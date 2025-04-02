import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils import *
from src.model import *
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import numpy as np
import random
import wandb
import warnings
from tqdm import tqdm
from data.data_loader import DDPImbalancedDatasetSampler
from data.data_loader import build_seq_dataset

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('NLA_ABAW', add_help=False)
    parser.add_argument('--backbone', default='swin', type=str)
    parser.add_argument('--exp_name', default='ddp', type=str)
    parser.add_argument('--im', default=True, type=bool)
    parser.add_argument('--type', default='NLA', choices=['jsd', 'NAW', 'NLA', 'None'])
    parser.add_argument('--clip', default=100, type=int)
    parser.add_argument('--batch_size', default=3*8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--opt', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR')
    parser.add_argument('--sch', type=str, default='cos', choices=['exp','cos'])
    parser.add_argument('--lam_a', type=float, default=0.5)
    parser.add_argument('--lam_b', type=float, default=0.4)
    parser.add_argument('--min_lr', type=float, default=0.00001, metavar='LR')
    parser.add_argument('--tras_n', type=int, default=1)
    parser.add_argument('--temporal_aug', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_csv_path', type=str, default='/workspace/ABAW8/6th ABAW Annotations/EXPR_Recognition_Challenge/Train_Set')
    parser.add_argument('--valid_csv_path', type=str, default='/workspace/ABAW8/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set')
    parser.add_argument('--output_dir', default='/workspace/weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:7767', type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.set_defaults(multiprocessing_distributed=True)
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # DDP를 위한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(args):
    train_dataset = build_seq_dataset(args, "train")
    valid_dataset = build_seq_dataset(args, "valid")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_labels = [seq[0] for seq in train_dataset.labels]  # 각 클립의 첫 번째 라벨 사용

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), sampler=DDPImbalancedDatasetSampler(train_dataset, labels=train_labels),
                              shuffle=(train_sampler is None),  num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size // dist.get_world_size(),
                              shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    return train_loader, valid_loader

def train(model, dataloader, optimizer, scheduler, criterion, args):
    torch.autograd.set_detect_anomaly(True)
    acc_logger = AccuracyLogger_torch(8)
    model.train()

    iter_cnt = 0
    running_loss = 0
    args.eps = exponential_scheduler(args.current_epoch, args.epochs)
    if args.gpu == 0:
        for imgs, flipped_imgs, labels in tqdm(dataloader):
            imgs = imgs.to(args.device, non_blocking=True)
            flipped_imgs = flipped_imgs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            
            combined_imgs = torch.cat([imgs, flipped_imgs], dim=1)
            output, flipped_output = model(combined_imgs, is_concat=True) 

            loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))
            
            if args.type == 'jsd':
                jsd_loss = jensen_shannon_divergence(output.reshape(-1, output.size(-1)), flipped_output.reshape(-1, flipped_output.size(-1)))
                loss = loss + args.lam_a*jsd_loss
            
            elif args.type == 'NAW':
                NLA_loss = Integrated_Co_GA_Loss(args)(output.reshape(-1, output.size(-1)), labels.reshape(-1))
                loss = loss + args.lam_b*NLA_loss
            
            elif args.type == 'NLA':
                jsd_loss = jensen_shannon_divergence(output.reshape(-1, output.size(-1)), flipped_output.reshape(-1, flipped_output.size(-1)))
                NLA_loss = Integrated_Co_GA_Loss(args)(output.reshape(-1, output.size(-1)), labels.reshape(-1))
                loss = args.lam_a*loss + (1-args.lam_a)*NLA_loss + args.lam_b*jsd_loss
            else:
                loss = loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iter_cnt += 1
            softmax_output = F.softmax(output, dim=-1)
            _, predicts = torch.max(softmax_output, dim=-1)
            acc_logger.update(predicts, labels)

        scheduler.step()
        running_loss = torch.tensor(running_loss / iter_cnt, device=args.device)
        dist.reduce(running_loss, dst=0, op=dist.ReduceOp.SUM)
        
        
    if args.gpu != 0:
        for imgs, flipped_imgs, labels in tqdm(dataloader):
            imgs = imgs.to(args.device, non_blocking=True)
            flipped_imgs = flipped_imgs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            
            combined_imgs = torch.cat([imgs, flipped_imgs], dim=1)
            output, flipped_output = model(combined_imgs, is_concat=True) 

            loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))
            
            if args.type == 'jsd':
                jsd_loss = jensen_shannon_divergence(output.reshape(-1, output.size(-1)), flipped_output.reshape(-1, flipped_output.size(-1)))
                loss = loss + args.lam_a*jsd_loss
            
            elif args.type == 'NAW':
                NLA_loss = Integrated_Co_GA_Loss(args)(output.reshape(-1, output.size(-1)), labels.reshape(-1))
                loss = loss + args.lam_b*NLA_loss
            
            elif args.type == 'NLA':
                jsd_loss = jensen_shannon_divergence(output.reshape(-1, output.size(-1)), flipped_output.reshape(-1, flipped_output.size(-1)))
                NLA_loss = Integrated_Co_GA_Loss(args)(output.reshape(-1, output.size(-1)), labels.reshape(-1))
                loss = args.lam_a*loss + (1-args.lam_a)*NLA_loss + args.lam_b*jsd_loss
            else:
                loss = loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iter_cnt += 1
            softmax_output = F.softmax(output, dim=-1)
            _, predicts = torch.max(softmax_output, dim=-1)
            acc_logger.update(predicts, labels)

        scheduler.step()
        running_loss = torch.tensor(running_loss / iter_cnt, device=args.device)
        dist.reduce(running_loss, dst=0, op=dist.ReduceOp.SUM)
        

    return acc_logger, running_loss

def validate(model, dataloader, args):
    model.eval()
    acc_logger = AccuracyLogger_torch(8)
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            output = model(imgs)
            softmax_output = F.softmax(output, dim=-1)
            _, predicts = torch.max(softmax_output, dim=-1)
            acc_logger.update(predicts, labels)

    return acc_logger

def main_worker(local_rank, ngpus_per_node, args):
    set_seed(args.seed + local_rank)
    args.local_rank = local_rank
    args.distributed = True  # DDP 모드에서 분산 학습 여부 명시
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank * ngpus_per_node + local_rank)

    if args.backbone == 'r50':
        feature_extractor = ResNet50FeatureExtractor().to(args.device)
        args.inc = 2048
    elif args.backbone == 'swin':
        feature_extractor = SwinTransformerFeatureExtractor().to(args.device)
        args.inc = 512
    else:
        raise ValueError('Invalid backbone')

    temporal_model = TransEncoder(inc=args.inc).to(args.device)
    model = VideoFeatureModel_concat(feature_extractor, temporal_model).to(args.device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        print('need to input the opt')

    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.sch == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

    train_dataloader, valid_dataloader = create_dataloader(args)

    if args.local_rank == 0:
        wandb.init(project="Jindo ABAW EXP", name=args.exp_name, config={"learning_rate": args.lr, "architecture": args.backbone, "clip": args.clip, "epochs": args.epochs})

    for epoch in range(args.epochs):
        args.current_epoch = epoch
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        acc_logger, running_loss = train(model, train_dataloader, optimizer, scheduler, criterion, args)
        acc_logger.gather()

        if args.local_rank == 0:
            classwise_acc, total_acc, f1_scores, mean_f1_score, total_data_num = acc_logger.final_score()
            print(f'------------- *{epoch}* train -------------')
            print(f'Train Loss {running_loss:.3f}\t')
            print(f'Train F1 : {[f"{score:.3f}" for score in f1_scores]}\t')
            print(f'Train Mean F1 : {mean_f1_score:.3f}\t')
            print(f'Train Acc@1 {total_acc:.3f}\t')
            print(f'Train Class Acc : {[f"{acc:.3f}" for acc in classwise_acc]}\t')
            print(f'Class-Wise Acc : {np.mean(classwise_acc.cpu().numpy()):.3f}\t') 
            print(f'Train Data size : {total_data_num.item()}') 

            wandb.log({"Train Acc@1": total_acc,
                'Train Loss' : running_loss,
                'Train F1' : mean_f1_score,
                'Train Mean Acc' : np.mean(classwise_acc.cpu().numpy()),
                }, step=epoch)


        acc_logger = validate(model, valid_dataloader, args)
        acc_logger.gather()
        
        if args.local_rank == 0:
            classwise_acc, total_acc, f1_scores, mean_f1_score, total_data_num= acc_logger.final_score()
            print(f'------------- *{epoch}* test-------------')
            print(f'F1 score : {[f"{score:.3f}" for score in f1_scores]}\t')
            print(f'Mean F1 score : {mean_f1_score:.3f}\t')
            print(f'Test Acc@1 {total_acc:.3f}\t')
            print(f'Test Class Acc : {[f"{acc:.3f}" for acc in classwise_acc]}\t')
            print(f'Class-Wise Acc : {np.mean(classwise_acc.cpu().numpy()):.3f}\t')
            print(f'Test Data size : {total_data_num.item()/ngpus_per_node}') 

            wandb.log({'Test Acc' : total_acc,
                'Test F1' : mean_f1_score,
                'Test Mean Acc' : np.mean(classwise_acc.cpu().numpy()),
                }, step=epoch)
            # model save
            if np.mean(mean_f1_score.cpu().numpy()) > args.max_f1_score:
                args.max_f1_score = np.mean(mean_f1_score.cpu().numpy())
                save_classifier(model, 'best', args)

        if args.local_rank == 0 and epoch % args.save_freq == 0:
            save_classifier(model.module, epoch, args)

    if args.local_rank == 0:
        save_classifier(model.module, 'final', args)
    dist.destroy_process_group()

def main(args):
    set_seed(args.seed) 
    args.max_f1_score = -1
    args.img_size = 224 if args.backbone == 'r50' else 112
    args.exp_name = "_".join([
    args.backbone,
    args.exp_name,
    args.type,
    f"clip_{args.clip}",
    f"opt_{args.opt}",
    f"lr_{args.lr}",
    f"sch_{args.sch}",
    f"gamma_{args.gamma}",
    f"lam_a_{args.lam_a}",
    f"lam_b_{args.lam_b}",
    f"tras_n_{args.tras_n}",
    f"temporal_aug_{args.temporal_aug}",
    f"im_sam_{args.im}"])
    
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    createDirectory(args.output_dir)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    args.distributed = True  # DDP 모드 설정

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
