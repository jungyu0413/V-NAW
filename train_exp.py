import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils import *
from src.model import *
import argparse
import numpy as np
import random
import wandb
from tqdm import tqdm
from data.data_loader import build_seq_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('NLA_ABAW', add_help=False)
    parser.add_argument('--backbone', default='swin', type=str)
    parser.add_argument('--exp_name', default='single_gpu', type=str)
    parser.add_argument('--im', default=True, type=bool)
    parser.add_argument('--type', default='NLA', choices=['jsd', 'NAW', 'NLA', 'None'])
    parser.add_argument('--clip', default=100, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--opt', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--blr', type=float, default=1.5e-4)
    parser.add_argument('--sch', type=str, default='cos', choices=['exp','cos'])
    parser.add_argument('--lam_a', type=float, default=0.5)
    parser.add_argument('--lam_b', type=float, default=0.4)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--tras_n', type=int, default=1)
    parser.add_argument('--temporal_aug', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train_csv_path', type=str, default='train_path')
    parser.add_argument('--valid_csv_path', type=str, default='valid_path')
    parser.add_argument('--output_dir', default='./weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    return parser
  
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloader(args):
    train_dataset = build_seq_dataset(args, "train")
    valid_dataset = build_seq_dataset(args, "valid")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_mem)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    return train_loader, valid_loader

def train(model, dataloader, optimizer, scheduler, criterion, args):
    model.train()
    acc_logger = AccuracyLogger_torch(8)
    iter_cnt = 0
    running_loss = 0
    args.eps = exponential_scheduler(args.current_epoch, args.epochs)

    for imgs, flipped_imgs, labels in tqdm(dataloader):
        imgs = imgs.to(args.device, non_blocking=True)
        flipped_imgs = flipped_imgs.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        optimizer.zero_grad()

        combined_imgs = torch.cat([imgs, flipped_imgs], dim=1)
        output, flipped_output = model(combined_imgs, is_concat=True)

        loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))

        if args.type == 'jsd':
            jsd_loss = jensen_shannon_divergence(output.reshape(-1, output.size(-1)),
                                                 flipped_output.reshape(-1, flipped_output.size(-1)))
            loss += args.lam_a * jsd_loss

        elif args.type == 'NAW':
            NLA_loss = Integrated_Co_GA_Loss(args)(output.reshape(-1, output.size(-1)), labels.reshape(-1))
            loss += args.lam_b * NLA_loss

        elif args.type == 'NLA':
            jsd_loss = jensen_shannon_divergence(output.reshape(-1, output.size(-1)),
                                                 flipped_output.reshape(-1, flipped_output.size(-1)))
            NLA_loss = Integrated_Co_GA_Loss(args)(output.reshape(-1, output.size(-1)), labels.reshape(-1))
            loss = args.lam_a * loss + (1 - args.lam_a) * NLA_loss + args.lam_b * jsd_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iter_cnt += 1

        softmax_output = F.softmax(output, dim=-1)
        _, predicts = torch.max(softmax_output, dim=-1)
        acc_logger.update(predicts, labels)

    scheduler.step()
    return acc_logger, running_loss / iter_cnt

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
        f"im_sam_{args.im}"
    ])
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    createDirectory(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    # Model 구성
    if args.backbone == 'r50':
        feature_extractor = ResNet50FeatureExtractor().to(device)
        args.inc = 2048
    elif args.backbone == 'swin':
        feature_extractor = SwinTransformerFeatureExtractor().to(device)
        args.inc = 512
    else:
        raise ValueError("Invalid backbone")

    temporal_model = TransEncoder(inc=args.inc).to(device)
    model = VideoFeatureModel_concat(feature_extractor, temporal_model).to(device)

    # Optimizer & Scheduler
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = (torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
                 if args.sch == 'exp' else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5))

    train_loader, valid_loader = create_dataloader(args)
    wandb.init(project="SingleGPU_ABAW_EXP", name=args.exp_name, config=vars(args))

    for epoch in range(args.epochs):
        args.current_epoch = epoch
        acc_logger, train_loss = train(model, train_loader, optimizer, scheduler, criterion, args)
        class_acc, total_acc, f1_scores, mean_f1_score, data_size = acc_logger.final_score()

        print(f"[Train] Epoch {epoch}: Loss {train_loss:.4f}, Acc@1 {total_acc:.4f}, F1 {mean_f1_score:.4f}")
        wandb.log({"Train Loss": train_loss, "Train Acc@1": total_acc, "Train F1": mean_f1_score}, step=epoch)

        acc_logger = validate(model, valid_loader, args)
        class_acc, total_acc, f1_scores, mean_f1_score, data_size = acc_logger.final_score()

        print(f"[Valid] Epoch {epoch}: Acc@1 {total_acc:.4f}, F1 {mean_f1_score:.4f}")
        wandb.log({"Valid Acc@1": total_acc, "Valid F1": mean_f1_score}, step=epoch)

        if mean_f1_score > args.max_f1_score:
            args.max_f1_score = mean_f1_score
            save_classifier(model, "best", args)

        if epoch % args.save_freq == 0:
            save_classifier(model, f"epoch_{epoch}", args)

    save_classifier(model, "final", args)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

