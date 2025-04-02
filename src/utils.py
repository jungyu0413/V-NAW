import torch
import cv2
import numpy as np
import random
import torch.nn.functional as F
import os
from torch import distributed as dist

# calculate accuracy      
class AccuracyLogger:
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.reset(num_class)

    def reset(self, n):
        self.classwise_sum = np.zeros(n, dtype=float)  # True Positives
        self.classwise_count = np.zeros(n, dtype=float)  # Total Labels (TP + FN)
        self.prediction_count = np.zeros(n, dtype=float)  # Total Predictions (TP + FP)
        self.total_sum = 0
        self.total_count = 0
    
    def update(self, predictions, labels):
        # Get number of images in current batch.
        num_imgs = predictions.shape[0]

        # Store total values.
        self.total_sum += np.sum((predictions == labels))
        self.total_count += num_imgs        

        # Store class-wise values.
        for i in range(self.classwise_sum.shape[0]):  # 7 classes: 0 ~ 6
            # True Positive (correctly predicted as class i)
            self.classwise_sum[i] += np.sum((predictions == i) & (labels == i)).astype(float)
            # Total actual labels (True Positives + False Negatives)
            self.classwise_count[i] += np.sum((labels == i)).astype(float)
            # Total predicted as class i (True Positives + False Positives)
            self.prediction_count[i] += np.sum((predictions == i)).astype(float)

    def final_score(self):
        # Calculate classwise accuracy.
        classwise_acc = self.classwise_sum / self.classwise_count
        for idx, cnt in enumerate(self.classwise_count):
            if cnt == 0:
                classwise_acc[idx] = 1

        # Calculate total mean accuracy.
        total_acc = self.total_sum / self.total_count

        return classwise_acc, total_acc

    def f1_score(self):
        # Precision: TP / (TP + FP)
        precision = self.classwise_sum / self.prediction_count
        # Recall: TP / (TP + FN)
        recall = self.classwise_sum / self.classwise_count

        # Handle division by zero
        precision[np.isnan(precision)] = 0
        recall[np.isnan(recall)] = 0

        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1[np.isnan(f1)] = 0  # Handle NaN where precision + recall == 0

        # Mean F1 Score
        mean_f1 = np.mean(f1)

        return f1, mean_f1



class AccuracyLogger_torch:
    """Computes and stores the average and current value, including F1 scores."""

    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.classwise_sum = torch.zeros(self.num_class, dtype=torch.float32, device='cuda')
        self.classwise_count = torch.zeros(self.num_class, dtype=torch.float32, device='cuda')
        self.total_sum = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        self.total_count = torch.tensor(0.0, dtype=torch.float32, device='cuda')

        self.true_positive = torch.zeros(self.num_class, dtype=torch.float32, device='cuda')
        self.false_positive = torch.zeros(self.num_class, dtype=torch.float32, device='cuda')
        self.false_negative = torch.zeros(self.num_class, dtype=torch.float32, device='cuda')
    
    def update(self, predictions, labels):
        predictions = predictions.flatten()
        labels = labels.flatten()

        # Update total values
        self.total_sum += (predictions == labels).sum().float()
        self.total_count += predictions.size(0)
        # Update class-wise values
        for i in range(self.num_class):
            class_mask = (labels == i) # label mask
            pred_mask = (predictions == i) # prediction mask

            self.classwise_sum[i] += (predictions[class_mask] == labels[class_mask]).sum().float()
            self.classwise_count[i] += class_mask.sum().float()

            # For F1 score calculation
            self.true_positive[i] += (pred_mask & class_mask).sum().float()
            self.false_positive[i] += (pred_mask & ~class_mask).sum().float()
            self.false_negative[i] += (~pred_mask & class_mask).sum().float()

    def gather(self):
        dist.barrier()
        dist.all_reduce(self.classwise_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.classwise_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_count, op=dist.ReduceOp.SUM)

        dist.all_reduce(self.true_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_negative, op=dist.ReduceOp.SUM)

    def final_score(self):
        # Calculate classwise accuracy
        classwise_acc = self.classwise_sum / (self.classwise_count + 1e-6)

        # Calculate total mean accuracy
        total_acc = self.total_sum / (self.total_count + 1e-6)

        # Calculate Precision, Recall, and F1 Score for each class
        precision = self.true_positive / (self.true_positive + self.false_positive + 1e-12)
        recall = self.true_positive / (self.true_positive + self.false_negative + 1e-12)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

        # Replace NaN F1 scores (e.g., for classes with no true positives) with 0
        f1_scores = torch.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate average F1 Score across all classes
        mean_f1_score = f1_scores.mean()

        data_num = self.total_count
        
        return classwise_acc, total_acc, f1_scores, mean_f1_score, data_num


class CCCLogger_torch:
    """Computes and stores the average and current value of CCC (Concordance Correlation Coefficient) for valence and arousal."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all stored values."""
        self.val_ccc_sum = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        self.aro_ccc_sum = torch.tensor(0.0, dtype=torch.float32, device='cuda')
        self.count = torch.tensor(0.0, dtype=torch.float32, device='cuda')

    def update(self, val, pred_val, aro, pred_aro):
        """Update CCC values for valence and arousal with new batch data.
        
        Args:
            val (torch.Tensor): Ground truth valence, shape [batch, num_samples]
            pred_val (torch.Tensor): Predicted valence, shape [batch, num_samples]
            aro (torch.Tensor): Ground truth arousal, shape [batch, num_samples]
            pred_aro (torch.Tensor): Predicted arousal, shape [batch, num_samples]
        """
        val_ccc = self.CCC_loss_cal(pred_val, val)
        aro_ccc = self.CCC_loss_cal(pred_aro, aro)
        
        self.val_ccc_sum += val_ccc
        self.aro_ccc_sum += aro_ccc
        self.count += 1

    def gather(self):
        """Synchronize and aggregate values across multiple processes (for distributed training)."""
        dist.barrier()
        dist.all_reduce(self.val_ccc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.aro_ccc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    def final_score(self):
        """Compute final CCC scores after aggregation."""
        val_ccc = self.val_ccc_sum / (self.count + 1e-6)
        aro_ccc = self.aro_ccc_sum / (self.count + 1e-6)
        total_ccc = (val_ccc + aro_ccc) / 2  # Mean CCC

        return total_ccc, val_ccc, aro_ccc, self.count

    def CCC_loss_cal(self, x, y):
        """Compute Concordance Correlation Coefficient (CCC) between two tensors."""
        x_m = torch.mean(x, dim=-1, keepdim=True)  # Mean along sample dimension
        y_m = torch.mean(y, dim=-1, keepdim=True)
        x_s = torch.std(x, dim=-1, keepdim=True)
        y_s = torch.std(y, dim=-1, keepdim=True)

        vx = x - x_m
        vy = y - y_m

        rho = torch.mean(vx * vy, dim=-1) / (x_s * y_s + 1e-8)
        ccc = (2 * rho * x_s * y_s) / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + 1e-8)

        return ccc.mean()  # Mean across batch

class AU_Logger_torch:
    """Computes and stores the F1 scores for each AU and overall average F1 Score."""

    def __init__(self, num_au=12):
        self.num_au = num_au
        self.reset()

    def reset(self):
        """Reset all stored values."""
        self.true_positive = torch.zeros(self.num_au, dtype=torch.float32, device='cuda')
        self.false_positive = torch.zeros(self.num_au, dtype=torch.float32, device='cuda')
        self.false_negative = torch.zeros(self.num_au, dtype=torch.float32, device='cuda')

    def update(self, predictions, labels):
        """Update TP, FP, FN for each AU.
        
        Args:
            predictions (torch.Tensor): Model predictions (0 or 1), shape [batch, num_au]
            labels (torch.Tensor): Ground truth labels (0 or 1), shape [batch, num_au]
        """
        # Ensure predictions and labels are both binary (0 or 1) and integer type
        predictions = (predictions > 0.5).int()  # 확률값 → 이진값 변환
        labels = labels.int()  # 🔥 labels도 int로 변환하여 타입 맞추기

        # Update TP, FP, FN for each AU
        self.true_positive += (predictions & labels).sum(dim=0).float()
        self.false_positive += (predictions & ~labels).sum(dim=0).float()
        self.false_negative += (~predictions & labels).sum(dim=0).float()

    def gather(self):
        """Synchronize values across multiple GPUs (if using Distributed Training)."""
        dist.barrier()
        dist.all_reduce(self.true_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_positive, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.false_negative, op=dist.ReduceOp.SUM)

    def final_score(self):
        """Compute F1 Scores for each AU and the overall mean F1 Score."""
        precision = self.true_positive / (self.true_positive + self.false_positive + 1e-12)
        recall = self.true_positive / (self.true_positive + self.false_negative + 1e-12)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

        # NaN 방지 및 -0.0 → 0.0 변환
        f1_scores = torch.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0).abs()

        # 평균 F1 Score 계산
        mean_f1_score = f1_scores.mean()

        return f1_scores, mean_f1_score

# JSD regularizer
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_div_p = F.kl_div(F.log_softmax(p, dim=1), F.softmax(m, dim=1), reduction='sum')
    kl_div_q = F.kl_div(F.log_softmax(q, dim=1), F.softmax(m, dim=1), reduction='sum')
    js_divergence = 0.5 * (kl_div_p + kl_div_q)
    return js_divergence


import torch.nn as nn
# NLA
class Integrated_Co_GA_Loss(nn.Module):
    def __init__(self, args, t_mean_x=0.50, t_mean_y=0.50, f_mean_x=0.30, f_mean_y=0.15, f_cm=0.80, f_std=0.85, t_cm=-0.50, t_std=0.75, t_lambda=1.0):
        super(Integrated_Co_GA_Loss, self).__init__()
        t_cm = args.eps * t_cm
        self.t_lambda = t_lambda
        self.t_mu = torch.tensor([t_mean_x, t_mean_y],dtype=torch.float).unsqueeze(1).to('cuda')
        self.f_mu = torch.tensor([f_mean_x, f_mean_y],dtype=torch.float).unsqueeze(1).to('cuda')
        self.t_cov = torch.tensor([[t_std, t_cm], [t_cm, t_std]],dtype=torch.float).to('cuda')
        self.f_cov = torch.tensor([[f_std, f_cm], [f_cm, f_std]],dtype=torch.float).to('cuda')
        self.w = torch.Tensor([1.0, 10.399988319803773, 16.23179290857716, 19.607905747632678, \
            1.8556467915720152, 2.225347712532647, 5.610554505356018, 1.0590043828089226]).to('cuda')
        
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # wce_loss = F.cross_entropy(inputs, targets, self.w, reduction='none')

        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze()

        top2_probs = probs.topk(2, dim=1).values
        max_probs = top2_probs[:, 0]
        second_max_probs = top2_probs[:, 1]

        is_pt_max = max_probs == p_t
        
        p_negative = torch.where(is_pt_max, second_max_probs, max_probs)
        # 가우시안 분포 계산
        SV_flat = torch.stack([p_t, p_negative], axis=0)
        
        t_value = self.t_lambda*self.co_gau(SV_flat, self.t_cov, self.t_mu) 
        f_value = self.co_gau(SV_flat, self.f_cov, self.f_mu)

        value = torch.where(is_pt_max, t_value, f_value)
        adjusted_loss = ce_loss * value
        # adjusted_loss = ce_loss * value + wce_loss
        
        return adjusted_loss.mean()
    
    
    def co_gau(self, SV_flat, cov, mu):
        diff = SV_flat - mu
        inv_cov = torch.linalg.inv(cov)
        value = torch.exp(-0.5 * torch.matmul(torch.matmul(diff.t(), inv_cov), (diff)))
        value = torch.diagonal(value)
        return value
    


def exponential_scheduler(epoch, epochs, slope=-15, sch_bool=True):
    if sch_bool:
        return (1 - np.exp(slope * epoch / epochs))
    else:
        return 1
    
# CCC Loss
class CCC_loss(nn.Module):
    def __init__(self):
        super(CCC_loss, self).__init__()
        
    def forward(self, v_label, Vout, a_label, Aout):
        v_ccc = self.CCC_loss_cal(Vout,v_label) 
        a_ccc = self.CCC_loss_cal(Aout,a_label)
        ccc_loss = v_ccc + a_ccc
        mae_loss = (nn.MSELoss()(Vout, v_label) + nn.MSELoss()(Aout, a_label)) / 2
        return ccc_loss, mae_loss, v_ccc, a_ccc

    def CCC_loss_cal(self, x, y):
        x_m = torch.mean(x, dim=-1, keepdim=True)  
        y_m = torch.mean(y, dim=-1, keepdim=True)
        x_s = torch.std(x, dim=-1, keepdim=True)
        y_s = torch.std(y, dim=-1, keepdim=True)

        # std가 0이면 작은 값 추가
        x_s = torch.where(x_s == 0, torch.tensor(1e-6, device=x.device), x_s)
        y_s = torch.where(y_s == 0, torch.tensor(1e-6, device=y.device), y_s)

        vx = x - x_m
        vy = y - y_m

        rho = torch.mean(vx * vy, dim=-1) / (x_s * y_s + 1e-6)  # 분모가 0이 되지 않도록 보완
        ccc = (2 * rho * x_s * y_s) / ((torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2)) + 1e-6)

        return 1 - ccc  # CCC Loss 반환 (batch-wise)
    

def CCC_metric(x,y):
    x_m = torch.mean(x) # valence mean
    y_m = torch.mean(y) # arousal mean
    x_s = torch.std(x) # valence std
    y_s = torch.std(y) # arousal std
    vx = x - x_m # valence 편차
    vy = y - y_m # arousal 편차
    # valence 편차 * arousal 편차의 평균 / valence std * arousal std
    rho =  torch.mean(vx*vy) / (x_s*y_s)
    ccc = 2*rho*x_s*y_s/((torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))+1e-8)
    return ccc



# save model
def save_classifier(model, epoch, args):
    path = os.path.join(args.output,f'{epoch}.pth')
    torch.save(model.state_dict(), path)
    print(f'save : {epoch} : {path}')

# generate folder
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")





# img augmentation
def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



