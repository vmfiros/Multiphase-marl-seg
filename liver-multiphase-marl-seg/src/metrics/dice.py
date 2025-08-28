
import torch

def dice_score(pred_bin, target, eps=1e-6):
    inter = (pred_bin*target).sum(dim=(1,2,3,4)).float()
    denom = pred_bin.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4)) + eps
    return (2*inter/denom).mean().item()
