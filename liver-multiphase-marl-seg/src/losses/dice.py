
import torch

def dice_loss(pred, target, eps=1e-6):
    # pred,target: (B,1,D,H,W) probabilities and binary masks
    pred = pred.clamp(0,1)
    target = target.float()
    inter = (pred*target).sum(dim=(1,2,3,4))
    denom = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4)) + eps
    dice = 2*inter/denom
    return 1 - dice.mean()
