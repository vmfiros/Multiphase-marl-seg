
import torch, torch.nn as nn, torch.optim as optim
from ..losses.dice import dice_loss

class SegTrainer:
    def __init__(self, model, lr=2e-4, amp=True):
        self.model = model
        self.opt = optim.AdamW(model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.amp = amp
        self.ce = nn.BCELoss()

    def train_one_epoch(self, loader, device='cuda'):
        self.model.train()
        total = 0.0
        for img,lbl,_ in loader:
            img,lbl = img.to(device), lbl.to(device).float()
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                p,_ = self.model(img)
                loss = 0.7*dice_loss(p,lbl) + 0.3*self.ce(p, lbl)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            total += float(loss.item())
        return total/len(loader)

    @torch.no_grad()
    def validate(self, loader, device='cuda'):
        self.model.eval()
        total = 0.0
        for img,lbl,_ in loader:
            img,lbl = img.to(device), lbl.to(device).float()
            p,_ = self.model(img)
            loss = dice_loss(p,lbl)
            total += float(loss.item())
        return total/len(loader)
