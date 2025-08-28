
import argparse, torch
from torch.utils.data import DataLoader, random_split
from ..utils.config import load_yaml
from ..models.swin_unet_cpaf import SwinUNetCPAF
from ..training.trainer_seg import SegTrainer
from ..data.datasets import MultiPhaseCTDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=False, default='configs/train_seg.yaml')
    ap.add_argument('--model', required=False, default='configs/model_swin_cpaf.yaml')
    ap.add_argument('--data', required=False, default='data/interim/SCIDB')
    ap.add_argument('--epochs', type=int, default=2)  # small default for demo
    args = ap.parse_args()

    cfg_m = load_yaml(args.model)
    model = SwinUNetCPAF(in_channels=2, out_channels=1,
                         channels=tuple(cfg_m['model']['channels']),
                         use_decoder_swin=cfg_m['model'].get('use_decoder_swin', True))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Simple split by folder names present in interim
    import pathlib
    ids = [p.name for p in pathlib.Path(args.data).glob("*") if p.is_dir()]
    n = len(ids); n_train = max(1,int(n*0.8)); n_val = max(1,n-n_train)
    ds = MultiPhaseCTDataset(args.data, ids)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train,n_val])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    trainer = SegTrainer(model, lr=2e-4, amp=True)
    best = 1e9; best_path = "checkpoints/swin_cpaf_demo.pt"
    import os; os.makedirs("checkpoints", exist_ok=True)
    for ep in range(1, args.epochs+1):
        tr = trainer.train_one_epoch(train_loader, device)
        va = trainer.validate(val_loader, device)
        print(f"Epoch {ep}: train={tr:.4f} val={va:.4f}")
        if va < best:
            best = va
            torch.save(model.state_dict(), best_path)
            print(f"Saved {best_path}")

if __name__ == '__main__':
    main()
