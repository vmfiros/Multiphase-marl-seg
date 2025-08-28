
import argparse, torch, pathlib
from torch.utils.data import DataLoader
from ..utils.config import load_yaml
from ..models.factory import build_model
from ..training.trainer_seg import SegTrainer
from ..data.datasets import MultiPhaseCTDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=['swin_unet_cpaf','unet3d','resunet3d','swinunet3d'])
    ap.add_argument('--model-config', required=True)
    ap.add_argument('--data', default='data/interim/SCIDB')
    ap.add_argument('--epochs', type=int, default=2)
    args = ap.parse_args()

    cfg = load_yaml(args.model_config)
    model = build_model(args.arch, cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # dataset
    ids = [p.name for p in pathlib.Path(args.data).glob("*") if p.is_dir()]
    ds = MultiPhaseCTDataset(args.data, ids)
    n = len(ds); n_train = max(1,int(n*0.8)); n_val = max(1,n-n_train)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train,n_val])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    trainer = SegTrainer(model, lr=2e-4, amp=True)
    best = 1e9; out = pathlib.Path("checkpoints"); out.mkdir(exist_ok=True, parents=True)
    ckpt = out / f"{args.arch}.pt"
    for ep in range(1, args.epochs+1):
        tr = trainer.train_one_epoch(train_loader, device)
        va = trainer.validate(val_loader, device)
        print(f"[{args.arch}] epoch {ep}: train={tr:.4f} val={va:.4f}")
        if va < best:
            best = va
            torch.save(model.state_dict(), ckpt)
            print("Saved", ckpt)

if __name__ == '__main__':
    main()
