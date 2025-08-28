
import argparse, torch, pathlib, numpy as np, nibabel as nib, os
from ..utils.config import load_yaml
from ..models.factory import build_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/infer.yaml')
    ap.add_argument('--arch', required=True, choices=['swin_unet_cpaf','unet3d','resunet3d','swinunet3d'])
    ap.add_argument('--model-config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--in_dir', default='data/interim/SCIDB')
    ap.add_argument('--out_dir', default='runs/preds')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_m = load_yaml(args.model_config)
    model = build_model(args.arch, cfg_m)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device); model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    for pdir in pathlib.Path(args.in_dir).glob("*"):
        if not pdir.is_dir(): continue
        a = nib.load(str(pdir/'arterial.nii.gz')).get_fdata().astype(np.float32)
        v = nib.load(str(pdir/'venous.nii.gz')).get_fdata().astype(np.float32)
        x = np.stack([a,v], axis=0)[None]  # (1,2,D,H,W)
        x_t = torch.from_numpy(x).to(device)
        with torch.no_grad():
            p,_ = model(x_t)
        p = p.squeeze().cpu().numpy()  # (D,H,W)
        nib.save(nib.Nifti1Image(p, np.eye(4)), str(pathlib.Path(args.out_dir)/f"{pdir.name}_{args.arch}_P.nii.gz"))
        m0 = (p >= 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(m0, np.eye(4)), str(pathlib.Path(args.out_dir)/f"{pdir.name}_{args.arch}_M0.nii.gz"))
        print("Saved:", pdir.name)

if __name__ == '__main__':
    main()
