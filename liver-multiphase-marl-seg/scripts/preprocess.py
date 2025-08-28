
import argparse, pathlib, os
import SimpleITK as sitk
import numpy as np
from src.data.transforms import resample_to_spacing, clip_and_normalize_hu, sitk_to_np, np_to_sitk
from src.utils.config import load_yaml, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    raw = pathlib.Path(cfg['root_raw'])
    out = pathlib.Path(cfg['root_interim'])
    ensure_dir(out)

    for pid_dir in sorted(raw.glob("*")):
        if not pid_dir.is_dir(): continue
        a = sitk.ReadImage(str(pid_dir/'arterial.nii.gz'))
        v = sitk.ReadImage(str(pid_dir/'venous.nii.gz'))
        y = sitk.ReadImage(str(pid_dir/'label.nii.gz'))

        a = resample_to_spacing(a, tuple(cfg['spacing_mm']), is_label=False)
        v = resample_to_spacing(v, tuple(cfg['spacing_mm']), is_label=False)
        y = resample_to_spacing(y, tuple(cfg['spacing_mm']), is_label=True)

        a_np = clip_and_normalize_hu(sitk_to_np(a), cfg['hu_clip'][0], cfg['hu_clip'][1], cfg.get('normalize','zscore'))
        v_np = clip_and_normalize_hu(sitk_to_np(v), cfg['hu_clip'][0], cfg['hu_clip'][1], cfg.get('normalize','zscore'))
        y_np = (sitk_to_np(y) > 0).astype(np.uint8)

        odir = out / pid_dir.name
        os.makedirs(odir, exist_ok=True)
        sitk.WriteImage(np_to_sitk(a_np, a), str(odir/'arterial.nii.gz'))
        sitk.WriteImage(np_to_sitk(v_np, v), str(odir/'venous.nii.gz'))
        sitk.WriteImage(np_to_sitk(y_np, y), str(odir/'label.nii.gz'))
        print("Preprocessed", pid_dir.name)

if __name__ == '__main__':
    main()
