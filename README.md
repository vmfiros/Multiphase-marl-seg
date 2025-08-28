# Liver Multiphase MARL Segmentation
[![CI](https://github.com/vmfiros/Liver-multiphase-marl-seg/actions/workflows/ci.yml/badge.svg)](https://github.com/vmfiros/Liver-multiphase-marl-seg/actions/workflows/ci.yml)

![Architecture](docs/figure/Archi1.png)

Implementation of a 3D Swin U-Net with Cross-Phase Attention Fusion (CPAF), followed by a Multi-Agent RL refinement loop and Grad-CAM++ explainability. Includes training/eval scripts and working baselines (UNet3D, ResUNet3D, SwinUNet3D).


## Quickstart

### Environment
```bash
python -m pip install -r requirements.txt

python scripts/preprocess.py --config configs/dataset_scidb.yaml


python -m src.training.train \
  --arch swin_unet_cpaf \
  --model-config configs/model_swin_cpaf.yaml \
  --data data/interim/SCIDB \
  --epochs 200

# UNet3D
python -m src.training.train --arch unet3d     --model-config configs/model_unet3d.yaml
# ResUNet3D
python -m src.training.train --arch resunet3d  --model-config configs/model_resunet3d.yaml
# SwinUNet3D (single-stream, no CPAF)
python -m src.training.train --arch swinunet3d --model-config configs/model_swinunet3d.yaml


python -m src.inference.predict \
  --arch swin_unet_cpaf \
  --model-config configs/model_swin_cpaf.yaml \
  --ckpt checkpoints/swin_unet_cpaf.pt \
  --in_dir data/interim/SCIDB \
  --out_dir runs/preds

python - <<'PY'
import torch, nibabel as nib, numpy as np
from src.models.swin_unet_cpaf import SwinUNetCPAF
from src.explainability.gradcampp_3d import grad_cam_pp_3d
m = SwinUNetCPAF().eval()
x = torch.randn(1,2,64,64,64)  # demo volume
cam = grad_cam_pp_3d(m, x, target_layer_name='up1')  # (D,H,W) in [0,1]
nib.save(nib.Nifti1Image(cam, np.eye(4)), 'runs/cam_demo.nii.gz')
print('Saved runs/cam_demo.nii.gz')
PY


data/
├── raw/SCIDB/<PID>/{arterial.nii.gz, venous.nii.gz, label.nii.gz}
└── interim/SCIDB/<PID>/{arterial.nii.gz, venous.nii.gz, label.nii.gz}   # produced by scripts/preprocess.py


root_raw:      data/raw/SCIDB
root_interim:  data/interim/SCIDB
spacing_mm:    [1.0, 1.0, 1.0]
hu_clip:       [-200, 300]
normalize:     zscore

model:
  channels: [96, 192, 384]   # encoder L1–L3
  use_decoder_swin: true


---

### And here are the **two small config files** again (for quick copy-paste):

**`configs/dataset_scidb.yaml`**
```yaml
root_raw:      data/raw/SCIDB
root_interim:  data/interim/SCIDB
spacing_mm:    [1.0, 1.0, 1.0]
hu_clip:       [-200, 300]
normalize:     zscore

model:
  channels: [96, 192, 384]
  use_decoder_swin: true





