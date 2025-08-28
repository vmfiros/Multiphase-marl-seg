
# Baseline Models & Commands

We include three baselines for comparison:

- **UNet3D** (classic 3D U-Net)
- **ResUNet3D** (residual variant)
- **SwinUNet3D** (single-stream Swin U-Net, no CPAF)

## Train
```bash
# UNet3D
bash scripts/train_baseline.sh unet3d configs/model_unet3d.yaml
# ResUNet3D
bash scripts/train_baseline.sh resunet3d configs/model_resunet3d.yaml
# SwinUNet3D
bash scripts/train_baseline.sh swinunet3d configs/model_swinunet3d.yaml
```

## Inference
```bash
# After training (checkpoints saved in checkpoints/)
bash scripts/infer_baseline.sh unet3d configs/model_unet3d.yaml checkpoints/unet3d.pt
bash scripts/infer_baseline.sh resunet3d configs/model_resunet3d.yaml checkpoints/resunet3d.pt
bash scripts/infer_baseline.sh swinunet3d configs/model_swinunet3d.yaml checkpoints/swinunet3d.pt
```

All models produce `*_P.nii.gz` (probability) and `*_M0.nii.gz` (thresholded mask).
