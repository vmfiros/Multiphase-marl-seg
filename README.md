# Liver Multiphase MARL Segmentation
[![CI](https://github.com/vmfiros/Liver-multiphase-marl-seg/actions/workflows/ci.yml/badge.svg)](https://github.com/vmfiros/Liver-multiphase-marl-seg/actions/workflows/ci.yml)

![Architecture](docs/figure/Archi1.png)

Implementation of a 3D Swin U-Net with Cross-Phase Attention Fusion (CPAF), followed by a Multi-Agent RL refinement loop and Grad-CAM++ explainability. Includes training/eval scripts and working baselines (UNet3D, ResUNet3D, SwinUNet3D).


## Quickstart

### Environment
```bash
python -m pip install -r requirements.txt

