
#!/usr/bin/env bash
set -e
ARCH=${1:-unet3d}
CFG=${2:-configs/model_unet3d.yaml}
CKPT=${3:-checkpoints/unet3d.pt}
python -m src.inference.predict --arch $ARCH --model-config $CFG --ckpt $CKPT
