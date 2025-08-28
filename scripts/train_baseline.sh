
#!/usr/bin/env bash
set -e
ARCH=${1:-unet3d}
CFG=${2:-configs/model_unet3d.yaml}
python -m src.training.train --arch $ARCH --model-config $CFG
