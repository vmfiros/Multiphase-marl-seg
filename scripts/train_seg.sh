#!/usr/bin/env bash
set -e
CFG_TRAIN=${1:-configs/train_seg.yaml}
CFG_MODEL=${2:-configs/model_swin_cpaf.yaml}
python -m src.training.train_seg --train $CFG_TRAIN --model $CFG_MODEL
