#!/usr/bin/env bash
set -e
CFG=${1:-configs/train_rl.yaml}
python -m src.training.train_rl --config $CFG
