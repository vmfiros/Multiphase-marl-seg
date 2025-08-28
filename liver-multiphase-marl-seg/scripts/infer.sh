#!/usr/bin/env bash
set -e
CFG=${1:-configs/infer.yaml}
shift
python -m src.inference.predict --config $CFG "$@"
