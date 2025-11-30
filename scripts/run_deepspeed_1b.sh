#!/usr/bin/env bash
# Example launcher script for training a ~1B model with DeepSpeed.
# Edit paths, partition, number of nodes / GPUs and environment to suit your setup.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
CONFIG=configs/model_1b.yaml
DEEPSPEED_CONF=configs/deepspeed_1b.json

# Example: 4x H100 (80GB) or multiple A100s. Adjust workers accordingly.
NUM_GPUS=4

deepspeed --num_gpus ${NUM_GPUS} \
  --module scripts.train \
  --config ${CONFIG} \
  --deepspeed_config ${DEEPSPEED_CONF} \
  --device cuda
