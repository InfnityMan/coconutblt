# Deployment & large-scale training notes

This document provides guidance and starting templates for training a ~1B parameter HybridTransformer.

Quick highlights
- Resource sizing: a well-tuned 1B model typically needs multiple high-memory GPUs (A100 40GB/H100 80GB or similar), preferably with FSDP or DeepSpeed zero-offload.
- Use bf16 or fp16 (with careful optimizer state handling) to drastically reduce memory and increase throughput.

Starter recipe (DeepSpeed)
- See `configs/deepspeed_1b.json` — this is a minimal starter config enabling fp16 and ZeRO stage 3. Update values for your HW and training budget.
- Launch with `scripts/run_deepspeed_1b.sh` after customizing `CUDA_VISIBLE_DEVICES` and node settings.

FSDP alternative
- If you prefer PyTorch FSDP, map the `scripts/train.py` into a torchrun + FSDP wrapper and configure parameter sharding.

Data & mixing
- For real training, prepare a byte-level (or tokenized) streaming pipeline and ensure large-scale shuffling.
- Use gradient accumulation to increase per-step effective batch size while staying within GPU memory limits.

Profilling and checkpoints
- Always run small debug/shmoo jobs (2–4 GPUs) to debug the data pipeline and checkpointing before scaling to a full 1B run.
- Save frequent small checkpoints to make restarts / debugging feasible.

Notes on licensing and datasets
- If you use BLT code or datasets shipped with a non-commercial license, ensure your project use is permitted by that license.
