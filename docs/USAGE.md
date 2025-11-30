# Usage & scaling notes

This repository is a prototype playground to design a hybrid transformer combining ideas from Coconut and BLT.

Quick local test (toy):

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 scripts/train_toy.py --steps 50 --device cpu
```

Estimating parameters for a 1B model:

```bash
python3 scripts/estimate_params.py --d_model 2048 --n_layers 20
```

Scaling to a real 1B training run requires:
- GPU fleet with sufficient memory (A100/H100 or equivalent) with FSDP / DeepSpeed
- Distributed training support (torch.distributed / FSDP / Deepspeed)
- Mixed precision (bf16/half) to reduce memory footprint
- Careful data pipeline and steps-per-epoch planning

This repo contains a minimal training script `scripts/train.py` which demonstrates many concepts but is not production-scale.

Entropy patcher and learned estimator
- Try the entropy patcher with a learned estimator on the toy trainer:

```bash
PYTHONPATH=src python3 scripts/train_toy.py --patcher entropy --use_learned_estimator --max_patch_size 8 --steps 20
```

To also train the estimator online with an auxiliary loss, pass --train_estimator and tuning params:

```bash
PYTHONPATH=src python3 scripts/train_toy.py --patcher entropy --use_learned_estimator --train_estimator --est_loss_weight 1.0 --steps 20
```

You can enable estimator training in the config-driven runner by setting `model.entropy_patcher.estimator: learned` and
`training.train_estimator: true` in `configs/model_1b.yaml`.

Byte↔Patch attention
- Enable simplified byte↔patch attention by using `HybridTransformer(..., use_byte_patch_attention=True)` in code or with `configs/model_1b.yaml` which enables it by default in this repo.

Generation (sampling)
- Use `scripts/generate.py` to produce byte-level continuations from a saved checkpoint:

```bash
PYTHONPATH=src python3 scripts/generate.py --ckpt checkpoints/final.pt --prompt "Hello" --max_new_tokens 64
```

HuggingFace-style export
- Use `scripts/export_hf.py` to write a simple HF-style export folder with `pytorch_model.bin` and `config.json`:

```bash
PYTHONPATH=src python3 scripts/export_hf.py --ckpt checkpoints/final.pt --out hf_export
```
