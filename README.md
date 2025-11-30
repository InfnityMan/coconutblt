# CoconutBLT — hybrid Coconut + BLT starter

This workspace is a starter scaffold to design and prototype a hybrid transformer architecture
combining ideas from Facebook Research's Coconut (continuous latent reasoning) and BLT (byte-latent / patching).

Important licensing note
- Coconut (MIT) — permissive
- BLT (CC-BY-NC-4.0) — non-commercial. If you reuse BLT-derived components, you must follow CC-BY-NC-4.0.

What is here
- `ARCHITECTURE.md` — design + parameter budget for a ~1B smol model
- `src/coconutblt` — minimal, original PyTorch implementation (toy patcher + hybrid transformer)
- `scripts/train_toy.py` — tiny test training run used to validate the scaffold
- `scripts/train.py` — a slightly more featureful trainer wired to `configs/model_1b.yaml`

Quickstart (toy run)

Install requirements (recommended virtualenv / conda):

```bash
pip install -r requirements.txt
```

Run a short toy training (this uses CPU by default and small toy sizes):
```bash
PYTHONPATH=src python3 scripts/train_toy.py --steps 20 --device cpu
```

To try the larger config (simulation only):
```bash
PYTHONPATH=src python3 scripts/train.py --config configs/model_1b.yaml --device cpu --steps 10 --batch_size 2
```

Next steps
- Implement an entropy-driven patcher inspired by BLT (requires consideration of the license if re-using code)
- Add tokenizer integration (HF tokenizer or byte-level tokenizer)
- Add distributed training, FSDP / DeepSpeed recipes for large runs

# coconutblt