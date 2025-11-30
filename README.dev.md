# coconutblt — development README

This workspace is a starter project to build a hybrid Transformer model inspired by:

- Facebook Research: Coconut (continuous latent reasoning) — MIT
- Facebook Research: BLT (byte-latent transformer / patching) — CC-BY-NC-4.0

This repository contains a minimal PyTorch scaffold that demonstrates how to combine
patch-level input processing with a continuous-latent module in a Transformer stack.

Files of interest
- `ARCHITECTURE.md` — design doc and parameter budget for a ~1B model
- `src/coconutblt` — package with a toy hybrid model implementation
- `scripts/train_toy.py` — an example script demonstrating a tiny training run
- `requirements.txt` — minimal Python deps

- `scripts/generate.py` — small generation tool to sample from a checkpoint
- `scripts/export_hf.py` — export weights/config to a HuggingFace-style folder

Notes about licensing
- Coconut: MIT — you can re-use with no commercial restriction
- BLT: CC-BY-NC-4.0 — non-commercial license. If you reuse BLT code/components, you must follow CC-BY-NC-4.0.

Next steps
- Expand the patcher to be entropy-driven (inspired by BLT) if you have rights for that code
- Implement a full training recipe (distributed, bf16) matching Coconut's multi-stage curriculum

Learned estimator & Byte↔Patch interaction
- `LearnedEntropyEstimator` — a small trainable estimator that can be attached to `EntropyPatcher`.
- `HybridTransformer` supports `use_byte_patch_attention=True`, which enables intra-patch positional embeddings
	and a simplified attention exchange between token-level bytes and patch-level embeddings (prototype BLT-style behavior).
