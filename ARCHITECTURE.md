# coconutblt — Design: Coconut + BLT hybrid (1B params)

This design document describes a hybrid transformer architecture inspired by:

- Facebook Research "Coconut" (MIT): continuous latent reasoning and training schedule ideas
- Facebook Research "BLT" (CC-BY-NC-4.0): byte-level patching, entropy-based patching, and patch-aware attention

Important licensing note
- Coconut (MIT) — fully permissive
- BLT (CC-BY-NC-4.0) — non-commercial restriction. Any derived code that re-uses BLT components must comply with CC-BY-NC-4.0. Carefully consider licensing before using or redistributing BLT-derived artifacts.

Goals
- Produce a compact, smol language model of roughly 1B parameters combining BLT-style patching and Coconut-style continuous latent reasoning.

High-level architecture
- Input pipeline
  - Byte-level input (0..255) or text tokens
  - BLT-style patcher (toy default: fixed-size patches or optional entropy-driven patcher)
  - Patch encoder: learn patch embeddings and positional information

- Core transformer
  - Transformer stack (decoder blocks) operating on patch embeddings
  - After every N layers, insert a small Continuous-Latent module (from Coconut)
    - A fixed-length set of continuous latent vectors (c_thought x latent_dim)
    - Cross-attention between patch tokens and latent tokens so the model can write/read continuous thoughts

- Output head
  - Project back to byte logits if modeling bytes, or to tokenizer vocabulary if tokens

Parameter budget (approximation)
- We target ~1B total params. A practical parameterization that meets the budget:
  - Model hidden dimension H = 2048
  - Layers L = 20
  - Attention heads = 16 (head dim = 128)
  - MLP expansion = 4x (MLP dim = 8192)

Rough param estimate (decoder only)
- Layer cost ≈ 12 * H^2
  - L * 12 * H^2 = 20 * 12 * 2048^2 ≈ 1.0e9 params

Extra modules
- Patch encoder (small) + patcher + continuous latent vectors + output head add ~1–20M params depending on implementation choices.

Design decisions / options
- Patcher
  - Default: fixed patch size (helps reproducibility and simplicity)
  - Advanced: implement BLT entropy-driven patcher and integrate the entropy model for adaptive patch sizes

- Continuous latent module
  - c_thought (number of continuous tokens) selectable per task — Coconut uses multiple continuous thoughts to encode reasoning steps
  - Interaction: cross-attention and gating mechanisms to control flow

- Training regimen
  - Multi-stage training (Coconut): CoT initialization then progressive increase in continuous thought length (optional)
  - BF16 / AMP training recommended for GPU-based training
  - Distributed training with torchrun / DeepSpeed / FSDP for >1 GPU

Next: basic repo scaffolding + minimal PyTorch implementation so you can run a tiny training experiment locally / on a small GPU. See the `coconutblt` package for the initial code.
