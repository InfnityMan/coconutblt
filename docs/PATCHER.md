# Entropy-driven patcher (prototype)

This repository now includes an original prototype of an entropy-driven patcher implemented at `src/coconutblt/entropy_patcher.py`.

Overview
- The patcher scans a byte-level sequence and grows variable-length patches up to a configurable `max_patch_size`.
- After reaching `min_patch_size`, the patcher evaluates the Shannon entropy of a small `lookahead` window at candidate boundaries.
- If the entropy is low (below `entropy_threshold`) the patch grows; if entropy is high the patch boundary is placed there (shorter patch).

Why this is useful
- Low-entropy regions (e.g. long runs of ASCII text) are grouped into larger patches, which allows the model to operate at a coarser granularity and save compute.
- High-entropy regions (e.g. code, non-compressible bytes) are segmented into smaller patches for finer-grained modelling.

How to try it
- The `scripts/train_toy.py` accepts `--patcher entropy` and patcher parameters such as `--max_patch_size`, `--min_patch_size`, `--lookahead`, and `--entropy_threshold`.

Learned estimator option
- The entropy patcher supports using a small learned estimator rather than the Shannon heuristic.
- Pass an estimator instance when constructing `EntropyPatcher` (e.g., `est = LearnedEntropyEstimator()` and `EntropyPatcher(..., estimator=est)`).
- The estimator exposes `mse_loss_against_shannon(tokens, lookahead)` which can be used as a self-supervised pretraining signal or online auxiliary loss during training.

Notes
- This is an original implementation (not copied from BLT). BLT has a more sophisticated entropy model and additional mechanisms — if you plan to use BLT code, remember it is CC-BY-NC-4.0.
