"""Entropy-driven patcher (original implementation inspired by BLT concepts).

This patcher creates variable-length patches (up to `max_patch_size`). It chooses boundaries
based on a simple local Shannon-entropy heuristic computed on the next `lookahead` bytes.

Behaviour (per sequence):
- Start at position i=0.
- Grow the current patch at least `min_patch_size` tokens.
- After reaching `min_patch_size`, examine the entropy of the next `lookahead` bytes starting at
  the candidate boundary. If the entropy is below `entropy_threshold`, the region is considered
  ``simple'' and the patch is allowed to grow; otherwise the patch is cut here.
- Repeat until the end of the sequence. Patches smaller than `max_patch_size` are allowed.

For batch inputs, we compute patches independently per sequence and pad all patches to `max_patch_size`.

Note: this is a lightweight, original reimplementation for research / prototype use — it is NOT a copy
of any BLT source code. It is intended to let you try adaptive patching without bringing BLT code into this
workspace (BLT is CC-BY-NC-4.0).
"""

from typing import List, Tuple, Optional

import torch
from torch import nn


def shannon_entropy_of_window(window: torch.Tensor) -> float:
    # window: 1D tensor of integers
    if window.numel() == 0:
        return 0.0
    vals, counts = torch.unique(window, return_counts=True)
    probs = counts.float() / counts.sum()
    # -sum(p * log2(p))
    ent = -(probs * torch.log2(probs)).sum().item()
    return float(ent)


class LearnedEntropyEstimator(nn.Module):
    """A tiny, trainable estimator that predicts entropy (or a proxy) for sliding windows.

    This lightweight estimator maps byte embeddings through a small conv/MLP stack to
    produce a scalar per position (representing estimated entropy or boundary score).

    The estimator is intended for prototyping; for production you may replace it with
    a larger learned model, or pretrain it to mimic true Shannon entropy on a dataset.
    """

    def __init__(self, vocab_size: int = 256, emb_dim: int = 64, hidden: int = 128, kernel: int = 5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.conv = nn.Conv1d(emb_dim, hidden, kernel_size=kernel, padding=kernel // 2)
        self.act = nn.GELU()
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, S) -> returns scores: (B, S) scalar estimates
        emb = self.emb(tokens)  # (B, S, E)
        x = emb.transpose(1, 2)  # (B, E, S)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, S, hidden)
        x = self.act(x)
        out = self.head(x).squeeze(-1)
        return out

    def mse_loss_against_shannon(self, tokens: torch.Tensor, lookahead: int) -> torch.Tensor:
        # compute target Shannon entropies for lookahead ahead-of-position windows
        B, S = tokens.shape
        device = tokens.device
        target = torch.zeros((B, S), dtype=torch.float32, device=device)
        for b in range(B):
            seq = tokens[b]
            for i in range(S):
                start = i
                end = min(S, i + lookahead)
                if start >= end:
                    target[b, i] = 0.0
                else:
                    target[b, i] = shannon_entropy_of_window(seq[start:end])

        pred = self.forward(tokens)
        return torch.nn.functional.mse_loss(pred, target)


class EntropyPatcher(nn.Module):
    """A simple local-entropy-based patcher.

    Parameters
    - min_patch_size: smallest allowed patch length
    - max_patch_size: largest allowed patch length (patches are padded to this width)
    - lookahead: number of bytes to consider when computing the candidate boundary entropy
    - entropy_threshold: if the lookahead entropy is *below* this threshold, we grow the patch,
                         otherwise we break (high entropy -> smaller patches)
    - pad_token: token used to pad tokens/patches
    """

    def __init__(self, min_patch_size: int = 2, max_patch_size: int = 16, lookahead: int = 4,
                 entropy_threshold: float = 3.0, pad_token: int = 0, estimator: Optional[nn.Module] = None):
        super().__init__()
        assert 1 <= min_patch_size <= max_patch_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.lookahead = lookahead
        self.entropy_threshold = entropy_threshold
        self.pad_token = pad_token
        self.estimator = estimator

    def _patch_one(self, seq: torch.Tensor, preds_seq: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        # seq: 1D tensor of tokens
        S = seq.shape[0]
        i = 0
        patches: List[torch.Tensor] = []

        while i < S:
            # base size at least min_patch_size
            size = min(self.min_patch_size, S - i)

            # try to grow patch up to max_patch_size based on local entropy
            while size < self.max_patch_size and (i + size) < S:
                # compute entropy of lookahead bytes starting at candidate boundary i+size
                start = i + size
                # if we have a learned estimator, use its predicted entropy at `start`
                if self.estimator is not None and preds_seq is not None:
                    ent = float(preds_seq[start].item())
                else:
                    end = min(S, start + self.lookahead)
                    look = seq[start:end]
                    ent = shannon_entropy_of_window(look)
                # if entropy is below threshold, region is 'simple', grow patch
                if ent < self.entropy_threshold:
                    size = min(size + 1, S - i)
                    # continue growing until threshold triggers or max reached
                    continue
                else:
                    # high entropy -> break the patch here
                    break

            patch = seq[i : i + size]
            patches.append(patch)
            i += size

        return patches

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, S)
        B, S = tokens.shape
        all_patches: List[List[torch.Tensor]] = []
        max_patches = 0

        # if using an estimator, precompute predictions over the full sequence
        if self.estimator is not None:
            with torch.no_grad():
                preds = self.estimator(tokens)  # (B, S) predicted scores
        else:
            preds = None

        for b in range(B):
            seq = tokens[b]
            patches = self._patch_one(seq, preds[b] if preds is not None else None)
            all_patches.append(patches)
            if len(patches) > max_patches:
                max_patches = len(patches)

        # pad per-batch patches to have same number of patches and same patch width
        padded_patches = tokens.new_full((B, max_patches, self.max_patch_size), fill_value=self.pad_token)
        patch_mask = torch.zeros((B, max_patches), dtype=torch.bool, device=tokens.device)

        for b in range(B):
            patches = all_patches[b]
            for p_idx, patch in enumerate(patches):
                L = patch.shape[0]
                L = min(L, self.max_patch_size)
                padded_patches[b, p_idx, :L] = patch[:L]
                patch_mask[b, p_idx] = True

        return padded_patches, patch_mask
