"""Simple patcher module (toy implementation inspired by BLT).

The real BLT patcher is entropy-based and dynamically sizes patches.
This implementation provides a fixed-size patcher for reproducible experiments and
easy integration in the hybrid model.
"""

from typing import List, Tuple

import torch
from torch import nn


class FixedPatcher(nn.Module):
    """Group token/byte sequences into fixed-size patches.

    Input: (batch, seq_len) of integer tokens
    Output: patches: (batch, n_patches, patch_size)
    and patch_mask: (batch, n_patches) boolean for padded patches
    """

    def __init__(self, patch_size: int = 8):
        super().__init__()
        assert patch_size >= 1
        self.patch_size = patch_size

    def forward(self, tokens: torch.Tensor, pad_token: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, S)
        B, S = tokens.shape
        pad_len = (-S) % self.patch_size
        if pad_len > 0:
            pad = tokens.new_full((B, pad_len), pad_token)
            tokens = torch.cat([tokens, pad], dim=1)
            S = S + pad_len

        n_patches = S // self.patch_size
        patches = tokens.view(B, n_patches, self.patch_size)  # (B, P, patch_size)
        patch_mask = (patches != pad_token).any(dim=-1)  # True for a non-empty patch
        return patches, patch_mask
