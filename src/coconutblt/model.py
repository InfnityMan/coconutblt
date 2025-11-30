"""Hybrid Transformer skeleton combining BLT-like patching and Coconut-style continuous latent vectors.

This is a minimal, educational implementation meant to be a starting point for experiments.
It intentionally avoids using any BLT code verbatim and implements original components inspired
by the concepts in the two research repos.
"""

from typing import Optional

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        res = x
        x = self.ln1(x)
        x2, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = res + x2
        res = x
        x = self.ln2(x)
        x = res + self.mlp(x)
        return x


class ContinuousLatentModule(nn.Module):
    """A small continuous-latent bank that interacts with token embeddings via cross-attention.

    The module holds `n_latents` learnable vectors and exposes a forward that performs two operations:
    - latents read tokens (latents attend over tokens)
    - tokens read latents (tokens attend over latents)
    """

    def __init__(self, n_latents: int, latent_dim: int, n_heads: int):
        super().__init__()
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.02)
        self.ln_tokens = nn.LayerNorm(latent_dim)
        self.ln_latents = nn.LayerNorm(latent_dim)
        self.read_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True)
        self.write_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        # tokens: (B, T, D) ; latents: (n_latents, D)
        B, T, D = tokens.shape
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, n_latents, D)

        # latents read tokens -> enrich latents with info from tokens
        q = self.ln_latents(latents)
        k = self.ln_tokens(tokens)
        v = k
        latents2, _ = self.read_attn(q, k, v, key_padding_mask=(~token_mask) if token_mask is not None else None)
        latents = latents + latents2

        # tokens read latents -> tokens incorporate latent knowledge
        q_t = self.ln_tokens(tokens)
        k_l = self.ln_latents(latents)
        v_l = k_l
        tokens2, _ = self.write_attn(q_t, k_l, v_l)
        tokens = tokens + tokens2

        return tokens, latents


class HybridTransformer(nn.Module):
    """A skeleton hybrid model.

    Configurable so a small toy variant (for tests) or a larger 1B variant can be used.
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 128, n_layers: int = 6, n_heads: int = 8,
                 mlp_ratio: float = 4.0, patch_size: int = 4,
                 n_latents: int = 8, latent_dim: Optional[int] = None,
                 latent_interval: int = 4,
                 use_byte_patch_attention: bool = False,
                 max_patch_pos: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.latent_interval = latent_interval

        self.token_embed = nn.Embedding(vocab_size, d_model)
        # patch encoder: pool token embeddings inside a patch (mean pool) then project to model dim
        # this supports variable-width patches (padded to a fixed max width by the patcher)
        self.patch_encoder = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, mlp_ratio=mlp_ratio) for _ in range(n_layers)])

        self.n_latents = n_latents
        self.latent_dim = latent_dim or d_model
        if self.latent_dim != d_model:
            # optional projection to allow different latent dims
            self.latent_proj = nn.Linear(self.latent_dim, d_model)
        else:
            self.latent_proj = None

        if n_latents > 0:
            self.latent_module = ContinuousLatentModule(n_latents, self.latent_dim, n_heads)
        else:
            self.latent_module = None

        # optional byte<->patch attention for richer interaction
        self.use_byte_patch_attention = use_byte_patch_attention
        if self.use_byte_patch_attention:
            # intra-patch positional embedding for token positions inside a patch
            self.intra_patch_pos = nn.Embedding(patch_size, d_model)
            # per-patch position (e.g., patch index in sequence)
            self.patch_pos = nn.Embedding(max_patch_pos, d_model)
            # attention modules to exchange information between tokens and patch tokens
            self.byte_to_patch_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
            self.patch_to_byte_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def encode_patches(self, tokens: torch.Tensor):
        # tokens: (B, n_patches, patch_size) - may include padding inside each patch
        B, P, S = tokens.shape
        tok_emb = self.token_embed(tokens)  # (B, P, S, D)
        # if using intra-patch pos emb add it (positions 0..S-1)
        B, P, S = tokens.shape
        if self.use_byte_patch_attention:
            pos_idx = torch.arange(S, device=tokens.device).unsqueeze(0).unsqueeze(0)  # (1,1,S)
            pos_emb = self.intra_patch_pos(pos_idx.squeeze(0))  # (1,S,D) -> broadcasting ok
            tok_emb = tok_emb + pos_emb.unsqueeze(0)

        # compute token-level mask (assume pad_token == 0)
        pad_mask = (tokens != 0).unsqueeze(-1)  # (B, P, S, 1)
        tok_emb = tok_emb * pad_mask
        denom = pad_mask.sum(dim=2).clamp_min(1)  # (B, P, 1)
        pooled = tok_emb.sum(dim=2) / denom  # (B, P, D)
        patches = self.patch_encoder(pooled)

        # add patch position embedding so patches know their relative position
        if self.use_byte_patch_attention:
            # patch indices (0..P-1)
            patch_idx = torch.arange(P, device=tokens.device).unsqueeze(0)  # (1,P)
            patches = patches + self.patch_pos(patch_idx)
        return patches

    def forward(self, patches_tokens: torch.Tensor, patch_mask: Optional[torch.Tensor] = None):
        # patches_tokens: (B, P, patch_size)
        x = self.encode_patches(patches_tokens)
        # x: (B, P, D)

        token_mask = patch_mask if patch_mask is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        # optionally run byte<->patch attention exchange
        if self.use_byte_patch_attention:
            # tokens: (B, P, S) -> token embeddings (B, P, S, D)
            tok_emb = self.token_embed(patches_tokens)  # re-embed tokens
            # reapply intra-patch pos emb
            S = patches_tokens.shape[2]
            pos_idx = torch.arange(S, device=patches_tokens.device).unsqueeze(0).unsqueeze(0)
            tok_emb = tok_emb + self.intra_patch_pos(pos_idx.squeeze(0)).unsqueeze(0)

            token_padding = (patches_tokens == 0)  # (B, P, S)
            # reshape to (B*P, S, D) and patches to (B*P, 1, D)
            Bn, Pn, Sn = tok_emb.shape[0], tok_emb.shape[1], tok_emb.shape[2]
            tok_flat = tok_emb.reshape(Bn * Pn, Sn, self.d_model)
            patch_flat = x.reshape(Bn * Pn, 1, self.d_model)
            token_mask_flat = token_padding.reshape(Bn * Pn, Sn)

            # patches read tokens -> enrich patches
            patch_read, _ = self.byte_to_patch_attn(patch_flat, tok_flat, tok_flat, key_padding_mask=~token_mask_flat)
            patch_read = patch_read.reshape(Bn, Pn, self.d_model)
            x = x + patch_read

            # tokens read patches -> tokens incorporate patch info
            token_read, _ = self.patch_to_byte_attn(tok_flat, patch_flat, patch_flat)
            token_read = token_read.reshape(Bn, Pn, Sn, self.d_model)
            # optionally merge token-level info (not used downstream currently)
            # For now, ignore token-level outputs except verifying shapes (keeps model simple)

        for i, block in enumerate(self.blocks):
            x = block(x, key_padding_mask=~token_mask)
            # every latent_interval layers, run latent module
            if self.latent_module is not None and (i + 1) % self.latent_interval == 0:
                x, latents = self.latent_module(x, token_mask)
                # if latent_dim differs from d_model optionally project
                if hasattr(self, 'latent_proj') and self.latent_proj is not None:
                    x = x + self.latent_proj(latents.mean(dim=1, keepdim=True))

        x = self.ln_final(x)
        logits = self.lm_head(x)  # logits per patch
        return logits
