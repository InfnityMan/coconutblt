#!/usr/bin/env python3
"""Robust generation script.

Features:
- byte-mode (default) uses patch_size=1 and generates autoregressively byte-by-byte
- supports temperature, top-k, top-p (nucleus) sampling, greedy sampling and beam search
- patch-mode is supported (if model was trained to generate full patches) but byte-mode is
  the recommended and tested path.
"""
import argparse
import json
import os
import math
import torch

from coconutblt.tokenizer import ByteTokenizer
from coconutblt.patcher import FixedPatcher
from coconutblt import HybridTransformer


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu')
    cfg = data.get('config', {})
    state = data.get('model_state_dict', data)
    return cfg, state


def build_model_from_cfg(cfg, patch_size_override=None):
    mc = cfg.get('model', {})
    patch_size = mc.get('patch_size', 1 if patch_size_override is None else patch_size_override)
    model = HybridTransformer(vocab_size=mc.get('vocab_size', 256),
                              d_model=mc.get('d_model', 128),
                              n_layers=mc.get('n_layers', 6),
                              n_heads=mc.get('n_heads', 8),
                              patch_size=patch_size,
                              n_latents=mc.get('n_latents', 0),
                              use_byte_patch_attention=mc.get('use_byte_patch_attention', False))
    return model


def top_k_logits(logits: torch.Tensor, k: int):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)


def top_p_logits(logits: torch.Tensor, p: float):
    """Apply nucleus (top-p) masking to logits. logits shape (B, V) returned masked.
    """
    if p <= 0 or p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    # create mask of tokens to keep
    keep = cumprobs <= p
    # ensure at least one token is kept per row
    keep[..., 0] = True
    # set to -inf for tokens we drop
    mask = ~keep
    filtered = sorted_logits.masked_fill(mask, -1e10)
    # scatter back to original order
    rev = torch.empty_like(filtered).scatter_(1, sorted_indices, filtered)
    return rev


def beam_search_step(model, cur_tokens, patcher, device, beam_size=3, max_new_tokens=10):
    """Simple beam search over bytes. cur_tokens is a list of token lists per beam.
    Returns the best sequence after max_new_tokens steps.
    This is a toy beam search that expands top-k vocabulary per beam to maintain tractability.
    """
    beams = [(cur_tokens, 0.0)]  # (tokens_list, logprob)
    for _ in range(max_new_tokens):
        new_beams = []
        for seq, score in beams:
            cur = torch.tensor([seq], dtype=torch.long).to(device)
            patches, mask = patcher(cur)
            with torch.no_grad():
                logits = model(patches, mask)  # (1, P, V)
            last_logits = logits[0, -1, :]
            logprobs = torch.log_softmax(last_logits, dim=-1)
            # pick top-k candidates from logits
            topk = 8
            vals, idxs = torch.topk(logprobs, topk)
            for v, i in zip(vals.tolist(), idxs.tolist()):
                new_seq = seq + [int(i)]
                new_score = score + float(v)
                new_beams.append((new_seq, new_score))

        # keep top beam_size beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # return best
    return beams[0][0]


def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0):
    # logits: (V,) -> torch.Tensor
    logits = logits.unsqueeze(0)
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p > 0.0 and top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--prompt', type=str, default='Hello world')
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.0, help='Nucleus sampling: cumulative probability to keep')
    parser.add_argument('--beam', type=int, default=0, help='Beam search width (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--beam', type=int, default=0, help='Beam size (0 means disable beam)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['byte', 'patch'], default='byte')
    parser.add_argument('--patch_size', type=int, default=None, help='Override patch size for generation')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg, state = load_checkpoint(args.ckpt)

    # build model
    model = build_model_from_cfg(cfg, patch_size_override=args.patch_size)
    try:
        model.load_state_dict(state)
    except Exception as e:
        print('Warning: strict state_dict load failed, trying non-strict load:', e)
        model.load_state_dict(state, strict=False)

    model.to(args.device)
    model.eval()

    tokenizer = ByteTokenizer()
    # Default to byte-mode (patch_size=1) for autoregressive correctness
    patch_size = 1 if args.mode == 'byte' else (args.patch_size or cfg.get('model', {}).get('patch_size', 1))
    patcher = FixedPatcher(patch_size=patch_size)

    # seed prompt
    tokens = tokenizer.encode(args.prompt)

    if args.beam > 0:
        # beam search mode (byte-level)
        assert args.mode == 'byte', 'beam search supported in byte mode only'
        best = beam_search_step(model, tokens, patcher, args.device, beam_size=args.beam, max_new_tokens=args.max_new_tokens)
        out = tokenizer.decode(best)
        print(out)
        return

    # sampling / greedy mode (byte-level recommended)
    generated = []
    for _ in range(args.max_new_tokens):
        cur = torch.tensor([tokens], dtype=torch.long).to(args.device)
        patches, mask = patcher(cur)
        with torch.no_grad():
            logits = model(patches, mask)
            last_logits = logits[0, -1, :]

        if args.greedy:
            idx = int(torch.argmax(last_logits).item())
        else:
            idx = sample_from_logits(last_logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

        generated.append(idx)
        tokens.append(idx)

    out = tokenizer.decode(tokens)
    print(out)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Generation script for HybridTransformer checkpoints.

Loads a checkpoint and performs autoregressive byte-level sampling. This is a small
utility for testing and demonstrations.
"""
import argparse
import json
import os
import torch

from coconutblt.tokenizer import ByteTokenizer
from coconutblt.patcher import FixedPatcher
from coconutblt import HybridTransformer


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu')
    cfg = data.get('config', {})
    state = data.get('model_state_dict', data)
    return cfg, state


def build_model_from_cfg(cfg, patch_size_override=None):
    mc = cfg.get('model', {})
    patch_size = mc.get('patch_size', 1 if patch_size_override is None else patch_size_override)
    model = HybridTransformer(vocab_size=mc.get('vocab_size', 256),
                              d_model=mc.get('d_model', 128),
                              n_layers=mc.get('n_layers', 6),
                              n_heads=mc.get('n_heads', 8),
                              patch_size=patch_size,
                              n_latents=mc.get('n_latents', 0),
                              use_byte_patch_attention=mc.get('use_byte_patch_attention', False))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--prompt', type=str, default='Hello world')
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patch_size', type=int, default=None, help='Override patch size for generation (default: use model cfg or 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg, state = load_checkpoint(args.ckpt)

    model = build_model_from_cfg(cfg, patch_size_override=args.patch_size)
    try:
        model.load_state_dict(state)
    except Exception as e:
        print('Warning: strict state_dict load failed, trying non-strict load:', e)
        model.load_state_dict(state, strict=False)

    model.to(args.device)
    model.eval()

    tokenizer = ByteTokenizer()
    patcher = FixedPatcher(patch_size=args.patch_size or cfg.get('model', {}).get('patch_size', 1))

    tokens = tokenizer.encode(args.prompt)
    generated = []

    def top_k_logits_np(logits, k=0):
        if k <= 0:
            return logits
        vals, _ = torch.topk(logits, k)
        min_vals = vals[:, -1].unsqueeze(1)
        return torch.where(logits < min_vals, torch.full_like(logits, -1e10), logits)

    def top_p_logits(logits, p=0.0):
        if p <= 0.0 or p >= 1.0:
            return logits
        # logits: (1, V)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # mask indices where cumulative probability > p
        mask = cumulative > p
        # keep at least one token
        mask[..., 0] = False
        # set masked logits to -inf
        sorted_logits[mask] = -1e10
        # scatter back to original ordering
        unsorted = torch.empty_like(sorted_logits)
        unsorted.scatter_(1, sorted_indices, sorted_logits)
        return unsorted

    def greedy_step(logits):
        return torch.argmax(logits, dim=-1).item()

    def beam_search_step(model, cur_tokens, generated, beam_width):
        # cur_tokens: list of token lists (beams)
        # perform one step of beam expansion and return new beams
        candidates = []
        for tokens in cur_tokens:
            cur = torch.tensor([tokens], dtype=torch.long).to(args.device)
            patches, mask = patcher(cur)
            with torch.no_grad():
                logits = model(patches, mask)
                last_logits = logits[0, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                topv, topidx = torch.topk(probs, beam_width)
                for v, idx in zip(topv.tolist(), topidx.tolist()):
                    candidates.append((tokens + [int(idx)], float(v)))

        # select top beam_width candidates by score (value)
        candidates.sort(key=lambda x: x[1], reverse=True)
        new_beams = [c[0] for c in candidates[:beam_width]]
        return new_beams

    # beam search path
    if args.beam and args.beam > 1:
        beams = [tokens]
        for _ in range(args.max_new_tokens):
            beams = beam_search_step(model, beams, generated, args.beam)
        # pick the top beam
        tokens = beams[0]
    else:
        for _ in range(args.max_new_tokens):
            cur = torch.tensor([tokens], dtype=torch.long).to(args.device)
            patches, mask = patcher(cur)
            with torch.no_grad():
                logits = model(patches, mask)
                last_logits = logits[0, -1, :].unsqueeze(0)

                # apply nucleus / top-p first
                if args.top_p and args.top_p > 0.0:
                    filtered = top_p_logits(last_logits, args.top_p)
                else:
                    filtered = last_logits

                # apply top-k
                filtered = top_k_logits_np(filtered, args.top_k)

                # sampling / greedy
                if args.top_k == 0 and args.top_p == 0.0:
                    # greedy by temperature adjusted logits
                    if args.temperature == 0.0:
                        idx = greedy_step(filtered)
                    else:
                        probs = torch.softmax(filtered / max(1e-8, args.temperature), dim=-1)
                        idx = torch.multinomial(probs, num_samples=1).item()
                else:
                    probs = torch.softmax(filtered / max(1e-8, args.temperature), dim=-1)
                    idx = torch.multinomial(probs, num_samples=1).item()

                generated.append(idx)
                tokens.append(idx)

    out = tokenizer.decode(tokens)
    print(out)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Simple generation script for the HybridTransformer scaffold.

This script loads a saved checkpoint (output of `scripts/train.py`) and performs
an autoregressive generation loop at the byte level using the model's predictions.

This is a minimal, educational generator for testing; for production-grade sampling you
would integrate a proper autoregressive decoder and beam/temperature/prefix handling.
"""
import argparse
import json
import torch

from coconutblt.tokenizer import ByteTokenizer
from coconutblt.patcher import FixedPatcher


def build_model_from_cfg(cfg):
    from coconutblt import HybridTransformer, EntropyPatcher
    mc = cfg.get('model', {})
    model = HybridTransformer(vocab_size=mc.get('vocab_size', 256),
                             d_model=mc.get('d_model', 128),
                             n_layers=mc.get('n_layers', 6),
                             n_heads=mc.get('n_heads', 8),
                             mlp_ratio=mc.get('mlp_ratio', 4.0),
                             patch_size=mc.get('patch_size', 4),
                             n_latents=mc.get('n_latents', 4),
                             latent_dim=mc.get('latent_dim', None),
                             latent_interval=mc.get('latent_interval', 4),
                             use_byte_patch_attention=mc.get('use_byte_patch_attention', False))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/final.pt')
    parser.add_argument('--prompt', type=str, default='Hello')
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    cfg = ck.get('config', {})
    model = build_model_from_cfg(cfg)
    try:
        model.load_state_dict(ck['model_state_dict'])
    except RuntimeError as e:
        print('Warning: strict load failed, falling back to non-strict load:', e)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model.to(args.device)
    model.eval()

    tokenizer = ByteTokenizer()
    # use fixed patcher for generation to keep deterministic control
    patch_size = cfg.get('model', {}).get('patch_size', 4)
    patcher = FixedPatcher(patch_size=patch_size)

    torch.manual_seed(args.seed)

    tokens = tokenizer.encode(args.prompt)
    # generate tokens autoregressively (byte by byte) using the model's last-patch prediction
    for _ in range(args.max_new_tokens):
        t_tensor = torch.tensor([tokens], dtype=torch.long).to(args.device)
        patches, mask = patcher(t_tensor)
        with torch.no_grad():
            logits = model(patches, mask)  # (1, P, V)
            # pick logits from the last patch
            last = logits[0, -1, :]
            probs = torch.softmax(last / max(1e-8, args.temperature), dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            tokens.append(int(idx))

    out = tokenizer.decode(tokens)
    print(out)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Simple generation tool for HybridTransformer checkpoints.

This script loads a checkpoint saved by `scripts/train.py` (pytorch state_dict) and generates
byte-level continuations token-by-token using a FixedPatcher (patch_size 1) for simplicity.

Supports greedy and top-k sampling.
"""
import argparse
import torch
import yaml
import os

from coconutblt import HybridTransformer, FixedPatcher
from coconutblt.tokenizer import ByteTokenizer


def top_k_logits(logits, k=0):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)


def load_ckpt(path):
    data = torch.load(path, map_location='cpu')
    cfg = data.get('config', {})
    state = data.get('model_state_dict', data)
    return cfg, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--prompt', type=str, default='Hello world')
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg, state = load_ckpt(args.ckpt)

    model_cfg = cfg.get('model', {})
    model = HybridTransformer(vocab_size=model_cfg.get('vocab_size', 256),
                             d_model=model_cfg.get('d_model', 128),
                             n_layers=model_cfg.get('n_layers', 6),
                             n_heads=model_cfg.get('n_heads', 8),
                             patch_size=1,
                             n_latents=model_cfg.get('n_latents', 0),
                             use_byte_patch_attention=model_cfg.get('use_byte_patch_attention', False))
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(args.prompt)
    cur = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(args.device)

    patcher = FixedPatcher(patch_size=1)

    generated = []
    for _ in range(args.max_new_tokens):
        patches, mask = patcher(cur)
        logits = model(patches, mask)  # (B, P, V)
        # take logits for last token in sequence (last patch)
        last_logits = logits[0, -1, :].unsqueeze(0)  # (1, V)
        last_logits = top_k_logits(last_logits, k=args.top_k)
        probs = torch.softmax(last_logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        generated.append(idx)
        cur = torch.cat([cur, torch.tensor([[idx]], dtype=torch.long, device=args.device)], dim=1)

    out = tokenizer.decode(tokens + generated)
    print(out)


if __name__ == '__main__':
    main()
