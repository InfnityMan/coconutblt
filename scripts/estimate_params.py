#!/usr/bin/env python3
"""Parameter estimation utility for a decoder-only transformer.

This is a rough estimator (not exact) for quick exploration. It assumes a decoder-only
architecture with attention projections and a 2-layer MLP per block.
"""
import math
import argparse


def estimate_params(d_model, n_layers, mlp_ratio=4.0, vocab_size=256):
    H = d_model
    L = n_layers
    mlp_hidden = int(H * mlp_ratio)
    # approximate per-layer: attn (4*H^2) + mlp (8*H^2)
    per_layer = 12 * (H ** 2)
    total = L * per_layer
    # embeddings / head (vocab*H * 2 approx)
    total += 2 * vocab_size * H
    return total


def human(n):
    for unit in ['','K','M','B','T']:
        if abs(n) < 1000.0:
            return "%3.2f%s" % (n, unit)
        n /= 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=2048)
    parser.add_argument('--n_layers', type=int, default=20)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--vocab', type=int, default=256)
    args = parser.parse_args()

    total = estimate_params(args.d_model, args.n_layers, args.mlp_ratio, args.vocab)
    print('Estimated parameters: ', total)
    print('Human: ', human(total))


if __name__ == '__main__':
    main()
