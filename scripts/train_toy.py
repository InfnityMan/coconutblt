#!/usr/bin/env python3
"""Toy training script for the minimal HybridTransformer.

This is a tiny example to verify the scaffold runs. It creates random data and trains
for a few small steps locally.
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from coconutblt import HybridTransformer, FixedPatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--patcher', type=str, default='fixed', choices=('fixed', 'entropy'))
    # entropy patcher options
    parser.add_argument('--min_patch_size', type=int, default=2)
    parser.add_argument('--max_patch_size', type=int, default=8)
    parser.add_argument('--lookahead', type=int, default=4)
    parser.add_argument('--entropy_threshold', type=float, default=3.0)
    parser.add_argument('--use_learned_estimator', action='store_true', help='Use a small learned estimator for entropy predictions')
    parser.add_argument('--n_latents', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--train_estimator', action='store_true', help='If set will train the learned entropy estimator with MSE-on-Shannon auxiliary loss')
    parser.add_argument('--est_loss_weight', type=float, default=1.0, help='Weight for estimator auxiliary loss')
    parser.add_argument('--est_lr', type=float, default=1e-3, help='Learning rate for estimator optimizer')
    parser.add_argument('--est_pretrain_steps', type=int, default=0, help='Steps to pretrain estimator before joint training')
    parser.add_argument('--est_warmup_steps', type=int, default=0, help='Warmup steps to ramp estimator loss weight')
    args = parser.parse_args()

    device = torch.device(args.device)

    # toy dataset: random sequences of length patches * patch_size
    n_patches = 8
    seq_len = n_patches * args.patch_size
    n_items = 256

    data = torch.randint(low=1, high=args.vocab_size, size=(n_items, seq_len), dtype=torch.long)
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = HybridTransformer(vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                               n_heads=args.n_heads, patch_size=args.patch_size, n_latents=args.n_latents)
    model.to(device)

    if args.patcher == 'fixed':
        patcher = FixedPatcher(patch_size=args.patch_size)
    else:
        from coconutblt import EntropyPatcher
        estimator = None
        if args.use_learned_estimator:
            from coconutblt.entropy_patcher import LearnedEntropyEstimator
            estimator = LearnedEntropyEstimator(vocab_size=args.vocab_size, emb_dim=64, hidden=128)

        patcher = EntropyPatcher(min_patch_size=args.min_patch_size,
                                 max_patch_size=args.max_patch_size,
                                 lookahead=args.lookahead,
                                 entropy_threshold=args.entropy_threshold)
        if estimator is not None:
            patcher.estimator = estimator

    # optimizer for model; separate optimizer for estimator (if training)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    est_opt = None
    if args.use_learned_estimator and args.train_estimator and estimator is not None:
        est_opt = torch.optim.AdamW(estimator.parameters(), lr=args.est_lr)

    loss_f = torch.nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    steps = 0
    # optionally pretrain estimator only
    if est_opt is not None and args.est_pretrain_steps > 0:
        psteps = 0
        while psteps < args.est_pretrain_steps:
            for batch in dl:
                tokens = batch[0].to(device)
                est_opt.zero_grad()
                est_loss = estimator.mse_loss_against_shannon(tokens, args.lookahead)
                est_loss.backward()
                est_opt.step()
                psteps += 1
                if psteps >= args.est_pretrain_steps:
                    break
    while steps < args.steps:
        for batch in dl:
            tokens = batch[0].to(device)
            patches, patch_mask = patcher(tokens)
            # patches: (B, P, S)
            logits = model(patches, patch_mask)
            # logits: (B, P, V)
            # make targets: predict the first token in each patch
            targets = patches[:, :, 0]
            B, P, V = logits.shape
            loss = loss_f(logits.view(B * P, V), targets.contiguous().view(B * P))

            # if we train estimator, compute its self-supervised MSE loss and add it
            if est_opt is not None:
                est_loss = estimator.mse_loss_against_shannon(tokens, args.lookahead)
                # warmup for estimator loss
                if args.est_warmup_steps > 0:
                    cur_warm = min(steps, args.est_warmup_steps)
                    est_weight = args.est_loss_weight * (cur_warm / float(args.est_warmup_steps))
                else:
                    est_weight = args.est_loss_weight
                total_loss = loss + est_weight * est_loss
            else:
                total_loss = loss

            opt.zero_grad()
            if est_opt is not None:
                est_opt.zero_grad()

            total_loss.backward()

            opt.step()
            if est_opt is not None:
                est_opt.step()

            steps += 1
            if steps % 10 == 0:
                print(f"step={steps} loss={loss.item():.4f}")
            if steps >= args.steps:
                break

    print("Finished", time.time() - t0)


if __name__ == '__main__':
    main()
