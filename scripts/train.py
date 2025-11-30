#!/usr/bin/env python3
"""Lightweight trainer for HybridTransformer.

This script is meant to be a convenient starting point. For large-scale training you'd
wire in distributed training and FSDP/DeepSpeed.
"""
import argparse
import os
import time
import yaml

import torch

from coconutblt import HybridTransformer, FixedPatcher


def build_model_from_config(cfg):
    mc = cfg.get('model', {})
    return HybridTransformer(vocab_size=mc.get('vocab_size', 256),
                             d_model=mc.get('d_model', 128),
                             n_layers=mc.get('n_layers', 6),
                             n_heads=mc.get('n_heads', 8),
                             mlp_ratio=mc.get('mlp_ratio', 4.0),
                             patch_size=mc.get('patch_size', 4),
                             n_latents=mc.get('n_latents', 4),
                             latent_dim=mc.get('latent_dim', None),
                             latent_interval=mc.get('latent_interval', 4)
                             )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_1b.yaml')
    parser.add_argument('--out', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    device = torch.device(args.device)
    model.to(device)

    # Toy dataset: random data for now
    vocab = cfg.get('model', {}).get('vocab_size', 256)
    patch_size = cfg.get('model', {}).get('patch_size', 8)
    n_patches = 16
    seq_len = patch_size * n_patches
    n_items = 1024

    data = torch.randint(low=1, high=vocab, size=(n_items, seq_len), dtype=torch.long)
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # choose patcher (supports 'fixed' or 'entropy')
    patch_choice = cfg.get('model', {}).get('patcher', 'fixed')
    if patch_choice == 'fixed':
        patcher = FixedPatcher(patch_size=patch_size)
    elif patch_choice == 'entropy':
        from coconutblt import EntropyPatcher
        epcfg = cfg.get('model', {}).get('entropy_patcher', {})
        # optionally build a learned estimator
        estimator = None
        if epcfg.get('estimator', None) == 'learned':
            from coconutblt.entropy_patcher import LearnedEntropyEstimator
            estimator = LearnedEntropyEstimator(vocab_size=cfg.get('model', {}).get('vocab_size', 256),
                                                emb_dim=epcfg.get('estimator_emb', 64), hidden=epcfg.get('estimator_hidden', 128))

        patcher = EntropyPatcher(min_patch_size=epcfg.get('min_patch_size', 2),
                                 max_patch_size=epcfg.get('max_patch_size', patch_size),
                                 lookahead=epcfg.get('lookahead', 4),
                                 entropy_threshold=epcfg.get('entropy_threshold', 3.0),
                                 estimator=estimator)
    else:
        raise ValueError('Unknown patcher: ' + str(patch_choice))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('training', {}).get('lr', 1e-4))
    est_opt = None
    # optionally create estimator optimizer if we will train the estimator
    if 'entropy_patcher' in cfg.get('model', {}) and cfg.get('model', {}).get('entropy_patcher', {}).get('estimator', None) == 'learned':
        if cfg.get('training', {}).get('train_estimator', False):
            est_lr = cfg.get('training', {}).get('est_lr', cfg.get('training', {}).get('lr', 1e-4))
            if estimator is not None:
                est_opt = torch.optim.AdamW(estimator.parameters(), lr=est_lr)
    loss_f = torch.nn.CrossEntropyLoss()

    os.makedirs(args.out, exist_ok=True)

    model.train()
    step = 0
    t0 = time.time()
    # optional estimator pretraining: train estimator alone for a fixed number of steps
    est_pretrain_steps = cfg.get('training', {}).get('est_pretrain_steps', 0)
    est_warmup_steps = cfg.get('training', {}).get('est_warmup_steps', 0)

    # pretrain estimator only
    if est_pretrain_steps > 0 and estimator is not None:
        print(f"Estimator pretraining for {est_pretrain_steps} steps...")
        pretrain_done = 0
        while pretrain_done < est_pretrain_steps:
            for batch in dl:
                tokens = batch[0].to(device)
                if est_opt is None:
                    raise RuntimeError('Estimator pretrain requested but estimator optimizer not configured')
                est_opt.zero_grad()
                est_loss = estimator.mse_loss_against_shannon(tokens, epcfg.get('lookahead', 4))
                est_loss.backward()
                est_opt.step()
                pretrain_done += 1
                if pretrain_done >= est_pretrain_steps:
                    break

    # main training loop
    while step < args.steps:
        for batch in dl:
            tokens = batch[0].to(device)
            patches, mask = patcher(tokens)
            logits = model(patches, mask)
            targets = patches[:, :, 0]
            B, P, V = logits.shape
            loss = loss_f(logits.view(B * P, V), targets.view(B * P))

            # compute optional estimator auxiliary loss
            est_loss = None
            if est_opt is not None and estimator is not None:
                est_loss = estimator.mse_loss_against_shannon(tokens, epcfg.get('lookahead', 4))

            total_loss = loss
            if est_loss is not None:
                # support warmup of estimator weight
                est_base_weight = cfg.get('training', {}).get('est_loss_weight', 1.0)
                if est_warmup_steps > 0:
                    cur_warm = min(step, est_warmup_steps)
                    weight = est_base_weight * (cur_warm / float(est_warmup_steps))
                else:
                    weight = est_base_weight
                total_loss = total_loss + weight * est_loss

            opt.zero_grad()
            if est_opt is not None:
                est_opt.zero_grad()

            total_loss.backward()

            opt.step()
            if est_opt is not None:
                est_opt.step()

            step += 1
            if step % 20 == 0:
                print(f"step {step} loss={loss.item():.4f}")

            if step >= args.steps:
                break

    ckpt = os.path.join(args.out, 'final.pt')
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, ckpt)
    print('Saved checkpoint', ckpt)
    print('Elapsed', time.time() - t0)


if __name__ == '__main__':
    main()
