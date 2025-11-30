import subprocess
import sys
import torch
from pathlib import Path


def test_generate_runs(tmp_path):
    from coconutblt import HybridTransformer

    cfg = {'model': {'vocab_size': 256, 'd_model': 64, 'n_layers': 2, 'n_heads': 4, 'patch_size': 4, 'n_latents': 8}}
    model = HybridTransformer(vocab_size=256, d_model=64, n_layers=2, n_heads=4, patch_size=4, n_latents=8)
    ck = {'model_state_dict': model.state_dict(), 'config': cfg}
    ckpt = tmp_path / 'ckpt.pt'
    torch.save(ck, ckpt)

    env = dict(**{**dict(), **{'PYTHONPATH': 'src'}})
    cmd = [sys.executable, 'scripts/generate.py', '--ckpt', str(ckpt), '--prompt', 'hi', '--max_new_tokens', '5', '--top_k', '5']
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0
    assert isinstance(r.stdout, str) and len(r.stdout) > 0


def test_generate_top_p_and_beam(tmp_path):
    from coconutblt import HybridTransformer

    cfg = {'model': {'vocab_size': 256, 'd_model': 64, 'n_layers': 2, 'n_heads': 4, 'patch_size': 1, 'n_latents': 0}}
    model = HybridTransformer(vocab_size=256, d_model=64, n_layers=2, n_heads=4, patch_size=1, n_latents=0)
    ck = {'model_state_dict': model.state_dict(), 'config': cfg}
    ckpt = tmp_path / 'ckpt2.pt'
    torch.save(ck, ckpt)

    env = dict(**{**dict(), **{'PYTHONPATH': 'src'}})
    # top-p sampling
    cmd = [sys.executable, 'scripts/generate.py', '--ckpt', str(ckpt), '--prompt', 'hi', '--max_new_tokens', '3', '--top_p', '0.9']
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0

    # beam search
    cmd = [sys.executable, 'scripts/generate.py', '--ckpt', str(ckpt), '--prompt', 'hi', '--max_new_tokens', '3', '--beam', '2']
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0
