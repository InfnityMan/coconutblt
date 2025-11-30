import os
import sys
import tempfile
import torch

from coconutblt import HybridTransformer


def _write_ckpt(path, model, cfg=None):
    data = {'model_state_dict': model.state_dict(), 'config': cfg or {}}
    torch.save(data, path)


def test_generate_runs(monkeypatch, tmp_path):
    # create tiny model and checkpoint
    model = HybridTransformer(vocab_size=256, d_model=32, n_layers=2, n_heads=4, patch_size=1, n_latents=0)
    ckpt = tmp_path / 'ckpt.pt'
    cfg = {'model': {'vocab_size': 256, 'd_model': 32, 'n_layers': 2, 'n_heads': 4}}
    _write_ckpt(str(ckpt), model, cfg)

    # run generate.py main by patching argv
    import subprocess
    env = dict(**{**dict(), **{'PYTHONPATH': 'src'}})
    cmd = [sys.executable, 'scripts/generate.py', '--ckpt', str(ckpt), '--prompt', 'Hi', '--max_new_tokens', '3']
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0


def test_export_hf_creates_files(tmp_path):
    model = HybridTransformer(vocab_size=256, d_model=32, n_layers=2, n_heads=4, patch_size=1, n_latents=0)
    ckpt = tmp_path / 'ckpt2.pt'
    cfg = {'model': {'vocab_size': 256}}
    _write_ckpt(str(ckpt), model, cfg)

    out = tmp_path / 'hf_out'
    import subprocess
    env = dict(**{**dict(), **{'PYTHONPATH': 'src'}})
    cmd = [sys.executable, 'scripts/export_hf.py', '--ckpt', str(ckpt), '--out', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0

    assert (out / 'pytorch_model.bin').exists()
    assert (out / 'config.json').exists()
    assert (out / 'tokenizer.json').exists()
