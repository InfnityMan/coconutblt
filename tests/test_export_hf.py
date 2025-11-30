import subprocess
import sys
import torch
from pathlib import Path


def test_export_hf(tmp_path):
    from coconutblt import HybridTransformer

    cfg = {'model': {'vocab_size': 32, 'd_model': 32, 'n_layers': 1, 'n_heads': 2, 'patch_size': 4}}
    model = HybridTransformer(vocab_size=32, d_model=32, n_layers=1, n_heads=2, patch_size=4)
    ck = {'model_state_dict': model.state_dict(), 'config': cfg}
    ckpt = tmp_path / 'ckpt2.pt'
    torch.save(ck, ckpt)

    outdir = tmp_path / 'hf'
    env = dict(**{**dict(), **{'PYTHONPATH': 'src'}})
    cmd = [sys.executable, 'scripts/export_hf.py', '--ckpt', str(ckpt), '--out', str(outdir)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0
    assert (outdir / 'pytorch_model.bin').exists()
    assert (outdir / 'config.json').exists()
