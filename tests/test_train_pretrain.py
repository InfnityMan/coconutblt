import subprocess
import sys
import yaml
from pathlib import Path


def test_train_pretrain(tmp_path):
    cfg = {
        'model': {
            'vocab_size': 256,
            'd_model': 64,
            'n_layers': 1,
            'n_heads': 4,
            'patch_size': 4,
            'patcher': 'entropy',
            'entropy_patcher': {
                'min_patch_size': 2,
                'max_patch_size': 4,
                'lookahead': 2,
                'entropy_threshold': 1.0,
                'estimator': 'learned',
                'estimator_emb': 16,
                'estimator_hidden': 32,
            }
        },
        'training': {
            'lr': 1e-3,
            'train_estimator': True,
            'est_lr': 1e-3,
            'est_pretrain_steps': 2,
            'est_warmup_steps': 2,
            'est_loss_weight': 0.1,
        }
    }

    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / 'ckpt'
    cmd = [sys.executable, 'scripts/train.py', '--config', str(cfg_path), '--out', str(out_dir), '--device', 'cpu', '--steps', '3', '--batch_size', '8']
    env = dict(**{**dict(), **{'PYTHONPATH': 'src'}})
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print('STDOUT', r.stdout)
    print('STDERR', r.stderr)
    assert r.returncode == 0
    assert (out_dir / 'final.pt').exists()
