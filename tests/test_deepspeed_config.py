import json
from pathlib import Path


def test_deepspeed_config_exists():
    p = Path('configs/deepspeed_1b.json')
    assert p.exists()
    cfg = json.loads(p.read_text())
    assert 'optimizer' in cfg
    assert 'zero_optimization' in cfg
