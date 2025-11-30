#!/usr/bin/env python3
"""Export a trained checkpoint into a lightweight HuggingFace-style layout.

This script writes a `config.json` and `pytorch_model.bin` into an output directory.
The resulting folder can be further adapted to full Transformers conversion if required.
"""
import argparse
import json
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/final.pt')
    parser.add_argument('--out', type=str, default='hf_export')
    args = parser.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    cfg = ck.get('config', {})
    model_state = ck['model_state_dict']

    os.makedirs(args.out, exist_ok=True)
    torch.save(model_state, os.path.join(args.out, 'pytorch_model.bin'))

    # export minimal model config
    model_cfg = cfg.get('model', {})
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(model_cfg, f, indent=2)

    # write a tiny tokenizer.json describing byte-level tokenizer
    tokenizer_json = {
        'tokenizer_type': 'byte',
        'vocab_size': model_cfg.get('vocab_size', 256),
        'pad_token': 0
    }
    with open(os.path.join(args.out, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer_json, f, indent=2)

    print('Exported HF-style files to', args.out)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Export a checkpoint into a simple HF-style directory (config.json + pytorch_model.bin + tokenizer.json).

This is a helpful minimal exporter; full conversion to HuggingFace Transformers model classes
requires mapping the architecture to a Transformers class and is left as an optional next step.
"""
import argparse
import json
import os
import torch

from coconutblt.tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    data = torch.load(args.ckpt, map_location='cpu')
    cfg = data.get('config', {})
    state = data.get('model_state_dict', data)

    os.makedirs(args.out, exist_ok=True)
    torch.save(state, os.path.join(args.out, 'pytorch_model.bin'))
    # minimal config
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # write simple tokenizer metadata
    tok = ByteTokenizer()
    tok_meta = {'type': 'byte', 'vocab_size': tok.vocab_size, 'pad_token': tok.pad_token}
    with open(os.path.join(args.out, 'tokenizer.json'), 'w') as f:
        json.dump(tok_meta, f)

    print('Exported to', args.out)


if __name__ == '__main__':
    main()
