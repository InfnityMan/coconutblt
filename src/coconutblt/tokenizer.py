"""Very small byte-level tokenizer utilities.

This tokenizer is intentionally minimal: it maps bytes (0-255) directly to integers.
If you want to use a real tokenization strategy, swap this for a HuggingFace tokenizer.
"""

from typing import List


class ByteTokenizer:
    def __init__(self, add_pad=True, pad_token=0):
        self.vocab_size = 256
        self.pad_token = pad_token
        self.add_pad = add_pad

    def encode(self, text: str) -> List[int]:
        b = text.encode('utf-8', errors='ignore')
        return [int(x) for x in b]

    def decode(self, ints: List[int]) -> str:
        try:
            return bytes(ints).decode('utf-8', errors='ignore')
        except Exception:
            return ''
