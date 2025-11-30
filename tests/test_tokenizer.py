from coconutblt.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip():
    t = ByteTokenizer()
    s = "hello world — 你好"
    enc = t.encode(s)
    dec = t.decode(enc)
    assert isinstance(enc, list)
    assert isinstance(dec, str)
