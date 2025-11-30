import torch

from coconutblt import HybridTransformer, FixedPatcher


def test_hybrid_forward_shape():
    vocab_size = 64
    d_model = 64
    n_layers = 2
    patch_size = 4
    model = HybridTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                              n_heads=4, patch_size=patch_size, n_latents=2)

    B = 2
    n_patches = 5
    seq_len = n_patches * patch_size
    tokens = torch.randint(low=1, high=vocab_size, size=(B, seq_len), dtype=torch.long)

    patcher = FixedPatcher(patch_size=patch_size)
    patches, mask = patcher(tokens)
    logits = model(patches, mask)

    assert logits.shape[0] == B
    assert logits.shape[1] == n_patches
    assert logits.shape[2] == vocab_size
