import torch

from coconutblt import HybridTransformer, EntropyPatcher


def test_model_with_entropy_patcher_forward():
    B = 2
    S = 40
    # make structured data: repeated zeros then noise
    seq = torch.cat([torch.zeros((B, S//2), dtype=torch.long), torch.randint(1, 255, (B, S - S//2), dtype=torch.long)], dim=1)

    patcher = EntropyPatcher(min_patch_size=2, max_patch_size=8, lookahead=4, entropy_threshold=1.0)
    patches, mask = patcher(seq)

    model = HybridTransformer(vocab_size=256, d_model=64, n_layers=4, n_heads=4, patch_size=patches.shape[2], n_latents=2, use_byte_patch_attention=True)
    logits = model(patches, mask)

    assert logits.shape[0] == B
    assert logits.shape[1] == patches.shape[1]
    assert logits.shape[2] == 256
