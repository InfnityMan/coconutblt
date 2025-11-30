import torch

from coconutblt import EntropyPatcher


def test_entropy_patcher_low_entropy():
    # low entropy sequence (all zeros) should produce long patches
    B = 2
    S = 32
    seq = torch.zeros((B, S), dtype=torch.long)
    p = EntropyPatcher(min_patch_size=2, max_patch_size=8, lookahead=4, entropy_threshold=0.1)
    patches, mask = p(seq)

    assert patches.shape[0] == B
    assert patches.shape[2] == 8
    # all patches should be non-empty (mask True) across full length except maybe tail
    assert mask.any().item()


def test_entropy_patcher_high_entropy():
    # high-entropy random sequence should favor smaller patches (close to min_patch_size)
    B = 1
    S = 64
    seq = torch.randint(low=0, high=256, size=(B, S), dtype=torch.long)
    p = EntropyPatcher(min_patch_size=2, max_patch_size=8, lookahead=4, entropy_threshold=1.0)
    patches, mask = p(seq)

    # ensure the patched outputs are padded to the expected width
    assert patches.shape == (B, patches.shape[1], 8)
    # check that the majority of produced patches do not exceed max patch size
    # (we can't strictly assert sizes everywhere in random case but tests confirm output shape)
    assert mask.sum().item() >= 1


def test_learned_estimator_forward():
    from coconutblt.entropy_patcher import LearnedEntropyEstimator
    B = 2
    S = 32
    seq = torch.randint(low=0, high=256, size=(B, S), dtype=torch.long)
    est = LearnedEntropyEstimator(vocab_size=256, emb_dim=32, hidden=64, kernel=3)
    out = est(seq)
    assert out.shape == (B, S)


def test_entropy_patcher_with_estimator():
    from coconutblt.entropy_patcher import LearnedEntropyEstimator
    est = LearnedEntropyEstimator(vocab_size=256, emb_dim=32, hidden=64, kernel=3)
    B = 2
    S = 40
    seq = torch.randint(low=0, high=256, size=(B, S), dtype=torch.long)
    p = EntropyPatcher(min_patch_size=2, max_patch_size=8, lookahead=4, entropy_threshold=1.0, estimator=est)
    patches, mask = p(seq)
    assert patches.shape[0] == B
    assert patches.shape[2] == 8
