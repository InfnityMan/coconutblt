import torch

from coconutblt import HybridTransformer, EntropyPatcher
from coconutblt.entropy_patcher import LearnedEntropyEstimator


def test_joint_training_step():
    # single-step joint training check
    B = 4
    S = 32
    vocab_size = 128

    model = HybridTransformer(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, patch_size=4, n_latents=1)
    est = LearnedEntropyEstimator(vocab_size=vocab_size, emb_dim=16, hidden=32, kernel=3)
    patcher = EntropyPatcher(min_patch_size=2, max_patch_size=4, lookahead=4, entropy_threshold=1.0, estimator=est)

    tokens = torch.randint(low=1, high=vocab_size, size=(B, S), dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    est_opt = torch.optim.AdamW(est.parameters(), lr=1e-3)
    loss_f = torch.nn.CrossEntropyLoss()

    patches, mask = patcher(tokens)
    logits = model(patches, mask)
    targets = patches[:, :, 0]
    Bp, Pp, V = logits.shape
    lm_loss = loss_f(logits.view(Bp * Pp, V), targets.view(Bp * Pp))

    est_loss = est.mse_loss_against_shannon(tokens, lookahead=4)
    total_loss = lm_loss + 0.5 * est_loss

    opt.zero_grad()
    est_opt.zero_grad()
    total_loss.backward()
    opt.step()
    est_opt.step()

    # if successful, parameters updated and backward executed
    assert True
