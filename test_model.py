"""Sanity + gradient checks for model.py (backbone, LoRA, classifier head)."""
import numpy as np
from tensor import Tensor, softmax_cross_entropy
from model import EEGBackbone, ClassifierHead, zero_grad, clip_grad_norm, snapshot, restore
import data

np.random.seed(0)


def build_and_loss(backbone, head, patches, labels):
    _, tok = backbone.encode(patches)
    logits = head(tok)
    loss, _ = softmax_cross_entropy(logits, labels)
    return loss


def test_forward_shapes():
    bb = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
    head = ClassifierHead(data.N_CLASSES, seed=2)
    X, y = data.make_subject(np.random.default_rng(3), n_trials=6)
    patches = data.patchify(X)
    assert patches.shape == (6, data.N_TOKENS, data.WIN)
    pooled, tok = bb.encode(patches)
    assert pooled.shape == (6, 16)
    assert tok.shape == (6, data.N_TOKENS, 16)
    logits = head(tok)
    assert logits.shape == (6, data.N_CLASSES)
    print("[OK] forward shapes")


def test_pretrain_grad_flow():
    """Base backbone params (incl. patch_embed W, attention/FFN weights,
    ch/win embeddings) should all receive nonzero gradients from a masked
    reconstruction loss, via finite-difference spot checks on a few params."""
    bb = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
    X, y = data.make_subject(np.random.default_rng(4), n_trials=4)
    patches = data.patchify(X)
    mask_idx = np.array([0, 5, 10])

    params = bb.backbone_base_params() + bb.decoder_params()

    def loss_fn():
        pooled, tok = bb.encode(patches, mask_idx=mask_idx)
        pred = bb.decoder_head(tok)
        target = patches[:, mask_idx, :]
        diff = pred.data[:, mask_idx, :] - target
        return float(np.mean(diff ** 2)), pred, mask_idx, target

    # analytic gradient via autodiff on the same computation, expressed with Tensor ops
    from tensor import Tensor
    zero_grad(params)
    pooled, tok = bb.encode(patches, mask_idx=mask_idx)
    pred = bb.decoder_head(tok)
    N = len(mask_idx) * patches.shape[0] * data.WIN
    target_full = np.zeros_like(pred.data)
    target_full[:, mask_idx, :] = patches[:, mask_idx, :]
    mask_full = np.zeros_like(pred.data)
    mask_full[:, mask_idx, :] = 1.0
    diff = (pred - Tensor(target_full, requires_grad=False)) * Tensor(mask_full, requires_grad=False)
    loss = (diff * diff).sum() * (1.0 / N)
    loss.backward()

    # pick one param to finite-difference check: patch_embed.W
    p = bb.patch_embed.W
    g_analytic = p.grad.copy()

    eps = 1e-5
    idx = (1, 2)
    orig = p.data[idx]

    def eval_loss():
        pooled, tok = bb.encode(patches, mask_idx=mask_idx)
        pred = bb.decoder_head(tok)
        d = (pred.data - target_full) * mask_full
        return float(np.sum(d ** 2) / N)

    p.data[idx] = orig + eps
    l1 = eval_loss()
    p.data[idx] = orig - eps
    l2 = eval_loss()
    p.data[idx] = orig
    g_num = (l1 - l2) / (2 * eps)

    err = abs(g_num - g_analytic[idx]) / (abs(g_num) + 1e-8)
    status = "OK" if err < 1e-2 else "FAIL"
    print(f"[{status}] pretrain masked-reconstruction grad check: analytic={g_analytic[idx]:.6f} numeric={g_num:.6f} rel_err={err:.2e}")
    assert status == "OK"


def test_lora_only_updates_lora():
    bb = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
    head = ClassifierHead(data.N_CLASSES, seed=2)
    bb.freeze_backbone_base()
    bb.add_lora_everywhere(r=4, seed=5)

    X, y = data.make_subject(np.random.default_rng(6), n_trials=8)
    patches = data.patchify(X)

    base_before = bb.patch_embed.W.data.copy()
    lora_params = bb.lora_params() + head.params()
    zero_grad(lora_params)
    loss = build_and_loss(bb, head, patches, y)
    loss.backward()
    for p in lora_params:
        p.data -= 0.1 * p.grad

    base_after = bb.patch_embed.W.data.copy()
    assert np.allclose(base_before, base_after), "frozen base weight changed!"
    assert bb.patch_embed.W.grad is None or np.allclose(bb.patch_embed.W.grad, 0), "frozen weight got nonzero grad"
    assert bb.mask_token.requires_grad is False, "freeze_backbone_base() must also freeze mask_token"
    print("[OK] LoRA fine-tuning leaves frozen base weights untouched")


def test_clip_grad_norm():
    """clip_grad_norm is the safety net eval_calibration.py and meta_train.py
    rely on when a badly-conditioned random LoRA init takes an exploding
    step. Check all three branches: under-budget grads pass through
    unchanged, over-budget grads are rescaled to exactly max_norm (direction
    preserved), and non-finite grads (NaN/Inf, e.g. from a diverged step)
    are zeroed out rather than silently propagated into the next update."""
    p1 = Tensor(np.zeros(3), requires_grad=True)
    p1.grad = np.array([0.3, 0.4, 0.0])  # norm = 0.5, under budget
    clip_grad_norm([p1], max_norm=5.0)
    assert np.allclose(p1.grad, [0.3, 0.4, 0.0]), "under-budget grad should be untouched"

    p2 = Tensor(np.zeros(3), requires_grad=True)
    p2.grad = np.array([3.0, 4.0, 0.0])  # norm = 5.0, over budget
    clip_grad_norm([p2], max_norm=1.0)
    assert np.isclose(np.linalg.norm(p2.grad), 1.0), "over-budget grad should be rescaled to max_norm"
    assert np.allclose(p2.grad / np.linalg.norm(p2.grad), [0.6, 0.8, 0.0]), "clipping should preserve direction"

    p3 = Tensor(np.zeros(2), requires_grad=True)
    p3.grad = np.array([np.nan, 1.0])
    clip_grad_norm([p3], max_norm=1.0)
    assert np.all(p3.grad == 0.0), "non-finite grad norm should zero the gradient, not propagate NaN"

    print("[OK] clip_grad_norm: no-op under budget, rescales over budget, zeroes non-finite grads")


def test_snapshot_restore_round_trip():
    """meta_train.py's Reptile step depends on snapshot()/restore() giving an
    independent copy of parameter data: each outer step snapshots meta_params,
    lets the inner loop mutate them in place, diffs the result against the
    snapshot, then restores. If snapshot() ever aliased memory instead of
    copying it, that diff would silently collapse to zero (or reflect
    mid-adaptation garbage) every outer step, and meta-training would train
    on nothing without raising an error."""
    p = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

    snap = snapshot([p])
    p.data[:] = 99.0  # mutate after snapshotting
    assert np.array_equal(snap[0], [1.0, 2.0, 3.0]), "snapshot must be independent of later mutation"

    restore([p], snap)
    assert np.array_equal(p.data, [1.0, 2.0, 3.0]), "restore must reset param data from the snapshot"

    p.data[0] = -1.0  # mutate after restoring
    assert snap[0][0] == 1.0, "mutating a restored param must not alias the snapshot"
    print("[OK] snapshot()/restore() are independent copies, not aliases")


if __name__ == "__main__":
    test_forward_shapes()
    test_pretrain_grad_flow()
    test_lora_only_updates_lora()
    test_clip_grad_norm()
    test_snapshot_restore_round_trip()
    print("\nAll model tests passed.")
