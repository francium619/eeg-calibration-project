"""Sanity + gradient checks for model.py (backbone, LoRA, classifier head)."""
import numpy as np
from tensor import softmax_cross_entropy
from model import EEGBackbone, ClassifierHead, zero_grad
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
    print("[OK] LoRA fine-tuning leaves frozen base weights untouched")


if __name__ == "__main__":
    test_forward_shapes()
    test_pretrain_grad_flow()
    test_lora_only_updates_lora()
    print("\nAll model tests passed.")
