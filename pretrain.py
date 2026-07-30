"""
Self-supervised pretraining of the EEG backbone: masked channel-window
reconstruction, in the style of LaBraM's masked neural-code prediction (we
predict raw patch values with an MSE decoder rather than VQ codes -- a
simplification, see README).

Pretraining pools trials from many *different* synthetic subjects so the
backbone is pushed to learn structure that's useful across subjects (the way
a real foundation model is pretrained on ~20 datasets / thousands of hours
spanning many people), rather than overfitting to one subject's quirks.

Run: python3 pretrain.py
Saves backbone weights to backbone_pretrained.npz
"""
import time
import numpy as np
from tensor import Tensor
from model import EEGBackbone, zero_grad, sgd_step
import data

SEED = 0
N_PRETRAIN_SUBJECTS = 30
TRIALS_PER_SUBJECT = 40
BATCH_SIZE = 32
N_STEPS = 1500
LR = 0.3
MASK_FRACTION = 0.3


def masked_reconstruction_loss(bb, patches_batch, mask_idx):
    pooled, tok = bb.encode(patches_batch, mask_idx=mask_idx)
    pred = bb.decoder_head(tok)
    B = patches_batch.shape[0]
    N = len(mask_idx) * B * data.WIN

    target_full = np.zeros_like(pred.data)
    target_full[:, mask_idx, :] = patches_batch[:, mask_idx, :]
    mask_full = np.zeros_like(pred.data)
    mask_full[:, mask_idx, :] = 1.0

    diff = (pred - Tensor(target_full, requires_grad=False)) * Tensor(mask_full, requires_grad=False)
    loss = (diff * diff).sum() * (1.0 / N)
    return loss


def main():
    rng = np.random.default_rng(SEED)
    print(f"Generating pretraining pool: {N_PRETRAIN_SUBJECTS} synthetic subjects x {TRIALS_PER_SUBJECT} trials...")
    pool = data.make_population(N_PRETRAIN_SUBJECTS, seed=SEED, n_trials=TRIALS_PER_SUBJECT)
    all_patches = np.concatenate([p for p, y in pool], axis=0)
    print(f"Pretraining pool shape: {all_patches.shape}")

    bb = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
    params = bb.backbone_base_params() + bb.decoder_params()
    n_params = sum(p.data.size for p in params)
    print(f"Backbone trainable params (pretraining phase): {n_params}")

    n_mask = max(1, int(MASK_FRACTION * data.N_TOKENS))
    t0 = time.time()
    losses = []
    for step in range(N_STEPS):
        idx = rng.integers(0, all_patches.shape[0], size=BATCH_SIZE)
        batch = all_patches[idx]
        mask_idx = rng.choice(data.N_TOKENS, size=n_mask, replace=False)

        zero_grad(params)
        loss = masked_reconstruction_loss(bb, batch, mask_idx)
        loss.backward()
        sgd_step(params, LR)
        losses.append(float(loss.data))

        if step % 50 == 0 or step == N_STEPS - 1:
            recent = np.mean(losses[-50:])
            print(f"  step {step:4d}/{N_STEPS}  masked-recon MSE (running avg) = {recent:.4f}")

    print(f"Pretraining done in {time.time()-t0:.1f}s. Initial loss ~{np.mean(losses[:20]):.4f} -> final ~{np.mean(losses[-20:]):.4f}")
    assert np.mean(losses[-20:]) < np.mean(losses[:20]), "pretraining loss did not decrease -- something is wrong"

    save = {}
    for name, p in [
        ("patch_embed_W", bb.patch_embed.W), ("patch_embed_b", bb.patch_embed.b),
        ("Wq", bb.Wq.W), ("Wk", bb.Wk.W), ("Wv", bb.Wv.W), ("Wo", bb.Wo.W),
        ("W1_W", bb.W1.W), ("W1_b", bb.W1.b), ("W2_W", bb.W2.W), ("W2_b", bb.W2.b),
        ("ch_embed", bb.ch_embed), ("win_embed", bb.win_embed),
        ("ln1_g", bb.ln1_g), ("ln1_b", bb.ln1_b), ("ln2_g", bb.ln2_g), ("ln2_b", bb.ln2_b),
    ]:
        save[name] = p.data
    np.savez("backbone_pretrained.npz", **save)
    print("Saved pretrained backbone to backbone_pretrained.npz")


if __name__ == "__main__":
    main()
