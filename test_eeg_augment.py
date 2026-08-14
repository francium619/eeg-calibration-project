"""Regression tests for eeg_augment.py.

This module is wired into both calibration (torch_eval_calibration.py's
calibrate_curve) and meta-training (torch_meta_train.py, torch_pretrain.py)
but had zero test coverage. These tests focus on the no-op edge cases each
transform promises -- augment_batch(strength<=0), mixup_batch(alpha<=0),
channel_drop/amplitude_scale/gaussian_noise at their identity parameter --
since a broken no-op would silently corrupt clean data on every "no
augmentation" call site, and on shape/dtype contracts the rest of the
pipeline relies on.

Run: python3 test_eeg_augment.py
"""
import torch
import torch.nn.functional as F

from eeg_augment import (time_shift, channel_drop, amplitude_scale, gaussian_noise,
                         freq_mask, augment_batch, mixup_batch, soft_cross_entropy)

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")


def main():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 64)

    # -- shape/dtype contracts (used to build a (B,C,T) batch through the
    #    same pipeline downstream ops assume is preserved) --
    check("time_shift preserves shape", time_shift(x).shape == x.shape)
    check("freq_mask preserves shape and dtype",
          freq_mask(x).shape == x.shape and freq_mask(x).dtype == x.dtype)

    # -- identity/no-op edge cases: each transform's "off" setting must be
    #    an exact no-op, since calibrate_curve and accuracy() call these with
    #    augment=0.0/tta=0 by default and any drift there would silently
    #    perturb the un-augmented eval path --
    check("channel_drop(p=0) is identity", torch.equal(channel_drop(x, p=0.0), x))
    check("channel_drop(p=1) zeros every channel",
          torch.equal(channel_drop(x, p=1.0), torch.zeros_like(x)))
    check("amplitude_scale(sigma=0) is identity", torch.equal(amplitude_scale(x, sigma=0.0), x))
    check("gaussian_noise(sigma=0) is identity", torch.equal(gaussian_noise(x, sigma=0.0), x))
    check("augment_batch(strength<=0) is identity", augment_batch(x, strength=0.0) is x)

    # -- mixup's alpha<=0 branch returns hard one-hot targets unmixed; this
    #    is the fallback torch_eval_calibration.py's default (no mixup) relies on --
    y = torch.randint(0, 4, (4,))
    x_mixed, y_mixed = mixup_batch(x, y, n_classes=4, alpha=0.0)
    check("mixup_batch(alpha<=0) leaves x unmixed", torch.equal(x_mixed, x))
    check("mixup_batch(alpha<=0) yields hard one-hot targets",
          torch.equal(y_mixed, F.one_hot(y, 4).float()))

    # -- soft_cross_entropy against hard one-hot targets must match plain
    #    cross-entropy exactly (it's the loss calibrate_curve would compute
    #    without mixup, just expressed via soft targets) --
    logits = torch.randn(4, 4)
    hard = F.one_hot(y, 4).float()
    ce_hard = F.cross_entropy(logits, y)
    ce_soft = soft_cross_entropy(logits, hard)
    check("soft_cross_entropy matches F.cross_entropy on hard targets",
          torch.allclose(ce_hard, ce_soft, atol=1e-6),
          f"hard={ce_hard.item():.6f} soft={ce_soft.item():.6f}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        raise SystemExit("FAILED: " + ", ".join(FAIL))


if __name__ == "__main__":
    main()
