"""
Regression tests for the PyTorch pipeline.

These target the failure modes that produce *good-looking but wrong* numbers,
which are the only ones that actually matter here -- a crash announces
itself, a leak does not.

Run: LD_LIBRARY_PATH=... python3 test_torch_pipeline.py
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch_data import (MetaEEGDataLoader, reference_covariance, euclidean_align,
                        _inv_sqrt_spd, zscore_per_trial, bandpass_fft, make_filter_bank,
                        DEFAULT_BANDS)
from torch_model import make_config, build, CalibModel, adapter_state_dict, load_adapter_state_dict

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")


def main():
    torch.manual_seed(0)
    print("== data ==")
    d = MetaEEGDataLoader([1, 2, 3], source="synthetic", verbose=False)

    # 1. Euclidean alignment really does whiten: mean covariance -> identity.
    X = d.store[1]["0train"].X.numpy().astype(np.float64)
    R = reference_covariance(X)
    Xa = euclidean_align(X, R)
    Ra = reference_covariance(Xa)
    err = np.abs(Ra / np.trace(Ra) * Ra.shape[0] - np.eye(Ra.shape[0])).max()
    check("euclidean alignment whitens mean covariance", err < 0.15, f"max dev {err:.3f}")

    # 2. R^{-1/2} is a real inverse square root.
    A = np.random.randn(8, 8); S = A @ A.T + np.eye(8)
    Ih = _inv_sqrt_spd(S)
    check("inv_sqrt_spd correct", np.allclose(Ih @ S @ Ih, np.eye(8), atol=1e-6))

    # 3. Standardization actually applied (raw EEG in volts would break training).
    check("trials are standardized", abs(float(d.store[1]["0train"].X.std()) - 1.0) < 0.3)

    # 4. Calibration sample is stratified -- an unstratified 12-trial draw
    #    routinely misses a class and caps accuracy at 75%.
    ok = True
    for seed in range(20):
        (_, ys), _ = d.calibration_and_query(1, 12, seed=seed)
        ok &= len(torch.unique(ys)) == 4 and (torch.bincount(ys, minlength=4) == 3).all().item()
    check("few-shot calibration sample is class-balanced", ok)

    # 4b. When calib_size isn't a multiple of the class count, the per-class
    #     quota rounds down, so the sample is shorter than requested rather
    #     than padded/overfilled -- document this so a caller that assumes
    #     len(idx) == n_total doesn't get a silent surprise.
    y_all = d.store[1]["0train"].y
    idx10 = d.stratified_indices(y_all, 10, seed=0)
    check("stratified_indices rounds down for a non-divisible n_total",
          len(idx10) == 8, f"len={len(idx10)} (expected 8 = (10//4)*4)")

    # 5. No trial overlap between the calibration and query sessions.
    (xs, _), (Xq, _) = d.calibration_and_query(1, 12, seed=0)
    overlap = any(bool((Xq == s).all(dim=-1).all(dim=-1).any()) for s in xs)
    check("calibration and query sets are disjoint", not overlap)

    # 5b. bandpass_fft passes in-band content, suppresses out-of-band content.
    fs, T = 250.0, 250
    t = np.arange(T) / fs
    x = (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 25 * t))[None, None, :]
    xf = bandpass_fft(x, fs, 8.0, 13.0)
    spec = np.abs(np.fft.rfft(xf[0, 0]))
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    in_band = spec[(freqs >= 9) & (freqs <= 11)].max()
    out_band = spec[(freqs >= 24) & (freqs <= 26)].max()
    check("bandpass_fft passes in-band, suppresses out-of-band",
          in_band > 10 * out_band, f"in-band={in_band:.3f} out-of-band={out_band:.3f}")

    # 5c. make_filter_bank stacks sub-bands along the channel axis.
    Xfb = make_filter_bank(np.random.randn(2, 3, T), fs, DEFAULT_BANDS)
    check("make_filter_bank stacks sub-bands along channel axis",
          Xfb.shape == (2, 3 * len(DEFAULT_BANDS), T))

    print("== model ==")
    cfg = make_config(d.n_channels, d.n_timepoints, 4, profile="sandbox")
    bb, head = build(cfg)

    # 6. Adapters are an exact no-op at init, so 'zero-shot' means zero-shot.
    bb.eval()
    x = torch.randn(4, d.n_channels, d.n_timepoints)
    before = bb.encode(x)
    bb.reset_adapters(zero_b=True)
    check("adapters are identity at init", torch.allclose(before, bb.encode(x), atol=1e-6))

    # 7. The frozen backbone really stays frozen through calibration -- the
    #    whole premise is that only the adapters move.
    bb.freeze_backbone_base(); bb.set_frozen_eval_mode()
    base_before = [p.detach().clone() for p in bb.backbone_base_parameters()]
    bn_before = [m.running_mean.clone() for m in bb.modules()
                 if isinstance(m, torch.nn.BatchNorm2d)]
    params = bb.lora_parameters() + head.parameters_for_calibration()
    adapters_before = [p.detach().clone() for p in params]
    y = torch.randint(0, 4, (4,))
    for _ in range(5):
        loss = F.cross_entropy(head(bb.encode(x)), y)
        g = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p_, gi in zip(params, g):
                if gi is not None:
                    p_ -= 0.05 * gi
    check("frozen base weights unchanged after calibration",
          all(torch.equal(a, b) for a, b in zip(base_before, bb.backbone_base_parameters())))
    check("BatchNorm running stats unchanged (frozen means frozen)",
          all(torch.equal(a, m.running_mean) for a, m in
              zip(bn_before, [m for m in bb.modules() if isinstance(m, torch.nn.BatchNorm2d)])))
    check("adapters did move", any(not torch.equal(a, b)
                                   for a, b in zip(adapters_before, params)))

    # 8. Adapter checkpoints round-trip and carry no base weights.
    sd = adapter_state_dict(bb, head)
    check("adapter checkpoint excludes frozen base",
          set(sd.keys()) <= {"lora", "spatial", "head"})
    bb2, head2 = build(cfg)
    load_adapter_state_dict(bb2, sd, head2)
    check("adapter checkpoint round-trips",
          all(torch.allclose(a, b) for a, b in
              zip(bb.lora_parameters(), bb2.lora_parameters())))

    # 9. Parameter ordering used by Meta-SGD is stable and complete.
    m = CalibModel(bb, head)
    names = m.adapter_names()
    check("adapter_names covers every calibrated parameter",
          len(names) == len(bb.lora_parameters()) + len(head.parameters_for_calibration()))
    check("adapter_names is deterministic", names == CalibModel(bb, head).adapter_names())

    print("== leakage controls ==")
    # 10. Shuffled-label control: if the pipeline could score above chance with
    #     randomized calibration labels, something is leaking the answer.
    from torch_eval_calibration import calibrate_curve
    bb3, head3 = build(cfg); bb3.freeze_backbone_base(); bb3.set_frozen_eval_mode()
    (xs, ys), (Xq, yq) = d.calibration_and_query(1, 12, seed=0)
    ys_shuf = ys[torch.randperm(len(ys))]
    accs = calibrate_curve(bb3, head3, bb3.lora_parameters() + head3.parameters_for_calibration(),
                           xs, ys_shuf, Xq[:120], yq[:120], [0, 5, 15], 0.05,
                           torch.device("cpu"))
    check("shuffled-label calibration stays near chance", max(accs) < 0.45,
          f"max acc {max(accs):.3f} (chance 0.250)")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        raise SystemExit("FAILED: " + ", ".join(FAIL))


if __name__ == "__main__":
    main()
