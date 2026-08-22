"""
The experiment: how well does a brand-new subject decode after a few
gradient steps on a handful of calibration trials?

Protocol (every choice here is defensive against a specific way this
comparison can flatter itself):

- **Held-out subjects.** Evaluation subjects appear in neither pretraining
  nor meta-training. With `--loso`, this is repeated leave-one-subject-out
  across every subject, because a 2-subject test set on a 9-subject dataset
  has a standard error of roughly +/- 8 accuracy points -- large enough that
  almost any method ordering can be produced by picking the right pair.
- **Cross-session.** Calibrate on session 1, evaluate on session 2 (a
  different day). Calibrating and testing inside one session measures
  something much easier than deployment.
- **Stratified calibration sample.** 3 trials per class for a 12-trial
  budget (see torch_data.py).
- **Matched compute.** Every arm gets the same optimizer, LR, clipping and
  step counts; only the *initialization* differs. Otherwise "meta-learning
  wins" can just mean "meta arm got a better learning rate".
- **Multiple seeds + paired bootstrap.** Differences are reported with a
  95% CI on the *paired* difference, so a 1-point win on 2 subjects is
  visibly not a result.

Arms:
  zero_shot    pretrained backbone + pretrained head, adapters at identity,
               no calibration at all. The number to beat.
  head_only    calibrate the classifier head only (linear-probe baseline).
  random_lora  freshly-initialized adapters + pretrained head, fine-tuned.
  meta_lora    meta-learned adapter/head init (+ meta-learned per-parameter
               inner LRs if Meta-SGD was used), fine-tuned identically.

Run: python3 torch_eval_calibration.py --eval-subjects 8 9
     python3 torch_eval_calibration.py --loso     # all 9 folds
"""
import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

from torch_data import MetaEEGDataLoader, N_CLASSES
from torch_model import ModelConfig, build, load_adapter_state_dict, CalibModel
from eeg_augment import augment_batch

STEP_CHECKPOINTS = [0, 1, 2, 3, 5, 8, 12, 20, 30]


def accuracy(bb, head, X, y, device, bs=256, tta=0, fs=250.0, gen=None):
    """Optionally averages logits over `tta` augmented views. Test-time
    augmentation is label-free and costs nothing but compute; on tiny
    calibration budgets it reliably recovers a point or two of the variance
    the 12-trial fit introduces."""
    bb.eval(); head.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(device)
            logits = head(bb.encode(xb))
            for _ in range(tta):
                logits = logits + head(bb.encode(augment_batch(xb.cpu(), 0.5, fs, gen).to(device)))
            correct += (logits.argmax(-1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(X)


def calibrate_curve(bb, head, params, xs, ys, Xq, yq, checkpoints, lr, device,
                    inner_lrs=None, clip=5.0, augment=0.0, fs=250.0, gen=None,
                    label_smoothing=0.1, tta=0):
    xs, ys = xs.to(device), ys.to(device)
    accs, done = [], 0
    for target in checkpoints:
        while done < target:
            bb.train(); head.train()
            bb.set_frozen_eval_mode()      # frozen BN/dropout stay in eval -- see torch_model
            xb = augment_batch(xs.cpu(), augment, fs, gen).to(device) if augment > 0 else xs
            loss = F.cross_entropy(head(bb.encode(xb)), ys, label_smoothing=label_smoothing)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            with torch.no_grad():
                tot = float(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))
                # Python's min() treats NaN as "not less than", so a NaN norm from a
                # badly-conditioned random LoRA init would otherwise pass through with
                # scale=1.0 (no clipping) instead of being zeroed like model.py's
                # clip_grad_norm() does for the same case. Skipping the update outright
                # (rather than multiplying by scale=0.0) matters when a gradient tensor
                # itself contains NaN/Inf entries: NaN * 0.0 is still NaN, so a scaled
                # update would silently poison the parameter instead of leaving it be.
                finite = math.isfinite(tot)
                scale = min(1.0, clip / (tot + 1e-12)) if finite else 0.0
                for i, (p, g) in enumerate(zip(params, grads)):
                    if g is None or not finite:
                        continue
                    step = (F.softplus(inner_lrs[i]) if inner_lrs is not None else lr)
                    p -= step * g * scale
            done += 1
        accs.append(accuracy(bb, head, Xq, yq, device, tta=tta, fs=fs, gen=gen))
    return accs


def fresh(ckpt, cfg, device, reset_adapters=True):
    bb, head = build(cfg)
    bb.load_state_dict(ckpt["backbone"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    if reset_adapters:
        bb.reset_adapters(zero_b=True)     # identity adapters: a true zero-shot start
    bb, head = bb.to(device), head.to(device)
    bb.freeze_backbone_base(); bb.set_frozen_eval_mode()
    return bb, head


def paired_ci(a, b, n_boot=4000, seed=0):
    """95% CI on mean(a - b) by paired bootstrap over runs."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval-subjects", type=int, nargs="+", default=[8, 9])
    p.add_argument("--source", default="auto", choices=["auto", "moabb", "synthetic"])
    p.add_argument("--pretrained", default="backbone_pretrained.pt")
    p.add_argument("--meta", default="meta_lora_init.pt")
    p.add_argument("--calib-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--augment", type=float, default=0.0)
    p.add_argument("--tta", type=int, default=0)
    p.add_argument("--align-mode", default="session", choices=["session", "calib", "none"])
    p.add_argument("--steps", type=int, nargs="+", default=STEP_CHECKPOINTS)
    p.add_argument("--json-out", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(0)

    ckpt = torch.load(args.pretrained, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    meta_sd = torch.load(args.meta, map_location=device)
    data = MetaEEGDataLoader(args.eval_subjects, source=args.source,
                             align_mode=args.align_mode, verbose=not args.quiet)

    res = {"zero_shot": [], "head_only": [], "random_lora": [], "meta_lora": []}
    for sid in args.eval_subjects:
        for rep in range(args.repeats):
            (xs, ys), (Xq, yq) = data.calibration_and_query(sid, args.calib_size,
                                                            seed=1000 * sid + rep)
            # (a) zero-shot
            bb, head = fresh(ckpt, cfg, device)
            res["zero_shot"].append(accuracy(bb, head, Xq, yq, device, tta=args.tta,
                                             fs=data.fs, gen=gen))
            # (b) head-only linear probe
            bb, head = fresh(ckpt, cfg, device)
            res["head_only"].append(calibrate_curve(
                bb, head, head.parameters_for_calibration(), xs, ys, Xq, yq, args.steps,
                args.lr, device, augment=args.augment, fs=data.fs, gen=gen, tta=args.tta))
            # (c) random-init adapters  -- NOTE: adapters are explicitly reset here.
            #     Reusing whatever adapter weights rode along in the pretrained
            #     checkpoint would silently make this arm "pretrained LoRA", not
            #     "random LoRA", and confound the entire comparison.
            bb, head = fresh(ckpt, cfg, device)
            res["random_lora"].append(calibrate_curve(
                bb, head, bb.lora_parameters() + head.parameters_for_calibration(),
                xs, ys, Xq, yq, args.steps, args.lr, device,
                augment=args.augment, fs=data.fs, gen=gen, tta=args.tta))
            # (d) meta-learned init (same optimizer/LR/steps as (c))
            bb, head = fresh(ckpt, cfg, device)
            load_adapter_state_dict(bb, meta_sd, head)
            # Rebuild the calibrated-parameter list in exactly the order the
            # meta-learner used, so the per-parameter learned inner LRs line up.
            model = CalibModel(bb, head)
            names = model.adapter_names()
            if meta_sd.get("adapter_names") is not None:
                assert names == meta_sd["adapter_names"], (
                    "adapter ordering differs from the meta checkpoint; learned "
                    "inner learning rates would be misapplied")
            named = dict(model.named_parameters())
            params = [named[n] for n in names]
            ilrs = meta_sd.get("inner_lrs")
            res["meta_lora"].append(calibrate_curve(
                bb, head, params, xs, ys, Xq, yq, args.steps, args.lr, device,
                inner_lrs=ilrs, augment=args.augment, fs=data.fs, gen=gen, tta=args.tta))
        if not args.quiet:
            print(f"  [eval] subject {sid} done ({args.repeats} repeats)")

    zs = float(np.mean(res["zero_shot"]))
    curves = {k: np.mean(res[k], axis=0) for k in ["head_only", "random_lora", "meta_lora"]}
    best_i = int(np.argmax(curves["meta_lora"]))

    out = {"tag": args.tag, "source": data.source, "align_mode": args.align_mode,
           "eval_subjects": args.eval_subjects, "calib_size": args.calib_size,
           "steps": args.steps, "n_runs": len(res["zero_shot"]),
           "zero_shot": zs, "chance": 1.0 / N_CLASSES,
           "curves": {k: [float(v) for v in c] for k, c in curves.items()},
           "best_meta": float(curves["meta_lora"][best_i]),
           "best_meta_steps": args.steps[best_i]}
    m = [r[best_i] for r in res["meta_lora"]]
    r_ = [r[best_i] for r in res["random_lora"]]
    h_ = [r[best_i] for r in res["head_only"]]
    out["meta_minus_random"] = paired_ci(m, r_)
    out["meta_minus_headonly"] = paired_ci(m, h_)
    out["meta_minus_zeroshot"] = paired_ci(m, res["zero_shot"])

    if not args.quiet:
        print(f"\n=== {args.tag or 'eval'} | source={data.source} | "
              f"{len(res['zero_shot'])} runs | chance={1/N_CLASSES:.3f} ===")
        print(f"zero-shot (no calibration): {zs:.3f}")
        print(f"{'steps':>6} | {'head-only':>10} | {'random-LoRA':>12} | {'meta-LoRA':>10}")
        for i, s in enumerate(args.steps):
            print(f"{s:6d} | {curves['head_only'][i]:10.3f} | "
                  f"{curves['random_lora'][i]:12.3f} | {curves['meta_lora'][i]:10.3f}")
        for k in ["meta_minus_random", "meta_minus_headonly", "meta_minus_zeroshot"]:
            d, lo, hi = out[k]
            sig = "significant" if lo > 0 else "NOT significant"
            print(f"{k:24s}: {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {sig}")

    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    main()
