"""
Leave-one-subject-out on real BCI IV 2a -- the number to actually quote.

A single held-out pair (subjects 8 and 9) has a standard error of roughly
+/- 8 accuracy points on a 9-subject dataset, which is wide enough that the
ordering of any two methods can be produced by choosing the right pair. LOSO
runs all 9 folds: for each subject, pretrain and meta-train on the other 8,
then evaluate on the held-out one. Nothing about the held-out subject touches
training at any stage.

This is the expensive path -- 9 x (pretrain + meta-train + eval). On a GPU
expect roughly 1-3 hours end to end; on CPU, plan for overnight. Every fold
appends to `loso_results.jsonl` and the script skips folds already present,
so it is safe to interrupt and re-run.

    python3 run_loso.py --source moabb              # all 9 folds
    python3 run_loso.py --source moabb --folds 1 2  # just two, to time it
    python3 run_loso.py --report                    # aggregate what exists
"""
import argparse
import json
import os
import time

import numpy as np

LEDGER = "loso_results.jsonl"
ALL_SUBJECTS = list(range(1, 10))


def done_folds():
    if not os.path.exists(LEDGER):
        return set()
    return {json.loads(l)["fold"] for l in open(LEDGER) if l.strip()}


def report():
    if not os.path.exists(LEDGER):
        print("no folds run yet")
        return
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    rows.sort(key=lambda r: r["fold"])
    print(f"{'subject':>7} {'zero-shot':>10} {'head-only':>10} {'random':>8} {'meta':>8} {'@steps':>7}")
    print("-" * 56)
    for r in rows:
        print(f"{r['fold']:>7} {r['zero_shot']:10.3f} {max(r['curves']['head_only']):10.3f} "
              f"{max(r['curves']['random_lora']):8.3f} {r['best_meta']:8.3f} {r['best_meta_steps']:7d}")
    zs = np.array([r["zero_shot"] for r in rows])
    mt = np.array([r["best_meta"] for r in rows])
    ho = np.array([max(r["curves"]["head_only"]) for r in rows])
    rl = np.array([max(r["curves"]["random_lora"]) for r in rows])
    print("-" * 56)
    print(f"{'MEAN':>7} {zs.mean():10.3f} {ho.mean():10.3f} {rl.mean():8.3f} {mt.mean():8.3f}")
    print(f"{'SD':>7} {zs.std(ddof=1):10.3f} {ho.std(ddof=1):10.3f} "
          f"{rl.std(ddof=1):8.3f} {mt.std(ddof=1):8.3f}")

    # Paired difference across folds, with a bootstrap CI over subjects.
    # Subjects are the unit of analysis here, not individual trials -- treating
    # trials as independent would shrink the interval by an order of magnitude
    # and claim significance that the 9-subject sample cannot support.
    for name, other in [("meta - zero-shot", zs), ("meta - head-only", ho),
                        ("meta - random-LoRA", rl)]:
        d = mt - other
        rng = np.random.default_rng(0)
        boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "significant" if lo > 0 else "not significant"
        print(f"{name:<20} {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    if len(rows) < len(ALL_SUBJECTS):
        print(f"\n(only {len(rows)}/9 folds present -- these numbers are partial)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="moabb", choices=["moabb", "synthetic"])
    p.add_argument("--folds", type=int, nargs="+", default=ALL_SUBJECTS)
    p.add_argument("--subjects", type=int, nargs="+", default=ALL_SUBJECTS,
                   help="restrict the subject pool (useful for timing a fold before "
                        "committing to the full 9-fold run)")
    p.add_argument("--profile", default="full", choices=["full", "sandbox"])
    p.add_argument("--calib-size", type=int, default=12)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--sup-epochs", type=int, default=200)
    p.add_argument("--outer-steps", type=int, default=400)
    p.add_argument("--align-mode", default="session", choices=["session", "calib", "none"])
    p.add_argument("--report", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run folds already in the ledger")
    args = p.parse_args()

    if args.report:
        report()
        return

    import torch_pretrain
    import torch_meta_train
    import torch_eval_calibration

    already = set() if args.force else done_folds()
    for held_out in [f for f in args.folds if f in args.subjects]:
        if held_out in already:
            print(f"[loso] fold {held_out} already in {LEDGER}; skipping (--force to redo)")
            continue
        src = [s for s in args.subjects if s != held_out]
        if not src:
            print(f"[loso] fold {held_out}: no source subjects left; skipping")
            continue
        bb_path = f"backbone_s{held_out}.pt"
        meta_path = f"meta_s{held_out}.pt"
        t0 = time.time()
        print(f"\n===== fold {held_out}/9 | train on {src} | eval on [{held_out}] =====")

        torch_pretrain.main([
            "--subjects", *map(str, src), "--source", args.source,
            "--profile", args.profile, "--align-mode", args.align_mode,
            "--sup-epochs", str(args.sup_epochs), "--out", bb_path])

        torch_meta_train.main([
            "--subjects", *map(str, src), "--source", args.source,
            "--pretrained", bb_path, "--align-mode", args.align_mode,
            "--outer-steps", str(args.outer_steps), "--calib-size", str(args.calib_size),
            "--out", meta_path])

        out = torch_eval_calibration.main([
            "--eval-subjects", str(held_out), "--source", args.source,
            "--pretrained", bb_path, "--meta", meta_path,
            "--align-mode", args.align_mode, "--calib-size", str(args.calib_size),
            "--repeats", str(args.repeats), "--tag", f"loso_s{held_out}"])

        out["fold"] = held_out
        out["source_subjects"] = src
        out["minutes"] = round((time.time() - t0) / 60, 1)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(out) + "\n")
        print(f"[loso] fold {held_out} done in {out['minutes']} min "
              f"-- zero-shot {out['zero_shot']:.3f} -> meta {out['best_meta']:.3f}")

    report()


if __name__ == "__main__":
    main()
