"""
The iteration ledger.

"Make every iteration better" only means something if each iteration is
measured the same way and the difference is bigger than the noise. This
script runs a fixed ladder of configurations against already-trained
checkpoints, appends every result to a JSONL ledger, and prints a table with
paired 95% confidence intervals against the previous rung.

Two kinds of rung:
  * `eval` rungs change only evaluation-time settings (alignment mode,
    calibration budget, test-time augmentation, augmentation during
    calibration). They reuse the existing checkpoints, so they are cheap.
  * `train` rungs would need re-running pretrain/meta-train; they are listed
    with the exact command rather than executed, so a long run is explicit
    rather than something this script silently starts.

Usage:
  python3 run_ablation.py --list
  python3 run_ablation.py --run align_none align_calib tta calib24
  python3 run_ablation.py --table            # print the ledger so far
"""
import argparse
import json
import os

import numpy as np

LEDGER = "results.jsonl"

EVAL_RUNGS = {
    "baseline":    dict(align_mode="session", calib_size=12, tta=0, augment=0.0),
    "align_none":  dict(align_mode="none",    calib_size=12, tta=0, augment=0.0),
    "align_calib": dict(align_mode="calib",   calib_size=12, tta=0, augment=0.0),
    "aug_calib":   dict(align_mode="session", calib_size=12, tta=0, augment=1.0),
    "tta":         dict(align_mode="session", calib_size=12, tta=4, augment=0.0),
    "aug_tta":     dict(align_mode="session", calib_size=12, tta=4, augment=1.0),
    "calib4":      dict(align_mode="session", calib_size=4,  tta=0, augment=0.0),
    "calib24":     dict(align_mode="session", calib_size=24, tta=0, augment=0.0),
    "calib48":     dict(align_mode="session", calib_size=48, tta=0, augment=0.0),
}

TRAIN_RUNGS = {
    "reptile": "python3 torch_meta_train.py --algo reptile --out meta_reptile.pt "
               "&& python3 torch_eval_calibration.py --meta meta_reptile.pt --tag reptile",
    "no_meta_sgd": "python3 torch_meta_train.py --no-meta-sgd --out meta_nolr.pt "
                   "&& python3 torch_eval_calibration.py --meta meta_nolr.pt --tag no_meta_sgd",
    "no_spatial_adapter": "retrain with make_config(..., use_spatial_adapter=False)",
    "filter_bank": "set bands=DEFAULT_BANDS in MetaEEGDataLoader, then rerun pretrain+meta+eval",
    "loso": "for s in 1..9: pretrain/meta-train on the other 8, eval on s",
}


def run_eval(name, cfg, args):
    from torch_eval_calibration import main as eval_main
    argv = ["--eval-subjects", *[str(s) for s in args.eval_subjects],
            "--repeats", str(args.repeats),
            "--calib-size", str(cfg["calib_size"]),
            "--align-mode", cfg["align_mode"],
            "--tta", str(cfg["tta"]),
            "--augment", str(cfg["augment"]),
            "--steps", *[str(s) for s in args.steps],
            "--threads", str(args.threads),
            "--tag", name, "--json-out", LEDGER, "--quiet"]
    return eval_main(argv)


def load_ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def table():
    rows = load_ledger()
    if not rows:
        print("ledger empty -- run some rungs first")
        return
    print(f"{'tag':<16} {'src':<10} {'zero-shot':>9} {'best meta':>10} {'@steps':>7} "
          f"{'meta-zs':>9} {'95% CI':>20}")
    print("-" * 88)
    for r in rows:
        d, lo, hi = r["meta_minus_zeroshot"]
        print(f"{r['tag']:<16} {r['source']:<10} {r['zero_shot']:9.3f} {r['best_meta']:10.3f} "
              f"{r['best_meta_steps']:7d} {d:+9.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20}")
    print(f"\nchance = {rows[0]['chance']:.3f}; n_runs per row = "
          f"{', '.join(str(r['n_runs']) for r in rows)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="*", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--table", action="store_true")
    p.add_argument("--eval-subjects", type=int, nargs="+", default=[8, 9])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--steps", type=int, nargs="+", default=[0, 1, 3, 8, 20])
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.list:
        print("eval rungs (cheap, reuse checkpoints):")
        for k, v in EVAL_RUNGS.items():
            print(f"  {k:<14} {v}")
        print("\ntrain rungs (need retraining -- run these commands yourself):")
        for k, v in TRAIN_RUNGS.items():
            print(f"  {k:<20} {v}")
        return
    if args.table or not args.run:
        table()
        return
    for name in args.run:
        if name not in EVAL_RUNGS:
            print(f"skip unknown rung {name!r} (see --list)")
            continue
        print(f"[ablation] {name} {EVAL_RUNGS[name]}")
        out = run_eval(name, EVAL_RUNGS[name], args)
        d, lo, hi = out["meta_minus_zeroshot"]
        print(f"  zero-shot {out['zero_shot']:.3f} -> best meta {out['best_meta']:.3f} "
              f"@ {out['best_meta_steps']} steps  (delta {d:+.3f} [{lo:+.3f},{hi:+.3f}])")
    table()


if __name__ == "__main__":
    main()
