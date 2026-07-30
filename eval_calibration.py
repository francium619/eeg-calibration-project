"""
The core evaluation: for subjects the model has NEVER seen (not in the
pretraining pool, not in the meta-training pool), compare three calibration
strategies as a function of how many gradient steps / calibration trials
are available:

  (a) Zero-shot        -- frozen backbone + meta-learned LoRA/head, NO
                           per-subject adaptation at all.
  (b) Random-init LoRA  -- fresh random LoRA + head, fine-tuned from scratch
                           on the subject's calibration trials (the "current
                           practice" baseline: no meta-learning benefit).
  (c) Meta-learned LoRA -- LoRA + head initialized from meta_train.py's
                           learned init, fine-tuned the same way.

(b) vs (c) isolates the effect of the meta-learned initialization: same
architecture, same optimizer, same number of steps and calibration trials,
only the starting point differs. This is the headline result for "can we
calibrate a foundation model to a new subject in real time with very few
samples."

Run: python3 eval_calibration.py  (requires backbone_pretrained.npz and
meta_lora_init.npz from the previous two scripts)
"""
import numpy as np
from model import EEGBackbone, ClassifierHead, zero_grad, sgd_step, clip_grad_norm, load_pretrained
from tensor import softmax_cross_entropy
import data

SEED = 777
LORA_RANK = 4
N_EVAL_SUBJECTS = 16
TRIALS_PER_SUBJECT = 60
CALIB_SIZE = 12           # trials available for calibration (the "few shots")
QUERY_SIZE = 40           # held-out trials used to measure real accuracy
STEP_CHECKPOINTS = [0, 1, 2, 3, 5, 8, 12]
LR = 0.08
GRAD_CLIP = 5.0            # gradient-norm clip, guards against rare exploding updates on badly-conditioned random inits
N_REPEATS = 4             # repeat each subject's calibration with different random calib subsets


def accuracy(bb, head, X, y):
    _, tok = bb.encode(X)
    logits = head(tok)
    preds = np.argmax(logits.data, axis=1)
    return float(np.mean(preds == y))


def fine_tune_curve(bb, head, params, X_calib, y_calib, X_query, y_query, step_checkpoints, lr):
    accs = []
    max_steps = max(step_checkpoints)
    done = 0
    for target in step_checkpoints:
        while done < target:
            zero_grad(params)
            _, tok = bb.encode(X_calib)
            logits = head(tok)
            loss, _ = softmax_cross_entropy(logits, y_calib)
            loss.backward()
            clip_grad_norm(params, GRAD_CLIP)
            sgd_step(params, lr)
            done += 1
        accs.append(accuracy(bb, head, X_query, y_query))
    return accs


def main():
    print(f"Generating {N_EVAL_SUBJECTS} held-out evaluation subjects "
          f"(unseen during pretraining and meta-training)...")
    eval_pool = data.make_population(N_EVAL_SUBJECTS, seed=SEED + 5000, n_trials=TRIALS_PER_SUBJECT)

    meta = np.load("meta_lora_init.npz")

    results = {"zero_shot": [], "random_lora": [], "meta_lora": []}

    for si, (X_all, y_all) in enumerate(eval_pool):
        rng = np.random.default_rng(SEED + si)
        for rep in range(N_REPEATS):
            perm = rng.permutation(len(y_all))
            calib_idx = perm[:CALIB_SIZE]
            query_idx = perm[CALIB_SIZE:CALIB_SIZE + QUERY_SIZE]
            X_calib, y_calib = X_all[calib_idx], y_all[calib_idx]
            X_query, y_query = X_all[query_idx], y_all[query_idx]

            # ---- (a) zero-shot: meta-learned LoRA/head, no adaptation ----
            bb = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
            load_pretrained(bb, "backbone_pretrained.npz")
            bb.freeze_backbone_base()
            bb.add_lora_everywhere(LORA_RANK, seed=1)
            head = ClassifierHead(data.N_CLASSES, seed=1)
            lora_layers = bb._lora_layers
            for i, lyr in enumerate(lora_layers):
                lyr.A.data = meta[f"lora_A_{i}"].copy()
                lyr.B.data = meta[f"lora_B_{i}"].copy()
            head.linear.W.data = meta["head_W"].copy()
            head.linear.b.data = meta["head_b"].copy()
            zs_acc = accuracy(bb, head, X_query, y_query)
            results["zero_shot"].append(zs_acc)

            # ---- (b) random-init LoRA, fine-tuned ----
            bb_r = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
            load_pretrained(bb_r, "backbone_pretrained.npz")
            bb_r.freeze_backbone_base()
            bb_r.add_lora_everywhere(LORA_RANK, seed=1000 + si * 10 + rep)
            head_r = ClassifierHead(data.N_CLASSES, seed=1000 + si * 10 + rep)
            params_r = bb_r.lora_params() + head_r.params()
            curve_r = fine_tune_curve(bb_r, head_r, params_r, X_calib, y_calib, X_query, y_query, STEP_CHECKPOINTS, LR)
            results["random_lora"].append(curve_r)

            # ---- (c) meta-learned LoRA init, fine-tuned ----
            bb_m = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
            load_pretrained(bb_m, "backbone_pretrained.npz")
            bb_m.freeze_backbone_base()
            bb_m.add_lora_everywhere(LORA_RANK, seed=1)
            head_m = ClassifierHead(data.N_CLASSES, seed=1)
            for i, lyr in enumerate(bb_m._lora_layers):
                lyr.A.data = meta[f"lora_A_{i}"].copy()
                lyr.B.data = meta[f"lora_B_{i}"].copy()
            head_m.linear.W.data = meta["head_W"].copy()
            head_m.linear.b.data = meta["head_b"].copy()
            params_m = bb_m.lora_params() + head_m.params()
            curve_m = fine_tune_curve(bb_m, head_m, params_m, X_calib, y_calib, X_query, y_query, STEP_CHECKPOINTS, LR)
            results["meta_lora"].append(curve_m)

        print(f"  subject {si+1}/{N_EVAL_SUBJECTS} done")

    zero_shot_mean = np.mean(results["zero_shot"])
    random_curve = np.mean(results["random_lora"], axis=0)
    meta_curve = np.mean(results["meta_lora"], axis=0)
    random_std = np.std(results["random_lora"], axis=0)
    meta_std = np.std(results["meta_lora"], axis=0)

    print("\n=== Calibration accuracy on held-out query trials, unseen subjects ===")
    print(f"Zero-shot (no adaptation, meta-learned init): {zero_shot_mean:.3f}")
    print(f"{'steps':>6} | {'random-init LoRA':>18} | {'meta-learned LoRA':>18}")
    for s, r, m in zip(STEP_CHECKPOINTS, random_curve, meta_curve):
        print(f"{s:6d} | {r:18.3f} | {m:18.3f}")

    np.savez(
        "eval_results.npz",
        step_checkpoints=np.array(STEP_CHECKPOINTS),
        zero_shot=np.array(results["zero_shot"]),
        random_curve=random_curve, meta_curve=meta_curve,
        random_std=random_std, meta_std=meta_std,
        random_all=np.array(results["random_lora"]), meta_all=np.array(results["meta_lora"]),
    )
    print("\nSaved eval_results.npz")


if __name__ == "__main__":
    main()
