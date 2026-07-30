"""
Reptile meta-learning of a LoRA + classifier-head *initialization* that
adapts to a brand-new subject in very few gradient steps on very few trials.

This is the actual "calibration" contribution: instead of (a) a single
frozen foundation model with no adaptation, or (b) fine-tuning LoRA adapters
from a random init on every new subject/session (slow, needs more data), we
meta-learn where in LoRA-parameter-space to start so that a handful of SGD
steps on a handful of calibration trials gets you most of the way to a good
subject-specific model. This mirrors real work like Li et al.'s MAML for
motor-imagery decoding and the meta-learning-for-BCI literature (see
README), applied specifically to LoRA adapters on top of a frozen
foundation-model backbone (per Stacked-LoRA / SuLoRA-style subject
adaptation papers).

Algorithm: Reptile (Nichol et al. 2018) -- simpler and more stable than full
MAML (no second-order gradients needed), well suited to a from-scratch
NumPy autodiff engine.

Run: python3 meta_train.py   (requires backbone_pretrained.npz from pretrain.py)
Saves meta-learned LoRA + head init to meta_lora_init.npz
"""
import time
import numpy as np
from model import EEGBackbone, ClassifierHead, zero_grad, sgd_step, clip_grad_norm, snapshot, restore, load_pretrained
from tensor import softmax_cross_entropy
import data

SEED = 42
LORA_RANK = 4
N_META_SUBJECTS = 50
TRIALS_PER_SUBJECT = 40
SUPPORT_SIZE = 12          # trials used per inner-loop adaptation ("calibration set")
INNER_STEPS = 5
INNER_LR = 0.15
OUTER_STEPS = 250
TASK_BATCH = 6
OUTER_LR = 0.5             # Reptile step size (epsilon)


def task_loss(bb, head, patches, labels):
    _, tok = bb.encode(patches)
    logits = head(tok)
    loss, probs = softmax_cross_entropy(logits, labels)
    acc = float(np.mean(np.argmax(probs, axis=1) == labels))
    return loss, acc


def inner_adapt(bb, head, meta_params, X, y, steps, lr):
    for _ in range(steps):
        zero_grad(meta_params)
        loss, acc = task_loss(bb, head, X, y)
        loss.backward()
        clip_grad_norm(meta_params, 5.0)
        sgd_step(meta_params, lr)
    return float(loss.data), acc


def main():
    rng = np.random.default_rng(SEED)

    bb = EEGBackbone(data.C, data.N_WIN, data.WIN, seed=1)
    load_pretrained(bb, "backbone_pretrained.npz")
    bb.freeze_backbone_base()
    bb.add_lora_everywhere(LORA_RANK, seed=SEED)
    head = ClassifierHead(data.N_CLASSES, seed=SEED + 1)
    meta_params = bb.lora_params() + head.params()
    n_meta = sum(p.data.size for p in meta_params)
    print(f"Meta-learned parameter count (LoRA + head, backbone frozen): {n_meta}")

    print(f"Generating meta-training pool: {N_META_SUBJECTS} subjects x {TRIALS_PER_SUBJECT} trials "
          f"(disjoint seed range from pretraining pool)...")
    pool = data.make_population(N_META_SUBJECTS, seed=SEED + 1000, n_trials=TRIALS_PER_SUBJECT)

    t0 = time.time()
    log_loss, log_acc = [], []
    for outer_step in range(OUTER_STEPS):
        meta_snap = snapshot(meta_params)
        deltas = [np.zeros_like(s) for s in meta_snap]

        task_ids = rng.choice(N_META_SUBJECTS, size=TASK_BATCH, replace=False)
        step_losses, step_accs = [], []
        for tid in task_ids:
            restore(meta_params, meta_snap)
            X_all, y_all = pool[tid]
            idx = rng.choice(len(y_all), size=SUPPORT_SIZE, replace=False)
            X_supp, y_supp = X_all[idx], y_all[idx]

            final_loss, final_acc = inner_adapt(bb, head, meta_params, X_supp, y_supp, INNER_STEPS, INNER_LR)
            step_losses.append(final_loss)
            step_accs.append(final_acc)

            adapted = snapshot(meta_params)
            for d, a, s in zip(deltas, adapted, meta_snap):
                d += (a - s)
            restore(meta_params, meta_snap)

        for p, s, d in zip(meta_params, meta_snap, deltas):
            p.data = s + OUTER_LR * d / TASK_BATCH

        log_loss.append(np.mean(step_losses))
        log_acc.append(np.mean(step_accs))
        if outer_step % 15 == 0 or outer_step == OUTER_STEPS - 1:
            print(f"  outer {outer_step:4d}/{OUTER_STEPS}  post-adapt support loss={np.mean(log_loss[-15:]):.3f} "
                  f"acc={np.mean(log_acc[-15:]):.3f}")

    print(f"Meta-training done in {time.time()-t0:.1f}s")
    print(f"Post-adaptation support accuracy: first 15 outer steps avg={np.mean(log_acc[:15]):.3f} "
          f"-> last 15 avg={np.mean(log_acc[-15:]):.3f}")

    save = {}
    for i, (A, B) in enumerate(zip(bb.lora_params()[0::2], bb.lora_params()[1::2])):
        save[f"lora_A_{i}"] = A.data
        save[f"lora_B_{i}"] = B.data
    save["head_W"] = head.linear.W.data
    save["head_b"] = head.linear.b.data
    np.savez("meta_lora_init.npz", **save)
    print("Saved meta-learned LoRA + head init to meta_lora_init.npz")


if __name__ == "__main__":
    main()
