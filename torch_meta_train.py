"""
PyTorch Reptile meta-training of the LoRA + head init, using real BCI IV 2a
data via torch_data.py, porting meta_train.py's logic (validated in the
NumPy prototype: post-adaptation support accuracy rose 62%->96% over
training there) onto the real backbone/data.

This fills in the three things left as pseudocode in your sketch, each of
which is a real correctness issue for Reptile specifically, not just
boilerplate:

1. `copy_lora_weights` / `meta_step_update` -- Reptile's actual update rule
   is `meta_param <- meta_param + eps * (adapted_param - meta_param)`,
   averaged (or applied sequentially) across tasks. This needs an explicit
   snapshot/restore of just the LoRA+head params (not the whole model).
2. **Freeze the backbone before building the optimizer.** Your sketch's
   inner loop calls `optimizer.step()` without showing that the optimizer
   only contains LoRA+head params -- if it accidentally includes the frozen
   backbone, "frozen" stops being true the moment you call `.step()`.
3. **Use a fresh, momentum-free optimizer per task's inner loop.** If you
   reuse one Adam optimizer across subjects, its running moment estimates
   carry over from one subject's few-shot adaptation into the next one's,
   which contaminates the very thing Reptile is trying to learn (a good
   *starting point* for adaptation, not a good *trajectory* biased by
   whatever subject came before). Plain SGD, reconstructed per task, avoids
   this -- same choice the NumPy prototype made.

Not executed in this sandbox -- see torch_model.py's docstring for why, and
README.md for how to validate before trusting results (e.g. run 5-10 outer
steps and confirm post-adaptation support accuracy climbs, the way the
NumPy version's log did).
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch_model import EEGBackbone, ClassifierHead, lora_state_dict
from torch_data import MetaEEGDataLoader, EVENTS

SEED = 42
INNER_STEPS = 5
INNER_LR = 0.15
OUTER_STEPS = 300
OUTER_LR = 0.5           # Reptile step size (epsilon)
CALIB_SIZE = 12          # trials per inner-loop adaptation, matched to eval_calibration's few-shot budget
GRAD_CLIP = 5.0


def meta_parameters(backbone, head):
    return backbone.lora_parameters() + head.parameters_for_calibration()


def snapshot(params):
    return [p.detach().clone() for p in params]


def restore(params, snap):
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


def inner_adapt(backbone, head, params, X, y, steps, lr, device):
    """Fresh, momentum-free SGD for the few-shot inner loop -- see module
    docstring point 3 for why this specifically must not be Adam-with-carried-state."""
    opt = torch.optim.SGD(params, lr=lr)
    X, y = X.to(device), y.to(device)
    patches = backbone.patchify(X)  # (B, C, T) raw trial -> (B, tokens, win_len)
    for _ in range(steps):
        opt.zero_grad()
        _, tok = backbone.encode(patches)
        logits = head(tok)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
    acc = float((logits.argmax(-1) == y).float().mean())
    return float(loss.item()), acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    meta_train_subjects = [1, 2, 3, 4, 5, 6, 7]  # e.g. hold out 8, 9 for eval -- see torch_eval_calibration.py
    pipeline = MetaEEGDataLoader(subject_ids=meta_train_subjects)

    backbone = EEGBackbone(
        n_channels=pipeline.n_channels,
        win_len=50,  # 0.2s windows @ resample_fs=250 -- tune to taste
        n_windows=pipeline.n_timepoints // 50,
    ).to(device)
    try:
        backbone.load_state_dict(torch.load("backbone_pretrained.pt", map_location=device), strict=False)
        print("Loaded backbone_pretrained.pt")
    except FileNotFoundError:
        print("WARNING: backbone_pretrained.pt not found -- meta-learning LoRA on top of an "
              "UNTRAINED, randomly-initialized backbone. Run torch_pretrain.py first; freezing "
              "random features and calibrating on top of them will run without error but defeats "
              "the point (same failure mode the NumPy prototype's README warns about).")
    backbone.freeze_backbone_base()
    head = ClassifierHead(n_classes=len(EVENTS)).to(device)

    params = meta_parameters(backbone, head)
    print(f"Meta-learned parameter count (LoRA + head, backbone frozen): {sum(p.numel() for p in params)}")

    rng = np.random.default_rng(SEED)
    log_acc = []
    for outer_step in range(OUTER_STEPS):
        meta_snap = snapshot(params)
        deltas = [torch.zeros_like(s) for s in meta_snap]

        task_subjects = rng.choice(meta_train_subjects, size=min(4, len(meta_train_subjects)), replace=False)
        step_accs = []
        for sub_id in task_subjects:
            restore(params, meta_snap)

            # Cross-session split even during meta-training: calibrate on
            # session 0, adapt, so the meta-init is optimized for the same
            # kind of shift it will face at eval time (a new subject AND a
            # session boundary), not just a new subject on a familiar day.
            calib_loader, _ = pipeline.get_subject_session_split_loaders(
                int(sub_id), calib_size=CALIB_SIZE, batch_size=CALIB_SIZE, seed=int(rng.integers(1e6))
            )
            X_calib, y_calib = next(iter(calib_loader))

            _, acc = inner_adapt(backbone, head, params, X_calib, y_calib, INNER_STEPS, INNER_LR, device)
            step_accs.append(acc)

            adapted = snapshot(params)
            for d, a, s in zip(deltas, adapted, meta_snap):
                d += (a - s)
            restore(params, meta_snap)

        with torch.no_grad():
            for p, s, d in zip(params, meta_snap, deltas):
                p.copy_(s + OUTER_LR * d / len(task_subjects))

        log_acc.append(np.mean(step_accs))
        if outer_step % 20 == 0 or outer_step == OUTER_STEPS - 1:
            print(f"  outer {outer_step:4d}/{OUTER_STEPS}  post-adapt support acc (running avg) = {np.mean(log_acc[-20:]):.3f}")

    torch.save(
        {"lora": lora_state_dict(backbone), "head": head.state_dict()},
        "meta_lora_init.pt",
    )
    print("Saved meta-learned LoRA + head init to meta_lora_init.pt")


if __name__ == "__main__":
    main()
