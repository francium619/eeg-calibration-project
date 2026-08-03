"""
Backbone pretraining on the source subjects' calibration sessions only.

Two objectives, run in sequence by default:

1. **Masked-token reconstruction** (LaBraM-style, `--objective masked`).
   Honest about its limits here: masked modelling is what makes a foundation
   model when you have ~2,500 hours across ~20 datasets. On 7 subjects x one
   session it is a warm start, not a foundation model, and treating it as
   equivalent is the main way this kind of project fools itself.

2. **Supervised cross-subject classification** (`--objective supervised`).
   With this little data this is what actually produces a backbone whose
   frozen features are linearly separable for motor imagery, and it is what
   the transfer-learning-for-BCI literature does (train on source subjects,
   adapt to target). The adapters are held at their identity init throughout
   so pretraining never bakes a particular adapter state into the base
   weights -- the previous version left random LoRA active during
   pretraining, so the base weights were optimized *around* a random
   perturbation that got thrown away at eval time.

Held-out subjects and every subject's second session are untouched here, so
the cross-session, cross-subject evaluation is clean at every stage.

Run:  python3 torch_pretrain.py --subjects 1 2 3 4 5 6 7
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from torch_data import MetaEEGDataLoader, N_CLASSES
from torch_model import make_config, build
from eeg_augment import augment_batch, mixup_batch, soft_cross_entropy


def masked_loss(bb, x, mask_frac, generator=None):
    """Predict the *unmasked* token representation at masked positions.
    Targets come from a no-grad forward of the same network, so this is a
    self-distillation objective; the stop-gradient is what keeps it from
    collapsing to a constant."""
    with torch.no_grad():
        target = bb.encode(x)
    n_mask = max(1, int(mask_frac * bb.n_tokens))
    idx = torch.randperm(bb.n_tokens, generator=generator)[:n_mask]
    pred = bb.decoder_head(bb.encode(x, mask_idx=idx))
    return F.mse_loss(pred[:, idx], target[:, idx])


def evaluate(bb, head, X, y, device, bs=256):
    bb.eval(); head.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            tok = bb.encode(X[i:i + bs].to(device))
            correct += (head(tok).argmax(-1).cpu() == y[i:i + bs]).sum().item()
    return correct / len(X)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--source", default="auto", choices=["auto", "moabb", "synthetic"])
    p.add_argument("--objective", default="both", choices=["masked", "supervised", "both"])
    p.add_argument("--masked-steps", type=int, default=300)
    p.add_argument("--sup-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--augment", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--align-mode", default="session", choices=["session", "calib", "none"])
    p.add_argument("--out", default="backbone_pretrained.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", default="full", choices=["full", "sandbox"])
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="wall-clock budget per invocation; >0 saves a resumable "
                        "state and exits, so long runs survive short command timeouts")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    data = MetaEEGDataLoader(args.subjects, source=args.source,
                             align_mode=args.align_mode, verbose=not args.quiet)
    X, y = data.pooled_session(args.subjects, which=0)   # calibration session only

    # a small in-source validation split, purely to report over/underfitting
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(X), generator=g)
    n_val = max(1, int(0.15 * len(X)))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = X[tr_i], y[tr_i], X[val_i], y[val_i]

    cfg = make_config(data.n_channels, data.n_timepoints, N_CLASSES, profile=args.profile)
    bb, head = build(cfg)
    bb, head = bb.to(device), head.to(device)
    if not args.quiet:
        print(f"[pretrain] {cfg} tokens={bb.n_tokens} train={len(Xtr)} val={len(Xva)}")

    base_params = bb.backbone_base_parameters() + list(head.parameters())

    if args.objective in ("masked", "both"):
        opt = torch.optim.AdamW(base_params, lr=args.lr, weight_decay=args.weight_decay)
        bb.train()
        for step in range(args.masked_steps):
            idx = torch.randint(0, len(Xtr), (args.batch_size,), generator=gen)
            xb = augment_batch(Xtr[idx], args.augment, data.fs, gen).to(device)
            opt.zero_grad()
            loss = masked_loss(bb, xb, 0.3, gen)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_params, 5.0)
            opt.step()
            if not args.quiet and (step % 100 == 0 or step == args.masked_steps - 1):
                print(f"  [masked] {step:4d}/{args.masked_steps} mse={loss.item():.4f}")

    best_val, best_state = -1.0, None
    start_ep = 0
    ckpt_path = args.out + ".resume"
    if args.objective in ("supervised", "both"):
        opt = torch.optim.AdamW(base_params, lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, len(Xtr) // args.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.sup_epochs * steps_per_epoch, pct_start=0.25)
        if args.resume and os.path.exists(ckpt_path):
            st = torch.load(ckpt_path, map_location=device)
            bb.load_state_dict(st["bb"]); head.load_state_dict(st["head"])
            opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            start_ep, best_val, best_state = st["ep"], st["best_val"], st["best_state"]
            if not args.quiet:
                print(f"[pretrain] resumed at epoch {start_ep} (best src-val {best_val:.3f})")
        t_start = time.time()
        for ep in range(start_ep, args.sup_epochs):
            bb.train(); head.train()
            order = torch.randperm(len(Xtr), generator=gen)
            tot = 0.0
            for b in range(steps_per_epoch):
                idx = order[b * args.batch_size:(b + 1) * args.batch_size]
                xb = augment_batch(Xtr[idx], args.augment, data.fs, gen)
                xb, soft = mixup_batch(xb, ytr[idx], N_CLASSES, args.mixup, gen)
                opt.zero_grad()
                logits = head(bb.encode(xb.to(device)))
                loss = soft_cross_entropy(logits, soft.to(device), args.label_smoothing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(base_params, 5.0)
                opt.step(); sched.step()
                tot += loss.item()
            va = evaluate(bb, head, Xva, yva, device)
            if va > best_val:
                best_val = va
                best_state = ({k: v.detach().clone() for k, v in bb.state_dict().items()},
                              {k: v.detach().clone() for k, v in head.state_dict().items()})
            if not args.quiet and (ep % 10 == 0 or ep == args.sup_epochs - 1):
                print(f"  [sup] epoch {ep:3d}/{args.sup_epochs} loss={tot/steps_per_epoch:.4f} "
                      f"src-val acc={va:.3f} (best {best_val:.3f})")
            if args.max_seconds and (time.time() - t_start) > args.max_seconds and ep < args.sup_epochs - 1:
                torch.save({"bb": bb.state_dict(), "head": head.state_dict(),
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "ep": ep + 1, "best_val": best_val, "best_state": best_state},
                           ckpt_path)
                print(f"[pretrain] time budget hit at epoch {ep+1}/{args.sup_epochs}; "
                      f"state saved -> rerun with --resume")
                return None

    # Early stopping on the *source* validation split -- never on the held-out
    # subjects, which would be selecting the checkpoint on the test set.
    if best_state is not None:
        bb.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])

    torch.save({"backbone": bb.state_dict(), "head": head.state_dict(),
                "cfg": cfg.__dict__, "subjects": args.subjects,
                "align_mode": args.align_mode, "source": data.source,
                "src_val_acc": best_val}, args.out)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if not args.quiet:
        print(f"[pretrain] saved {args.out} (source-val acc {best_val:.3f})")
    return best_val


if __name__ == "__main__":
    main()
