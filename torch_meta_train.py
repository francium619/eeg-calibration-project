"""
Meta-learning the adapter initialization (Reptile / first-order MAML).

The meta-objective is written to be *literally the deployment objective*:
"after k gradient steps on 12 stratified trials recorded on day 1, how well
does this subject decode on day 2?" Concretely, for each source subject the
support set is a few-shot sample from session 1 and the query set is all of
session 2. The first draft optimized post-adaptation accuracy on the support
set itself, which rewards memorizing 12 trials -- the exact failure mode the
experiment is supposed to detect.

Three things here are not cosmetic:

1. **Fresh, momentum-free inner optimizer per task.** Reusing one Adam across
   subjects lets subject A's adaptation trajectory leak into subject B's
   through the moment buffers, corrupting the initialization Reptile is
   trying to estimate. Inner loop is plain SGD, rebuilt per task.

2. **Meta-SGD: per-parameter learned inner learning rates** (`--meta-sgd`).
   A single global inner LR has to be simultaneously right for a rank-8 LoRA
   matrix and for the classifier bias. Learning a per-parameter step size
   alongside the initialization (Li et al., "Meta-SGD") is a small change
   that consistently buys more than tuning the scalar LR by hand, and in the
   few-shot regime the step size matters as much as the starting point.

3. **Adam on the outer loop.** Reptile's `(adapted - init)` is used as a
   pseudo-gradient fed to an outer Adam rather than applied as a raw
   interpolation. Same fixed point, far less sensitive to the outer LR.

FOMAML (`--algo fomaml`) differentiates the *query* loss at the adapted
parameters and is the better-aligned objective; Reptile is cheaper and more
stable. Both are implemented so the choice is an empirical one.

Run: python3 torch_meta_train.py --subjects 1 2 3 4 5 6 7
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from torch_data import MetaEEGDataLoader, N_CLASSES
from torch_model import ModelConfig, build, adapter_state_dict, CalibModel
from eeg_augment import augment_batch


def snapshot(params):
    return [p.detach().clone() for p in params]


@torch.no_grad()
def restore(params, snap):
    for p, s in zip(params, snap):
        p.copy_(s)


def forward_loss(bb, head, x, y, label_smoothing=0.0):
    return F.cross_entropy(head(bb.encode(x)), y, label_smoothing=label_smoothing)


def functional_inner_adapt(model, names, theta, x, y, steps, lr, inner_lrs=None,
                           clip=5.0, augment=0.0, fs=250.0, gen=None, label_smoothing=0.0):
    """A *differentiable* inner loop.

    `theta` is a list of tensors standing in for the calibrated parameters;
    each step produces new tensors rather than mutating leaves, so the query
    loss computed at the end is a differentiable function of the inner
    learning rates. The parameter gradients are detached at each step, which
    is exactly the first-order MAML approximation -- we keep the (cheap,
    exact) gradient path into `inner_lrs` and drop the (expensive) second-
    order path through the parameters.

    The previous in-place `with torch.no_grad(): p -= lr*g` version silently
    severed *both* paths, which meant the "learned" per-parameter learning
    rates of Meta-SGD received no gradient at all and never moved from their
    initialization. The meta-learner was quietly a plain fixed-LR Reptile.
    """
    from torch.func import functional_call
    for _ in range(steps):
        xb = augment_batch(x, augment, fs, gen) if augment > 0 else x
        out = functional_call(model, dict(zip(names, theta)), (xb,))
        loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)
        grads = torch.autograd.grad(loss, theta, allow_unused=True)
        total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None))
        scale = torch.clamp(clip / (total + 1e-12), max=1.0)
        new = []
        for i, (t, g) in enumerate(zip(theta, grads)):
            if g is None:
                new.append(t)
                continue
            step = F.softplus(inner_lrs[i]) if inner_lrs is not None else lr
            new.append(t - step * g.detach() * scale)
        theta = new
    return theta, float(loss.item())


def accuracy(bb, head, X, y, device, bs=256):
    bb.eval(); head.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            correct += (head(bb.encode(X[i:i + bs].to(device))).argmax(-1).cpu()
                        == y[i:i + bs]).sum().item()
    return correct / len(X)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--source", default="auto", choices=["auto", "moabb", "synthetic"])
    p.add_argument("--pretrained", default="backbone_pretrained.pt")
    p.add_argument("--algo", default="fomaml", choices=["reptile", "fomaml"])
    p.add_argument("--meta-sgd", action="store_true", default=True)
    p.add_argument("--no-meta-sgd", dest="meta_sgd", action="store_false")
    p.add_argument("--outer-steps", type=int, default=200)
    p.add_argument("--outer-lr", type=float, default=5e-3)
    p.add_argument("--inner-steps", type=int, default=5)
    p.add_argument("--inner-lr", type=float, default=0.05)
    p.add_argument("--tasks-per-step", type=int, default=4)
    p.add_argument("--calib-size", type=int, default=12)
    p.add_argument("--query-size", type=int, default=64)
    p.add_argument("--augment", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--align-mode", default="session", choices=["session", "calib", "none"])
    p.add_argument("--out", default="meta_lora_init.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="wall-clock budget per invocation; saves resumable state and exits")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.pretrained, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    data = MetaEEGDataLoader(args.subjects, source=args.source,
                             align_mode=args.align_mode, verbose=not args.quiet)
    assert data.n_channels == cfg.n_channels, "channel mismatch with pretrained checkpoint"

    bb, head = build(cfg)
    # strict=True: a silently-partial load here is the difference between
    # "meta-learning on pretrained features" and "meta-learning on noise".
    bb.load_state_dict(ckpt["backbone"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    bb, head = bb.to(device), head.to(device)
    bb.freeze_backbone_base()
    bb.set_frozen_eval_mode()

    params = bb.lora_parameters() + head.parameters_for_calibration()
    for p_ in params:
        p_.requires_grad_(True)
    if not args.quiet:
        print(f"[meta] algo={args.algo} meta_sgd={args.meta_sgd} "
              f"meta-params={sum(p_.numel() for p_ in params)} "
              f"(frozen base={sum(p_.numel() for p_ in bb.backbone_base_parameters())})")

    model = CalibModel(bb, head).to(device)
    names = model.adapter_names()
    theta0 = [dict(model.named_parameters())[n] for n in names]
    assert len(theta0) == len(params), "adapter name lookup missed a parameter"

    inner_lrs = None
    outer_targets = list(theta0)
    if args.meta_sgd:
        raw = float(np.log(np.expm1(args.inner_lr)))   # softplus(raw) == inner_lr
        inner_lrs = [torch.full_like(t, raw, requires_grad=True) for t in theta0]
        outer_targets = list(theta0) + inner_lrs

    opt = torch.optim.Adam(outer_targets, lr=args.outer_lr)

    history, start_step = [], 0
    resume_path = args.out + ".resume"
    if args.resume and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=device)
        restore(theta0, st["params"])
        if inner_lrs is not None and st["inner_lrs"] is not None:
            with torch.no_grad():
                for l, v in zip(inner_lrs, st["inner_lrs"]):
                    l.copy_(v)
        opt.load_state_dict(st["opt"]); history = st["history"]; start_step = st["step"]
        if not args.quiet:
            print(f"[meta] resumed at outer step {start_step}")

    t_start = time.time()
    for step in range(start_step, args.outer_steps):
        accum = [torch.zeros_like(t) for t in outer_targets]
        task_ids = rng.choice(args.subjects,
                              size=min(args.tasks_per_step, len(args.subjects)), replace=False)
        q_accs = []
        for sid in task_ids:
            (xs, ys), (xq_all, yq_all) = data.calibration_and_query(
                int(sid), args.calib_size, seed=int(rng.integers(1e9)))
            qi = data.stratified_indices(yq_all, args.query_size, seed=int(rng.integers(1e9)))
            xq, yq = xq_all[qi].to(device), yq_all[qi].to(device)
            xs, ys = xs.to(device), ys.to(device)

            theta_adapted, _ = functional_inner_adapt(
                model, names, theta0, xs, ys, args.inner_steps, args.inner_lr,
                inner_lrs=inner_lrs, augment=args.augment, fs=data.fs, gen=gen,
                label_smoothing=args.label_smoothing)

            # ---- meta-objective: loss on the *other day's* session ----
            from torch.func import functional_call
            q_out = functional_call(model, dict(zip(names, theta_adapted)), (xq,))
            q_loss = F.cross_entropy(q_out, yq, label_smoothing=args.label_smoothing)
            q_accs.append(float((q_out.argmax(-1) == yq).float().mean()))

            if args.algo == "fomaml":
                g = torch.autograd.grad(q_loss, outer_targets, allow_unused=True, retain_graph=False)
                for a, gi in zip(accum, g):
                    if gi is not None:
                        a += gi
            else:  # reptile: pseudo-gradient = -(adapted - init); LRs still via query loss
                with torch.no_grad():
                    for i, (a, t_new, t_old) in enumerate(zip(accum, theta_adapted, theta0)):
                        a += -(t_new.detach() - t_old.detach())
                if inner_lrs is not None:
                    g = torch.autograd.grad(q_loss, inner_lrs, allow_unused=True)
                    for j, gi in enumerate(g):
                        if gi is not None:
                            accum[len(theta0) + j] += gi

        opt.zero_grad()
        for t, a in zip(outer_targets, accum):
            t.grad = a / len(task_ids)
        torch.nn.utils.clip_grad_norm_(outer_targets, 5.0)
        opt.step()

        history.append(float(np.mean(q_accs)))
        if not args.quiet and (step % 20 == 0 or step == args.outer_steps - 1):
            lr_note = ""
            if inner_lrs is not None:
                lr_note = f" inner-lr mean={float(F.softplus(torch.cat([l.flatten() for l in inner_lrs])).mean()):.4f}"
            print(f"  [meta] outer {step:4d}/{args.outer_steps} "
                  f"post-adapt QUERY-session acc (avg last 20) = {np.mean(history[-20:]):.3f}{lr_note}")

        if args.max_seconds and (time.time() - t_start) > args.max_seconds and step < args.outer_steps - 1:
            torch.save({"params": snapshot(theta0),
                        "inner_lrs": [l.detach().clone() for l in inner_lrs] if inner_lrs else None,
                        "opt": opt.state_dict(), "history": history, "step": step + 1}, resume_path)
            print(f"[meta] time budget hit at outer {step+1}/{args.outer_steps}; "
                  f"state saved -> rerun with --resume")
            return None

    if os.path.exists(resume_path):
        os.remove(resume_path)
    sd = adapter_state_dict(bb, head)
    # Save the parameter *names* alongside the learned inner LRs. The LR list
    # is positional, so a consumer that rebuilds the parameter list in a
    # different order would apply the LoRA-A learning rate to the classifier
    # bias and never notice. The eval script asserts on this.
    sd["adapter_names"] = names
    sd["inner_lrs"] = [l.detach().clone() for l in inner_lrs] if inner_lrs else None
    sd["algo"] = args.algo
    sd["meta_history"] = history
    sd["cfg"] = cfg.__dict__
    torch.save(sd, args.out)
    if not args.quiet:
        print(f"[meta] saved {args.out}  final query acc {np.mean(history[-20:]):.3f}")
    return float(np.mean(history[-20:]))


if __name__ == "__main__":
    main()
