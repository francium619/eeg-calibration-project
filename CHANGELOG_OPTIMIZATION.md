# Optimization log

Each rung is a change, the reason, and the measured effect. All PyTorch
numbers below are from the offline synthetic fixture (`--source synthetic`)
because this development environment has no route to the real dataset servers
(only PyPI is reachable). The fixture is tuned so within-subject cross-session
decoding sits near 0.53 and cross-subject zero-shot near chance — the same
regime as real BCI IV 2a — precisely so the ladder measures the code rather
than an easy dataset. **These numbers characterize the pipeline, not the
brain.** Re-run with `--source moabb` for real results.

Evaluation is identical at every rung: subjects 8 and 9 held out of both
pretraining and meta-training, calibrate on session 1, test on session 2,
12 stratified calibration trials, matched optimizer/LR/step counts across
arms, paired bootstrap 95% CIs.

## Rung 0 — starting state

The PyTorch port had never been executed. Getting it to run surfaced the
following, each of which independently invalidates results:

| # | bug | consequence |
|---|---|---|
| 1 | MOABB output (volts, ~1e-5) fed to the network unnormalized | gradients ~1e-10; the model learns essentially nothing |
| 2 | `load_state_dict(..., strict=False)` inside `try/except FileNotFoundError` | a renamed or mismatched key loads *nothing* and prints success |
| 3 | pretrained checkpoint carried LoRA weights; the "random-init LoRA" arm loaded them | the control arm was not a control — it was pretrained-LoRA |
| 4 | LoRA active but untrained during pretraining | base weights were optimized around a random perturbation that got discarded at eval |
| 5 | calibration trials drawn uniformly at random | 12 trials over 4 classes routinely misses a class; accuracy capped at 0.75 |
| 6 | meta-objective was post-adaptation accuracy on the *support* set | rewards memorizing the 12 calibration trials — the exact failure being studied |
| 7 | BatchNorm left in train mode during "frozen backbone" calibration | running statistics updated; the frozen backbone was not frozen |
| 8 | Meta-SGD inner LRs updated inside `torch.no_grad()` | they received no gradient and never moved — Meta-SGD was silently plain Reptile |
| 9 | learned inner-LR list positional, parameter list rebuilt independently in eval | LoRA-A's learning rate could be applied to the classifier bias |

## Rung 1 — preprocessing

Per-trial z-scoring, Euclidean Alignment, stratified few-shot sampling,
optional filter bank.

| | zero-shot | best meta |
|---|---|---|
| no alignment | 0.549 | 0.698 |
| alignment from calibration session (inductive) | 0.597 | 0.715 |
| alignment per session (default) | **0.611** | **0.734** |

Euclidean Alignment alone: **+6.2 points zero-shot**. Label-free, ~10 lines.

## Rung 2 — architecture: square/log power stem

Symptom: with the harder fixture, cross-subject pretraining sat at chance
(0.25) no matter how long it trained.

Cause: the stem was patch-embed / ELU only. Motor imagery is a *band-power*
phenomenon — ERD is a percentage power change in mu/beta. A network with no
squaring nonlinearity has to approximate power with piecewise-linear pieces,
and at this data scale it does not.

Fix: ShallowFBCSPNet-style stem — temporal conv, learned spatial (CSP-like)
depthwise conv, then **square -> sliding-window average pool -> log** before
the transformer blocks.

| | source-val accuracy (7 subjects) |
|---|---|
| ELU stem | 0.250 (chance) |
| square/log power stem | **0.649** |

Largest single architecture change in the project.

## Rung 3 — meta-learning: Reptile -> FOMAML with a differentiable inner loop

Reptile with the (broken) Meta-SGD made the meta-learned initialization
*worse than no calibration at all*:

| arm | best accuracy | vs zero-shot |
|---|---|---|
| zero-shot | 0.611 | — |
| head-only probe | 0.620 | +0.009 |
| random-init LoRA | 0.611 | +0.000 |
| Reptile meta-LoRA | 0.575 | **-0.036** |

Reptile's update moves toward each task's few-shot optimum, which on 12
trials *is* the overfitted solution — with only 7 source subjects and an
already-good pretrained head, that degrades the initialization. And because
the inner loop mutated parameters under `torch.no_grad()`, the per-parameter
inner learning rates were receiving no gradient at all.

Fix: a functional, differentiable inner loop (`torch.func.functional_call`).
Parameter gradients are still detached each step — that is first-order MAML —
but the dependence on the inner learning rates is preserved exactly, so
Meta-SGD works. The meta-objective is the loss on the subject's *other-day*
session, i.e. literally the deployment objective.

| arm | best accuracy | vs zero-shot | paired 95% CI |
|---|---|---|---|
| zero-shot | 0.611 | — | — |
| head-only probe | 0.620 | +0.009 | not significant |
| random-init LoRA | 0.612 | +0.001 | not significant |
| **FOMAML + Meta-SGD meta-LoRA** | **0.734** | **+0.123** | **[+0.094, +0.155]** |

Meta-init reaches 0.712 after a *single* gradient step on 12 trials.

## Rung 4 — evaluation harness

`run_ablation.py` keeps an append-only JSONL ledger; every row carries the
paired bootstrap CI against zero-shot. Current ledger:

| tag | zero-shot | best meta | @steps | meta - zero-shot | 95% CI |
|---|---|---|---|---|---|
| baseline | 0.611 | 0.734 | 8 | +0.123 | [+0.094, +0.155] |
| align_none | 0.549 | 0.698 | 8 | +0.149 | [+0.090, +0.208] |
| align_calib | 0.597 | 0.715 | 3 | +0.118 | [+0.092, +0.139] |
| calib4 (4 trials) | 0.611 | 0.720 | 1 | +0.109 | [+0.069, +0.149] |
| calib24 (24 trials) | 0.611 | 0.726 | 20 | +0.115 | [+0.066, +0.163] |
| aug_calib | 0.611 | 0.736 | 8 | +0.125 | [+0.109, +0.141] |

## Rung 5 — leakage controls

`test_torch_pipeline.py`, 14 checks, all passing. The ones that matter:

- **shuffled-label control**: calibrating on randomly permuted labels stays at
  0.267 (chance 0.250). If this ever clears chance, the pipeline is leaking.
- calibration and query sets are disjoint, and come from different sessions
- frozen base weights bit-identical after calibration
- BatchNorm running statistics unchanged (frozen means frozen)
- adapters are an exact no-op at initialization, so "zero-shot" is zero-shot
- adapter checkpoints contain no base weights
- Euclidean alignment verified to actually whiten (max deviation 0.000)

## Still open

- Real BNCI2014_001 numbers (blocked on network here — see the playbook).
- Leave-one-subject-out over all 9 folds rather than a fixed pair.
- Larger source-subject pool; this is the biggest remaining lever.
- LaBraM backbone swap.
