# Meta-Learned LoRA Calibration for EEG Foundation Models

## The problem

EEG-based brain-computer interfaces work in the lab and fail in the field for
one main reason: a decoding model trained on today's brain signals doesn't
transfer to tomorrow's, or to a different person's, without new calibration
data. Electrode placement, skin conductivity, and even the same person's
mental state drift session to session. There is currently no standard way to
take a large pretrained EEG model and *quickly* re-align it to a new brain
without collecting a full new labeled dataset and retraining.

Real EEG foundation models now exist — most notably **LaBraM**
([paper](https://arxiv.org/abs/2405.18765),
[code + weights](https://github.com/935963004/LaBraM)), pretrained on ~2,500
hours of EEG across ~20 datasets using masked neural-code reconstruction —
but they're typically deployed with either a frozen linear probe (doesn't
adapt at all) or full fine-tuning (needs lots of subject-specific data, too
slow for real-time use).

## The approach this project implements

Freeze the foundation model's backbone. Attach small trainable **LoRA**
(low-rank adaptation) matrices to its linear layers. Instead of initializing
those LoRA matrices randomly for every new subject (slow — needs many
calibration trials to converge), **meta-learn a LoRA initialization** across
many training subjects such that a handful of gradient steps on a handful of
calibration trials from a *brand-new* subject gets you most of the way to a
subject-specific model.

This combines two lines of real, current research (see citations below):

1. **LoRA for EEG subject adaptation** — e.g. "Stacked LoRA for
   Subject-Adaptive EEG Foundation Models" and "Subject-Specific Low-Rank
   Adapters (SuLoRA)" both add frozen-backbone + LoRA adapters for
   per-subject calibration.
2. **Meta-learning for BCI few-shot calibration** — MAML/Reptile-style
   approaches (Li et al. 2021 and others) that learn an initialization
   which adapts to a new subject/session in very few gradient steps.

Nobody in the papers I found combines the two as the actual adapter being
meta-learned (rather than meta-learning the whole network or a separate
subject-embedding). That combination — meta-learn a LoRA init on top of a
frozen EEG foundation model — is the specific idea this project prototypes
and is a reasonable, well-scoped angle for a project.

## Status

| stage | state |
|---|---|
| NumPy prototype (`tensor.py`, `model.py`, `data.py`, ...) | built, executed, gradient-checked |
| PyTorch pipeline (`torch_*.py`) | **built, executed, tested** — 20-check regression suite incl. a shuffled-label leakage control |
| Real BCI IV 2a numbers | **download verified, training not yet run** — see the caveat under "Headline result" |

## Quickstart

```bash
pip install torch                        # add `moabb` for the real dataset
python3 test_torch_pipeline.py           # 20 checks, all green
python3 torch_pretrain.py         --subjects 1 2 3 4 5 6 7
python3 torch_meta_train.py       --subjects 1 2 3 4 5 6 7
python3 torch_eval_calibration.py --eval-subjects 8 9
python3 run_ablation.py --run baseline align_none tta calib24
```

With `moabb` installed and network available it uses real BCI Competition
IV 2a (`BNCI2014_001`: 9 subjects × 2 sessions recorded on different days,
4-class motor imagery, 22 channels @ 250 Hz). Without it, it falls back to a
shape-identical synthetic fixture so the code stays runnable and testable
offline. The source is printed and recorded in every result row, so a fixture
number can never be mistaken for a real one.

## Headline result

Subjects 8 and 9 held out of pretraining *and* meta-training; calibrate on
session 1, evaluate on session 2 (a different day); 12 stratified calibration
trials; identical optimizer, learning rate and step budget for every arm.

| arm | best accuracy | vs zero-shot | paired 95% CI |
|---|---|---|---|
| chance | 0.250 | | |
| zero-shot (no calibration) | 0.611 | — | — |
| head-only linear probe | 0.620 | +0.009 | not significant |
| random-init LoRA | 0.612 | +0.001 | not significant |
| **meta-learned LoRA (FOMAML + Meta-SGD)** | **0.734** | **+0.123** | **[+0.094, +0.155]** |

The meta-learned initialization reaches 0.712 after a **single** gradient step
on 12 trials — which is the actual claim this project exists to test.

> **These numbers are from the synthetic fixture, not real EEG.** Real
> BNCI2014_001 downloads now verify end to end (9/9 subjects, 18 files,
> 779.9 MB, checked against the published spec — see `moabb_path_fix.py`
> and the `n_classes` fix in `torch_data.py`), but the training benchmark
> itself has not yet been re-run against it, so the table above still comes
> from the fixture. The fixture is deliberately tuned to the same difficulty
> regime as BCI IV 2a (within-subject cross-session ≈ 0.53, cross-subject
> zero-shot near chance) so the ablation ladder measures the *pipeline*
> rather than an easy dataset. Run with `--source moabb` for numbers about
> brains.

## What changed and what it was worth

Full log in [CHANGELOG_OPTIMIZATION.md](CHANGELOG_OPTIMIZATION.md).

| change | effect |
|---|---|
| per-trial standardization (MOABB volts were being fed in raw) | pipeline learns at all |
| Euclidean Alignment (He & Wu 2020) | **+6.2 points zero-shot** |
| square/log power stem (ShallowFBCSPNet-style) instead of ELU-only | source-val **0.250 → 0.649** — from not learning to learning |
| Reptile → FOMAML with a differentiable inner loop | **−0.036 → +0.123** vs zero-shot |
| stratified few-shot sampling | removes a silent 0.75 accuracy ceiling |
| explicit adapter reset in the control arm | the "random LoRA" baseline had been *pretrained* LoRA |
| BatchNorm frozen during calibration | the "frozen" backbone was not frozen |

Nine correctness bugs were found by actually executing the port; they are
listed individually in the changelog.

## On "perfect accuracy"

Four-class motor imagery does not go to 100%. Published within-subject work on
BCI IV 2a lands at 68–80%, cross-subject few-shot at 55–70%, and 15–30% of
people are "BCI illiterate" for motor imagery at all. Any pipeline reporting
near-perfect accuracy on this task has a leak — which is why
`test_torch_pipeline.py` includes a shuffled-label control (calibrating on
permuted labels must stay at chance; it scores 0.267 vs 0.250) and why every
result carries a paired confidence interval. The target this project optimizes
is: **beat the zero-shot baseline by a statistically significant margin at the
smallest possible calibration budget.**

## What's in this folder

### The PyTorch pipeline (current)

| file | what it does |
|---|---|
| `torch_data.py` | MOABB loading + synthetic fixture, per-trial standardization, Euclidean Alignment (two modes), stratified few-shot sampling, optional filter bank |
| `torch_model.py` | `SpatialAdapter` (low-rank channel remix for montage/impedance shift) → ShallowFBCSP-style square/log power stem → multi-head transformer with `LoRALinear` on every projection → attention-pooling head. `CalibModel` exposes the whole thing for functional/differentiable inner loops |
| `eeg_augment.py` | time shift, channel dropout, amplitude jitter, noise, frequency masking, mixup — each chosen to preserve the label |
| `torch_pretrain.py` | masked-token + supervised cross-subject pretraining on source subjects' first sessions only; early-stops on a *source* validation split |
| `torch_meta_train.py` | FOMAML / Reptile with Meta-SGD per-parameter inner learning rates; meta-objective is the loss on the subject's *other-day* session |
| `torch_eval_calibration.py` | the experiment: zero-shot vs head-only vs random-LoRA vs meta-LoRA at matched compute, with paired bootstrap CIs |
| `run_ablation.py` | append-only JSONL results ledger + ablation ladder |
| `test_torch_pipeline.py` | 20 regression checks, including the shuffled-label leakage control |

Both `torch_pretrain.py` and `torch_meta_train.py` support
`--max-seconds` / `--resume`, so long runs survive short command timeouts.

### The NumPy prototype (stage 1, kept)

A dependency-free implementation of the same idea, written before PyTorch was
available in the development environment. Still useful as a from-scratch
reference and still passing its tests.

| file | what it does |
|---|---|
| `tensor.py` | ~250-line reverse-mode autodiff engine, gradient-checked against finite differences |
| `model.py` | small EEG transformer with `Linear.add_lora()` |
| `data.py` | synthetic multi-subject/multi-session generator |
| `pretrain.py`, `meta_train.py`, `eval_calibration.py`, `plot_results.py` | the original pipeline |
| `test_tensor.py`, `test_model.py` | gradient checks + freeze/LoRA sanity checks — all passing |

Run: `python3 pretrain.py && python3 meta_train.py && python3 eval_calibration.py && python3 plot_results.py`

Its result (`calibration_comparison.png`): the meta-learned LoRA init beat a
random init at every calibration budget, improving 0.511 → 0.518 with more
steps while the random init *degraded* 0.509 → 0.485 — few-shot overfitting,
which is a real documented phenomenon in the BCI meta-learning literature and
a useful sign the setup behaved like the actual problem. Those margins are
small; the PyTorch pipeline above is the version to trust.

## Honest limitations

- **No real EEG data has been through the training benchmark yet.** The
  `--source moabb` download path is now verified working end to end (see
  `moabb_path_fix.py`), so producing real headline numbers is a matter of
  running the pipeline, not further debugging the loader.
- **The backbone is trained from scratch**, not a real foundation model. It is
  much smaller than LaBraM (~200-dim, 12 layers). The `LoRALinear` +
  `SpatialAdapter` pattern applies directly to any loaded `nn.Linear`, which
  is what the LaBraM swap in the playbook describes.
- **First-order MAML, not full second-order.** The inner loop is already
  functional and differentiable, so full MAML is now a one-line change
  (`create_graph=True`, stop detaching parameter gradients) — expect a small
  gain for ~3× cost.
- **Only 7 source subjects** for meta-training. Published meta-learning-for-BCI
  work pools dozens to low hundreds. This is the single biggest remaining
  lever and step 2 of the playbook.
- **`align_mode="session"` is transductive-unsupervised** — it estimates the
  whitening matrix from the query session's *unlabeled* inputs. Standard and
  realistic (you record ~1 min of unlabeled EEG at session start), but the
  strictly inductive `align_mode="calib"` is implemented and reported
  alongside so nobody has to take it on faith.

## Where to go next

[FINE_TUNING_PLAYBOOK.md](FINE_TUNING_PLAYBOOK.md) is the prioritized work
order for real raw data: the dataset ladder (BNCI2014_004, PhysionetMI's 109
subjects, Cho2017's 52, ...), the LaBraM backbone swap, Riemannian tangent-
space ensembling, test-time adaptation, and the evaluation protocol to hold
to. Short version, if only three things get done:

1. Run the existing pipeline on real BNCI2014_001 and check the acceptance
   criteria.
2. Pool `PhysionetMI` + `Cho2017` to get from 7 source subjects to ~100.
3. Swap the from-scratch backbone for LaBraM.

## Key references

- Jiang et al., "Large Brain Model for Learning Generic Representations
  with Tremendous EEG Data in BCI" (LaBraM), ICLR 2024 spotlight —
  [arXiv](https://arxiv.org/abs/2405.18765) /
  [code+weights](https://github.com/935963004/LaBraM)
- "Stacked LoRA for Subject-Adaptive EEG Foundation Models in Motor
  Imagery Decoding" — [arXiv](https://arxiv.org/html/2607.03094v1)
- "Mitigating Subject Dependency in EEG Decoding with Subject-Specific
  Low-Rank Adapters" (SuLoRA) — [arXiv](https://arxiv.org/html/2510.08059)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" —
  [arXiv](https://arxiv.org/abs/2106.09685)
- Nichol, Achiam, Schulman, "On First-Order Meta-Learning Algorithms"
  (Reptile) — the outer-loop update this project uses.
- "Meta-Learning for Fast and Privacy-Preserving Source Knowledge
  Transfer of EEG-Based BCIs" —
  [IEEE](https://ieeexplore.ieee.org/document/9942685/)
- "TCPL: task-conditioned prompt learning for few-shot cross-subject
  motor imagery EEG decoding" (2025) —
  [Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1689286/full)
