# Fine-tuning on raw real EEG: what to do, in priority order

Written as a work order, not a literature review. Every item states what to
change, why it should move accuracy, and roughly how much to expect. Numbers
in "expected" are from published cross-subject motor-imagery work on BCI
Competition IV 2a unless noted; treat them as direction and magnitude, not
promises.

## First, the honest ceiling

Four-class motor imagery on BCI IV 2a does not go to 100%, and any pipeline
that reports it has a leak. For calibration:

| regime | typical accuracy |
|---|---|
| chance | 25% |
| cross-subject **zero-shot** (no calibration) | 35-50% |
| cross-subject + **few-shot calibration** (this project's target) | 55-70% |
| **within-subject**, full session of labeled data (FBCSP / EEGNet / Conformer) | 68-80% |
| best published within-subject ensembles | ~84-88% |
| BCI illiteracy floor: 15-30% of people never clear ~60% on MI at all | — |

So "perfect accuracy" is the wrong target. The right target is **beating the
zero-shot baseline by a significant margin at a small calibration budget**,
and the harness in `run_ablation.py` now reports exactly that with paired
confidence intervals. A method that raises mean accuracy 2 points on 2
subjects has not beaten anything — the standard error on 2 subjects is
roughly +/- 8 points.

## Step 0 — get the real data flowing (half a day)

```bash
pip install torch moabb mne
python3 torch_pretrain.py       --subjects 1 2 3 4 5 6 7 --source moabb
python3 torch_meta_train.py     --subjects 1 2 3 4 5 6 7 --source moabb
python3 torch_eval_calibration.py --eval-subjects 8 9    --source moabb
```

First MOABB call downloads BNCI2014_001 (a few hundred MB). Everything in
this repo already runs end-to-end; `--source moabb` swaps the synthetic
fixture for real recordings and changes nothing else.

**Acceptance check before believing any number:** `zero_shot` must clear 25%
chance by a real margin, and `test_torch_pipeline.py` must be all-green
(it includes a shuffled-label control that catches leakage). If zero-shot is
at chance, the problem is backbone pretraining, not the calibration method,
and every arm will look equally bad.

## Step 1 — the cheap wins, in descending order of expected effect

These are ranked by (expected gain) / (effort). Do them in this order.

### 1.1 Euclidean Alignment — already implemented, verify on real data
`torch_data.py`, `align_mode="session"`. Whitens each session by its own mean
spatial covariance so subjects/sessions share a reference frame. Label-free,
so it does not leak; it does look at unlabeled query inputs, which is why the
strictly-inductive `align_mode="calib"` is also implemented and should be
reported alongside.
**Expected: +5 to +15 points cross-subject.** In this repo's fixture it was
worth +6.2 zero-shot. He & Wu, IEEE TBME 2020.

### 1.2 Riemannian tangent-space features as a parallel head
Project trial covariance matrices to the tangent space at the Riemannian mean
and feed those features alongside the network's. On small data, Riemannian
pipelines (`pyriemann`: `Covariances -> TangentSpace -> LogisticRegression`)
are still extremely competitive with deep nets and fail differently, so an
ensemble of the two beats either.
**Expected: +3 to +8 points, and a strong sanity baseline.** Barachant et al.

### 1.3 Filter bank input
`bands=DEFAULT_BANDS` in `MetaEEGDataLoader` splits 4-38 Hz into five
sub-bands stacked as channels (FBCSP's core idea in neural form). Mu and beta
carry motor imagery; a single wide band dilutes them with drift.
**Expected: +2 to +6 points.** Ang et al., FBCSP.

### 1.4 More calibration trials, and spend them well
The budget curve is already in the eval output. On real data expect the knee
around 20-40 trials. Keep the sampling stratified (implemented) — an
unstratified 12-trial draw regularly misses a class and caps you at 75%.

### 1.5 Test-time augmentation and crop ensembling
Average logits over shifted crops of the trial (`--tta 4`). Free at inference,
label-free.
**Expected: +1 to +3 points.**

## Step 2 — more source subjects (this is the real unlock)

BNCI2014_001 has 9 subjects. Meta-learning an initialization from 7 of them
is a demonstration, not a trained meta-learner; published meta-learning-for-
BCI work pools dozens to low hundreds of subjects. `MetaEEGDataLoader` works
with any MOABB dataset exposing `subjects` / `get_data()`.

Pooling ladder, in the order worth doing:

| dataset | subjects | classes | note |
|---|---|---|---|
| `BNCI2014_001` (IV 2a) | 9 | 4 | the evaluation set — hold out |
| `BNCI2014_004` (IV 2b) | 9 | 2 | 5 sessions each, great for cross-session |
| `PhysionetMI` | 109 | 4 | biggest single MI subject pool, 64ch |
| `Cho2017` (GigaScience) | 52 | 2 | clean, well-documented |
| `Weibo2014`, `Zhou2016`, `Shin2017A` | 10-25 | 2-4 | fill out the tail |
| `Lee2019_MI` | 54 | 2 | two sessions per subject |

Channel counts differ (22 vs 64), so either restrict to the intersecting
10-20 montage subset (C3/Cz/C4-centred is the standard choice for MI) or add a
learned channel-projection layer per dataset. The `SpatialAdapter` already in
`torch_model.py` is the natural place to absorb montage differences.

**Expected: this is the single largest remaining lever — meta-learning with
7 source subjects is data-starved, and going to 50-100 subjects is where
"meta-learned init" stops being a demo.**

## Step 3 — swap in a real EEG foundation model

The current backbone is trained from scratch on a few hundred trials. The
project's premise is adapting a *foundation* model.

- **LaBraM** — https://github.com/935963004/LaBraM, ICLR 2024 spotlight,
  pretrained on ~2,500 hours across ~20 datasets. Weights in-repo.
- **EEGPT**, **BIOT**, **Brant** — alternatives worth benchmarking.
- **BENDR** — older, wav2vec-style, small and easy to run.

Integration is mechanical: the `LoRALinear` pattern wraps any `nn.Linear`,
so you replace `EEGBackbone.encode` with the foundation model's encoder and
walk its modules substituting `nn.Linear` -> `LoRALinear` (this is precisely
what `peft` does for LLMs). Keep the `SpatialAdapter` at the input — montage
mismatch between LaBraM's pretraining montages and your recording montage is
handled there.

**Expected: +5 to +15 points zero-shot, and a much better starting point for
few-shot calibration.** Caveat: published foundation-model gains on MI
specifically are more modest than on seizure/sleep tasks, so measure, do not
assume.

## Step 4 — adaptation methods worth trying, ranked

1. **Test-time / online adaptation.** Entropy minimization or batch-norm
   statistic adaptation on the unlabeled query stream (TENT-style). Directly
   targets session drift and needs no labels. *Report separately — it is
   transductive.* Expected +2 to +6.
2. **FOMAML over Reptile** (already the default). In this repo's fixture,
   switching from Reptile to FOMAML with a working differentiable inner loop
   moved the meta-init from *worse* than zero-shot (-3.6 points) to +11.7.
   That was the largest single change in the whole project.
3. **Meta-SGD per-parameter inner learning rates** (already implemented). The
   inner LR matters as much as the initialization at 5 gradient steps.
4. **Full second-order MAML.** Now feasible — the inner loop is already
   functional/differentiable; set `create_graph=True` and stop detaching the
   parameter gradients. Expect a small gain for ~3x cost.
5. **Adapter rank sweep.** r in {2,4,8,16}. Lower rank is a stronger prior
   and usually wins at 12 trials.
6. **Prototypical / metric-based calibration** instead of gradient steps:
   compute class prototypes in feature space from the 12 calibration trials
   and classify by distance. Zero gradient steps, no overfitting, often
   startlingly strong in the very-low-shot regime. Worth having as an arm.
7. **Adversarial / domain-invariant subject alignment** (DANN, CORAL, MMD) on
   the pretraining stage so the frozen features are subject-agnostic before
   the adapter ever runs.

## Step 5 — evaluation protocol you must hold to

- **Leave-one-subject-out over all 9 subjects.** A fixed 2-subject test set on
  9 subjects is not an evaluation. `run_ablation.py` lists the LOSO rung.
- **Cross-session always.** Calibrate day 1, test day 2.
- **Multiple seeds**, paired CIs on the difference (implemented).
- **Never select checkpoints or hyperparameters on the held-out subjects.**
  `torch_pretrain.py` early-stops on a *source*-subject validation split for
  this reason.
- **Report the shuffled-label control.** It is one line and it is what
  distinguishes a result from a bug.

## What to do first, if you only do three things

1. Run the existing pipeline on real BNCI2014_001 and confirm the acceptance
   checks (Step 0).
2. Pool `PhysionetMI` + `Cho2017` into meta-training to get from 7 source
   subjects to ~100 (Step 2).
3. Swap the from-scratch backbone for LaBraM (Step 3).

Steps 1.1-1.5 are already coded and can be ablated in minutes with
`run_ablation.py`.

## References

- He & Wu, "Transfer Learning for Brain-Computer Interfaces: A Euclidean
  Space Data Alignment Approach", IEEE TBME 2020.
- Schirrmeister et al., "Deep learning with convolutional neural networks for
  EEG decoding and visualization" (ShallowFBCSPNet), HBM 2017.
- Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
  EEG-based Brain-Computer Interfaces", J. Neural Eng. 2018.
- Ang et al., "Filter Bank Common Spatial Pattern (FBCSP) in BCI
  Competition IV Datasets 2a and 2b", Front. Neurosci. 2012.
- Barachant et al., "Multiclass Brain-Computer Interface Classification by
  Riemannian Geometry", IEEE TBME 2012.
- Jiang et al., "Large Brain Model for Learning Generic Representations with
  Tremendous EEG Data in BCI" (LaBraM), ICLR 2024 — arXiv:2405.18765.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  arXiv:2106.09685.
- Finn et al., "Model-Agnostic Meta-Learning" (MAML), ICML 2017;
  Nichol et al., "On First-Order Meta-Learning Algorithms" (Reptile).
- Li et al., "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning",
  arXiv:1707.09835.
- Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization",
  ICLR 2021.
