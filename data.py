"""
Synthetic multi-subject / multi-session EEG generator.

Why synthetic: this project's point is the *calibration* method (LoRA +
meta-learning), not sourcing a dataset. A real download (BCI Competition IV
2a, PhysioNet MI, etc. via MOABB/MNE) needs network access to fetch ~100s of
MB, which this sandbox couldn't do reliably (see README). So instead this
generates data with the same qualitative structure that motivates the
project:

  - A shared, class-discriminative latent signal (like a mu-rhythm ERD/ERS
    lateralization pattern for left- vs right-hand motor imagery).
  - A per-subject linear mixing matrix (like different head geometry /
    electrode placement / impedance -- this is what makes a foundation
    model trained on subject A transfer poorly, zero-shot, to subject B).
  - A per-session perturbation on top of the subject mixing (like day-to-day
    nonstationarity: electrode drift, skin conductivity, mental state).
  - Colored (1/f-ish) background noise, which is what real EEG looks like
    far more than white noise.

Swap-in path to real data: replace `make_subject_trials()` with a loader
that reads BCI IV 2a / PhysioNet epochs (via moabb.datasets +
mne.Epochs.get_data()), z-scored per-channel, reshaped into the same
(trials, channels, time) array this file returns. Everything downstream
(patchify, backbone, LoRA, meta-learning) is agnostic to that swap.
"""
import numpy as np

C = 8          # channels
T = 128        # timepoints per trial
WIN = 32       # patch window length (samples)
N_WIN = T // WIN  # windows per channel
N_TOKENS = C * N_WIN
N_CLASSES = 2


def _colored_noise(n_trials, channels, t, rng):
    # 1/f-ish noise: filter white noise in the frequency domain by 1/f.
    white = rng.standard_normal((n_trials, channels, t))
    freqs = np.fft.rfftfreq(t)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    shaping = 1.0 / np.sqrt(freqs)
    spec = np.fft.rfft(white, axis=-1) * shaping
    colored = np.fft.irfft(spec, n=t, axis=-1)
    colored /= colored.std(axis=-1, keepdims=True) + 1e-8
    return colored


def _latent_source(n_trials, t, labels, rng):
    """Two class-discriminative oscillatory sources (mu-rhythm-like)."""
    time = np.arange(t)
    src = np.zeros((n_trials, 2, t))
    for i, y in enumerate(labels):
        f_left = 10.0 + rng.uniform(-0.5, 0.5)   # ~mu band
        f_right = 10.0 + rng.uniform(-0.5, 0.5)
        amp_left = 1.8 if y == 0 else 0.25   # ERD/ERS lateralization
        amp_right = 0.25 if y == 0 else 1.8
        phase = rng.uniform(0, 2 * np.pi, size=2)
        src[i, 0] = amp_left * np.sin(2 * np.pi * f_left * time / t + phase[0])
        src[i, 1] = amp_right * np.sin(2 * np.pi * f_right * time / t + phase[1])
    return src


def make_subject(rng, n_trials=40, class_balance=0.5, session_shift_scale=0.15):
    """Generate one synthetic subject's trials + labels.

    Returns X: (n_trials, C, T) float64, y: (n_trials,) int in {0,1}.
    A fresh per-subject mixing matrix + per-subject noise gain are drawn so
    that different subjects look meaningfully different to a fixed model,
    which is exactly the calibration problem this project addresses.
    """
    labels = (rng.uniform(size=n_trials) > class_balance).astype(int)
    latent = _latent_source(n_trials, T, labels, rng)  # (n_trials, 2, T)

    # Subject-specific mixing: maps 2 latent sources -> C channels.
    subj_mix = rng.standard_normal((C, 2)) * 0.8
    # Subject-specific per-channel gain (impedance-like) and noise level.
    subj_gain = rng.uniform(0.6, 1.4, size=(C, 1))
    subj_noise_level = rng.uniform(0.2, 0.45)

    # Session drift: small extra random rotation-like perturbation applied
    # per-trial-block to emulate within-subject, cross-session nonstationarity.
    session_perturb = rng.standard_normal((C, 2)) * session_shift_scale

    mixed = np.einsum("cs,nst->nct", subj_mix + session_perturb, latent)
    mixed = mixed * subj_gain[None, :, :]
    noise = _colored_noise(n_trials, C, T, rng) * subj_noise_level
    X = mixed + noise

    # z-score per channel per trial (standard EEG preprocessing)
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
    return X, labels


def patchify(X):
    """(n_trials, C, T) -> (n_trials, N_TOKENS, WIN) channel-window patches,
    the same tokenization style LaBraM/BENDR-style EEG foundation models use
    (each token = one channel's short time window)."""
    n = X.shape[0]
    X = X.reshape(n, C, N_WIN, WIN)          # split time into windows
    X = X.transpose(0, 1, 2, 3)               # (n, C, N_WIN, WIN)
    X = X.reshape(n, C * N_WIN, WIN)          # (n, tokens, WIN)
    return X


def channel_and_window_ids():
    """Token metadata used for the backbone's channel/position embeddings."""
    ch_ids = np.repeat(np.arange(C), N_WIN)
    win_ids = np.tile(np.arange(N_WIN), C)
    return ch_ids, win_ids


def make_population(n_subjects, seed, n_trials=40):
    """A list of (X_patches, y) per subject, for pretraining / meta-training
    across many distinct synthetic subjects."""
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_subjects):
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        X, y = make_subject(sub_rng, n_trials=n_trials)
        pop.append((patchify(X), y))
    return pop
