"""
Real-data loading + preprocessing for the PyTorch pipeline.

Primary source is BCI Competition IV 2a (MOABB `BNCI2014_001`): 9 subjects,
2 sessions recorded on *different days*, 4-class motor imagery, 22 EEG
channels @ 250 Hz. A synthetic source with the identical shape/interface is
included so the whole pipeline can be executed (and regression-tested)
without network access -- it is a *test fixture*, never a result.

What changed vs. the first draft of this file, and why each change matters
for accuracy (not just tidiness):

1. **Per-trial standardization.** MOABB returns volts (~1e-5). Feeding that
   straight into a randomly-initialized linear layer produces activations
   ~1e-5, gradients ~1e-10, and a model that learns essentially nothing in
   the few hundred steps a calibration budget allows. Every trial is now
   z-scored per channel. This was the single largest silent accuracy bug.

2. **Euclidean Alignment (He & Wu, IEEE TBME 2020).** For each recording
   session, compute the mean spatial covariance R = mean_i (X_i X_i^T / T)
   and whiten every trial by R^{-1/2}. This maps every session's data into a
   common reference frame where the mean covariance is the identity, which
   removes most of the between-subject and between-session covariance shift
   that the calibration step would otherwise have to spend its 12 trials
   undoing. It is *label-free*, so it does not leak. Reported gains in the
   literature for cross-subject MI are ~5-15 points; it is the cheapest real
   win available and it composes with everything else here.

   Two modes, because the honest answer depends on deployment assumptions:
     - `align_mode="session"` (default): each session is whitened by its own
       R, estimated from that session's *unlabeled* trials. This is
       transductive-unsupervised -- legitimate and standard in the EA
       literature, and realistic (you record ~1 min of unlabeled EEG at the
       start of a session), but it does see the query session's inputs.
     - `align_mode="calib"`: the query session is whitened with the
       *calibration* session's R. Fully inductive, strictly weaker, and
       reported alongside so nobody has to take the transductive number on
       faith.

3. **Stratified few-shot calibration sampling.** A 12-trial budget drawn
   uniformly at random from a 4-class pool routinely yields 0 examples of
   some class -- after which that class can never be predicted and the
   accuracy ceiling for the run is 75%. Sampling is now stratified
   (3 per class for a 12-trial budget), which is also what a real
   calibration protocol does: you cue each class equally.

4. **Optional filter bank.** Motor imagery is a mu (8-13 Hz) / beta
   (13-30 Hz) phenomenon; a single 4-38 Hz band mixes discriminative rhythm
   power with non-discriminative low-frequency drift. `bands=[(4,8),(8,13),
   (13,20),(20,30),(30,38)]` splits the signal into sub-bands stacked as
   extra channels (FBCSP's core insight, kept in neural form).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EVENTS = ["left_hand", "right_hand", "feet", "tongue"]
LABEL_MAP = {name: i for i, name in enumerate(EVENTS)}
N_CLASSES = len(EVENTS)

DEFAULT_BANDS = [(4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 38.0)]


# --------------------------------------------------------------------------
# signal preprocessing (numpy, no scipy dependency)
# --------------------------------------------------------------------------
def bandpass_fft(X, fs, lo, hi):
    """Zero-phase brick-wall bandpass via rFFT. Fine for offline epochs;
    for online use swap in a causal IIR filter (scipy.signal.sosfilt)."""
    n = X.shape[-1]
    spec = np.fft.rfft(X, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    keep = (freqs >= lo) & (freqs <= hi)
    spec = spec * keep
    return np.fft.irfft(spec, n=n, axis=-1)


def make_filter_bank(X, fs, bands):
    """(n, C, T) -> (n, C*len(bands), T), sub-bands stacked along channels."""
    return np.concatenate([bandpass_fft(X, fs, lo, hi) for lo, hi in bands], axis=1)


def zscore_per_trial(X, eps=1e-6):
    """Per-trial, per-channel standardization. See docstring point 1."""
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return (X - mu) / (sd + eps)


def _inv_sqrt_spd(R, eps=1e-8):
    """R^{-1/2} for a symmetric positive-definite matrix, via eigh."""
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, eps * float(w.max()) if w.max() > 0 else eps)
    return (V * (w ** -0.5)) @ V.T


def reference_covariance(X, eps=1e-8):
    """Mean spatial covariance across trials: R = mean_i (X_i X_i^T / T).
    Label-free, so computing it on unlabeled query data leaks nothing."""
    T = X.shape[-1]
    covs = np.einsum("nct,ndt->ncd", X, X) / T
    R = covs.mean(axis=0)
    R += eps * np.trace(R) / R.shape[0] * np.eye(R.shape[0])  # ridge, for stability
    return R


def euclidean_align(X, R):
    """Whiten every trial by R^{-1/2}: after this the set's mean covariance
    is the identity, so different subjects/sessions become comparable."""
    return np.einsum("cd,ndt->nct", _inv_sqrt_spd(R), X)


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------
class EEGTrials(Dataset):
    """(trials, channels, time) float32 + int64 labels for one session."""

    def __init__(self, X, y):
        self.X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def tensors(self):
        return self.X, self.y


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def load_moabb(subject_ids, fmin=4.0, fmax=38.0, resample=250.0):
    """Real BNCI2014_001. Requires `pip install moabb` and network access on
    first call (raw files are a few hundred MB)."""
    import moabb_path_fix
    moabb_path_fix.apply()   # MOABB 1.5.0 mangles the Windows drive letter and
                             # downloads relative to cwd; see that module.

    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    # n_classes is required alongside `events`: MOABB only infers it when
    # `events is None`, otherwise it stays None and used_events() raises
    # `TypeError: '<' not supported between 'int' and 'NoneType'`.
    paradigm = MotorImagery(n_classes=len(EVENTS), fmin=fmin, fmax=fmax,
                            events=EVENTS, resample=resample)
    X, y, meta = paradigm.get_data(dataset=BNCI2014_001(), subjects=list(subject_ids))
    out = {}
    for sid in subject_ids:
        m = meta["subject"] == sid
        out[sid] = {}
        for sess in sorted(meta.loc[m, "session"].unique()):
            sel = (m & (meta["session"] == sess)).values
            out[sid][str(sess)] = (X[sel].astype(np.float64),
                                   np.array([LABEL_MAP[str(v)] for v in y[sel]]))
    return out, float(resample)


def load_synthetic(subject_ids, n_channels=22, fs=250.0, dur_s=2.0,
                   trials_per_session=144, seed=7, erd_contrast=0.30,
                   noise_range=(1.6, 3.2), n_distractors=3):
    """A BNCI2014_001-shaped *fixture*: 4 classes, 2 sessions per subject on
    'different days', per-subject spatial mixing (head geometry / montage),
    per-session perturbation (electrode drift), 1/f background noise, and a
    class-discriminative mu/beta ERD pattern. Same shapes and same dict
    layout as `load_moabb`, so every downstream line of code is identical.

    This exists so the pipeline is runnable and regression-testable in an
    offline environment. Numbers produced from it describe the code, not
    the brain -- see FINE_TUNING_PLAYBOOK.md.
    """
    T = int(fs * dur_s)
    t = np.arange(T) / fs
    rng = np.random.default_rng(seed)

    # Four class-linked sources plus non-discriminative distractor sources.
    # The fixture is deliberately tuned so that *within*-subject decoding is
    # hard-but-possible (~0.6-0.8) and *cross*-subject zero-shot decoding is
    # near chance, which is the regime real BCI IV 2a sits in. A fixture that
    # is trivially separable would make every ablation below look identical
    # and prove nothing.
    n_src = 4
    src_freqs = [10.0, 11.5, 21.0, 9.0]

    out = {}
    for sid in subject_ids:
        srng = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
        mixing = srng.standard_normal((n_channels, n_src)) * 0.9
        dist_mix = srng.standard_normal((n_channels, n_distractors)) * 1.1
        dist_freqs = srng.uniform(5.0, 35.0, size=n_distractors)
        gain = srng.uniform(0.5, 1.6, size=(n_channels, 1))
        noise_lvl = srng.uniform(*noise_range)
        out[sid] = {}
        for s_i, sess in enumerate(["0train", "1test"]):
            erng = np.random.default_rng(srng.integers(0, 2 ** 31 - 1))
            drift = erng.standard_normal((n_channels, n_src)) * (0.10 + 0.10 * s_i)
            y = np.tile(np.arange(N_CLASSES), trials_per_session // N_CLASSES)
            erng.shuffle(y)
            lat = np.zeros((len(y), n_src, T))
            for i, cls in enumerate(y):
                for s in range(n_src):
                    # ERD: the source matching the class is suppressed, the
                    # others are not -- the actual physiology of motor imagery.
                    # ERD: the class-matched source is *suppressed*. The
                    # contrast is small on purpose -- real mu-ERD is a
                    # modest percentage power change, not an on/off switch.
                    amp = (1.0 - erd_contrast) if s == cls else 1.0
                    amp *= erng.uniform(0.75, 1.25)   # per-trial variability
                    ph = erng.uniform(0, 2 * np.pi)
                    env = 1.0 + 0.3 * np.sin(2 * np.pi * 0.7 * t + erng.uniform(0, 6.28))
                    lat[i, s] = amp * env * np.sin(2 * np.pi * src_freqs[s] * t + ph)
            # non-discriminative but high-variance activity (alpha bursts,
            # EMG-ish drift) -- the stuff a decoder must learn to ignore
            dist = np.zeros((len(y), n_distractors, T))
            for s in range(n_distractors):
                for i in range(len(y)):
                    dist[i, s] = erng.uniform(0.7, 1.8) * np.sin(
                        2 * np.pi * dist_freqs[s] * t + erng.uniform(0, 6.28))
            X = (np.einsum("cs,nst->nct", mixing + drift, lat)
                 + np.einsum("cs,nst->nct", dist_mix, dist)) * gain[None]
            white = erng.standard_normal((len(y), n_channels, T))
            f = np.fft.rfftfreq(T, d=1.0 / fs)
            f[0] = f[1]
            pink = np.fft.irfft(np.fft.rfft(white, axis=-1) / np.sqrt(f), n=T, axis=-1)
            pink /= pink.std(axis=-1, keepdims=True) + 1e-9
            out[sid][sess] = (X + noise_lvl * pink, y)
    return out, fs


# --------------------------------------------------------------------------
# main loader
# --------------------------------------------------------------------------
class MetaEEGDataLoader:
    """Per-subject, per-session trial pools with the preprocessing stack
    applied, plus stratified few-shot calibration sampling."""

    def __init__(self, subject_ids=None, source="auto", align_mode="session",
                 bands=None, fmin=4.0, fmax=38.0, resample=250.0, verbose=True):
        self.subject_ids = list(subject_ids) if subject_ids else list(range(1, 10))
        self.align_mode = align_mode
        self.bands = bands

        source = self._resolve_source(source)
        self.source = source
        if verbose:
            print(f"[data] source={source} subjects={self.subject_ids} "
                  f"align={align_mode} bands={'none' if not bands else len(bands)}")

        if source == "moabb":
            raw, fs = load_moabb(self.subject_ids, fmin=fmin, fmax=fmax, resample=resample)
        else:
            raw, fs = load_synthetic(self.subject_ids)
        self.fs = fs

        self.store = {}
        for sid, sessions in raw.items():
            self.store[sid] = {}
            calib_R = None
            for s_i, sess in enumerate(sorted(sessions)):
                X, y = sessions[sess]
                if self.bands:
                    X = make_filter_bank(X, fs, self.bands)
                X = zscore_per_trial(X)
                R = reference_covariance(X)
                if s_i == 0:
                    calib_R = R
                use_R = R if align_mode == "session" else calib_R
                if align_mode != "none":
                    X = euclidean_align(X, use_R)
                    X = zscore_per_trial(X)  # re-standardize after whitening
                self.store[sid][sess] = EEGTrials(X, y)

        any_sid = self.subject_ids[0]
        any_sess = sorted(self.store[any_sid])[0]
        self.n_channels = self.store[any_sid][any_sess].X.shape[1]
        self.n_timepoints = self.store[any_sid][any_sess].X.shape[2]
        if verbose:
            print(f"[data] {self.n_channels} channels x {self.n_timepoints} timepoints, fs={fs}")

    @staticmethod
    def _resolve_source(source):
        if source in ("moabb", "synthetic"):
            return source
        if os.environ.get("EEG_FORCE_SYNTHETIC"):
            return "synthetic"
        try:
            import moabb  # noqa: F401
            return "moabb"
        except Exception:
            print("[data] moabb unavailable -> synthetic fixture. "
                  "`pip install moabb` + network for the real experiment.")
            return "synthetic"

    def sessions(self, subject_id):
        return sorted(self.store[subject_id])

    def stratified_indices(self, y, n_total, seed):
        """Equal trials per class (see docstring point 3)."""
        rng = np.random.default_rng(seed)
        y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
        classes = np.unique(y)
        per = max(1, n_total // len(classes))
        idx = []
        for c in classes:
            pool = np.where(y == c)[0]
            idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False))
        rng.shuffle(idx)
        return np.array(idx[:n_total])

    def calibration_and_query(self, subject_id, calib_size=12, seed=0):
        """The realistic split: calibrate on session 1 (one day), evaluate on
        session 2 (a different day). Returns raw tensors, not DataLoaders --
        few-shot calibration is full-batch by construction."""
        sess = self.sessions(subject_id)
        if len(sess) < 2:
            raise ValueError(f"subject {subject_id} has sessions {sess}; need >= 2")
        calib, query = self.store[subject_id][sess[0]], self.store[subject_id][sess[-1]]
        idx = self.stratified_indices(calib.y, calib_size, seed)
        return (calib.X[idx], calib.y[idx]), (query.X, query.y)

    def pooled_session(self, subject_ids, which=0):
        """All trials from one session index, pooled over subjects -- used for
        self-supervised pretraining (never touches the query session)."""
        Xs, ys = [], []
        for sid in subject_ids:
            s = self.sessions(sid)[which]
            Xs.append(self.store[sid][s].X)
            ys.append(self.store[sid][s].y)
        return torch.cat(Xs), torch.cat(ys)
