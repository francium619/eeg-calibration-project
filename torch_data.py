"""
Real-data loading for the PyTorch port: BCI Competition IV 2a (BNCI2014_001)
via MOABB, wrapped for the Reptile meta-learning loop.

This builds on the loader sketch you provided, with two fixes that matter
for the actual experiment:

1. **Session-based support/query split, not a random split.** BNCI2014_001
   records each subject on two separate days ("0train" and "1test" session
   labels in MOABB's metadata) -- that day-to-day drift is *exactly* the
   nonstationarity this whole project is about. A `random_split` over the
   pooled two-session data lets support and query trials land in the same
   session, which leaks session-specific state between calibration and
   evaluation and would make calibration look better than it will in
   deployment (calibrate once, use across a different day). We split by
   `metadata['session']` instead: the earlier session is the calibration
   pool, the later session is always the query/evaluation pool.
2. **A single, fixed label ordering shared across all subjects.** Building
   `label_mapping` independently per subject (as in the original sketch) is
   almost certainly fine here since every subject has the same 4 classes,
   but it's a silent footgun if that ever stops being true (e.g. a subject
   missing a class). We fix the mapping once from `paradigm.events` instead.

Install: pip install moabb torch  (moabb pulls mne + the dataset's raw
files on first use -- the BNCI2014_001 download is a few hundred MB per
subject and was not something this development sandbox could fetch, so
this file is written carefully but not executed here; see README).
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

EVENTS = ["left_hand", "right_hand", "feet", "tongue"]
LABEL_MAP = {name: i for i, name in enumerate(EVENTS)}  # fixed, not per-subject


class MoabbSubjectDataset(Dataset):
    """(trials, channels, time) + integer labels for one subject/session slice."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array([LABEL_MAP[label] for label in y]), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MetaEEGDataLoader:
    """Fetches BNCI2014_001 via MOABB and exposes per-subject, per-session
    trial pools for episodic (Reptile) meta-training and calibration eval."""

    def __init__(self, subject_ids=None, resample_fs=250, fmin=4, fmax=38):
        if subject_ids is None:
            subject_ids = list(range(1, 10))  # BNCI2014_001 has 9 subjects
        self.subject_ids = subject_ids
        self.dataset = BNCI2014_001()
        self.paradigm = MotorImagery(fmin=fmin, fmax=fmax, events=EVENTS, resample=resample_fs)

        # subject_id -> {session_name: MoabbSubjectDataset}
        self.subject_session_datasets = {}
        self.n_channels = None
        self.n_timepoints = None
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        print(f"Fetching and epoching MOABB data for subjects: {self.subject_ids} "
              f"(first call downloads raw files -- can take a while / a few hundred MB per subject)...")
        X, y, metadata = self.paradigm.get_data(dataset=self.dataset, subjects=self.subject_ids)
        self.n_channels, self.n_timepoints = X.shape[1], X.shape[2]

        for sub_id in self.subject_ids:
            sub_mask = (metadata["subject"] == sub_id)
            sessions = sorted(metadata.loc[sub_mask, "session"].unique())
            self.subject_session_datasets[sub_id] = {}
            for sess in sessions:
                mask = sub_mask & (metadata["session"] == sess)
                self.subject_session_datasets[sub_id][sess] = MoabbSubjectDataset(X[mask], y[mask])
        print(f"Done. {self.n_channels} channels x {self.n_timepoints} timepoints per trial. "
              f"Sessions per subject: { {s: list(d.keys()) for s, d in self.subject_session_datasets.items()} }")

    def get_subject_session_split_loaders(self, subject_id, calib_size=None, batch_size=16, seed=0):
        """The realistic split: calibrate on the earlier session, evaluate on
        the later session (a different day). If `calib_size` is given, only
        that many trials (the realistic "few-shot calibration budget") are
        drawn from the calibration session; the rest of that session is
        discarded rather than leaked into query."""
        sessions = sorted(self.subject_session_datasets[subject_id].keys())
        if len(sessions) < 2:
            raise ValueError(
                f"Subject {subject_id} only has session(s) {sessions}; need >=2 for a "
                f"cross-session calibration/query split. Fall back to "
                f"get_subject_random_split_loaders if you specifically want a within-session split."
            )
        calib_ds = self.subject_session_datasets[subject_id][sessions[0]]
        query_ds = self.subject_session_datasets[subject_id][sessions[-1]]

        if calib_size is not None and calib_size < len(calib_ds):
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(calib_ds), generator=g)[:calib_size]
            calib_ds = torch.utils.data.Subset(calib_ds, idx.tolist())

        calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=True)
        query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False)
        return calib_loader, query_loader

    def get_subject_random_split_loaders(self, subject_id, support_ratio=0.2, batch_size=16, seed=42):
        """Within-session random split -- kept for parity with the original
        sketch / for datasets that only have one session, but note this
        does NOT test cross-session generalization, only within-session."""
        pooled_X, pooled_y = [], []
        for sess_ds in self.subject_session_datasets[subject_id].values():
            pooled_X.append(sess_ds.X)
            pooled_y.append(sess_ds.y)
        full = torch.utils.data.TensorDataset(torch.cat(pooled_X), torch.cat(pooled_y))

        total_len = len(full)
        support_len = int(total_len * support_ratio)
        query_len = total_len - support_len
        support_set, query_set = torch.utils.data.random_split(
            full, [support_len, query_len], generator=torch.Generator().manual_seed(seed)
        )
        support_loader = DataLoader(support_set, batch_size=batch_size, shuffle=True)
        query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False)
        return support_loader, query_loader
