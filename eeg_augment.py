"""
EEG-specific augmentation.

With 12 calibration trials, the model sees 12 points and has ~15k adapter
parameters -- the random-init LoRA arm in the original experiment did not
just fail to improve with more steps, it got *worse* (0.509 -> 0.485),
which is textbook few-shot overfitting. Augmentation is the direct
countermeasure, and unlike most of the other levers it costs no extra data.

Each transform below preserves the label by construction, which is the bar
an EEG augmentation has to clear:

- `time_shift`   : circular shift. Motor-imagery ERD is sustained over the
                   analysis window, so its class identity does not depend on
                   the exact onset sample; shifting also stops the model from
                   memorizing a fixed temporal template.
- `channel_drop` : zero a random subset of electrodes. Directly simulates the
                   bad/loose electrode that ruins a real session, and forces
                   the spatial filters not to depend on any single channel.
- `amplitude_scale`: per-channel gain jitter -- impedance drift, in one line.
- `gaussian_noise` : lowers effective SNR; the standard regularizer.
- `freq_mask`    : zero a random narrow band (SpecAugment for EEG). Must be
                   used gently: masking the whole mu band deletes the label.
- `mixup`        : convex combinations of trials with mixed soft labels. On
                   tiny calibration sets this is the strongest single
                   regularizer here, because it manufactures interpolated
                   examples instead of just perturbing the 12 you have.

`augment_batch` is the composed default; `mixup_batch` is applied separately
because it also changes the targets.
"""
import torch


def time_shift(x, max_frac=0.25, generator=None):
    T = x.shape[-1]
    max_s = max(1, int(T * max_frac))
    shifts = torch.randint(-max_s, max_s + 1, (x.shape[0],), generator=generator)
    return torch.stack([torch.roll(xi, int(s), dims=-1) for xi, s in zip(x, shifts)])


def channel_drop(x, p=0.1, generator=None):
    mask = (torch.rand(x.shape[0], x.shape[1], 1, generator=generator) > p).to(x.dtype)
    return x * mask


def amplitude_scale(x, sigma=0.15, generator=None):
    g = 1.0 + sigma * torch.randn(x.shape[0], x.shape[1], 1, generator=generator)
    return x * g.to(x.dtype)


def gaussian_noise(x, sigma=0.10, generator=None):
    return x + sigma * torch.randn(x.shape, generator=generator).to(x.dtype)


def freq_mask(x, max_width=4, fs=250.0, generator=None):
    """Zero a random narrow band. Width is capped so the mu/beta rhythm
    carrying the label survives."""
    n = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs)
    lo = float(torch.rand(1, generator=generator)) * max(1.0, float(freqs[-1]) - max_width)
    band = (freqs >= lo) & (freqs < lo + max_width)
    spec[..., band] = 0
    return torch.fft.irfft(spec, n=n, dim=-1).to(x.dtype)


def augment_batch(x, strength=1.0, fs=250.0, generator=None):
    if strength <= 0:
        return x
    x = time_shift(x, 0.25 * strength, generator)
    x = channel_drop(x, 0.10 * strength, generator)
    x = amplitude_scale(x, 0.15 * strength, generator)
    x = gaussian_noise(x, 0.10 * strength, generator)
    if float(torch.rand(1, generator=generator)) < 0.3 * strength:
        x = freq_mask(x, 4.0, fs, generator)
    return x


def mixup_batch(x, y, n_classes, alpha=0.4, generator=None):
    """Returns (x_mixed, soft_targets). Cross-entropy against soft targets:
    `-(soft * log_softmax(logits)).sum(-1).mean()`."""
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, n_classes).float()
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.shape[0], generator=generator)
    y1 = torch.nn.functional.one_hot(y, n_classes).float()
    return lam * x + (1 - lam) * x[perm], lam * y1 + (1 - lam) * y1[perm]


def soft_cross_entropy(logits, soft_targets, label_smoothing=0.0):
    if label_smoothing > 0:
        k = soft_targets.shape[-1]
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / k
    return -(soft_targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
