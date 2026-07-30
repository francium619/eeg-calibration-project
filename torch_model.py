"""
PyTorch port of model.py: same architecture (patchify -> patch embed ->
channel/window embeddings -> 1 self-attention block -> 1 FFN block ->
attention-pooling head), same LoRA-on-every-linear-layer pattern, now as
real nn.Modules so it can run on GPU with real data via torch_data.py.

Sized up from the NumPy prototype (D_MODEL 16->64, LORA_RANK 4->8) since
this targets real hardware, not a 45-second sandbox budget -- tune these
for whatever compute you actually have.

Not executed in this sandbox (no PyTorch available here, see README) --
ported carefully from the NumPy version, which *is* gradient-checked and
tested (test_tensor.py, test_model.py), but treat this file as reviewed-not-run
and re-verify (e.g. with a gradient check against torch.autograd.gradcheck,
or just a quick overfit-one-batch sanity test) before trusting results.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
FFN_HIDDEN = 128
LORA_RANK = 8
LORA_ALPHA = 4 * LORA_RANK


class LoRALinear(nn.Module):
    """y = x @ W_frozen (+b) + scaling * (x @ A) @ B.

    W/b are frozen after `freeze_base()`; A/B are what gets meta-learned /
    per-subject calibrated. As in the NumPy version, B is small-random
    (not the conventional zero) init, trading the "LoRA starts as an exact
    no-op" property for a nonzero gradient on A from step 1 -- this matters
    when the adaptation budget is a handful of steps, and it's the LoRA
    init that gets overwritten by the meta-learned init anyway.
    """

    def __init__(self, d_in, d_out, bias=True, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.05)
        self.B = nn.Parameter(torch.randn(r, d_out) * 0.01)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def lora_parameters(self):
        return [self.A, self.B]

    def reset_lora(self, generator=None):
        with torch.no_grad():
            self.A.copy_(torch.randn(*self.A.shape, generator=generator) * 0.05)
            self.B.copy_(torch.randn(*self.B.shape, generator=generator) * 0.01)

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A) @ self.B


class EEGBackbone(nn.Module):
    def __init__(self, n_channels, win_len, n_windows, d_model=D_MODEL, ffn_hidden=FFN_HIDDEN):
        super().__init__()
        self.n_channels, self.win_len, self.n_windows = n_channels, win_len, n_windows
        self.n_tokens = n_channels * n_windows
        d = d_model

        self.patch_embed = LoRALinear(win_len, d)
        self.ch_embed = nn.Embedding(n_channels, d)
        self.win_embed = nn.Embedding(n_windows, d)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.ln1 = nn.LayerNorm(d)
        self.Wq = LoRALinear(d, d, bias=False)
        self.Wk = LoRALinear(d, d, bias=False)
        self.Wv = LoRALinear(d, d, bias=False)
        self.Wo = LoRALinear(d, d, bias=False)

        self.ln2 = nn.LayerNorm(d)
        self.W1 = LoRALinear(d, ffn_hidden)
        self.W2 = LoRALinear(ffn_hidden, d)

        self.decoder_head = LoRALinear(d, win_len)  # pretraining-only

        ch_ids = torch.arange(n_channels).repeat_interleave(n_windows)
        win_ids = torch.arange(n_windows).repeat(n_channels)
        self.register_buffer("ch_ids", ch_ids)
        self.register_buffer("win_ids", win_ids)

        self._lora_layers = [self.patch_embed, self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2]

    # ---- patchify raw (B, C, T) trials into (B, tokens, win_len) ----
    def patchify(self, x):
        B, C, T = x.shape
        assert C == self.n_channels, f"expected {self.n_channels} channels, got {C}"
        needed = self.n_windows * self.win_len
        assert T >= needed, f"trial has only {T} timepoints, need >= {needed} ({self.n_windows}x{self.win_len})"
        if T > needed:
            x = x[:, :, :needed]  # drop trailing samples that don't fill a full window
        x = x.reshape(B, C, self.n_windows, self.win_len)
        return x.reshape(B, C * self.n_windows, self.win_len)

    def freeze_backbone_base(self):
        for lyr in self._lora_layers:
            lyr.freeze_base()
        for p in [self.ch_embed.weight, self.win_embed.weight, self.mask_token]:
            p.requires_grad = False
        for p in list(self.ln1.parameters()) + list(self.ln2.parameters()):
            p.requires_grad = False

    def lora_parameters(self):
        ps = []
        for lyr in self._lora_layers:
            ps += lyr.lora_parameters()
        return ps

    def backbone_base_parameters(self):
        ps = []
        for lyr in self._lora_layers:
            ps += list(lyr.base.parameters())
        ps += [self.ch_embed.weight, self.win_embed.weight, self.mask_token]
        ps += list(self.ln1.parameters()) + list(self.ln2.parameters())
        return ps

    def encode(self, patches, mask_idx=None):
        """patches: (B, N_TOKENS, WIN). Returns (pooled_mean (B,D) -- unused
        by the attention-pooling head but handy for probing, tok (B,N,D))."""
        tok = self.patch_embed(patches) + self.ch_embed(self.ch_ids) + self.win_embed(self.win_ids)

        if mask_idx is not None:
            keep = torch.ones(1, tok.shape[1], 1, device=tok.device)
            keep[0, mask_idx, 0] = 0.0
            tok = tok * keep + self.mask_token * (1.0 - keep)

        h = self.ln1(tok)
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        attn = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        tok = tok + self.Wo(attn @ v)

        h = self.ln2(tok)
        tok = tok + self.W2(F.gelu(self.W1(h)))

        return tok.mean(dim=1), tok


def lora_state_dict(backbone):
    """Only the A/B pairs -- NOT the frozen base weights. Saving/loading the
    full backbone.state_dict() as "the meta-learned init" is a real footgun:
    it silently also carries the base weights around, so if you ever load a
    'meta init' checkpoint onto a backbone with *different* pretrained base
    weights, you won't get an error, just quietly wrong base weights."""
    sd = {}
    for i, lyr in enumerate(backbone._lora_layers):
        sd[f"lora_A_{i}"] = lyr.A.detach().clone()
        sd[f"lora_B_{i}"] = lyr.B.detach().clone()
    return sd


def load_lora_state_dict(backbone, sd):
    with torch.no_grad():
        for i, lyr in enumerate(backbone._lora_layers):
            lyr.A.copy_(sd[f"lora_A_{i}"])
            lyr.B.copy_(sd[f"lora_B_{i}"])


class ClassifierHead(nn.Module):
    """Attention-pooling readout: a learnable query attends over token
    representations; the pooled feature is classified. Matches model.py's
    ClassifierHead -- pooling is task-adaptive and part of what gets
    meta-learned/calibrated, not a fixed mean."""

    def __init__(self, n_classes, d_model=D_MODEL):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.1)
        self.linear = nn.Linear(d_model, n_classes)

    def pool(self, tok):
        scores = (tok @ self.query) / math.sqrt(tok.shape[-1])  # (B, N)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)       # (B, N, 1)
        return (tok * weights).sum(dim=1)                        # (B, D)

    def forward(self, tok):
        return self.linear(self.pool(tok))

    def parameters_for_calibration(self):
        return [self.query] + list(self.linear.parameters())
