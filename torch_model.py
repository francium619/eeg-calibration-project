"""
Model: an EEGNet-style convolutional front-end feeding a LoRA-adapted
transformer, with two levels of subject adapter.

Why this shape rather than the previous "patchify raw samples -> linear
embed -> 1 attention block":

1. **Spatial filtering has to come first.** Motor imagery is a *spatially
   distributed* rhythm change: the discriminative signal is the power in the
   mu/beta band at a particular linear combination of electrodes (that is
   exactly what CSP computes). Tokenizing each channel's time window
   independently and hoping self-attention rediscovers spatial filtering
   wastes both capacity and data. The depthwise spatial convolution here
   (`Conv2d(F1, F1*D, (C,1), groups=F1)`) *is* a learned bank of spatial
   filters, one set per temporal filter -- EEGNet's core design, which is
   still competitive with far larger models on BCI IV 2a.

2. **Then temporal structure.** After spatial filtering + power pooling, the
   sequence is short (tens of tokens rather than C*n_windows = 440), so a
   real multi-head transformer over it is cheap and models the ERD/ERS time
   course. Token count dropping ~20x is also why this version trains in
   seconds where the previous one did not.

3. **Two adapters, because subjects differ in two different ways.**
   - `SpatialAdapter`: a low-rank, per-subject remix of the *input channels*
     (X <- X + (alpha/r)(A B) X). Electrode placement, head geometry, and
     impedance shift are first and foremost a linear channel-space change,
     so an adapter in channel space is the right inductive bias, and it is
     tiny (2*C*r parameters). This is the piece the previous version was
     missing entirely.
   - `LoRALinear` on the transformer's projections: the standard low-rank
     adaptation, for whatever subject-specific structure survives the
     spatial remix.

LoRA `B` is zero-initialized here (the conventional choice), which makes an
un-adapted LoRA an exact no-op so the "zero-shot" arm really measures the
pretrained backbone. The earlier small-random-B choice quietly perturbed the
backbone at step 0 and made zero-shot look worse than it is. Because the
meta-learned init overwrites B anyway, we lose nothing.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- defaults (overridable per-experiment via ModelConfig) ----
D_MODEL = 64
N_HEADS = 4
N_BLOCKS = 2
FFN_MULT = 2
LORA_RANK = 8
LORA_ALPHA = 16
DROPOUT = 0.25


class ModelConfig:
    def __init__(self, n_channels, n_timepoints, n_classes=4, d_model=D_MODEL,
                 n_heads=N_HEADS, n_blocks=N_BLOCKS, ffn_mult=FFN_MULT,
                 lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA, dropout=DROPOUT,
                 f1=16, depth_mult=2, kern_len=64, use_spatial_adapter=True,
                 spatial_rank=4, pool_len=64, pool_stride=16):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def __repr__(self):
        keys = ["d_model", "n_heads", "n_blocks", "lora_rank", "f1", "depth_mult",
                "kern_len", "pool_len", "pool_stride", "use_spatial_adapter",
                "spatial_rank", "dropout"]
        return "ModelConfig(" + ", ".join(f"{k}={getattr(self, k)}" for k in keys) + ")"


# --------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = x W_frozen + b + (alpha/r) * (x A) B, with B zero-initialized."""

    def __init__(self, d_in, d_out, bias=True, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.empty(d_in, r))
        self.B = nn.Parameter(torch.zeros(r, d_out))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad_(False)

    def lora_parameters(self):
        return [self.A, self.B]

    @torch.no_grad()
    def reset_lora(self, zero_b=True, generator=None, scale=0.02):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        if zero_b:
            self.B.zero_()
        else:
            self.B.copy_(torch.randn(self.B.shape, generator=generator) * scale)

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.A) @ self.B)


class SpatialAdapter(nn.Module):
    """Per-subject low-rank channel remix: X <- X + (alpha/r) (A B) X.

    Identity at init (B=0), so it is a no-op until calibrated/meta-learned.
    This is where montage/impedance/head-geometry differences between
    subjects get absorbed, and it is the cheapest adapter in the model
    (2*C*r parameters, e.g. 176 for 22 channels at rank 4)."""

    def __init__(self, n_channels, r=4, alpha=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n_channels, r) * (1.0 / math.sqrt(n_channels)))
        self.B = nn.Parameter(torch.zeros(r, n_channels))
        self.scaling = alpha / r

    def adapter_parameters(self):
        return [self.A, self.B]

    @torch.no_grad()
    def reset(self):
        nn.init.normal_(self.A, std=1.0 / math.sqrt(self.A.shape[0]))
        self.B.zero_()

    def forward(self, x):                       # x: (B, C, T)
        W = self.scaling * (self.A @ self.B)    # (C, C)
        return x + torch.einsum("cd,ndt->nct", W, x)


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, ffn_hidden, r, alpha, dropout):
        super().__init__()
        assert d % n_heads == 0, f"d_model {d} not divisible by n_heads {n_heads}"
        self.h = n_heads
        self.dh = d // n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wq = LoRALinear(d, d, bias=False, r=r, alpha=alpha)
        self.Wk = LoRALinear(d, d, bias=False, r=r, alpha=alpha)
        self.Wv = LoRALinear(d, d, bias=False, r=r, alpha=alpha)
        self.Wo = LoRALinear(d, d, bias=False, r=r, alpha=alpha)
        self.W1 = LoRALinear(d, ffn_hidden, r=r, alpha=alpha)
        self.W2 = LoRALinear(ffn_hidden, d, r=r, alpha=alpha)
        self.drop = nn.Dropout(dropout)

    def lora_layers(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2]

    def _heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dh).transpose(1, 2)     # (B,h,N,dh)

    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self._heads(self.Wq(h)), self._heads(self.Wk(h)), self._heads(self.Wv(h))
        att = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(x.shape)
        x = x + self.drop(self.Wo(ctx))
        h = self.ln2(x)
        return x + self.drop(self.W2(F.gelu(self.W1(h))))


class EEGBackbone(nn.Module):
    """SpatialAdapter -> EEGNet conv stem -> transformer blocks -> tokens."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        C, T, d = cfg.n_channels, cfg.n_timepoints, cfg.d_model
        f1, dm, kl = cfg.f1, cfg.depth_mult, cfg.kern_len
        f2 = f1 * dm

        self.spatial_adapter = SpatialAdapter(C, r=cfg.spatial_rank) if cfg.use_spatial_adapter else None

        self.temporal = nn.Conv2d(1, f1, (1, kl), padding=(0, kl // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, f2, (C, 1), groups=f1, bias=False)   # depthwise spatial filters
        self.bn2 = nn.BatchNorm2d(f2)
        # Square -> average-pool -> log: this computes *band power in a
        # sliding window*, which is precisely the quantity motor imagery
        # modulates (ERD/ERS is a percentage power change in mu/beta). It is
        # ShallowFBCSPNet's (Schirrmeister et al. 2017) key nonlinearity and
        # it is the reason that network is still a strong BCI IV 2a baseline.
        # An ELU-only stem has to approximate a squaring operation with
        # piecewise-linear pieces and, at this data scale, does not.
        self.pool_pow = nn.AvgPool2d((1, cfg.pool_len), stride=(1, cfg.pool_stride))
        self.proj = nn.Conv2d(f2, d, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(d)
        self.drop = nn.Dropout(cfg.dropout)

        self.n_tokens = self._infer_tokens(C, T)
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens, d) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.n_heads, d * cfg.ffn_mult, cfg.lora_rank,
                             cfg.lora_alpha, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])
        self.ln_out = nn.LayerNorm(d)
        self.decoder_head = nn.Linear(d, d)   # pretraining only; never calibrated

    def _infer_tokens(self, C, T):
        with torch.no_grad():
            return self._stem(torch.zeros(1, C, T)).shape[1]

    # ---- convolutional stem ----
    def _stem(self, x):                       # x: (B, C, T)
        if self.spatial_adapter is not None:
            x = self.spatial_adapter(x)
        x = x.unsqueeze(1)                    # (B,1,C,T)
        x = self.bn1(self.temporal(x))        # temporal (band) filters
        x = self.bn2(self.spatial(x))         # learned spatial filters (CSP-like)
        x = self.pool_pow(x * x)              # sliding-window power
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        x = F.elu(self.bn3(self.proj(x)))     # (B,d,1,T')
        return x.squeeze(2).transpose(1, 2)   # (B,T',d)

    def encode(self, x, mask_idx=None):
        tok = self._stem(x)
        tok = tok + self.pos[:, :tok.shape[1]]
        if mask_idx is not None:
            keep = torch.ones(1, tok.shape[1], 1, device=tok.device)
            keep[0, mask_idx, 0] = 0.0
            tok = tok * keep + self.mask_token * (1.0 - keep)
        for blk in self.blocks:
            tok = blk(tok)
        return self.ln_out(tok)

    # ---- parameter groups ----
    def lora_layers(self):
        out = []
        for blk in self.blocks:
            out += blk.lora_layers()
        return out

    def lora_parameters(self):
        ps = []
        for lyr in self.lora_layers():
            ps += lyr.lora_parameters()
        if self.spatial_adapter is not None:
            ps += self.spatial_adapter.adapter_parameters()
        return ps

    def backbone_base_parameters(self):
        """Everything that is *not* an adapter -- what pretraining trains."""
        lora_ids = {id(p) for p in self.lora_parameters()}
        return [p for p in self.parameters() if id(p) not in lora_ids]

    def freeze_backbone_base(self):
        for p in self.backbone_base_parameters():
            p.requires_grad_(False)
        return self

    def set_frozen_eval_mode(self):
        """BatchNorm must use running statistics during few-shot calibration:
        with 12 trials, batch statistics are noise, and updating the running
        stats would silently un-freeze the 'frozen' backbone. Dropout is
        likewise disabled on the frozen path."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
                m.eval()

    @torch.no_grad()
    def reset_adapters(self, zero_b=True, generator=None):
        for lyr in self.lora_layers():
            lyr.reset_lora(zero_b=zero_b, generator=generator)
        if self.spatial_adapter is not None:
            self.spatial_adapter.reset()


class ClassifierHead(nn.Module):
    """Attention pooling over tokens + linear readout. Pooling is learned and
    calibrated, so the model can decide *when* in the trial the ERD lives for
    this particular subject -- which does vary between people."""

    def __init__(self, d_model, n_classes, dropout=0.25):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.1)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, n_classes)

    def pool(self, tok):
        w = F.softmax((tok @ self.query) / math.sqrt(tok.shape[-1]), dim=-1)
        return (tok * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, tok):
        return self.linear(self.drop(self.pool(tok)))

    def parameters_for_calibration(self):
        return [self.query] + list(self.linear.parameters())


# --------------------------------------------------------------------------
def adapter_state_dict(backbone, head=None):
    """Only adapters (+ head) -- never the frozen base. Saving a full
    state_dict as 'the meta init' silently smuggles base weights along, so a
    later load onto a differently-pretrained backbone would not error, it
    would just be wrong."""
    sd = {"lora": {}, "spatial": None, "head": None}
    for i, lyr in enumerate(backbone.lora_layers()):
        sd["lora"][f"A{i}"] = lyr.A.detach().clone()
        sd["lora"][f"B{i}"] = lyr.B.detach().clone()
    if backbone.spatial_adapter is not None:
        sd["spatial"] = {"A": backbone.spatial_adapter.A.detach().clone(),
                         "B": backbone.spatial_adapter.B.detach().clone()}
    if head is not None:
        sd["head"] = {k: v.detach().clone() for k, v in head.state_dict().items()}
    return sd


@torch.no_grad()
def load_adapter_state_dict(backbone, sd, head=None):
    layers = backbone.lora_layers()
    assert len(layers) == len(sd["lora"]) // 2, "adapter checkpoint/model mismatch"
    for i, lyr in enumerate(layers):
        lyr.A.copy_(sd["lora"][f"A{i}"])
        lyr.B.copy_(sd["lora"][f"B{i}"])
    if backbone.spatial_adapter is not None and sd.get("spatial") is not None:
        backbone.spatial_adapter.A.copy_(sd["spatial"]["A"])
        backbone.spatial_adapter.B.copy_(sd["spatial"]["B"])
    if head is not None and sd.get("head") is not None:
        head.load_state_dict(sd["head"])


# Compute profiles. `full` is what you should run on real BCI IV 2a with a
# GPU (or any machine without a 45-second-per-command cap). `sandbox` is a
# deliberately shrunk version used for the in-repo regression runs so the
# whole ablation ladder completes in an offline CI-style environment -- same
# code path, ~4x cheaper, and clearly labelled in every result it produces.
PROFILES = {
    "full":    dict(d_model=64, n_heads=4, n_blocks=2, f1=16, depth_mult=2,
                    kern_len=64, pool_len=64, pool_stride=16, lora_rank=8, spatial_rank=4),
    "sandbox": dict(d_model=48, n_heads=4, n_blocks=2, f1=8, depth_mult=2,
                    kern_len=32, pool_len=64, pool_stride=16, lora_rank=8, spatial_rank=4),
}


def make_config(n_channels, n_timepoints, n_classes=4, profile="full", **overrides):
    kw = dict(PROFILES[profile])
    kw.update(overrides)
    return ModelConfig(n_channels=n_channels, n_timepoints=n_timepoints,
                       n_classes=n_classes, **kw)


def build(cfg):
    return EEGBackbone(cfg), ClassifierHead(cfg.d_model, cfg.n_classes, cfg.dropout)


class CalibModel(nn.Module):
    """Backbone + head as one module, so the meta-learner can run a
    *functional* (differentiable) inner loop over it with
    `torch.func.functional_call`. Without a single container you cannot
    cleanly express "forward this model with these hypothetical parameter
    values", which is exactly what an inner loop is."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone.encode(x))

    def adapter_names(self):
        """Names (as `named_parameters` keys) of everything calibrated:
        LoRA A/B, the spatial adapter, and the classifier head."""
        wanted = {id(p) for p in
                  self.backbone.lora_parameters() + self.head.parameters_for_calibration()}
        return [n for n, p in self.named_parameters() if id(p) in wanted]
