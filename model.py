"""
A small "EEG foundation model"-style backbone: patchify -> linear patch
embed -> channel/window position embeddings -> one self-attention block ->
one FFN block -> mean-pool -> head. This mirrors the coarse architecture of
real EEG foundation models (LaBraM, BENDR): tokenize short per-channel time
windows, run a transformer encoder, pool to a representation.

The novelty this project adds on top is `Linear.add_lora()`: every linear
layer in the backbone can carry a frozen base weight W plus a trainable
low-rank pair (A, B) with W_eff = W + scaling * A @ B (standard LoRA). After
self-supervised pretraining (see pretrain.py), the base weights are frozen
and only the LoRA pairs + task head are adapted per subject -- and
meta_train.py learns a LoRA *initialization* that adapts in very few
gradient steps (see README for why this matters for real-time calibration).
"""
import numpy as np
from tensor import Tensor

D_MODEL = 16
FFN_HIDDEN = 32
LORA_RANK = 4


class Linear:
    def __init__(self, d_in, d_out, rng, bias=True):
        scale = 1.0 / np.sqrt(d_in)
        self.W = Tensor(rng.uniform(-scale, scale, size=(d_in, d_out)), requires_grad=True)
        self.b = Tensor(np.zeros(d_out), requires_grad=True) if bias else None
        self.A = None
        self.B = None
        self.r = 0
        self.scaling = 0.0

    def add_lora(self, r, rng, alpha=None):
        d_in, d_out = self.W.data.shape
        self.r = r
        self.alpha = alpha if alpha is not None else 4 * r
        self.scaling = self.alpha / r
        # Standard LoRA zero-inits B so the adapter starts as an exact no-op.
        # In this small-scale prototype that also makes dL/dA == 0 at step 0
        # (it flows through B), which starves the adapter of gradient in a
        # very-few-step calibration regime. We instead use a small random B,
        # which sacrifices the "exact no-op at init" property for a working
        # gradient signal from step 1 -- appropriate here since the LoRA
        # init itself gets overwritten by the meta-learned init anyway.
        self.A = Tensor(rng.standard_normal((d_in, r)) * 0.05, requires_grad=True)
        self.B = Tensor(rng.standard_normal((r, d_out)) * 0.01, requires_grad=True)

    def reset_lora(self, rng):
        assert self.r > 0
        d_in, d_out = self.W.data.shape
        self.A = Tensor(rng.standard_normal((d_in, self.r)) * 0.05, requires_grad=True)
        self.B = Tensor(rng.standard_normal((self.r, d_out)) * 0.01, requires_grad=True)

    def freeze_base(self):
        self.W.requires_grad = False
        self.W.grad = None
        if self.b is not None:
            self.b.requires_grad = False
            self.b.grad = None

    def __call__(self, x):
        out = x.matmul(self.W)
        if self.b is not None:
            out = out + self.b
        if self.A is not None:
            out = out + (x.matmul(self.A)).matmul(self.B) * self.scaling
        return out

    def base_params(self):
        ps = [self.W]
        if self.b is not None:
            ps.append(self.b)
        return ps

    def lora_params(self):
        return [self.A, self.B] if self.A is not None else []


class EEGBackbone:
    """One attention block + one FFN block over channel-window tokens."""

    def __init__(self, n_channels, n_windows, win_len, seed=0):
        rng = np.random.default_rng(seed)
        self.n_channels = n_channels
        self.n_windows = n_windows
        self.win_len = win_len
        d = D_MODEL

        self.patch_embed = Linear(win_len, d, rng)
        self.ch_embed = Tensor(rng.standard_normal((n_channels, d)) * 0.02, requires_grad=True)
        self.win_embed = Tensor(rng.standard_normal((n_windows, d)) * 0.02, requires_grad=True)
        self.mask_token = Tensor(rng.standard_normal((1, 1, d)) * 0.02, requires_grad=True)

        self.ln1_g = Tensor(np.ones(d), requires_grad=True)
        self.ln1_b = Tensor(np.zeros(d), requires_grad=True)
        self.Wq = Linear(d, d, rng, bias=False)
        self.Wk = Linear(d, d, rng, bias=False)
        self.Wv = Linear(d, d, rng, bias=False)
        self.Wo = Linear(d, d, rng, bias=False)

        self.ln2_g = Tensor(np.ones(d), requires_grad=True)
        self.ln2_b = Tensor(np.zeros(d), requires_grad=True)
        self.W1 = Linear(d, FFN_HIDDEN, rng)
        self.W2 = Linear(FFN_HIDDEN, d, rng)

        self.decoder_head = Linear(d, win_len, rng)  # pretraining-only

        ch_ids = np.repeat(np.arange(n_channels), n_windows)
        win_ids = np.tile(np.arange(n_windows), n_channels)
        self._ch_ids = ch_ids
        self._win_ids = win_ids
        # One-hot selection matrices so the channel/window embedding lookup
        # is expressed as a matmul (keeps it inside the autodiff graph, so
        # ch_embed/win_embed get real gradients during pretraining).
        self._ch_onehot = Tensor(np.eye(n_channels)[ch_ids], requires_grad=False)
        self._win_onehot = Tensor(np.eye(n_windows)[win_ids], requires_grad=False)

        self._lora_layers = [self.patch_embed, self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2]

    # ---- parameter groups ----
    def backbone_base_params(self):
        ps = []
        for lyr in self._lora_layers:
            ps += lyr.base_params()
        ps += [self.ch_embed, self.win_embed, self.mask_token, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b]
        return ps

    def decoder_params(self):
        return self.decoder_head.base_params()

    def add_lora_everywhere(self, r, seed):
        rng = np.random.default_rng(seed)
        for lyr in self._lora_layers:
            lyr.add_lora(r, rng)

    def reset_lora_everywhere(self, seed):
        rng = np.random.default_rng(seed)
        for lyr in self._lora_layers:
            lyr.reset_lora(rng)

    def lora_params(self):
        ps = []
        for lyr in self._lora_layers:
            ps += lyr.lora_params()
        return ps

    def freeze_backbone_base(self):
        for lyr in self._lora_layers:
            lyr.freeze_base()
        for t in [self.ch_embed, self.win_embed, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b]:
            t.requires_grad = False
            t.grad = None

    # ---- forward ----
    def embed_tokens(self, patches_np):
        """patches_np: (B, N_TOKENS, WIN) -> Tensor (B, N_TOKENS, D_MODEL)."""
        x = Tensor(patches_np, requires_grad=False)
        tok = self.patch_embed(x)
        ch_sel = self._ch_onehot.matmul(self.ch_embed)    # (N_TOKENS, D), gradient flows to ch_embed
        win_sel = self._win_onehot.matmul(self.win_embed)  # (N_TOKENS, D), gradient flows to win_embed
        tok = tok + ch_sel + win_sel
        return tok

    def _attn_block(self, tok):
        d = D_MODEL
        h = tok.layernorm(self.ln1_g, self.ln1_b)
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        scores = q.matmul(k.transpose_last2()) * (1.0 / np.sqrt(d))
        attn = scores.softmax(axis=-1)
        ctx = attn.matmul(v)
        out = self.Wo(ctx)
        return tok + out

    def _ffn_block(self, tok):
        h = tok.layernorm(self.ln2_g, self.ln2_b)
        h = self.W1(h).gelu()
        h = self.W2(h)
        return tok + h

    def encode(self, patches_np, mask_idx=None):
        """Returns (pooled (B,D), token_repr (B,N,D))."""
        tok = self.embed_tokens(patches_np)
        if mask_idx is not None:
            B, N, D = tok.data.shape
            keep = np.ones((1, N, 1))
            keep[0, mask_idx, 0] = 0.0
            tok = tok * Tensor(keep, requires_grad=False) + self.mask_token * Tensor(1.0 - keep, requires_grad=False)
        tok = self._attn_block(tok)
        tok = self._ffn_block(tok)
        pooled = tok.mean(axis=1)
        return pooled, tok


class ClassifierHead:
    """Attention-pooling classifier head: a learnable query vector attends
    over the backbone's token representations to produce a pooled feature,
    which a linear layer classifies. Unlike plain mean-pooling, the pooling
    itself is task-adaptive -- it is part of what gets meta-learned and
    per-subject calibrated, so calibration can learn *which* channel/window
    tokens matter for this subject, not just how to reweight fixed features.
    """

    def __init__(self, n_classes, seed):
        rng = np.random.default_rng(seed)
        self.query = Tensor(rng.standard_normal(D_MODEL) * 0.1, requires_grad=True)
        self.linear = Linear(D_MODEL, n_classes, rng)

    def pool(self, tok):
        B, N, D = tok.shape
        q2d = self.query.reshape(D, 1)
        scores = tok.matmul(q2d).reshape(B, N) * (1.0 / np.sqrt(D))
        weights = scores.softmax(axis=-1).reshape(B, N, 1)
        pooled = (tok * weights).sum(axis=1)
        return pooled

    def __call__(self, tok):
        return self.linear(self.pool(tok))

    def params(self):
        return [self.query] + self.linear.base_params()

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.query = Tensor(rng.standard_normal(D_MODEL) * 0.1, requires_grad=True)
        d_in, d_out = self.linear.W.data.shape
        scale = 1.0 / np.sqrt(d_in)
        self.linear.W = Tensor(rng.uniform(-scale, scale, size=(d_in, d_out)), requires_grad=True)
        self.linear.b = Tensor(np.zeros(d_out), requires_grad=True)


def zero_grad(params):
    for p in params:
        p.grad = np.zeros_like(p.data)


def clip_grad_norm(params, max_norm):
    """Global-norm gradient clipping across all params, in place. Guards
    against rare exploding updates -- most likely with badly-conditioned
    random LoRA inits taking several full-batch steps on a tiny (e.g. 12
    trial) calibration set."""
    total = np.sqrt(sum(float(np.sum(p.grad ** 2)) for p in params))
    if total > max_norm and np.isfinite(total):
        scale = max_norm / (total + 1e-8)
        for p in params:
            p.grad *= scale
    elif not np.isfinite(total):
        for p in params:
            p.grad[:] = 0.0


def sgd_step(params, lr):
    for p in params:
        p.data -= lr * p.grad


def load_pretrained(bb, path):
    d = np.load(path)
    bb.patch_embed.W.data = d["patch_embed_W"]; bb.patch_embed.b.data = d["patch_embed_b"]
    bb.Wq.W.data = d["Wq"]; bb.Wk.W.data = d["Wk"]; bb.Wv.W.data = d["Wv"]; bb.Wo.W.data = d["Wo"]
    bb.W1.W.data = d["W1_W"]; bb.W1.b.data = d["W1_b"]
    bb.W2.W.data = d["W2_W"]; bb.W2.b.data = d["W2_b"]
    bb.ch_embed.data = d["ch_embed"]; bb.win_embed.data = d["win_embed"]
    bb.ln1_g.data = d["ln1_g"]; bb.ln1_b.data = d["ln1_b"]
    bb.ln2_g.data = d["ln2_g"]; bb.ln2_b.data = d["ln2_b"]


def snapshot(params):
    return [p.data.copy() for p in params]


def restore(params, snap):
    for p, s in zip(params, snap):
        p.data = s.copy()
