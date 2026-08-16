"""Gradient check for tensor.py via finite differences. Run: python3 test_tensor.py"""
import numpy as np
from tensor import Tensor, softmax_cross_entropy

np.random.seed(0)


def numeric_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f1 = f(x)
        x[idx] = orig - eps
        f2 = f(x)
        x[idx] = orig
        g[idx] = (f1 - f2) / (2 * eps)
    return g


def check(name, build_fn, x0):
    x0 = x0.astype(np.float64)

    def f(xv):
        t = Tensor(xv, requires_grad=True)
        out = build_fn(t)
        return float(out.data)

    ng = numeric_grad(f, x0.copy())

    t = Tensor(x0.copy(), requires_grad=True)
    out = build_fn(t)
    out.backward()
    ag = t.grad

    err = np.max(np.abs(ng - ag)) / (np.max(np.abs(ng)) + 1e-8)
    status = "OK" if err < 1e-3 else "FAIL"
    print(f"[{status}] {name}: max rel err = {err:.2e}")
    assert status == "OK", f"{name} gradient check failed"


# matmul + relu + sum
W = Tensor(np.random.randn(4, 3), requires_grad=False)
check("matmul+relu+sum", lambda t: t.matmul(W).relu().sum(), np.random.randn(5, 4))

# gelu
check("gelu+sum", lambda t: t.gelu().sum(), np.random.randn(6, 6))

# softmax + sum of squares
check("softmax+sumsq", lambda t: (t.softmax(axis=-1) * t.softmax(axis=-1)).sum(), np.random.randn(3, 5))

# layernorm (note: plain .sum() after layernorm is ~invariant to x since xhat is
# mean-zero, so we use a non-degenerate probe: dot with random weights + gelu)
gamma = Tensor(np.random.randn(4) * 0.5 + 1.0, requires_grad=False)
beta = Tensor(np.random.randn(4) * 0.1, requires_grad=False)
probe = Tensor(np.random.randn(4, 4), requires_grad=False)
check(
    "layernorm+matmul+gelu+sum",
    lambda t: t.layernorm(gamma, beta).matmul(probe).gelu().sum(),
    np.random.randn(3, 4),
)

# mean over an axis (used by EEGBackbone.encode's pooled = tok.mean(axis=1))
Wm = Tensor(np.random.randn(4, 3), requires_grad=False)
check("mean(axis)+matmul+sum", lambda t: t.mean(axis=1).matmul(Wm).sum(), np.random.randn(5, 6, 4))

# reshape (used by ClassifierHead.pool's query.reshape(D,1) / scores.reshape(B,N)
# to move a vector between attention-pooling's matmul and softmax shapes)
Wr = Tensor(np.random.randn(4, 5), requires_grad=False)
check(
    "reshape+matmul+relu+sum",
    lambda t: t.reshape(3, 4).matmul(Wr).relu().sum(),
    np.random.randn(12),
)

# pow / truediv -- implemented on Tensor but not exercised by any model code
# (mean() divides by n as a plain Python float, not via Tensor.__truediv__),
# so this is their only regression coverage.
check("pow+sum", lambda t: ((t.relu() + 0.5) ** 1.7).sum(), np.random.randn(4, 5))

Wd = Tensor(np.random.randn(4, 3), requires_grad=False)
check(
    "truediv+matmul+sum",
    lambda t: ((t.relu() + 0.5) / 2.0).matmul(Wd).sum(),
    np.random.randn(5, 4),
)

# transpose_last2 (used by attention's q.matmul(k.transpose_last2()) to form
# the (..., d, N) key matrix for scores = q @ k^T), had no regression coverage
Wt = Tensor(np.random.randn(4, 3), requires_grad=False)
check(
    "transpose_last2+matmul+sum",
    lambda t: t.transpose_last2().matmul(Wt).relu().sum(),
    np.random.randn(2, 4, 5),
)

# subtraction (used by pretrain.py's masked-reconstruction loss: pred - target)
# -- every other binary op above has a direct check; __sub__ had none.
Ws = Tensor(np.random.randn(4, 3), requires_grad=False)
target = Tensor(np.random.randn(5, 3), requires_grad=False)
check("sub+matmul+sum", lambda t: (t.matmul(Ws) - target).relu().sum(), np.random.randn(5, 4))

# cross entropy
labels = np.array([0, 2, 1])


def xent_build(t):
    loss, _ = softmax_cross_entropy(t, labels)
    return loss


check("cross_entropy", xent_build, np.random.randn(3, 4))

# batched linear layer: X (B,N,L) @ W (L,d) -- 2D weight broadcast over a 3D
# batched/token input, as used throughout the EEG backbone. Checks that
# matmul's backward correctly sums gradients over the broadcast batch dims.
Wb = Tensor(np.random.randn(4, 3), requires_grad=False)
check(
    "batched_linear (B,N,L)@(L,d)",
    lambda t: t.matmul(Wb).relu().sum(),
    np.random.randn(2, 5, 4),
)

# backward() should refuse to run on a non-scalar output (e.g. forgetting a
# final .sum()/.mean()), since the reverse pass assumes a single seed gradient
# of 1.0 -- this guard had no regression coverage.
try:
    Tensor(np.random.randn(3, 4), requires_grad=True).relu().backward()
    status = "FAIL"
except AssertionError as e:
    status = "OK" if "scalar" in str(e) else "FAIL"
print(f"[{status}] backward() rejects non-scalar tensor")
assert status == "OK", "backward() should raise on a non-scalar tensor"

print("\nAll gradient checks passed.")
