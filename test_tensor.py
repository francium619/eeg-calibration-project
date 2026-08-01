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

print("\nAll gradient checks passed.")
