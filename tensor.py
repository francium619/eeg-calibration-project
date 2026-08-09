"""
Minimal reverse-mode autodiff engine, built on NumPy.

This exists so the EEG backbone / LoRA / meta-learning code in this project
can be expressed as ordinary matrix ops (matmul, softmax, layernorm, GELU...)
without hand-deriving gradients for every layer. It predates the PyTorch
port (`torch_*.py`) and is kept as a dependency-free reference -- see
README for how the two pipelines relate.

Design: each `Tensor` wraps a NumPy array and, if `requires_grad`, records a
`_backward` closure that accumulates gradients into its parents. Calling
`.backward()` on a scalar loss walks the graph in reverse topological order.

This is intentionally small (~200 lines) and only implements the ops this
project needs: add, sub, mul, matmul, transpose, reshape, relu, gelu,
softmax, layernorm, mean, sum, cross-entropy. It is not meant to be a
general-purpose replacement for autograd/PyTorch.
"""
import numpy as np


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_backward", "_parents", "_op")

    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._parents = _parents
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # ---- graph construction helpers ----
    def _wrap(self, out_data, parents, backward, op=""):
        req = any(p.requires_grad for p in parents)
        out = Tensor(out_data, requires_grad=req, _parents=parents, _op=op)
        if req:
            out._backward = backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data

        def backward():
            if self.requires_grad:
                g = out.grad
                self.grad += _unbroadcast(g, self.data.shape)
            if other.requires_grad:
                g = out.grad
                other.grad += _unbroadcast(g, other.data.shape)

        out = self._wrap(out_data, (self, other), backward, "add")
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (other * -1.0)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data

        def backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out = self._wrap(out_data, (self, other), backward, "mul")
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __pow__(self, p):
        out_data = self.data ** p

        def backward():
            if self.requires_grad:
                self.grad += out.grad * (p * self.data ** (p - 1))

        out = self._wrap(out_data, (self,), backward, "pow")
        return out

    def matmul(self, other):
        out_data = self.data @ other.data

        def backward():
            # Handles broadcasting (e.g. (B,N,L)@(L,d) with a plain 2D weight
            # matrix): the raw product can have extra leading batch dims
            # relative to the parent's own shape, which must be summed out.
            if self.requires_grad:
                raw = out.grad @ _swap_last2(other.data)
                self.grad += _unbroadcast(raw, self.data.shape)
            if other.requires_grad:
                raw = _swap_last2(self.data) @ out.grad
                other.grad += _unbroadcast(raw, other.data.shape)

        out = self._wrap(out_data, (self, other), backward, "matmul")
        return out

    def transpose_last2(self):
        out_data = _swap_last2(self.data)

        def backward():
            if self.requires_grad:
                self.grad += _swap_last2(out.grad)

        out = self._wrap(out_data, (self,), backward, "T")
        return out

    def reshape(self, *shape):
        orig_shape = self.data.shape
        out_data = self.data.reshape(*shape)

        def backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(orig_shape)

        out = self._wrap(out_data, (self,), backward, "reshape")
        return out

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)

        def backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self.grad += np.broadcast_to(g, self.data.shape)

        out = self._wrap(out_data, (self,), backward, "sum")
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def relu(self):
        out_data = np.maximum(self.data, 0.0)

        def backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out = self._wrap(out_data, (self,), backward, "relu")
        return out

    def gelu(self):
        # tanh approximation of GELU
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out_data = 0.5 * x * (1.0 + t)

        def backward():
            if self.requires_grad:
                sech2 = 1.0 - t ** 2
                dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
                dgelu = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner
                self.grad += out.grad * dgelu

        out = self._wrap(out_data, (self,), backward, "gelu")
        return out

    def softmax(self, axis=-1):
        x = self.data
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        s = e / np.sum(e, axis=axis, keepdims=True)

        def backward():
            if self.requires_grad:
                g = out.grad
                dot = np.sum(g * s, axis=axis, keepdims=True)
                self.grad += s * (g - dot)

        out = self._wrap(s, (self,), backward, "softmax")
        return out

    def layernorm(self, gamma, beta, eps=1e-5):
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        xhat = (x - mu) / std
        out_data = xhat * gamma.data + beta.data

        def backward():
            g = out.grad
            if gamma.requires_grad:
                gamma.grad += _unbroadcast(g * xhat, gamma.data.shape)
            if beta.requires_grad:
                beta.grad += _unbroadcast(g, beta.data.shape)
            if self.requires_grad:
                N = x.shape[-1]
                dxhat = g * gamma.data
                dvar_term = dxhat * xhat
                self.grad += (1.0 / (N * std)) * (
                    N * dxhat
                    - dxhat.sum(axis=-1, keepdims=True)
                    - xhat * dvar_term.sum(axis=-1, keepdims=True)
                )

        out = self._wrap(out_data, (self, gamma, beta), backward, "layernorm")
        return out

    def backward(self):
        assert self.data.size == 1, "backward() only supported on scalar tensors"
        topo, visited = [], set()

        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._parents:
                    build(p)
                topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()


def _swap_last2(a):
    axes = list(range(a.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return np.transpose(a, axes)


def _unbroadcast(grad, shape):
    """Sum-reduce `grad` down to `shape` to undo NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def softmax_cross_entropy(logits: Tensor, labels: np.ndarray):
    """logits: Tensor of shape (B, C); labels: int array of shape (B,)."""
    probs = logits.softmax(axis=-1)
    B = logits.shape[0]
    logp = np.log(np.clip(probs.data[np.arange(B), labels], 1e-12, 1.0))
    loss_val = -np.mean(logp)
    loss = Tensor(loss_val, requires_grad=True, _parents=(logits,), _op="xent")

    def backward():
        if logits.requires_grad:
            g = probs.data.copy()
            g[np.arange(B), labels] -= 1.0
            g /= B
            logits.grad += g * loss.grad

    loss._backward = backward
    return loss, probs.data
