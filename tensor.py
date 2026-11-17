import math
import numpy as np
from backend import xp, scatter_add
from contextlib import contextmanager


@contextmanager
def no_grad():
    prev = Tensor.grad_enabled
    Tensor.grad_enabled = False
    try:
        yield                    # the with-block runs here
    finally:
        Tensor.grad_enabled = prev


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis, keepdims=True)

    return grad


class Tensor:
    grad_enabled = True

    def __init__(self, data, _children=(), _op=''):
        self.data = xp.asarray(data, dtype=xp.float32)
        self._prev = tuple(_children) if self.grad_enabled else ()
        self._op = _op
        self._backward = lambda: None
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    def item(self):
        # plain Python float, on CPU or GPU
        return float(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def _set_backward(self, fn):
        if Tensor.grad_enabled:
            self._backward = fn

    def _accum(self, g):
        if self.grad is None:
            self.grad = xp.broadcast_to(g, self.data.shape).copy()
        else:
            self.grad += g

    def zero_grad(self):
        self.grad = None

    @staticmethod
    def randn(*shape, scale=1):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        # drawn with NumPy's RNG on every device, so a seed gives the same init on CPU and GPU
        return Tensor(np.random.randn(*shape) * scale)

    @staticmethod
    def cat(tensors, axis=-1):
        out = Tensor(xp.concatenate([t.data for t in tensors], axis=axis), tuple(tensors), 'cat')
        def _backward():
            sizes = [t.data.shape[axis] for t in tensors]
            pieces = xp.split(out.grad, np.cumsum(sizes)[:-1].tolist(), axis=axis)
            for t, g in zip(tensors, pieces):
                t._accum(g)
        out._set_backward(_backward)
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self._accum(_unbroadcast(out.grad, self.data.shape))
            other._accum(_unbroadcast(out.grad, other.data.shape))
        out._set_backward(_backward)
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self._accum(_unbroadcast((other.data * out.grad), self.data.shape))
            other._accum(_unbroadcast((self.data * out.grad), other.data.shape))
        out._set_backward(_backward)
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Value Error"
        out = Tensor(self.data**other, (self, ), f'**{other}')

        def _backward():
            self._accum(other * self.data**(other-1) * out.grad)
        out._set_backward(_backward)

        return out

    def __truediv__(self, other):
        out = self * other**-1

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            dA = out.grad @ xp.swapaxes(other.data, -1, -2)
            dB = xp.swapaxes(self.data, -1, -2) @ out.grad
            self._accum(_unbroadcast(dA, self.data.shape))
            other._accum(_unbroadcast(dB, other.data.shape))
        out._set_backward(_backward)
        return out

    def exp(self):
        e = xp.exp(self.data)
        out = Tensor(e, (self,), 'exp')
        def _backward():
            self._accum(e * out.grad)
        out._set_backward(_backward)
        return out

    def log(self):
        out = Tensor(xp.log(self.data), (self,), 'log')
        def _backward():
            self._accum((1 / self.data) * out.grad)
        out._set_backward(_backward)
        return out

    def tanh(self):
        t = xp.tanh(self.data)
        out = Tensor(t, (self, ), 'tanh')

        def _backward():
            self._accum((1-t**2) * out.grad)
        out._set_backward(_backward)
        return out

    def sigmoid(self):
        s = 1 / (1+xp.exp(-self.data))
        out = Tensor(s, (self, ), 'sig')
        def _backward():
            self._accum(s * (1-s) * out.grad)
        out._set_backward(_backward)
        return out

    def relu(self):
        out = Tensor(xp.maximum(self.data, 0), (self,), 'relu')
        def _backward():
            self._accum((self.data > 0) * out.grad)
        out._set_backward(_backward)
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = xp.expand_dims(g, axis)
            self._accum(xp.broadcast_to(g, self.data.shape))
        out._set_backward(_backward)
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = xp.expand_dims(g, axis)
            self._accum(g / n)
        out._set_backward(_backward)
        return out

    def var(self, axis=None, keepdims=False, correction=1):
        n = self.data.size if axis is None else self.data.shape[axis]
        diff = self.data - self.data.mean(axis=axis, keepdims=True)
        v = (diff**2).sum(axis=axis, keepdims=True) / (n - correction)
        out = Tensor(v if keepdims else xp.squeeze(v, axis=axis), (self,), 'var')

        def _backward():
            g = out.grad.reshape(v.shape)
            self._accum(g * 2 * diff / (n - correction))
        out._set_backward(_backward)
        return out

    def std(self, axis=None, keepdims=False, correction=1, eps=0.0):
        n = self.data.size if axis is None else self.data.shape[axis]
        diff = self.data - self.data.mean(axis=axis, keepdims=True)
        s = xp.sqrt((diff**2).sum(axis=axis, keepdims=True) / (n - correction) + eps)
        out = Tensor(s if keepdims else xp.squeeze(s, axis=axis), (self,), 'std')

        def _backward():
            g = out.grad.reshape(s.shape)
            self._accum(g * diff / ((n - correction) * s))
        out._set_backward(_backward)
        return out

    def __getitem__(self, idx):
        def unwrap(i):
            if isinstance(i, Tensor):
                return i.data.astype(int)
            if isinstance(i, np.ndarray):
                return xp.asarray(i)    # move NumPy index arrays to the device (no-op on CPU)
            return i
        idx = tuple(unwrap(i) for i in idx) if isinstance(idx, tuple) else unwrap(idx)
        out = Tensor(self.data[idx], (self,), 'index')
        def _backward():
            if self.grad is None:
                self.grad = xp.zeros_like(self.data)
            scatter_add(self.grad, idx, out.grad)
        out._set_backward(_backward)
        return out

    def view(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'view')
        def _backward():
            self._accum(out.grad.reshape(self.data.shape))
        out._set_backward(_backward)
        return out

    def squeeze(self, axis):
        out = Tensor(xp.squeeze(self.data, axis=axis), (self,), 'squeeze')
        def _backward():
            self._accum(out.grad.reshape(self.data.shape))
        out._set_backward(_backward)
        return out

    def transpose(self, dim0, dim1):
        out = Tensor(xp.swapaxes(self.data, dim0, dim1), (self,), 'transpose')
        def _backward():
            self._accum(xp.swapaxes(out.grad, dim0, dim1))
        out._set_backward(_backward)
        return out

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = xp.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,), 'softmax')
        def _backward():
            g = out.grad
            self._accum(s * (g - (g*s).sum(axis=axis, keepdims=True)))
        out._set_backward(_backward)
        return out

    def cross_entropy(self, targets):
        t = targets.data.astype(int) if isinstance(targets, Tensor) else xp.asarray(targets, dtype=int)
        n, c = self.data.shape
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        logp = shifted - xp.log(xp.exp(shifted).sum(axis=1, keepdims=True))
        probs = xp.exp(logp)
        out = Tensor(-logp[xp.arange(n), t].mean(), (self, ), 'cross_entropy')

        def _backward():
            d = probs.copy()
            d[xp.arange(n), t] -= 1
            self._accum(d / n * out.grad)
        out._set_backward(_backward)
        return out

    def layer_norm(self, gamma, beta, eps=1e-5):
        X = self.data
        N = X.shape[-1]
        mu = X.mean(-1, keepdims=True)
        var = ((X - mu)**2).mean(-1, keepdims=True)
        inv = 1 / xp.sqrt(var + eps)
        xhat = (X - mu) * inv
        out = Tensor(gamma.data * xhat + beta.data, (self, gamma, beta), 'layernorm')

        def _backward():
            d = out.grad
            gamma._accum(_unbroadcast(d * xhat, gamma.data.shape))
            beta._accum(_unbroadcast(d, beta.data.shape))
            dxhat = d * gamma.data
            self._accum(inv / N * (N*dxhat - dxhat.sum(-1, keepdims=True) - xhat * (dxhat * xhat).sum(-1, keepdims=True)))

        out._set_backward(_backward)
        return out

    def batch_norm(self, gamma, beta, eps=1e-5, axes=(0,)):
        X = self.data
        assert X.ndim - len(axes) == 1, f"batch_norm on {X.ndim}D input must reduce over all but the last dim, got axes={axes}"
        N = math.prod(X.shape[a] for a in axes)
        mu = X.mean(axes, keepdims=True)
        var = ((X - mu)**2).sum(axes, keepdims=True) / (N - 1)
        inv = 1 / xp.sqrt(var + eps)
        xhat = (X - mu) * inv
        out = Tensor(gamma.data * xhat + beta.data, (self, gamma, beta), 'batchnorm')

        def _backward():
            d = out.grad
            gamma._accum(_unbroadcast(d * xhat, gamma.data.shape))
            beta._accum(_unbroadcast(d, beta.data.shape))
            dsum = d.sum(axes, keepdims=True)
            dxsum = (d * xhat).sum(axes, keepdims=True)
            self._accum(gamma.data * inv / N * (N*d - dsum - N/(N-1) * xhat * dxsum))

        out._set_backward(_backward)
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = xp.ones_like(self.data)

        for i in reversed(topo):
            i._backward()

        # Optimization - break closure tensor cycles so the graph fress
        for i in topo:
            i._backward = lambda: None
            i._prev = ()
