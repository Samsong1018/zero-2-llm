import numpy as np
import math

def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(0)
    
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis, keepdims=True)

    return grad

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
       return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast((other.data * out.grad), self.data.shape)
            other.grad += _unbroadcast((self.data * out.grad), other.data.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        out = self * other**-1

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def sigmoid(self):
        s = 1 / (1+np.exp(-self.data))
        out = Tensor(s, (self, ), 'sig')
        def _backward():
            self.grad += s * (1-s) * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Value Error"
        out = Tensor(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out


    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += g / n
        out._backward = _backward
        return out

    def tanh(self):
        n = self.data
        t = np.tanh(self.data)
        out = Tensor(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def randn(*shape, scale=1):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return Tensor(np.random.randn(*shape) * scale)

    def __getitem__(self, idx):
        def unwrap(i):
            return i.data.astype(int) if isinstance(i, Tensor) else i
        idx = tuple(unwrap(i) for i in idx) if isinstance(idx, tuple) else unwrap(idx)
        out = Tensor(self.data[idx], (self,), 'index')
        def _backward():
            np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    def view(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'view')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        t = targets.data.astype(int) if isinstance(targets, Tensor) else np.asarray(targets, dtype=int)
        n, c = self.data.shape
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        probs = np.exp(logp)
        out = Tensor(-logp[np.arange(n), t].mean(), (self, ), 'cross_entropy')

        def _backward():
            self.grad += (probs - np.eye(c)[t]) / n * out.grad
        out._backward = _backward
        return out

    def var(self, axis=None, keepdims=False, correction=1):
        n = self.data.size if axis is None else self.data.shape[axis]
        diff = self.data - self.data.mean(axis=axis, keepdims=True)
        v = (diff**2).sum(axis=axis, keepdims=True) / (n - correction)
        out = Tensor(v if keepdims else np.squeeze(v, axis=axis), (self,), 'var')

        def _backward():
            g = out.grad.reshape(v.shape)
            self.grad += g * 2 * diff / (n - correction)
        out._backward = _backward
        return out
    
    def std(self, axis=None, keepdims=False, correction=1, eps=0.0):
        n = self.data.size if axis is None else self.data.shape[axis]
        diff = self.data - self.data.mean(axis=axis, keepdims=True)
        s = np.sqrt((diff**2).sum(axis=axis, keepdims=True) / (n - correction) + eps)
        out = Tensor(s if keepdims else np.squeeze(s, axis=axis), (self,), 'std')

        def _backward():
            g = out.grad.reshape(s.shape)
            self.grad += g * diff / ((n - correction) * s)
        out._backward = _backward
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
        
        self.grad = np.ones_like(self.data)

        for i in reversed(topo):
            i._backward()