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

    def sum(self):
        out = Tensor(np.sum(self.data), (self, ), 'sum')
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        return self.sum() * (1 /  self.data.size)

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