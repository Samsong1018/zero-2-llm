from tensor import Tensor, no_grad
import numpy as np

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = Tensor(np.random.randn(fan_in, fan_out) / fan_in**0.5)
        self.bias = Tensor(np.zeros(fan_out)) if bias else None

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias
        return out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def __call__(self, x):
        axes = tuple(range(x.data.ndim - 1))   # (0,) for 2D, (0, 1) for 3D
        if self.training:
            out = x.batch_norm(self.gamma, self.beta, self.eps, axes)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * x.data.mean(axes)
            self.running_var  = (1-self.momentum) * self.running_var  + self.momentum * x.data.var(axes, ddof=1)
        else:
            xhat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * xhat + self.beta
        return out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        out = x.tanh()
        return out
    def parameters(self):
        return []