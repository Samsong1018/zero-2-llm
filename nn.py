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
        if self.training:
            xmean = x.mean(0, keepdims=True)
            xvar = x.var(0, keepdims=True, correction=1)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / (xvar + self.eps)**0.5
        out = self.gamma * xhat + self.beta
        if self.training:
            with no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean.data
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar.data
        return out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        out = x.tanh()
        return out
    def parameters(self):
        return []