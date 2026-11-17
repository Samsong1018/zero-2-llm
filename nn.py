from tensor import Tensor, no_grad
from backend import xp
import numpy as np    # weight init only, so a seed gives the same weights on CPU and GPU

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
        self.running_mean = xp.zeros(dim)
        self.running_var = xp.ones(dim)

    def __call__(self, x):
        axes = tuple(range(x.data.ndim - 1))   # (0,) for 2D, (0, 1) for 3D
        if self.training:
            out = x.batch_norm(self.gamma, self.beta, self.eps, axes)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * x.data.mean(axes)
            self.running_var  = (1-self.momentum) * self.running_var  + self.momentum * x.data.var(axes, ddof=1)
        else:
            xhat = (x - self.running_mean) / xp.sqrt(self.running_var + self.eps)
            out = self.gamma * xhat + self.beta
        return out

    def parameters(self):
        return [self.gamma, self.beta]

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))

    def __call__(self, x):
        return x.layer_norm(self.gamma, self.beta, self.eps)
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        out = x.tanh()
        return out
    def parameters(self):
        return []

class ReLU:
    def __call__(self, x):
        return x.relu()
    def parameters(self):
        return []

class Dropout:
    training = True

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x):
        if not Dropout.training or self.p == 0:
            return x
        mask = (xp.random.rand(*x.shape) > self.p) / (1 - self.p)
        return x * mask
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        out = self.weight[IX]
        return out

    def parameters(self):
        return [self.weight]

class Flatten:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        
        out = x
        return out

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

class Adam:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [xp.zeros_like(p.data) for p in parameters]
        self.v = [xp.zeros_like(p.data) for p in parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        self.t += 1
        c1 = 1 - self.beta1 ** self.t   # bias corrections
        c2 = 1 - self.beta2 ** self.t
        for p, m, v in zip(self.parameters, self.m, self.v):
            if p.grad is None:          # param wasn't used in this forward pass
                continue
            g = p.grad
            m *= self.beta1; m += (1 - self.beta1) * g        # update in place so self.m keeps it
            v *= self.beta2; v += (1 - self.beta2) * g * g
            p.data -= self.lr * (m / c1) / (xp.sqrt(v / c2) + self.eps)