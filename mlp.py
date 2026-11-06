from tensor import Tensor
import numpy as np

class Layer:
    def __init__(self, nin, nout, activation=True):
        self.W = Tensor(np.random.randn(nin, nout) * (1/nin) ** 0.5)
        self.activation = activation
        self.b = Tensor(np.zeros(nout))
        
    def __call__(self, x):
        out = x @ self.W + self.b
        return out.sigmoid() if self.activation else out

    def parameters(self):
        return [self.W, self.b]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], activation=True) for i in range(len(nouts))
            ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
