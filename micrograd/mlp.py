from value import Value
import random

class Neuron:
    def __init__ (self, nin, activation="tanh"):
        self.w = [Value(random.gauss(0, (1 / nin) ** 0.5)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        if self.activation == "tanh":
            out = act.tanh()
        elif self.activation == "sig":
            out = act.sigmoid()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, activation="tanh"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]

        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts, activation="tanh"):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], activation) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]