import json, sys, os, string
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from tensor import Tensor
from mlp import MLP
import numpy as np

LR = 2.0
BATCH = 32
EPOCHS = 20

# "." = 0 (start/end token), "a" = 1 ... "z" = 26
index = {".": 0}
for i, ch in enumerate(string.ascii_lowercase, start=1):
    index[ch] = i
itos = {i: ch for ch, i in index.items()}  # reverse lookup, for sampling later

x_train = []
y_train = []
words = open(os.path.join(HERE, "names.txt"), 'r').read().splitlines()

# Create Bigram dataset based of names.txt
for w in words:
    s = "." + w + "."
    for ch1, ch2 in zip(s, s[1:]):
        x_train.append(index[ch1])
        y_train.append(index[ch2])

x_train = np.eye(27)[x_train]
y_train = np.array(y_train)
Y_train = np.eye(27)[y_train]

# Training Regime

def save_weights(model, path=os.path.join(HERE, "weightsSimplified.json")):
    weights = {}
    for n, layer in enumerate(model.layers, start=1):
        weights[f"W{n}"] = layer.W.data.tolist()
        weights[f"b{n}"] = layer.b.data.tolist()
    with open(path, "w") as f:
        json.dump(weights, f)

def train(model):
    params = model.parameters()
    for epoch in range(EPOCHS):
        lr = LR * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
        total_cost, correct, seen = 0.0, 0, 0

        order = np.random.permutation(len(x_train))
        for step, s in enumerate(range(0, len(order), BATCH), start=1):
            idx = order[s:s + BATCH]
            X = Tensor(x_train[idx])
            Y = Tensor(Y_train[idx])

            out = model(X)
            loss = ((out - Y) ** 2).sum() * (1/len(idx))

            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= lr * p.grad
            
            total_cost += loss.data * len(idx)
            seen += len(idx)
            correct += (np.argmax(out.data, axis=1) == y_train[idx]).sum()

            if step % 300 == 0:
                print(f'{total_cost / seen} | {correct / seen}')
        
        print(f"epoch {epoch} done, avg cost {total_cost / seen:.4f}")
        save_weights(model)

model = MLP(27, [27])
train(model)
