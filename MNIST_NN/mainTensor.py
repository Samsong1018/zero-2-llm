import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from mlp import MLP
from tensor import Tensor

LR = 2.0
BATCH = 32
EPOCHS = 100

data = np.load(os.path.join(HERE, "mnist_24.npz"))

x_train = data["x_train"].reshape(-1, 576) / 255.0
y_train = data["y_train"]
Y_train = np.eye(10)[y_train]
x_test = data["x_test"].reshape(-1, 576) / 255.0
y_test = data["y_test"]

def save_weights(model, path=os.path.join(HERE, "weightsTensor.json")):
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

def evaluate(model):
    pred = np.argmax(model(Tensor(x_test)).data, axis=1)
    print(f"test acc {(pred == y_test).mean():.1%} on {len(y_test)} images")


model = MLP(576, [64, 64, 10])
train(model)
evaluate(model)