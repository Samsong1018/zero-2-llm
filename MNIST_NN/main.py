import json, sys, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from mlp import MLP

LR = 0.5
EPOCHS = 20
TRAIN_SIZE = 1000
TEST_SIZE = 200

data = np.load(os.path.join(HERE, "mnist_24.npz"))

x_train = (data["x_train"].reshape(-1, 576) / 255.0).tolist()
y_train = data["y_train"]
x_test = (data["x_test"].reshape(-1, 576) / 255.0).tolist()
y_test = data["y_test"]

def one_hot(label):
    return [1.0 if j == label else 0.0 for j in range(10)]

def guess(out):
    return max(range(10), key=lambda j: out[j].data)

def save_weights(model, path=os.path.join(HERE, "weights.json")):
    weights = {}
    for n, layer in enumerate(model.layers, start=1):
        nin = len(layer.neurons[0].w)
        weights[f"W{n}"] = [[neuron.w[i].data for neuron in layer.neurons] for i in range(nin)]
        weights[f"b{n}"] = [neuron.b.data for neuron in layer.neurons]
    with open(path, "w") as f:
        json.dump(weights, f)

def train(model):
    params = model.parameters()
    for epoch in range(EPOCHS):
        order = random.sample(range(len(x_train)), TRAIN_SIZE)
        total_cost, correct = 0.0, 0

        for step, idx in enumerate(order, start=1):
            x, y = x_train[idx], y_train[idx]

            out = model(x)
            loss = sum((o - t) ** 2 for o, t in zip(out, one_hot(y)))

            for p in params:
                p.grad = 0.0
            loss.backward()

            for p in params:
                p.data -= LR * p.grad

            total_cost += loss.data
            correct += guess(out) == y
            if step % 100 == 0:
                print(f" step {step} avg cost {total_cost/step:.4f} train acc {correct / step:.1%}")

        save_weights(model)
        print(f"epoch {epoch} avg cost {total_cost / TRAIN_SIZE:.4f}")

def evaluate(model):
    correct = sum(guess(model(x)) == y for x, y in zip(x_test[:TEST_SIZE], y_test[:TEST_SIZE]))
    print(f"test acc {correct / TEST_SIZE:.1%} on {TEST_SIZE} images")

model = MLP(576, [64, 64, 10], activation="sig")
train(model)
evaluate(model)