import json, sys, os
from cube import SOLVED, R, Rprime, R2, F, Fprime, F2, U, Uprime, U2, MOVES, MOVE_NAMES, INV, STOP, build_model
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from tensor import Tensor, no_grad
from collections import deque
import nn
import numpy as np

BATCH_SIZE=256
LR = 1e-3
CHUNK_SIZE = 10000
STEPS_PER_DEPTH = [0, 2000, 2000, 3000, 5000, 20000, 40000, 80000, 110000, 130000, 130000, 130000]
MAX_STEPS = 20
TOTAL_TESTS = 1000


# Data set

# Build out all possible permutations as dataset

data = np.load(os.path.join(HERE, "dataset.npz"))
X, Y, D = data["X"], data["Y"], data["D"]

n1 = int(0.8*len(X))
n2 = int(0.9*len(X))

Xtr, Xdev, Xte = X[:n1], X[n1:n2], X[n2:]
Ytr, Ydev, Yte = Y[:n1], Y[n1:n2], Y[n2:]
Dtr, Ddev, Dte = D[:n1], D[n1:n2], D[n2:]

BY_DEPTH = [np.where(Dtr == d)[0] for d in range(12)]

# Model init

model = build_model()

with no_grad():
    model.layers[-1].weight.data *= 0.1

parameters = model.parameters()
optimizer = nn.Adam(parameters, lr=LR)
print(sum(p.data.size for p in parameters))
with open(os.path.join(HERE, "training_log.txt"), "a") as file:
    file.write(f'{sum(p.data.size for p in parameters)}\n')

# Training

def save_weights():
    # BatchNorm running stats aren't parameters (no gradient), but eval mode uses them,
    # so save one [mean, var] pair per BatchNorm1d layer, in layer order
    bn = [[layer.running_mean.tolist(), layer.running_var.tolist()] for layer in model.layers if isinstance(layer, nn.BatchNorm1d)]
    with open(os.path.join(HERE, "weights.json"), "w") as file:
        weights = {"params": [param.data.tolist() for param in parameters], "bn": bn}
        json.dump(weights, file)

def solve(s):
    # greedy solve: returns True if solved within MAX_STEPS
    last = None
    for step in range(MAX_STEPS):
        if np.array_equal(s, SOLVED):
            return True
        logits = model(s[None]).data
        logits[0, STOP] = -np.inf                   # never pick STOP
        if last is not None:
            logits[0, INV[last]] = -np.inf          # never undo the previous move
        last = logits.argmax(1)[0]
        s = s[MOVES[last]]
    return np.array_equal(s, SOLVED)

def solve_rate(pool):
    # pool = row indices into Xte to draw scrambles from
    if len(pool) == 0:
        return None
    picks = np.random.choice(pool, TOTAL_TESTS)
    with no_grad():
        return sum(solve(Xte[j]) for j in picks) / TOTAL_TESTS

import time
T0 = time.time()

def log(msg):
    # print and append to the log with elapsed hours, so you can check progress in the morning
    line = f'[{(time.time() - T0) / 3600:5.2f}h] {msg}'
    print(line, flush=True)
    with open(os.path.join(HERE, "training_log.txt"), "a") as file:
        file.write(line + '\n')

for max_d in range(1, 12):
    for layer in model.layers:
        layer.training = True
    steps = STEPS_PER_DEPTH[max_d]
    for i in range(steps):
        lr = LR if i < steps * 2 // 3 else LR / 4
        optimizer.lr = lr
        #minibatch construct

        ds = np.random.randint(1, max_d + 1, BATCH_SIZE)
        ix = np.array([np.random.choice(BY_DEPTH[d]) for d in ds])
        Xb, Yb = Xtr[ix], Ytr[ix]

        #forward pass
        logits = model(Xb)
        loss = Tensor.cross_entropy(logits, Yb)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update
        optimizer.step()

        if i % 10000 == 0:
            log(f'depth<={max_d} step {i:6d}/{steps}: loss {float(loss.data):.4f} | LR: {lr}')

    # Eval on each stage

    for layer in model.layers:
        layer.training = False

    # solve rate on test scrambles exactly max_d moves from solved
    rate = solve_rate(np.where(Dte == max_d)[0])
    log(f'DONE depth<={max_d}: loss {float(loss.data):.4f} | Success rate @ depth {max_d}: {rate}')
    save_weights()

# Eval

for layer in model.layers:
    layer.training = False

def evaluate(X, Y):
    with no_grad():
        total = 0.0
        totalacc = 0.0
        for start in range(0, len(X), CHUNK_SIZE):
            slicex = X[start:start+CHUNK_SIZE]
            slicey = Y[start:start+CHUNK_SIZE]

            logits = model(slicex)
            loss = Tensor.cross_entropy(logits, slicey)

            totalacc += (logits.data.argmax(1) == slicey).sum()
            total += (loss.data.item() * len(slicex))

        print(f'Accuracy: {totalacc / len(X)}')
        return (total / len(X))

with no_grad():
    loss = evaluate(Xtr, Ytr)

    print(f'Training loss: {loss}')

    loss = evaluate(Xdev, Ydev)

    print(f'Dev loss: {loss}')

    loss = evaluate(Xte, Yte)

    print(f'Test loss: {loss}')

# final solve rate on random test scrambles of any depth, then broken down by depth
log(f'FINAL success rate (all depths): {solve_rate(np.where(Dte > 0)[0])}')
for d in range(1, 12):
    log(f'  depth {d:2d}: {solve_rate(np.where(Dte == d)[0])}')
