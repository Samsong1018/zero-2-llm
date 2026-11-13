import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from tensor import Tensor, no_grad
from collections import deque
import nn
import numpy as np

EMBEDDING_DIMENSIONS = 20
HIDDEN_LAYER = 512
BATCH_SIZE=256
LR = 1e-3
CHUNK_SIZE = 10000
STEPS_PER_DEPTH = [0, 2000, 2000, 3000, 5000, 20000, 40000, 80000, 110000, 130000, 130000, 130000]
MAX_STEPS = 20
TOTAL_TESTS = 1000


# Data set

SOLVED = np.repeat(np.arange(6), 4).astype(np.int8)

# Right
R = np.arange(24)
R[4], R[5], R[7], R[6] = 6, 4, 5, 7
R[1], R[3] = 9, 11
R[22], R[20] = 1, 3
R[15], R[13] = 20, 22
R[11], R[9] = 15, 13

Rprime = np.argsort(R)
R2 = R[R]

# Front

F = np.arange(24)
F[8], F[9], F[11], F[10] = 10, 8, 9, 11
F[4], F[6] = 2, 3
F[13], F[12] = 4, 6
F[19], F[17] = 13, 12
F[2], F[3] = 19, 17

Fprime = np.argsort(F)
F2 = F[F]

# Up
U = np.arange(24)
U[0], U[1], U[3], U[2] = 2, 0, 1, 3
U[4], U[5] = 20, 21
U[8], U[9] = 4, 5
U[16], U[17] = 8, 9
U[20], U[21] = 16, 17

Uprime = np.argsort(U)
U2 = U[U]

# index = move label for the network, 9 = STOP
MOVES = [R, Rprime, R2, U, Uprime, U2, F, Fprime, F2]
MOVE_NAMES = ["R", "R'", "R2", "U", "U'", "U2", "F", "F'", "F2"]

# Build out all possible permutations as dataset

# TODO save dataset to optomize
INV = [1, 0, 2, 4, 3, 5, 7, 6, 8]   # undo move for each index in MOVES
STOP = 9

seen = {SOLVED.tobytes(): (0, STOP)}   # state -> (distance, label)
q = deque([SOLVED]); counts = {}
while q:
    x = q.popleft(); d, _ = seen[x.tobytes()]; counts[d] = counts.get(d, 0) + 1
    for i, p in enumerate(MOVES):
        y = x[p]; k = y.tobytes()
        if k not in seen:
            seen[k] = (d + 1, INV[i]); q.append(y)
print(len(seen), [counts[i] for i in sorted(counts)])

keys = list(seen.keys())
X = np.frombuffer(b"".join(keys), dtype=SOLVED.dtype).reshape(-1, 24)  # (N, 24) colors 0-5
Y = np.array([seen[k][1] for k in keys])                                # (N,) labels 0-9
D = np.array([seen[k][0] for k in keys])

perm = np.random.permutation(len(X))
X, Y, D = X[perm], Y[perm], D[perm]
n1 = int(0.8*len(X))
n2 = int(0.9*len(X))

Xtr, Xdev, Xte = X[:n1], X[n1:n2], X[n2:]
Ytr, Ydev, Yte = Y[:n1], Y[n1:n2], Y[n2:]
Dtr, Ddev, Dte = D[:n1], D[n1:n2], D[n2:]

BY_DEPTH = [np.where(Dtr == d)[0] for d in range(12)]

# Model init

model = nn.Sequential([
    nn.Embedding(6, EMBEDDING_DIMENSIONS), nn.Flatten(24),
    nn.Linear(24*EMBEDDING_DIMENSIONS, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, 10)
])

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
