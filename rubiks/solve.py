import cube, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import nn
from tensor import no_grad

model = cube.build_model()
data = json.load(open(os.path.join(HERE, "weights.json")))

for p, w in zip(model.parameters(), data["params"]):
    p.data = np.array(w, dtype=np.float32)

bn_layers = [l for l in model.layers if isinstance(l, nn.BatchNorm1d)]
for layer, (mean, var) in zip(bn_layers, data["bn"]):
    layer.running_mean = np.array(mean)
    layer.running_var = np.array(var)

for l in model.layers:
    l.training = False

def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def beam_solve(s, K=50, max_depth=20):
    with no_grad():
        beam = [(s, 0.0, [])]                       # (state, score, path of move indices)
        for depth in range(max_depth):
            for st, sc, path in beam:
                if np.array_equal(st, cube.SOLVED):
                    return path

            S = np.stack([b[0] for b in beam])      # (B, 24): the whole beam in ONE forward pass
            logp = log_softmax(model(S).data)       # (B, 10)
            logp = logp[:, :9]                      # drop STOP

            cands = {}                              # state bytes -> (score, state, path)
            for b, (st, sc, path) in enumerate(beam):
                for m in range(9):
                    if path and m == cube.INV[path[-1]]:
                        continue                    # no undo
                    child = st[cube.MOVES[m]]
                    score = sc + logp[b, m]
                    k = child.tobytes()
                    if k not in cands or score > cands[k][0]:
                        cands[k] = (score, child, path + [m])

            best = sorted(cands.values(), key=lambda c: c[0], reverse=True)[:K]
            beam = [(st, sc, path) for sc, st, path in best]
        return None  

dataset = np.load(os.path.join(HERE, "dataset.npz"))
X, Y, D = dataset["X"], dataset["Y"], dataset["D"]

n2 = int(0.9*len(X))
Xte, Yte, Dte = X[n2:], Y[n2:], D[n2:]

tests = [1, 10, 50, 200]

for K in tests:
    print(f'K: {K}')
    for d in range(1, 12):
        idx = np.random.choice(np.where(Dte == d)[0], 100)    # 100 test cubes at depth d
        results = [beam_solve(Xte[j], K) for j in idx]
        solved = [r for r in results if r is not None]
        solve_rate = len(solved) / 100
        print(f'Results: {solve_rate}')
        # extra moves = average len(r) - d over the solved ones
