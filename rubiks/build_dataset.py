import os, time
from collections import deque
import numpy as np
from cube import SOLVED, MOVES, INV, STOP

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset.npz")
SEED = 42

t0 = time.time()
seen = {SOLVED.tobytes(): (0, STOP)}   # state -> (distance, label)
q = deque([SOLVED])
while q:
    x = q.popleft()
    d, _ = seen[x.tobytes()]
    for i, p in enumerate(MOVES):
        y = x[p]
        k = y.tobytes()
        if k not in seen:
            seen[k] = (d + 1, INV[i])
            q.append(y)
print(f"BFS: {len(seen)} states in {time.time() - t0:.0f}s")

keys = list(seen.keys())
X = np.frombuffer(b"".join(keys), dtype=SOLVED.dtype).reshape(-1, 24)
Y = np.array([seen[k][1] for k in keys], dtype=np.int8)
D = np.array([seen[k][0] for k in keys], dtype=np.int8)

perm = np.random.default_rng(SEED).permutation(len(X))
X, Y, D = X[perm], Y[perm], D[perm]

# sanity checks before saving
assert len(X) == 3674160
assert np.bincount(D).tolist() == [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]
for j in np.random.default_rng(0).choice(len(X), 2000):
    if Y[j] != STOP:
        assert seen[X[j][MOVES[Y[j]]].tobytes()][0] == D[j] - 1

np.savez_compressed(OUT, X=X, Y=Y, D=D)
print(f"saved {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB) in {time.time() - t0:.0f}s total")
