import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import nn

# model shape -- must match whatever weights.json was trained with
EMBEDDING_DIMENSIONS = 20
HIDDEN_LAYER = 512

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

MOVES = [R, Rprime, R2, U, Uprime, U2, F, Fprime, F2]
MOVE_NAMES = ["R", "R'", "R2", "U", "U'", "U2", "F", "F'", "F2"]

INV = [1, 0, 2, 4, 3, 5, 7, 6, 8]   # undo move for each index in MOVES
STOP = 9

def build_model():
    return nn.Sequential([
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