"""
This is for part three of makemore which hopefully means it's a little better
"""

import json, sys, os, string, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from tensor import Tensor
from mlp import MLP
import numpy as np
from datetime import date

BLOCK_SIZE = 3 # Context window (how many characters are considered when deciding next)
LR=0.2
BATCH_SIZE = 32
STEPS = 200000
HIDDEN_LAYER = 200
EMBEDDING_DIMENSIONS = 10

words = open(os.path.join(HERE, "names.txt"), 'r').read().splitlines()

# Build the vocabulary of characters and mappings to/from intgers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# compile dataset (train, dev, test)

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * BLOCK_SIZE
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = Tensor(X)
    Y = Tensor(Y)

    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

C = Tensor.randn((27, EMBEDDING_DIMENSIONS))

W1 = Tensor.randn((EMBEDDING_DIMENSIONS*BLOCK_SIZE, HIDDEN_LAYER)) * (5/3)/(EMBEDDING_DIMENSIONS*BLOCK_SIZE)**0.5
#b1 = Tensor.randn(HIDDEN_LAYER) * 0.01 Because of normalization b1 gets subtracted and creates a gradient of zero
W2 = Tensor.randn((HIDDEN_LAYER, 27)) * 0.1
b2 = Tensor.randn(27) * 0

bngain = Tensor(np.ones((1, HIDDEN_LAYER)))
bnbias = Tensor(np.zeros((1, HIDDEN_LAYER)))
bnmean_running = np.zeros((1, HIDDEN_LAYER))
bnstd_running = np.ones((1, HIDDEN_LAYER))

parameters = [W1, W2, b2, C, bngain, bnbias]

#lre = np.linspace(-3, -1, STEPS)
#lrs = 10**lre

lr = LR

for i in range(STEPS):
    lr = 0.1 if i < 100000 else 0.01

    # define mini batch
    ix = np.random.randint(0, Xtr.shape[0], (BATCH_SIZE, ))

    # forward pass
    emb = C[Xtr[ix]]
    embcat = emb.view((-1, EMBEDDING_DIMENSIONS*BLOCK_SIZE)) @ W1# + b1
    bnmeani = embcat.mean(0, keepdims=True)
    bnstdi = embcat.std(0, keepdims=True)
    hpreact = bngain * (embcat - bnmeani) / bnstdi + bnbias

    bnmean_running = 0.999*bnmean_running + 0.001*bnmeani.data
    bnstd_running = 0.999*bnstd_running + 0.001*bnstdi.data

    h = hpreact.tanh()
    logits = h @ W2 + b2
    loss = logits.cross_entropy(Ytr[ix])

    #backward pass and nudge opposite of gradient
    for p in parameters:
        p.zero_grad()

    loss.backward()

    for p in parameters:
        p.data -= lr*p.grad

    if i % 5000 == 0:
        print(loss.data)

# Evaluate

with open('BigramEngine/embedding_output.txt', 'a') as file:

    file.write(f'\n\n {date.today()}\n')
    file.write("Improved Version (part three)\n")
    file.write(f'Current Settings:\nBlock Size: {BLOCK_SIZE}\n')
    file.write(f'Batch size: {BATCH_SIZE}\nSteps: {STEPS}\n')
    file.write(f'Hidden Layer: {HIDDEN_LAYER}\nEmbedding Dimension: {EMBEDDING_DIMENSIONS}\n\n')

    emb = C[Xtr]
    embcat = emb.view((-1, EMBEDDING_DIMENSIONS*BLOCK_SIZE)) @ W1# + b1
    hpreact = bngain * (embcat - bnmean_running) / bnstd_running + bnbias
    h = hpreact.tanh()
    logits = h @ W2 + b2
    loss = logits.cross_entropy(Ytr)

    print(f'Training loss: {loss.data}')
    file.write(f'Training loss: {loss.data}\n')

    emb = C[Xdev]
    embcat = emb.view((-1, EMBEDDING_DIMENSIONS*BLOCK_SIZE)) @ W1# + b1
    hpreact = bngain * (embcat - bnmean_running) / bnstd_running + bnbias
    h = hpreact.tanh()
    logits = h @ W2 + b2
    loss = logits.cross_entropy(Ydev)

    print(f'Dev loss: {loss.data}')
    file.write(f'Dev loss: {loss.data}\n')

    emb = C[Xte]
    embcat = emb.view((-1, EMBEDDING_DIMENSIONS*BLOCK_SIZE)) @ W1# + b1
    hpreact = bngain * (embcat - bnmean_running) / bnstd_running + bnbias
    h = hpreact.tanh()
    logits = h @ W2 + b2
    loss = logits.cross_entropy(Yte)

    print(f'Test loss: {loss.data}')
    file.write(f'Test loss: {loss.data}\n\n')

    print("20 Generated Names: ")
    file.write("20 Generated Names: \n")

    for _ in range(20):
        out = []
        context = [0] * BLOCK_SIZE
        while True:
            emb = C[np.array([context])]
            embcat = emb.view((-1, EMBEDDING_DIMENSIONS*BLOCK_SIZE)) @ W1# + b1
            hpreact = bngain * (embcat - bnmean_running) / bnstd_running + bnbias
            h = hpreact.tanh()
            logits = (h @ W2 + b2).data

            counts = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = counts / counts.sum(axis=1, keepdims=True)

            ix = np.random.choice(27, p=probs[0])
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        
        print(''.join(out))
        file.write(f"{''.join(out)}\n")