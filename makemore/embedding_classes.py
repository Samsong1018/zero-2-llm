import json, sys, os, string, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from tensor import Tensor, no_grad
import nn
import numpy as np
from datetime import date

BLOCK_SIZE = 3 # Context window (how many characters are considered when deciding next)
LR=0.4
BATCH_SIZE = 256 # Batch size for training
CHUNK_SIZE = 10000 # batch size for evaluation
STEPS = 50000
HIDDEN_LAYER = 100
EMBEDDING_DIMENSIONS = 10

words = open(os.path.join(HERE, "names.txt"), 'r').read().splitlines()

# Build the vocabulary of characters and mappings to/from intgers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

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

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# Define layers

C = Tensor.randn((vocab_size, EMBEDDING_DIMENSIONS))
layers = [
    nn.Linear(EMBEDDING_DIMENSIONS*BLOCK_SIZE, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER, bias=False), nn.BatchNorm1d(HIDDEN_LAYER), nn.Tanh(),
    nn.Linear(HIDDEN_LAYER, vocab_size)
]

with no_grad():
    layers[-1].weight.data *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, nn.Linear):
            layer.weight.data *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.data.size for p in parameters))

# Training

for i in range(STEPS):

    lr = LR if i < STEPS/2 else LR*0.1

    #minibatch construct

    ix = np.random.randint(0, Xtr.shape[0], (BATCH_SIZE, ))
    Xb, Yb = Xtr[ix], Ytr[ix]

    #forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = Tensor.cross_entropy(x, Yb)

    # backward pass
    for p in parameters:
        p.zero_grad()
    loss.backward()

    # update

    for p in parameters:
        p.data -= lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{STEPS:7d}: {loss.data:.4f}')

# Evaluate

def evaluate(X, Y):
    with no_grad():

        total = 0.0
        for start in range(0, len(X), CHUNK_SIZE):
            slicex = X[start:start+CHUNK_SIZE]
            slicey = Y[start:start+CHUNK_SIZE]

            emb = C[slicex]
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
            loss = Tensor.cross_entropy(x, slicey)

            total += (loss.data.item() * len(slicex))
        
    
        return (total / len(X))

word_set = set(words[:n1])

for layer in layers:
    layer.training = False

with open(os.path.join(HERE, "embedding_output.txt"), 'a') as file:

    file.write(f'\n\n {date.today()}\n')
    file.write("Final Version (uses classes)\n")
    file.write(f'Current Settings:\nBlock Size: {BLOCK_SIZE}\n')
    file.write(f'Batch size: {BATCH_SIZE}\nSteps: {STEPS}\n')
    file.write(f'Hidden Layer: {HIDDEN_LAYER}\nEmbedding Dimension: {EMBEDDING_DIMENSIONS}\n\n')

    with no_grad():
        loss = evaluate(Xtr, Ytr)

        print(f'Training loss: {loss}')
        file.write(f'Training loss: {loss}\n')

        loss = evaluate(Xdev, Ydev)

        print(f'Dev loss: {loss}')
        file.write(f'Dev loss: {loss}\n')

        loss = evaluate(Xte, Yte)

        print(f'Test loss: {loss}')
        file.write(f'Test loss: {loss}\n\n')

        print("20 Generated Names: ")
        file.write("20 Generated Names: \n")

        for _ in range(20):
            out = []
            context = [0] * BLOCK_SIZE
            while True:
                emb = C[np.array([context])]
                x = emb.view(emb.shape[0], -1)
                for layer in layers:
                    x = layer(x)
                logits = x.data

                counts = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = counts / counts.sum(axis=1, keepdims=True)

                ix = np.random.choice(vocab_size, p=probs[0])
                context = context[1:] + [ix]
                if ix == 0:
                    break
                out.append(itos[ix])
        
            print(''.join(out))
            file.write(f"{''.join(out)}")
            if ''.join(out) in word_set:
                file.write("*\n")
            else:
                file.write("\n")