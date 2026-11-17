"""
Run this file to sample my trained weights!
Needs gpt_weights.npz, which gpt_shakespear.py saves while it trains.
"""

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
# "cpu" or "gpu"; must be set before tensor/nn are imported. MLP_DEVICE=cpu in the shell overrides it.
os.environ.setdefault("MLP_DEVICE", "cpu")
from tensor import Tensor, no_grad
from backend import xp, to_numpy, DEVICE
import numpy as np
import nn

# must match the settings gpt_shakespear.py trained with, or the weights won't fit
BLOCK_SIZE = 256
EMBEDDING_DIMENSIONS = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.0 # only matters for training
MAX_NEW_TOKENS = 500
WEIGHTS_FILE = os.path.join(HERE, "gpt_weights.npz")

# the vocab comes from the training text, so the ids line up with the weights
with open(os.path.join(HERE, "input.txt"), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder string to int
decode = lambda l: ''.join([itos[i] for i in l]) # decoder integers to string

class Head:
    def __init__(self, n_emb, head_size, block_size, dropout):
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.tril = xp.tril(xp.ones((block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        T = x.shape[1]
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei + xp.where(self.tril[:T, :T] == 0, -xp.inf, 0.0)
        wei = wei.softmax()
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

    def parameters(self):
        return self.key.parameters() + self.query.parameters() + self.value.parameters()

class MultiHeadAttention:
    def __init__(self, n_emb, num_heads, head_size, block_size, dropout):
        self.heads = [Head(n_emb, head_size, block_size, dropout) for _ in range(num_heads)]
        self.proj = nn.Linear(num_heads * head_size, n_emb)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = Tensor.cat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out

    def parameters(self):
        return [p for head in self.heads for p in head.parameters()] + self.proj.parameters()

class FeedForward:
    def __init__(self, n_emb, dropout):
        self.net = nn.Sequential([
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb),
            nn.Dropout(dropout)
            ])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()

class Block:
    def __init__(self, n_emb, num_heads, block_size, dropout):
        head_size = n_emb // num_heads
        self.sa = MultiHeadAttention(n_emb, num_heads, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_emb, dropout)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def parameters(self):
        return self.sa.parameters() + self.ffwd.parameters() + self.ln1.parameters() + self.ln2.parameters()

class GPTLanguageModel:
    def __init__(self, vocab_size, n_emb, n_head, n_layer, block_size, dropout):
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential([Block(n_emb, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def __call__(self, idx):
        return self.forward(idx)

    def forward(self, idx):
        T = idx.shape[1]

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(np.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def parameters(self):
        layers = [self.token_embedding_table, self.position_embedding_table, self.lm_head, self.blocks, self.ln_f]
        return [p for layer in layers for p in layer.parameters()]

    def generate(self, idx, max_new_tokens):
        idx = xp.asarray(idx)
        with no_grad():
            nn.Dropout.training = False
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits = self(idx_cond)
                logits = logits[:, -1, :]
                probs = logits.softmax()
                idx_next = (probs.data.cumsum(1) > xp.random.rand(idx.shape[0], 1)).argmax(1, keepdims=True)
                idx = xp.concatenate((idx, idx_next), axis=1)
            nn.Dropout.training = True
        return to_numpy(idx)

def load_weights(model):
    # saved in model.parameters() order, so the shapes have to line up one for one
    if not os.path.exists(WEIGHTS_FILE):
        sys.exit(f"no weights at {WEIGHTS_FILE}, run gpt_shakespear.py first")
    saved = np.load(WEIGHTS_FILE)
    params = model.parameters()
    if len(params) != len(saved.files):
        sys.exit(f"weights file has {len(saved.files)} tensors but the model has {len(params)}, do the settings match?")
    for p, key in zip(params, saved.files):
        w = saved[key]
        if p.data.shape != w.shape:
            sys.exit(f"shape mismatch on {key}: model {p.data.shape}, file {w.shape}, do the settings match?")
        p.data = xp.asarray(w)

model = GPTLanguageModel(vocab_size, EMBEDDING_DIMENSIONS, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT)
load_weights(model)
print(f'Device: {DEVICE}, parameters: {sum(p.data.size for p in model.parameters())}\n')

context = np.zeros((1,1), dtype=int)
print(decode(model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()))
