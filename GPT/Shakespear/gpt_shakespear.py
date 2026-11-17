import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
# "cpu" or "gpu"; must be set before tensor/nn are imported. MLP_DEVICE=gpu in the shell overrides it.
os.environ.setdefault("MLP_DEVICE", "gpu")
from tensor import Tensor, no_grad
from backend import xp, to_numpy, DEVICE
import numpy as np
import nn

# constants

BLOCK_SIZE = 256
BATCH_SIZE = 16
MAX_ITERS = 15000
EVAL_INTERVAL = 1000
LR = 3e-4
EVAL_ITERS = 100
EMBEDDING_DIMENSIONS = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
TRAIN_SPLIT = 0.9
MAX_NEW_TOKENS = 500
LOG_FILE = os.path.join(HERE, "training_log.txt")
WEIGHTS_FILE = os.path.join(HERE, "gpt_weights.npz")
PATIENCE = 3
MIN_DELTA = 0.002

best_val = float('inf')
bad_evals = 0

# load dataset
with open(os.path.join(HERE, "input.txt"), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder string to int
decode = lambda l: ''.join([itos[i] for i in l]) # decoder integers to string

data = np.array(encode(text))

# split to training dataset

n = int(TRAIN_SPLIT*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    split_data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(split_data) - BLOCK_SIZE, size=BATCH_SIZE)
    x = np.stack([split_data[i:i+BLOCK_SIZE] for i in ix])
    y = np.stack([split_data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

def estimate_loss():
    with no_grad():
        out = {}
        nn.Dropout.training = False
        for split in ['train', 'val']:
            losses = np.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        nn.Dropout.training = True
        return out

def log(line):
    # print and append to the log file, so the run survives a closed terminal
    print(line)
    with open(LOG_FILE, "a") as file:
        file.write(line + "\n")

def save_weights():
    # saved in model.parameters() order
    np.savez(WEIGHTS_FILE, *[to_numpy(p.data) for p in model.parameters()])

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

    def __call__(self, idx, targets=None):
        return self.forward(idx, targets)

    def forward(self, idx, targets=None):
        T = idx.shape[1]

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(np.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.reshape(B*T)
        loss = logits.cross_entropy(targets)

        return logits, loss

    def parameters(self):
        layers = [self.token_embedding_table, self.position_embedding_table, self.lm_head, self.blocks, self.ln_f]
        return [p for layer in layers for p in layer.parameters()]

    def generate(self, idx, max_new_tokens):
        idx = xp.asarray(idx)
        with no_grad():
            nn.Dropout.training = False
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = logits.softmax()
                idx_next = (probs.data.cumsum(1) > xp.random.rand(idx.shape[0], 1)).argmax(1, keepdims=True)
                idx = xp.concatenate((idx, idx_next), axis=1)
            nn.Dropout.training = True
        return to_numpy(idx)

model = GPTLanguageModel(vocab_size, EMBEDDING_DIMENSIONS, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT)

optimizer = nn.Adam(model.parameters(), lr=LR)

log(f'Device: {DEVICE}, parameters: {sum(p.data.size for p in model.parameters())}')

def load_weights():
    saved = np.load(WEIGHTS_FILE)
    for p, key in zip(model.parameters(), saved.files):
        p.data = xp.asarray(saved[key])

def evaluate(step):
    # log losses and checkpoint whenever val loss hits a new best
    global best_val, bad_evals
    losses = estimate_loss()
    log(f'step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    if losses['val'] < best_val - MIN_DELTA:
        best_val = losses['val']
        bad_evals = 0
        save_weights()
    else:
        bad_evals += 1
    return bad_evals >= PATIENCE

MIN_LR = LR / 10
def get_lr(step):
    progress = step / MAX_ITERS
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))

for step in range(MAX_ITERS):
    optimizer.lr = get_lr(step)

    if step % EVAL_INTERVAL == 0 and evaluate(step):
        log(f'early stop at step {step}, best val {best_val:.4f}')
        break

    xb, yb = get_batch('train')

    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

evaluate(MAX_ITERS)
load_weights()

context = np.zeros((1,1), dtype=int)
print(decode(model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()))
with open(os.path.join(HERE, "output.txt"), "a") as f:
    f.write(decode(model.generate(context, max_new_tokens=(MAX_NEW_TOKENS*7))[0].tolist()))

# To qoute an earlier training iteration: To withll your hangell with
