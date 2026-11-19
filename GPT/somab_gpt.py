from dataclasses import dataclass
import os, time, math, glob, threading
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))

def apply_rotary_emb(x, cos, sin):
    # x is (B, nh, T, hs), rotate the two halves of each head by the position angle
    hs = x.size(-1)
    x1, x2 = x[..., :hs//2], x[..., hs//2:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        # q gets n_head heads, k and v only get n_kv_head heads each (GQA)
        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.head_size, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_norm = nn.RMSNorm(self.head_size)
        self.k_norm = nn.RMSNorm(self.head_size)
    
    def forward(self, x, cos, sin):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.head_size, self.n_kv_head * self.head_size, self.n_kv_head * self.head_size], dim=2)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # each k/v head is shared by n_head // n_kv_head query heads
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU, c_fc makes the gate and the up projection in one matmul
        self.c_fc = nn.Linear(config.n_embd, 2 * config.mlp_hidden, bias=False)
        self.c_proj = nn.Linear(config.mlp_hidden, config.n_embd, bias=False)
    
    def forward(self, x):
        gate, up = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 49152 # SmolLM2 tokenizer, already a multiple of 64
    n_layer: int = 26
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 1536
    mlp_hidden: int = 4096
    rope_base: float = 10000.0
    logit_softcap: float = 30.0

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.RMSNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # RoPE angles replace wpe, precomputed once for every position up to block_size
        head_size = config.n_embd // config.n_head
        inv_freq = 1.0 / (config.rope_base ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        angles = torch.outer(torch.arange(config.block_size, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

        # recompute each block's activations during backward instead of storing them. ~30% slower, way less memory
        self.grad_checkpoint = False

        self.apply(self._init_weights)
        # the c_proj layers add into the residual stream, 2 per block, so they get scaled down
        # so the residual doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        cos = self.cos[:T].to(self.transformer.wte.weight.dtype)
        sin = self.sin[:T].to(self.transformer.wte.weight.dtype)
        x = self.transformer.wte(idx)

        for block in self.transformer.h:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.transformer.ln_f(x)
        # .float() so the soft cap and the loss happen in fp32 even under fp16 autocast
        logits = self.lm_head(x).float()
        # soft cap keeps logits in (-cap, cap) so fp16 can't blow up
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizer(self, weight_decay, lr, use_8bit):
        # weight decay on the matrices and embeddings only, not the norms
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if use_8bit:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
            # bitsandbytes recommends full 32-bit Adam state for the embedding, 8-bit there can be unstable
            bnb.optim.GlobalOptimManager.get_instance().override_config(self.transformer.wte.weight, "optim_bits", 32)
        else:
            optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available())
        return optimizer

class DataLoader:
    """ reads the uint16 shards from gen_dataset.py in order. state() is saved in checkpoints so a new session picks up on the exact next token """

    def __init__(self, pattern, B, T, rank, world_size, state=None):
        self.pattern = pattern # glob for the shard files, e.g. ".../train_*.npy"
        self.B, self.T = B, T
        self.rank, self.world_size = rank, world_size
        self.shard = state["shard"] if state else 0
        self.pos = state["pos"] if state else 0
        self.tokens = None

    def shards(self):
        # looked up again each time, so shards gen_dataset.py makes mid-run get picked up
        return sorted(glob.glob(self.pattern))

    def state(self):
        return {"shard": self.shard, "pos": self.pos}

    def next_batch(self):
        """ returns (x, y), or None once every shard is used up """
        span = self.B * self.T * self.world_size # tokens all ranks use per batch together
        while True:
            shards = self.shards()
            if self.shard >= len(shards):
                return None
            if self.tokens is None:
                # mmap so a 200MB shard isn't copied into RAM
                self.tokens = np.load(shards[self.shard], mmap_mode="r")
            if self.pos + span + 1 <= len(self.tokens):
                break
            # not enough left in this shard for a full batch, the leftover gets skipped
            self.shard += 1
            self.pos = 0
            self.tokens = None

        start = self.pos + self.rank * self.B * self.T
        buf = torch.from_numpy(self.tokens[start:start + self.B * self.T + 1].astype(np.int64))
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.pos += span
        return x, y

# model + batch settings per run. pick one with SOMAB_SIZE=tiny|pilot|full (default full)
#   tiny: for testing on the laptop
#   pilot: ~150M, the practice run on Kaggle before the real one
#   full: the 805M run
SIZES = {
    "tiny": dict(config=dict(block_size=256, n_layer=4, n_head=4, n_kv_head=2, n_embd=256, mlp_hidden=688),
                B=8, total_batch=8 * 256 * 4, max_lr=1e-3, warmup=50),
    "pilot": dict(config=dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768, mlp_hidden=2048),
                B=16, total_batch=524288, max_lr=6e-4, warmup=700),
    "full": dict(config=dict(),
                B=4, total_batch=524288, max_lr=3e-4, warmup=2000),
}

def get_lr(step, max_lr, warmup, decay_start, decay_steps):
    """ WSD: linear warmup, then flat at max_lr until decay_start, then linear down to 0 over decay_steps """
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if decay_start is None or step < decay_start:
        return max_lr
    return max_lr * max(0.0, 1 - (step - decay_start) / decay_steps)

if __name__ == "__main__":
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from tokenizers import Tokenizer

    # ---- run settings, all can be set with environment variables ----
    SIZE = os.environ.get("SOMAB_SIZE", "full")
    SHARD_DIR = os.environ.get("SOMAB_SHARDS", os.path.join(HERE, "datasets", "shards"))
    CKPT_DIR = os.environ.get("SOMAB_CKPT", os.path.join(HERE, "checkpoints", SIZE))
    HF_REPO = os.environ.get("SOMAB_HF_REPO") # e.g. "yourname/somab-checkpoints", leave unset to skip uploading
    SESSION_HOURS = float(os.environ.get("SOMAB_HOURS", 11.5)) # save and stop before Kaggle's 12h kill
    SAVE_EVERY_MIN = float(os.environ.get("SOMAB_SAVE_MIN", 30))
    # set DECAY_START (a step number) when you decide to end the run, the LR goes to 0 over DECAY_STEPS after it
    DECAY_START = int(os.environ["SOMAB_DECAY_START"]) if "SOMAB_DECAY_START" in os.environ else None
    DECAY_STEPS = int(os.environ.get("SOMAB_DECAY_STEPS", 5000))
    VAL_EVERY = 250
    VAL_BATCHES = 20 # per val file
    SAMPLE_EVERY = 2000
    WEIGHT_DECAY = 0.1
    OVERFIT = os.environ.get("SOMAB_OVERFIT") == "1" # train on one batch over and over, loss should go to ~0
    MAX_STEPS = int(os.environ["SOMAB_MAX_STEPS"]) if "SOMAB_MAX_STEPS" in os.environ else None # for test runs

    run = SIZES[SIZE]
    config = GPTConfig(**run["config"])
    B, T = int(os.environ.get("SOMAB_B", run["B"])), config.block_size # SOMAB_B to shrink the micro batch on a small GPU

    # ---- DDP: torchrun sets RANK, LOCAL_RANK and WORLD_SIZE. plain python is one process ----
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, local_rank, world_size = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    master = rank == 0 # only this process logs, samples and saves
    device_type = "cuda" if device.startswith("cuda") else device
    # fp16 because T4s have no bf16. SOMAB_FP16=0 to test in fp32
    use_fp16 = device_type == "cuda" and os.environ.get("SOMAB_FP16", "1") == "1"
    use_8bit = device_type == "cuda" and os.environ.get("SOMAB_8BIT", "1") == "1"

    def log(msg):
        if master:
            print(msg, flush=True)
            with open(os.path.join(CKPT_DIR, "log.txt"), "a") as f:
                f.write(msg + "\n")

    assert run["total_batch"] % (B * T * world_size) == 0, "total_batch has to divide evenly by B * T * world_size"
    grad_accum = run["total_batch"] // (B * T * world_size)
    os.makedirs(CKPT_DIR, exist_ok=True)

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # ---- model + optimizer ----
    model = GPT(config)
    model.grad_checkpoint = SIZE != "tiny"
    model.to(device)
    model.train()
    raw_model = model # the plain module, this is what gets saved
    optimizer = raw_model.configure_optimizer(WEIGHT_DECAY, run["max_lr"], use_8bit)
    scaler = torch.amp.GradScaler(device_type, enabled=use_fp16)

    enc = Tokenizer.from_file(os.path.join(HERE, "tokenizer", "smollm2_tokenizer.json"))

    # ---- checkpoints ----
    def ckpt_paths():
        return sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.pt")))

    upload_thread = None

    def save_weights(step, loader_state):
        ckpt = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "loader": loader_state,
            "config": config.__dict__,
            "size": SIZE,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device_type == "cuda" else None,
        }
        path = os.path.join(CKPT_DIR, f"ckpt_{step:07d}.pt")
        # write to .tmp and rename, so a session dying mid-save can't leave a broken checkpoint
        torch.save(ckpt, path + ".tmp")
        os.replace(path + ".tmp", path)
        for old in ckpt_paths()[:-2]: # keep the last 2
            os.remove(old)
        log(f"saved {os.path.basename(path)}")
        if HF_REPO:
            upload(path)

    def upload(path):
        global upload_thread
        from huggingface_hub import HfApi
        if upload_thread is not None:
            upload_thread.join() # only one upload at a time
        def go():
            try:
                HfApi().upload_file(path_or_fileobj=path, path_in_repo=f"{SIZE}/latest.pt", repo_id=HF_REPO,
                                    commit_message=f"{SIZE} {os.path.basename(path)}")
                log(f"uploaded {os.path.basename(path)} to {HF_REPO}")
            except Exception as e:
                log(f"upload failed, will try again next save: {e}")
        upload_thread = threading.Thread(target=go)
        upload_thread.start()

    def load_weights():
        """ newest local checkpoint, or the one on Hugging Face if there isn't one here. returns step, loader state """
        paths = ckpt_paths()
        path = paths[-1] if paths else None
        if path is None and HF_REPO:
            from huggingface_hub import hf_hub_download
            try:
                path = hf_hub_download(HF_REPO, f"{SIZE}/latest.pt", local_dir=CKPT_DIR)
            except Exception as e:
                log(f"no checkpoint on {HF_REPO} ({type(e).__name__}), starting fresh")
        if path is None:
            return 0, None
        ckpt = torch.load(path, map_location=device, weights_only=False)
        assert ckpt["size"] == SIZE, f"checkpoint is for {ckpt['size']}, not {SIZE}"
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng_cpu"].cpu())
        if ckpt["rng_cuda"] is not None and device_type == "cuda":
            torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu())
        log(f"resumed from {os.path.basename(path)} at step {ckpt['step']}")
        return ckpt["step"], ckpt["loader"]

    step, loader_state = load_weights()
    train_loader = DataLoader(os.path.join(SHARD_DIR, "train_*.npy"), B, T, rank, world_size, loader_state)

    # compile and wrap after loading, so the checkpoint keys stay plain
    if device_type == "cuda" and os.environ.get("SOMAB_COMPILE", "1") == "1":
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    log(f"size {SIZE} | {raw_model.num_params() / 1e6:.1f}M params | device {device} x{world_size} | "
        f"B {B} T {T} grad_accum {grad_accum} | 8-bit adam {use_8bit} | fp16 {use_fp16}")

    # ---- val + sampling ----
    def val_loss():
        model.eval()
        losses = {}
        for path in sorted(glob.glob(os.path.join(SHARD_DIR, "val_*.npy"))):
            name = os.path.basename(path)[4:-4]
            loader = DataLoader(path, B, T, rank, world_size)
            total = torch.zeros((), device=device)
            n = 0
            with torch.no_grad():
                for _ in range(VAL_BATCHES):
                    batch = loader.next_batch()
                    if batch is None:
                        break
                    x, y = batch[0].to(device), batch[1].to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
                        _, loss = model(x, y)
                    total += loss
                    n += 1
            if ddp:
                dist.all_reduce(total, op=dist.ReduceOp.AVG)
            losses[name] = (total / max(n, 1)).item()
        model.train()
        return losses

    def sample(prompt, n=2, max_len=48):
        raw_model.eval()
        x = torch.tensor(enc.encode(prompt).ids, dtype=torch.long, device=device).unsqueeze(0).repeat(n, 1)
        g = torch.Generator(device=device).manual_seed(42)
        with torch.no_grad():
            while x.size(1) < max_len:
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
                    logits, _ = raw_model(x[:, -T:])
                probs = F.softmax(logits[:, -1, :], dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=g)
                x = torch.cat((x, torch.gather(topk_indices, -1, ix)), dim=1)
        raw_model.train()
        for row in x:
            log("> " + enc.decode(row.tolist()).replace("\n", " "))

    # ---- training loop ----
    session_start = time.time()
    last_save = time.time()
    if OVERFIT:
        overfit_batch = train_loader.next_batch()

    while True:
        if DECAY_START is not None and step >= DECAY_START + DECAY_STEPS:
            log("decay finished, run is done")
            break
        if MAX_STEPS is not None and step >= MAX_STEPS:
            log(f"hit SOMAB_MAX_STEPS={MAX_STEPS}, stopping")
            if master and not OVERFIT:
                save_weights(step, train_loader.state())
            break

        if step % VAL_EVERY == 0 and not OVERFIT:
            losses = val_loss()
            log(f"step {step} val | " + " | ".join(f"{k} {v:.4f}" for k, v in losses.items()))
        if step % SAMPLE_EVERY == 0 and step > 0 and master:
            sample("The most important thing to know about")

        t0 = time.time()
        state_before = train_loader.state() # if the data runs out mid-step, this is where to resume
        lr = get_lr(step, run["max_lr"], run["warmup"], DECAY_START, DECAY_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_accum = torch.zeros((), device=device)
        out_of_data = False
        for micro in range(grad_accum):
            batch = overfit_batch if OVERFIT else train_loader.next_batch()
            if batch is None:
                out_of_data = True
                break
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            if ddp:
                # only sync gradients between the GPUs on the last micro step
                model.require_backward_grad_sync = micro == grad_accum - 1
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_fp16):
                _, loss = model(x, y)
            # divide so the accumulated gradient is the mean over the whole batch, not the sum
            loss = loss / grad_accum
            loss_accum += loss.detach()
            scaler.scale(loss).backward()

        if out_of_data:
            optimizer.zero_grad(set_to_none=True)
            log("out of training shards, run gen_dataset.py (or pull the next chunk) and start again")
            if master:
                save_weights(step, state_before)
            break

        scaler.unscale_(optimizer) # clip the real gradients, not the scaled ones
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer) # skips the step if fp16 overflowed
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok_per_s = run["total_batch"] / dt
        log(f"step {step:6d} | loss {loss_accum.item():.4f} | lr {lr:.2e} | norm {norm:.3f} | "
            f"scale {scaler.get_scale():.0f} | {dt * 1000:.0f}ms | {tok_per_s:,.0f} tok/s")

        # every process has to agree on saving/stopping, so rank 0 decides and tells the rest
        flags = torch.tensor([time.time() - last_save > SAVE_EVERY_MIN * 60,
                              time.time() - session_start > SESSION_HOURS * 3600], device=device, dtype=torch.int32)
        if ddp:
            dist.broadcast(flags, 0)
        save_now, stop_now = bool(flags[0]), bool(flags[1])
        if (save_now or stop_now) and master and not OVERFIT:
            save_weights(step, train_loader.state())
            last_save = time.time()
        if stop_now:
            log(f"session hit {SESSION_HOURS}h, stopping. start a new session to keep going")
            break

    if upload_thread is not None:
        upload_thread.join() # don't let the session end mid-upload
    if ddp:
        dist.destroy_process_group()
