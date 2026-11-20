import os, json, glob, time
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "datasets")
OUT_DIR = os.path.join(DATA_DIR, "shards")
STATE_PATH = os.path.join(OUT_DIR, "state.json")
TOKENIZER_PATH = os.path.join(HERE, "tokenizer", "smollm2_tokenizer.json")

SHARD_SIZE = 100_000_000 # tokens per train shard, 200MB as uint16
VAL_TOKENS = 2_000_000 # per source
ENCODE_BATCH = 1024 # docs tokenized at a time

# folder in datasets/ -> share of the mix. shards before train_000023 were made without stack-edu (code),
# with its 8% spread over the rest. stack-edu is Python + Markdown, see datasets/stack-edu-index/fetch.py
MIX = {
    "fineweb-edu": 0.598,
    "dclm": 0.2484,
    "finemath-3plus": 0.046,
    "wikipedia": 0.0138,
    "cosmopedia-v2": 0.0138,
    "stack-edu": 0.08,
}

# junk filter: spam like "how many zeros in a duotrigintillion" is up to a million tokens of just "0" and ",".
# a doc gets skipped when its two most common tokens are over half of it. real text sits around 0.05-0.15,
# so 0.5 only catches number dumps, blank forms and the like (~32 docs per 800M tokens on the first chunk)
JUNK_MIN_TOKENS = 200 # shorter docs aren't checked
JUNK_TOP2_SHARE = 0.5

enc = Tokenizer.from_file(TOKENIZER_PATH)
EOT = enc.token_to_id("<|endoftext|>")
assert enc.get_vocab_size() <= 2**16, "token ids have to fit in uint16"

def is_junk(doc):
    ids = doc[1:] # skip the <|endoftext|> in front
    if len(ids) < JUNK_MIN_TOKENS:
        return False
    counts = np.bincount(ids)
    top2 = np.partition(counts, -2)[-2:].sum() if len(counts) > 1 else counts.sum()
    return top2 / len(ids) > JUNK_TOP2_SHARE

class Source:
    """ reads one source's parquet files in name order, one row group at a time, and hands out tokenized docs """

    def __init__(self, name, pos):
        self.name = name
        # pos is where the next doc comes from: {"file": name, "row_group": i, "row": j}, or None for the very start
        self.pos = pos
        self.texts = [] # raw text of the current row group
        self.row = 0 # next row of texts to hand out
        self.docs = [] # the encoded batch, docs[doc_i] is row
        self.doc_i = 0
        self.done = False
        self.skipped_docs = 0 # junk docs skipped since the last reset_skipped()
        self.skipped_tokens = 0

    def files(self):
        return sorted(os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, self.name, "*.parquet")))

    def load_next_group(self):
        files = self.files()
        if self.pos is None:
            if not files:
                self.done = True
                return
            self.pos = {"file": files[0], "row_group": 0, "row": 0}

        # once a file is finished, move to the next file name after it. going by name instead of index
        # means already-used files can be deleted without breaking this
        while True:
            path = os.path.join(DATA_DIR, self.name, self.pos["file"])
            n_groups = pq.ParquetFile(path).num_row_groups if os.path.exists(path) else 0
            if self.pos["row_group"] < n_groups:
                break
            later = [f for f in files if f > self.pos["file"]]
            if not later:
                self.done = True
                return
            self.pos = {"file": later[0], "row_group": 0, "row": 0}

        self.texts = pq.ParquetFile(path).read_row_group(self.pos["row_group"], columns=["text"]).column("text").to_pylist()
        self.row = self.pos["row"]

    def encode_some(self):
        # encode the next ENCODE_BATCH docs of the row group. encode_batch runs on every CPU core,
        # and doing it in pieces keeps memory down (one dclm row group is 61k docs)
        batch = self.texts[self.row:self.row + ENCODE_BATCH]
        self.docs = [np.array([EOT] + e.ids, dtype=np.uint16) for e in enc.encode_batch(batch)]
        self.doc_i = 0

    def next_doc(self):
        """ the next doc that isn't junk, or None once the downloaded files run out """
        while True:
            doc = self.next_raw_doc()
            if doc is None or not is_junk(doc):
                return doc
            self.skipped_docs += 1
            self.skipped_tokens += len(doc)

    def reset_skipped(self):
        self.skipped_docs = self.skipped_tokens = 0

    def next_raw_doc(self):
        while self.doc_i >= len(self.docs):
            if self.done:
                return None
            if self.row >= len(self.texts):
                if self.texts:
                    # finished this row group, the next one starts at row 0
                    self.pos = {**self.pos, "row_group": self.pos["row_group"] + 1, "row": 0}
                self.load_next_group()
                if self.done:
                    return None
                if self.row >= len(self.texts):
                    # row group was empty or already used up
                    self.pos = {**self.pos, "row_group": self.pos["row_group"] + 1, "row": 0}
                    self.texts = []
                    continue
            self.encode_some()
        doc = self.docs[self.doc_i]
        self.doc_i += 1
        self.row += 1
        self.pos = {**self.pos, "row": self.row}
        return doc

def save_tokens(path, chunks):
    tokens = np.concatenate(chunks)
    np.save(path, tokens)
    return len(tokens)

def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH) # so a crash mid-write can't leave a broken state file

def make_val(sources, state):
    """ val set from the first VAL_TOKENS of every source that doesn't have one yet, so a source added later still gets one """
    for name, src in sources.items():
        if os.path.exists(os.path.join(OUT_DIR, f"val_{name}.npy")):
            continue
        chunks, n = [], 0
        while n < VAL_TOKENS:
            doc = src.next_doc()
            if doc is None:
                break
            chunks.append(doc)
            n += len(doc)
        if not chunks:
            print(f"warning: no data for {name}, no val set")
            continue
        n = save_tokens(os.path.join(OUT_DIR, f"val_{name}.npy"), chunks)
        print(f"val_{name}.npy: {n:,} tokens")
        state["sources"][name]["pos"] = src.pos # train starts after the val docs
    state["val_done"] = True
    save_state(state)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert abs(sum(MIX.values()) - 1.0) < 1e-6, "MIX has to add up to 1"

    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
    else:
        state = {"next_shard": 0, "val_done": False, "sources": {}}

    sources = {name: Source(name, state["sources"].get(name, {}).get("pos")) for name in MIX}
    for name in MIX:
        state["sources"].setdefault(name, {"pos": None, "tokens": 0})

    make_val(sources, state) # only does sources with no val_<name>.npy yet

    while True:
        chunks, n = [], 0
        per_source = {name: 0 for name in MIX}
        for src in sources.values():
            src.reset_skipped()
        t0 = time.time()
        ran_out = None
        while n < SHARD_SIZE:
            # take the next doc from whichever source is furthest behind its share
            name = min(MIX, key=lambda s: per_source[s] / MIX[s])
            doc = sources[name].next_doc()
            if doc is None:
                ran_out = name
                break
            chunks.append(doc)
            n += len(doc)
            per_source[name] += len(doc)

        if ran_out is not None:
            # the unfinished shard isn't saved, and state.json still points at its first doc, so those docs get used next run
            print(f"{ran_out} ran out of downloaded files, stopping. pull the next chunk and run this again.")
            print(f"(the {n:,} tokens of the unfinished shard will be redone next run)")
            break

        shard_name = f"train_{state['next_shard']:06d}.npy"
        save_tokens(os.path.join(OUT_DIR, shard_name), chunks)
        state["next_shard"] += 1
        for name in MIX:
            state["sources"][name]["pos"] = sources[name].pos
            state["sources"][name]["tokens"] += per_source[name]
            s = state["sources"][name]
            s["skipped_docs"] = s.get("skipped_docs", 0) + sources[name].skipped_docs
            s["skipped_tokens"] = s.get("skipped_tokens", 0) + sources[name].skipped_tokens
        save_state(state)

        dt = time.time() - t0
        mix = ", ".join(f"{s} {per_source[s] / n:.1%}" for s in MIX)
        skipped = sum(src.skipped_docs for src in sources.values())
        skipped_tok = sum(src.skipped_tokens for src in sources.values())
        print(f"{shard_name}: {n:,} tokens in {dt:.0f}s ({n / dt / 1e6:.2f}M tok/s) | {mix} | junk skipped: {skipped} docs, {skipped_tok:,} tokens")

    total = sum(s["tokens"] for s in state["sources"].values())
    print(f"done. {state['next_shard']} train shards, {total:,} train tokens total")

    # only files before the saved position are fully in shards. the file the position is in is still needed
    safe = []
    for name in MIX:
        pos = state["sources"][name]["pos"]
        if pos is not None:
            safe += [os.path.join(name, f) for f in sources[name].files() if f < pos["file"]]
    if safe:
        print("these parquet files are fully tokenized and safe to delete:")
        for f in safe:
            print("  datasets/" + f)

if __name__ == "__main__":
    main()
