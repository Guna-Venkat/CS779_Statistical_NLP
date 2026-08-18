import argparse
import os
import unicodedata
import re
import time
import gc
from collections import defaultdict
import numpy as np

# -------------------------
# Utilities & IO
# -------------------------
SPECIAL_SPACE = "\u2581"  # '▁' boundary token

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = unicodedata.normalize("NFKC", text)
    # collapse whitespace -> single space, then convert space to boundary symbol
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", SPECIAL_SPACE)
    return text

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        # write in chunks for speed
        chunk_size = 10000
        for i in range(0, len(tokens), chunk_size):
            f.write("\n".join(tokens[i:i+chunk_size]) + ("\n" if i+chunk_size < len(tokens) else ""))

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

# -------------------------
# Core numeric conversion helpers
# -------------------------
def text_to_id_array(text):
    """
    Convert str -> array of uint32 codepoints (UTF-32-LE), and produce compact unique_chars array.
    Returns:
      ids: np.ndarray(dtype=uint32)  # length n, mapped to 0..m-1
      unique_codepoints: np.ndarray(dtype=uint32)  # length m, codepoint value for each id
    """
    # encode to utf-32-le (no BOM), then view as uint32
    b = text.encode("utf-32-le")
    arr_cp = np.frombuffer(b, dtype=np.uint32)   # view into bytes, no copy
    unique_cp = np.unique(arr_cp)                # sorted unique codepoints (C-level)
    ids = np.searchsorted(unique_cp, arr_cp).astype(np.uint32, copy=False)
    # free memory
    del arr_cp, b
    gc.collect()
    return ids, unique_cp

# -------------------------
# Pruning helper (numpy)
# -------------------------
def top_k_indices(counts_list, keep_k):
    """Return indices of top keep_k items in counts_list (list or numpy array)."""
    cnt = np.asarray(counts_list, dtype=np.int64)
    if keep_k >= cnt.size:
        return np.arange(cnt.size, dtype=np.int64)
    # argpartition gives top-k (unordered); we'll return those indices
    idx = np.argpartition(-cnt, keep_k - 1)[:keep_k]
    # optionally sort them by descending counts for deterministic output
    idx = idx[np.argsort(-cnt[idx])]
    return idx

# -------------------------
# Training (beam/positions + numpy)
# -------------------------
def train_unigram_tokenizer(text, vocab_size, prune_ratio=0.2, max_sub_len=10):
    """
    Train a unigram-style token set using beam / position-tracking approach.
    Returns vocab (list of strings) and tokenizer mapping (token -> id).
    """
    ids, unique_cp = text_to_id_array(text)   # ids are compact (0..m-1)
    n = ids.shape[0]
    m = unique_cp.shape[0]

    # --- k = 1: keep ALL single chars (no UNK)
    counts_k1 = np.bincount(ids, minlength=m)
    prefixes = {}           # token_key (tuple of compact ids) -> {'positions': np.uint32 array, 'count': int}
    global_counts = {}      # token_key -> count (only store counts globally, not positions)
    # generate single-char prefixes
    for char_id in range(m):
        pos = np.nonzero(ids == char_id)[0].astype(np.uint32)
        cnt = int(counts_k1[char_id])
        key = (int(char_id),)
        prefixes[key] = {"positions": pos, "count": cnt}
        global_counts[key] = cnt

    # --- extend for k = 2..max_sub_len using beam + prune
    for k in range(2, max_sub_len + 1):
        t_iter = time.time()
        cand_keys = []
        cand_counts = []
        cand_positions = []

        # For each surviving prefix, compute next-char extensions via vectorized numpy ops
        max_start = n - (k - 1)
        for prefix_key, info in prefixes.items():
            pos = info["positions"]
            if pos.size == 0:
                continue
            # restrict to valid starts where the k-length substring fits
            if pos[-1] >= max_start:
                pos = pos[pos < max_start]
                if pos.size == 0:
                    continue
            # vectorized gather of next character ids
            next_ids = ids[pos + (k - 1)]
            vals, counts_vals, inv = np.unique(next_ids, return_counts=True, return_inverse=True)
            # create candidate extension tokens
            for j in range(vals.size):
                new_key = prefix_key + (int(vals[j]),)
                mask = (inv == j)
                survivors = np.nonzero(mask)[0]   # indices of survivors
                pos_j = survivors.astype(np.uint32, copy=False)
                cand_keys.append(new_key)
                cand_counts.append(int(counts_vals[j]))
                cand_positions.append(pos_j)

        total_cands = len(cand_keys)
        if total_cands == 0:
            break

        # prune bottom prune_ratio (keep top (1 - prune_ratio))
        keep_k = max(1, int(total_cands * (1.0 - prune_ratio)))
        if keep_k < total_cands:
            chosen_idx = top_k_indices(cand_counts, keep_k)
        else:
            chosen_idx = np.arange(total_cands, dtype=np.int64)

        # build new prefixes dict from chosen indices and update global pool
        new_prefixes = {}
        for idx in chosen_idx:
            key = cand_keys[int(idx)]
            pos = cand_positions[int(idx)]
            cnt = cand_counts[int(idx)]
            new_prefixes[key] = {"positions": pos, "count": cnt}
            # record into global candidates pool
            prev = global_counts.get(key)
            if prev is None or cnt > prev:
                global_counts[key] = cnt

        # replace prefixes with survivors for next round
        prefixes = new_prefixes
        # free temporary lists & memory
        del cand_keys, cand_counts, cand_positions
        gc.collect()

    # --- finalize: pick top vocab_size tokens from global_counts
    items = list(global_counts.items())   # (key, count)
    keys = [kv[0] for kv in items]
    counts_all = np.array([kv[1] for kv in items], dtype=np.int64)
    total_candidates = counts_all.size
    if total_candidates <= vocab_size:
        top_idx = np.arange(total_candidates, dtype=np.int64)
    else:
        top_idx = top_k_indices(counts_all, vocab_size)

    # build final vocab strings (convert compact ids -> codepoints -> chars)
    vocab = []
    for i in top_idx:
        key = keys[int(i)]
        chars = [chr(int(unique_cp[id_])) for id_ in key]
        token = "".join(chars)
        vocab.append(token)

    tokenizer = {tok: i for i, tok in enumerate(vocab)}

    # free big arrays
    del ids, unique_cp
    gc.collect()

    return vocab, tokenizer

# -------------------------
# Trie for tokenization (char-level)
# -------------------------
def build_trie_from_vocab(vocab):
    """
    Build a nested-dict trie: node is dict mapping char -> node; terminal nodes hold '_id' -> int
    """
    root = {}
    for i, tok in enumerate(vocab):
        node = root
        for ch in tok:
            node = node.setdefault(ch, {})
        node['_id'] = i
    return root

def tokenize(text, tokenizer_trie, vocab):
    """
    Greedy longest-match tokenization using char-level trie.
    Returns list of token strings.
    """
    tokens = []
    n = len(text)
    i = 0
    while i < n:
        node = tokenizer_trie
        last_id = None
        last_len = 0
        j = i
        # walk as far as possible
        while j < n and text[j] in node:
            node = node[text[j]]
            j += 1
            if '_id' in node:
                last_id = node['_id']
                last_len = j - i
        if last_len > 0:
            tokens.append(vocab[last_id])
            i += last_len
        else:
            # fallback single-char token (guaranteed present in vocab)
            tokens.append(text[i])
            i += 1
    return tokens

def detokenize(tokens):
    """
    Join tokens and convert special boundary symbol back to space.
    """
    txt = "".join(tokens)
    return txt.replace(SPECIAL_SPACE, " ")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251140009"
    max_sub_len = 10
    prune_ratio = 0.2

    train_text = load_training_data(args.train)

    vocab, tokenizer = train_unigram_tokenizer(
        train_text,
        args.vocab_size,
        prune_ratio=prune_ratio,
        max_sub_len=max_sub_len
    )
    save_vocab(vocab, rollno, args.vocab_size)

    trie = build_trie_from_vocab(vocab)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    sample_text = unicodedata.normalize("NFKC", sample_text)
    sample_text = re.sub(r"\s+", " ", sample_text).replace(" ", SPECIAL_SPACE)

    tokens = tokenize(sample_text, trie, vocab)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)

