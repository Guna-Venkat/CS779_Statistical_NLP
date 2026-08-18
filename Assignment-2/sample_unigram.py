import argparse
import os
import unicodedata
import re
import gc
from collections import defaultdict
import numpy as np

# Special space marker used for segmentation
SPECIAL_SPACE = "\u2581"  # '▁'


def load_training_data(train_path):
    """
    Load and normalize training text.
    - Converts to NFKC form.
    - Collapses multiple whitespaces into one.
    - Replaces spaces with SPECIAL_SPACE marker.
    """
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    text = text.replace(" ", SPECIAL_SPACE)
    return text


def save_vocab(vocab, rollno, vocab_size):
    """Save learned vocabulary tokens to a file."""
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")


def save_tokens(tokens, rollno):
    """Save tokenized sequence to file (writes in chunks for efficiency)."""
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        chunk_size = 10000
        for i in range(0, len(tokens), chunk_size):
            f.write("\n".join(tokens[i:i+chunk_size]) + ("\n" if i+chunk_size < len(tokens) else ""))


def save_detokenized(text, rollno):
    """Save detokenized text to file."""
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)


def text_to_id_array(text):
    """
    Convert string into array of compact integer IDs.
    Steps:
    - Encode text into UTF-32 (so each codepoint = fixed width 4 bytes).
    - Map codepoints to a compact range 0..m-1.
    Returns:
        ids: np.ndarray of int IDs (mapped chars).
        unique_cp: array of unique codepoints.
    """
    b = text.encode("utf-32-le")
    arr_cp = np.frombuffer(b, dtype=np.uint32)  # raw codepoints
    unique_cp = np.unique(arr_cp)               # sorted unique set of codepoints
    ids = np.searchsorted(unique_cp, arr_cp).astype(np.uint32, copy=False)
    del arr_cp, b
    gc.collect()
    return ids, unique_cp


def top_k_indices(counts_list, keep_k):
    """
    Return indices of top-k highest values in counts_list.
    Uses argpartition for efficiency.
    """
    cnt = np.asarray(counts_list, dtype=np.int64)
    if keep_k >= cnt.size:
        return np.arange(cnt.size, dtype=np.int64)
    idx = np.argpartition(-cnt, keep_k - 1)[:keep_k]
    idx = idx[np.argsort(-cnt[idx])]  # sort top-k by frequency
    return idx


def train_unigram_tokenizer(text, vocab_size, prune_ratio=0.2, max_sub_len=10):
    """
    Train a unigram tokenizer (frequency-based, not EM).
    Steps:
    1. Start with single-character tokens.
    2. Iteratively extend substrings up to max_sub_len.
    3. Prune less frequent substrings at each step.
    4. Collect global counts and pick top vocab_size.
    5. Convert counts into approximate probabilities.
    """
    ids, unique_cp = text_to_id_array(text)
    n = ids.shape[0]
    m = unique_cp.shape[0]

    # k=1: single characters
    counts_k1 = np.bincount(ids, minlength=m)
    prefixes = {}
    global_counts = {}
    for char_id in range(m):
        pos = np.nonzero(ids == char_id)[0].astype(np.uint32)
        cnt = int(counts_k1[char_id])
        key = (int(char_id),)
        prefixes[key] = {"positions": pos, "count": cnt}
        global_counts[key] = cnt

    # Extend substrings up to max_sub_len
    for k in range(2, max_sub_len + 1):
        cand_keys, cand_counts, cand_positions = [], [], []
        max_start = n - (k - 1)

        # Extend each surviving prefix
        for prefix_key, info in prefixes.items():
            pos = info["positions"]
            if pos.size == 0:
                continue
            if pos[-1] >= max_start:  # remove positions too close to end
                pos = pos[pos < max_start]
                if pos.size == 0:
                    continue
            next_ids = ids[pos + (k - 1)]
            vals, counts_vals, inv = np.unique(next_ids, return_counts=True, return_inverse=True)

            # Create new candidate substrings
            for j in range(vals.size):
                new_key = prefix_key + (int(vals[j]),)
                survivors = np.nonzero(inv == j)[0]
                pos_j = survivors.astype(np.uint32, copy=False)
                cand_keys.append(new_key)
                cand_counts.append(int(counts_vals[j]))
                cand_positions.append(pos_j)

        # Stop if no candidates
        total_cands = len(cand_keys)
        if total_cands == 0:
            break

        # Prune to keep top candidates
        keep_k = max(1, int(total_cands * (1.0 - prune_ratio)))
        if keep_k < total_cands:
            chosen_idx = top_k_indices(cand_counts, keep_k)
        else:
            chosen_idx = np.arange(total_cands, dtype=np.int64)

        # Build next prefixes
        new_prefixes = {}
        for idx in chosen_idx:
            key = cand_keys[int(idx)]
            pos = cand_positions[int(idx)]
            cnt = cand_counts[int(idx)]
            new_prefixes[key] = {"positions": pos, "count": cnt}
            prev = global_counts.get(key)
            if prev is None or cnt > prev:
                global_counts[key] = cnt
        prefixes = new_prefixes

        del cand_keys, cand_counts, cand_positions
        gc.collect()

    # Select top vocab_size substrings
    items = list(global_counts.items())
    keys = [kv[0] for kv in items]
    counts_all = np.array([kv[1] for kv in items], dtype=np.int64)
    if counts_all.size <= vocab_size:
        top_idx = np.arange(counts_all.size, dtype=np.int64)
    else:
        top_idx = top_k_indices(counts_all, vocab_size)

    vocab = []
    for i in top_idx:
        key = keys[int(i)]
        chars = [chr(int(unique_cp[id_])) for id_ in key]
        token = "".join(chars)
        vocab.append(token)

    tokenizer = {tok: i for i, tok in enumerate(vocab)}

    # Approximate probabilities from frequencies
    total_count = counts_all[top_idx].sum() if counts_all.size > 0 else 0
    probs = {}
    eps = 1e-12
    for i, idx in enumerate(top_idx):
        cnt = int(counts_all[int(idx)])
        probs[vocab[i]] = (cnt / total_count) if total_count > 0 else eps
    if total_count == 0:
        for tok in vocab:
            probs[tok] = 1.0 / max(1, len(vocab))

    del ids, unique_cp
    gc.collect()
    return vocab, {"token_to_id": tokenizer, "probs": probs}


def tokenize(text, vocab, token_probs, unk_prob=1e-12):
    """
    Tokenize text using Viterbi dynamic programming.
    Finds globally optimal segmentation based on token probabilities.
    Fallback: if no match, break text into single characters.
    """
    max_token_len = max((len(t) for t in vocab), default=0)
    token_set = set(vocab)

    n = len(text)
    dp = np.full(n + 1, -np.inf)   # dp[i] = best log-prob up to position i
    back = [-1] * (n + 1)          # backpointer to reconstruct tokens
    dp[0] = 0.0

    # Forward pass: dynamic programming
    for i in range(n):
        if dp[i] == -np.inf:
            continue
        end_limit = min(n, i + max_token_len)
        matched = False

        # Try substrings starting at position i
        for j in range(i + 1, end_limit + 1):
            substr = text[i:j]
            if substr in token_set:
                p = token_probs.get(substr, unk_prob)
                score = dp[i] + float(np.log(p + 1e-300))
                if score > dp[j]:
                    dp[j] = score
                    back[j] = i
                matched = True

        # If no match, fallback to single char
        if not matched:
            substr = text[i:i+1]
            p = token_probs.get(substr, unk_prob)
            score = dp[i] + float(np.log(p + 1e-300))
            if score > dp[i+1]:
                dp[i+1] = score
                back[i+1] = i

    # Backtrack to recover tokens
    tokens = []
    idx = n
    if dp[n] == -np.inf:  # fallback: split into characters
        i = 0
        while i < n:
            tokens.append(text[i])
            i += 1
        return tokens

    while idx > 0:
        j = back[idx]
        if j is None or j < 0:
            j = idx - 1
        tokens.append(text[j:idx])
        idx = j
    tokens.reverse()
    return tokens


def detokenize(tokens):
    """Join tokens and replace special marker with space."""
    txt = "".join(tokens)
    return txt.replace(SPECIAL_SPACE, " ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251140009"
    max_sub_len = 10
    prune_ratio = 0.2

    # Train
    train_text = load_training_data(args.train)
    vocab, tokenizer = train_unigram_tokenizer(train_text, args.vocab_size, prune_ratio, max_sub_len)
    save_vocab(vocab, rollno, args.vocab_size)

    # Tokenize
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    sample_text = unicodedata.normalize("NFKC", sample_text)
    sample_text = re.sub(r"\s+", " ", sample_text).replace(" ", SPECIAL_SPACE)
    tokens = tokenize(sample_text, vocab, tokenizer["probs"])
    save_tokens(tokens, rollno)

    # Detokenize
    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)
