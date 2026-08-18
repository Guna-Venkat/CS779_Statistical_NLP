import argparse
import math
import heapq
from collections import defaultdict
from typing import List, Dict
import numpy as np
import time
from multiprocessing import Pool, cpu_count

# ------------------- Constants -------------------
RESERVED_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
CONTINUATION = "##"
MIN_PAIR_FREQ = 2  # keep same logic threshold used in original (f_new > 1)

# ------------------- Trie (moved top-level for reuse) -------------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token: str):
        node = self.root
        for c in token:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def longest_match(self, word: str, start: int):
        node = self.root
        match_end = None
        matched_token = None
        i = start
        while i < len(word) and word[i] in node.children:
            node = node.children[word[i]]
            i += 1
            if node.is_end:
                match_end = i
                matched_token = word[start:i]
        return match_end, matched_token

# ------------------- Load Training Data -------------------
def load_training_data(train_path: str) -> List[str]:
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split()

# ------------------- Initial Vocabulary -------------------
def get_initial_vocab(words: List[str]) -> List[str]:
    chars = set()
    for w in words:
        chars.update(w)
    return RESERVED_TOKENS + sorted(chars)

# ------------------- Delta L (keeps same math but reduces log calls) -------------------
def compute_delta_L(f_new: int, f1: int, f2: int, logN: float):
    # delta = f_new*log(f_new/N) - f1*log(f1/N) - f2*log(f2/N)
    # = f_new*log(f_new) - f1*log(f1) - f2*log(f2) - (f_new - f1 - f2)*logN
    # compute only when counts > 0
    delta = 0.0
    if f_new > 0:
        delta += f_new * math.log(f_new)
    if f1 > 0:
        delta -= f1 * math.log(f1)
    if f2 > 0:
        delta -= f2 * math.log(f2)
    delta -= (f_new - f1 - f2) * logN
    return delta

# ------------------- Multiprocessing Helper (optimized local list usage) -------------------
def count_pairs_chunk(chunk):
    # chunk: list of (w_idx, arr)
    pair_counts = {}
    for w_idx, arr in chunk:
        alen = len(arr)
        # iterate with local vars for speed
        for i in range(alen - 1):
            pair = (int(arr[i]), int(arr[i + 1]))
            if pair in pair_counts:
                pair_counts[pair].append((w_idx, i))
            else:
                pair_counts[pair] = [(w_idx, i)]
    return pair_counts

def merge_pair_counts(list_of_dicts):
    merged = {}
    for d in list_of_dicts:
        for k, v in d.items():
            if k in merged:
                merged[k].extend(v)
            else:
                merged[k] = v[:]  # make a shallow copy to avoid aliasing
    return merged

# ------------------- WordPiece Training (Optimized) -------------------
def train_wordpiece_tokenizer(words: List[str], vocab_size: int):
    t0 = time.time()
    vocab = get_initial_vocab(words)
    token2id = {tok: idx for idx, tok in enumerate(vocab)}
    id2token = {idx: tok for tok, idx in token2id.items()}
    print(f"[Timer] Initial vocab created in {time.time() - t0:.2f}s")

    # Convert words to integer arrays
    t1 = time.time()
    # convert using local variables for speed
    token2id_local = token2id
    corpus = [np.array([token2id_local[c] for c in w], dtype=np.int32) for w in words]
    masks = [np.ones(len(arr), dtype=bool) for arr in corpus]
    print(f"[Timer] Converted words to arrays in {time.time() - t1:.2f}s")

    # Token counts
    token_counts = defaultdict(int)
    for arr in corpus:
        for t in arr:
            token_counts[int(t)] += 1
    N = sum(token_counts.values())
    if N <= 0:
        return vocab, {"vocab": vocab, "merges": [], "trie": Trie()}

    # Prepare enumerated corpus once and chunk it for multiprocessing
    enumerated_corpus = list(enumerate(corpus))
    num_processes = max(1, cpu_count() - 1)
    chunk_size = (len(enumerated_corpus) + num_processes - 1) // num_processes
    chunks = [enumerated_corpus[i:i+chunk_size] for i in range(0, len(enumerated_corpus), chunk_size)]

    # Parallel pair counting
    with Pool(num_processes) as pool:
        pair_positions_list = pool.map(count_pairs_chunk, chunks)

    pair_positions = merge_pair_counts(pair_positions_list)

    # Initialize heap with significant pairs only
    heap = []
    logN = math.log(N)
    for pair, positions in pair_positions.items():
        f_new = len(positions)
        if f_new >= MIN_PAIR_FREQ:
            f1 = token_counts.get(pair[0], 0)
            f2 = token_counts.get(pair[1], 0)
            delta = compute_delta_L(f_new, f1, f2, logN)
            if delta > 0.0:
                heapq.heappush(heap, (-delta, pair))

    merges = []
    merge_counter = 0
    start_merge = time.time()

    # Local references for speed
    corpus_local = corpus
    masks_local = masks
    pair_positions_local = pair_positions
    token_counts_local = token_counts
    id2token_local = id2token
    vocab_local = vocab
    heap_local = heap
    compute_delta = compute_delta_L

    while len(vocab_local) < vocab_size and heap_local:
        merge_counter += 1
        neg_delta, best_pair = heapq.heappop(heap_local)

        positions = pair_positions_local.get(best_pair)
        if not positions:
            continue

        # Filter valid positions lazily
        valid_positions = []
        for w_idx, i in positions:
            arr = corpus_local[w_idx]
            mask = masks_local[w_idx]
            # check bounds and equality using ints
            if i + 1 < len(arr) and mask[i] and mask[i+1] and arr[i] == best_pair[0] and arr[i+1] == best_pair[1]:
                valid_positions.append((w_idx, i))
        if len(valid_positions) <= 1:
            continue

        # Recompute true delta with up-to-date counts
        f_new = len(valid_positions)
        f1 = token_counts_local.get(best_pair[0], 0)
        f2 = token_counts_local.get(best_pair[1], 0)
        true_delta = compute_delta(f_new, f1, f2, math.log(N) if N > 0 else 0.0)
        if -neg_delta != true_delta:
            # push updated delta if positive
            if true_delta > 0.0:
                heapq.heappush(heap_local, (-true_delta, best_pair))
            continue

        # ---- Perform merge (core logic unchanged) ----
        t1_tok = id2token_local[best_pair[0]]
        t2_tok = id2token_local[best_pair[1]]
        if t1_tok.startswith(CONTINUATION) or t2_tok.startswith(CONTINUATION):
            new_token = CONTINUATION + t1_tok.lstrip(CONTINUATION) + t2_tok.lstrip(CONTINUATION)
        else:
            new_token = t1_tok + t2_tok

        new_id = len(vocab_local)
        vocab_local.append(new_token)
        token2id_local[new_token] = new_id
        id2token_local[new_id] = new_token
        merges.append(best_pair)

        # Update corpus and collect neighbor counts in bulk
        left_neighbors = defaultdict(int)
        right_neighbors = defaultdict(int)

        for w_idx, i in valid_positions:
            arr = corpus_local[w_idx]
            mask = masks_local[w_idx]
            # replace left token id with new_id and deactivate right slot
            arr[i] = new_id
            mask[i+1] = False  # mark merged-out position

            # left neighbor (a, new_id)
            if i > 0 and mask[i-1]:
                left_neighbors[(int(arr[i-1]), new_id)] += 1
            # right neighbor (new_id, c)
            if i + 2 < len(arr) and mask[i+2]:
                right_neighbors[(new_id, int(arr[i+2]))] += 1

        # Update heap with neighbor pairs (only significant counts)
        for neighbor_dict in (left_neighbors, right_neighbors):
            for pair_n, count in neighbor_dict.items():
                if count >= MIN_PAIR_FREQ:
                    f1n = token_counts_local.get(pair_n[0], 0)
                    f2n = token_counts_local.get(pair_n[1], 0)
                    delta_n = compute_delta(count, f1n, f2n, math.log(N) if N > 0 else 0.0)
                    if delta_n > 0.0:
                        heapq.heappush(heap_local, (-delta_n, pair_n))

        # Update token counts and N (core logic)
        token_counts_local[new_id] = f_new
        token_counts_local[best_pair[0]] -= f_new
        token_counts_local[best_pair[1]] -= f_new
        N -= f_new
        if N <= 0:
            break  # safety

    print(f"[Timer] Merge loop completed in {time.time() - start_merge:.2f}s (merges: {merge_counter})")

    # Build Trie for fast tokenization
    trie = Trie()
    for tok in vocab_local:
        trie.insert(tok)

    tokenizer = {"vocab": vocab_local, "merges": merges, "trie": trie}
    return vocab_local, tokenizer

# ------------------- Tokenization / Detokenization -------------------
def tokenize(text: str, tokenizer: Dict) -> List[str]:
    output_tokens = []
    trie = tokenizer["trie"]
    for word in text.split():
        i = 0
        sub_tokens = []
        while i < len(word):
            end, match = trie.longest_match(word, i)
            if match is None:
                sub_tokens.append("<unk>")
                i += 1
            else:
                if i > 0 and not match.startswith(CONTINUATION):
                    match = CONTINUATION + match
                sub_tokens.append(match)
                i = end
        output_tokens.extend(sub_tokens)
    return output_tokens

def detokenize(tokens: List[str]) -> str:
    words = []
    cur_word = ""
    for tok in tokens:
        if tok in RESERVED_TOKENS:
            continue
        if tok.startswith(CONTINUATION):
            cur_word += tok[len(CONTINUATION):]
        else:
            if cur_word:
                words.append(cur_word)
            cur_word = tok
    if cur_word:
        words.append(cur_word)
    return " ".join(words)

# ------------------- Saving -------------------
def save_vocab(vocab: List[str], rollno: str, vocab_size: int) -> None:
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for t in vocab:
            f.write(t + "\n")

def save_tokens(tokens: List[str], rollno: str) -> None:
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")

def save_detokenized(text: str, rollno: str) -> None:
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

# ------------------- Main -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251140009"

    words = load_training_data(args.train)
    vocab, tokenizer = train_wordpiece_tokenizer(words, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
