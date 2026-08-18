import argparse
import math
import heapq
from collections import defaultdict
from typing import List, Dict
import numpy as np
from multiprocessing import Pool, cpu_count

# ------------------- Constants -------------------
RESERVED_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
CONTINUATION = "##"

# ------------------- Load Training Data -------------------
def load_training_data(train_path: str) -> List[str]:
    """
    Load and split training data into a list of words.
    
    Args:
        train_path (str): Path to the training text file.
    
    Returns:
        List[str]: List of words from the training data.
    """
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split()

# ------------------- Initial Vocabulary -------------------
def get_initial_vocab(words: List[str]) -> List[str]:
    """
    Initialize vocabulary with reserved tokens and all unique characters from words.
    
    Args:
        words (List[str]): List of training words.
    
    Returns:
        List[str]: Initial vocabulary.
    """
    chars = set()
    for w in words:
        chars.update(w)
    return RESERVED_TOKENS + sorted(chars)

# ------------------- Delta L -------------------
def compute_delta_L(f_new, f1, f2, N):
    """
    Compute the change in log-likelihood for a potential merge.
    
    Args:
        f_new (int): Frequency of the new merged token.
        f1 (int): Frequency of first token.
        f2 (int): Frequency of second token.
        N (int): Total token count.
    
    Returns:
        float: Delta log-likelihood.
    """
    delta = 0.0
    if f_new > 0:
        delta += f_new * math.log(f_new / N)
    if f1 > 0:
        delta -= f1 * math.log(f1 / N)
    if f2 > 0:
        delta -= f2 * math.log(f2 / N)
    return delta

# ------------------- Multiprocessing Helper -------------------
def count_pairs_chunk(chunk):
    """
    Count adjacent token pairs in a chunk of the corpus.
    
    Args:
        chunk (List[Tuple[int, np.ndarray]]): List of word indices and token arrays.
    
    Returns:
        Dict[Tuple[int,int], List[Tuple[int,int]]]: Mapping of token pairs to positions.
    """
    pair_counts = defaultdict(list)
    for w_idx, arr in chunk:
        for i in range(len(arr)-1):
            pair_counts[(arr[i], arr[i+1])].append((w_idx, i))
    return pair_counts

def merge_pair_counts(list_of_dicts):
    """
    Merge pair counts from multiple chunks into a single dictionary.
    
    Args:
        list_of_dicts (List[Dict]): List of pair count dictionaries.
    
    Returns:
        Dict: Merged dictionary of pair counts.
    """
    merged = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            merged[k].extend(v)
    return merged

# ------------------- WordPiece Training -------------------
def train_wordpiece_tokenizer(words: List[str], vocab_size: int):
    """
    Train a WordPiece tokenizer using a frequency-based greedy merge algorithm.
    
    Args:
        words (List[str]): Training words.
        vocab_size (int): Desired final vocabulary size.
    
    Returns:
        Tuple[List[str], Dict]: Vocabulary list and tokenizer dictionary.
    """
    vocab = get_initial_vocab(words)
    token2id = {tok: idx for idx, tok in enumerate(vocab)}
    id2token = {idx: tok for tok, idx in token2id.items()}

    # Convert words to arrays of token IDs
    corpus = [np.array([token2id[c] for c in w], dtype=np.int32) for w in words]
    masks = [np.ones(len(arr), dtype=bool) for arr in corpus]  # active positions

    # Initialize token counts
    token_counts = defaultdict(int)
    for arr in corpus:
        for t in arr:
            token_counts[t] += 1
    N = sum(token_counts.values())

    # Split corpus into chunks for parallel processing
    num_processes = max(1, cpu_count() - 1)
    chunk_size = (len(corpus) + num_processes - 1) // num_processes
    chunks = [(list(enumerate(corpus))[i:i+chunk_size]) for i in range(0, len(corpus), chunk_size)]

    # Count all adjacent token pairs in parallel
    with Pool(num_processes) as pool:
        pair_positions_list = pool.map(count_pairs_chunk, chunks)
    pair_positions = merge_pair_counts(pair_positions_list)

    # Initialize priority heap for candidate merges
    heap = []
    for pair, positions in pair_positions.items():
        f_new = len(positions)
        if f_new > 1:  # skip low-impact merges
            f1, f2 = token_counts[pair[0]], token_counts[pair[1]]
            delta = compute_delta_L(f_new, f1, f2, N)
            heapq.heappush(heap, (-delta, pair))

    merges = []
    merge_counter = 0

    # ------------------- Main Merge Loop -------------------
    while len(vocab) < vocab_size and heap:
        merge_counter += 1
        neg_delta, best_pair = heapq.heappop(heap)
        if best_pair not in pair_positions or not pair_positions[best_pair]:
            continue

        positions = pair_positions[best_pair]
        valid_positions = []
        for w_idx, i in positions:
            arr, mask = corpus[w_idx], masks[w_idx]
            if mask[i] and i+1 < len(arr) and arr[i]==best_pair[0] and arr[i+1]==best_pair[1]:
                valid_positions.append((w_idx,i))
        if len(valid_positions) <= 1:
            continue

        # Perform merge and create new token
        t1, t2 = id2token[best_pair[0]], id2token[best_pair[1]]
        new_token = CONTINUATION + t1.lstrip(CONTINUATION) + t2.lstrip(CONTINUATION) \
                    if t1.startswith(CONTINUATION) or t2.startswith(CONTINUATION) else t1 + t2
        new_id = len(vocab)
        vocab.append(new_token)
        token2id[new_token] = new_id
        id2token[new_id] = new_token
        merges.append(best_pair)

        # Update corpus and neighbor pairs
        left_neighbors = defaultdict(int)
        right_neighbors = defaultdict(int)
        for w_idx,i in valid_positions:
            arr, mask = corpus[w_idx], masks[w_idx]
            arr[i] = new_id
            mask[i+1] = False
            if i>0 and mask[i-1]:
                left_neighbors[(arr[i-1], new_id)] += 1
            if i+2 < len(arr) and mask[i+2]:
                right_neighbors[(new_id, arr[i+2])] += 1

        # Push neighbor pairs to heap
        for neighbor_dict in [left_neighbors, right_neighbors]:
            for pair_n, count in neighbor_dict.items():
                if count > 1:
                    f1n, f2n = token_counts.get(pair_n[0],0), token_counts.get(pair_n[1],0)
                    delta = compute_delta_L(count, f1n, f2n, N)
                    heapq.heappush(heap, (-delta, pair_n))

        # Update token counts
        f_new = len(valid_positions)
        token_counts[new_id] = f_new
        token_counts[best_pair[0]] -= f_new
        token_counts[best_pair[1]] -= f_new
        N -= f_new

    # ------------------- Build Trie for Fast Tokenization -------------------
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    class Trie:
        def __init__(self):
            self.root = TrieNode()

        def insert(self, token):
            """Insert a token into the trie."""
            node = self.root
            for c in token:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
            node.is_end = True

        def longest_match(self, word, start):
            """
            Find the longest token in the trie that matches the word starting at index `start`.
            
            Returns:
                Tuple[int, str]: End index and matched token string (or None if no match).
            """
            node = self.root
            match_end, matched_token = None, None
            i = start
            while i < len(word) and word[i] in node.children:
                node = node.children[word[i]]
                i += 1
                if node.is_end:
                    match_end = i
                    matched_token = word[start:i]
            return match_end, matched_token

    trie = Trie()
    for tok in vocab:
        trie.insert(tok)

    tokenizer = {"vocab": vocab, "merges": merges, "trie": trie}
    return vocab, tokenizer

# ------------------- Tokenization / Detokenization -------------------
def tokenize(text: str, tokenizer: Dict) -> List[str]:
    """
    Tokenize input text using the trained WordPiece tokenizer.
    """
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
                if i>0 and not match.startswith(CONTINUATION):
                    match = CONTINUATION + match
                sub_tokens.append(match)
                i = end
        output_tokens.extend(sub_tokens)
    return output_tokens

def detokenize(tokens: List[str]) -> str:
    """
    Reconstruct text from WordPiece tokens.
    """
    words, cur_word = [], ""
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
    """Save vocabulary to a file."""
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for t in vocab:
            f.write(t + "\n")

def save_tokens(tokens: List[str], rollno: str) -> None:
    """Save tokenized output to a file."""
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")

def save_detokenized(text: str, rollno: str) -> None:
    """Save detokenized text to a file."""
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

    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)
