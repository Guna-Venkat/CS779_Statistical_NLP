import argparse
import unicodedata
from collections import defaultdict
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count, Manager

END_WORD = "</w>"
SPACE_MARKER = "▁"

# ------------------- Load & Normalize -------------------
def load_training_data(train_path: str) -> str:
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_text(text: str) -> str:
    """Unicode NFKC normalization + lowercase"""
    return unicodedata.normalize("NFKC", text).lower()

# ------------------- BPE / SentencePiece Utilities -------------------
def words_to_symbols(text: str) -> List[List[str]]:
    """Convert words in text to lists of symbols with space marker"""
    return [[SPACE_MARKER] + list(word) + [END_WORD] for word in text.split()]

# ------------------- Parallel Pair Counting -------------------
def count_pairs_chunk(words_chunk):
    local_pairs = defaultdict(int)
    for word in words_chunk:
        for i in range(len(word) - 1):
            local_pairs[(word[i], word[i + 1])] += 1
    return local_pairs

def merge_pair_counts(pair_counts_list):
    merged = defaultdict(int)
    for d in pair_counts_list:
        for k, v in d.items():
            merged[k] += v
    return merged

# ------------------- Optimized BPE Training with Multiprocessing -------------------
def train_sp_tokenizer_parallel(text: str, vocab_size: int, num_processes=None) -> Tuple[List[str], Dict]:
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    # Prepare symbol words
    words = text.split()
    symbol_words = [[SPACE_MARKER] + list(word) + [END_WORD] for word in words]

    # Initialize vocab with unique symbols
    vocab_symbols = set(s for word in symbol_words for s in word)
    vocab = ["<pad>", "<unk>", "<s>", "</s>"] + sorted(vocab_symbols)
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}

    max_merges = max(0, vocab_size - len(vocab))
    merges = []

    for _ in range(max_merges):
        # Split words for multiprocessing
        chunk_size = (len(symbol_words) + num_processes - 1) // num_processes
        chunks = [symbol_words[i:i+chunk_size] for i in range(0, len(symbol_words), chunk_size)]

        with Pool(num_processes) as pool:
            pair_counts_list = pool.map(count_pairs_chunk, chunks)

        pair_counts = merge_pair_counts(pair_counts_list)

        if not pair_counts:
            break

        # Find most frequent pair
        pair, freq = max(pair_counts.items(), key=lambda x: x[1])
        if freq < 2:
            break

        merges.append(pair)
        merged_symbol = pair[0] + pair[1]

        # Merge pair in all words sequentially
        for word in symbol_words:
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i+1]) == pair:
                    word[i:i+2] = [merged_symbol]
                else:
                    i += 1

        # Add merged symbol to vocab
        if merged_symbol not in token_to_id:
            token_to_id[merged_symbol] = len(token_to_id)
            vocab.append(merged_symbol)

    return vocab[:vocab_size], {"merges": merges, "token_to_id": token_to_id}

# ------------------- Parallel Tokenization -------------------
def tokenize_word_chunk(words_chunk, merges):
    merge_dict = {pair: pair[0] + pair[1] for pair in merges}
    tokens = []
    for word in words_chunk:
        w = word[:]
        merged = True
        while merged:
            merged = False
            i = 0
            new_w = []
            while i < len(w):
                if i < len(w)-1 and (w[i], w[i+1]) in merge_dict:
                    new_w.append(merge_dict[(w[i], w[i+1])])
                    i += 2
                    merged = True
                else:
                    new_w.append(w[i])
                    i += 1
            w = new_w
        tokens.extend(w)
    return tokens

def tokenize_parallel(text: str, tokenizer: Dict, num_processes=None):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)

    symbol_words = words_to_symbols(text)
    chunk_size = (len(symbol_words) + num_processes - 1) // num_processes
    chunks = [symbol_words[i:i+chunk_size] for i in range(0, len(symbol_words), chunk_size)]

    with Pool(num_processes) as pool:
        results = pool.starmap(tokenize_word_chunk, [(chunk, tokenizer["merges"]) for chunk in chunks])

    # Flatten results
    tokens = [tok for sublist in results for tok in sublist]
    return tokens

# ------------------- Detokenization -------------------
def detokenize(tokens: List[str]) -> str:
    text = "".join(tokens)
    text = text.replace(END_WORD, "")
    text = text.replace(SPACE_MARKER, " ")
    return text.strip()

# ------------------- File Saving -------------------
def save_vocab(vocab: List[str], rollno: str, vocab_size: int) -> None:
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def save_tokens(tokens: List[str], rollno: str) -> None:
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text: str, rollno: str) -> None:
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
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

    train_text = normalize_text(load_training_data(args.train))
    vocab, tokenizer = train_sp_tokenizer_parallel(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = normalize_text(f.read())

    tokens = tokenize_parallel(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)

    print("✅ Training and tokenization completed with multiprocessing!")