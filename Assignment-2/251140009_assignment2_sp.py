import argparse
import unicodedata
from typing import List, Tuple, Dict
import numpy as np

# ------------------- Constants -------------------
END_WORD = "</w>"
SPACE_MARKER = "▁"

# ------------------- Text Preprocessing -------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()

def preprocess_text(text: str) -> List[str]:
    """
    Convert text into list of symbols.
    SPACE_MARKER for space, END_WORD at end.
    """
    symbols = [SPACE_MARKER]  # start-of-text
    for char in text:
        if char == " ":
            symbols.append(SPACE_MARKER)
        else:
            symbols.append(char)
    symbols.append(END_WORD)
    return symbols

# ------------------- Train Tokenizer -------------------
def train_sp_tokenizer(symbols: List[str], vocab_size: int):
    # Step 1: Map symbols to IDs
    unique_symbols = []
    seen = set()
    for s in symbols:
        if s not in seen:
            unique_symbols.append(s)
            seen.add(s)

    # Add special tokens at the end
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", END_WORD]
    unique_symbols += special_tokens

    symbol_to_id = {s:i for i,s in enumerate(unique_symbols)}
    id_to_symbol = {i:s for s,i in symbol_to_id.items()}

    # Step 2: Convert full sequence to numpy array
    seq = np.array([symbol_to_id[s] for s in symbols], dtype=np.int32)

    # Step 3: Initialize vocab
    vocab = unique_symbols[:]
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}

    merges: List[Tuple[int,int]] = []

    # Step 4: Count initial pairs
    left = seq[:-1]
    right = seq[1:]
    pair_array = np.stack([left,right],axis=1)
    pair_array_view = pair_array.view([('l',np.int32),('r',np.int32)])
    unique_pairs, counts = np.unique(pair_array_view, return_counts=True)
    pair_counts = {(pair['l'], pair['r']): c for pair, c in zip(unique_pairs, counts)}

    # Step 5: Iterative merges
    next_id = len(token_to_id)
    while len(vocab) < vocab_size and pair_counts:
        # Most frequent pair
        most_freq_pair, freq = max(pair_counts.items(), key=lambda x: x[1])
        if freq < 2: break

        a,b = most_freq_pair
        merged_symbol = id_to_symbol[a] + id_to_symbol[b]

        # Add new merged symbol to vocab
        if merged_symbol not in token_to_id:
            token_to_id[merged_symbol] = next_id
            id_to_symbol[next_id] = merged_symbol
            vocab.append(merged_symbol)
            next_id += 1

        # Save merge as IDs
        merges.append((a,b))
        pair_counts.pop(most_freq_pair)

    return vocab[:vocab_size], {"merges": merges, "token_to_id": token_to_id, "id_to_symbol": id_to_symbol}

# ------------------- Lazy Encoding -------------------
def lazy_encode_word(word_seq: List[int], merges: List[Tuple[int,int]], 
                     id_to_symbol: Dict[int,str], token_to_id: Dict[str,int]):
    """
    Apply BPE merges lazily using IDs only.
    Returns list of IDs.
    """
    # Build lookup: (left_id, right_id) -> merged_id
    next_id = max(id_to_symbol.keys()) + 1
    merge_lookup = {}
    for l, r in merges:
        merged_symbol = id_to_symbol[l] + id_to_symbol[r]
        if merged_symbol in token_to_id:
            merge_id = token_to_id[merged_symbol]
        else:
            merge_id = next_id
            token_to_id[merged_symbol] = merge_id
            id_to_symbol[merge_id] = merged_symbol
            next_id += 1
        merge_lookup[(l, r)] = merge_id

    # Apply merges on word_seq
    seq = word_seq[:]
    i = 0
    while i < len(seq) - 1:
        pair = (seq[i], seq[i+1])
        if pair in merge_lookup:
            seq[i] = merge_lookup[pair]
            del seq[i+1]
            i = max(i-1,0)
        else:
            i += 1
    return seq

# ------------------- Tokenization -------------------
def tokenize(text: str, token_to_id: Dict[str,int], merges: List[Tuple[int,int]], id_to_symbol: Dict[int,str]) -> List[int]:
    symbols = preprocess_text(text)
    word_seq = [token_to_id[s] for s in symbols]
    tokens = lazy_encode_word(word_seq, merges, id_to_symbol, token_to_id)
    return tokens

# ------------------- Detokenization -------------------
def detokenize(tokens: List[int], id_to_symbol: Dict[int,str]) -> str:
    text = "".join([id_to_symbol[t] for t in tokens])
    text = text.replace(END_WORD, "").replace(SPACE_MARKER, " ").strip()
    return text

# ------------------- File saving -------------------
def save_vocab(vocab: List[str], rollno: str, vocab_size: int):
    with open(f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt","w",encoding="utf-8") as f:
        f.write("\n".join(vocab))

def save_tokens(tokens: List[int], id_to_symbol: Dict[int,str], rollno: str) -> None:
    """Save tokenized output as actual token strings"""
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(id_to_symbol[t] + "\n")

def save_detokenized(text: str, rollno: str):
    with open(f"{rollno}_assignment2_sp_detokenized.txt","w",encoding="utf-8") as f:
        f.write(text)

# ------------------- Main -------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251140009"

    # Load and preprocess
    train_text = normalize_text(load_text(args.train))
    preprocessed_symbols = preprocess_text(train_text)

    # Train tokenizer
    vocab, tokenizer = train_sp_tokenizer(preprocessed_symbols, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Load input text
    sample_text = normalize_text(load_text(args.input))

    # Tokenize
    tokens = tokenize(sample_text, tokenizer["token_to_id"], tokenizer["merges"], tokenizer["id_to_symbol"])
    save_tokens(tokens, tokenizer["id_to_symbol"], rollno)

    # Detokenize
    detok_text = detokenize(tokens, tokenizer["id_to_symbol"])
    save_detokenized(detok_text, rollno)
