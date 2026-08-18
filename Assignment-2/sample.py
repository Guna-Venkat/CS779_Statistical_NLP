import argparse
import unicodedata
from typing import List, Tuple, Dict
import numpy as np
import re

# ------------------- Constants -------------------
END_WORD = "</w>"        # Special marker to denote end of sequence
SPACE_MARKER = "▁"       # Marker for spaces (used in SentencePiece-like tokenizers)


# ------------------- Text Preprocessing -------------------
def load_training_data(path: str) -> str:
    """
    Load text from a file.

    Args:
        path (str): Path to the file.

    Returns:
        str: Content of the file as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text: str) -> str:
    """
    Normalize text using Unicode NFKC normalization
    and convert to lowercase.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized lowercase text.
    """
    return unicodedata.normalize("NFKC", text).lower()


def preprocess_text(text: str) -> List[str]:
    """
    Convert text into a list of symbols for training.
    - Replaces spaces with SPACE_MARKER.
    - Appends END_WORD at the end.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of symbols (characters + markers).
    """
    symbols = []
    for char in text:
        if char == " " or char == "\n":
            symbols.append(SPACE_MARKER)
        else:
            symbols.append(char)
    symbols.append(END_WORD)
    return symbols


# ------------------- Train Tokenizer -------------------
def train_sp_tokenizer(symbols: List[str], vocab_size: int):
    """
    Train a simple SentencePiece-like BPE tokenizer.

    Args:
        symbols (List[str]): Preprocessed symbols from text.
        vocab_size (int): Target vocabulary size.

    Returns:
        Tuple[List[str], Dict]: 
            - Vocabulary list
            - Tokenizer dictionary (merges, token_to_id, id_to_symbol)
    """
    # Step 1: Collect unique symbols from input sequence
    unique_symbols = []
    seen = set()
    for s in symbols:
        if s not in seen:
            unique_symbols.append(s)
            seen.add(s)

    # Add special tokens (always included)
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", END_WORD]
    unique_symbols += special_tokens

    symbol_to_id = {s: i for i, s in enumerate(unique_symbols)}
    id_to_symbol = {i: s for s, i in symbol_to_id.items()}

    # Step 2: Convert the input sequence to IDs (numpy array for efficiency)
    seq = np.array([symbol_to_id[s] for s in symbols], dtype=np.int32)

    # Step 3: Initialize vocabulary and mappings
    vocab = unique_symbols[:]
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}

    merges: List[Tuple[int, int]] = []

    # Step 4: Count all adjacent symbol pairs
    left = seq[:-1]
    right = seq[1:]
    pair_array = np.stack([left, right], axis=1)
    # Efficient view to treat (l, r) as a single item
    pair_array_view = pair_array.view([('l', np.int32), ('r', np.int32)])
    unique_pairs, counts = np.unique(pair_array_view, return_counts=True)
    pair_counts = {(pair['l'], pair['r']): c for pair, c in zip(unique_pairs, counts)}

    # Step 5: Iteratively merge most frequent pairs until vocab_size reached
    next_id = len(token_to_id)
    while len(vocab) < vocab_size and pair_counts:
        # Find most frequent pair
        most_freq_pair, freq = max(pair_counts.items(), key=lambda x: x[1])
        if freq < 2:  # Stop if no frequent pairs remain
            break

        a, b = most_freq_pair
        merged_symbol = id_to_symbol[a] + id_to_symbol[b]

        # Add merged symbol to vocab if new
        if merged_symbol not in token_to_id:
            token_to_id[merged_symbol] = next_id
            id_to_symbol[next_id] = merged_symbol
            vocab.append(merged_symbol)
            next_id += 1

        # Save merge and remove processed pair
        merges.append((a, b))
        pair_counts.pop(most_freq_pair)

    return vocab[:vocab_size], {
        "merges": merges,
        "token_to_id": token_to_id,
        "id_to_symbol": id_to_symbol
    }


# ------------------- Lazy Encoding -------------------
def lazy_encode_word(
    word_seq: List[int],
    merges: List[Tuple[int, int]],
    id_to_symbol: Dict[int, str],
    token_to_id: Dict[str, int]
) -> List[int]:
    """
    Apply BPE merges lazily using IDs only.

    Args:
        word_seq (List[int]): Sequence of token IDs for a word.
        merges (List[Tuple[int,int]]): List of merge operations.
        id_to_symbol (Dict[int,str]): Map from ID -> symbol.
        token_to_id (Dict[str,int]): Map from symbol -> ID.

    Returns:
        List[int]: Encoded token sequence after merges.
    """
    # Build lookup: (left_id, right_id) -> merged_id
    next_id = max(id_to_symbol.keys()) + 1
    merge_lookup = {}
    for l, r in merges:
        merged_symbol = id_to_symbol[l] + id_to_symbol[r]
        if merged_symbol in token_to_id:
            merge_id = token_to_id[merged_symbol]
        else:
            # Assign new ID if not seen
            merge_id = next_id
            token_to_id[merged_symbol] = merge_id
            id_to_symbol[merge_id] = merged_symbol
            next_id += 1
        merge_lookup[(l, r)] = merge_id

    # Apply merges on sequence
    seq = word_seq[:]
    i = 0
    while i < len(seq) - 1:
        pair = (seq[i], seq[i + 1])
        if pair in merge_lookup:
            seq[i] = merge_lookup[pair]  # Replace left with merged
            del seq[i + 1]               # Remove right
            i = max(i - 1, 0)            # Re-check previous position
        else:
            i += 1
    return seq


# ------------------- Tokenization -------------------
def tokenize(
    text: str,
    token_to_id: Dict[str, int],
    merges: List[Tuple[int, int]],
    id_to_symbol: Dict[int, str]
) -> List[int]:
    """
    Tokenize input text into sequence of token IDs.

    Args:
        text (str): Input text.
        token_to_id (Dict[str,int]): Symbol-to-ID mapping.
        merges (List[Tuple[int,int]]): Merge operations.
        id_to_symbol (Dict[int,str]): ID-to-symbol mapping.

    Returns:
        List[int]: List of token IDs.
    """
    symbols = preprocess_text(text)
    word_seq = [token_to_id[s] for s in symbols]
    tokens = lazy_encode_word(word_seq, merges, id_to_symbol, token_to_id)
    return tokens


# ------------------- Detokenization -------------------
def detokenize(tokens: List[int], id_to_symbol: Dict[int, str]) -> str:
    """
    Convert token IDs back to human-readable text.

    Args:
        tokens (List[int]): List of token IDs.
        id_to_symbol (Dict[int,str]): ID-to-symbol mapping.

    Returns:
        str: Reconstructed text string.
    """
    text = "".join([id_to_symbol[t] for t in tokens])
    text = text.replace(END_WORD, "").replace(SPACE_MARKER, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------- File saving -------------------
def save_vocab(vocab: List[str], rollno: str, vocab_size: int):
    """
    Save vocabulary to a file.

    Args:
        vocab (List[str]): Vocabulary list.
        rollno (str): Roll number for filename.
        vocab_size (int): Vocabulary size for filename.
    """
    with open(f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))


def save_tokens(tokens: List[int], id_to_symbol: Dict[int, str], rollno: str) -> None:
    """
    Save tokenized output as actual token strings.

    Args:
        tokens (List[int]): List of token IDs.
        id_to_symbol (Dict[int,str]): ID-to-symbol mapping.
        rollno (str): Roll number for filename.
    """
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(id_to_symbol[t] + "\n")


def save_detokenized(text: str, rollno: str):
    """
    Save detokenized text to a file.

    Args:
        text (str): Detokenized text string.
        rollno (str): Roll number for filename.
    """
    with open(f"{rollno}_assignment2_sp_detokenized.txt", "w", encoding="utf-8") as f:
        f.write(text)


# ------------------- Main -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251140009"

    # Step 1: Load and preprocess training text
    train_text = normalize_text(load_training_data(args.train))
    preprocessed_symbols = preprocess_text(train_text)

    # Step 2: Train tokenizer
    vocab, tokenizer = train_sp_tokenizer(preprocessed_symbols, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Step 3: Load and preprocess input text
    sample_text = normalize_text(load_training_data(args.input))

    # Step 4: Tokenize
    tokens = tokenize(sample_text, tokenizer["token_to_id"], tokenizer["merges"], tokenizer["id_to_symbol"])
    save_tokens(tokens, tokenizer["id_to_symbol"], rollno)

    # Step 5: Detokenize
    detok_text = detokenize(tokens, tokenizer["id_to_symbol"])
    save_detokenized(detok_text, rollno)
