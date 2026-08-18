import argparse
from collections import Counter
from typing import List, Tuple, Dict, Set

END_WORD = "</w>"


def load_training_data(train_path: str) -> List[str]:
    """
    Load text from a file and split it into words.
    """
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    return words


def word_to_symbols(word: str, initial_tokens: Set[str] = None) -> Tuple[str, ...]:
    """
    Convert a word into a tuple of symbols for BPE.
    """
    if initial_tokens and word in initial_tokens:
        return (word + END_WORD,)
    # represent word as UTF-8 byte chars
    byte_symbols = [chr(b) for b in word.encode("utf-8")]
    return tuple(byte_symbols + [END_WORD])


def replace_pair(symbols: Tuple[str, ...], pair: Tuple[str, str], merged_symbol: str) -> Tuple[str, ...]:
    """
    Replace all occurrences of a symbol pair with a merged symbol.
    """
    a, b = pair
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged_symbol)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe_tokenizer(
    words: List[str],
    vocab_size: int,
    top_word_count: int = 5000,
    verbose: bool = False
) -> Tuple[List[str], Dict]:
    """
    Train a BPE tokenizer on a list of words.
    """
    freq = Counter(words)

    # Step 1: Preserve top frequent words
    top_words = set(w for w, _ in freq.most_common(top_word_count))

    # Step 2: Initialize symbols for all words
    current_symbols = {w: word_to_symbols(w, top_words) for w in freq.keys()}

    # Step 3: Collect initial single-character symbols
    initial_symbols = sorted({s for syms in current_symbols.values() for s in syms})
    include_end_word = END_WORD in initial_symbols
    num_initial = len(initial_symbols)
    max_merges = max(0, vocab_size - 4 - num_initial)  # 4 reserved tokens

    merges: List[str] = []
    merge_pairs: List[Tuple[str, str]] = []

    # Step 4: Count initial pairs weighted by word frequency
    pair_counts = Counter()
    for w, syms in current_symbols.items():
        f = freq[w]
        for i in range(len(syms) - 1):
            pair_counts[(syms[i], syms[i + 1])] += f

    # Step 5: Iteratively perform merges
    for _ in range(max_merges):
        if not pair_counts:
            break
        max_count = max(pair_counts.values())
        if max_count < 2:
            break

        # Tie-breaker: lexicographic
        candidates = [p for p, c in pair_counts.items() if c == max_count]
        chosen_pair = min(candidates)
        merged_symbol = chosen_pair[0] + chosen_pair[1]

        # Replace the chosen pair everywhere
        new_pair_counts = Counter()
        for w, syms in current_symbols.items():
            if chosen_pair in zip(syms, syms[1:]):
                new_syms = replace_pair(syms, chosen_pair, merged_symbol)
                current_symbols[w] = new_syms
            else:
                new_syms = syms
            f = freq[w]
            for i in range(len(new_syms) - 1):
                new_pair_counts[(new_syms[i], new_syms[i + 1])] += f

        pair_counts = new_pair_counts
        merges.append(merged_symbol)
        merge_pairs.append(chosen_pair)

        if verbose:
            print(f"Merged {chosen_pair} -> {merged_symbol}")

    # Step 6: Build vocabulary
    vocab = ["<pad>", "<unk>", "<s>", "</s>"] + initial_symbols
    if include_end_word and END_WORD not in vocab:
        vocab.append(END_WORD)
    vocab += merges

    # Trim if vocab got too large
    vocab = vocab[:vocab_size]

    token_to_id = {token: idx for idx, token in enumerate(vocab)}

    return vocab, {"merges": merge_pairs, "token_to_id": token_to_id, "top_words": top_words}


def tokenize(text: str, tokenizer: Dict) -> List[str]:
    """
    Tokenize input text using the trained BPE tokenizer.
    """
    tokens_out: List[str] = []
    merges = tokenizer["merges"]
    merge_dict = {pair: pair[0] + pair[1] for pair in merges}
    top_words = tokenizer.get("top_words", set())

    for word in text.split():
        s = word_to_symbols(word, top_words)
        merged = True
        while merged:
            merged = False
            for pair, merged_symbol in merge_dict.items():
                if pair in zip(s, s[1:]):
                    s = replace_pair(s, pair, merged_symbol)
                    merged = True
        tokens_out.extend(s)
    return tokens_out


def detokenize(tokens: List[str]) -> str:
    """
    Convert BPE tokens back into text.
    """
    words = []
    cur = []
    for tok in tokens:
        if tok.endswith(END_WORD):
            tok_clean = tok.replace(END_WORD, "")
            if cur:
                cur.append(tok_clean)
                # Reconstruct from bytes
                word_bytes = bytes([ord(c) for c in cur if c])
                words.append(word_bytes.decode("utf-8"))
                cur = []
            else:
                if tok_clean:
                    words.append(tok_clean)  # preserved whole word
        else:
            if tok:
                cur.append(tok)

    if cur:
        word_bytes = bytes([ord(c) for c in cur if c])
        words.append(word_bytes.decode("utf-8"))

    return " ".join(words)


def save_vocab(vocab: List[str], rollno: str, vocab_size: int) -> None:
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")


def save_tokens(tokens: List[str], rollno: str) -> None:
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")


def save_detokenized(text: str, rollno: str) -> None:
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training text file")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file for tokenization")
    parser.add_argument("--vocab_size", type=int, required=True, help="Desired vocabulary size")
    args = parser.parse_args()

    rollno = "251140009"

    # Step 1: Load training data
    words = load_training_data(args.train)

    # Step 2: Train BPE tokenizer
    vocab, tokenizer = train_bpe_tokenizer(words, args.vocab_size)

    # Step 3: Save vocabulary
    save_vocab(vocab, rollno, args.vocab_size)

    # Step 4: Tokenize input text
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    # Step 5: Detokenize tokens and save
    detok_text = detokenize(tokens)
    save_detokenized(detok_text, rollno)
