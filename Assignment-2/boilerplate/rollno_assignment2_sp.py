import argparse
import os
import unicodedata

def load_training_data(train_path):
    pass

def normalize_text(text):
    """Apply Unicode NFKC normalization and lowercasing."""
    return unicodedata.normalize("NFKC", text).lower()

def train_sp_tokenizer(text, vocab_size):
    """Adapt BPE training with space marker ▁."""
    pass

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize(text, tokenizer, seed=42):
    """Implement sampling-based tokenization (deterministic if seed fixed)."""
    pass

def detokenize(tokens, tokenizer):
    pass

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "<your_rollno_here>"

    train_text = load_training_data(args.train)
    train_text = normalize_text(train_text)
    vocab, tokenizer = train_sp_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = normalize_text(f.read())
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
