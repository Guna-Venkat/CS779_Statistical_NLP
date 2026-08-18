import argparse
import os

def load_training_data(train_path):
    """Load and return raw text for training."""
    pass

def train_bpe_tokenizer(text, vocab_size):
    """Learn BPE merges and return vocabulary."""
    pass

def save_vocab(vocab, rollno, vocab_size):
    """Save vocabulary file in required format."""
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize(text, tokenizer):
    """Tokenize input text using trained BPE model."""
    pass

def detokenize(tokens, tokenizer):
    """Detokenize tokens back to original text."""
    pass

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    # Replace with your actual roll number
    rollno = "<your_rollno_here>"

    # Training
    train_text = load_training_data(args.train)
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Tokenization
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    # Detokenization
    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
