import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import math

# ------------------- Constants -------------------
RESERVED_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]  # Special tokens
CONTINUATION = "##"  # Prefix for continuation subwords

# ------------------- Trie for efficient tokenization -------------------
class TrieNode:
    """A node in the Trie used for efficient token matching."""
    def __init__(self):
        self.children = {}  # Character → TrieNode mapping
        self.is_end = False  # True if this node marks the end of a token


class Trie:
    """Trie data structure for efficient subword matching in tokenization."""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token: str):
        """Insert a token into the Trie."""
        node = self.root
        for c in token:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def longest_match(self, word: str, start: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the longest token in the Trie starting from word[start:].

        Args:
            word (str): The word being tokenized.
            start (int): Starting index within the word.

        Returns:
            (end_index, matched_token): If match is found, end_index is the position 
                                        after the match and matched_token is the token.
                                        If no match, returns (None, None).
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

# ------------------- Load Data -------------------
def load_training_data(train_path: str) -> List[str]:
    """
    Load and split training data into words.

    Args:
        train_path (str): Path to the training text file.

    Returns:
        List[str]: List of words from the training text.
    """
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split()  # Split on whitespace

# ------------------- Initial Vocabulary -------------------
def get_initial_vocab(words: List[str]) -> List[str]:
    """
    Construct initial vocabulary consisting of characters + reserved tokens.

    Args:
        words (List[str]): Training words.

    Returns:
        List[str]: Initial vocabulary.
    """
    chars = set()
    for w in words:
        chars.update(w)  # Collect all unique characters
    return RESERVED_TOKENS + sorted(chars)

# ------------------- Corpus as lists + pair positions -------------------
def words_to_ids(words: List[str], token2id: Dict[str,int]):
    """
    Convert words into token IDs and record adjacent token pairs.

    Args:
        words (List[str]): List of training words.
        token2id (Dict[str,int]): Mapping from token string to ID.

    Returns:
        corpus_ids (List[List[int]]): Each word represented as a list of token IDs.
        pair_positions (defaultdict): Map of token pairs → positions where they occur.
    """
    corpus_ids = []
    pair_positions = defaultdict(set)

    for w_idx, w in enumerate(words):
        token_ids = [token2id[c] for c in w]  # word → char IDs
        corpus_ids.append(token_ids)
        for i in range(len(token_ids)-1):  # collect adjacent pairs
            pair = (token_ids[i], token_ids[i+1])
            pair_positions[pair].add((w_idx, i))
    return corpus_ids, pair_positions

# ------------------- Get pair counts -------------------
def get_pair_counts(pair_positions: Dict[Tuple[int,int], set]):
    """
    Count occurrences of token pairs.

    Args:
        pair_positions (dict): Mapping of pairs to positions.

    Returns:
        dict: Pair → frequency (only pairs with count > 1).
    """
    return {pair: len(pos_set) for pair, pos_set in pair_positions.items() if len(pos_set) > 1}

# ------------------- Merge pair -------------------
def merge_pair(corpus_ids, pair_positions, pair_to_merge, new_id):
    """
    Merge a specific token pair into a new token.

    Args:
        corpus_ids (List[List[int]]): Corpus as token IDs.
        pair_positions (dict): Pair → positions in corpus.
        pair_to_merge (Tuple[int,int]): The pair being merged.
        new_id (int): ID for the newly created token.
    """
    if pair_to_merge not in pair_positions:
        return

    positions = list(pair_positions[pair_to_merge])  # All positions of the pair
    pair_positions.pop(pair_to_merge)  # Remove old pair

    for w_idx, i in positions:
        word = corpus_ids[w_idx]
        # Skip if pair is no longer valid (already merged earlier)
        if i >= len(word)-1 or (word[i], word[i+1]) != pair_to_merge:
            continue

        # Merge pair into new token
        word[i] = new_id
        word.pop(i+1)

        # Update left neighbor pair
        if i > 0:
            left_pair_old = (word[i-1], pair_to_merge[0])
            pair_positions[left_pair_old].discard((w_idx, i-1))
            left_pair_new = (word[i-1], new_id)
            pair_positions[left_pair_new].add((w_idx, i-1))

        # Update right neighbor pair
        if i < len(word)-1:
            right_pair_old = (pair_to_merge[1], word[i+1])
            pair_positions[right_pair_old].discard((w_idx, i+1))
            right_pair_new = (new_id, word[i+1])
            pair_positions[right_pair_new].add((w_idx, i))


# ------------------- Compute ΔL -----------------------
def compute_delta_L(pair, f_new, f1, f2, N):
    # Avoid log(0)
    delta = 0.0
    if f_new > 0:
        delta += f_new * math.log(f_new / N)
    if f1 > 0:
        delta -= f1 * math.log(f1 / N)
    if f2 > 0:
        delta -= f2 * math.log(f2 / N)
    return delta

# ------------------- WordPiece Training -----------------------
def train_wordpiece_tokenizer(words: List[str], vocab_size: int):
    """
    Train a WordPiece tokenizer on a list of words using likelihood-maximizing merges.

    Args:
        words (List[str]): Training words.
        vocab_size (int): Desired vocabulary size.

    Returns:
        vocab (List[str]): Final vocabulary.
        tokenizer (dict): Dict with vocab, merges, and Trie.
    """
    vocab = get_initial_vocab(words)
    token2id = {tok: idx for idx, tok in enumerate(vocab)}
    id2token = {idx: tok for tok, idx in token2id.items()}

    # Initial corpus & pair positions
    corpus_ids, pair_positions = words_to_ids(words, token2id)

    # Token frequencies
    token_counts = defaultdict(int)
    for word in corpus_ids:
        for t in word:
            token_counts[t] += 1

    merges = []

    N = sum(token_counts.values())

    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(pair_positions)
        if not pair_counts:
            break

        # Compute ΔL for all candidate pairs
        best_delta, best_pair = float("-inf"), None
        for pair, f_new in pair_counts.items():
            f1 = token_counts[pair[0]]
            f2 = token_counts[pair[1]]
            delta = compute_delta_L(pair, f_new, f1, f2, N)
            if delta > best_delta or (delta == best_delta and pair < best_pair):
                best_delta = delta
                best_pair = pair

        # Stop if no positive ΔL
        if best_delta <= 0:
            break

        # Form new token string
        t1, t2 = id2token[best_pair[0]], id2token[best_pair[1]]
        if t1.startswith(CONTINUATION) or t2.startswith(CONTINUATION):
            new_token = CONTINUATION + t1.lstrip(CONTINUATION) + t2.lstrip(CONTINUATION)
        else:
            new_token = t1 + t2

        # Register new token
        new_id = len(vocab)
        vocab.append(new_token)
        token2id[new_token] = new_id
        id2token[new_id] = new_token
        merges.append(best_pair)

        # Update corpus with merge
        merge_pair(corpus_ids, pair_positions, best_pair, new_id)

        # Update token counts
        token_counts[new_id] = pair_counts[best_pair]
        token_counts[best_pair[0]] -= pair_counts[best_pair]
        token_counts[best_pair[1]] -= pair_counts[best_pair]
        N = N - pair_counts[best_pair]  # total token occurrences updated

    # Build Trie for fast tokenization
    trie = Trie()
    for tok in vocab:
        trie.insert(tok)

    tokenizer = {"vocab": vocab, "merges": merges, "trie": trie}
    return vocab, tokenizer

# ------------------- Tokenization -------------------
def tokenize(text: str, tokenizer: Dict) -> List[str]:
    """
    Tokenize input text using trained WordPiece tokenizer.

    Args:
        text (str): Input text.
        tokenizer (dict): Trained tokenizer with Trie.

    Returns:
        List[str]: List of tokens.
    """
    output_tokens = []
    trie: Trie = tokenizer["trie"]

    for word in text.split():
        i = 0
        sub_tokens = []
        while i < len(word):
            end, match = trie.longest_match(word, i)
            if match is None:
                sub_tokens.append("<unk>")  # Unknown token
                i += 1
            else:
                # Add continuation marker if subword is not at word start
                if i > 0 and not match.startswith(CONTINUATION):
                    match = CONTINUATION + match
                sub_tokens.append(match)
                i = end
        output_tokens.extend(sub_tokens)
    return output_tokens


def detokenize(tokens: List[str], tokenizer: Dict) -> str:
    """
    Convert tokens back into natural text.

    Args:
        tokens (List[str]): Tokenized sequence.
        tokenizer (dict): Trained tokenizer.

    Returns:
        str: Detokenized text.
    """
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
    """Save vocabulary to file."""
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname,"w",encoding="utf-8") as f:
        for t in vocab: 
            f.write(t+"\n")


def save_tokens(tokens: List[str], rollno: str) -> None:
    """Save tokenized sequence to file."""
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname,"w",encoding="utf-8") as f:
        for t in tokens: 
            f.write(t+"\n")


def save_detokenized(text: str, rollno: str) -> None:
    """Save detokenized text to file."""
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
    with open(fname,"w",encoding="utf-8") as f:
        f.write(text)

# ------------------- Main -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and run a WordPiece tokenizer.")
    parser.add_argument("--train", type=str, required=True, help="Path to training text file")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--vocab_size", type=int, required=True, help="Target vocabulary size")
    args = parser.parse_args()

    rollno = "251140009"  # Student ID or identifier

    # Train tokenizer
    words = load_training_data(args.train)
    vocab, tokenizer = train_wordpiece_tokenizer(words, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Tokenize input text
    with open(args.input,"r",encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    # Detokenize tokens back to text
    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
