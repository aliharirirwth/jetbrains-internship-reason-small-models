from typing import List, Optional, Tuple

import numpy as np

from word2vec.data import Corpus

# Build a Corpus from raw text (space-separated tokens). For demo use a tiny inline corpus
# or load from file. No external tokenizer required.


def tokenize_simple(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric; keep only letter/digit sequences.

    Args:
        text: Raw input string.

    Returns:
        List of token strings.
    """
    import re

    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return tokens


def tokens_to_ids(tokens: List[str], word2id: dict) -> np.ndarray:
    """Map token strings to vocabulary ids; OOV maps to word2id.get(w, 0).

    Args:
        tokens: List of token strings.
        word2id: Mapping from token string to integer id.

    Returns:
        One-dimensional array of integer ids (int64).
    """
    return np.array([word2id.get(w, 0) for w in tokens], dtype=np.int64)


def build_corpus_from_text(
    text: str,
    min_count: int = 2,
    max_vocab: Optional[int] = None,
    subsample_t: float = 1e-5,
) -> Tuple[Corpus, List[str], dict]:
    """Tokenize text, build vocabulary, and return a Corpus with id_to_word and word2id.

    Args:
        text: Raw text (space-separated or any format; tokenized by regex).
        min_count: Minimum token count to include in vocabulary. Defaults to 2.
        max_vocab: Maximum vocabulary size (by frequency). Defaults to None (no cap).
        subsample_t: Subsampling threshold for Corpus. Defaults to 1e-5.

    Returns:
        Tuple of (corpus, id_to_word, word2id).

    Raises:
        ValueError: If text yields no tokens.
    """
    tokens = tokenize_simple(text)
    if not tokens:
        raise ValueError("No tokens in text")
    # Build vocab from token strings: we need unique tokens and counts
    from collections import Counter

    cnt = Counter(tokens)
    # Apply min_count and max_vocab
    kept = [w for w, c in cnt.most_common(max_vocab or len(cnt)) if c >= min_count]
    word2id = {w: i for i, w in enumerate(kept)}
    id_to_word = kept
    V = len(kept)
    counts = np.zeros(V, dtype=np.float64)
    for w, c in cnt.items():
        if w in word2id:
            counts[word2id[w]] = c
    token_ids = np.array([word2id.get(w, 0) for w in tokens], dtype=np.int64)
    # build_vocab expects token_ids and returns word_ids (remapped), counts, id_to_word
    # We already built our own vocab; so we can just build Corpus from token_ids and counts.
    # But token_ids might have 0 for OOV - we filtered so all tokens are in vocab. So token_ids = word_ids.
    corpus = Corpus(token_ids, counts, subsample_t=subsample_t)
    return corpus, id_to_word, word2id


def build_corpus_from_file(
    path: str,
    min_count: int = 2,
    max_vocab: Optional[int] = None,
    subsample_t: float = 1e-5,
) -> Tuple[Corpus, List[str], dict]:
    """Build a Corpus from a text file (read as a single blob).

    Args:
        path: Path to the text file.
        min_count: Minimum token count for vocabulary. Defaults to 2.
        max_vocab: Maximum vocabulary size. Defaults to None.
        subsample_t: Subsampling threshold for Corpus. Defaults to 1e-5.

    Returns:
        Tuple of (corpus, id_to_word, word2id).
    """
    with open(path) as f:
        text = f.read()
    return build_corpus_from_text(
        text, min_count=min_count, max_vocab=max_vocab, subsample_t=subsample_t
    )


def build_corpus_from_tokens(
    tokens: List[str],
    min_count: int = 2,
    max_vocab: Optional[int] = None,
    subsample_t: float = 1e-5,
) -> Tuple[Corpus, List[str], dict]:
    """Build a Corpus from a list of token strings (same logic as build_corpus_from_text).

    Args:
        tokens: List of token strings.
        min_count: Minimum token count for vocabulary. Defaults to 2.
        max_vocab: Maximum vocabulary size. Defaults to None.
        subsample_t: Subsampling threshold for Corpus. Defaults to 1e-5.

    Returns:
        Tuple of (corpus, id_to_word, word2id).
    """
    from collections import Counter

    cnt = Counter(tokens)
    kept = [w for w, c in cnt.most_common(max_vocab or len(cnt)) if c >= min_count]
    word2id = {w: i for i, w in enumerate(kept)}
    id_to_word = kept
    V = len(kept)
    counts = np.zeros(V, dtype=np.float64)
    for w, c in cnt.items():
        if w in word2id:
            counts[word2id[w]] = c
    token_ids = np.array([word2id.get(w, 0) for w in tokens], dtype=np.int64)
    corpus = Corpus(token_ids, counts, subsample_t=subsample_t)
    return corpus, id_to_word, word2id
