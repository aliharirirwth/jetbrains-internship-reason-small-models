from typing import Iterator, List, Optional, Tuple

import numpy as np

# Data pipeline for skip-gram: vocab, subsampling (sqrt(t/f) capped at 1), unigram^0.75
# negatives. We do not use the same word as its own negative.


class Corpus:
    """Tokenized corpus as word indices; supports subsampling and skip-gram iteration.

    Attributes:
        word_ids (np.ndarray): 1D array of vocabulary indices for the full corpus.
        counts (np.ndarray): 1D array of length vocab_size; counts[i] is count of word i.
        subsample_t (float): Subsampling threshold (Mikolov: sqrt(t/f) capped at 1).
        n_tokens (int): Length of corpus (len(word_ids)).
    """

    def __init__(self, word_ids: np.ndarray, counts: np.ndarray, subsample_t: float = 1e-5):
        """Initialize corpus from token ids and vocabulary counts.

        Args:
            word_ids: 1D array of vocabulary indices for the whole corpus.
            counts: 1D array of length vocab_size; counts[i] = count of word i.
            subsample_t: Subsampling threshold; typical 1e-5. Higher = more drop. Defaults to 1e-5.
        """
        self.word_ids = np.asarray(word_ids, dtype=np.int64)
        self.counts = np.asarray(counts, dtype=np.float64)
        self.subsample_t = subsample_t
        self.n_tokens = len(self.word_ids)

    def subsample(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Probabilistic subsampling of frequent words (Mikolov et al.: P(keep) = sqrt(t/f) cap 1).

        Args:
            rng: Random generator for reproducibility. Defaults to None (new default_rng).

        Returns:
            Indices of kept tokens (positions in corpus), not word ids.
        """
        if rng is None:
            rng = np.random.default_rng()
        total_count = self.counts.sum()
        if total_count <= 0:
            return np.arange(self.n_tokens)
        freqs = self.counts[self.word_ids] / total_count
        # Avoid div by zero; then keep_prob = sqrt(t/f) capped at 1
        keep_prob = np.sqrt(self.subsample_t / np.clip(freqs, 1e-12, None))
        keep_prob = np.minimum(keep_prob, 1.0)
        r = rng.random(self.n_tokens)
        return np.where(r < keep_prob)[0]

    def iter_skipgram_pairs(
        self,
        window_size: int,
        subsample: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[int, int]]:
        """Yield (center_word_id, context_word_id) pairs from the corpus.

        Symmetric window: context in [center - window_size, center + window_size], excluding center.
        If subsample=True, subsample first then iterate over kept indices in order.

        Args:
            window_size: Half-window size (each side).
            subsample: Whether to subsample before iterating. Defaults to True.
            seed: Random seed for subsampling. Defaults to None.

        Yields:
            Tuples (center_id, context_id).
        """
        rng = np.random.default_rng(seed)
        if subsample:
            kept = self.subsample(rng)
            # Sort so we traverse corpus in order (optional but reproducible)
            kept = np.sort(kept)
        else:
            kept = np.arange(self.n_tokens)

        for idx in kept:
            center = self.word_ids[idx]
            start = max(0, idx - window_size)
            end = min(self.n_tokens, idx + window_size + 1)
            for j in range(start, end):
                if j == idx:
                    continue
                context = self.word_ids[j]
                yield (center, context)


def build_vocab(
    token_ids: np.ndarray,
    min_count: int = 5,
    max_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build vocabulary from token ids; remap to 0..V-1 and filter by count/size.

    Keeps tokens with count >= min_count; optionally caps vocabulary at max_size by frequency.
    OOV positions in the corpus get id 0.

    Args:
        token_ids: 1D array of integer token ids (from any tokenizer).
        min_count: Minimum count to include a token. Defaults to 5.
        max_size: Maximum vocabulary size (by frequency). Defaults to None.

    Returns:
        Tuple of (word_ids, counts, id_to_word). word_ids is remapped corpus; counts shape (V,);
        id_to_word is list of length V (placeholder strings in this implementation).
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    unique, cnt = np.unique(token_ids, return_counts=True)
    # Sort by count descending for max_size truncation
    order = np.argsort(-cnt)
    unique = unique[order]
    cnt = cnt[order]
    # min_count filter
    mask = cnt >= min_count
    unique = unique[mask]
    cnt = cnt[mask]
    if max_size is not None and len(unique) > max_size:
        unique = unique[:max_size]
        cnt = cnt[:max_size]
    V = len(unique)
    old_to_new = np.full(unique.max() + 1, -1, dtype=np.int64)
    old_to_new[unique] = np.arange(V)
    # Remap corpus
    in_vocab = np.isin(token_ids, unique)
    word_ids = np.where(in_vocab, old_to_new[token_ids], 0)  # use 0 for OOV if any
    counts = np.zeros(V, dtype=np.float64)
    for i, c in zip(unique, cnt):
        counts[old_to_new[i]] = c
    id_to_word = [str(i) for i in range(V)]  # placeholder; real impl would pass word list
    return word_ids, counts, id_to_word


def negative_sampling_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """Unigram distribution raised to power and normalized (Mikolov et al.: power=0.75).

    Args:
        counts: 1D array of vocabulary counts.
        power: Exponent for counts; 0.75 is standard. Defaults to 0.75.

    Returns:
        1D array of probabilities (sum 1), same length as counts.
    """
    probs = np.power(np.maximum(counts, 1e-10), power)
    probs /= probs.sum()
    return probs


def skipgram_batches(
    corpus: Corpus,
    batch_size: int,
    window_size: int,
    num_negatives: int,
    neg_probs: np.ndarray,
    subsample: bool = True,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield batches of (center, context_pos, context_neg) for skip-gram training.

    Negatives are sampled from neg_probs; center and positive are never used as negatives.

    Args:
        corpus: Corpus instance to iterate over.
        batch_size: Number of (center, context) pairs per batch.
        window_size: Window size for iter_skipgram_pairs.
        num_negatives: Number of negative samples per positive (K).
        neg_probs: 1D probability distribution over vocab for negative sampling.
        subsample: Whether to subsample corpus. Defaults to True.
        seed: Random seed. Defaults to None.

    Yields:
        Tuples (center, context_pos, context_neg) with shapes (B,), (B,), (B, K).
    """
    rng = np.random.default_rng(seed)
    V = len(neg_probs)
    centers: List[int] = []
    positives: List[int] = []
    for c, p in corpus.iter_skipgram_pairs(window_size, subsample=subsample, seed=seed):
        centers.append(c)
        positives.append(p)
        if len(centers) >= batch_size:
            centers_arr = np.array(centers, dtype=np.int64)
            pos_arr = np.array(positives, dtype=np.int64)
            # Negative samples: (B, K); ensure we don't use center as negative
            negs = rng.choice(V, size=(batch_size, num_negatives), p=neg_probs)
            for i in range(batch_size):
                bad = (negs[i] == centers_arr[i]) | (negs[i] == pos_arr[i])
                while np.any(bad):
                    negs[i, bad] = rng.choice(V, size=bad.sum(), p=neg_probs)
                    bad = (negs[i] == centers_arr[i]) | (negs[i] == pos_arr[i])
            yield centers_arr, pos_arr, negs
            centers = []
            positives = []
    if centers:
        n = len(centers)
        centers_arr = np.array(centers, dtype=np.int64)
        pos_arr = np.array(positives, dtype=np.int64)
        negs = rng.choice(V, size=(n, num_negatives), p=neg_probs)
        for i in range(n):
            bad = (negs[i] == centers_arr[i]) | (negs[i] == pos_arr[i])
            while np.any(bad):
                negs[i, bad] = rng.choice(V, size=bad.sum(), p=neg_probs)
                bad = (negs[i] == centers_arr[i]) | (negs[i] == pos_arr[i])
        yield centers_arr, pos_arr, negs
