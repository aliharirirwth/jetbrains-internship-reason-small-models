from typing import List, Optional, Tuple

import numpy as np

# Evaluation: similarity (related vs unrelated), k-NN neighbours, analogy (a - b + c ≈ ?).


def l2_normalize(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize array along the given axis (zero vectors get divisor 1).

    Args:
        X: Input array.
        axis: Axis along which to normalize. Defaults to -1.

    Returns:
        Normalized array, same shape as X.
    """
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    return X / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (flattened).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Scalar in [-1, 1] (plus small epsilon in denominator).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def similarity_sanity_check(
    model,
    id_to_word: List[str],
    word2id: dict,
    pairs: Optional[List[tuple]] = None,
) -> None:
    """Print similarity for related vs unrelated pairs (sanity check).

    If pairs is None, uses first vocab words and a few random pairs for comparison.

    Args:
        model: Model with W_in attribute (V, D).
        id_to_word: List of words by id.
        word2id: Mapping word -> id.
        pairs: Optional list of (word1, word2) for related pairs. Defaults to None.
    """
    W = model.W_in
    V, D = W.shape
    if pairs is None:
        # Use any two words that exist
        words = list(word2id.keys())[: min(10, len(word2id))]
        if len(words) < 2:
            print("Eval: vocab too small for similarity check")
            return
        related = [(words[0], words[1])] if len(words) >= 2 else []
        unrelated = [(words[i % len(words)], words[(i + 3) % len(words)]) for i in range(3)]
    else:
        related = [(a, b) for a, b in pairs if a in word2id and b in word2id]
        unrelated = []
    if not related:
        related = [(id_to_word[0], id_to_word[1])] if V >= 2 else []
    if not unrelated and V >= 4:
        unrelated = [(id_to_word[0], id_to_word[3]), (id_to_word[1], id_to_word[2])]
    rel_sims = [cosine_similarity(W[word2id[a]], W[word2id[b]]) for a, b in related]
    unrel_sims = (
        [cosine_similarity(W[word2id[a]], W[word2id[b]]) for a, b in unrelated]
        if unrelated
        else [0.0]
    )
    print("Similarity sanity: related pairs", rel_sims, "unrelated", unrel_sims)
    return


def print_nearest(
    embeddings: np.ndarray,
    word2id: dict,
    id_to_word: List[str],
    k: int = 5,
    query_words: Optional[List[str]] = None,
) -> None:
    """Print k nearest neighbours (cosine) for given or default query words.

    Args:
        embeddings: (V, D) embedding matrix.
        word2id: Mapping word -> id.
        id_to_word: List of words by id.
        k: Number of neighbours to show. Defaults to 5.
        query_words: Words to query; if None, use first 3 vocab words. Defaults to None.
    """
    V, D = embeddings.shape
    E = l2_normalize(embeddings, axis=1)
    if query_words is None:
        query_words = [id_to_word[i] for i in range(min(3, V))]
    for w in query_words:
        if w not in word2id:
            continue
        i = word2id[w]
        sims = np.dot(E, E[i])
        sims[i] = -2.0  # exclude self
        nearest = np.argsort(-sims)[:k]
        nn_str = ", ".join(f"{id_to_word[j]}({sims[j]:.3f})" for j in nearest)
        print(f"  '{w}' -> {nn_str}")


def analogy(
    embeddings: np.ndarray,
    word2id: dict,
    id_to_word: List[str],
    a: str,
    b: str,
    c: str,
    k: int = 1,
) -> Optional[List[str]]:
    """Solve "a is to b as c is to ?" via vector offset; return k nearest (excluding a, b, c).

    Args:
        embeddings: (V, D) embedding matrix.
        word2id: Mapping word -> id.
        id_to_word: List of words by id.
        a: First word of analogy.
        b: Second word.
        c: Third word.
        k: Number of nearest neighbours to return. Defaults to 1.

    Returns:
        List of k nearest word strings, or None if any of a, b, c not in vocab.
    """
    for w in (a, b, c):
        if w not in word2id:
            return None
    ia, ib, ic = word2id[a], word2id[b], word2id[c]
    vec = embeddings[ia] - embeddings[ib] + embeddings[ic]
    vec = vec.reshape(1, -1)
    E = l2_normalize(embeddings, axis=1)
    vec_n = l2_normalize(vec, axis=1)
    sims = np.dot(E, vec_n.ravel())
    exclude = {ia, ib, ic}
    for idx in exclude:
        sims[idx] = -2.0
    nearest = np.argsort(-sims)[:k]
    return [id_to_word[j] for j in nearest]


# Small built-in set (no download). Expand or load from file for full evaluation.
DEFAULT_ANALOGIES = [
    ("king", "man", "woman", "queen"),
    ("paris", "france", "germany", "berlin"),
    ("big", "biggest", "small", "smallest"),
    ("run", "running", "walk", "walking"),
]


def run_analogy_eval(
    embeddings: np.ndarray,
    word2id: dict,
    id_to_word: List[str],
    analogies: Optional[List[Tuple[str, str, str, str]]] = None,
) -> Tuple[int, int]:
    """Run analogy evaluation on (a, b, c, expected_d) quadruples; print and return accuracy.

    Args:
        embeddings: (V, D) embedding matrix.
        word2id: Mapping word -> id.
        id_to_word: List of words by id.
        analogies: List of (a, b, c, expected) tuples. Defaults to DEFAULT_ANALOGIES.

    Returns:
        Tuple (correct_count, total_count) for analogies where a, b, c, expected are in vocab.
    """
    if analogies is None:
        analogies = DEFAULT_ANALOGIES
    correct = 0
    total = 0
    for a, b, c, expected in analogies:
        if expected not in word2id:
            continue
        preds = analogy(embeddings, word2id, id_to_word, a, b, c, k=1)
        if preds is None:
            continue
        total += 1
        if preds[0].lower() == expected.lower():
            correct += 1
        print(f"  {a} - {b} + {c} = {preds[0]} (expected {expected})")
    if total > 0:
        print(f"Analogy accuracy: {correct}/{total} = {100.0 * correct / total:.1f}%")
    else:
        print("  (no analogies in vocab; train on a larger corpus e.g. text8 for king/queen etc.)")
    return correct, total
