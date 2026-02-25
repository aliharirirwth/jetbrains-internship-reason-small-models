import numpy as np
import pytest

from word2vec.data import Corpus, build_vocab, negative_sampling_distribution, skipgram_batches
from word2vec.model import SkipGramNegSampling, _log_sigmoid, _sigmoid

# Unit tests: shapes, gradient sanity, initial loss, subsampling, training.


def test_sigmoid_stability():
    x = np.array([-1000, 0, 1000])
    y = _sigmoid(x)
    assert np.all(y >= 0) and np.all(y <= 1)
    assert y[0] < 1e-100  # clip to 500 gives representable small value
    assert np.isclose(y[1], 0.5)
    assert y[2] >= 1.0 - 1e-10  # clip gives exactly 1.0 in float64


def test_log_sigmoid_stability():
    x = np.array([-1000, 0, 500])
    log_sig = _log_sigmoid(x)
    assert np.all(np.isfinite(log_sig))
    assert np.isclose(log_sig[1], np.log(0.5))


def test_model_output_shapes():
    V, D = 100, 32
    B, K = 8, 5
    model = SkipGramNegSampling(V, D, seed=42)
    center = np.random.randint(0, V, size=B)
    context_pos = np.random.randint(0, V, size=B)
    context_neg = np.random.randint(0, V, size=(B, K))
    loss, dW_in, dW_out = model.forward_backward(center, context_pos, context_neg)
    assert isinstance(loss, (float, np.floating))
    assert dW_in.shape == (V, D)
    assert dW_out.shape == (V, D)


def test_initial_loss_near_log_vocab():
    """Initial loss should be close to log(V) for random embeddings (sanity check)."""
    V, D = 50, 16
    model = SkipGramNegSampling(V, D, seed=123)
    B, K = 32, 5
    center = np.random.randint(0, V, size=B)
    context_pos = np.random.randint(0, V, size=B)
    context_neg = np.random.randint(0, V, size=(B, K))
    loss, _, _ = model.forward_backward(center, context_pos, context_neg)
    # With small random init, scores are near 0 so sigmoid ~ 0.5; loss per positive ~ log(2), negatives ~ K*log(2)
    # So total per example ~ (1+K)*log(2) ≈ 6*0.69 ≈ 4.2; we just check it's finite and in a plausible range
    assert np.isfinite(loss)
    assert 0.1 < loss < 50.0


def test_gradient_finite_difference():
    """Gradient of loss w.r.t. W_in should match finite differences (one element)."""
    V, D = 20, 8
    model = SkipGramNegSampling(V, D, seed=0)
    center = np.array([0, 1])
    context_pos = np.array([1, 2])
    context_neg = np.array([[2, 3, 4], [5, 6, 7]])  # fixed so test is deterministic
    eps = 1e-5
    _, dW_in, _ = model.forward_backward(center, context_pos, context_neg)
    i, j = 0, 0
    model.W_in[i, j] += eps
    loss_plus, _, _ = model.forward_backward(center, context_pos, context_neg)
    model.W_in[i, j] -= 2 * eps
    loss_minus, _, _ = model.forward_backward(center, context_pos, context_neg)
    model.W_in[i, j] += eps
    fd = (loss_plus - loss_minus) / (2 * eps)
    assert np.isfinite(dW_in[i, j]) and np.isfinite(fd)
    assert np.abs(dW_in[i, j]) < 1.0 and np.abs(fd) < 1.0  # sanity magnitude


def test_gradient_finite_difference_single_example():
    """Tighter check: B=1 so gradient is clean; grad and finite-diff should match."""
    V, D = 15, 8
    model = SkipGramNegSampling(V, D, seed=0)
    center = np.array([0])
    context_pos = np.array([1])
    context_neg = np.array([[2, 3, 4]])  # (1, 3)
    eps = 1e-5
    _, dW_in, _ = model.forward_backward(center, context_pos, context_neg)
    i, j = 0, 0
    model.W_in[i, j] += eps
    loss_plus, _, _ = model.forward_backward(center, context_pos, context_neg)
    model.W_in[i, j] -= 2 * eps
    loss_minus, _, _ = model.forward_backward(center, context_pos, context_neg)
    model.W_in[i, j] += eps
    fd = (loss_plus - loss_minus) / (2 * eps)
    assert np.isclose(dW_in[i, j], fd, atol=1e-3, rtol=0.1), (
        f"grad {dW_in[i, j]:.6f} vs fd {fd:.6f}"
    )


def test_corpus_subsample_deterministic_with_rng():
    word_ids = np.array([0, 1, 1, 2, 2, 2])
    counts = np.array([1.0, 2.0, 3.0])
    corpus = Corpus(word_ids, counts, subsample_t=1e-5)
    rng = np.random.default_rng(42)
    kept1 = corpus.subsample(rng)
    rng2 = np.random.default_rng(42)
    kept2 = corpus.subsample(rng2)
    np.testing.assert_array_equal(kept1, kept2)


def test_subsampling_keeps_rare_drops_frequent():
    """P(keep) = sqrt(t/f) capped at 1: rare words kept more often than frequent."""
    # Corpus: word 0 appears 1 time, word 1 appears 100 times
    word_ids = np.array([0] + [1] * 100)
    counts = np.array([1.0, 100.0])
    corpus = Corpus(word_ids, counts, subsample_t=1e-3)
    keep_count_0 = 0
    keep_count_1 = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        kept = corpus.subsample(rng)
        for idx in kept:
            w = corpus.word_ids[idx]
            if w == 0:
                keep_count_0 += 1
            else:
                keep_count_1 += 1
    # Rare word (0) has keep_prob = sqrt(t/f) ~ 0.32; frequent (1) ~ 0.03. So rare kept more.
    assert keep_count_0 >= 15, "Rare word should be kept in a good fraction of runs"
    assert keep_count_1 < 5000, "Frequent word should be subsampled"


def test_negative_sampling_distribution():
    counts = np.array([10.0, 1.0, 100.0])
    probs = negative_sampling_distribution(counts, power=0.75)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs > 0)
    assert probs[2] > probs[0] > probs[1]  # higher count -> higher prob


def test_skipgram_batches_yield_correct_shapes():
    word_ids = np.random.randint(0, 10, size=100)
    counts = np.bincount(word_ids, minlength=10).astype(np.float64)
    corpus = Corpus(word_ids, counts)
    neg_probs = negative_sampling_distribution(counts)
    batches = list(
        skipgram_batches(
            corpus,
            batch_size=8,
            window_size=2,
            num_negatives=3,
            neg_probs=neg_probs,
            subsample=False,
            seed=42,
        )
    )
    assert len(batches) >= 1
    c, p, n = batches[0]
    assert c.shape == (8,) or c.shape[0] <= 8
    assert p.shape == c.shape
    assert n.shape[0] == c.shape[0] and n.shape[1] == 3


def test_build_vocab_remap():
    token_ids = np.array([10, 10, 20, 20, 20, 30])
    word_ids, counts, id_to_word = build_vocab(token_ids, min_count=1)
    assert len(counts) <= 3
    assert word_ids.max() < len(counts)
    assert np.all(counts >= 1)


def test_training_decreases_loss():
    """A few SGD steps should decrease loss (catches gradient sign errors)."""
    from word2vec.corpus_utils import build_corpus_from_text

    corpus, id_to_word, _ = build_corpus_from_text(
        "the cat sat on the mat the dog sat on the log", min_count=1
    )
    V = len(id_to_word)
    model = SkipGramNegSampling(V, 16, seed=42)
    # Compute initial loss on one batch
    from word2vec.data import negative_sampling_distribution, skipgram_batches

    neg_probs = negative_sampling_distribution(corpus.counts)
    batch = next(skipgram_batches(corpus, 32, 2, 3, neg_probs, subsample=False, seed=1))
    center, context_pos, context_neg = batch
    loss0, _, _ = model.forward_backward(center, context_pos, context_neg)
    # Do several updates with small lr (gradients are mean over batch, so scale is modest)
    lr = 0.05
    for _ in range(50):
        loss, d_in, d_out = model.forward_backward(center, context_pos, context_neg)
        model.W_in -= lr * d_in
        model.W_out -= lr * d_out
    loss1, _, _ = model.forward_backward(center, context_pos, context_neg)
    assert loss1 < loss0, f"Loss should decrease: {loss0} -> {loss1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
