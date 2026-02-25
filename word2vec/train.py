from typing import Optional

import numpy as np

from word2vec.data import Corpus, negative_sampling_distribution, skipgram_batches
from word2vec.model import SkipGramNegSampling

# Training loop: Adagrad or SGD, optional linear LR decay (Mikolov et al.). Pure NumPy.


def train(
    model: SkipGramNegSampling,
    corpus: Corpus,
    *,
    num_epochs: int = 1,
    batch_size: int = 128,
    window_size: int = 5,
    num_negatives: int = 5,
    lr: float = 0.025,
    lr_min_ratio: float = 0.0001,
    use_adagrad: bool = True,
    use_lr_decay: bool = True,
    subsample: bool = True,
    seed: Optional[int] = None,
    log_every: int = 1000,
) -> list:
    """Train with Adagrad (default) or SGD; optional linear LR decay over words processed.

    LR decays from lr to lr * lr_min_ratio over the run when use_lr_decay is True (Mikolov et al.).

    Args:
        model: SkipGramNegSampling instance (modified in place).
        corpus: Corpus to train on.
        num_epochs: Number of passes over the corpus. Defaults to 1.
        batch_size: Batch size for skip-gram pairs. Defaults to 128.
        window_size: Context window size. Defaults to 5.
        num_negatives: Number of negative samples per positive. Defaults to 5.
        lr: Initial learning rate. Defaults to 0.025.
        lr_min_ratio: Minimum LR as fraction of initial (for decay). Defaults to 0.0001.
        use_adagrad: Whether to use Adagrad. Defaults to True.
        use_lr_decay: Whether to apply linear LR decay. Defaults to True.
        subsample: Whether to subsample corpus. Defaults to True.
        seed: Random seed. Defaults to None.
        log_every: Log and record history every this many steps. Defaults to 1000.

    Returns:
        List of dicts with keys "step", "loss", "lr" for plotting.
    """
    rng = np.random.default_rng(seed)
    neg_probs = negative_sampling_distribution(corpus.counts)
    history = []
    step = 0
    words_processed = 0
    # Approximate total (center, context) pairs per epoch for LR schedule
    total_pairs = 0
    if use_lr_decay:
        pairs_per_epoch = corpus.n_tokens * max(1, 2 * window_size)
        total_pairs = pairs_per_epoch * num_epochs
        if subsample:
            total_pairs = int(total_pairs * 0.5)
    # Estimate total steps so user knows when training ends (approx: pairs per epoch / batch_size * epochs)
    pairs_per_epoch_est = corpus.n_tokens * max(1, 2 * window_size) * (0.5 if subsample else 1.0)
    steps_per_epoch_est = max(1, int(pairs_per_epoch_est / batch_size))
    total_steps_est = steps_per_epoch_est * num_epochs
    print(
        f"Training: {num_epochs} epochs, ~{total_steps_est} steps total (est. ~{steps_per_epoch_est} steps/epoch)"
    )

    if use_adagrad:
        G_in = np.zeros_like(model.W_in)
        G_out = np.zeros_like(model.W_out)

    last_step, last_loss, last_lr = 0, 0.0, lr
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        batch_gen = skipgram_batches(
            corpus,
            batch_size,
            window_size,
            num_negatives,
            neg_probs,
            subsample=subsample,
            seed=int(rng.integers(0, 2**31)) if seed is not None else None,
        )
        for center, context_pos, context_neg in batch_gen:
            B = center.shape[0]
            if use_lr_decay and total_pairs > 0:
                lr_current = lr * max(lr_min_ratio, 1.0 - words_processed / (total_pairs + 1))
            else:
                lr_current = lr
            loss, dW_in, dW_out = model.forward_backward(center, context_pos, context_neg)
            if use_adagrad:
                G_in += dW_in**2
                G_out += dW_out**2
                model.W_in -= lr_current * dW_in / (np.sqrt(G_in) + 1e-10)
                model.W_out -= lr_current * dW_out / (np.sqrt(G_out) + 1e-10)
            else:
                model.W_in -= lr_current * dW_in
                model.W_out -= lr_current * dW_out
            words_processed += B
            step += 1
            if step % log_every == 0:
                history.append({"step": step, "loss": float(loss), "lr": lr_current})
                print(f"step {step} loss {loss:.4f} lr {lr_current:.6f}")
        last_step, last_loss, last_lr = step, float(loss), lr_current
    # Ensure we have at least the final step for plotting (avoids empty loss curve)
    if step > 0 and (not history or history[-1]["step"] != last_step):
        history.append({"step": last_step, "loss": last_loss, "lr": last_lr})
    return history
