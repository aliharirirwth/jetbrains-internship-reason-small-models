import argparse
import json
import os

import numpy as np

from word2vec.corpus_utils import build_corpus_from_file, build_corpus_from_text
from word2vec.model import SkipGramNegSampling
from word2vec.train import train

# Generate word2vec figures: loss curve and 2D PCA of embeddings. Run: python -m word2vec.visualize

# Repeated so we get enough steps for a visible loss curve (~100+ steps with default batch_size)
DEMO_TEXT = (
    "the quick brown fox jumps over the lazy dog "
    "the dog and the fox are animals quick animals jump over lazy dogs "
    "brown foxes and lazy dogs the quick brown fox runs the lazy dog sleeps "
) * 8


def _pca2(X: np.ndarray) -> np.ndarray:
    """Project rows of X onto first 2 principal components (pure NumPy SVD).

    Args:
        X: Array of shape (n_samples, n_features).

    Returns:
        Array of shape (n_samples, 2).
    """
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return (X_centered @ Vt[:2].T).astype(np.float64)


def main() -> None:
    """Train word2vec, save loss curve and PCA embedding figure to save_dir."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", type=str, default="word2vec/figures")
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.file and os.path.isfile(args.file):
        corpus, id_to_word, word2id = build_corpus_from_file(args.file, min_count=2)
    else:
        text = args.text or DEMO_TEXT
        corpus, id_to_word, word2id = build_corpus_from_text(text, min_count=1)

    V = len(id_to_word)
    np.random.seed(args.seed)
    model = SkipGramNegSampling(V, args.dim, seed=args.seed)

    # Log every step when corpus is small so the loss curve has enough points
    est_steps = max(
        1, (corpus.n_tokens * 2 * 3) // min(32, max(1, corpus.n_tokens // 5)) * args.epochs
    )
    log_every = 1 if est_steps <= 100 else max(1, est_steps // 50)

    history = train(
        model,
        corpus,
        num_epochs=args.epochs,
        batch_size=min(32, max(1, corpus.n_tokens // 5)),
        window_size=3,
        num_negatives=5,
        lr=0.025,
        use_adagrad=True,
        use_lr_decay=True,
        subsample=False,  # keep all pairs so we get enough steps for a visible loss curve
        seed=args.seed,
        log_every=log_every,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Loss curve (ensure we have at least one point; use markers when few points)
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        if len(steps) == 0:
            plt.text(0.5, 0.5, "No steps logged", ha="center", va="center")
        else:
            kwargs = {"color": "C0"}
            if len(steps) <= 20:
                kwargs["marker"] = "o"
                kwargs["markersize"] = 4
            plt.plot(steps, losses, **kwargs)
            if len(steps) >= 2 and min(losses) != max(losses):
                plt.ylim(min(losses) - 0.05 * (max(losses) - min(losses)), max(losses) * 1.05)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("SGNS training loss")
        plt.tight_layout()
        loss_path = os.path.join(args.save_dir, "loss_curve.png")
        plt.savefig(loss_path, dpi=120)
        plt.close()
        print(f"Saved {loss_path}")
    except ImportError:
        print("matplotlib not installed; skipping loss curve. pip install matplotlib")

    # Save history for reproducibility
    with open(os.path.join(args.save_dir, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=0)

    # 2D PCA of input embeddings
    coords = _pca2(model.W_in)
    max_labels = min(50, V)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=20)
        for i in range(max_labels):
            plt.annotate(
                id_to_word[i],
                (coords[i, 0], coords[i, 1]),
                fontsize=7,
                alpha=0.9,
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Word2Vec embeddings (PCA)")
        plt.tight_layout()
        emb_path = os.path.join(args.save_dir, "embeddings_pca.png")
        plt.savefig(emb_path, dpi=120)
        plt.close()
        print(f"Saved {emb_path}")
    except ImportError:
        print("matplotlib not installed; skipping PCA plot. pip install matplotlib")


if __name__ == "__main__":
    main()
