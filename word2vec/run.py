import argparse

import numpy as np

from word2vec.corpus_utils import build_corpus_from_file, build_corpus_from_text
from word2vec.eval import print_nearest, run_analogy_eval, similarity_sanity_check
from word2vec.model import SkipGramNegSampling
from word2vec.train import train

# Entry point: train word2vec on demo corpus or file. Usage: python -m word2vec.run [--file path]

DEMO_TEXT = """
the quick brown fox jumps over the lazy dog
the dog and the fox are animals
quick animals jump over lazy dogs
brown foxes and lazy dogs
the quick brown fox runs
the lazy dog sleeps
""".replace("\n", " ").strip()


def main():
    """Train word2vec on demo corpus or file; print similarity and analogy evaluation."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None, help="Train on this string")
    ap.add_argument("--file", type=str, default=None, help="Train on file (one big text)")
    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="For large corpora (e.g. text8) use 1-2 for a quicker run",
    )
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-adagrad", action="store_true", help="Use vanilla SGD")
    ap.add_argument(
        "--no-lr-decay", action="store_true", help="Disable linear LR decay (Mikolov-style)"
    )
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    if args.file:
        corpus, id_to_word, word2id = build_corpus_from_file(args.file, min_count=args.min_count)
    else:
        text = args.text or DEMO_TEXT
        corpus, id_to_word, word2id = build_corpus_from_text(text, min_count=args.min_count)

    V = len(id_to_word)
    print(f"Vocab size {V}, corpus tokens {corpus.n_tokens}")

    np.random.seed(args.seed)
    model = SkipGramNegSampling(V, args.dim, seed=args.seed)
    train(
        model,
        corpus,
        num_epochs=args.epochs,
        batch_size=min(args.batch_size, max(1, corpus.n_tokens // 10)),
        window_size=args.window,
        num_negatives=args.negatives,
        lr=args.lr,
        use_adagrad=not args.no_adagrad,
        use_lr_decay=not args.no_lr_decay,
        subsample=True,
        seed=args.seed,
        log_every=50,
    )

    similarity_sanity_check(model, id_to_word, word2id)
    print_nearest(model.W_in, word2id, id_to_word, k=5)
    print("Analogy (a - b + c = ?):")
    run_analogy_eval(model.W_in, word2id, id_to_word)


if __name__ == "__main__":
    main()
