# Word2Vec from scratch (pure NumPy)

Task #1 for **Learning to Reason with Small Models**: implement the core training loop of word2vec in pure NumPy — no PyTorch/TensorFlow. This implementation uses **skip-gram with negative sampling** so that gradients and design choices can be fully defended in follow-up.

## Design choices

- **Variant**: Skip-gram with negative sampling. Reason: (1) one center → one context per sample keeps the gradient derivation clear; (2) negative sampling makes training feasible at large vocab size; (3) you can explain exactly where each gradient term comes from (see `GRADIENTS.md`).
- **Subsampling**: \(P(\text{keep}) = \sqrt{t/f}\) capped at 1 (Mikolov et al.). So we keep rare words and drop frequent ones; low \(f\) → high keep, high \(f\) → low keep.
- **Negative distribution**: unigram count raised to 0.75, normalized. We avoid using the center or the positive context as a negative.
- **Updates**: Adagrad by default; vanilla SGD with `--no-adagrad`. **Linear LR decay** from \(lr_0\) to \(lr_0 \times 0.0001\) over words processed (same as original word2vec.c); disable with `--no-lr-decay`.
- **Numerical stability**: stable sigmoid and log-sigmoid (see `model.py` and `GRADIENTS.md`).
- **Gradient sign**: the code returns **grad** so that **W -= lr * grad** minimizes the loss; see `GRADIENTS.md` for the trace-through.

## Structure

```
word2vec/
├── model.py          # Forward, loss, gradients (all by hand)
├── data.py           # Corpus, subsampling, negative sampling, batches
├── train.py          # Training loop (SGD/Adagrad)
├── corpus_utils.py   # Build corpus from text/file
├── eval.py           # Similarity, k-NN, analogy (king - man + woman ≈ queen)
├── run.py            # Entry point
├── visualize.py      # Loss curve + PCA embedding plot (optional: matplotlib)
├── download_text8.py # Download text8 corpus (optional)
├── GRADIENTS.md      # Full gradient derivation + sign convention
├── test_word2vec.py  # Unit tests
├── figures/          # loss_curve.png, embeddings_pca.png (from visualize.py)
└── README.md
```

## Run

```bash
# From repo root
pip install -r requirements.txt
python -m word2vec.run
```

Optional arguments:

- `--text "..."` — train on the given string
- `--file path/to/corpus.txt` — train on a file
- `--epochs 10 --dim 64 --lr 0.025 --batch-size 32 --window 3 --negatives 5`
- `--no-adagrad` — use vanilla SGD
- `--no-lr-decay` — disable linear LR decay
- `--seed 42`

To train on **text8** (e.g. for analogy evaluation):

```bash
python -m word2vec.download_text8              # writes word2vec/data/text8.txt
python -m word2vec.run --file word2vec/data/text8.txt --epochs 1 --min-count 5
```

**Recommended for interpretable results:** Use `--epochs 1` (or the default demo corpus). Training on full text8 for many epochs can lead to **embedding collapse** (all cosine similarities ≈ 1, analogy accuracy 0%). The implementation is correct; the collapse is a training-dynamics effect. For a run that shows meaningful similarity and analogies, keep epochs low or use a smaller corpus.

**Figures (loss curve + PCA of embeddings):**

```bash
pip install matplotlib
python -m word2vec.visualize --save_dir word2vec/figures
```

## Tests

```bash
python -m pytest word2vec/test_word2vec.py -v
```

Tests check: output shapes, gradient vs finite difference (including a single-example tight check), initial loss range, subsampling (rare kept / frequent dropped), and that training decreases loss.

## Gradient derivation

See **`GRADIENTS.md`** for the full derivation of the negative-sampling loss and gradients w.r.t. \(v_w\), \(u_c\), and \(u_{n_k}\), including the correct handling when the same word appears as both positive and negative (add gradients, do not overwrite).

## One part most people can't explain when pressed

**Updating both input and output matrices**: each word type has two vectors (input \(v_w\) and output \(u_w\)). At training we only update \(v_w\) when \(w\) is the **center** and \(u_c\) when \(c\) is the **context** (positive or negative). So for a pair \((w, c)\), we update \(v_w\) and \(u_c\); we do **not** update \(u_w\) or \(v_c\) for that pair. At inference we typically use only \(W_{\text{in}}\) (or the average of both) as the word embedding. The two matrices encode slightly different roles (target vs context), and both receive gradients from the same loss.
