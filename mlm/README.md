# Phase 1 — MLM pretraining (accuracy vs compute)

We compare two model families under a **matched compute budget** (same number of training steps) on masked language modeling (WikiText):

- **BERT-small:** 4-layer Transformer encoder, 256 hidden, 4 heads.
- **TRM (Transformer Recurrent Model):** 1 layer applied **recurrently** 4 times (same hidden size); fewer parameters, same conceptual depth via recurrence.

**Outcome:** A single plot of **accuracy (1/perplexity) vs compute (training steps)** for both models, showing how each architecture uses the same budget.

## Quick start

```bash
# From repo root
pip install -r mlm/requirements.txt
python -m mlm.run_compute_experiment --out_dir mlm/figures
```

This downloads WikiText-2 (smaller than 103 for speed), tokenizes, and trains both BERT and TRM for 500 steps each, then saves:

- `mlm/figures/accuracy_vs_compute.png`
- `mlm/figures/accuracy_vs_compute.json`

### TRM curve near zero

With default hyperparameters, the TRM curve often sits near zero (accuracy = 1/perplexity). TRM’s perplexity stays very high in this MLM setup, so the plot is correct—it reflects that TRM may need extra tuning (e.g. learning rate, initialization, or more steps) for comparable perplexity to BERT.

## Options

- `--max_steps 1000` — more training steps.
- `--wikitext wikitext-103-v1` — full WikiText-103 (use `--max_examples 10000` to keep time reasonable).
- `--max_examples 5000` — more training lines.
- `--batch_size 16` — larger batches.

## Design

- **Matched compute:** Same number of optimizer steps and batch size for both models so the x-axis (steps) is a fair compute proxy.
- **Matched capacity:** Same hidden size (256) and similar depth (4 layers BERT vs 4 recurrent steps TRM).
- **Metric:** Perplexity on MLM; we plot 1/perplexity as “accuracy” so higher is better.
