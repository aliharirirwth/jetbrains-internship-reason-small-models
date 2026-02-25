# Learning to Reason with Small Models — Roadmap

All deliverables are implemented; nothing is left in "planned" state. This document lists every item and its status.

---

## Task #1 — Word2vec (pure NumPy)

| Item | Status | Location / Command |
|------|--------|--------------------|
| Skip-gram with negative sampling (forward, loss, gradients, updates) | ✅ Done | `word2vec/model.py`, `word2vec/data.py`, `word2vec/train.py` |
| No PyTorch/TensorFlow | ✅ Done | NumPy only |
| Subsampling, negative sampling, Adagrad, LR decay | ✅ Done | `word2vec/data.py`, `word2vec/train.py` |
| Gradient derivation documented | ✅ Done | `word2vec/GRADIENTS.md` |
| Unit tests (shapes, grad check, subsampling, training) | ✅ Done | `word2vec/test_word2vec.py`, `pytest word2vec/ -v` |
| Loss curve + embedding figure | ✅ Done | `python -m word2vec.visualize --save_dir word2vec/figures` |
| Run on demo / text8 | ✅ Done | `python -m word2vec.run` |

---

## Phase 1 — MLM pretraining (accuracy vs compute)

| Item | Status | Location / Command |
|------|--------|--------------------|
| Dataset: WikiText-103 (or subset) | ✅ Done | `mlm/data.py`, HuggingFace `datasets` |
| BERT-style small model (MLM head) | ✅ Done | `mlm/models/bert_mlm.py` |
| TRM: Transformer Recurrent Model (matched capacity) | ✅ Done | `mlm/models/trm.py` |
| MLM training loop (masked LM loss, perplexity) | ✅ Done | `mlm/train.py` |
| Matched compute budget (same steps or same param-steps proxy) | ✅ Done | `mlm/run_compute_experiment.py` |
| Accuracy vs compute figure | ✅ Done | `mlm/figures/accuracy_vs_compute.png` |
| Reproduce instructions | ✅ Done | `mlm/README.md` |

**Metrics:** Perplexity (or token accuracy) vs total training steps (compute proxy). Compare BERT vs TRM under the same budget.

---

## Phase 2 — Recurrent reasoning (depth vs accuracy)

| Item | Status | Location / Command |
|------|--------|--------------------|
| Task: Sudoku (4×4) | ✅ Done | `reasoning/sudoku.py` |
| BERT-like model embedded in recurrent loop | ✅ Done | `reasoning/model.py` (`SudokuBERT`), `reasoning/recurrent_loop.py` |
| TRM (Transformer Recurrent Model) in recurrent loop | ✅ Done | `reasoning/model.py` (`SudokuTRM`) |
| State passed between steps | ✅ Done | Grid → model → one move → updated grid |
| Depth vs accuracy experiment (BERT + TRM + MLP) | ✅ Done | `python -m reasoning.run_depth_experiment --model all --out_dir reasoning/figures` |
| Depth vs accuracy figure | ✅ Done | `reasoning/figures/depth_vs_accuracy.png` |
| Reproduce instructions | ✅ Done | `reasoning/README.md` |

---

## Repository layout (all implemented)

| Path | Description |
|------|-------------|
| `word2vec/` | Task #1: Skip-gram with negative sampling (NumPy) |
| `mlm/` | Phase 1: MLM pretraining, BERT vs TRM, accuracy vs compute |
| `reasoning/` | Phase 2: Recurrent Sudoku, depth vs accuracy |
| `ROADMAP.md` | This file: full checklist and status |

---

## How to run everything

```bash
# Task #1
pip install -r requirements.txt
python -m word2vec.run
python -m word2vec.visualize --save_dir word2vec/figures

# Phase 1 (MLM)
pip install -r mlm/requirements.txt
python -m mlm.run_compute_experiment --out_dir mlm/figures

# Phase 2 (Reasoning)
pip install torch numpy matplotlib
python -m reasoning.run_depth_experiment --out_dir reasoning/figures
```

---

## Optional / future extensions (not required for submission)

- PubMed abstracts in addition to WikiText-103
- 9×9 Sudoku (Phase 2) or Sudoku-Extreme benchmark
- FLOP counting for exact compute (Phase 1)
- Notebooks for analysis and narrative
