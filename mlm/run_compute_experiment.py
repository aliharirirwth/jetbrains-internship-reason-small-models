import argparse
import json
import os

import torch

from mlm.data import get_tokenizer, load_wikitext, tokenize_and_chunk
from mlm.models.bert_mlm import get_bert_mlm
from mlm.models.trm import TRMForMaskedLM
from mlm.train import train_mlm

# Phase 1 experiment: BERT vs TRM under matched compute (same steps). Output: accuracy vs compute.


def main():
    """Run BERT vs TRM under matched compute; save accuracy-vs-compute figure and JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="mlm/figures")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--max_examples", type=int, default=3000)
    ap.add_argument("--wikitext", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print("Loading WikiText and tokenizing...")
    texts = load_wikitext(name=args.wikitext, split="train", max_examples=args.max_examples)
    tokenizer = get_tokenizer("bert-base-uncased")
    chunks = tokenize_and_chunk(texts, tokenizer, max_length=128)
    print(f"  {len(chunks)} chunks")

    vocab_size = tokenizer.vocab_size
    hidden_size = 256
    num_layers_bert = 4
    num_steps_trm = 4

    results = []

    # BERT
    print("Training BERT (small)...")
    bert = get_bert_mlm(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers_bert,
        num_attention_heads=4,
        intermediate_size=1024,
    )
    hist_bert = train_mlm(
        bert,
        chunks,
        tokenizer,
        device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=5e-5,
        log_every=100,
        seed=args.seed,
    )
    for h in hist_bert:
        results.append(
            {"model": "bert", "step": h["step"], "ppl": h["ppl"], "accuracy": 1.0 / h["ppl"]}
        )

    # TRM (often needs lower LR for stability with recurrence)
    print("Training TRM (recurrent)...")
    trm = TRMForMaskedLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=4,
        intermediate_size=1024,
        num_recurrent_steps=num_steps_trm,
    )
    hist_trm = train_mlm(
        trm,
        chunks,
        tokenizer,
        device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=1e-4,
        log_every=100,
        seed=args.seed + 1,
    )
    for h in hist_trm:
        results.append(
            {"model": "trm", "step": h["step"], "ppl": h["ppl"], "accuracy": 1.0 / h["ppl"]}
        )

    with open(os.path.join(args.out_dir, "accuracy_vs_compute.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot: step vs 1/ppl (accuracy) for both models
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps_bert = [r["step"] for r in results if r["model"] == "bert"]
        acc_bert = [r["accuracy"] for r in results if r["model"] == "bert"]
        steps_trm = [r["step"] for r in results if r["model"] == "trm"]
        acc_trm = [r["accuracy"] for r in results if r["model"] == "trm"]

        plt.figure(figsize=(6, 4))
        plt.plot(steps_bert, acc_bert, marker="o", label="BERT-small", markersize=4)
        plt.plot(steps_trm, acc_trm, marker="s", label="TRM", markersize=4)
        plt.xlabel("Training steps (compute budget)")
        plt.ylabel("Accuracy (1 / perplexity)")
        plt.title("Phase 1: Accuracy vs compute (WikiText MLM)")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(args.out_dir, "accuracy_vs_compute.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Saved {path}")
    except ImportError:
        print("matplotlib not installed; skipping plot.")

    return results


if __name__ == "__main__":
    main()
