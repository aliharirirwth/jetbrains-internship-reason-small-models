import argparse
import json
import os

import numpy as np

from reasoning.model import SudokuBERT, SudokuPredictor, SudokuTRM, grid_to_features
from reasoning.recurrent_loop import run_recurrent
from reasoning.sudoku import cell_accuracy, generate_puzzles

# Phase 2: vary depth K, measure accuracy. --model mlp|bert|trm|all. Plots depth vs accuracy.


def build_model(model_type: str, device, seed: int = 42):
    """Build a Sudoku model by type; all have forward(x) -> (B, 16, 4).

    Args:
        model_type: One of "mlp", "bert", "trm".
        device: Torch device to place the model on.
        seed: Random seed. Defaults to 42.

    Returns:
        Model instance on the given device.

    Raises:
        ValueError: If model_type is not mlp, bert, or trm.
    """
    if model_type == "mlp":
        return SudokuPredictor(hidden=128, num_layers=2, seed=seed).to(device)
    if model_type == "bert":
        return SudokuBERT(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            seed=seed,
        ).to(device)
    if model_type == "trm":
        return SudokuTRM(
            hidden_size=128,
            num_attention_heads=4,
            intermediate_size=256,
            num_recurrent_steps=4,
            seed=seed,
        ).to(device)
    raise ValueError(f"Unknown model: {model_type}. Use mlp, bert, or trm.")


def train_epoch(model, optimizer, puzzles, solutions, device):
    """One epoch: cross-entropy on empty cells (predict solution digit from puzzle).

    Args:
        model: Sudoku model (modified in place).
        optimizer: Torch optimizer.
        puzzles: List of 4x4 puzzle arrays.
        solutions: List of 4x4 solution arrays.
        device: Device string.

    Returns:
        Mean loss over batches (one batch per puzzle).
    """
    model.train()
    total_loss = 0.0
    n = 0
    for puzzle, solution in zip(puzzles, solutions):
        x = grid_to_features(puzzle)
        import torch

        inp = torch.from_numpy(x).float().unsqueeze(0).to(device)
        target = solution.ravel()  # 0-4, we only care about 1-4 for CE
        logits = model(inp).squeeze(0)  # (16, 4)
        # For each cell: if empty in puzzle, target is solution[cell]-1 (0-3)
        loss = 0.0
        count = 0
        for cell in range(16):
            if puzzle.ravel()[cell] == 0:
                t = int(target[cell]) - 1  # 0-3
                loss = loss + torch.nn.functional.cross_entropy(
                    logits[cell : cell + 1], torch.tensor([t], device=logits.device)
                )
                count += 1
        if count > 0:
            loss = loss / count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def evaluate_depth(model, puzzles, solutions, depth: int, device: str) -> float:
    """Run model recurrently for depth steps on each puzzle; return mean cell accuracy.

    Accuracy is over initially empty cells only (mask = puzzle == 0).

    Args:
        model: Sudoku model.
        puzzles: List of 4x4 puzzles.
        solutions: List of 4x4 solutions.
        depth: Number of recurrent steps (max_steps).
        device: Device string.

    Returns:
        Mean accuracy in [0, 1].
    """
    model.eval()
    accs = []
    for puzzle, solution in zip(puzzles, solutions):
        mask = puzzle == 0
        if mask.sum() == 0:
            accs.append(1.0)
            continue
        pred = run_recurrent(model, puzzle, max_steps=depth, device=device)
        acc = cell_accuracy(pred, solution, mask)
        accs.append(acc)
    return float(np.mean(accs))


def main():
    """Train and evaluate MLP/BERT/TRM on 4x4 Sudoku; plot depth vs accuracy; save figure and JSON."""
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="reasoning/figures")
    ap.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["mlp", "bert", "trm", "all"],
        help="Model to train and evaluate: mlp, bert, trm, or all (compare BERT + TRM + MLP)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_size", type=int, default=200)
    ap.add_argument("--val_size", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--depths", type=str, default="1,2,4,8,16")
    ap.add_argument(
        "--no_train", action="store_true", help="Skip training, use random init (for quick plot)"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = generate_puzzles(args.train_size, num_holes=8, seed=args.seed)
    val_data = generate_puzzles(args.val_size, num_holes=8, seed=args.seed + 1)
    train_puzzles = [p for p, _ in train_data]
    train_solutions = [s for _, s in train_data]
    val_puzzles = [p for p, _ in val_data]
    val_solutions = [s for _, s in val_data]

    models_to_run = ["mlp", "bert", "trm"] if args.model == "all" else [args.model]
    all_results = {}

    for model_type in models_to_run:
        print(f"\n--- Model: {model_type.upper()} ---")
        model = build_model(model_type, device, seed=args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if not args.no_train:
            for epoch in range(args.epochs):
                loss = train_epoch(model, optimizer, train_puzzles, train_solutions, device)
                if (epoch + 1) % 5 == 0:
                    acc1 = evaluate_depth(model, val_puzzles, val_solutions, depth=1, device=device)
                    print(f"Epoch {epoch + 1} loss {loss:.4f} val_acc(depth=1) {acc1:.4f}")
        else:
            print("Skipping training (--no_train); using random init.")

        depths = [int(d) for d in args.depths.split(",")]
        results = []
        for k in depths:
            acc = evaluate_depth(model, val_puzzles, val_solutions, depth=k, device=device)
            results.append({"depth": k, "accuracy": acc})
            print(f"Depth {k} accuracy {acc:.4f}")
        all_results[model_type] = results

    # Save JSON: single model or dict of model -> results
    out_json = os.path.join(args.out_dir, "depth_vs_accuracy.json")
    if len(models_to_run) == 1:
        with open(out_json, "w") as f:
            json.dump(all_results[models_to_run[0]], f, indent=2)
    else:
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        for model_type in models_to_run:
            res = all_results[model_type]
            label = {"mlp": "MLP", "bert": "BERT-Small", "trm": "TRM"}[model_type]
            plt.plot(
                [r["depth"] for r in res], [r["accuracy"] for r in res], marker="o", label=label
            )
        plt.xlabel("Recurrent depth (steps)")
        plt.ylabel("Cell accuracy (initially empty)")
        plt.title("Depth vs accuracy (4×4 Sudoku)")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(args.out_dir, "depth_vs_accuracy.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Saved {path}")
    except ImportError:
        print("matplotlib not installed; skipping plot.")

    return all_results


if __name__ == "__main__":
    main()
