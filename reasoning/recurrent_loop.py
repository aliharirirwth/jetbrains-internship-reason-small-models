from typing import Tuple

import numpy as np

from reasoning.model import grid_to_features
from reasoning.sudoku import all_valid_moves, is_valid_grid

try:
    import torch
except ImportError:
    torch = None

# Recurrent loop: grid -> model -> one move -> updated grid. State passed between steps; K steps.


def run_step(
    model,
    grid: np.ndarray,
    device: str = "cpu",
) -> Tuple[np.ndarray, bool]:
    """One step: encode grid, run model, pick best valid (cell, digit), fill that cell.

    Args:
        model: Model with forward(x) -> (B, 16, 4) logits.
        grid: 4x4 current grid (0 = empty, 1-4 = digit).
        device: Device string. Defaults to "cpu".

    Returns:
        Tuple (new_grid, changed). new_grid is a copy with one cell filled, or unchanged if no valid move.
        changed is True iff a cell was filled.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if torch is None:
        raise ImportError("PyTorch required")
    model.eval()
    x = grid_to_features(grid)
    with torch.no_grad():
        inp = torch.from_numpy(x).float().unsqueeze(0).to(device)
        logits = model(inp)  # (1, 16, 4)
    logits = logits.cpu().numpy().squeeze(0)  # (16, 4)

    best_score = -1e9
    best_r, best_c, best_d = -1, -1, -1
    for cell in range(16):
        r, c = cell // 4, cell % 4
        if grid[r, c] != 0:
            continue
        valid = all_valid_moves(grid, r, c)
        if not valid:
            continue
        for d in valid:
            # logits: digit 0 -> value 1, digit 1 -> value 2, ...
            score = logits[cell, d - 1]
            if score > best_score:
                best_score = score
                best_r, best_c, best_d = r, c, d

    if best_r < 0:
        return grid.copy(), False
    new_grid = grid.copy()
    new_grid[best_r, best_c] = best_d
    return new_grid, True


def run_recurrent(
    model,
    puzzle: np.ndarray,
    max_steps: int,
    device: str = "cpu",
) -> np.ndarray:
    """Run the model recurrently for up to max_steps steps; stop if solved or no change.

    Args:
        model: Model with forward(x) -> (B, 16, 4).
        puzzle: 4x4 initial puzzle.
        max_steps: Maximum number of steps (depth).
        device: Device string. Defaults to "cpu".

    Returns:
        4x4 grid after at most max_steps updates (may be solved, partial, or unchanged).
    """
    grid = puzzle.copy()
    for _ in range(max_steps):
        if is_valid_grid(grid):
            break
        grid, changed = run_step(model, grid, device)
        if not changed:
            break
    return grid
