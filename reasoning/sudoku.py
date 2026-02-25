from typing import List, Tuple

import numpy as np

# 4x4 Sudoku: grid representation, validity, puzzle generation. Digits 1-4, 2x2 blocks. 0 = empty.


def is_valid_grid(grid: np.ndarray) -> bool:
    """Check if a 4x4 grid is a valid solution (rows, cols, 2x2 blocks each have 1-4).

    Args:
        grid: 4x4 array with values in 0..4.

    Returns:
        True if valid solution, False otherwise.
    """
    if grid.shape != (4, 4):
        return False
    for i in range(4):
        row = grid[i, :]
        if not _valid_unit(row):
            return False
        col = grid[:, i]
        if not _valid_unit(col):
            return False
    for br in range(0, 4, 2):
        for bc in range(0, 4, 2):
            block = grid[br : br + 2, bc : bc + 2].ravel()
            if not _valid_unit(block):
                return False
    return True


def _valid_unit(arr: np.ndarray) -> bool:
    """Unit (row/col/block) has exactly 1,2,3,4 (no zeros)."""
    return set(arr.ravel()) == {1, 2, 3, 4}


def all_valid_moves(grid: np.ndarray, r: int, c: int) -> List[int]:
    """Return list of valid digits (1-4) for cell (r, c) given current grid state.

    Args:
        grid: 4x4 array.
        r: Row index 0..3.
        c: Column index 0..3.

    Returns:
        List of digits that can be placed without violating row/col/block constraints.
    """
    if grid[r, c] != 0:
        return []
    used = set()
    for j in range(4):
        if grid[r, j] != 0:
            used.add(grid[r, j])
    for i in range(4):
        if grid[i, c] != 0:
            used.add(grid[i, c])
    br, bc = 2 * (r // 2), 2 * (c // 2)
    for i in range(br, br + 2):
        for j in range(bc, bc + 2):
            if grid[i, j] != 0:
                used.add(grid[i, j])
    return [d for d in range(1, 5) if d not in used]


def generate_solved() -> np.ndarray:
    """Generate a random valid 4x4 Sudoku solution via backtracking.

    Returns:
        4x4 array with values 1-4, valid in rows, columns, and 2x2 blocks.
    """
    while True:
        grid = np.zeros((4, 4), dtype=np.int64)
        if _fill(grid, 0, 0):
            return grid
    return grid


def _fill(grid: np.ndarray, r: int, c: int) -> bool:
    """Backtrack fill; return True if solved."""
    if r == 4:
        return True
    next_r, next_c = (r, c + 1) if c < 3 else (r + 1, 0)
    if grid[r, c] != 0:
        return _fill(grid, next_r, next_c)
    opts = all_valid_moves(grid, r, c)
    np.random.shuffle(opts)
    for d in opts:
        grid[r, c] = d
        if _fill(grid, next_r, next_c):
            return True
        grid[r, c] = 0
    return False


def punch_holes(solution: np.ndarray, num_holes: int, rng: np.random.Generator) -> np.ndarray:
    """Return a puzzle by zeroing num_holes randomly chosen cells in a copy of solution.

    Args:
        solution: 4x4 valid solution.
        num_holes: Number of cells to clear (max 16).
        rng: Random generator for reproducibility.

    Returns:
        4x4 puzzle (copy of solution with some cells set to 0).
    """
    puzzle = solution.copy()
    positions = [(i, j) for i in range(4) for j in range(4)]
    rng.shuffle(positions)
    for idx in range(min(num_holes, 16)):
        i, j = positions[idx]
        puzzle[i, j] = 0
    return puzzle


def generate_puzzles(
    num_puzzles: int, num_holes: int = 8, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (puzzle, solution) pairs; each puzzle has num_holes empty cells.

    Args:
        num_puzzles: Number of pairs to generate.
        num_holes: Number of cells to clear per puzzle. Defaults to 8.
        seed: Random seed. Defaults to 42.

    Returns:
        List of (puzzle, solution) tuples; puzzle is solution with num_holes zeros.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_puzzles):
        solution = generate_solved()
        puzzle = punch_holes(solution, num_holes, rng)
        out.append((puzzle, solution))
    return out


def cell_accuracy(pred: np.ndarray, solution: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of masked cells where pred matches solution.

    Args:
        pred: 4x4 predicted grid.
        solution: 4x4 ground truth.
        mask: 4x4 boolean (True = count this cell).

    Returns:
        Scalar in [0, 1]; 1.0 if mask has no True cells.
    """
    if mask.sum() == 0:
        return 1.0
    return np.float64((pred[mask] == solution[mask]).sum()) / mask.sum()
