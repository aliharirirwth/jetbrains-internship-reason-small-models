import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

# 4x4 Sudoku models: MLP, BERT-like encoder, TRM. Input (B, 80), output (B, 16, 4) logits.
NUM_CELLS = 16
NUM_VALUES = 5
NUM_DIGITS = 4  # 1-4 only (we predict digit to fill)
INPUT_DIM = 80


def grid_to_features(grid: np.ndarray) -> np.ndarray:
    """Flatten 4x4 grid to one-hot encoding (16 cells * 5 values = 80 floats).

    Args:
        grid: 4x4 array with values in 0..4 (0 = empty, 1-4 = digit).

    Returns:
        1D array of shape (80,) dtype float32.
    """
    x = np.zeros((16, 5), dtype=np.float32)
    flat = grid.ravel()
    for i in range(16):
        x[i, int(flat[i])] = 1.0
    return x.ravel()


def _grid_features_to_value_ids(x):
    """Convert (B, 80) grid features to (B, 16) value indices in 0..4.

    Args:
        x: Tensor of shape (B, 80) (one-hot per cell).

    Returns:
        Tensor of shape (B, 16) with dtype long.
    """
    return x.view(-1, NUM_CELLS, NUM_VALUES).argmax(dim=-1)


class _TransformerBlock(nn.Module):
    """Single transformer encoder layer: self-attention + FFN, residual, LayerNorm."""

    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1
    ):
        """Initialize block (same signature as mlm TRM for consistency)."""
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, H)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class SudokuPredictor(nn.Module if torch is not None else object):
    """MLP baseline: maps (B, 80) grid features to (B, 16, 4) logits per cell per digit.

    No attention; used as baseline in depth-vs-accuracy comparison.
    """

    def __init__(self, hidden: int = 128, num_layers: int = 2, seed: int = 42):
        """Initialize MLP.

        Args:
            hidden: Hidden layer size. Defaults to 128.
            num_layers: Number of hidden layers. Defaults to 2.
            seed: Random seed. Defaults to 42.
        """
        if torch is None:
            raise ImportError("PyTorch required for reasoning.model")
        super().__init__()
        torch.manual_seed(seed)
        self.input_dim = INPUT_DIM
        self.num_cells = NUM_CELLS
        self.num_digits = NUM_DIGITS
        layers = [
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
        ]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, self.num_cells * self.num_digits))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Grid features (B, 80).

        Returns:
            Logits (B, 16, 4).
        """
        out = self.mlp(x)
        return out.view(-1, self.num_cells, self.num_digits)


class SudokuBERT(nn.Module if torch is not None else object):
    """BERT-like encoder for 4x4 Sudoku: 16 cells as sequence, value+position embeddings, then head.

    Stacked transformer layers; output (B, 16, 4) logits. Used in recurrent loop (state = grid).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        """Initialize Sudoku BERT.

        Args:
            hidden_size: Hidden size. Defaults to 128.
            num_hidden_layers: Number of transformer layers. Defaults to 2.
            num_attention_heads: Number of heads. Defaults to 4.
            intermediate_size: FFN size. Defaults to 256.
            dropout: Dropout. Defaults to 0.1.
            seed: Random seed. Defaults to 42.
        """
        if torch is None:
            raise ImportError("PyTorch required for reasoning.model")
        super().__init__()
        torch.manual_seed(seed)
        self.num_cells = NUM_CELLS
        self.num_digits = NUM_DIGITS
        self.value_embed = nn.Embedding(NUM_VALUES, hidden_size)
        self.position_embed = nn.Embedding(NUM_CELLS, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(hidden_size, num_attention_heads, intermediate_size, dropout)
                for _ in range(num_hidden_layers)
            ]
        )
        self.head = nn.Linear(hidden_size, NUM_DIGITS)
        self.register_buffer("position_ids", torch.arange(NUM_CELLS, dtype=torch.long).unsqueeze(0))

    def forward(self, x):
        """Forward pass.

        Args:
            x: Grid features (B, 80).

        Returns:
            Logits (B, 16, 4).
        """
        value_ids = _grid_features_to_value_ids(x).long()
        B = value_ids.size(0)
        pos = self.position_ids.expand(B, -1)
        h = self.value_embed(value_ids) + self.position_embed(pos)
        h = self.layer_norm(h)
        for block in self.blocks:
            h = block(h)
        logits = self.head(h)
        return logits


class SudokuTRM(nn.Module if torch is not None else object):
    """TRM for Sudoku: value+position embeddings, one block applied K times, then head.

    Same interface as SudokuBERT; recurrent application of a single transformer block.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        num_recurrent_steps: int = 4,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        """Initialize Sudoku TRM.

        Args:
            hidden_size: Hidden size. Defaults to 128.
            num_attention_heads: Number of heads. Defaults to 4.
            intermediate_size: FFN size. Defaults to 256.
            num_recurrent_steps: Number of block applications. Defaults to 4.
            dropout: Dropout. Defaults to 0.1.
            seed: Random seed. Defaults to 42.
        """
        if torch is None:
            raise ImportError("PyTorch required for reasoning.model")
        super().__init__()
        torch.manual_seed(seed)
        self.num_cells = NUM_CELLS
        self.num_digits = NUM_DIGITS
        self.num_recurrent_steps = num_recurrent_steps
        self.value_embed = nn.Embedding(NUM_VALUES, hidden_size)
        self.position_embed = nn.Embedding(NUM_CELLS, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.block = _TransformerBlock(hidden_size, num_attention_heads, intermediate_size, dropout)
        self.head = nn.Linear(hidden_size, NUM_DIGITS)
        self.register_buffer("position_ids", torch.arange(NUM_CELLS, dtype=torch.long).unsqueeze(0))

    def forward(self, x):
        """Forward pass.

        Args:
            x: Grid features (B, 80).

        Returns:
            Logits (B, 16, 4).
        """
        value_ids = _grid_features_to_value_ids(x).long()
        B = value_ids.size(0)
        pos = self.position_ids.expand(B, -1)
        h = self.value_embed(value_ids) + self.position_embed(pos)
        h = self.layer_norm(h)
        for _ in range(self.num_recurrent_steps):
            h = self.block(h)
        logits = self.head(h)
        return logits
