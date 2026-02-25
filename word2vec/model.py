from typing import Tuple

import numpy as np

# Skip-gram with negative sampling: forward, loss, gradients in pure NumPy.
# Stability: sigmoid/log_sigmoid use clipping and -softplus(-x); see GRADIENTS.md.


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid; clips input to avoid overflow in exp.

    Args:
        x: Input array (any shape).

    Returns:
        Sigmoid of x, same shape; values in (0, 1).
    """
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Log of sigmoid: -softplus(-x), computed in a numerically stable way.

    Args:
        x: Input array (any shape).

    Returns:
        log(sigmoid(x)), same shape as x.
    """
    x = np.clip(x, -500.0, 500.0)
    return -np.maximum(x, 0) - np.log(1.0 + np.exp(-np.abs(x)))


class SkipGramNegSampling:
    """Skip-gram with negative sampling (two embedding matrices, no autograd).

    Score for (center, context) is W_out[context] @ W_in[center]. Loss is
    -log(sigmoid(score_pos)) - sum over negatives of log(sigmoid(-score_neg)).
    See GRADIENTS.md for derivative details.

    Attributes:
        W_in (np.ndarray): Input/center embeddings, shape (V, D).
        W_out (np.ndarray): Output/context embeddings, shape (V, D).
        V (int): Vocabulary size.
        D (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int, seed: int = 42):
        """Initialize embedding matrices with small random values.

        Args:
            vocab_size: Vocabulary size V.
            dim: Embedding dimension D.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        rng = np.random.default_rng(seed)
        # Small init so sigmoid isn't saturated
        self.W_in = (rng.standard_normal((vocab_size, dim)) * 0.01).astype(np.float64)
        self.W_out = (rng.standard_normal((vocab_size, dim)) * 0.01).astype(np.float64)
        self.V = vocab_size
        self.D = dim

    def forward(
        self,
        center: np.ndarray,
        context_pos: np.ndarray,
        context_neg: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Compute loss and gradients for a batch (gradients are negated for descent).

        Args:
            center: Center word ids, shape (B,).
            context_pos: Positive context word ids, shape (B,).
            context_neg: Negative context word ids, shape (B, K).

        Returns:
            Tuple of (loss, dW_in, dW_out). loss is mean over batch; dW_in and dW_out
            are (V, D) and are the negative of dL/dW so that W -= lr * grad decreases loss.
        """
        B = center.shape[0]
        K = context_neg.shape[1]

        # Embeddings: (B, D) each
        v_w = self.W_in[center]  # (B, D)
        u_c = self.W_out[context_pos]  # (B, D)
        # Positive scores: (B,)
        score_pos = np.sum(u_c * v_w, axis=1)

        # Negative scores: (B, K) — we need W_out for each negative
        # u_neg: (B, K, D) then score_neg = sum over D -> (B, K)
        u_neg = self.W_out[context_neg]  # (B, K, D)
        score_neg = np.sum(u_neg * v_w[:, np.newaxis, :], axis=2)  # (B, K)

        # Loss: -log(sigmoid(score_pos)) - sum_k log(sigmoid(-score_neg))
        loss_pos = -_log_sigmoid(score_pos).sum()
        loss_neg = -_log_sigmoid(-score_neg).sum()
        loss = (loss_pos + loss_neg) / B  # mean over batch for reporting

        # Gradients (see GRADIENTS.md).
        # dL/d(score_pos): for positive, d/dx [-log(sigmoid(x))] = sigmoid(x) - 1.
        sigma_pos = _sigmoid(score_pos)  # (B,)
        grad_score_pos = sigma_pos - 1.0  # (B,)
        # dL/d(score_neg): for negative, d/dx [-log(sigmoid(-x))] = sigma(x).
        sigma_neg = _sigmoid(score_neg)  # (B, K)
        grad_score_neg = sigma_neg  # (B, K)

        # dL/dv_w: from positive (B,) and from negatives (B, K). Sum over context dimension.
        # d(v_w · u_c)/d v_w = u_c, so dL/dv_w += (sigma_pos - 1) * u_c per row.
        d_v_w = grad_score_pos[:, np.newaxis] * u_c  # (B, D)
        # From negatives: d(v_w · u_neg)/d v_w = u_neg, so dL/dv_w += sigma_neg * u_neg summed over K.
        d_v_w += np.einsum("bk,bkd->bd", grad_score_neg, u_neg)
        # dL/du_c: (sigma_pos - 1) * v_w for each positive.
        d_u_c = grad_score_pos[:, np.newaxis] * v_w  # (B, D)
        # dL/du_neg: sigma_neg * v_w for each (b,k); shape (B, K, D).
        d_u_neg = grad_score_neg[:, :, np.newaxis] * v_w[:, np.newaxis, :]  # (B, K, D)

        # Accumulate gradients into full matrices (scatter-add).
        dW_in = np.zeros_like(self.W_in)
        dW_out = np.zeros_like(self.W_out)
        np.add.at(dW_in, center, d_v_w)
        np.add.at(dW_out, context_pos, d_u_c)
        for k in range(K):
            np.add.at(dW_out, context_neg[:, k], d_u_neg[:, k, :])

        # Scale by 1/B so that loss and gradients are on same scale (we minimized mean loss).
        dW_in /= B
        dW_out /= B
        # Sign convention: the formulas above give dL/dW (gradient of L). Descent uses W -= lr * grad,
        # so we must return grad = -dL/dW. Then W -= lr * (-dL/dW) = W + lr * dL/dW is ascent in W
        # along the gradient of L, i.e. descent in L. So we negate before returning.
        return loss, -dW_in, -dW_out

    def forward_backward(
        self,
        center: np.ndarray,
        context_pos: np.ndarray,
        context_neg: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Convenience wrapper: run forward and return loss plus gradient arrays.

        Args:
            center: Center word ids, shape (B,).
            context_pos: Positive context ids, shape (B,).
            context_neg: Negative context ids, shape (B, K).

        Returns:
            Tuple of (loss, dW_in, dW_out).
        """
        loss, dW_in, dW_out = self.forward(center, context_pos, context_neg)
        return loss, dW_in, dW_out
