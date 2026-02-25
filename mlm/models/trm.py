import torch
import torch.nn as nn

# TRM: one transformer layer applied K times (recurrent). Matched to BERT for comparison.


class TransformerBlock(nn.Module):
    """Single transformer encoder layer: self-attention + FFN with residual and LayerNorm.

    Attributes:
        attn: MultiheadAttention module.
        ff: Two-layer FFN (Linear-GELU-Linear).
        norm1, norm2: LayerNorm layers.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1
    ):
        """Initialize the block.

        Args:
            hidden_size: Hidden size and attention dimension.
            num_heads: Number of attention heads.
            intermediate_size: FFN intermediate dimension.
            dropout: Dropout rate. Defaults to 0.1.
        """
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

    def forward(self, x, key_padding_mask=None):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, S, H).
            key_padding_mask: Optional mask (True = mask out). Defaults to None.

        Returns:
            Tensor of shape (B, S, H).
        """
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TRMForMaskedLM(nn.Module):
    """TRM: embedding + K recurrent steps of one transformer block + LM head (weight-tied).

    Same transformer block is applied num_recurrent_steps times; no extra layers.
    Interface matches BERT: forward(input_ids, attention_mask=None, labels=None) -> .loss, .logits.

    Attributes:
        hidden_size (int): Hidden dimension.
        num_recurrent_steps (int): Number of times the block is applied.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        num_recurrent_steps: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize TRM.

        Args:
            vocab_size: Vocabulary size. Defaults to 30522.
            hidden_size: Hidden size. Defaults to 256.
            num_attention_heads: Number of heads. Defaults to 4.
            intermediate_size: FFN size. Defaults to 1024.
            max_position_embeddings: Max sequence length. Defaults to 512.
            num_recurrent_steps: Number of recurrent block applications. Defaults to 4.
            dropout: Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_recurrent_steps = num_recurrent_steps
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.block = TransformerBlock(hidden_size, num_attention_heads, intermediate_size, dropout)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tying: share embedding and lm_head weights
        self.lm_head.weight = self.embeddings.weight
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).unsqueeze(0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass; returns object with .loss and .logits (BERT-compatible).

        Args:
            input_ids: Token ids (B, S).
            attention_mask: Optional (B, S), 1 = attend. Defaults to None.
            labels: Optional labels, -100 ignored. Defaults to None.

        Returns:
            Object with .loss (scalar or None) and .logits (B, S, vocab_size).
        """
        B, S = input_ids.shape
        position_ids = self.position_ids[:, :S].expand(B, -1)
        x = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (1 - attention_mask).bool()  # True = mask out
        for _ in range(self.num_recurrent_steps):
            x = self.block(x, key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type("Output", (), {"loss": loss, "logits": logits})()
