try:
    from transformers import BertConfig, BertForMaskedLM
except ImportError:
    BertConfig = None
    BertForMaskedLM = None

# Small BERT for masked LM (HuggingFace). Matched to TRM for fair accuracy-vs-compute comparison.


def get_bert_mlm(
    vocab_size: int = 30522,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 512,
):
    """Return a small BERT for Masked LM (HuggingFace BertForMaskedLM with small config).

    Args:
        vocab_size: Vocabulary size. Defaults to 30522.
        hidden_size: Hidden size and embedding dim. Defaults to 256.
        num_hidden_layers: Number of transformer layers. Defaults to 4.
        num_attention_heads: Number of attention heads. Defaults to 4.
        intermediate_size: FFN intermediate size. Defaults to 1024.
        max_position_embeddings: Max sequence length. Defaults to 512.

    Returns:
        BertForMaskedLM instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    if BertForMaskedLM is None:
        raise ImportError("pip install transformers")
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    return BertForMaskedLM(config)
