import random
from typing import List, Optional

import torch

# MLM training loop: train BERT or TRM on WikiText, return loss/perplexity history.


def train_mlm(
    model: torch.nn.Module,
    chunks: List[List[int]],
    tokenizer,
    device: str,
    max_steps: int,
    batch_size: int = 16,
    lr: float = 5e-5,
    max_length: int = 128,
    mask_prob: float = 0.15,
    log_every: int = 50,
    seed: Optional[int] = None,
) -> List[dict]:
    """Train an MLM for max_steps; use mlm_collate for masking and AdamW with grad clipping.

    Args:
        model: Model with forward(input_ids, attention_mask, labels) returning .loss.
        chunks: List of token id lists (from tokenize_and_chunk).
        tokenizer: Tokenizer (pad/mask/vocab_size).
        device: Device string (e.g. "cuda" or "cpu").
        max_steps: Number of training steps.
        batch_size: Batch size. Defaults to 16.
        lr: Learning rate. Defaults to 5e-5.
        max_length: Max sequence length (for collate). Defaults to 128.
        mask_prob: Masking probability. Defaults to 0.15.
        log_every: Log and record every this many steps. Defaults to 50.
        seed: Random seed. Defaults to None.

    Returns:
        List of dicts with keys "step", "loss", "ppl".
    """
    from .data import mlm_collate

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    history = []
    step = 0
    idx = 0
    while step < max_steps:
        batch_chunks = []
        for _ in range(batch_size):
            batch_chunks.append(chunks[idx % len(chunks)])
            idx += 1
        input_ids, labels = mlm_collate(
            batch_chunks, tokenizer, pad_id, mask_id, vocab_size, mask_prob=mask_prob, seed=seed
        )
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        attention_mask = (input_ids_t != pad_id).long()
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask, labels=labels_t)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % log_every == 0:
            ppl = torch.exp(loss.detach()).item()
            history.append({"step": step, "loss": loss.item(), "ppl": ppl})
            print(f"  step {step} loss {loss.item():.4f} ppl {ppl:.2f}")
    return history


def eval_ppl(
    model: torch.nn.Module,
    chunks: List[List[int]],
    tokenizer,
    device: str,
    batch_size: int = 16,
    max_batches: Optional[int] = 50,
    max_length: int = 128,
    mask_prob: float = 0.15,
    seed: int = 42,
) -> float:
    """Compute mean perplexity on a subset of chunks (for validation).

    Args:
        model: MLM model.
        chunks: List of token id lists.
        tokenizer: Tokenizer.
        device: Device string.
        batch_size: Batch size. Defaults to 16.
        max_batches: Maximum number of batches to evaluate; None = use all. Defaults to 50.
        max_length: Max length for collate. Defaults to 128.
        mask_prob: Masking probability. Defaults to 0.15.
        seed: Random seed. Defaults to 42.

    Returns:
        Scalar perplexity (exp(mean loss)).
    """
    from .data import mlm_collate

    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_id = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size
    total_loss = 0.0
    n_batches = 0
    for start in range(0, min(len(chunks), (max_batches or len(chunks)) * batch_size), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        if not batch_chunks:
            break
        input_ids, labels = mlm_collate(
            batch_chunks, tokenizer, pad_id, mask_id, vocab_size, mask_prob=mask_prob, seed=seed
        )
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        attention_mask = (input_ids_t != pad_id).long()
        with torch.no_grad():
            outputs = model(input_ids=input_ids_t, attention_mask=attention_mask, labels=labels_t)
        total_loss += outputs.loss.item()
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    return torch.exp(torch.tensor(total_loss / max(n_batches, 1))).item()
