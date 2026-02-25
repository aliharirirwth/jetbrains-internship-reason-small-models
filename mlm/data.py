from typing import List, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# WikiText-103 (or subset) for MLM: load, tokenize, masked batches. Uses HuggingFace datasets.


def load_wikitext(
    name: str = "wikitext-103-v1",
    split: str = "train",
    max_examples: Optional[int] = 5000,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """Load WikiText dataset and return a list of text lines (non-empty, no section headers).

    Args:
        name: Dataset variant, e.g. 'wikitext-103-v1' or 'wikitext-2-raw-v1'. Defaults to wikitext-103-v1.
        split: Dataset split. Defaults to "train".
        max_examples: Maximum number of lines to load. Defaults to 5000.
        cache_dir: HuggingFace cache directory. Defaults to None.

    Returns:
        List of non-empty text strings (section headers starting with "=" are skipped).

    Raises:
        ImportError: If datasets is not installed.
    """
    if load_dataset is None:
        raise ImportError("pip install datasets")
    dataset = load_dataset("Salesforce/wikitext", name, split=split, cache_dir=cache_dir)
    texts = []
    for i, ex in enumerate(dataset):
        if max_examples and i >= max_examples:
            break
        t = ex.get("text", "").strip()
        if t and not t.startswith("="):
            texts.append(t)
    return texts


def tokenize_and_chunk(
    texts: List[str],
    tokenizer,
    max_length: int = 128,
    stride: Optional[int] = None,
) -> List[List[int]]:
    """Tokenize texts and split into overlapping chunks of max_length.

    Args:
        texts: List of raw text strings.
        tokenizer: HuggingFace tokenizer (e.g. BERT).
        max_length: Maximum chunk length in tokens. Defaults to 128.
        stride: Step between chunk starts; if None, set to max_length. Defaults to None.

    Returns:
        List of token id lists (each of length <= max_length; chunks with len < 8 are dropped).
    """
    if stride is None:
        stride = max_length
    all_ids = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        ids = enc["input_ids"]
        for start in range(0, len(ids), stride):
            chunk = ids[start : start + max_length]
            if len(chunk) >= 8:
                all_ids.append(chunk)
    return all_ids


def get_tokenizer(model_name: str = "bert-base-uncased"):
    """Return a HuggingFace pretrained tokenizer.

    Args:
        model_name: Pretrained model name (e.g. bert-base-uncased). Defaults to "bert-base-uncased".

    Returns:
        Tokenizer instance.

    Raises:
        ImportError: If transformers is not installed.
    """
    if AutoTokenizer is None:
        raise ImportError("pip install transformers")
    return AutoTokenizer.from_pretrained(model_name)


def mlm_collate(
    batch: List[List[int]],
    tokenizer,
    pad_id: int,
    mask_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    seed: Optional[int] = None,
):
    """Collate a batch of token id lists into padded input_ids and labels for MLM.

    For mask_prob of non-special tokens: 80% replaced by mask_id, 10% random token, 10% unchanged.
    Labels are -100 for non-masked positions and the true token id for masked positions.

    Args:
        batch: List of token id lists (variable length).
        tokenizer: Tokenizer (used for cls/sep token ids).
        pad_id: Padding token id.
        mask_id: Mask token id.
        vocab_size: Vocabulary size (for random replacement).
        mask_prob: Probability of masking a non-special token. Defaults to 0.15.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple (input_ids, labels), each a list of length batch_size (padded to max length).
    """
    import random

    if seed is not None:
        random.seed(seed)
    max_len = max(len(x) for x in batch)
    input_ids = []
    labels = []
    for ids in batch:
        inp = list(ids) + [pad_id] * (max_len - len(ids))
        lab = [-100] * max_len
        for i in range(len(ids)):
            special = {
                pad_id,
                getattr(tokenizer, "cls_token_id", None),
                getattr(tokenizer, "sep_token_id", None),
            }
            special.discard(None)
            if ids[i] in special:
                continue
            if random.random() < mask_prob:
                lab[i] = ids[i]
                r = random.random()
                if r < 0.8:
                    inp[i] = mask_id
                elif r < 0.9:
                    inp[i] = random.randint(0, vocab_size - 1)
        input_ids.append(inp)
        labels.append(lab)
    return input_ids, labels
