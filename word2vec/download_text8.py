import os
import sys
import urllib.request
from typing import Optional

# Download text8 corpus for word2vec. Saves to word2vec/data/text8.txt. No ML deps.

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "text8.txt")


def download_text8(save_path: str = DEFAULT_PATH, max_chars: Optional[int] = None) -> str:
    """Download text8.zip, unzip in memory, and write text to save_path.

    Args:
        save_path: Path to write the extracted text. Defaults to word2vec/data/text8.txt.
        max_chars: If set, only write the first max_chars characters. Defaults to None.

    Returns:
        The path written (save_path).

    Raises:
        RuntimeError: If the zip contains no files.
    """
    import io
    import zipfile

    req = urllib.request.Request(TEXT8_URL, headers={"User-Agent": "word2vec-download/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data), "r") as z:
        names = z.namelist()
        if not names:
            raise RuntimeError("Empty zip")
        text = z.read(names[0]).decode("utf-8", errors="replace")
    if max_chars is not None:
        text = text[:max_chars]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write(text)
    return save_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    # Optional: 5MB slice for quick runs
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out = download_text8(path, max_chars=max_chars)
    print(f"Wrote {os.path.getsize(out) // 1024} KB to {out}")
