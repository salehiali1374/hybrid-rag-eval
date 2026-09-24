"""Disk cache for embeddings, so the corpus is embedded once per model.

The cache key is a hash of the model name, the kind of text ("query" or
"passage") and the exact texts, so any change to the model or the data
creates a new entry instead of silently reusing stale vectors.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


def cache_key(model_name: str, kind: str, texts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in (model_name, kind, *texts):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def load_or_compute(
    cache_dir: Path,
    model_name: str,
    kind: str,
    texts: list[str],
    compute: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """Return cached embeddings for `texts`, computing and saving them on a miss."""
    key = cache_key(model_name, kind, texts)
    safe_model = model_name.replace("/", "__")
    path = cache_dir / f"{safe_model}.{kind}.{key}.npy"
    if path.exists():
        return np.load(path)

    embeddings = compute(texts)
    if embeddings.shape[0] != len(texts):
        raise ValueError(f"expected {len(texts)} embeddings, got {embeddings.shape[0]}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    meta = {"model": model_name, "kind": kind, "count": len(texts), "dim": embeddings.shape[1]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return embeddings
