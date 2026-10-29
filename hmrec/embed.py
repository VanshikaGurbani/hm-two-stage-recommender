"""Text embeddings of article descriptions (product name, type, colour, department, description).

MiniLM (``all-MiniLM-L6-v2``) by default; a TF-IDF + SVD fallback of the same text keeps tests and
CI free of model downloads. Results are cached under ``data/cache`` keyed by a hash of the text.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .data import DATA_DIR, HMData

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def item_embeddings(
    data: HMData,
    encoder: str = "minilm",
    cache_dir: Path | None = None,
    dim: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """(n_items, d) float32, rows L2-normalised and aligned with ``data.items``."""
    texts = data.items["text"].tolist()
    key = hashlib.sha1(("\n".join(texts) + encoder + MODEL_NAME).encode()).hexdigest()[:16]
    cache = (cache_dir or DATA_DIR / "cache") / f"emb_{encoder}_{key}.npy"
    if cache.exists():
        return np.load(cache)
    if encoder == "minilm":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, device="cpu")
        emb = model.encode(
            texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False
        )
        emb = np.asarray(emb, dtype=np.float32)
    elif encoder == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        x = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
        d = min(dim, x.shape[1] - 1, x.shape[0] - 1)
        emb = _l2(TruncatedSVD(d, random_state=seed).fit_transform(x).astype(np.float32))
    else:
        raise ValueError(f"unknown encoder {encoder!r}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb
