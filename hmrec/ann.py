"""FAISS wrapper: HNSW (approximate) or flat (exact) inner-product search over article vectors."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np


class ItemIndex:
    def __init__(self, vectors: np.ndarray, kind: str = "hnsw", m: int = 32, ef_search: int = 128):
        self.index: Any
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        d = self.vectors.shape[1]
        if kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = ef_search
        elif kind == "flat":
            self.index = faiss.IndexFlatIP(d)
        else:
            raise ValueError(f"unknown index kind {kind!r}")
        self.kind = kind
        self.index.add(self.vectors)

    def set_ef(self, ef: int) -> None:
        if self.kind == "hnsw":
            self.index.hnsw.efSearch = ef

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        scores, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return scores, ids
