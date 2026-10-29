"""Diversity re-ranking with maximal marginal relevance (MMR)."""

from __future__ import annotations

import numpy as np


def mmr_select(
    cand: np.ndarray, rel: np.ndarray, vectors: np.ndarray, k: int, lam: float
) -> np.ndarray:
    """Greedy MMR: argmax of ``lam * relevance - (1 - lam) * max cosine similarity to picks``.

    ``cand`` are article indices, ``rel`` their ranker scores (min-max scaled inside the list) and
    ``vectors`` unit-norm embeddings. ``lam = 1`` returns the plain top-k.
    """
    ok = cand >= 0
    cand, rel = cand[ok], rel[ok].astype(np.float64)
    out = np.full(k, -1, dtype=np.int64)
    if len(cand) == 0:
        return out
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 1e-12 else np.zeros_like(rel)
    sim = vectors[cand] @ vectors[cand].T
    taken = np.zeros(len(cand), dtype=bool)
    max_sim = np.zeros(len(cand))
    for t in range(min(k, len(cand))):
        obj = lam * rel - (1.0 - lam) * max_sim
        obj[taken] = -np.inf
        j = int(np.argmax(obj))
        out[t] = cand[j]
        taken[j] = True
        max_sim = np.maximum(max_sim, sim[j])
    return out


def mmr_rerank(
    ranked: np.ndarray,
    scores: np.ndarray,
    vectors: np.ndarray,
    k: int,
    lam: float,
    pool: int = 50,
) -> np.ndarray:
    """Re-rank each customer's top-``pool`` articles with MMR and keep ``k``."""
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    out = np.full((ranked.shape[0], k), -1, dtype=np.int64)
    for u in range(ranked.shape[0]):
        out[u] = mmr_select(ranked[u, :pool], scores[u, :pool], v, k, lam)
    return out
