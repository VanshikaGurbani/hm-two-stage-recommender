"""Ranking, candidate-generation and beyond-accuracy metrics.

``ranked`` is an ``(n_users, K)`` int array of item indices, best first, ``-1`` for padding.
``rel`` is a sparse boolean ``(n_users, n_items)`` matrix of the items each customer bought in
the target week. Accuracy is computed over the whole catalogue (no sampled negatives).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _lookup(rel, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if sp.issparse(rel):
        return np.asarray(rel[rows, cols]).ravel().astype(bool)
    return rel[rows, cols].astype(bool)


def hit_matrix(ranked: np.ndarray, rel, k: int) -> np.ndarray:
    """(n_users, k) bool: is the item at each rank a purchased one?"""
    r = ranked[:, :k]
    valid = r >= 0
    rows = np.repeat(np.arange(r.shape[0]), r.shape[1])
    hits = _lookup(rel, rows, np.where(valid, r, 0).ravel()).reshape(r.shape)
    return hits & valid


def n_relevant(rel) -> np.ndarray:
    return np.asarray(rel.sum(axis=1)).ravel()


def average_precision_at_k(ranked, rel, k: int = 12) -> np.ndarray:
    """AP@k normalised by min(n_relevant, k) - the competition's per-customer score."""
    hits = hit_matrix(ranked, rel, k)
    prec = np.cumsum(hits, axis=1) / np.arange(1, hits.shape[1] + 1)
    denom = np.minimum(n_relevant(rel), k)
    return np.where(denom > 0, (prec * hits).sum(1) / np.maximum(denom, 1), 0.0)


def map_at_12(ranked, rel) -> np.ndarray:
    return average_precision_at_k(ranked, rel, 12)


def recall_at_k(ranked, rel, k: int) -> np.ndarray:
    return hit_matrix(ranked, rel, k).sum(1) / np.maximum(n_relevant(rel), 1)


def hit_rate_at_k(ranked, rel, k: int) -> np.ndarray:
    return hit_matrix(ranked, rel, k).any(1).astype(float)


def ndcg_at_k(ranked, rel, k: int) -> np.ndarray:
    hits = hit_matrix(ranked, rel, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(1)
    ideal_len = np.minimum(n_relevant(rel), k).astype(int)
    idcg = np.cumsum(discounts)[np.maximum(ideal_len, 1) - 1]
    return np.where(ideal_len > 0, dcg / idcg, 0.0)


def candidate_recall(candidates: np.ndarray, rel, k: int | None = None) -> np.ndarray:
    """Per-customer share of purchased items present among the first ``k`` candidates.

    Order is ignored and duplicates (e.g. a union of several sources) count once: this scores a
    candidate generator, not a ranker.
    """
    cand = candidates if k is None else candidates[:, :k]
    valid = cand >= 0
    rows = np.repeat(np.arange(cand.shape[0]), cand.shape[1])
    hit = _lookup(rel, rows, np.where(valid, cand, 0).ravel()).reshape(cand.shape) & valid
    n_hit = np.array([len(np.unique(c[h])) for c, h in zip(cand, hit, strict=True)], dtype=float)
    return n_hit / np.maximum(n_relevant(rel), 1)


# ---------------------------------------------------------------- beyond accuracy


def _flat(ranked: np.ndarray, k: int) -> np.ndarray:
    r = ranked[:, :k].ravel()
    return r[r >= 0]


def catalog_coverage(ranked: np.ndarray, n_items: int, k: int) -> float:
    return len(np.unique(_flat(ranked, k))) / n_items


def exposure_gini(ranked: np.ndarray, n_items: int, k: int) -> float:
    """Gini of how often each article is recommended (0 = even exposure, 1 = a single article)."""
    counts = np.sort(np.bincount(_flat(ranked, k), minlength=n_items).astype(float))
    if counts.sum() == 0:
        return 0.0
    cum = np.cumsum(counts)
    n = len(counts)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def novelty(ranked: np.ndarray, item_pop: np.ndarray, n_users: int, k: int) -> float:
    """Mean self-information -log2 p(i) of recommended items; p(i) = buyers / customers."""
    p = np.maximum(item_pop, 1) / n_users
    return float(np.mean(-np.log2(np.minimum(p[_flat(ranked, k)], 1.0))))


def avg_recommendation_popularity(ranked: np.ndarray, item_pop: np.ndarray, k: int) -> float:
    return float(np.mean(item_pop[_flat(ranked, k)]))


def head_share(ranked: np.ndarray, item_pop: np.ndarray, k: int, head_frac: float = 0.01) -> float:
    """Share of slots filled by the most popular ``head_frac`` of the catalogue."""
    n_head = max(1, round(head_frac * len(item_pop)))
    head = np.zeros(len(item_pop), dtype=bool)
    head[np.argsort(-item_pop, kind="stable")[:n_head]] = True
    return float(head[_flat(ranked, k)].mean())


def tail_share(ranked: np.ndarray, item_pop: np.ndarray, k: int, tail_frac: float = 0.5) -> float:
    """Share of slots filled by the least popular ``tail_frac`` of the catalogue."""
    n_tail = max(1, round(tail_frac * len(item_pop)))
    tail = np.zeros(len(item_pop), dtype=bool)
    tail[np.argsort(item_pop, kind="stable")[:n_tail]] = True
    return float(tail[_flat(ranked, k)].mean())


def intra_list_diversity(ranked: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """Per-customer mean pairwise cosine distance inside the top-k list."""
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    out = np.zeros(ranked.shape[0])
    for u in range(ranked.shape[0]):
        items = ranked[u, :k]
        items = items[items >= 0]
        n = len(items)
        if n >= 2:
            sim = v[items] @ v[items].T
            out[u] = 1.0 - (sim.sum() - n) / (n * (n - 1))
    return out


def tag_diversity(ranked: np.ndarray, tag_matrix: np.ndarray, k: int) -> np.ndarray:
    """Per-customer mean pairwise Jaccard distance of article category tags in the top-k list."""
    g = tag_matrix.astype(np.float32)
    out = np.zeros(ranked.shape[0])
    for u in range(ranked.shape[0]):
        items = ranked[u, :k]
        items = items[items >= 0]
        n = len(items)
        if n < 2:
            continue
        gi = g[items]
        inter = gi @ gi.T
        sizes = gi.sum(1)
        union = sizes[:, None] + sizes[None, :] - inter
        jac = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        out[u] = 1.0 - (jac.sum() - n) / (n * (n - 1))
    return out
