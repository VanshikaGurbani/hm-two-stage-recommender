"""Full-catalogue evaluator with bootstrap confidence intervals over customers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import metrics as M
from .data import HMData
from .splits import relevant_items

KMAX = 200


def purchase_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """Binary customer x article matrix (repeat purchases collapse to 1)."""
    m = sp.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df["user"].to_numpy(), df["item"].to_numpy())),
        shape=(n_users, n_items),
    )
    m.data[:] = 1.0
    return m


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k column indices per row, best first; -inf scores become -1 padding."""
    scores = scores.astype(np.float32, copy=False)
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1, kind="stable")
    top = np.take_along_axis(part, order, 1)
    return np.where(np.isfinite(np.take_along_axis(scores, top, 1)), top, -1)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean difference a - b with a 95% CI over customers (same customers, paired)."""
    d = np.asarray(a) - np.asarray(b)
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}


@dataclass
class EvalResult:
    model: str
    users: np.ndarray
    ranked: np.ndarray
    per_user: dict[str, np.ndarray]
    summary: dict[str, float]
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)


class Evaluator:
    """Ranks the whole catalogue for each customer who bought something in the target week.

    Already-bought articles are *not* masked: repurchases are legitimate H&M hits, and the
    competition metric does not exclude them either. Customers without any prior purchase in
    ``history`` are skipped (they are the cold-start case, reported separately).
    """

    def __init__(
        self,
        data: HMData,
        history: pd.DataFrame,
        target: pd.DataFrame,
        k: int = 12,
        users: np.ndarray | None = None,
        batch: int = 256,
        content_vectors: np.ndarray | None = None,
    ):
        self.data, self.k, self.batch = data, k, batch
        self.history_matrix = purchase_matrix(history, data.n_users, data.n_items)
        rel = relevant_items(target, history)
        eval_users = np.array(sorted(rel), dtype=int)
        if users is not None:
            eval_users = np.intersect1d(eval_users, users)
        self.users = eval_users
        pos = {int(u): j for j, u in enumerate(self.users)}
        rows = np.concatenate([np.full(len(rel[int(u)]), pos[int(u)]) for u in self.users])
        cols = np.concatenate([rel[int(u)] for u in self.users])
        self.rel = sp.csr_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(self.users), data.n_items)
        )
        self.item_pop = np.asarray(self.history_matrix.sum(0)).ravel()  # buyers per article
        self.n_hist_users = int((np.asarray(self.history_matrix.sum(1)).ravel() > 0).sum())
        self.tag_matrix = data.tag_matrix()
        self.content_vectors = content_vectors

    def rank(self, model) -> np.ndarray:
        out = []
        for s in range(0, len(self.users), self.batch):
            out.append(topk(model.score(self.users[s : s + self.batch]), KMAX))
        return np.vstack(out)

    def evaluate(self, model, name: str | None = None) -> EvalResult:
        return self.metrics_from_ranking(self.rank(model), name or getattr(model, "name", "model"))

    def metrics_from_ranking(self, ranked: np.ndarray, name: str) -> EvalResult:
        k, rel = self.k, self.rel
        per = {
            f"map@{k}": M.average_precision_at_k(ranked, rel, k),
            f"recall@{k}": M.recall_at_k(ranked, rel, k),
            f"ndcg@{k}": M.ndcg_at_k(ranked, rel, k),
            f"hit@{k}": M.hit_rate_at_k(ranked, rel, k),
        }
        for kk in (50, 100, 200):
            if ranked.shape[1] >= kk:
                per[f"recall@{kk}"] = M.recall_at_k(ranked, rel, kk)
        s = {m: float(v.mean()) for m, v in per.items()}
        n_items = self.data.n_items
        s["coverage"] = M.catalog_coverage(ranked, n_items, k)
        s["gini"] = M.exposure_gini(ranked, n_items, k)
        s["novelty"] = M.novelty(ranked, self.item_pop, self.n_hist_users, k)
        s["arp"] = M.avg_recommendation_popularity(ranked, self.item_pop, k)
        s["head_share"] = M.head_share(ranked, self.item_pop, k)
        s["tail_share"] = M.tail_share(ranked, self.item_pop, k)
        td = M.tag_diversity(ranked, self.tag_matrix, k)
        per["tag_ild"] = td
        s["tag_ild"] = float(td.mean())
        if self.content_vectors is not None:
            cd = M.intra_list_diversity(ranked, self.content_vectors, k)
            per["content_ild"] = cd
            s["content_ild"] = float(cd.mean())
        s["n_customers"] = float(len(self.users))
        ci = {m: bootstrap_ci(per[m]) for m in (f"map@{k}", f"recall@{k}", f"ndcg@{k}")}
        return EvalResult(name, self.users, ranked, per, s, ci)
