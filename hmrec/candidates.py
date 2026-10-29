"""Stage 1: merge several candidate sources into one pool per customer.

Each source proposes its top-N articles; the union is the candidate set. A source's raw score is
kept as a ranking feature for *every* candidate (not only the ones it proposed), together with its
rank when the source proposed the article. Candidate recall is reported per source and for the
union.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .models import Recommender


@dataclass
class Source:
    model: Recommender
    n: int  # how many articles this source proposes per customer


@dataclass
class CandidatePool:
    frame: pd.DataFrame  # user, item, label, src_* features
    users: np.ndarray
    recall: dict[str, float] = field(default_factory=dict)  # mean candidate recall per source
    avg_candidates: float = 0.0


def generate(
    sources: dict[str, Source],
    users: np.ndarray,
    relevant=None,
    batch: int = 128,
) -> CandidatePool:
    """Union of top-N per source. ``relevant`` is the evaluator's sparse purchase matrix (rows
    aligned with ``users``) used for labels and recall; omit it when serving."""
    frames = []
    cand_by_source: dict[str, list[np.ndarray]] = {name: [] for name in sources}
    union_rows: list[np.ndarray] = []
    for s in range(0, len(users), batch):
        u = users[s : s + batch]
        scores = {name: src.model.score(u) for name, src in sources.items()}
        tops, topn_rank = {}, {}
        for name, src in sources.items():
            sc = scores[name]
            n = min(src.n, sc.shape[1])
            part = np.argpartition(-sc, n - 1, axis=1)[:, :n]
            order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1, kind="stable")
            top = np.take_along_axis(part, order, 1)
            ok = np.isfinite(np.take_along_axis(sc, top, 1))
            tops[name] = np.where(ok, top, -1)
            topn_rank[name] = np.where(ok, np.arange(n)[None, :], -1)
            cand_by_source[name].append(
                np.pad(tops[name], ((0, 0), (0, src.n - n)), constant_values=-1)
            )
        for j, uu in enumerate(u):
            cand = np.unique(np.concatenate([tops[n][j] for n in sources]))
            cand = cand[cand >= 0]
            cols = {"user": np.full(len(cand), uu), "item": cand}
            for name in sources:
                raw = scores[name][j, cand]
                cols[f"src_{name}_score"] = np.where(np.isfinite(raw), raw, np.nan)
                rk = np.full(len(cand), np.nan)
                pos = {int(i): r for r, i in enumerate(tops[name][j]) if i >= 0}
                for t, i in enumerate(cand):
                    if int(i) in pos:
                        rk[t] = pos[int(i)]
                cols[f"src_{name}_rank"] = rk
            frames.append(pd.DataFrame(cols))
            union_rows.append(cand)
    frame = pd.concat(frames, ignore_index=True)
    pool = CandidatePool(frame, users)
    pool.avg_candidates = len(frame) / max(len(users), 1)
    if relevant is not None:
        r = relevant.tocsr()
        pos_of = {int(u): j for j, u in enumerate(users)}
        rows = frame["user"].map(pos_of).to_numpy()
        frame["label"] = np.asarray(r[rows, frame["item"].to_numpy()]).ravel().astype(int)
        width = max(len(c) for c in union_rows)
        union = np.full((len(users), width), -1)
        for j, c in enumerate(union_rows):
            union[j, : len(c)] = c
        pool.recall["union"] = float(M.candidate_recall(union, r).mean())
        for name, chunks in cand_by_source.items():
            pool.recall[name] = float(M.candidate_recall(np.vstack(chunks), r).mean())
    return pool
