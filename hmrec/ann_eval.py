"""Measures what serving the two-tower through FAISS HNSW costs and saves.

``python -m hmrec.ann_eval`` -> ``results/ann.json``: training log, candidate recall of the exact
dense scores vs the HNSW index, overlap of the two top-100 lists, and per-query latency.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from . import workspace
from .ann import ItemIndex
from .twotower import TwoTower


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--history-weeks", type=int, default=26)
    ap.add_argument("--out", default="results/ann.json")
    args = ap.parse_args()
    w = workspace.setup(sample_frac=args.sample_frac, seed=0, history_weeks=args.history_weeks)
    t0 = time.time()
    tt = TwoTower(epochs=args.epochs, days=args.history_weeks * 7).fit(w.ctx_fit)
    train_s = time.time() - t0
    users = w.ev_test.users
    q = tt.user_vectors(tt.user_hist[users])

    exact = ItemIndex(tt.item_vecs, "flat")
    t0 = time.time()
    hnsw = ItemIndex(tt.item_vecs, "hnsw")
    build_s = time.time() - t0
    n = 100
    t0 = time.perf_counter()
    _, ids_exact = exact.search(q, n)
    ms_exact = (time.perf_counter() - t0) / len(q) * 1e3
    rows = {}
    for ef in (16, 64, 128, 256):
        hnsw.set_ef(ef)
        t0 = time.perf_counter()
        _, ids = hnsw.search(q, n)
        ms = (time.perf_counter() - t0) / len(q) * 1e3
        overlap = float(
            np.mean([len(set(a) & set(b)) / n for a, b in zip(ids, ids_exact, strict=True)])
        )
        rows[ef] = {
            "overlap_with_exact_top100": round(overlap, 4),
            "candidate_recall@100": round(float(M.candidate_recall(ids, w.ev_test.rel).mean()), 4),
            "ms_per_query": round(ms, 3),
        }
    res = {
        "catalogue_articles": len(tt.item_vecs),
        "dim": tt.item_vecs.shape[1],
        "train_seconds": round(train_s, 1),
        "final_train_loss": tt.log[-1]["loss"],
        "hnsw_build_seconds": round(build_s, 2),
        "exact_candidate_recall@100": round(
            float(M.candidate_recall(ids_exact, w.ev_test.rel).mean()), 4
        ),
        "exact_ms_per_query": round(ms_exact, 3),
        "hnsw_by_efSearch": rows,
        "eval_customers": len(users),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
