"""Baselines and candidate sources. Every model exposes ``score(users) -> (n, n_items)`` floats.

All models are fitted on a :class:`FitContext` holding only purchases *before* the target week.
Scores of -inf mean "not a candidate".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import HMData
from .splits import DAY


@dataclass
class FitContext:
    data: HMData
    frame: pd.DataFrame  # purchases strictly before the target week
    seed: int = 0

    @cached_property
    def ref_ts(self) -> int:
        """First second after the history; the moment recommendations are made."""
        return int(self.frame["ts"].max()) + DAY

    @cached_property
    def counts(self) -> sp.csr_matrix:
        """customer x article number of purchases."""
        f = self.frame
        m = sp.coo_matrix(
            (np.ones(len(f), dtype=np.float32), (f["user"], f["item"])),
            shape=(self.data.n_users, self.data.n_items),
        )
        return m.tocsr()

    def recent(self, days: int) -> pd.DataFrame:
        return self.frame[self.frame["ts"] >= self.ref_ts - days * DAY]

    def decayed(self, half_life_days: float, window_days: int | None = None) -> sp.csr_matrix:
        """customer x article sum of 0.5 ** (age_days / half_life) over purchases."""
        f = self.frame if window_days is None else self.recent(window_days)
        age = (self.ref_ts - f["ts"].to_numpy()) / DAY
        w = np.power(0.5, age / half_life_days).astype(np.float32)
        m = sp.coo_matrix((w, (f["user"], f["item"])), shape=(self.data.n_users, self.data.n_items))
        return m.tocsr()

    def popularity(self, days: int) -> np.ndarray:
        r = self.recent(days)
        return np.bincount(r["item"], minlength=self.data.n_items).astype(np.float32)


class Recommender:
    name = "base"

    def fit(self, ctx: FitContext):  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, users: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _dense(m: sp.spmatrix) -> np.ndarray:
    return np.asarray(m.todense(), dtype=np.float32)


class Popularity(Recommender):
    """Most purchased articles over the last ``days`` days of history."""

    def __init__(self, days: int = 7):
        self.days = days
        self.name = f"popularity-{days}d"

    def fit(self, ctx: FitContext):
        self.s = ctx.popularity(self.days)
        # break ties (and fill the list when few articles sold) with all-time popularity
        self.s = self.s + 1e-3 * np.log1p(np.asarray(ctx.counts.sum(0)).ravel())
        return self

    def score(self, users):
        return np.tile(self.s, (len(users), 1))


class Repurchase(Recommender):
    """Articles the customer already bought, newest and most repeated first."""

    def __init__(self, half_life_days: float = 14.0):
        self.hl = half_life_days
        self.name = f"repurchase-hl{half_life_days:g}d"

    def fit(self, ctx: FitContext):
        self.m = ctx.decayed(self.hl)
        return self

    def score(self, users):
        s = _dense(self.m[users])
        return np.where(s > 0, s, -np.inf)


class AgePopularity(Recommender):
    """Popularity within the customer's age bucket (recent window)."""

    def __init__(self, days: int = 14, bucket: int = 10):
        self.days, self.bucket = days, bucket
        self.name = f"age-popularity-{days}d"

    def fit(self, ctx: FitContext):
        age = ctx.data.customers["age"].to_numpy(dtype=float)
        fill = np.nanmedian(age)
        self.group = (np.where(np.isnan(age), fill, age) // self.bucket).astype(int)
        r = ctx.recent(self.days)
        g = self.group[r["user"].to_numpy()]
        self.n_groups = int(self.group.max()) + 1
        flat = np.bincount(
            g * ctx.data.n_items + r["item"].to_numpy(), minlength=self.n_groups * ctx.data.n_items
        )
        self.table = flat.reshape(self.n_groups, ctx.data.n_items).astype(np.float32)
        self.table += 1e-3 * np.log1p(np.asarray(ctx.counts.sum(0)).ravel())[None, :]
        return self

    def score(self, users):
        return self.table[self.group[users]]


class CoPurchase(Recommender):
    """Item-item cosine similarity from co-purchase by the same customer (recent window)."""

    def __init__(self, days: int = 56, k: int = 30, half_life_days: float = 14.0):
        self.days, self.k, self.hl = days, k, half_life_days
        self.name = f"co-purchase-{days}d"

    def fit(self, ctx: FitContext):
        f = ctx.recent(self.days)
        n_items = ctx.data.n_items
        x = sp.coo_matrix(
            (np.ones(len(f), dtype=np.float32), (f["user"], f["item"])),
            shape=(ctx.data.n_users, n_items),
        ).tocsr()
        x.data[:] = 1.0
        active = np.flatnonzero(np.asarray(x.sum(0)).ravel() >= 2)
        xa = x[:, active].tocsc()
        norms = np.sqrt(np.asarray(xa.sum(0)).ravel()) + 1e-9
        xt = xa.T.tocsr()
        rows, cols, vals = [], [], []
        for s in range(0, len(active), 1000):
            e = min(s + 1000, len(active))
            co = (xt[s:e] @ xa).toarray()
            sim = co / (norms[s:e, None] * norms[None, :])
            sim[np.arange(e - s), np.arange(s, e)] = 0.0
            kk = min(self.k, len(active) - 1)
            idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            v = np.take_along_axis(sim, idx, 1)
            rows.append(np.repeat(active[s:e], kk))
            cols.append(active[idx.ravel()])
            vals.append(v.ravel())
        self.sim = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
            dtype=np.float32,
        )
        self.hist = ctx.decayed(self.hl, window_days=self.days)
        return self

    def score(self, users):
        s = _dense(self.hist[users] @ self.sim.T)
        return np.where(s > 0, s, -np.inf)


class ALS(Recommender):
    """Implicit-feedback ALS on log purchase counts over a recent window."""

    def __init__(self, factors=64, reg=0.05, alpha=20.0, iterations=15, days=180, seed=0):
        self.factors, self.reg, self.alpha, self.iterations = factors, reg, alpha, iterations
        self.days, self.seed = days, seed
        self.name = f"als-{factors}f"

    def fit(self, ctx: FitContext):
        from implicit.cpu.als import AlternatingLeastSquares

        f = ctx.recent(self.days)
        m = sp.coo_matrix(
            (np.ones(len(f), dtype=np.float32), (f["user"], f["item"])),
            shape=(ctx.data.n_users, ctx.data.n_items),
        ).tocsr()
        m.data = 1.0 + self.alpha * np.log1p(m.data)
        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iterations,
            random_state=self.seed,
            calculate_training_loss=False,
        )
        model.fit(m, show_progress=False)
        self.user_f = np.asarray(model.user_factors, dtype=np.float32)
        self.item_f = np.asarray(model.item_factors, dtype=np.float32)
        self.active = np.asarray(m.sum(0)).ravel() > 0
        return self

    def score(self, users):
        s = self.user_f[users] @ self.item_f.T
        return np.where(self.active[None, :], s, -np.inf)
