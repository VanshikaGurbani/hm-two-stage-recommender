"""Stage 2: LightGBM LambdaRank over the candidate pool.

Feature groups (used for the ablation table):

* ``source``   - each candidate source's score and rank
* ``customer`` - age, loyalty flags, purchase count / recency / spend / channel mix
* ``item``     - recent popularity, trend, price, article age, categorical attributes
* ``pair``     - customer x article: times bought, days since last bought, affinity to the
                 article's product type / colour / department, price vs the customer's usual
* ``text``     - cosine between the article's description embedding and the customer's
                 recently bought articles

Every feature is computed from purchases strictly before the week being predicted.
Training labels come from the validation week; the test week is only used for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import HMData, customer_features
from .models import FitContext
from .splits import DAY

GROUPS = ("source", "customer", "item", "pair", "text")
CATEGORICAL = ("item_product_type", "item_colour", "item_department", "item_index_group")


def _codes(series: pd.Series) -> np.ndarray:
    return pd.factorize(series)[0].astype(np.int32)


class FeatureBuilder:
    """Precomputes per-customer / per-article statistics from the history, then featurises pools."""

    def __init__(self, data: HMData, ctx: FitContext, embeddings: np.ndarray | None = None):
        self.data, self.ctx, self.emb = data, ctx, embeddings
        f = ctx.frame
        n_u, n_i = data.n_users, data.n_items
        ref = ctx.ref_ts
        self.cust = customer_features(data).to_numpy(dtype=np.float32)
        self.cust_names = list(customer_features(data).columns)
        g = f.groupby("user")
        idx = np.arange(n_u)
        self.u_n = g.size().reindex(idx, fill_value=0).to_numpy(dtype=np.float32)
        self.u_last = g["ts"].max().reindex(idx).to_numpy(dtype=np.float64)
        self.u_first = g["ts"].min().reindex(idx).to_numpy(dtype=np.float64)
        self.u_price = g["price"].mean().reindex(idx).to_numpy(dtype=np.float32)
        self.u_spend = g["price"].sum().reindex(idx, fill_value=0).to_numpy(dtype=np.float32)
        self.u_online = (
            (f["channel"] == 2).groupby(f["user"]).mean().reindex(idx).to_numpy(dtype=np.float32)
        )
        self.u_distinct = (
            f.groupby("user")["item"]
            .nunique()
            .reindex(idx, fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        self.u_recent_n = (
            ctx.recent(28)
            .groupby("user")
            .size()
            .reindex(idx, fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        # article statistics
        pop = {d: ctx.popularity(d) for d in (7, 14, 28)}
        self.pop = pop
        self.trend = (pop[7] + 1) / (pop[28] / 4 + 1)
        self.i_price = (
            f.groupby("item")["price"].mean().reindex(np.arange(n_i)).to_numpy(dtype=np.float32)
        )
        first = f.groupby("item")["ts"].min().reindex(np.arange(n_i)).to_numpy(dtype=np.float64)
        self.i_age = (ref - first) / DAY
        items = data.items
        self.cat = {
            "product_type": _codes(items["product_type"]),
            "colour": _codes(items["colour"]),
            "department": _codes(items["department"]),
            "index_group": _codes(
                items["tags"].apply(
                    lambda t: next((x for x in t if x.startswith("index_group")), "")
                )
            ),
        }
        # affinity: share of the customer's history in each attribute value
        self.aff = {}
        items_idx = f["item"].to_numpy()
        for name, codes in self.cat.items():
            m = sp.coo_matrix(
                (np.ones(len(f), dtype=np.float32), (f["user"], codes[items_idx])),
                shape=(n_u, int(codes.max()) + 1),
            ).tocsr()
            row_sum = np.asarray(m.sum(1)).ravel()
            self.aff[name] = sp.diags(1.0 / np.maximum(row_sum, 1.0)) @ m
        self.counts = ctx.counts
        last = sp.coo_matrix(
            ((f["ts"].to_numpy() - f["ts"].min() + 1).astype(np.float64), (f["user"], f["item"])),
            shape=(n_u, n_i),
        ).tocsr()
        # coo -> csr sums duplicates; recompute max per pair separately
        mx = f.groupby(["user", "item"])["ts"].max().reset_index()
        self.last_pair = sp.csr_matrix(
            (
                (mx["ts"].to_numpy() - f["ts"].min() + 1).astype(np.float64),
                (mx["user"], mx["item"]),
            ),
            shape=(n_u, n_i),
        )
        self.ts_min = f["ts"].min()
        del last
        self.user_text = None
        if embeddings is not None:
            self.user_text = self._user_text_profile(f)

    def _user_text_profile(self, f: pd.DataFrame) -> np.ndarray:
        """Mean embedding of each customer's 5 most recent purchases."""
        n_u = self.data.n_users
        prof = np.zeros((n_u, self.emb.shape[1]), dtype=np.float32)
        last5 = f.sort_values("ts").groupby("user").tail(5)
        np.add.at(prof, last5["user"].to_numpy(), self.emb[last5["item"].to_numpy()])
        return prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-9)

    def featurize(self, pool_frame: pd.DataFrame) -> pd.DataFrame:
        df = pool_frame.copy()
        u = df["user"].to_numpy()
        i = df["item"].to_numpy()
        ref = self.ctx.ref_ts
        out = {}
        for j, name in enumerate(self.cust_names):
            out[f"cust_{name}"] = self.cust[u, j]
        out["cust_n_purchases"] = np.log1p(self.u_n[u])
        out["cust_days_since_last"] = (ref - self.u_last[u]) / DAY
        out["cust_tenure_days"] = (ref - self.u_first[u]) / DAY
        out["cust_mean_price"] = self.u_price[u]
        out["cust_total_spend"] = self.u_spend[u]
        out["cust_online_share"] = self.u_online[u]
        out["cust_distinct_items"] = self.u_distinct[u]
        out["cust_purchases_28d"] = self.u_recent_n[u]
        for d, p in self.pop.items():
            out[f"item_pop_{d}d"] = p[i]
        out["item_trend"] = self.trend[i]
        out["item_price"] = self.i_price[i]
        out["item_age_days"] = self.i_age[i]
        out["item_product_type"] = self.cat["product_type"][i]
        out["item_colour"] = self.cat["colour"][i]
        out["item_department"] = self.cat["department"][i]
        out["item_index_group"] = self.cat["index_group"][i]
        out["pair_times_bought"] = np.asarray(self.counts[u, i]).ravel()
        last = np.asarray(self.last_pair[u, i]).ravel()
        out["pair_days_since_bought"] = np.where(
            last > 0, (ref - (last + self.ts_min - 1)) / DAY, np.nan
        )
        for name, codes in self.cat.items():
            out[f"pair_aff_{name}"] = np.asarray(self.aff[name][u, codes[i]]).ravel()
        out["pair_price_ratio"] = self.i_price[i] / np.where(
            self.u_price[u] > 0, self.u_price[u], np.nan
        )
        if self.user_text is not None:
            out["text_cos_recent"] = (self.emb[i] * self.user_text[u]).sum(1)
        feats = pd.DataFrame(out, index=df.index)
        return pd.concat([df, feats], axis=1)


def feature_groups(columns) -> dict[str, list[str]]:
    prefix = {
        "src_": "source",
        "cust_": "customer",
        "item_": "item",
        "pair_": "pair",
        "text_": "text",
    }
    out: dict[str, list[str]] = {g: [] for g in GROUPS}
    for c in columns:
        for p, g in prefix.items():
            if c.startswith(p):
                out[g].append(c)
    return out


@dataclass
class RankerResult:
    booster: lgb.Booster
    features: list[str]
    best_iteration: int


def train_ranker(
    frame: pd.DataFrame,
    features: list[str],
    seed: int = 0,
    valid_frac: float = 0.2,
    num_rounds: int = 500,
    params: dict | None = None,
) -> RankerResult:
    """LambdaRank; a held-out slice of customers chooses the number of trees."""
    users = frame["user"].unique()
    rng = np.random.default_rng(seed)
    valid_users = set(rng.choice(users, size=max(1, int(len(users) * valid_frac)), replace=False))
    is_val = frame["user"].isin(valid_users).to_numpy()
    cats = [c for c in CATEGORICAL if c in features]

    def part(mask: np.ndarray) -> lgb.Dataset:
        sub = frame[mask].sort_values("user", kind="stable")
        grp = sub.groupby("user", sort=False).size().to_numpy()
        return lgb.Dataset(
            sub[features], sub["label"], group=grp, categorical_feature=cats, free_raw_data=False
        )

    p = {
        "objective": "lambdarank",
        "metric": "map",
        "eval_at": [12],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambdarank_truncation_level": 24,
        "seed": seed,
        "verbosity": -1,
        "num_threads": 12,
        "deterministic": True,
        "force_row_wise": True,
    }
    p.update(params or {})
    booster = lgb.train(
        p,
        part(~is_val),
        num_boost_round=num_rounds,
        valid_sets=[part(is_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return RankerResult(booster, features, booster.best_iteration or num_rounds)


def rank_pool(
    result: RankerResult, frame: pd.DataFrame, users: np.ndarray, k: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer top-k articles and scores from the trained ranker (-1 pads short lists)."""
    pred = result.booster.predict(frame[result.features], num_iteration=result.best_iteration)
    scored = (
        frame[["user", "item"]].assign(s=pred).sort_values(["user", "s"], ascending=[True, False])
    )
    ranked = np.full((len(users), k), -1, dtype=np.int64)
    scores = np.full((len(users), k), -np.inf)
    pos = {int(u): j for j, u in enumerate(users)}
    for u, g in scored.groupby("user", sort=False):
        g = g.head(k)
        j = pos[int(u)]
        ranked[j, : len(g)] = g["item"].to_numpy()
        scores[j, : len(g)] = g["s"].to_numpy()
    return ranked, scores
