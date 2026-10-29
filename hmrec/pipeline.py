"""End-to-end recommender used by the Streamlit app and the API.

``System.build`` trains the LightGBM ranker on the validation week (retrievers fitted on the
training weeks), then refits the retrievers on train + validation so that ``recommend`` serves
"what will this customer buy next week". Explanations are assembled from facts the system actually
holds (which sources proposed the article, how often the customer bought it, ...), never invented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import candidates as C
from . import models as Mo
from . import ranker as R
from .rerank import mmr_select
from .workspace import World

SOURCE_LABELS = {
    "repurchase": "you bought it before",
    "popularity": "trending this week",
    "age_pop": "popular with customers your age",
    "co_purchase": "often bought together with your recent items",
    "als": "liked by customers with similar taste",
    "two_tower": "matches the style of your recent purchases",
}


def default_sources(
    ctx: Mo.FitContext, history_weeks: int = 26, tt_epochs: int = 6, ann: bool = False
) -> dict[str, C.Source]:
    sources: dict[str, C.Source] = {}
    try:  # the two-tower source needs torch; the rest of the system works without it
        from .twotower import TwoTower

        tt = TwoTower(epochs=tt_epochs, days=history_weeks * 7).fit(ctx)
        sources["two_tower"] = C.Source(tt.ann_source(100) if ann else tt, 100)
    except ImportError:
        pass
    sources.update(
        {
            "repurchase": C.Source(Mo.Repurchase(28).fit(ctx), 40),
            "popularity": C.Source(Mo.Popularity(7).fit(ctx), 30),
            "age_pop": C.Source(Mo.AgePopularity(14).fit(ctx), 30),
            "co_purchase": C.Source(Mo.CoPurchase(days=56).fit(ctx), 50),
            "als": C.Source(Mo.ALS(factors=64, days=history_weeks * 7).fit(ctx), 50),
        }
    )
    return sources


@dataclass
class System:
    world: World
    sources: dict[str, C.Source]
    ranker: R.RankerResult
    features: R.FeatureBuilder
    build_seconds: float = 0.0

    @classmethod
    def build(
        cls, world: World, history_weeks: int = 26, threads: int = 8, tt_epochs: int = 6
    ) -> System:
        t0 = time.time()
        ev = world.ev_val
        pool = C.generate(
            default_sources(world.ctx_train, history_weeks, tt_epochs), ev.users, ev.rel
        )
        frame = R.FeatureBuilder(world.hm, world.ctx_train, world.emb).featurize(pool.frame)
        feats = [c for c in frame.columns if c not in ("user", "item", "label")]
        ranker = R.train_ranker(frame, feats, params={"num_threads": threads})
        sources = default_sources(world.ctx_fit, history_weeks, tt_epochs, ann=True)
        fb = R.FeatureBuilder(world.hm, world.ctx_fit, world.emb)
        return cls(world, sources, ranker, fb, time.time() - t0)

    # ------------------------------------------------------------------ serving
    def candidates(self, user: int) -> pd.DataFrame:
        return C.generate(self.sources, np.array([user]), None).frame

    def recommend(self, user: int, k: int = 12, lam: float = 1.0, pool: int = 50) -> pd.DataFrame:
        """Top-k articles for one customer; ``lam < 1`` applies MMR diversity re-ranking."""
        frame = self.features.featurize(self.candidates(user))
        scores = self.ranker.booster.predict(
            frame[self.ranker.features], num_iteration=self.ranker.best_iteration
        )
        frame = frame.assign(score=scores).sort_values("score", ascending=False)
        top = frame.head(pool)
        if lam < 1.0:
            v = self.world.emb
            chosen = mmr_select(top["item"].to_numpy(), top["score"].to_numpy(), v, k, lam)
            top = top.set_index("item").loc[chosen].reset_index()
        else:
            top = top.head(k)
        return self._decorate(top.reset_index(drop=True))

    def _decorate(self, top: pd.DataFrame) -> pd.DataFrame:
        items = self.world.hm.items
        out = top[["item", "score"]].copy()
        out["article_id"] = self.world.hm.item_ids[out["item"].to_numpy()]
        for col in ("name", "product_type", "colour", "department"):
            out[col] = items[col].to_numpy()[out["item"].to_numpy()]
        out["why"] = [self.explain(r) for _, r in top.iterrows()]
        return out

    def explain(self, row: pd.Series) -> str:
        """Plain-language reasons built only from recorded facts about this (customer, article)."""
        reasons = []
        times = row.get("pair_times_bought", 0)
        if times and times > 0:
            days = row.get("pair_days_since_bought")
            when = f", last {int(days)} days ago" if days == days and days is not None else ""
            reasons.append(f"you bought it {int(times)}x{when}")
        for name, label in SOURCE_LABELS.items():
            rank = row.get(f"src_{name}_rank")
            if name != "repurchase" and rank == rank and rank is not None:
                reasons.append(f"{label} (ranked #{int(rank) + 1} by that signal)")
        aff = row.get("pair_aff_product_type", 0)
        if aff and aff >= 0.2:
            reasons.append(f"{aff:.0%} of your purchases are this product type")
        return "; ".join(reasons) if reasons else "general popularity among similar candidates"

    def history(self, user: int, n: int = 10) -> pd.DataFrame:
        f = self.world.ctx_fit.frame
        h = f[f["user"] == user].sort_values("ts", ascending=False).head(n)
        items = self.world.hm.items
        return pd.DataFrame(
            {
                "date": pd.to_datetime(h["ts"], unit="s").dt.date.to_numpy(),
                "name": items["name"].to_numpy()[h["item"].to_numpy()],
                "product_type": items["product_type"].to_numpy()[h["item"].to_numpy()],
                "colour": items["colour"].to_numpy()[h["item"].to_numpy()],
                "price": h["price"].to_numpy(),
            }
        )
