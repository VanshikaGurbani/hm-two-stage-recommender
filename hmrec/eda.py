"""Descriptive statistics of the purchase data, for the app's "Data analysis" tab.

``python -m hmrec.eda`` -> ``results/eda.json`` (and ``docs/results/eda.json``, the committed copy
the hosted app reads). The file holds aggregate counts only - no customer ids and no per-customer
rows - so it is safe to publish. ``summarize`` also runs on any loaded ``HMData`` (the app uses it
for the synthetic fixture when no JSON is available).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import splits as S

PURCHASE_BANDS = [(1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 24, "10-24")]
PURCHASE_BANDS += [(25, 49, "25-49"), (50, 99, "50-99"), (100, 10**9, "100+")]
AGE_BANDS = [(0, 19, "under 20"), (20, 29, "20s"), (30, 39, "30s"), (40, 49, "40s")]
AGE_BANDS += [(50, 59, "50s"), (60, 200, "60+")]


def _counts(values: np.ndarray, bands: list[tuple[int, int, str]]) -> list[dict]:
    out = []
    for lo, hi, label in bands:
        out.append({"band": label, "count": int(((values >= lo) & (values <= hi)).sum())})
    return out


def _top(items: pd.DataFrame, bought: pd.Series, column: str, n: int) -> list[dict]:
    names = items[column].to_numpy()[bought.to_numpy()]
    vc = pd.Series(names).replace("", "unknown").value_counts()
    total = int(vc.sum())
    return [
        {"name": str(k), "purchases": int(v), "share": float(v) / total}
        for k, v in vc.head(n).items()
    ]


def summarize(hm: D.HMData, history_weeks: int | None = 26) -> dict:
    df = hm.interactions.sort_values("ts", kind="stable")
    n = len(df)
    day = df["ts"].to_numpy() // S.DAY
    week = (day - day.min()) // 7
    start = pd.to_datetime(day.min() * S.DAY, unit="s")
    per_week = np.bincount(week)
    weekly = [
        {"week_start": str((start + pd.Timedelta(days=7 * i)).date()), "purchases": int(c)}
        for i, c in enumerate(per_week)
    ]

    per_customer = np.bincount(df["user"].to_numpy(), minlength=hm.n_users)
    prices = df["price"].to_numpy()
    top_price = float(np.quantile(prices, 0.99))
    counts, edges = np.histogram(np.clip(prices, 0, top_price), bins=20, range=(0, top_price))

    repeat = df.duplicated(["user", "item"]).to_numpy()
    per_item = np.bincount(df["item"].to_numpy(), minlength=hm.n_items)
    sold = np.sort(per_item[per_item > 0])[::-1]
    top1 = int(max(1, round(0.01 * len(sold))))

    cust = D.customer_features(hm)
    known_age = cust["age_missing"].to_numpy() == 0
    bought_items = df["item"]

    split = S.weekly(hm.interactions, history_weeks=history_weeks)
    seen = split.fit_frame[["user", "item"]].drop_duplicates()
    week_pairs = split.test[["user", "item"]].drop_duplicates()
    joined = week_pairs.merge(seen, on=["user", "item"], how="left", indicator=True)
    test_repeat = float((joined["_merge"] == "both").mean()) if len(joined) else 0.0

    return {
        "customers": int(hm.n_users),
        "articles_in_catalogue": int(hm.n_items),
        "articles_purchased": int((per_item > 0).sum()),
        "purchases": int(n),
        "days": int(day.max() - day.min() + 1),
        "first_day": str(start.date()),
        "weekly_purchases": weekly,
        "purchases_per_customer": _counts(per_customer, PURCHASE_BANDS),
        "median_purchases_per_customer": float(np.median(per_customer)),
        "mean_purchases_per_customer": float(per_customer.mean()),
        "repeat_purchase_share": float(repeat.mean()),
        "test_week_repeat_share": test_repeat,
        "product_groups": _top(hm.items, bought_items, "product_group", 12),
        "product_types": _top(hm.items, bought_items, "product_type", 12),
        "colours": _top(hm.items, bought_items, "colour", 10),
        "age_customers": _counts(cust["age"].to_numpy()[known_age], AGE_BANDS),
        "top1pct_articles_share_of_purchases": float(sold[:top1].sum() / n),
        "top1pct_articles_count": top1,
        "median_purchases_per_sold_article": float(np.median(sold)),
        "price_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "median_price": float(np.median(prices)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/eda.json")
    ap.add_argument("--docs-copy", default="docs/results/eda.json")
    args = ap.parse_args()
    hm = D.load(Path(args.data_dir), sample_frac=args.sample_frac, seed=args.seed)
    res = summarize(hm)
    res["sample_frac"], res["seed"] = args.sample_frac, args.seed
    for path in (args.out, args.docs_copy):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list | dict)}, indent=2))


if __name__ == "__main__":
    main()
