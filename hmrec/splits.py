"""Week-based temporal split for purchase logs.

Weeks are 7-day blocks counted back from the last day in the data:

* **test**  - the final week (what we are scored on);
* **val**   - the week before it (the ranker's training labels);
* **train** - everything older (optionally only the newest ``history_weeks`` weeks).

Retrievers and features used for a target week may only see data strictly before that week.
``check_no_leakage`` verifies this on the cut-offs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86400


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_start: int  # first second of the validation week
    test_start: int  # first second of the test week

    @property
    def fit_frame(self) -> pd.DataFrame:
        """Everything a final model may see before the test week (train + validation)."""
        return sort_purchases(pd.concat([self.train, self.val], ignore_index=True))


def sort_purchases(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["ts", "item", "user"], kind="stable").reset_index(drop=True)


def weekly(df: pd.DataFrame, history_weeks: int | None = None) -> Split:
    df = sort_purchases(df)
    day = df["ts"] // DAY
    last_day = int(day.max())
    weeks_ago = (last_day - day) // 7
    train_mask = weeks_ago >= 2
    if history_weeks is not None:
        train_mask &= weeks_ago < history_weeks + 2
    return Split(
        df[train_mask].reset_index(drop=True),
        df[weeks_ago == 1].reset_index(drop=True),
        df[weeks_ago == 0].reset_index(drop=True),
        val_start=(last_day - 13) * DAY,
        test_start=(last_day - 6) * DAY,
    )


def check_no_leakage(split: Split) -> None:
    """Raise AssertionError if train/val/test overlap in time or are out of order."""
    assert split.train["ts"].max() < split.val_start <= split.val["ts"].min()
    assert split.val["ts"].max() < split.test_start <= split.test["ts"].min()


def relevant_items(
    target: pd.DataFrame, history: pd.DataFrame, require_history: bool = True
) -> dict[int, np.ndarray]:
    """customer -> unique items bought in the target week (only customers with prior history)."""
    t = target
    if require_history:
        t = t[t["user"].isin(set(history["user"].unique()))]
    return {int(u): g["item"].unique() for u, g in t.groupby("user")}  # type: ignore[arg-type]
