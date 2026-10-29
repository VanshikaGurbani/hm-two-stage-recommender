import zipfile

import numpy as np
import pandas as pd
import pytest

from hmrec import data as D
from hmrec import splits as S
from hmrec.env import has_kaggle_credentials, kaggle_env, read_env


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("hm")
    D.make_fixture(root, n_customers=240, n_articles=90, n_weeks=10, seed=3)
    return root


@pytest.fixture(scope="module")
def hm(raw):
    return D.load(raw, sample_frac=1.0, seed=0)


def test_missing_data_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="accept"):
        D.load(tmp_path)


def test_shapes_and_indexing(hm):
    assert hm.n_users == 240 and hm.n_items == 90
    inter = hm.interactions
    assert inter["user"].between(0, hm.n_users - 1).all()
    assert inter["item"].between(0, hm.n_items - 1).all()
    assert inter["ts"].is_monotonic_increasing and (inter["ts"] % 86400 == 0).all()
    assert hm.tag_matrix().shape == (90, len(hm.tags))
    assert any(t.startswith("colour_group=") for t in hm.tags)
    assert "text" in hm.items.columns
    feats = D.customer_features(hm)
    assert len(feats) == 240 and feats["age"].notna().all()


def test_parquet_conversion_keeps_every_row(raw, hm):
    raw_rows = len(pd.read_csv(raw / "transactions_train.csv"))
    assert len(pd.read_parquet(raw / "transactions_train.parquet")) == raw_rows
    assert len(hm.interactions) == raw_rows


def test_customer_sampling_is_seeded_and_proportional(raw):
    a = D.load(raw, sample_frac=0.25, seed=1)
    b = D.load(raw, sample_frac=0.25, seed=1)
    c = D.load(raw, sample_frac=0.25, seed=2)
    assert a.n_users == 60
    assert (a.user_ids == b.user_ids).all() and not (a.user_ids == c.user_ids).all()
    assert a.interactions.equals(b.interactions)


def test_weekly_split_windows_and_no_leakage(hm):
    sp = S.weekly(hm.interactions)
    S.check_no_leakage(sp)
    last_day = hm.interactions["ts"].max() // 86400
    assert (sp.test["ts"] // 86400 > last_day - 7).all()
    assert (sp.val["ts"] // 86400).between(last_day - 13, last_day - 7).all()
    assert (sp.train["ts"] // 86400 <= last_day - 14).all()
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(hm.interactions)
    assert sp.fit_frame["ts"].max() < sp.test_start


def test_history_weeks_trims_only_train(hm):
    full = S.weekly(hm.interactions)
    short = S.weekly(hm.interactions, history_weeks=3)
    last_day = hm.interactions["ts"].max() // 86400
    assert len(short.train) < len(full.train)
    assert (short.train["ts"] // 86400 > last_day - 14 - 21).all()
    assert short.val.equals(full.val) and short.test.equals(full.test)


def test_leak_checker_catches_future_purchase_in_train(hm):
    sp = S.weekly(hm.interactions)
    bad = S.Split(sp.train.copy(), sp.val, sp.test, sp.val_start, sp.test_start)
    bad.train.loc[bad.train.index[-1], "ts"] = sp.test["ts"].max()
    with pytest.raises(AssertionError):
        S.check_no_leakage(bad)


def test_relevant_items_requires_prior_history(hm):
    sp = S.weekly(hm.interactions)
    rel = S.relevant_items(sp.test, sp.fit_frame)
    prior = set(sp.fit_frame["user"])
    assert rel and all(u in prior for u in rel)
    bought = sp.test.groupby("user")["item"].apply(set)
    assert all(set(items) == bought[u] for u, items in rel.items())


def test_extract_only_the_three_csvs_from_a_zip(raw, tmp_path):
    archive = tmp_path / "h-and-m-personalized-fashion-recommendations.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in D.FILES:
            zf.write(raw / name, name)
        zf.writestr("images/010/0108775015.jpg", b"not an image")
        zf.writestr("../escape.csv", "evil")
    out = D.extract_archive(archive, tmp_path / "data")
    assert sorted(p.name for p in out.iterdir()) == sorted(D.FILES)
    assert not (tmp_path / "escape.csv").exists()


def test_extract_reports_missing_member(tmp_path):
    archive = tmp_path / "x.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("articles.csv", "a")
    with pytest.raises(FileNotFoundError, match="transactions_train.csv"):
        D.extract_archive(archive, tmp_path / "d")


def test_env_parser_never_needs_values_exposed(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# c\nKAGGLE_USERNAME='someone'\nKAGGLE_KEY=abc\nOTHER=1\n", encoding="utf-8")
    assert read_env(p)["KAGGLE_KEY"] == "abc"
    assert has_kaggle_credentials(p)
    assert "OTHER" not in {
        k for k in kaggle_env(p) if k == "OTHER" and k not in __import__("os").environ
    }
    p.write_text("KAGGLE_USERNAME=\n", encoding="utf-8")
    assert not has_kaggle_credentials(p) or "KAGGLE_API_TOKEN" in kaggle_env(p)


def test_no_empty_candidate_universe(hm):
    # sanity on fixture: repeat purchases exist, which is why seen items are not masked
    pairs = hm.interactions.groupby(["user", "item"]).size()
    assert (pairs > 1).any() and np.isfinite(pairs.mean())
