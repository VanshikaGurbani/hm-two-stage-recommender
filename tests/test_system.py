import numpy as np
import pytest

from hmrec import workspace
from hmrec.pipeline import System


@pytest.fixture(scope="module")
def system():
    world = workspace.setup(allow_fixture=True, data_dir="__no_such_dir__")
    assert world.synthetic
    return System.build(world, history_weeks=8, threads=2)


def test_recommend_returns_k_distinct_articles_with_facts(system):
    user = int(system.world.ev_test.users[0])
    recs = system.recommend(user, k=12)
    assert len(recs) == 12 and recs["item"].is_unique
    assert recs["score"].is_monotonic_decreasing
    assert recs["why"].str.len().gt(0).all()
    assert set(recs["article_id"]) <= set(system.world.hm.item_ids)


def test_explanations_only_cite_true_purchase_facts(system):
    user = int(system.world.ev_test.users[3])
    recs = system.recommend(user, k=12)
    hist = system.world.ctx_fit.frame
    bought = hist[hist["user"] == user].groupby("item").size()
    for _, r in recs.iterrows():
        if "you bought it" in r["why"]:
            n = int(r["why"].split("you bought it ")[1].split("x")[0])
            assert bought.get(int(r["item"]), 0) == n
        else:
            assert int(r["item"]) not in bought.index or "you bought" not in r["why"]


def test_mmr_changes_the_list_but_keeps_size(system):
    user = int(system.world.ev_test.users[1])
    plain = system.recommend(user, k=12, lam=1.0)
    diverse = system.recommend(user, k=12, lam=0.3)
    assert len(diverse) == 12 and diverse["item"].is_unique
    assert set(plain["item"]) != set(diverse["item"]) or not np.allclose(
        plain["score"].to_numpy(), diverse["score"].to_numpy()
    )


def test_history_view(system):
    user = int(system.world.ev_test.users[0])
    h = system.history(user, 5)
    assert 1 <= len(h) <= 5 and h["date"].is_monotonic_decreasing


def test_system_builds_without_torch(monkeypatch):
    """Hosting the demo without PyTorch must skip the two-tower source, not crash."""
    import sys

    monkeypatch.setitem(sys.modules, "hmrec.twotower", None)  # makes the import raise
    world = workspace.setup(allow_fixture=True, data_dir="__no_such_dir__")
    s = System.build(world, history_weeks=8, threads=2)
    assert "two_tower" not in s.sources and len(s.sources) == 5
    user = int(world.ev_test.users[0])
    assert len(s.recommend(user, k=5)) == 5
