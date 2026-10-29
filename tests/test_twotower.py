import numpy as np
import pandas as pd
import pytest

from hmrec import workspace
from hmrec.ann import ItemIndex
from hmrec.twotower import TwoTower, build_samples, last_history


def _frame():
    # customer 0 buys items 5, 6, 7, 8 on consecutive days; customer 1 buys 1, 2
    rows = [(0, 5, 1), (0, 6, 2), (0, 7, 3), (0, 8, 4), (1, 1, 1), (1, 2, 2)]
    return pd.DataFrame(rows, columns=["user", "item", "ts"]).assign(price=0.1, channel=1)


def test_build_samples_uses_only_previous_purchases_of_the_same_customer():
    hist, target = build_samples(_frame(), n_users=2, hist_len=3)
    pairs = {(tuple(h[h >= 0]), int(t)) for h, t in zip(hist, target, strict=True)}
    assert pairs == {
        ((5,), 6),
        ((5, 6), 7),
        ((5, 6, 7), 8),
        ((1,), 2),
    }
    # padding is on the left and the newest purchase is last
    h, t = next((h, t) for h, t in zip(hist, target, strict=True) if t == 7)
    assert list(h) == [-1, 5, 6]


def test_last_history_keeps_newest_and_pads_left():
    out = last_history(_frame(), n_users=3, hist_len=3)
    assert list(out[0]) == [6, 7, 8]
    assert list(out[1]) == [-1, 1, 2]
    assert list(out[2]) == [-1, -1, -1]  # customer without purchases


@pytest.fixture(scope="module")
def trained():
    world = workspace.setup(allow_fixture=True, data_dir="__no_such_dir__")
    model = TwoTower(dim=16, hidden=32, hist_len=6, epochs=3, batch=256, days=400).fit(
        world.ctx_fit
    )
    return world, model


def test_two_tower_scores_and_learns(trained):
    world, model = trained
    users = world.ev_test.users[:10]
    s = model.score(users)
    assert s.shape == (10, world.hm.n_items) and np.isfinite(s).all()
    assert model.log[-1]["loss"] < model.log[0]["loss"]
    np.testing.assert_allclose(np.linalg.norm(model.item_vecs, axis=1), 1.0, atol=1e-4)


def test_ann_matches_exact_search_on_small_catalogue(trained):
    world, model = trained
    users = world.ev_test.users[:50]
    assert model.ann_recall(users, n=20) > 0.95
    exact = ItemIndex(model.item_vecs, "flat")
    _, ids = exact.search(model.user_vectors(model.user_hist[users]), 5)
    best_dense = np.argsort(-model.score(users), axis=1)[:, :5]
    assert (ids[:, 0] == best_dense[:, 0]).mean() > 0.95
