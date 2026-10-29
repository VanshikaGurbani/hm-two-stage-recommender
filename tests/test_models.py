import numpy as np
import pytest

from hmrec import data as D
from hmrec import models as Mo
from hmrec import splits as S
from hmrec.evaluate import Evaluator


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("hm")
    D.make_fixture(root, n_customers=300, n_articles=90, n_weeks=12, seed=5)
    hm = D.load(root, sample_frac=1.0)
    sp = S.weekly(hm.interactions)
    hist = sp.fit_frame
    return hm, sp, hist, Evaluator(hm, hist, sp.test), Mo.FitContext(hm, hist)


def _map(world, model):
    hm, sp, hist, ev, ctx = world
    return ev.evaluate(model.fit(ctx)).summary["map@12"]


def test_context_never_sees_the_future(world):
    hm, sp, hist, ev, ctx = world
    assert ctx.frame["ts"].max() < sp.test_start
    assert ctx.ref_ts <= sp.test_start
    assert ctx.popularity(7).sum() == (ctx.frame["ts"] >= ctx.ref_ts - 7 * S.DAY).sum()


def test_decayed_weights_halve_per_half_life(world):
    hm, sp, hist, ev, ctx = world
    one = ctx.decayed(7.0)
    two = ctx.decayed(14.0)
    assert one.sum() < two.sum() <= len(ctx.frame)
    assert (one.data > 0).all()
    # exact check: a purchase exactly one half-life before the reference weighs 0.5
    f = ctx.frame.iloc[:1].copy()
    f["ts"] = ctx.ref_ts - 7 * S.DAY
    tiny = Mo.FitContext(hm, f)
    tiny.__dict__["ref_ts"] = ctx.ref_ts
    assert tiny.decayed(7.0).sum() == pytest.approx(0.5)


def test_score_shapes_and_masks(world):
    hm, sp, hist, ev, ctx = world
    users = ev.users[:7]
    for m in (
        Mo.Popularity(7),
        Mo.Repurchase(),
        Mo.AgePopularity(),
        Mo.CoPurchase(days=84),
        Mo.ALS(factors=8, iterations=3),
    ):
        s = m.fit(ctx).score(users)
        assert s.shape == (7, hm.n_items), m.name
        assert not np.isnan(s).any(), m.name
    rep = Mo.Repurchase().fit(ctx).score(users)
    bought = np.asarray(ctx.counts[users].todense()) > 0
    assert (np.isfinite(rep) == bought).all()  # only previously bought articles are candidates


def test_taste_aware_models_beat_global_popularity_on_planted_data(world):
    pop = _map(world, Mo.Popularity(14))
    assert _map(world, Mo.Repurchase()) > 1.5 * pop
    assert _map(world, Mo.CoPurchase(days=84)) > pop
    assert _map(world, Mo.ALS(factors=8, iterations=5, days=84)) > pop


def test_models_are_deterministic(world):
    a = Mo.ALS(factors=8, iterations=3, seed=1).fit(world[4]).score(world[3].users[:5])
    b = Mo.ALS(factors=8, iterations=3, seed=1).fit(world[4]).score(world[3].users[:5])
    finite = np.isfinite(a)
    assert (finite == np.isfinite(b)).all()
    np.testing.assert_allclose(a[finite], b[finite], rtol=1e-3, atol=1e-4)
