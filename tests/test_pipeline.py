import numpy as np
import pytest

from hmrec import candidates as C
from hmrec import data as D
from hmrec import embed
from hmrec import models as Mo
from hmrec import ranker as R
from hmrec import splits as S
from hmrec.evaluate import Evaluator


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("hm")
    D.make_fixture(root, n_customers=400, n_articles=90, n_weeks=14, seed=7)
    hm = D.load(root, sample_frac=1.0)
    sp = S.weekly(hm.interactions)
    emb = embed.item_embeddings(hm, "tfidf", cache_dir=root / "cache")
    return hm, sp, emb


def _sources(ctx):
    return {
        "pop": C.Source(Mo.Popularity(14).fit(ctx), 20),
        "repeat": C.Source(Mo.Repurchase().fit(ctx), 20),
        "co": C.Source(Mo.CoPurchase(days=98).fit(ctx), 20),
    }


def test_embeddings_are_unit_norm_and_cached(world, tmp_path):
    hm, sp, emb = world
    assert emb.shape[0] == hm.n_items
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    again = embed.item_embeddings(hm, "tfidf", cache_dir=tmp_path)
    np.testing.assert_allclose(again, emb, atol=1e-5)


def test_candidate_pool_labels_recall_and_union_bound(world):
    hm, sp, emb = world
    ctx = Mo.FitContext(hm, sp.train)
    ev = Evaluator(hm, sp.train, sp.val)
    pool = C.generate(_sources(ctx), ev.users, ev.rel)
    f = pool.frame
    assert set(f["user"]) <= set(ev.users) and f.duplicated(["user", "item"]).sum() == 0
    assert pool.recall["union"] >= max(v for k, v in pool.recall.items() if k != "union") - 1e-9
    # labels agree with the evaluator's relevance matrix
    pos = {int(u): j for j, u in enumerate(ev.users)}
    check = f.sample(200, random_state=0)
    for _, r in check.iterrows():
        assert r["label"] == int(ev.rel[pos[int(r["user"])], int(r["item"])])
    assert f["label"].sum() > 0


def test_features_use_only_history_and_cover_all_groups(world):
    hm, sp, emb = world
    ctx = Mo.FitContext(hm, sp.train)
    ev = Evaluator(hm, sp.train, sp.val)
    pool = C.generate(_sources(ctx), ev.users, ev.rel)
    fb = R.FeatureBuilder(hm, ctx, emb)
    feats = fb.featurize(pool.frame)
    groups = R.feature_groups([c for c in feats.columns if c not in ("user", "item", "label")])
    assert all(groups[g] for g in R.GROUPS)
    # recency features are measured at the end of the history, never past it
    assert (feats["cust_days_since_last"] >= 0).all()
    assert (feats["pair_days_since_bought"].dropna() >= 0).all()
    # a changed validation-week purchase must not change any feature
    sp2 = S.weekly(hm.interactions)
    sp2.val["item"] = (sp2.val["item"] + 1) % hm.n_items
    ctx2 = Mo.FitContext(hm, sp2.train)
    feats2 = R.FeatureBuilder(hm, ctx2, emb).featurize(pool.frame)
    cols = [c for c in feats.columns if c not in ("label",) and not c.startswith("src_")]
    np.testing.assert_allclose(
        feats[cols].to_numpy(dtype=float), feats2[cols].to_numpy(dtype=float), equal_nan=True
    )


def test_ranker_end_to_end_beats_its_weakest_source(world):
    hm, sp, emb = world
    ctx_tr = Mo.FitContext(hm, sp.train)
    ev_val = Evaluator(hm, sp.train, sp.val)
    pool_val = C.generate(_sources(ctx_tr), ev_val.users, ev_val.rel)
    fb = R.FeatureBuilder(hm, ctx_tr, emb)
    frame = fb.featurize(pool_val.frame)
    feats = [c for c in frame.columns if c not in ("user", "item", "label")]
    res = R.train_ranker(frame, feats, params={"num_threads": 2, "min_data_in_leaf": 20})

    hist = sp.fit_frame
    ctx_fit = Mo.FitContext(hm, hist)
    ev = Evaluator(hm, hist, sp.test)
    pool = C.generate(_sources(ctx_fit), ev.users, ev.rel)
    frame_t = R.FeatureBuilder(hm, ctx_fit, emb).featurize(pool.frame)
    ranked, scores = R.rank_pool(res, frame_t, ev.users)
    out = ev.metrics_from_ranking(ranked, "ranker")
    pop = ev.evaluate(Mo.Popularity(14).fit(ctx_fit))
    assert out.summary["map@12"] > pop.summary["map@12"]
    assert ranked.shape == (len(ev.users), 200) and (ranked[:, 0] >= 0).all()
