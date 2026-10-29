import numpy as np
import pytest
import scipy.sparse as sp

from hmrec import metrics as M
from hmrec.data import load as load_hm
from hmrec.data import make_fixture
from hmrec.evaluate import Evaluator, bootstrap_ci, paired_bootstrap, topk
from hmrec.splits import weekly


def rel_matrix(rows, n_items):
    r = sp.lil_matrix((len(rows), n_items), dtype=bool)
    for u, items in enumerate(rows):
        for i in items:
            r[u, i] = True
    return r.tocsr()


def test_map_at_12_hand_computed():
    ranked = np.array([[5, 9, 1, 8, 2, 7, 3, 6, 4, 0, 10, 11]])
    rel = rel_matrix([[5, 8, 2]], 20)
    # purchased items sit at ranks 1, 4, 5: (1/1 + 2/4 + 3/5) / min(3, 12)
    assert M.map_at_12(ranked, rel)[0] == pytest.approx((1 + 0.5 + 0.6) / 3)


def test_map_denominator_is_capped_at_12():
    many = rel_matrix([list(range(15))], 30)
    assert M.map_at_12(np.arange(12)[None, :], many)[0] == pytest.approx(1.0)


def test_map_zero_without_hits_and_padding_ignored():
    rel = rel_matrix([[3]], 10)
    assert M.map_at_12(np.array([[0, 1, -1, -1]]), rel)[0] == 0.0
    assert M.hit_rate_at_k(np.array([[3, -1]]), rel, 2)[0] == 1.0


def test_recall_and_ndcg_hand_computed():
    ranked = np.array([[3, 1, 7, 0], [5, 6, 2, 9]])
    rel = rel_matrix([[1, 0, 8], [4]], 10)
    np.testing.assert_allclose(M.recall_at_k(ranked, rel, 4), [2 / 3, 0])
    np.testing.assert_allclose(M.recall_at_k(ranked, rel, 1), [0, 0])
    r2 = rel_matrix([[1]], 10)
    np.testing.assert_allclose(M.ndcg_at_k(np.array([[9, 1, 2]]), r2, 3), [1 / np.log2(3)])
    perfect = rel_matrix([[0, 1, 2, 3, 4]], 10)
    np.testing.assert_allclose(M.ndcg_at_k(np.array([[0, 1]]), perfect, 2), [1.0])


def test_candidate_recall_counts_union_once_and_ignores_padding():
    rel = rel_matrix([[1, 2, 3], [4]], 10)
    cands = np.array([[1, 1, 2, -1, 7], [9, 8, -1, -1, -1]])
    np.testing.assert_allclose(M.candidate_recall(cands, rel), [2 / 3, 0.0])
    np.testing.assert_allclose(M.candidate_recall(cands, rel, k=2), [1 / 3, 0.0])


def test_coverage_gini_novelty_and_popularity_bias():
    even = np.array([[0, 1], [2, 3]])
    assert M.catalog_coverage(even, 8, 2) == 0.5
    assert M.exposure_gini(even, 4, 2) == pytest.approx(0.0)
    assert M.exposure_gini(np.array([[0, 1], [0, 1]]), 4, 2) > 0.4
    pop = np.array([8, 4, 2, 1])
    ranked = np.array([[0], [3]])
    assert M.novelty(ranked, pop, 8, 1) == pytest.approx(1.5)
    assert M.avg_recommendation_popularity(ranked, pop, 1) == pytest.approx(4.5)
    assert M.head_share(ranked, pop, 1, head_frac=0.25) == pytest.approx(0.5)
    assert M.tail_share(ranked, pop, 1, tail_frac=0.25) == pytest.approx(0.5)


def test_diversity_metrics():
    vec = np.array([[1.0, 0], [1.0, 0], [0, 1.0]])
    ranked = np.array([[0, 1], [0, 2]])
    np.testing.assert_allclose(M.intra_list_diversity(ranked, vec, 2), [0.0, 1.0])
    tags = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    np.testing.assert_allclose(M.tag_diversity(ranked, tags, 2), [0.0, 1.0])


def test_topk_handles_neg_inf_and_ties():
    scores = np.array([[0.9, 0.8, -np.inf, 0.8]])
    assert list(topk(scores, 4)[0]) == [0, 1, 3, -1]


def test_bootstrap_is_seeded_and_covers_the_mean():
    x = np.random.default_rng(0).normal(1.0, 1.0, 500)
    lo, hi = bootstrap_ci(x, seed=3)
    assert lo < x.mean() < hi and (lo, hi) == bootstrap_ci(x, seed=3)
    res = paired_bootstrap(x + 0.5, x)
    assert res["significant"] and res["diff"] == pytest.approx(0.5)


class _Repeat:
    """Scores articles by how many times the customer already bought them."""

    name = "repeat"

    def __init__(self, matrix):
        self.m = matrix

    def score(self, users):
        return np.asarray(self.m[users].todense(), dtype=np.float32)


def test_evaluator_rewards_repurchase_and_only_uses_customers_with_history(tmp_path):
    make_fixture(tmp_path, n_customers=240, n_articles=90, n_weeks=10, seed=3)
    hm = load_hm(tmp_path, sample_frac=1.0)
    sp_ = weekly(hm.interactions)
    from hmrec.evaluate import purchase_matrix

    hist = sp_.fit_frame
    ev = Evaluator(hm, hist, sp_.test)
    assert len(ev.users) > 20 and set(ev.users) <= set(hist["user"])
    res = ev.evaluate(_Repeat(purchase_matrix(hist, hm.n_users, hm.n_items)))
    rand = ev.evaluate(
        type(
            "R",
            (),
            {
                "name": "r",
                "score": lambda s, u: np.random.default_rng(0).random((len(u), hm.n_items)),
            },
        )()
    )
    # tiny 90-article catalogue: 12 slots already cover 13% of it, so random is not near zero
    assert res.summary["map@12"] > 2.5 * rand.summary["map@12"]
    assert 0 <= res.summary["coverage"] <= 1 and len(res.ci["map@12"]) == 2
