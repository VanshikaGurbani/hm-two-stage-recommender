import json

import pytest

from hmrec import eda, workspace


@pytest.fixture(scope="module")
def stats():
    world = workspace.setup(allow_fixture=True, data_dir="__no_such_dir__")
    return world, eda.summarize(world.hm, history_weeks=8)


def test_counts_match_the_data(stats):
    world, e = stats
    assert e["purchases"] == len(world.hm.interactions)
    assert sum(w["purchases"] for w in e["weekly_purchases"]) == e["purchases"]
    assert sum(b["count"] for b in e["purchases_per_customer"]) <= e["customers"]
    assert sum(h for h in e["price_histogram"]["counts"]) == e["purchases"]


def test_shares_are_proportions(stats):
    _, e = stats
    for key in (
        "repeat_purchase_share",
        "test_week_repeat_share",
        "top1pct_articles_share_of_purchases",
    ):
        assert 0.0 <= e[key] <= 1.0
    assert sum(g["share"] for g in e["product_groups"]) <= 1.0 + 1e-9


def test_summary_is_json_serialisable_and_has_no_customer_ids(stats):
    world, e = stats
    text = json.dumps(e)
    assert str(world.hm.user_ids[0]) not in text
