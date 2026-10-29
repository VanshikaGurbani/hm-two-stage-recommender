import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    import os

    os.environ["HM_SAMPLE_FRAC"] = "1.0"
    os.environ["HM_DATA_DIR"] = "__no_such_dir__"
    from hmrec.api import app

    with fastapi_testclient.TestClient(app) as c:
        yield c


def test_health_reports_synthetic_data(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["synthetic_data"] is True and r["customers"] > 0


def test_recommend_validates_and_caches(client):
    cid = client.get("/customers", params={"limit": 1}).json()["customer_ids"][0]
    a = client.get("/recommend", params={"customer_id": cid, "k": 5}).json()
    b = client.get("/recommend", params={"customer_id": cid, "k": 5}).json()
    assert len(a["recommendations"]) == 5 and a["cached"] is False and b["cached"] is True
    assert a["recommendations"] == b["recommendations"]
    assert all(r["why"] for r in a["recommendations"])
    assert len({r["article_id"] for r in a["recommendations"]}) == 5


def test_unknown_customer_and_bad_params(client):
    assert client.get("/recommend", params={"customer_id": "nope"}).status_code == 404
    cid = client.get("/customers").json()["customer_ids"][0]
    assert client.get("/recommend", params={"customer_id": cid, "k": 0}).status_code == 422
    assert client.get("/recommend", params={"customer_id": cid, "lam": 2}).status_code == 422


def test_mmr_parameter_changes_the_response(client):
    cid = client.get("/customers").json()["customer_ids"][1]
    plain = client.get("/recommend", params={"customer_id": cid, "lam": 1.0}).json()
    diverse = client.get("/recommend", params={"customer_id": cid, "lam": 0.3}).json()
    assert plain["recommendations"] != diverse["recommendations"]
