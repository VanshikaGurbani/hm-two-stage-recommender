"""Streamlit demo: ``streamlit run app/streamlit_app.py``.

Pick a customer, see what they bought, then what the two-stage system recommends for next week and
why. The MMR slider trades accuracy for variety. Result tables come from ``results/results.json``
(produced by ``python -m hmrec.run``), never typed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from hmrec import workspace
from hmrec.images import ImageStore
from hmrec.pipeline import System

st.set_page_config(page_title="H&M two-stage recommender", layout="wide")


@st.cache_resource(show_spinner="Loading data and training the ranker (first run only)...")
def load_system(sample_frac: float) -> System:
    world = workspace.setup(sample_frac=sample_frac, seed=0)
    return System.build(world)


@st.cache_resource
def image_store() -> ImageStore:
    return ImageStore()


def results() -> dict | None:
    for folder in ("results", "docs/results"):  # the second is the committed copy
        p = Path(folder) / "results.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


st.title("H&M personalized recommendations")
sample = st.sidebar.select_slider("Customer sample", [0.02, 0.05, 0.1], value=0.1)
system = load_system(sample)
world = system.world
if world.synthetic:
    st.warning("Real H&M data not found: showing a tiny SYNTHETIC dataset. Numbers are not real.")

users = world.ev_test.users
st.sidebar.caption(f"{len(users):,} customers with purchases in the test week")
pick = st.sidebar.number_input("Customer #", min_value=0, max_value=len(users) - 1, value=0, step=1)
user = int(users[int(pick)])
k = st.sidebar.slider("How many recommendations", 4, 24, 12)
lam = st.sidebar.slider(
    "Relevance vs variety (MMR lambda)", 0.3, 1.0, 1.0, 0.1, help="1.0 = most relevant only"
)

images = image_store()
if not images.available:
    st.info("No product photos found (extract the competition zip or its images/ folder).")
tab_recs, tab_hist, tab_eval = st.tabs(["Recommendations", "Purchase history", "How good is it?"])

with tab_recs:
    recs = system.recommend(user, k=k, lam=lam)
    bought_next = set(
        world.split.test.loc[world.split.test["user"] == user, "item"].astype(int).tolist()
    )
    hits = int(recs["item"].isin(bought_next).sum())
    st.caption(
        f"In the held-out test week this customer actually bought {len(bought_next)} article(s); "
        f"{hits} of the {len(recs)} shown below were among them."
    )
    for _, r in recs.iterrows():
        mark = "✅ " if int(r["item"]) in bought_next else ""
        with st.container(border=True):
            pic, text = st.columns([1, 4])
            photo = images.get(int(r["article_id"])) if images.available else None
            if photo:
                pic.image(photo, width=110)
            text.markdown(f"**{mark}{r['name']}** · {r['product_type']} · {r['colour']}")
            text.caption(f"Why: {r['why']}  ·  score {r['score']:.3f}")

with tab_hist:
    st.dataframe(system.history(user, 25), use_container_width=True, hide_index=True)

with tab_eval:
    res = results()
    if res is None:
        st.info("Run `python -m hmrec.run` to generate results/results.json.")
    else:
        st.caption(f"Data: {res['data']}  |  evaluation customers: {res['eval_customers']}")
        base = pd.DataFrame(res["baselines_test"]).T[["map@12", "recall@12", "coverage"]]
        abl = pd.DataFrame(res["ranker"]["ablation"]).T[["map@12", "recall@12", "coverage"]]
        st.subheader("Baselines (MAP@12, test week)")
        st.bar_chart(base["map@12"])
        st.subheader("Ranker and feature-group ablation")
        st.dataframe(abl.round(4))
        st.subheader("Accuracy vs variety (MMR)")
        st.dataframe(pd.DataFrame(res["mmr_tradeoff"]).T.round(4))
