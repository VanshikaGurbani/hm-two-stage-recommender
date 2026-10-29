"""Generates the analysis notebooks (``python notebooks/build_notebooks.py``).

Notebooks are committed *without outputs*; run them in Jupyter, or execute them all with
``python notebooks/build_notebooks.py --execute`` to embed the outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


SETUP = """
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from hmrec import workspace
pd.set_option("display.width", 140); plt.rcParams["figure.dpi"] = 110

# 10% of customers, seeded. Falls back to a tiny synthetic fixture (with a warning) if the
# real data has not been downloaded yet.
W = workspace.setup(sample_frac=0.10, seed=0)
hm, split = W.hm, W.split
if W.synthetic:
    print("*** SYNTHETIC DATA - numbers below are not H&M results ***")
"""

NOTEBOOKS: dict[str, list] = {}

NOTEBOOKS["01_data_exploration.ipynb"] = [
    md("""# 1. Data exploration
What does the purchase log look like, and what does that imply for modelling?
Key questions: how sparse is it, how much do customers *re*-buy, how skewed is popularity, and how
fast does demand drift week to week (which decides how recent the history must be)."""),
    code(SETUP),
    code("""
from hmrec.data import describe
pd.Series(describe(hm)).to_frame("value")"""),
    md("## Purchases per week"),
    code("""
inter = hm.interactions.copy()
inter["day"] = inter["ts"] // 86400
last = inter["day"].max()
inter["weeks_ago"] = (last - inter["day"]) // 7
weekly = inter.groupby("weeks_ago").size().sort_index(ascending=False)
ax = weekly.plot(figsize=(10, 3), title="Purchases per week (right edge = test week)")
ax.set_xlabel("weeks before the end of the data"); plt.show()"""),
    md("## Customer activity and repurchasing"),
    code("""
per_cust = inter.groupby("user").size()
print(per_cust.describe().round(2).to_string())
pairs = inter.groupby(["user", "item"]).size()
repeat_share = (pairs > 1).sum() / len(pairs)
bought_before = inter.sort_values("ts").duplicated(["user", "item"]).mean()
print(f"share of (customer, article) pairs bought more than once: {repeat_share:.1%}")
print(f"share of ALL purchases that are a repeat of an earlier purchase: {bought_before:.1%}")
per_cust.clip(upper=per_cust.quantile(0.99)).hist(bins=50, figsize=(8, 3)); plt.title("Purchases per customer"); plt.show()"""),
    md("""**Reading:** if a large share of purchases are repeats, a model that only recommends
*new* articles throws away its best signal - which is why the evaluator in this project does not
mask articles a customer already owns."""),
    md("## Popularity is a long tail"),
    code("""
pop = inter.groupby("item").size().sort_values(ascending=False)
cum = pop.cumsum() / pop.sum()
for q in (0.01, 0.05, 0.2):
    n = int(len(hm.items) * q)
    print(f"top {q:.0%} of articles ({n}) account for {cum.iloc[min(n, len(cum)) - 1]:.1%} of purchases")
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].plot(np.arange(len(pop)), pop.to_numpy()); ax[0].set_yscale("log"); ax[0].set_title("articles ranked by purchases (log y)")
ax[1].plot(np.arange(len(cum)) / len(hm.items), cum.to_numpy()); ax[1].set_title("cumulative share of purchases"); plt.show()"""),
    md("## Demand drift: how long does 'popular' stay popular?"),
    code("""
top_last_week = set(inter[inter.weeks_ago == 0].groupby("item").size().nlargest(50).index)
rows = []
for w in range(1, min(12, int(inter.weeks_ago.max())) + 1):
    top_w = set(inter[inter.weeks_ago == w].groupby("item").size().nlargest(50).index)
    rows.append((w, len(top_last_week & top_w) / 50))
pd.DataFrame(rows, columns=["weeks_before", "overlap_of_top50_with_last_week"]).set_index("weeks_before").plot(
    kind="bar", figsize=(8, 3), legend=False, title="Overlap of weekly top-50 with the final week's top-50"); plt.show()"""),
    md("## Who do we evaluate?"),
    code("""
print(f"validation week: {len(W.ev_val.users):,} customers with purchases and prior history")
print(f"test week:       {len(W.ev_test.users):,} customers with purchases and prior history")
print(f"split sizes (purchases): train={len(split.train):,} val={len(split.val):,} test={len(split.test):,}")
print("No future leakage:", "checked" if split.train.ts.max() < split.val_start <= split.val.ts.min() else "FAILED")"""),
]

NOTEBOOKS["02_baselines.ipynb"] = [
    md("""# 2. Baselines
Simple models first. Every number comes from the same evaluator: MAP@12 over the **whole catalogue**,
for customers who bought something in the test week. Hyper-parameters are chosen on the *validation*
week; the test week is only used to report."""),
    code(SETUP),
    code("""
from hmrec import models as Mo
from hmrec.evaluate import paired_bootstrap
grids = {
    "popularity": [Mo.Popularity(d) for d in (7, 14, 28)],
    "repurchase": [Mo.Repurchase(h) for h in (7, 14, 28, 56)],
    "age-popularity": [Mo.AgePopularity(d) for d in (7, 14)],
    "co-purchase": [Mo.CoPurchase(days=d) for d in (28, 56)],
    "als": [Mo.ALS(factors=f, days=(26 * 7)) for f in (32, 64)],
}
val_rows = []
for family, models in grids.items():
    for m in models:
        m.fit(W.ctx_train)
        val_rows.append((family, m.name, W.ev_val.evaluate(m).summary["map@12"]))
val = pd.DataFrame(val_rows, columns=["family", "config", "val_map@12"])
val.sort_values("val_map@12", ascending=False)"""),
    md("### Pick the best config per family on validation, report on test"),
    code("""
best = val.loc[val.groupby("family")["val_map@12"].idxmax()]
factory = {m.name: m for ms in grids.values() for m in ms}
results = {}
for _, row in best.iterrows():
    m = type(factory[row.config])  # refit the chosen config on train+val
    model = factory[row.config].fit(W.ctx_fit)
    results[row.family] = W.ev_test.evaluate(model, row.family)
table = pd.DataFrame({k: v.summary for k, v in results.items()}).T
ci = pd.Series({k: v.ci["map@12"] for k, v in results.items()})
table["map@12 95% CI"] = ci.map(lambda x: f"[{x[0]:.4f}, {x[1]:.4f}]")
table[["map@12", "map@12 95% CI", "recall@12", "recall@100", "coverage", "novelty", "head_share"]].sort_values("map@12", ascending=False).round(4)"""),
    code("""
names = list(results)
fig, ax = plt.subplots(figsize=(8, 3))
means = [results[n].summary["map@12"] for n in names]
errs = [[results[n].summary["map@12"] - results[n].ci["map@12"][0] for n in names],
        [results[n].ci["map@12"][1] - results[n].summary["map@12"] for n in names]]
ax.barh(names, means, xerr=errs); ax.set_xlabel("MAP@12 (test week, 95% bootstrap CI)"); plt.show()"""),
    md("""**Reading:** repurchase usually beats global popularity by a wide margin on this kind of
data. Is the best simple model *significantly* better than the runner-up? Check with a paired bootstrap:"""),
    code("""
ranked_models = sorted(results.values(), key=lambda r: -r.summary["map@12"])
a, b = ranked_models[0], ranked_models[1]
print(a.model, "vs", b.model, paired_bootstrap(a.per_user["map@12"], b.per_user["map@12"]))"""),
]

NOTEBOOKS["03_candidates_and_ranker.ipynb"] = [
    md("""# 3. Two-stage recommender: candidates then ranker
Stage 1 proposes a few hundred articles per customer from several cheap sources; stage 2 (LightGBM
LambdaRank) orders them. Candidate **recall** is the ceiling for the whole system: the ranker cannot
recommend what stage 1 never proposed."""),
    code(SETUP),
    code("""
from hmrec import models as Mo, candidates as C, ranker as R
def sources(ctx):
    return {
        "repurchase": C.Source(Mo.Repurchase(28).fit(ctx), 40),
        "popularity": C.Source(Mo.Popularity(7).fit(ctx), 30),
        "age_pop":    C.Source(Mo.AgePopularity(14).fit(ctx), 30),
        "co_purchase": C.Source(Mo.CoPurchase(days=56).fit(ctx), 50),
        "als":        C.Source(Mo.ALS(factors=64, days=26 * 7).fit(ctx), 50),
    }
pool_val = C.generate(sources(W.ctx_train), W.ev_val.users, W.ev_val.rel)   # labels: validation week
pool_test = C.generate(sources(W.ctx_fit), W.ev_test.users, W.ev_test.rel)  # scored on: test week
print(f"avg candidates per customer: {pool_test.avg_candidates:.0f}")
pd.Series(pool_test.recall, name="candidate recall (test)").sort_values(ascending=False).round(4).to_frame()"""),
    md(
        "## Features\nBuilt only from purchases before the week being predicted (a unit test fails if that is violated)."
    ),
    code("""
fb_val = R.FeatureBuilder(hm, W.ctx_train, W.emb); fb_test = R.FeatureBuilder(hm, W.ctx_fit, W.emb)
frame_val = fb_val.featurize(pool_val.frame); frame_test = fb_test.featurize(pool_test.frame)
feats = [c for c in frame_val.columns if c not in ("user", "item", "label")]
groups = R.feature_groups(feats)
{g: len(v) for g, v in groups.items()}"""),
    md("## Train, evaluate, and ablate feature groups"),
    code("""
def fit_eval(name, cols):
    rk = R.train_ranker(frame_val, cols, seed=0)
    ranked, scores = R.rank_pool(rk, frame_test, W.ev_test.users)
    return rk, ranked, scores, W.ev_test.metrics_from_ranking(ranked, name)
runs = {"full": fit_eval("full", feats)}
for g in R.GROUPS:
    runs[f"without {g}"] = fit_eval(g, [c for c in feats if c not in groups[g]])
runs["source scores only"] = fit_eval("src", groups["source"])
abl = pd.DataFrame({k: v[3].summary for k, v in runs.items()}).T[["map@12", "recall@12", "ndcg@12"]]
abl["delta vs full"] = abl["map@12"] - abl.loc["full", "map@12"]
abl.round(4)"""),
    code("""
abl["delta vs full"].drop("full").sort_values().plot(kind="barh", figsize=(8, 3), title="MAP@12 change when a feature group is removed"); plt.show()"""),
    md("""Differences between runs are small next to the sampling noise of a few thousand customers, so
test each against the full model with a **paired bootstrap** (an interval containing 0 is not evidence
of anything):"""),
    code("""
from hmrec.evaluate import paired_bootstrap
full = runs["full"][3].per_user["map@12"]
pd.DataFrame({k: paired_bootstrap(full, v[3].per_user["map@12"]) for k, v in runs.items() if k != "full"}).T.round(5)"""),
    md("## What the model uses"),
    code("""
rk = runs["full"][0]
imp = pd.Series(rk.booster.feature_importance("gain"), index=rk.features).sort_values()
imp.tail(15).plot(kind="barh", figsize=(8, 4), title="top features by gain"); plt.show()"""),
    md("""**Reading:** compare the full ranker to the best baseline from notebook 2 using a paired bootstrap
before claiming an improvement - with a few thousand evaluation customers, small gaps can be noise."""),
]

NOTEBOOKS["04_diversity_and_mmr.ipynb"] = [
    md("""# 4. Beyond accuracy: diversity, novelty, popularity bias
Accuracy is not the only goal. This notebook measures what the ranker recommends (coverage,
novelty, concentration on popular articles) and what MMR re-ranking trades away for variety."""),
    code(SETUP),
    code("""
from hmrec import models as Mo, candidates as C, ranker as R
from hmrec.rerank import mmr_rerank
def sources(ctx):
    return {"repurchase": C.Source(Mo.Repurchase(28).fit(ctx), 40), "popularity": C.Source(Mo.Popularity(7).fit(ctx), 30),
            "co_purchase": C.Source(Mo.CoPurchase(days=56).fit(ctx), 50), "als": C.Source(Mo.ALS(factors=64, days=26 * 7).fit(ctx), 50)}
pool_val = C.generate(sources(W.ctx_train), W.ev_val.users, W.ev_val.rel)
pool_test = C.generate(sources(W.ctx_fit), W.ev_test.users, W.ev_test.rel)
fv = R.FeatureBuilder(hm, W.ctx_train, W.emb).featurize(pool_val.frame)
ft = R.FeatureBuilder(hm, W.ctx_fit, W.emb).featurize(pool_test.frame)
feats = [c for c in fv.columns if c not in ("user", "item", "label")]
rk = R.train_ranker(fv, feats, seed=0)
ranked, scores = R.rank_pool(rk, ft, W.ev_test.users)
base = W.ev_test.metrics_from_ranking(ranked, "ranker")
pd.Series({k: v for k, v in base.summary.items() if k in ("map@12","coverage","novelty","gini","head_share","tail_share","tag_ild","content_ild","arp")}).round(4).to_frame("ranker")"""),
    md(
        "## MMR trade-off\n`lambda=1` is the plain ranker; smaller values favour variety (cosine distance of description embeddings)."
    ),
    code("""
rows = {}
for lam in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
    r = W.ev_test.metrics_from_ranking(mmr_rerank(ranked, scores, W.emb, 12, lam), f"mmr {lam}")
    rows[lam] = {k: r.summary[k] for k in ("map@12", "recall@12", "coverage", "novelty", "head_share", "tag_ild", "content_ild")}
sweep = pd.DataFrame(rows).T; sweep.index.name = "lambda"; sweep.round(4)"""),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(sweep["content_ild"], sweep["map@12"], "o-"); ax[0].set_xlabel("intra-list diversity (embedding)"); ax[0].set_ylabel("MAP@12")
for lam, r in sweep.iterrows(): ax[0].annotate(f"{lam}", (r["content_ild"], r["map@12"]))
ax[1].plot(sweep["coverage"], sweep["map@12"], "o-", color="C1"); ax[1].set_xlabel("catalogue coverage"); plt.tight_layout(); plt.show()"""),
    md("""**Reading:** find the knee where diversity rises while MAP@12 barely moves. Note the diversity
metric uses the same embeddings MMR optimises, so `tag_ild` (category-based) is the independent check."""),
]


def build(execute: bool = False) -> None:
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python (hmrec)",
            "language": "python",
            "name": "hmrec",
        }
        path = HERE / name
        if execute:
            from nbclient import NotebookClient

            NotebookClient(
                nb,
                timeout=3600,
                kernel_name="hmrec",
                resources={"metadata": {"path": str(HERE.parent)}},
            ).execute()
        nbf.write(nb, path)
        print("wrote", path.name, "(executed)" if execute else "")


if __name__ == "__main__":
    build("--execute" in sys.argv)
