"""One command that produces every number in the README: ``python -m hmrec.run``.

Steps: load + split -> tune and evaluate baselines -> candidate pool and recall -> LightGBM ranker
with feature-group ablation -> beyond-accuracy metrics and the MMR trade-off. Writes
``results/results.json`` and ``docs/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import candidates as C
from . import data as D
from . import embed
from . import models as Mo
from . import ranker as R
from . import splits as S
from .evaluate import Evaluator, paired_bootstrap
from .rerank import mmr_rerank
from .twotower import TwoTower


def _pick(
    summary: dict,
    keys=(
        "map@12",
        "recall@12",
        "ndcg@12",
        "recall@100",
        "coverage",
        "novelty",
        "gini",
        "head_share",
        "tag_ild",
        "arp",
    ),
) -> dict:
    return {k: round(float(summary[k]), 5) for k in keys if k in summary}


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_dir)
    if args.fixture:
        D.make_fixture(root, n_customers=600, n_articles=120, n_weeks=16, seed=0)
    hm = D.load(root, sample_frac=args.sample_frac, seed=args.seed)
    split = S.weekly(hm.interactions, history_weeks=args.history_weeks)
    S.check_no_leakage(split)
    emb = embed.item_embeddings(hm, args.encoder, cache_dir=root / "cache")
    res: dict = {
        "config": {**vars(args), "python": platform.python_version()},
        "data": D.describe(hm),
        "split": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
    }
    ev_val = Evaluator(hm, split.train, split.val, content_vectors=emb)
    ctx_tr = Mo.FitContext(hm, split.train, args.seed, {"text_embeddings": emb})
    hist = split.fit_frame
    ev_test = Evaluator(hm, hist, split.test, content_vectors=emb)
    ctx_fit = Mo.FitContext(hm, hist, args.seed, {"text_embeddings": emb})
    res["eval_customers"] = {"val": len(ev_val.users), "test": len(ev_test.users)}
    print(res["data"], res["eval_customers"], flush=True)

    # ---------------------------------------------------------------- baselines
    grids: dict[str, list] = {
        "popularity": [lambda d=d: Mo.Popularity(d) for d in (7, 14, 28)],
        "repurchase": [lambda h=h: Mo.Repurchase(h) for h in (7, 14, 28, 56)],
        "age-popularity": [lambda d=d: Mo.AgePopularity(d) for d in (7, 14)],
        "co-purchase": [lambda d=d: Mo.CoPurchase(days=d) for d in (28, 56)],
        "als": [
            lambda f=f: Mo.ALS(factors=f, days=args.history_weeks * 7 or 180) for f in (32, 64)
        ],
    }
    tuning: dict = {}
    chosen: dict = {}
    test_results: dict = {}
    for family, factories in grids.items():
        rows = []
        for fac in factories:
            m = fac().fit(ctx_tr)
            rows.append((ev_val.evaluate(m).summary["map@12"], fac, m.name))
        tuning[family] = {name: round(v, 5) for v, _, name in rows}
        best = max(rows, key=lambda r: r[0])
        chosen[family] = best[1]
        test_results[family] = ev_test.evaluate(best[1]().fit(ctx_fit), family)
        print(family, "->", best[2], round(test_results[family].summary["map@12"], 5), flush=True)
    res["baseline_tuning_val_map12"] = tuning
    res["baselines_test"] = {
        k: {**_pick(v.summary), "ci_map@12": [round(x, 5) for x in v.ci["map@12"]]}
        for k, v in test_results.items()
    }

    # ---------------------------------------------------------------- candidate pool
    def sources(ctx):
        return {
            "repurchase": C.Source(chosen["repurchase"]().fit(ctx), args.n_repurchase),
            "popularity": C.Source(chosen["popularity"]().fit(ctx), args.n_popular),
            "age_pop": C.Source(chosen["age-popularity"]().fit(ctx), args.n_popular),
            "co_purchase": C.Source(chosen["co-purchase"]().fit(ctx), args.n_neighbors),
            "als": C.Source(chosen["als"]().fit(ctx), args.n_neighbors),
            "two_tower": C.Source(
                TwoTower(epochs=args.tt_epochs, days=args.history_weeks * 7).fit(ctx),
                args.n_neighbors,
            ),
        }

    pool_val = C.generate(sources(ctx_tr), ev_val.users, ev_val.rel)
    pool_test = C.generate(sources(ctx_fit), ev_test.users, ev_test.rel)
    res["candidates"] = {
        "val_recall": {k: round(v, 4) for k, v in pool_val.recall.items()},
        "test_recall": {k: round(v, 4) for k, v in pool_test.recall.items()},
        "avg_candidates_per_customer": round(pool_test.avg_candidates, 1),
    }
    print(res["candidates"], flush=True)

    # ---------------------------------------------------------------- ranker + ablation
    frame_val = R.FeatureBuilder(hm, ctx_tr, emb).featurize(pool_val.frame)
    frame_test = R.FeatureBuilder(hm, ctx_fit, emb).featurize(pool_test.frame)
    all_feats = [c for c in frame_val.columns if c not in ("user", "item", "label")]
    groups = R.feature_groups(all_feats)
    variants = {"full": all_feats}
    for g in R.GROUPS:
        variants[f"without {g}"] = [c for c in all_feats if c not in groups[g]]
    variants["source scores only"] = groups["source"]
    ablation, rankings = {}, {}
    for name, feats in variants.items():
        rk = R.train_ranker(frame_val, feats, seed=args.seed, params={"num_threads": args.threads})
        ranked, scores = R.rank_pool(rk, frame_test, ev_test.users)
        r = ev_test.metrics_from_ranking(ranked, name)
        ablation[name] = {
            **_pick(r.summary),
            "n_features": len(feats),
            "ci_map@12": [round(x, 5) for x in r.ci["map@12"]],
            "trees": rk.best_iteration,
        }
        rankings[name] = (ranked, scores, r)
        print(name, ablation[name]["map@12"], flush=True)
    best_base = max(test_results.values(), key=lambda r: r.summary["map@12"])
    full = rankings["full"][2]
    res["ranker"] = {
        "ablation": ablation,
        "full_vs_best_baseline": {
            "baseline": best_base.model,
            **paired_bootstrap(full.per_user["map@12"], best_base.per_user["map@12"]),
        },
        "full_vs_ablated": {
            n: paired_bootstrap(full.per_user["map@12"], rankings[n][2].per_user["map@12"])
            for n in variants
            if n != "full"
        },
    }

    # ---------------------------------------------------------------- MMR trade-off
    ranked, scores, _ = rankings["full"]
    sweep = {}
    for lam in (1.0, 0.9, 0.8, 0.7, 0.5):
        rr = mmr_rerank(ranked, scores, emb, 12, lam)
        wide = np.pad(rr, ((0, 0), (0, 0)))
        sweep[str(lam)] = _pick(ev_test.metrics_from_ranking(wide, f"mmr {lam}").summary)
    res["mmr_tradeoff"] = sweep
    res["runtime_sec"] = round(time.time() - t0)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, Path(args.docs) / "RESULTS.md")
    update_readme(res)
    return res


def update_readme(res: dict, readme: Path = Path("README.md")) -> None:
    """Replace the block between the RESULTS markers with the freshly measured tables."""
    if not readme.exists() or res["config"].get("fixture"):
        return
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return
    block = (Path(res["config"]["docs"]) / "RESULTS.md").read_text(encoding="utf-8")
    block = block.replace("# Results (generated by `python -m hmrec.run`)", "")
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    new = "\n".join([f"{head}{start}", block.strip(), f"{end}{tail}"])
    readme.write_text(new, encoding="utf-8")


def write_markdown(res: dict, path: Path) -> None:
    def table(rows: dict, cols: list[str]) -> str:
        head = "| model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        body = ""
        for name, r in rows.items():
            body += f"| {name} | " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
        return head + body

    cols = ["map@12", "recall@12", "recall@100", "coverage", "novelty", "head_share"]
    base = {
        k: {**v, "map@12": f"{v['map@12']} {v['ci_map@12']}"}
        for k, v in res["baselines_test"].items()
    }
    abl = {
        k: {**v, "map@12": f"{v['map@12']} {v['ci_map@12']}"}
        for k, v in res["ranker"]["ablation"].items()
    }
    text = (
        "# Results (generated by `python -m hmrec.run`)\n\n"
        f"Data: {res['data']}\n\nEvaluation customers: {res['eval_customers']}\n\n"
        "## Baselines (test week)\n\n"
        + table(base, cols)
        + "\n## Candidate recall\n\n```\n"
        + json.dumps(res["candidates"], indent=1)
        + "\n```\n"
        "\n## Ranker and feature-group ablation (test week)\n\n"
        + table(abl, cols + ["n_features"])
        + "\n## MMR trade-off\n\n"
        + table(res["mmr_tradeoff"], ["map@12", "coverage", "novelty", "tag_ild", "head_share"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m hmrec.run")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history-weeks", type=int, default=26)
    ap.add_argument("--encoder", default="minilm", choices=["minilm", "tfidf"])
    ap.add_argument("--n-repurchase", type=int, default=40)
    ap.add_argument("--n-popular", type=int, default=30)
    ap.add_argument("--n-neighbors", type=int, default=50)
    ap.add_argument("--tt-epochs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default="results")
    ap.add_argument("--docs", default="docs")
    ap.add_argument(
        "--fixture", action="store_true", help="run on a synthetic dataset (smoke test)"
    )
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
