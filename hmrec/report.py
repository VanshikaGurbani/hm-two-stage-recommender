"""Turns ``results/*.json`` into ``docs/RESULTS.md`` and the README results block.

Nothing here is typed by hand: every cell comes from a JSON file written by ``hmrec.run``,
``hmrec.ann_eval`` or ``hmrec.loadtest``. ``python -m hmrec.report`` regenerates the tables
without re-running any experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _ci(v: dict, key: str = "map@12") -> str:
    lo, hi = v["ci_" + key]
    return f"{v[key]:.4f} [{lo:.4f}, {hi:.4f}]"


def _table(rows: list[list[str]], header: list[str]) -> str:
    line = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return line + "".join("| " + " | ".join(r) + " |\n" for r in rows)


def build(results_dir: Path = Path("results")) -> str:
    res = json.loads((results_dir / "results.json").read_text())
    out = []
    d = res["data"]
    out.append(
        f"**Data:** {d['customers']:,} customers "
        f"(seeded {res['config']['sample_frac']:.0%} sample), "
        f"{d['articles_in_catalogue']:,} articles, {d['purchases']:,} purchases over "
        f"{d['span_days']:.0f} days. **Evaluated customers:** {res['eval_customers']['test']:,} "
        f"(test week), {res['eval_customers']['val']:,} (validation week). "
        f"Metric: MAP@12 over the full catalogue, 95% bootstrap CI over customers.\n"
    )
    cols = ["recall@12", "recall@100", "coverage", "novelty", "head_share"]
    rows = [
        [name, _ci(v)] + [f"{v[c]:.4f}" for c in cols]
        for name, v in sorted(res["baselines_test"].items(), key=lambda kv: -kv[1]["map@12"])
    ]
    out.append("### Baselines (test week)\n\n" + _table(rows, ["model", "MAP@12 [95% CI]", *cols]))

    rec = res["candidates"]["test_recall"]
    rows = [[k, f"{v:.4f}"] for k, v in sorted(rec.items(), key=lambda kv: -kv[1])]
    out.append(
        f"### Candidate generation: share of purchased articles proposed (test week)\n\n"
        f"Average {res['candidates']['avg_candidates_per_customer']:.0f} candidates per customer. "
        "The union is the ceiling for the ranker.\n\n" + _table(rows, ["source", "recall"])
    )

    abl = res["ranker"]["ablation"]
    paired = res["ranker"]["full_vs_ablated"]
    rows = []
    for name, v in abl.items():
        p = paired.get(name)
        delta = (
            "-"
            if p is None
            else f"{-p['diff']:+.4f} [{-p['hi']:+.4f}, {-p['lo']:+.4f}]"
            + ("" if p["significant"] else " (n.s.)")
        )
        rows.append(
            [
                name,
                _ci(v),
                delta,
                str(v["n_features"]),
                f"{v['recall@12']:.4f}",
                f"{v['coverage']:.4f}",
            ]
        )
    b = res["ranker"]["full_vs_best_baseline"]
    out.append(
        "### Ranker and feature-group ablation (test week)\n\n"
        + _table(
            rows,
            [
                "variant",
                "MAP@12 [95% CI]",
                "change vs full (paired CI)",
                "features",
                "recall@12",
                "coverage",
            ],
        )
        + f"\nFull ranker minus best baseline (`{b['baseline']}`): {b['diff']:+.4f} "
        f"[{b['lo']:+.4f}, {b['hi']:+.4f}] MAP@12 (paired bootstrap over customers). "
        "`(n.s.)` = the interval includes 0.\n"
    )

    rows = [
        [
            lam,
            f"{v['map@12']:.4f}",
            f"{v['coverage']:.4f}",
            f"{v['novelty']:.2f}",
            f"{v['tag_ild']:.3f}",
            f"{v['head_share']:.3f}",
        ]
        for lam, v in res["mmr_tradeoff"].items()
    ]
    out.append(
        "### MMR: accuracy vs variety\n\n`lambda = 1` is the plain ranker; `tag_ild` is category "
        "diversity (independent of the embeddings MMR optimises).\n\n"
        + _table(rows, ["lambda", "MAP@12", "coverage", "novelty", "tag_ild", "head_share"])
    )

    ann = results_dir / "ann.json"
    if ann.exists():
        a = json.loads(ann.read_text())
        rows = [[k, str(v)] for k, v in a.items()]
        out.append("### Two-tower + FAISS HNSW\n\n" + _table(rows, ["measure", "value"]))
    lt = results_dir / "loadtest.json"
    if lt.exists():
        t = json.loads(lt.read_text())
        rows = [
            [
                k,
                str(v["n"]),
                f"{v['p50_ms']}",
                f"{v['p95_ms']}",
                f"{v['p99_ms']}",
                f"{v['server_p50_ms']}",
                f"{v['throughput_rps']}",
                f"{v['cache_hit_share']}",
            ]
            for k, v in t.items()
            if isinstance(v, dict) and "p50_ms" in v
        ]
        out.append(
            f"### API latency (one machine, {t['concurrency']} concurrent clients)\n\n"
            + _table(
                rows,
                [
                    "pass",
                    "requests",
                    "p50 ms",
                    "p95 ms",
                    "p99 ms",
                    "server-side p50 ms",
                    "req/s",
                    "cache hits",
                ],
            )
        )
    out.append(
        f"_Full run took {res['runtime_sec'] / 60:.0f} min on CPU; seed {res['config']['seed']}._\n"
    )
    return "\n".join(out)


def write(
    results_dir: Path = Path("results"), docs: Path = Path("docs"), readme: Path = Path("README.md")
) -> None:
    body = build(results_dir)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RESULTS.md").write_text(
        "# Results (generated by `python -m hmrec.report`)\n\n" + body, encoding="utf-8"
    )
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if START in text and END in text:
            head, rest = text.split(START, 1)
            tail = rest.split(END, 1)[1]
            readme.write_text(
                "\n".join([f"{head}{START}", body.strip(), f"{END}{tail}"]), encoding="utf-8"
            )


if __name__ == "__main__":
    write()
