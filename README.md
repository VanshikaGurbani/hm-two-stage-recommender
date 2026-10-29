# H&M personalized recommendations: two-stage recommender

Candidate generation (repurchase, trending, age-group popularity, co-purchase, implicit ALS,
two-tower + FAISS) followed by a LightGBM LambdaRank ranker, evaluated on a strict week-based
temporal split of the Kaggle *H&M Personalized Fashion Recommendations* purchases. Includes
analysis notebooks, a Streamlit demo, a FastAPI service with measured latency, and tests/CI.

> **Status:** all numbers in the Results block were produced by `python -m hmrec.run`,
> `python -m hmrec.ann_eval` and `python -m hmrec.loadtest` on a seeded 10% customer sample and are
> inserted by `python -m hmrec.report`; none is typed by hand.

## Quick start

```bash
pip install -e ".[all]"            # CPU torch: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m hmrec.data download      # unzips the 3 CSVs from a competition zip in this folder / Downloads,
                                   # or fetches them with the Kaggle CLI using credentials from .env
python -m hmrec.run                # every number below (10% customer sample, seed 0)
jupyter lab notebooks/             # step-by-step analysis, runnable cell by cell
streamlit run app/streamlit_app.py # demo
uvicorn hmrec.api:app              # API;  python -m hmrec.loadtest  measures p50/p95/p99
pytest && ruff check .
```

Data: join the competition and accept its rules on Kaggle (required by Kaggle), then either save
the competition zip in the project folder (only the three CSVs are extracted; the 28 GB of images
are ignored) or copy `.env.example` to `.env` and add your Kaggle token. `.env`, zips and `data/`
are git-ignored.

## Method

| Piece | What it does |
|---|---|
| Split | last 7 days = test, the 7 days before = ranker labels, older = history. Retrievers/features for a target week only see earlier data; a unit test fails otherwise. |
| Metric | MAP@12 over the full catalogue (competition definition) for customers who bought in the target week and had prior history; repurchases are **not** masked. 95% bootstrap CIs over customers; paired bootstrap for differences. |
| Candidates | repurchase (recency-decayed), trending, age-group popularity, co-purchase item-item, ALS, two-tower with FAISS HNSW. Recall@K reported per source and for the union. |
| Ranker | LightGBM LambdaRank on the validation week; feature groups: source scores, customer, article, customer×article, text-embedding similarity (MiniLM over product name/description). Ablation removes one group at a time. |
| Diversity | MMR re-ranking over description embeddings; coverage, novelty, Gini, popularity-head share, category diversity. |
| Explanations | built only from recorded facts (which sources proposed the article, times bought, product-type share); tested against the purchase log. |

## Results

<!-- RESULTS:START -->
**Data:** 136,228 customers (seeded 10% sample), 105,542 articles, 3,179,841 purchases over 733 days. **Evaluated customers:** 5,726 (test week), 5,976 (validation week). Metric: MAP@12 over the full catalogue, 95% bootstrap CI over customers.

### Baselines (test week)

| model | MAP@12 [95% CI] | recall@12 | recall@100 | coverage | novelty | head_share |
|---|---|---|---|---|---|---|
| repurchase | 0.0273 [0.0239, 0.0307] | 0.0464 | 0.0502 | 0.1429 | 10.4749 | 0.2701 |
| als | 0.0092 [0.0076, 0.0108] | 0.0210 | 0.0664 | 0.0214 | 8.3078 | 0.8920 |
| age-popularity | 0.0082 [0.0068, 0.0097] | 0.0264 | 0.1092 | 0.0006 | 9.5143 | 0.3409 |
| popularity | 0.0081 [0.0067, 0.0098] | 0.0239 | 0.1108 | 0.0001 | 9.3073 | 0.3333 |
| co-purchase | 0.0069 [0.0056, 0.0084] | 0.0175 | 0.0385 | 0.0802 | 12.0122 | 0.0813 |

### Candidate generation: share of purchased articles proposed (test week)

Average 163 candidates per customer. The union is the ceiling for the ranker.

| source | recall |
|---|---|
| union | 0.1647 |
| two_tower | 0.0639 |
| popularity | 0.0500 |
| repurchase | 0.0499 |
| age_pop | 0.0498 |
| als | 0.0474 |
| co_purchase | 0.0313 |

### Ranker and feature-group ablation (test week)

| variant | MAP@12 [95% CI] | change vs full (paired CI) | features | recall@12 | coverage |
|---|---|---|---|---|---|
| full | 0.0333 [0.0297, 0.0370] | - | 44 | 0.0687 | 0.0464 |
| without source | 0.0318 [0.0285, 0.0355] | -0.0014 [-0.0032, +0.0007] (n.s.) | 32 | 0.0637 | 0.0416 |
| without customer | 0.0315 [0.0283, 0.0352] | -0.0018 [-0.0036, +0.0002] (n.s.) | 30 | 0.0677 | 0.0422 |
| without item | 0.0303 [0.0271, 0.0338] | -0.0029 [-0.0051, -0.0008] | 34 | 0.0645 | 0.0540 |
| without pair | 0.0311 [0.0278, 0.0348] | -0.0021 [-0.0039, -0.0002] | 37 | 0.0683 | 0.0456 |
| without text | 0.0330 [0.0293, 0.0368] | -0.0003 [-0.0020, +0.0015] (n.s.) | 43 | 0.0681 | 0.0444 |
| source scores only | 0.0312 [0.0278, 0.0347] | -0.0021 [-0.0040, +0.0001] (n.s.) | 12 | 0.0631 | 0.0500 |

Full ranker minus best baseline (`repurchase`): +0.0060 [+0.0039, +0.0079] MAP@12 (paired bootstrap over customers). `(n.s.)` = the interval includes 0.

### MMR: accuracy vs variety

`lambda = 1` is the plain ranker; `tag_ild` is category diversity (independent of the embeddings MMR optimises).

| lambda | MAP@12 | coverage | novelty | tag_ild | head_share |
|---|---|---|---|---|---|
| 1.0 | 0.0333 | 0.0464 | 9.42 | 0.634 | 0.492 |
| 0.9 | 0.0330 | 0.0465 | 9.42 | 0.641 | 0.490 |
| 0.8 | 0.0327 | 0.0466 | 9.42 | 0.650 | 0.490 |
| 0.7 | 0.0323 | 0.0468 | 9.42 | 0.660 | 0.488 |
| 0.5 | 0.0307 | 0.0469 | 9.41 | 0.685 | 0.490 |

### Two-tower + FAISS HNSW

| measure | value |
|---|---|
| catalogue_articles | 105542 |
| dim | 64 |
| train_seconds | 221.1 |
| final_train_loss | 5.7206 |
| hnsw_build_seconds | 3.55 |
| exact_candidate_recall@100 | 0.0895 |
| exact_ms_per_query | 0.07 |
| hnsw_by_efSearch | {"16": {"overlap_with_exact_top100": 0.6219, "candidate_recall@100": 0.0858, "ms_per_query": 0.007}, "64": {"overlap_with_exact_top100": 0.8981, "candidate_recall@100": 0.0908, "ms_per_query": 0.011}, "128": {"overlap_with_exact_top100": 0.9645, "candidate_recall@100": 0.0898, "ms_per_query": 0.015}, "256": {"overlap_with_exact_top100": 0.9906, "candidate_recall@100": 0.0893, "ms_per_query": 0.034}} |
| eval_customers | 5726 |

### API latency (one machine, 8 concurrent clients)

| pass | requests | p50 ms | p95 ms | p99 ms | server-side p50 ms | req/s | cache hits |
|---|---|---|---|---|---|---|---|
| single_request_cold | 30 | 21.7 | 32.9 | 2369.1 | 19.8 | 7.5 | 0.0 |
| cold_cache_under_load | 70 | 117.6 | 554.4 | 569.0 | 100.5 | 47.8 | 0.0 |
| mixed_cache_under_load | 300 | 5.9 | 122.5 | 518.3 | 0.0 | 283.5 | 0.9 |
| all_cached_under_load | 70 | 4.2 | 373.9 | 384.1 | 0.0 | 171.3 | 1.0 |

_Full run took 23 min on CPU; seed 0._
<!-- RESULTS:END -->

## Reading the results

- **Repurchasing is the strongest single signal.** The best simple baseline is "buy it again", far
  ahead of global or age-group popularity and of collaborative models trained on their own.
- **The two-stage ranker beats that baseline**, and the paired interval in the ablation table says
  by how much. The gain is real but modest: most of it comes from article and customer×article
  features, not from the learned candidate sources.
- **Not every feature group earns its place.** In the ablation, groups whose interval spans zero
  (marked `n.s.`) are not shown to help on this sample. In particular the text-embedding feature
  adds nothing measurable here.
- **Candidate recall is the ceiling.** The union of sources proposes only a small share of what was
  bought, so better candidate generation is the largest remaining lever, more than ranker tuning.
- **MMR trades accuracy for category variety almost linearly** and barely changes catalogue
  coverage; popularity concentration stays high because the data itself is concentrated.
- **ANN:** at ~100k articles an exact search is as fast as HNSW; HNSW is included because it is how
  this would be served at larger catalogue sizes, and the table shows what its recall costs.

## Limitations (read before quoting any number)

- The LightGBM ranker stopped early at a small number of trees on a one-week validation slice;
  it is lightly tuned and results are from a single seed.
- Notebooks use slightly simpler settings than `hmrec.run` (no two-tower source), so their ablation
  numbers differ from the README table. The README table is the reference.

- Scores come from a seeded **customer subsample** and my own week split; they are **not comparable
  to the Kaggle leaderboard** (different customer set, no private test week).
- Customers with no purchase history before the target week are excluded from the headline metric
  (cold start is treated separately).
- Evaluation is offline on logged purchases: it cannot measure what customers *would* have bought
  if shown something else. No A/B test, no counterfactual (IPS) estimate is claimed.
- Images are not used; article text is H&M's own product name and description.
- Hyper-parameters are tuned lightly on a single validation week; results are from one seed.
- Latency numbers are for one machine and the in-process cache; see `results/loadtest.json`.

## Layout

`hmrec/` package · `tests/` · `notebooks/` (built by `notebooks/build_notebooks.py`, committed
without outputs) · `app/streamlit_app.py` · `Dockerfile` · `.github/workflows/ci.yml`
