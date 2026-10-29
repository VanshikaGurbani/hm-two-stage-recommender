# H&M personalized recommendations: two-stage recommender

Candidate generation (repurchase, trending, age-group popularity, co-purchase, implicit ALS,
two-tower + FAISS) followed by a LightGBM LambdaRank ranker, evaluated on a strict week-based
temporal split of the Kaggle *H&M Personalized Fashion Recommendations* purchases. Includes
analysis notebooks, a Streamlit demo, a FastAPI service with measured latency, and tests/CI.

> **Status:** the code and tests are complete; **real-data results have not been generated yet**.
> The results block below is filled in automatically by `python -m hmrec.run` and is empty until
> that has run on the real data. No number in this README is typed by hand.

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
_Not generated yet. Run `python -m hmrec.run` after the data is downloaded._
<!-- RESULTS:END -->

## Limitations (read before quoting any number)

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
