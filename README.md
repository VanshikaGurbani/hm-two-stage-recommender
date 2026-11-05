# Fashion Recommendation System

*A fashion recommender built on H&M purchase data*

### 👉 [Try the live app: fashion-recommendersystem.streamlit.app](https://fashion-recommendersystem.streamlit.app/)

## What is this?

You know the "You might also like..." row on every online clothing shop? Someone (or rather
something) has to decide which handful of items, out of more than 100,000, show up in that row for
*you*. This project builds that something from scratch, and then checks honestly whether it works.

It learns from **31.8 million real H&M purchases** (105,000 products, 1.37 million customers) and
tries to guess the **12 items each shopper will buy in the coming week**. Think of a shop assistant
who has quietly watched a million shoppers and can say, "this person will probably want another
pair of those black trousers, and maybe that jacket everyone is buying this week."

The fun part is the test. We hide the most recent week, ask the system for its guesses, then
reveal what people really bought and see how many guesses were right.

**Did it work?** Yes, modestly but measurably. The hardest rival is a very simple rule: "show
people what they already bought," because shoppers re-buy a lot. The system beats that rule, and
the gap is large enough that it is very unlikely to be luck. The exact numbers are just below.

It is an independent portfolio project, not made by or affiliated with H&M.

## The result in one paragraph

<!-- HEADLINE:START -->
On 5,726 real shoppers, the system scored **0.033** (higher is better) against **0.027** for the best simple rule, 'suggest what they bought before'. The improvement is statistically reliable: with 95% confidence it lies between +0.004 and +0.008. The shortlist step finds about 16% of what shoppers go on to buy. A single recommendation takes about 22 milliseconds.
<!-- HEADLINE:END -->

## How it thinks

Picture an online shop with over 100,000 products. Checking every product for every shopper would
be slow, so the system works in two steps, like a clever shop assistant would:

1. **Make a shortlist.** Several quick "scouts" each suggest items: things the shopper bought
   before, what is trending this week, what people of the same age buy, items often bought
   together, items liked by shoppers with similar taste, and a small neural network that
   matches the style of recent purchases. Together they produce about 160 candidates.
2. **Sort the shortlist.** A learned ranking model looks at 44 clues (how often the shopper buys
   this kind of product, how popular the item is, its price compared with what they usually
   spend, how recently they shopped, and so on) and puts the likeliest items first.

Every suggestion comes with a one-line reason in everyday language, for example "you bought it
2 times, last 14 days ago" or "trending this week". The reasons are built from facts in the
purchase records, never invented. A "variety" slider lets you trade a little accuracy for a more
varied list.

## How it was tested

The test is a replay of real life. The system is only shown purchases up to a certain day, and we
then check whether its 12 suggestions match what shoppers really bought in the following 7 days.
Nothing from the future can leak in, and automated tests check that.

The score used is the one from the original H&M competition, **MAP@12**. It runs from 0 to 1 and
is higher when the products a shopper really bought sit nearer the top of the 12 suggestions.
Scores on this kind of data are small, because shoppers buy only a few of 100,000 products in
a week. What matters is the comparison with simple alternatives, shown below.

## Try it

**Live demo:** [fashion-recommendersystem.streamlit.app](https://fashion-recommendersystem.streamlit.app/).
The online demo uses made-up sample shoppers, because H&M's real data is not allowed to be shared
publicly. Its charts and scores come from measurements on the real data.

**Using the demo (no technical knowledge needed).** The app opens on a **Start here** tab that
explains everything in plain words. In short:

| Tab | What you can do there |
|---|---|
| Start here | read what the app is, how to use it and what to try |
| Recommendations | pick a shopper (left panel) and see the 12 suggestions, each with a reason |
| Purchase history | see what that shopper bought before, to judge if the suggestions fit |
| Data analysis | a guided tour of the purchase data in simple charts (busy weeks, how much people buy, favourite products, ages, prices) |
| How good is it? | the scores against simpler ideas, which clues helped, and the variety trade-off, each chart with a sentence explaining how to read it |

Things to try: press *Pick a random shopper* a few times, and drag the *Variety* slider to the
left to see the list become less repetitive.

**Run it on your own computer** (Windows, Mac or Linux, with Python 3.11 or newer). Use a
virtual environment, which keeps this project's libraries separate from your other projects.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Mac or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process Bypass` once
in that window and try again. After activation the prompt starts with `(.venv)`; repeat the
activation step in any new terminal before running the app.

A browser tab opens at http://localhost:8501. Pick a customer number on the left, move the
"variety" slider, and read why each product was suggested. Without the real data it starts
instantly on made-up shoppers (a yellow notice says so). To see the real thing, download the
H&M competition data from Kaggle, put the competition zip in this folder, run
`python -m hmrec.data download` once, and start the app again. The first start then takes a few
minutes while it prepares everything. Product photos are shown when the `images` folder is
present.

## What is in this project

| Folder or file | What it is |
|---|---|
| `hmrec/` | the recommender itself (data loading, scouts, ranking model, scoring, API) |
| `app/streamlit_app.py` | the interactive demo (guide, recommendations, history, data analysis, results) |
| `notebooks/` | four step-by-step notebooks with charts: data exploration, simple baselines, the two-step system, and variety versus accuracy |
| `tests/` | 48 automated checks, including "no peeking at the future" |
| `Dockerfile`, `.github/workflows/ci.yml` | packaging and automatic checks on every change |

Everything below the line is detail for engineers. All numbers are produced by the project's own
code (`python -m hmrec.run`) and inserted automatically; none is typed by hand.

---

## Things to keep in mind

- Results come from a random 10% of customers (136,228 people), so the whole experiment runs on an ordinary
  computer in about 25 minutes. They are not comparable with the public Kaggle leaderboard, which uses a
  different customer set and a hidden test week.
- The test is a replay of past purchases. It shows how well the system predicts what shoppers
  bought; it cannot show what they would have bought if shown something else.
- Shoppers with no earlier purchases are left out of the headline score (they need a different
  approach, since there is nothing to learn from).
- Product photos are used for display only; the models use the product names and descriptions.
- The ranking model was lightly tuned on one validation week and trained with one random seed.

## What the experiments showed

- Suggesting what a shopper bought before is the strongest single rule, far ahead of "most
  popular right now" or "popular for your age".
- The two-step system beats that rule, and the gain comes mostly from clues about the product
  and about the shopper-product pair.
- Some clue groups did not help measurably on this sample. In particular, adding a text-understanding model
  that reads product descriptions added nothing detectable.
- The shortlist only contains a minority of what shoppers buy, so better scouts are the biggest
  opportunity for improvement.
- Making the list more varied lowers accuracy gradually; the table below shows how much.

## Technical details

Two-stage retrieval and ranking: six candidate sources (recency-weighted repurchase, trending,
age-group popularity, item-item co-purchase, implicit ALS, two-tower served through FAISS HNSW)
feed a LightGBM LambdaRank model trained on the validation week. Split: last 7 days = test, the 7
days before = ranker labels, older = history. MAP@12 over the full catalogue with 95% bootstrap
confidence intervals over customers; paired bootstrap for differences.

```bash
pip install -e ".[all]"            # CPU PyTorch: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m hmrec.data download      # extracts the 3 CSVs from the competition zip, or uses the Kaggle CLI with .env credentials
python -m hmrec.run                # all experiments (10% sample, seed 0), regenerates the tables below
python -m hmrec.eda                # data-analysis numbers for the app's Data analysis tab (aggregates only)
python -m hmrec.ann_eval           # two-tower + FAISS measurements
python -m hmrec.loadtest           # API latency
python -m hmrec.report             # rebuilds docs/RESULTS.md and this README's tables
jupyter lab notebooks/             # notebooks
uvicorn hmrec.api:app              # API (/recommend, /image/{article_id}, /health)
pytest && ruff check . && mypy hmrec --ignore-missing-imports
```

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

Data access: join the competition on Kaggle and accept its rules, then either save the
competition zip in the project folder (only the three CSVs are read) or copy `.env.example` to
`.env` and add a Kaggle token. `.env`, zips and `data/` are git-ignored.
