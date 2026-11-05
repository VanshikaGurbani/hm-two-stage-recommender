"""Streamlit demo: ``streamlit run app/streamlit_app.py``.

Tabs: Start here (plain-language guide), Recommendations, Purchase history, Data analysis and
How good is it? Every number on the analysis tabs comes from JSON files written by the project's
own code (``python -m hmrec.run`` and ``python -m hmrec.eda``), never typed in by hand.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from hmrec import eda, report, workspace
from hmrec.images import ImageStore
from hmrec.pipeline import System

TITLE = "Fashion Recommendation System"
SUBTITLE = "A fashion recommender built on H&M purchase data"

st.set_page_config(page_title=TITLE, layout="wide")

BASELINE_NAMES = {
    "repurchase": "Re-suggest what they bought before",
    "als": "Shoppers with similar taste",
    "age-popularity": "Popular with their age group",
    "popularity": "Most popular overall",
    "co-purchase": "Often bought together",
}
SCOUT_NAMES = {
    "union": "All scouts combined",
    "two_tower": "Style-matching neural network",
    "popularity": "Trending this week",
    "repurchase": "Bought before",
    "age_pop": "Popular with their age group",
    "als": "Shoppers with similar taste",
    "co_purchase": "Often bought together",
}
CLUE_NAMES = {
    "without source": "Which scout suggested the item",
    "without customer": "About the shopper (age, habits)",
    "without item": "About the product (price, popularity)",
    "without pair": "Shopper and product together (past buying)",
    "without text": "Words in the product description",
    "source scores only": "Using only the scouts' scores",
}
OURS = "Our two-step system"


@st.cache_resource(show_spinner="Loading data and training the ranker (first run only)...")
def load_system(sample_frac: float) -> System:
    world = workspace.setup(sample_frac=sample_frac, seed=0)
    return System.build(world)


@st.cache_resource
def image_store() -> ImageStore:
    return ImageStore()


@st.cache_data
def results() -> tuple[dict, Path] | None:
    for folder in ("results", "docs/results"):  # the second is the committed copy
        p = Path(folder) / "results.json"
        if p.exists():
            return json.loads(p.read_text()), Path(folder)
    return None


def analysis(system: System) -> tuple[dict, str]:
    """Data statistics plus a sentence saying which data they describe."""
    for folder in ("results", "docs/results"):
        p = Path(folder) / "eda.json"
        if p.exists():
            e = json.loads(p.read_text())
            return e, (
                f"These charts describe the real H&M purchase records (a random "
                f"{e['sample_frac']:.0%} of customers, {e['customers']:,} shoppers). "
                "Only totals and averages are shown, nothing about individual people."
            )
    e = eda.summarize(system.world.hm, history_weeks=8)
    return e, "These charts describe the data currently loaded" + (
        " (made-up sample shoppers)." if system.world.synthetic else "."
    )


def bars(
    df: pd.DataFrame,
    label: str,
    value: str,
    title: str,
    horizontal: bool = True,
    percent: bool = False,
    color: str | None = None,
) -> alt.Chart:
    order = df[label].tolist()
    fmt = ".0%" if percent else ",.0f"
    cat = (
        alt.Y(
            f"{label}:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=220, labelOverlap=False),
        )
        if horizontal
        else alt.X(f"{label}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
    )
    val = (
        alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=fmt))
        if horizontal
        else alt.Y(f"{value}:Q", title=title, axis=alt.Axis(format=fmt))
    )
    enc = {"y": cat, "x": val} if horizontal else {"x": cat, "y": val}
    chart = alt.Chart(df).mark_bar(color=color) if color else alt.Chart(df).mark_bar()
    return chart.encode(**enc, tooltip=[label, alt.Tooltip(value, format=fmt)]).properties(
        height=max(120, 28 * len(df)) if horizontal else 260
    )


def range_bars(
    df: pd.DataFrame, title: str, zero_line: bool = False, signed: bool = False
) -> alt.Chart:
    """Bars with a thin 95% range; columns: label, value, lo, hi, group."""
    order = df["label"].tolist()
    axis = alt.Axis(labelLimit=320, labelOverlap=False)
    base = alt.Chart(df).encode(y=alt.Y("label:N", sort=order, title=None, axis=axis))
    colors = alt.Scale(
        domain=["ours", "other", "matters", "unclear"],
        range=["#d9534f", "#7f8fa6", "#3b7dd8", "#b8b8b8"],
    )
    bar = base.mark_bar().encode(
        x=alt.X("value:Q", title=title),
        color=alt.Color("group:N", scale=colors, legend=None),
        tooltip=[
            "label",
            alt.Tooltip("value", format=".4f"),
            alt.Tooltip("lo", format=".4f"),
            alt.Tooltip("hi", format=".4f"),
        ],
    )
    whisker = base.mark_rule(color="#555", strokeWidth=2).encode(x="lo:Q", x2="hi:Q")
    layers = [bar, whisker]
    if zero_line:
        layers.append(
            alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        )
    return alt.layer(*layers).properties(height=max(140, 40 * len(df)))


def pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.095 else f"{x:.1%}"


st.title(TITLE)
st.caption(SUBTITLE)
sample = st.sidebar.select_slider(
    "Customer sample",
    [0.02, 0.05, 0.1],
    value=0.1,
    help="Only used with the real data: how many customers to load. 10% is what the results use.",
)
system = load_system(sample)
world = system.world
if world.synthetic:
    st.warning(
        "This online demo runs on made-up sample shoppers, because H&M's real data may not be "
        "shared publicly. The Data analysis and How good is it? tabs show measurements taken "
        "on the real data."
    )

users = world.ev_test.users
st.sidebar.header("Choose a shopper")
st.sidebar.caption(f"{len(users):,} shoppers are available (all made purchases in the final week).")
st.session_state.setdefault("customer", 0)
st.sidebar.number_input(
    "Shopper number",
    min_value=0,
    max_value=len(users) - 1,
    step=1,
    key="customer",
    help="Any number works. Each number is a different shopper with their own purchase history.",
)
st.sidebar.button(
    "Pick a random shopper",
    on_click=lambda: st.session_state.update(customer=random.randrange(len(users))),
)
pick = int(st.session_state["customer"])
user = int(users[pick])
st.sidebar.header("Tune the list")
k = st.sidebar.slider("How many suggestions", 4, 24, 12)
lam = st.sidebar.slider(
    "Variety: 1.0 = best guesses only, lower = more varied",
    0.3,
    1.0,
    1.0,
    0.1,
    help="Technical name: MMR lambda. Lower values swap some top picks for items that look "
    "different from each other.",
)

images = image_store()
tab_start, tab_recs, tab_hist, tab_data, tab_eval = st.tabs(
    ["Start here", "Recommendations", "Purchase history", "Data analysis", "How good is it?"]
)

with tab_start:
    st.header("What is this?")
    st.markdown(
        "Imagine an online clothes shop with more than 100,000 products. Nobody can browse all of "
        "them, so the shop wants to **show each shopper the 12 items they are most likely to "
        "buy next week**. This app does exactly that, and lets you see how well it works.\n\n"
        "It learned from real H&M purchase records (31.8 million purchases over two years). To "
        "check it honestly, it was **tested like a time machine**: it only saw purchases up to a "
        "certain day, then we compared its suggestions with what shoppers really bought in the "
        "following week."
    )
    st.header("How to use this app")
    st.markdown(
        "1. **Pick a shopper** in the left panel (type a number, or press *Pick a random "
        "shopper*).\n"
        "2. Open **Purchase history** to see what that shopper bought in the past.\n"
        "3. Open **Recommendations** to see the 12 items the system suggests for next week, "
        "with a one-line reason for each. A green tick means the shopper really bought that "
        "item in the week we hid from the system.\n"
        "4. Open **Data analysis** for a tour of the purchase data in plain charts.\n"
        "5. Open **How good is it?** to see how the system scored against simpler ideas."
    )
    st.header("Things to try")
    st.markdown(
        "- **Press *Pick a random shopper* a few times.** Do the suggestions look sensible next "
        "to what each shopper bought before? Some shoppers are easy to predict, others are not.\n"
        "- **Move the *Variety* slider to the left.** The list becomes less repetitive (for "
        "example fewer near-identical T-shirts). The cost is that accuracy drops slightly; the "
        "last tab measures how much.\n"
        "- **Change *How many suggestions*.** A shorter list shows only the surest picks.\n"
        "- **Look at the reasons.** They are built only from facts in the records, such as "
        '"you bought it 2 times, last 14 days ago" or "trending this week". Nothing is '
        "invented."
    )
    st.header("How it works, in two steps")
    st.markdown(
        '1. **Make a shortlist.** Several quick "scouts" each suggest items: what the shopper '
        "bought before, what is trending, what people of the same age buy, items often bought "
        "together, items liked by shoppers with similar taste, and a small neural network that "
        "matches the style of recent purchases. Together they give about 160 candidates.\n"
        "2. **Sort the shortlist.** A learned ranking model looks at around 40 clues (for "
        "example how often the shopper buys this kind of product, or how the price compares with "
        "what they usually spend) and puts the likeliest items first."
    )
    if "two_tower" not in system.sources:
        st.caption(
            "Note: this online version leaves out the neural-network scout to stay light. The "
            "measurements in the last tab were taken with the full system."
        )
    with st.expander("Words used in this app"):
        st.markdown(
            "- **MAP@12**: the score. It runs from 0 to 1 and is higher when the items a "
            "shopper really bought appear near the top of the 12 suggestions. Scores on this "
            "kind of data are small (0.03 is good) because a shopper buys only a few of 100,000 "
            "products in a week, so compare scores with each other, not with 1.\n"
            '- **Baseline**: a simple rule to beat, such as "suggest the most popular items".\n'
            "- **95% range**: the thin line on a bar. The true value very likely lies inside it. "
            "If the ranges of two bars overlap a lot, the difference might be luck.\n"
            "- **Shortlist coverage**: the share of what shoppers went on to buy that the "
            "scouts managed to put on the shortlist. The ranking step can only pick from it.\n"
            "- **Variety (MMR)**: a setting that trades a little accuracy for a more varied list."
        )

with tab_recs:
    recs = system.recommend(user, k=k, lam=lam)
    bought_next = set(
        world.split.test.loc[world.split.test["user"] == user, "item"].astype(int).tolist()
    )
    past = int((world.ctx_fit.frame["user"] == user).sum())
    age = world.hm.customers.iloc[user]["age"]
    who = (
        f"Shopper {pick}: "
        + (f"age {age:.0f}, " if pd.notna(age) else "")
        + (f"{past} purchases before the final week.")
    )
    hits = int(recs["item"].isin(bought_next).sum())
    st.markdown(f"**{who}**")
    st.caption(
        f"In the final week, which the system never saw, this shopper bought {len(bought_next)} "
        f"item(s). {hits} of the {len(recs)} suggestions below are among them (marked ✅)."
        + (" Not seeing a tick is normal: shoppers buy few items in a week." if hits == 0 else "")
    )
    if not images.available:
        st.info("No product photos found, so only text is shown.")
    for _, r in recs.iterrows():
        mark = "✅ " if int(r["item"]) in bought_next else ""
        with st.container(border=True):
            pic, text = st.columns([1, 4])
            photo = images.get(int(r["article_id"])) if images.available else None
            if photo:
                pic.image(photo, width=110)
            text.markdown(f"**{mark}{r['name']}** · {r['product_type']} · {r['colour']}")
            text.caption(f"Why: {r['why']}  ·  score {r['score']:.3f} (only for sorting)")

with tab_hist:
    st.markdown(
        f"**What shopper {pick} bought most recently** (newest first). Compare this with "
        "the suggestions: do they fit the same taste?"
    )
    st.dataframe(system.history(user, 25), use_container_width=True, hide_index=True)

with tab_data:
    e, source = analysis(system)
    st.header("A tour of the purchase data")
    st.caption(source)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shoppers", f"{e['customers']:,}")
    c2.metric("Products on sale", f"{e['articles_in_catalogue']:,}")
    c3.metric("Purchases", f"{e['purchases']:,}")
    c4.metric("Days covered", f"{e['days']:,}")

    st.subheader("1. How busy was each week?")
    wk = pd.DataFrame(e["weekly_purchases"])
    wk["week_start"] = pd.to_datetime(wk["week_start"])
    st.line_chart(wk.set_index("week_start")["purchases"], height=220)
    st.caption(
        "Each point is one week of purchases. Rises and dips follow seasons and sales. The "
        "system is tested on the final week, so it must cope with whatever is in fashion that "
        "week; this is why one of its scouts looks at what is trending."
    )

    st.subheader("2. How much does a typical shopper buy?")
    pc = pd.DataFrame(e["purchases_per_customer"]).rename(
        columns={"band": "Purchases in two years", "count": "Shoppers"}
    )
    st.altair_chart(
        bars(pc, "Purchases in two years", "Shoppers", "Number of shoppers", horizontal=False),
        use_container_width=True,
    )
    st.caption(
        f"The middle shopper made {e['median_purchases_per_customer']:.0f} purchases, but the "
        f"average is {e['mean_purchases_per_customer']:.0f} because a minority shop very often. "
        "Shoppers with only a few purchases leave little to learn from, so they are the hardest "
        "to recommend for."
    )

    st.subheader("3. Do people buy the same thing again?")
    r1, r2 = st.columns(2)
    r1.metric(
        "Purchases that repeat an earlier purchase",
        pct(e["repeat_purchase_share"]),
        help="Share of all purchases where the same shopper had bought the same product before.",
    )
    r2.metric(
        "Final-week items the shopper had bought before",
        pct(e["test_week_repeat_share"]),
        help="Among the distinct items each shopper bought in the final week, the share "
        "they had also bought in the 26 weeks before.",
    )
    st.caption(
        "Repeat buying is real but not common in fashion: most of what a shopper buys next is "
        'something new to them. The simple rule "suggest what they bought before" still '
        "turned out to be the strongest single rule we tried, because the other simple rules "
        "are weaker. The two-step system has to find new items and also remember old ones."
    )

    st.subheader("4. What do people buy?")
    g1, g2 = st.columns(2)
    pg = pd.DataFrame(e["product_groups"]).head(8)
    pg["name"] = pg["name"].str.replace("Garment ", "", regex=False)
    g1.altair_chart(
        bars(
            pg.rename(columns={"name": "Product group"}),
            "Product group",
            "share",
            "Share of all purchases",
            percent=True,
        ),
        use_container_width=True,
    )
    pt = pd.DataFrame(e["product_types"]).head(10)
    g2.altair_chart(
        bars(
            pt.rename(columns={"name": "Product type"}),
            "Product type",
            "share",
            "Share of all purchases",
            percent=True,
        ),
        use_container_width=True,
    )
    colours = pd.DataFrame(e["colours"]).rename(columns={"name": "Colour"})
    st.altair_chart(
        bars(colours, "Colour", "share", "Share of all purchases", percent=True),
        use_container_width=True,
    )
    st.caption(
        "Tops and bottoms dominate, and black is by far the most common colour. The system "
        "reads the product group, type, colour and description of each item, so it can tell "
        "that two different items are alike."
    )

    st.subheader("5. Who shops?")
    ages = pd.DataFrame(e["age_customers"]).rename(columns={"band": "Age", "count": "Shoppers"})
    st.altair_chart(
        bars(ages, "Age", "Shoppers", "Number of shoppers (age known)", horizontal=False),
        use_container_width=True,
    )
    st.caption(
        "Age is one of the clues the system uses, and one of its scouts suggests what is popular "
        "with the shopper's own age group."
    )

    st.subheader("6. A few products sell far more than the rest")
    s1, s2, s3 = st.columns(3)
    s1.metric(
        f"Top 1% of products ({e['top1pct_articles_count']:,})",
        pct(e["top1pct_articles_share_of_purchases"]) + " of purchases",
    )
    s2.metric(
        "Products never bought", pct(1 - e["articles_purchased"] / e["articles_in_catalogue"])
    )
    s3.metric("Typical sold product bought", f"{e['median_purchases_per_sold_article']:.0f} times")
    st.caption(
        "Sales are concentrated in a small set of bestsellers while most products sell rarely. "
        "A system that only shows bestsellers would be easy to build but boring and unhelpful "
        "for shoppers with their own taste. That is why the next tab also measures how varied "
        "the suggestions are."
    )

    st.subheader("7. What do items cost?")
    h = e["price_histogram"]
    hist = pd.DataFrame({"from": h["edges"][:-1], "to": h["edges"][1:], "items": h["counts"]})
    price_chart = (
        alt.Chart(hist)
        .mark_bar()
        .encode(
            x=alt.X("from:Q", title="Price (in the dataset's own scaled units)"),
            x2="to:Q",
            y=alt.Y("items:Q", title="Purchases"),
            tooltip=["from", "to", "items"],
        )
        .properties(height=220)
    )
    st.altair_chart(price_chart, use_container_width=True)
    st.caption(
        "The dataset hides real currencies, so prices are shown on a scaled range (the highest "
        "1% are grouped into the last bar). Most purchases are low-priced items. Price compared "
        "with what a shopper usually spends is one of the clues the ranking step uses."
    )

with tab_eval:
    loaded = results()
    if loaded is None:
        st.info("Run `python -m hmrec.run` to generate results/results.json.")
    else:
        res, folder = loaded
        st.header("Is the system any good?")
        st.success(report.headline(folder))
        st.caption(
            f"Measured on {res['eval_customers']['test']:,} real shoppers over one hidden week, "
            f"using a {res['config']['sample_frac']:.0%} random sample of H&M customers. "
            "Higher is better, but scores on this kind of data are small, so what matters is the "
            "comparison between the bars."
        )

        st.subheader("1. Our system against simpler ideas")
        full = res["ranker"]["ablation"]["full"]
        rows = [
            {
                "label": OURS,
                "value": full["map@12"],
                "lo": full["ci_map@12"][0],
                "hi": full["ci_map@12"][1],
                "group": "ours",
            }
        ]
        for name, v in sorted(res["baselines_test"].items(), key=lambda kv: -kv[1]["map@12"]):
            rows.append(
                {
                    "label": BASELINE_NAMES.get(name, name),
                    "value": v["map@12"],
                    "lo": v["ci_map@12"][0],
                    "hi": v["ci_map@12"][1],
                    "group": "other",
                }
            )
        st.altair_chart(
            range_bars(pd.DataFrame(rows), "Score (MAP@12, higher is better)"),
            use_container_width=True,
        )
        best = res["ranker"]["full_vs_best_baseline"]
        st.caption(
            "The red bar is the two-step system; the grey-blue bars are simple rules. The thin "
            "line on each bar is its 95% range. Against the best simple rule, the system scores "
            f"{best['diff']:+.4f} higher, and the 95% range of that gain is {best['lo']:+.4f} to "
            f"{best['hi']:+.4f}. "
            + (
                "Because that range stays above zero, the gain is unlikely to be luck."
                if best["lo"] > 0
                else "That range includes zero, so the gain could be luck."
            )
        )

        st.subheader("2. Which clues helped?")
        paired = res["ranker"]["full_vs_ablated"]
        rows = []
        for name, p in paired.items():
            rows.append(
                {
                    "label": CLUE_NAMES.get(name, name),
                    "value": -p["diff"],
                    "lo": -p["hi"],
                    "hi": -p["lo"],
                    "group": "matters" if p["significant"] else "unclear",
                }
            )
        st.altair_chart(
            range_bars(
                pd.DataFrame(rows), "Change in score when this is taken away", zero_line=True
            ),
            use_container_width=True,
        )
        st.caption(
            "We took away one group of clues at a time and measured how much worse the score "
            "became. A bar to the left of the dashed zero line means the group helped. "
            "**Blue bars** have a range that stays on one side of zero, so we are fairly sure "
            "of the effect. **Grey bars** cross zero: on this sample the effect is too small to "
            "tell from luck."
            + (
                " The words in product descriptions added nothing we could detect."
                if not paired.get("without text", {}).get("significant", True)
                else ""
            )
        )

        st.subheader("3. How good is the shortlist?")
        rec = res["candidates"]["test_recall"]
        sc = pd.DataFrame(
            [
                {"Scout": SCOUT_NAMES.get(k, k), "share": v}
                for k, v in sorted(rec.items(), key=lambda kv: -kv[1])
            ]
        )
        st.altair_chart(
            bars(sc, "Scout", "share", "Share of what shoppers bought that it found", percent=True),
            use_container_width=True,
        )
        st.caption(
            f"Each scout suggests items; this is the share of the shoppers' real purchases it "
            f"found. Together they find {pct(rec['union'])} (about "
            f"{res['candidates']['avg_candidates_per_customer']:.0f} candidates per shopper). The "
            "ranking step can only choose from this list, so finding more of what shoppers "
            "buy is the biggest room for improvement."
        )

        st.subheader("4. Variety versus accuracy")
        mm = pd.DataFrame(res["mmr_tradeoff"]).T
        mm.index = [f"{float(i):.1f}" for i in mm.index]
        mm.index.name = "Variety setting"
        mm = mm.reset_index()
        m1, m2 = st.columns(2)
        m1.altair_chart(
            alt.Chart(mm)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Variety setting:N",
                    sort=mm["Variety setting"].tolist(),
                    title="Variety setting (1.0 = off, lower = more varied)",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("map@12:Q", title="Score (MAP@12)", scale=alt.Scale(zero=False)),
            )
            .properties(height=240),
            use_container_width=True,
        )
        m2.altair_chart(
            alt.Chart(mm)
            .mark_line(point=True, color="#3b7dd8")
            .encode(
                x=alt.X(
                    "Variety setting:N",
                    sort=mm["Variety setting"].tolist(),
                    title="Variety setting (1.0 = off, lower = more varied)",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "tag_ild:Q", title="Category variety of the list", scale=alt.Scale(zero=False)
                ),
            )
            .properties(height=240),
            use_container_width=True,
        )
        first, last = mm.iloc[0], mm.iloc[-1]
        st.caption(
            f"Moving the Variety slider from {first['Variety setting']} to "
            f"{last['Variety setting']} lowers the score from {first['map@12']:.4f} to "
            f"{last['map@12']:.4f} (left chart) and raises category variety from "
            f"{first['tag_ild']:.3f} to {last['tag_ild']:.3f} (right chart). It is a trade: "
            "a more varied list is less likely to hit what the shopper buys next, but is "
            "less repetitive to look at."
        )

        with st.expander("Honest limits of these results"):
            st.markdown(
                "- The test replays past purchases. It shows how well the system predicts what "
                "shoppers bought; it cannot show what they would have bought if shown something "
                "else.\n"
                "- Results come from a random 10% of customers, so they are not comparable with "
                "the public Kaggle leaderboard.\n"
                "- Shoppers with no earlier purchases are left out of the score.\n"
                "- The ranking model was lightly tuned on one validation week and trained with "
                "one random seed."
            )
