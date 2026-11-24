"""ReviewSense AI v2 - laptop review analysis and recommendation (Streamlit)."""

from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

import core

st.set_page_config(page_title="ReviewSense AI", page_icon="💻", layout="wide")

# Minimal styling only: colours come from Streamlit's own theme (.streamlit/config.toml),
# so dark/light mode (Settings menu) works without CSS hacks.
st.markdown(
    """<style>
    [data-testid="stMainBlockContainer"] { max-width: 1300px; padding-top: 2rem; }
    .rs-brand { font-size: 1.5rem; font-weight: 700; }
    .rs-brand span { color: #2fbf9f; }
    </style>""",
    unsafe_allow_html=True,
)

SENT_COLORS = {"positive": "#2fbf9f", "neutral": "#e0b04a", "negative": "#e5645f"}
ASPECTS = list(core.ASPECTS)


# ----------------------------------------------------------- version-safe helpers
def show_chart(fig) -> None:
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


# --------------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner="Loading data…")
def load(path: str, mtime: float):
    del mtime  # only used to refresh the cache when the file changes
    return core.load_master(path)


@st.cache_resource(show_spinner="Training sentiment model (first run only)…")
def get_model(path: str, mtime: float):
    reviews, _ = load(path, mtime)
    return core.train_model(reviews)


@st.cache_data(show_spinner="Analyzing reviews…")
def get_analysis(path: str, mtime: float) -> pd.DataFrame:
    reviews, _ = load(path, mtime)
    model, _info = get_model(path, mtime)
    return core.analyze_reviews(reviews, model)


@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


data_path, is_sample = core.resolve_data_path()
if not data_path.exists():
    st.error("No dataset found. Put `reviewsense_master.csv` in the `data/` folder, "
             "or run `python make_sample_data.py` to create demo data.")
    st.stop()
mtime = data_path.stat().st_mtime
try:
    reviews_raw, catalog = load(str(data_path), mtime)
except Exception as exc:  # unreadable / malformed CSV
    st.error(f"Could not read {data_path.name}: {exc}")
    st.stop()
if reviews_raw.empty:
    st.error("The dataset has no usable review rows (need `laptop` and `review` columns).")
    st.stop()

# ------------------------------------------------------------------------ header/nav
st.markdown('<div class="rs-brand">◈ ReviewSense <span>AI</span></div>', unsafe_allow_html=True)
if is_sample:
    st.warning("Showing **synthetic demo data** because `data/reviewsense_master.csv` was not found. "
               "Copy your real data folder into this project to use your dataset.")

PAGES = ["Overview", "Reviews", "Compare", "Try a review", "Recommendations"]
page = st.radio("Navigate", PAGES, horizontal=True, label_visibility="collapsed", key="page")
st.divider()

model, model_info = get_model(str(data_path), mtime)
analyzed = get_analysis(str(data_path), mtime)

# ============================================================================ OVERVIEW
if page == "Overview":
    st.title("Laptop review overview")
    st.caption("A snapshot of what laptop owners are saying.")

    counts = analyzed.sentiment.value_counts().reindex(core.SENTIMENTS, fill_value=0)
    complaint = pd.DataFrame({
        "Aspect": ASPECTS,
        "Complaints": [int((analyzed[f"{a} feedback"] == "negative").sum()) for a in ASPECTS],
    })
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analyzed", f"{len(analyzed):,}")
    c2.metric("Laptops", f"{analyzed.laptop.nunique():,}")
    c3.metric("Positive reviews", f"{counts['positive'] / len(analyzed):.0%}")
    top = complaint.sort_values("Complaints", ascending=False).iloc[0]
    c4.metric("Top complaint", top.Aspect if top.Complaints else "None detected")

    left, right = st.columns(2)
    with left:
        sent_df = counts.rename_axis("Sentiment").reset_index(name="Reviews")
        fig = px.pie(sent_df, names="Sentiment", values="Reviews", hole=0.55, color="Sentiment",
                     color_discrete_map=SENT_COLORS, title="Sentiment breakdown")
        show_chart(fig)
    with right:
        fig = px.bar(complaint, x="Complaints", y="Aspect", orientation="h", title="Reviews with a complaint",
                     color_discrete_sequence=[SENT_COLORS["negative"]])
        fig.update_layout(yaxis_title=None)
        show_chart(fig)

    st.subheader("Most-reviewed laptops")
    grp = analyzed.groupby("laptop").agg(
        Reviews=("review", "size"),
        Avg_rating=("rating", "mean"),
        Positive=("sentiment", lambda s: (s == "positive").mean() * 100),
    ).reset_index().sort_values("Reviews", ascending=False).head(10)
    grp = grp.rename(columns={"laptop": "Laptop", "Avg_rating": "Avg rating", "Positive": "Positive %"})
    show_table(grp.round(1), column_config={
        "Positive %": st.column_config.ProgressColumn("Positive %", min_value=0, max_value=100, format="%.0f%%"),
    })
    st.info("Sentiment comes from star ratings where available, otherwise from the trained model. "
            "Aspect signals are keyword-based estimates, so treat them as a guide.")

# ============================================================================ REVIEWS
elif page == "Reviews":
    st.title("Browse reviews")
    counts_by_laptop = analyzed.laptop.value_counts()
    query = st.text_input("Search laptop name", placeholder="e.g. Lenovo, HP, 16GB")
    names = [n for n in counts_by_laptop.index if query.lower() in n.lower()] if query else list(counts_by_laptop.index)
    if not names:
        st.info("No laptop matches that search.")
        st.stop()
    selected = st.selectbox("Laptop", names, format_func=lambda n: f"{n}  ({counts_by_laptop[n]} reviews)")
    subset = analyzed[analyzed.laptop == selected]

    st.markdown("**Review summary**")
    st.write(core.summarize(subset.review))
    chosen = st.multiselect("Show sentiment", core.SENTIMENTS, default=core.SENTIMENTS)
    visible = subset[subset.sentiment.isin(chosen)]
    if visible.empty:
        st.info("No reviews match the selected sentiment filter.")
    else:
        page_size = 50
        pages = max(1, -(-len(visible) // page_size))
        pg = st.number_input("Page", 1, pages, 1, step=1) if pages > 1 else 1
        start = (int(pg) - 1) * page_size
        chunk = visible.iloc[start:start + page_size]
        st.caption(f"Showing {start + 1:,}–{start + len(chunk):,} of {len(visible):,} reviews")
        cols = ["rating", "review", "sentiment"] + [f"{a} feedback" for a in ASPECTS]
        show_table(chunk[cols].rename(columns={"rating": "Rating", "review": "Review", "sentiment": "Sentiment"}))

    export = analyzed[["laptop", "rating", "review", "sentiment"] + [f"{a} feedback" for a in ASPECTS]]
    st.download_button("⬇ Download all analyzed reviews (CSV)", to_csv_bytes(export),
                       "reviewsense_analyzed.csv", "text/csv")

# ============================================================================ COMPARE
elif page == "Compare":
    st.title("Compare laptops")
    st.caption("Pick two or more laptops to compare sentiment and complaint rates side by side.")
    counts_by_laptop = analyzed.laptop.value_counts()
    options = list(counts_by_laptop.index)
    selected = st.multiselect("Laptops", options, default=options[:3],
                              format_func=lambda n: f"{n}  ({counts_by_laptop[n]} reviews)")
    if len(selected) < 2:
        st.info("Choose at least two laptops to compare.")
    else:
        sub = analyzed[analyzed.laptop.isin(selected)]
        comp = sub.groupby("laptop").agg(
            Reviews=("review", "size"), Avg_rating=("rating", "mean"),
            Positive=("sentiment", lambda s: (s == "positive").mean() * 100),
            Neutral=("sentiment", lambda s: (s == "neutral").mean() * 100),
            Negative=("sentiment", lambda s: (s == "negative").mean() * 100),
        )
        for a in ASPECTS:
            comp[f"{a} complaints %"] = sub.groupby("laptop")[f"{a} feedback"].apply(lambda s: (s == "negative").mean() * 100)
        comp = comp.reset_index().rename(columns={"laptop": "Laptop", "Avg_rating": "Avg rating",
                                                  "Positive": "Positive %", "Neutral": "Neutral %",
                                                  "Negative": "Negative %"})
        show_table(comp.round(1))
        melt = comp.melt(id_vars="Laptop", value_vars=[f"{a} complaints %" for a in ASPECTS],
                         var_name="Aspect", value_name="% of reviews")
        melt["Aspect"] = melt.Aspect.str.replace(" complaints %", "", regex=False)
        fig = px.bar(melt, x="Laptop", y="% of reviews", color="Aspect", barmode="group",
                     title="Share of reviews with a complaint, by aspect")
        show_chart(fig)
        mix = comp.melt(id_vars="Laptop", value_vars=["Positive %", "Neutral %", "Negative %"],
                        var_name="Sentiment", value_name="% of reviews")
        mix["Sentiment"] = mix.Sentiment.str.replace(" %", "", regex=False).str.lower()
        fig = px.bar(mix, x="Laptop", y="% of reviews", color="Sentiment", barmode="stack",
                     color_discrete_map=SENT_COLORS, title="Sentiment mix")
        show_chart(fig)

# ======================================================================== TRY A REVIEW
elif page == "Try a review":
    st.title("Try a review")
    st.caption("Type or paste any laptop review to see the predicted sentiment and aspect signals.")
    if model is None:
        st.warning("Not enough star-rated reviews to train a model, so predictions are unavailable.")
    else:
        text = st.text_area("Review text", height=140,
                            placeholder="The battery barely lasts 3 hours and the fan gets really loud…")
        if st.button("Analyze", type="primary"):
            if not text.strip():
                st.info("Please enter some review text first.")
            else:
                label, scores = core.predict_with_scores(model, text)
                st.metric("Predicted sentiment", label.capitalize())
                if scores:
                    sdf = pd.DataFrame({"Sentiment": list(scores), "Probability": list(scores.values())})
                    fig = px.bar(sdf, x="Probability", y="Sentiment", orientation="h", color="Sentiment",
                                 color_discrete_map=SENT_COLORS, range_x=[0, 1])
                    fig.update_layout(showlegend=False, height=220)
                    show_chart(fig)
                tokens = core.tokenize(text)
                st.markdown("**Aspect signals**")
                cols = st.columns(len(ASPECTS))
                for col, aspect in zip(cols, ASPECTS):
                    col.metric(aspect, core.aspect_opinion(tokens, core.ASPECTS[aspect]).capitalize())
        acc = model_info["accuracy"]
        st.caption(f"Model: {model_info['kind']} · trained on {model_info['n_train']:,} star-rated reviews"
                   + (f" · held-out accuracy {acc:.0%}" if acc is not None else ""))

# ================================================================== RECOMMENDATIONS
else:
    st.title("Find your best match")
    st.caption("Set a budget and priorities and we'll shortlist laptops from the price snapshot.")
    if catalog.empty:
        st.info("The dataset has no priced laptops for recommendations.")
        st.stop()
    max_price = int(catalog.price.max())
    c1, c2 = st.columns(2)
    budget = c1.number_input("Maximum budget (₹)", min_value=10000, max_value=max(1_000_000, max_price),
                             value=100000, step=5000)
    usage = c2.selectbox("Main use", list(core.USE_CASE_MAP))
    prefs = st.multiselect("Priorities", core.PREFERENCES)
    st.caption("“Dedicated graphics” is a strict filter; the other priorities raise the match score.")

    recs = core.recommend(catalog, float(budget), usage, prefs)
    if recs.empty:
        st.info("No laptops match that graphics preference. Try clearing it.")
        st.stop()
    within = recs[recs["Within budget"]]
    if within.empty:
        st.warning("Nothing is listed within this budget. Showing the closest matches instead.")
    shown = (within if not within.empty else recs).head(15)

    top = shown.head(3)
    for col, (rank, (_, pick)) in zip(st.columns(len(top)), enumerate(top.iterrows(), 1)):
        with col.container(border=True):
            st.caption(f"MATCH {rank} · {pick['Match score']}%")
            st.markdown(f"**{html.escape(str(pick['Laptop']))}**")
            st.subheader(core.money(pick["Price (₹)"]))
            st.write(pick["Specs"])
    show_table(shown.drop(columns=["Within budget"]), column_config={
        "Price (₹)": st.column_config.NumberColumn("Price (₹)", format="₹%d"),
        "Match score": st.column_config.ProgressColumn("Match score", min_value=0, max_value=100, format="%d"),
    })
    fig = px.bar(shown.head(10), x="Laptop", y="Match score", range_y=[0, 100], title="Top match scores",
                 color_discrete_sequence=["#2fbf9f"])
    show_chart(fig)
    st.caption("Prices come from a dated snapshot and can change. Confirm price, stock and exact configuration "
               "with the seller. Match scores are heuristic estimates, not benchmarks.")
