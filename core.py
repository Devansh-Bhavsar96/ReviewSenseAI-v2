"""ReviewSense AI v2 - data loading, sentiment model and recommendation logic.

This module has no Streamlit code, so it can be tested on its own.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except (ImportError, OSError):  # e.g. blocked compiled libraries on some Windows PCs
    SKLEARN_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
MASTER_PATH = BASE_DIR / "data" / "reviewsense_master.csv"
SAMPLE_PATH = BASE_DIR / "data" / "sample_master.csv"

SENTIMENTS = ["positive", "neutral", "negative"]
UNKNOWN_LAPTOPS = {"", "laptop model not specified", "unknown", "nan"}

# Exact-word keyword sets (exact matching avoids false hits like "fan" in "fantastic").
ASPECTS = {
    "Battery": {"battery", "charge", "charging", "charger", "backup", "drains", "drain", "lasts"},
    "Heating": {"heat", "heats", "heating", "hot", "warm", "overheat", "overheating", "overheats", "fan", "fans"},
    "Display": {"display", "screen", "brightness", "bright", "dim", "colors", "colours", "color", "colour", "oled", "resolution"},
    "Performance": {"performance", "slow", "slows", "stutter", "stutters", "lag", "lags", "laggy", "fast", "smooth",
                    "processor", "speed", "gaming", "render", "hang", "hangs"},
}
POSITIVE_WORDS = {
    "good", "great", "excellent", "strong", "smooth", "fast", "bright", "crisp", "sharp", "beautiful",
    "reliable", "lasts", "long", "quick", "quiet", "impressive", "accurate", "responsive", "comfortable",
    "enjoyable", "solid", "easy", "clear", "holds", "amazing", "awesome", "superb", "best", "nice", "perfect",
}
NEGATIVE_WORDS = {
    "bad", "poor", "short", "drains", "drain", "hot", "heats", "heating", "warm", "loud", "noisy", "dim",
    "washed", "flat", "slow", "slows", "stutter", "lag", "laggy", "expensive", "heavy", "quickly", "average",
    "stutters", "unreliable", "weak", "worst", "terrible", "awful", "hangs", "overheating", "overheats",
}
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "doesn't", "don't", "didn't", "cannot", "can't", "hardly"}


# --------------------------------------------------------------------------- data
def money(value: float | None) -> str:
    return f"₹{value:,.0f}" if value and not pd.isna(value) else "Price not listed"


def resolve_data_path() -> tuple[Path, bool]:
    """Return (path, is_sample). Falls back to the bundled demo data."""
    if MASTER_PATH.exists():
        return MASTER_PATH, False
    return SAMPLE_PATH, True


def load_master(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the master CSV and split it into (reviews, catalog)."""
    master = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    master.columns = [str(c).strip().lower() for c in master.columns]
    if "record_type" not in master:
        master["record_type"] = "review"
    for col in ["laptop", "review", "title", "rating", "sentiment", "aspect_terms", "catalog_group",
                "price", "price_currency", "price_as_of", "ram_gb", "storage_gb", "screen_inches",
                "display_resolution", "processor", "gpu", "use_case", "recommendation_eligible",
                "price_source_url"]:
        if col not in master:
            master[col] = pd.NA

    reviews = master[master.record_type.eq("review")].copy()
    reviews["laptop"] = reviews.laptop.fillna("").astype(str).str.strip()
    reviews["review"] = reviews.review.fillna("").astype(str).str.strip()
    reviews = reviews[~reviews.laptop.str.casefold().isin(UNKNOWN_LAPTOPS) & reviews.review.ne("")].copy()
    title = reviews.title.fillna("").astype(str).str.strip()
    has_title = title.ne("") & ~title.str.casefold().eq("nan")
    reviews["text"] = reviews.review
    reviews.loc[has_title, "text"] = title[has_title] + ". " + reviews.loc[has_title, "review"]
    reviews["rating"] = pd.to_numeric(reviews.rating, errors="coerce")
    derived = pd.Series(pd.NA, index=reviews.index, dtype="object")
    derived[reviews.rating >= 4] = "positive"
    derived[reviews.rating == 3] = "neutral"
    derived[reviews.rating <= 2] = "negative"
    given = reviews.sentiment.astype("object").where(reviews.sentiment.isin(SENTIMENTS), pd.NA)
    reviews["rating_sentiment"] = given.fillna(derived)
    reviews = reviews.reset_index(drop=True)

    catalog = master[master.record_type.eq("catalog")].copy()
    catalog["laptop"] = catalog.laptop.fillna("").astype(str).str.strip()
    for col in ["price", "ram_gb", "storage_gb", "screen_inches"]:
        catalog[col] = pd.to_numeric(catalog[col], errors="coerce")
    eligible = catalog.recommendation_eligible.astype(str).str.lower().eq("true")
    catalog = catalog[eligible & catalog.price.notna() & catalog.laptop.ne("")]
    catalog = catalog[catalog.price_currency.fillna("INR").astype(str).str.upper().eq("INR")].copy()
    catalog["use_case"] = catalog.use_case.fillna("general").astype(str)
    catalog["price_as_of"] = catalog.price_as_of.fillna("").astype(str)
    return reviews, catalog.reset_index(drop=True)


# ------------------------------------------------------------------ sentiment model
class NaiveBayesFallback:
    """Tiny pure-Python TF-IDF-style Naive Bayes used only if scikit-learn can't load."""

    def __init__(self, max_features: int = 8000, alpha: float = 1.0):
        self.max_features, self.alpha = max_features, alpha
        self.idf: dict[str, float] = {}
        self.logp: dict[str, dict[str, float]] = {}
        self.classes_: list[str] = []

    @staticmethod
    def _tok(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9']+", str(text).lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def fit(self, texts, labels):
        docs = [self._tok(t) for t in texts]
        labels = [str(x) for x in labels]
        df: dict[str, int] = {}
        tf: dict[str, int] = {}
        for words in docs:
            for w in set(words):
                df[w] = df.get(w, 0) + 1
            for w in words:
                tf[w] = tf.get(w, 0) + 1
        vocab = sorted(tf, key=tf.get, reverse=True)[: self.max_features]
        self.idf = {w: 1.0 + math.log((1 + len(docs)) / (1 + df[w])) for w in vocab}
        self.classes_ = sorted(set(labels))
        sums = {c: {} for c in self.classes_}
        for words, label in zip(docs, labels):
            for w in words:
                if w in self.idf:
                    sums[label][w] = sums[label].get(w, 0.0) + self.idf[w]
        for c in self.classes_:
            total = sum(sums[c].values()) + self.alpha * len(self.idf)
            self.logp[c] = {w: math.log((sums[c].get(w, 0.0) + self.alpha) / total) for w in self.idf}
        return self

    def predict(self, texts):
        out = []
        for t in texts:
            scores = {c: sum(self.logp[c][w] * self.idf[w] for w in self._tok(t) if w in self.idf)
                      for c in self.classes_}
            out.append(max(scores, key=scores.get))
        return out


def train_model(reviews: pd.DataFrame):
    """Train on star-rated reviews. Returns (model_or_None, info_dict)."""
    train = reviews[reviews.rating_sentiment.isin(SENTIMENTS)]
    if train.rating_sentiment.nunique() < 2 or len(train) < 30:
        return None, {"accuracy": None, "n_train": 0, "kind": "none"}
    texts, labels = train.text.astype(str), train.rating_sentiment.astype(str)
    try:
        x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    except ValueError:
        x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42)
    if SKLEARN_AVAILABLE:
        def make():
            return Pipeline([
                ("tfidf", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=2,
                                          max_features=20000, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
        kind = "TF-IDF + Logistic Regression"
    else:
        make, kind = NaiveBayesFallback, "TF-IDF + Naive Bayes (fallback)"
    held_out = make().fit(x_tr, y_tr)
    accuracy = float((pd.Series(held_out.predict(x_te)).to_numpy() == y_te.to_numpy()).mean())
    final = make().fit(texts, labels)  # final model uses all labelled data
    return final, {"accuracy": accuracy, "n_train": len(train), "kind": kind}


def predict_with_scores(model, text: str) -> tuple[str, dict[str, float]]:
    label = str(model.predict([text])[0])
    scores: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        scores = {str(c): float(p) for c, p in zip(model.classes_, probs)}
    return label, scores


# ------------------------------------------------------------------ aspect analysis
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def aspect_opinion(tokens: list[str], keywords: set[str]) -> str:
    hits = [i for i, w in enumerate(tokens) if w in keywords]
    if not hits:
        return "not mentioned"
    pos = neg = 0
    for i in hits:
        for j in range(max(0, i - 4), min(len(tokens), i + 5)):
            w = tokens[j]
            flipped = j > 0 and tokens[j - 1] in NEGATORS
            if w in POSITIVE_WORDS:
                pos, neg = (pos, neg + 1) if flipped else (pos + 1, neg)
            elif w in NEGATIVE_WORDS:
                pos, neg = (pos + 1, neg) if flipped else (pos, neg + 1)
    return "negative" if neg > pos else ("positive" if pos > neg else "mentioned")


def analyze_reviews(reviews: pd.DataFrame, model) -> pd.DataFrame:
    """Add final sentiment and per-aspect opinions to the review table."""
    out = reviews.copy()
    out["sentiment"] = out.rating_sentiment
    missing = ~out.sentiment.isin(SENTIMENTS)
    if missing.any():
        if model is not None:
            out.loc[missing, "sentiment"] = model.predict(out.loc[missing, "text"].astype(str))
        else:
            out.loc[missing, "sentiment"] = "neutral"
    tokens = out.text.map(tokenize)
    for aspect, kw in ASPECTS.items():
        out[f"{aspect} feedback"] = tokens.map(lambda t, kw=kw: aspect_opinion(t, kw))
    return out


def summarize(reviews: pd.Series, limit: int = 3) -> str:
    """Pick representative sentences by word-frequency scoring (extractive summary)."""
    sentences: list[str] = []
    for review in reviews.astype(str).head(400):
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", review) if len(s.strip()) > 18)
    sentences = list(dict.fromkeys(sentences))  # drop repeated sentences
    if not sentences:
        return "There is not enough review text to summarize."
    stop = {"the", "and", "for", "with", "this", "that", "are", "but", "when", "from", "into", "have", "has",
            "its", "you", "your", "was", "were", "very", "not", "can", "will", "also", "just", "than"}
    toks = [[w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in stop] for s in sentences]
    freq: dict[str, int] = {}
    for row in toks:
        for w in set(row):
            freq[w] = freq.get(w, 0) + 1
    scored = sorted(((sum(freq[w] for w in row) / max(len(row), 1), i) for i, row in enumerate(toks)),
                    reverse=True)
    chosen = sorted(i for _, i in scored[:limit])
    return " ".join(sentences[i] for i in chosen)


# --------------------------------------------------------------- recommendations
USE_CASE_MAP = {
    "Study / office": ["study_office", "office", "general"],
    "Gaming": ["gaming"],
    "Creative work": ["creator", "gaming"],
    "Everyday use": ["office", "study_office", "creator", "general"],
}
PREFERENCES = ["16 GB RAM or more", "Dedicated graphics", "14-inch or smaller",
               "High-resolution display", "Affordable within my budget"]


def is_dedicated_gpu(name: object) -> bool:
    gpu = str(name or "").casefold()
    patterns = (r"\b(?:geforce\s*)?(?:rtx|gtx)\s*\d", r"\bgeforce\s*mx\s*\d", r"\bquadro\b",
                r"\bradeon\s*(?:rx|pro)\s*[a-z0-9]", r"\barc\s*a\s*\d{3,4}m?\b")
    return any(re.search(p, gpu) for p in patterns)


def recommend(catalog: pd.DataFrame, budget: float, usage: str, prefs: list[str]) -> pd.DataFrame:
    """Rank catalog rows. Heuristic score 0-100; not a benchmark."""
    cat = catalog.copy()
    if "Dedicated graphics" in prefs:
        cat = cat[cat.gpu.map(is_dedicated_gpu)]
    wanted = USE_CASE_MAP[usage]
    rows = []
    for _, p in cat.iterrows():
        price = float(p.price)
        within = price <= budget
        score = 45.0 if within else max(0.0, 45.0 * budget / price)
        fits_use = p.use_case in wanted
        score += 25 if fits_use else 8
        why = ["Within budget" if within else "Over budget"]
        if fits_use:
            why.append(f"Suits {usage.lower()}")
        if "16 GB RAM or more" in prefs and pd.notna(p.ram_gb) and p.ram_gb >= 16:
            score += 10
            why.append("16 GB+ RAM")
        if "Dedicated graphics" in prefs:
            score += 10
            why.append(f"GPU: {p.gpu}")
        if "14-inch or smaller" in prefs and pd.notna(p.screen_inches) and p.screen_inches <= 14:
            score += 10
            why.append(f"{p.screen_inches:g}-inch screen")
        if "High-resolution display" in prefs:
            m = re.search(r"\d{3,4}\s*[x×]\s*(\d{3,4})", str(p.display_resolution))
            if m and int(m.group(1)) >= 1440:
                score += 10
                why.append(f"{p.display_resolution} display")
        if "Affordable within my budget" in prefs and within:
            score += 8 * max(0.0, 1 - price / budget)
            why.append("Good value for budget")
        specs = []
        if pd.notna(p.ram_gb):
            specs.append(f"{p.ram_gb:g} GB RAM")
        if pd.notna(p.storage_gb):
            specs.append(f"{p.storage_gb:g} GB storage")
        if pd.notna(p.screen_inches):
            specs.append(f'{p.screen_inches:g}"')
        rows.append({"Laptop": p.laptop, "Match score": int(round(min(100, score))), "Price (₹)": price,
                     "Within budget": within, "Specs": ", ".join(specs) or "Limited specs listed",
                     "Why it matches": "; ".join(why), "Price checked": p.price_as_of})
    if not rows:
        return pd.DataFrame(columns=["Laptop", "Match score", "Price (₹)", "Within budget", "Specs",
                                     "Why it matches", "Price checked"])
    return pd.DataFrame(rows).sort_values(["Match score", "Price (₹)"], ascending=[False, True]).reset_index(drop=True)
