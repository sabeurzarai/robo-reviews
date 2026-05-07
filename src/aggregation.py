"""Category-level insights and product rankings."""
from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import POSITIVE_RATIO_WEIGHT, RATING_WEIGHT, REVIEW_COUNT_WEIGHT, TOP_N_PRODUCTS
from src.preprocessing import normalize_text

logger = logging.getLogger(__name__)

COMPLAINT_TERMS = [
    "battery", "charge", "slow", "broken", "screen", "sound", "price", "quality",
    "setup", "shipping", "return", "wifi", "connection", "app", "remote", "durable",
]

# Truly universal noise; everything else (fire, echo, kindle, tablet, tv,
# speaker, reader, voice, remote, hd) is left to TF-IDF, which naturally
# penalises terms that appear across many clusters.
CATEGORY_NAME_STOPWORDS = frozenset({
    "amazon", "new", "all", "with", "for", "and", "the", "of", "or", "in", "on",
    "by", "to", "from", "this", "that",
    "case", "edition", "version", "series", "model", "device", "devices",
    "includes", "include", "special", "offers", "offer",
    "wifi", "wi", "fi", "bluetooth", "wireless",
    "gb", "mb", "inch", "ounce", "ips",
    "display", "color", "colour",
    "black", "white", "blue", "red", "silver", "gray", "grey",
    "green", "yellow", "tangerine", "orange", "purple", "pink",
    "kid", "proof", "kids",
})

_ACRONYMS = {"hd", "tv", "led", "lcd", "usb", "ssd"}


def _tokenize_product_name(name: str) -> list[str]:
    """Lowercase, alpha-only tokens of length ≥ 3, with stopwords removed."""
    cleaned = re.sub(r"[^a-z\s]", " ", str(name).lower())
    return [
        token for token in cleaned.split()
        if len(token) >= 3 and token not in CATEGORY_NAME_STOPWORDS
    ]


# Columns that carry useful text signal for naming; others (rating, sentiment) are skipped.
_TEXT_NAMING_COLUMNS = {"name", "categories", "reviews.text"}


def _build_naming_document(group_df: pd.DataFrame, naming_columns: list[str]) -> str:
    """Build one TF-IDF document for a cluster from the chosen naming columns.

    `name` and `categories` are tokenised cleanly.  `reviews.text` is included
    verbatim (already normalised upstream).  Numeric / label columns such as
    `reviews.rating` and `sentiment` are skipped — they carry no term signal.
    """
    tokens: list[str] = []
    effective = [c for c in naming_columns if c in _TEXT_NAMING_COLUMNS and c in group_df.columns]
    if not effective:
        effective = ["name"]  # safe fallback

    for col in effective:
        for value in group_df[col].astype(str).unique():
            cleaned = re.sub(r"[^a-z\s]", " ", value.lower())
            tokens.extend(
                t for t in cleaned.split()
                if len(t) >= 3 and t not in CATEGORY_NAME_STOPWORDS
            )
    return " ".join(tokens)


def _format_term(term: str) -> str:
    """Pretty-print a TF-IDF term — uppercase known acronyms, title-case the rest."""
    return term.upper() if term in _ACRONYMS else term.title()


def derive_category_names(
    df: pd.DataFrame,
    top_n_terms: int = 5,
    name_max_words: int = 3,
    naming_columns: list[str] | None = None,
) -> dict[int, dict]:
    """Label each cluster via TF-IDF over the chosen naming columns.

    Returns ``{category_id: {"name": str, "top_terms": [(term, weight), ...]}}``.
    Each cluster becomes one TF-IDF document built from ``naming_columns``
    (defaults to ``["name"]`` for backward compatibility). TF-IDF rewards terms
    frequent in this cluster but rare across others — the "what makes this
    cluster different" signal.  Numeric/label columns (rating, sentiment) are
    silently skipped; only text columns carry useful term signal.
    """
    if "category_id" not in df.columns or "name" not in df.columns:
        raise ValueError("derive_category_names requires 'category_id' and 'name' columns")

    cols = naming_columns or ["name"]

    docs_by_id: dict[int, str] = {}
    for cat_id, sub in df.groupby("category_id"):
        docs_by_id[int(cat_id)] = _build_naming_document(sub, cols)

    cat_ids = sorted(docs_by_id.keys())
    fallback = {cid: {"name": f"Category {cid + 1}", "top_terms": []} for cid in cat_ids}

    docs = [docs_by_id[cid] for cid in cat_ids]
    if len(docs) < 2 or all(not d.strip() for d in docs):
        return fallback

    try:
        vectorizer = TfidfVectorizer(token_pattern=r"\b[a-z]{3,}\b")
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return fallback

    feature_names = vectorizer.get_feature_names_out()
    result: dict[int, dict] = {}
    used_names: set[str] = set()
    for i, cid in enumerate(cat_ids):
        row = matrix[i].toarray().flatten()
        top_idx = row.argsort()[::-1][:top_n_terms]
        top_terms = [
            (feature_names[j], float(row[j])) for j in top_idx if row[j] > 0
        ]
        words = [_format_term(t) for t, _ in top_terms[:name_max_words]]
        name = " ".join(words) if words else f"Category {cid + 1}"
        # Make sure two clusters don't end up with the same label
        if name in used_names:
            name = f"{name} ({cid + 1})"
        used_names.add(name)
        result[cid] = {"name": name, "top_terms": top_terms}
    return result


def name_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``category_name`` replaced by TF-IDF-derived labels."""
    derived = derive_category_names(df)
    out = df.copy()
    out["category_name"] = out["category_id"].astype(int).map(
        lambda cid: derived[int(cid)]["name"]
    )
    return out


def _complaints_for_group(group: pd.DataFrame) -> list[str]:
    """Extract plain-English complaint themes from negative and neutral reviews."""
    unhappy = group[group["sentiment"].isin(["negative", "neutral"])]
    if unhappy.empty:
        return ["Very few repeated complaints showed up in this category."]

    counts: Counter[str] = Counter()
    text = " ".join(unhappy["reviews.text"].map(normalize_text).tolist())
    for term in COMPLAINT_TERMS:
        if term in text:
            counts[term] += text.count(term)

    if not counts:
        return ["Buyers were mixed, but no single complaint dominated the reviews."]

    return [f"{term} issues" for term, _ in counts.most_common(5)]


def aggregate_category_insights(df: pd.DataFrame) -> list[dict]:
    """Compute category metrics, top products, worst product, and complaints.

    The ranking formula from the brief is used exactly, with review_count normalized
    inside each category so large catalogs do not swamp rating quality.
    """
    required = {"name", "reviews.rating", "sentiment", "category_id", "category_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot aggregate insights; missing columns: {', '.join(sorted(missing))}")

    insights: list[dict] = []
    for (category_id, category_name), category_df in df.groupby(["category_id", "category_name"]):
        product_stats = (
            category_df.groupby("name")
            .agg(
                avg_rating=("reviews.rating", "mean"),
                review_count=("reviews.rating", "size"),
                positive_reviews=("sentiment", lambda s: int((s == "positive").sum())),
                neutral_reviews=("sentiment", lambda s: int((s == "neutral").sum())),
                negative_reviews=("sentiment", lambda s: int((s == "negative").sum())),
            )
            .reset_index()
        )
        product_stats["positive_ratio"] = product_stats["positive_reviews"] / product_stats["review_count"]
        max_count = max(float(product_stats["review_count"].max()), 1.0)
        normalized_count = product_stats["review_count"] / max_count
        product_stats["score"] = (
            RATING_WEIGHT * product_stats["avg_rating"]
            + POSITIVE_RATIO_WEIGHT * product_stats["positive_ratio"]
            + REVIEW_COUNT_WEIGHT * normalized_count
        )

        ranked = product_stats.sort_values(["score", "review_count"], ascending=False)
        worst = product_stats.sort_values(["avg_rating", "negative_reviews"], ascending=[True, False]).iloc[0]

        total_reviews = len(category_df)
        sentiment_counts = category_df["sentiment"].value_counts().to_dict()
        insight = {
            "category_id": int(category_id),
            "category_name": category_name,
            "avg_rating": round(float(category_df["reviews.rating"].mean()), 2),
            "review_count": int(total_reviews),
            "sentiment_ratio": {
                "positive": round(sentiment_counts.get("positive", 0) / total_reviews, 3),
                "neutral": round(sentiment_counts.get("neutral", 0) / total_reviews, 3),
                "negative": round(sentiment_counts.get("negative", 0) / total_reviews, 3),
            },
            "top_products": [
                {
                    "name": row["name"],
                    "avg_rating": round(float(row["avg_rating"]), 2),
                    "positive_ratio": round(float(row["positive_ratio"]), 3),
                    "review_count": int(row["review_count"]),
                    "score": round(float(row["score"]), 3),
                }
                for _, row in ranked.head(TOP_N_PRODUCTS).iterrows()
            ],
            "worst_product": {
                "name": worst["name"],
                "avg_rating": round(float(worst["avg_rating"]), 2),
                "negative_reviews": int(worst["negative_reviews"]),
                "review_count": int(worst["review_count"]),
            },
            "complaints": _complaints_for_group(category_df),
        }
        insights.append(insight)

    logger.info("Built insights for %s clustered categories", len(insights))
    return sorted(insights, key=lambda item: item["category_id"])
