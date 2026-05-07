from __future__ import annotations

import pandas as pd

from src.aggregation import aggregate_category_insights, derive_category_names
from src.preprocessing import fill_missing_categories, normalize_category_path
from src.sentiment import sentiment_from_rating


def test_sentiment_mapping() -> None:
    assert sentiment_from_rating(1) == "negative"
    assert sentiment_from_rating(2) == "negative"
    assert sentiment_from_rating(3) == "neutral"
    assert sentiment_from_rating(4) == "positive"
    assert sentiment_from_rating(5) == "positive"


def test_missing_categories_are_inferred() -> None:
    df = pd.DataFrame(
        {
            "name": ["AmazonBasics USB Cable", "Laptop Sleeve"],
            "reviews.text": ["charging cable works", "fits my notebook well"],
            "categories": [None, ""],
        }
    )

    filled = fill_missing_categories(df)

    assert filled["categories"].isna().sum() == 0
    assert filled.loc[0, "categories"] == "Cable"
    # infer_category returns the first matching rule for a clean single label
    # rather than compound paths like "Laptop > Case Sleeve Bag".
    assert filled.loc[1, "categories"] == "Laptop"


def test_category_paths_collapse_ambiguous_segments_to_empty() -> None:
    # "Computers & Tablets" matches both the computer and tablet keyword
    # families, so it's collapsed to "" and re-inferred from the review text.
    category = "Electronics,iPad & Tablets,All Tablets,Fire Tablets,Tablets,Computers & Tablets"

    assert normalize_category_path(category) == ""


def test_aggregation_outputs_top_worst_and_complaints() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "A", "B", "B", "C"],
            "reviews.text": [
                "great battery",
                "excellent charge",
                "slow setup",
                "broken remote",
                "bad battery",
            ],
            "reviews.rating": [5, 4, 2, 3, 1],
            "sentiment": ["positive", "positive", "negative", "neutral", "negative"],
            "category_id": [0, 0, 0, 0, 0],
            "category_name": ["Category 1"] * 5,
        }
    )

    insights = aggregate_category_insights(df)

    assert len(insights) == 1
    assert insights[0]["top_products"][0]["name"] == "A"
    assert len(insights[0]["top_products"]) == 3
    assert insights[0]["worst_product"]["name"] == "C"
    assert insights[0]["review_count"] == 5
    assert insights[0]["complaints"]


def test_category_names_filter_generic_source_terms() -> None:
    df = pd.DataFrame(
        {
            "name": [
                "Amazon Digital Remote Voice",
                "Amazon Digital Remote Voice",
                "Electronics Features Laptop",
                "Electronics Features Laptop",
            ],
            "reviews.text": [
                "fire tv remote voice control",
                "remote streaming tv stick",
                "laptop keyboard computer",
                "notebook laptop sleeve",
            ],
            "category_id": [0, 0, 1, 1],
            "categories": ["Electronics"] * 4,
        }
    )

    derived = derive_category_names(df)
    names = " ".join(item["name"] for item in derived.values()).lower()

    assert "electronics" not in names
    assert "digital" not in names
    assert "features" not in names
    assert "remote" in names
    assert "laptop" in names
