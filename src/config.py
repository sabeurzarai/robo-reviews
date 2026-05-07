"""Project settings.

The values live in one place so the API, Streamlit app, and tests do not drift apart.
Environment variables can override paths in Docker or on an EC2 host.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("ROBO_REVIEWS_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = Path(os.getenv("ROBO_REVIEWS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = Path(os.getenv("ROBO_REVIEWS_RAW_DATA_DIR", DATA_DIR / "raw"))
OUTPUTS_DIR = Path(os.getenv("ROBO_REVIEWS_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
MODELS_DIR = Path(os.getenv("ROBO_REVIEWS_MODELS_DIR", PROJECT_ROOT / "models"))

BLOGPOSTS_DIR = OUTPUTS_DIR / "blogposts"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

PRIMARY_DATASET = "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
REQUIRED_COLUMNS = ["name", "reviews.text", "reviews.rating", "categories"]

SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARY_MODEL_NAME = "google/flan-t5-base"

# SST-2 is binary; we treat low-confidence model outputs as "neutral" so the
# 3-class evaluation against rating-derived labels is meaningful.
NEUTRAL_CONFIDENCE_THRESHOLD = 0.85

MIN_CLUSTERS = 4
MAX_CLUSTERS = 10
DEFAULT_CLUSTERS = 6
TOP_N_PRODUCTS = 3

# Review count can be much larger than rating/ratio, so we normalize it before scoring.
RATING_WEIGHT = 0.5
POSITIVE_RATIO_WEIGHT = 0.3
REVIEW_COUNT_WEIGHT = 0.2
