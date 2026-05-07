"""Sentiment utilities.

Two label sources live here side by side:

1. ``sentiment_from_rating`` — the project-required rating mapping (1-2 negative,
   3 neutral, 4-5 positive). Used as **ground truth** during evaluation and as the
   batch label for downstream aggregation, since reviewers themselves picked the
   rating.

2. ``SentimentAnalyzer.predict_text`` / ``predict_texts`` — a real fine-tuned
   DistilBERT classifier (``distilbert-base-uncased-finetuned-sst-2-english``)
   running on review text. SST-2 is binary; we collapse low-confidence outputs
   to ``neutral`` so the 3-class API contract holds.

Keeping rating labels as ground truth and model labels as predictions lets the
evaluation notebook compute precision/recall/F1 honestly — without leaking the
rating into the feature.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Literal

import pandas as pd
import torch
from transformers import pipeline

from src.config import NEUTRAL_CONFIDENCE_THRESHOLD, SENTIMENT_MODEL_NAME

logger = logging.getLogger(__name__)
SentimentLabel = Literal["negative", "neutral", "positive"]

_MAX_LENGTH = 256
_BATCH_SIZE = 32


def sentiment_from_rating(rating: float) -> SentimentLabel:
    """Apply the project-required rating-to-sentiment mapping."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


@lru_cache(maxsize=1)
def load_sentiment_pipeline():
    """Load the fine-tuned DistilBERT SST-2 classifier once per process."""
    logger.info("Loading sentiment classifier: %s", SENTIMENT_MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
        truncation=True,
        max_length=_MAX_LENGTH,
        device=device,
    )


def _label_from_pipeline_output(out: dict) -> SentimentLabel:
    """Map a HuggingFace SST-2 result to one of {negative, neutral, positive}.

    SST-2 returns ``POSITIVE``/``NEGATIVE``. We collapse predictions whose
    confidence is below ``NEUTRAL_CONFIDENCE_THRESHOLD`` to ``neutral`` — the
    classifier is most uncertain on ambivalent / mixed reviews, which is exactly
    where the rating-based ground truth tends to land at 3 stars.
    """
    label = str(out["label"]).upper()
    score = float(out["score"])
    if score < NEUTRAL_CONFIDENCE_THRESHOLD:
        return "neutral"
    return "positive" if label == "POSITIVE" else "negative"


class SentimentAnalyzer:
    """Sentiment service shared by the API and data pipeline."""

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach sentiment labels using the rating mapping mandated by the brief.

        This is the **batch** label used by aggregation. It is also the ground
        truth that ``predict_dataframe`` is evaluated against.
        """
        out = df.copy()
        out["sentiment"] = out["reviews.rating"].map(sentiment_from_rating)
        return out

    def predict_text(self, text: str) -> SentimentLabel:
        """Predict sentiment for a single text snippet using DistilBERT SST-2."""
        clf = load_sentiment_pipeline()
        result = clf(str(text)[: _MAX_LENGTH * 4])[0]
        return _label_from_pipeline_output(result)

    def predict_texts(self, texts: Iterable[str]) -> list[SentimentLabel]:
        """Batch-predict sentiment for evaluation.

        Returns a list aligned with ``texts``. Used by the evaluation notebook to
        score the classifier against rating-derived ground truth.
        """
        clf = load_sentiment_pipeline()
        cleaned = [str(t)[: _MAX_LENGTH * 4] for t in texts]
        results = clf(cleaned, batch_size=_BATCH_SIZE)
        return [_label_from_pipeline_output(r) for r in results]

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "reviews.text") -> pd.DataFrame:
        """Add a ``predicted_sentiment`` column from the model's output."""
        out = df.copy()
        out["predicted_sentiment"] = self.predict_texts(out[text_column].astype(str).tolist())
        return out
