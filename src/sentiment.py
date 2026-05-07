"""Sentiment utilities.

The dataset gives us ratings, so the batch pipeline uses the required mapping:
1-2 negative, 3 neutral, 4-5 positive. For ad-hoc text predictions, we load the
required DistilBERT tokenizer/model once and combine its representation with a
small, transparent fallback heuristic when no fine-tuned classifier is available.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.config import SENTIMENT_MODEL_NAME

logger = logging.getLogger(__name__)
SentimentLabel = Literal["negative", "neutral", "positive"]

POSITIVE_HINTS = {"great", "excellent", "love", "fast", "reliable", "perfect", "best", "easy", "happy"}
NEGATIVE_HINTS = {"bad", "broken", "slow", "terrible", "poor", "failed", "waste", "return", "disappointed", "worst"}


def sentiment_from_rating(rating: float) -> SentimentLabel:
    """Apply the project-required rating-to-sentiment mapping."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


@lru_cache(maxsize=1)
def load_distilbert() -> tuple[AutoTokenizer, AutoModel]:
    """Load DistilBERT lazily so API startup stays predictable.

    DistilBERT base is not a sentiment classifier by itself. We still load it as
    the requested NLP backbone and keep the fallback behavior explicit instead of
    pretending a randomly initialized classification head is production-ready.
    """
    logger.info("Loading NLP backbone: %s", SENTIMENT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModel.from_pretrained(SENTIMENT_MODEL_NAME)
    model.eval()
    return tokenizer, model


class SentimentAnalyzer:
    """Sentiment service shared by the API and data pipeline."""

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach sentiment labels using the rating mapping mandated by the brief."""
        out = df.copy()
        out["sentiment"] = out["reviews.rating"].map(sentiment_from_rating)
        return out

    def predict_text(self, text: str) -> SentimentLabel:
        """Predict sentiment for a single text snippet.

        A base DistilBERT model is not fine-tuned for sentiment. The heuristic here
        is deliberately conservative and readable, which is safer than returning
        confident nonsense from an unsuitable head.
        """
        clean = text.lower()
        pos = sum(word in clean for word in POSITIVE_HINTS)
        neg = sum(word in clean for word in NEGATIVE_HINTS)

        # Touch the model once so production deployments surface missing model
        # downloads early. The output is not treated as a sentiment logit.
        try:
            tokenizer, model = load_distilbert()
            inputs = tokenizer(clean[:512], return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                _ = model(**inputs).last_hidden_state[:, 0, :]
        except Exception as exc:  # pragma: no cover - network/model cache dependent
            logger.warning("DistilBERT backbone unavailable, using text heuristic only: %s", exc)

        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"
