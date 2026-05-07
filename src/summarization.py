"""Recommendation article generation."""
from __future__ import annotations

import logging
from functools import lru_cache

from transformers import pipeline

from src.config import SUMMARY_MODEL_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_summarizer():
    """Load FLAN-T5 lazily; this keeps tests and health checks lightweight."""
    logger.info("Loading summary model: %s", SUMMARY_MODEL_NAME)
    return pipeline(
        "text2text-generation",
        model=SUMMARY_MODEL_NAME,
    )


def build_safe_prompt(category_insight: dict, tone: str = "Professional buying guide") -> str:
    """Build an LLM prompt from structured insights only.

    Raw reviews are intentionally excluded. This reduces privacy risk and gives the
    model the clean facts it needs without inviting it to quote customers.
    """
    top = "\n".join(
        f"- {p['name']} | rating {p['avg_rating']} | positive ratio {p['positive_ratio']} | reviews {p['review_count']} | score {p['score']}"
        for p in category_insight["top_products"]
    )
    complaints = ", ".join(category_insight["complaints"])
    worst = category_insight["worst_product"]

    return f"""
Write a helpful, natural product recommendation article.
Tone: {tone}.

Use only these structured facts. Do not mention raw customer reviews or invent test results.

Category: {category_insight['category_name']}
Average rating: {category_insight['avg_rating']}
Review count: {category_insight['review_count']}
Sentiment ratio: {category_insight['sentiment_ratio']}
Top products:
{top}
Worst product:
- {worst['name']} | rating {worst['avg_rating']} | negative reviews {worst['negative_reviews']} | reviews {worst['review_count']}
Common complaints: {complaints}

Format:
- Clear headline
- Best overall
- Also worth considering
- What to watch out for
- Bottom line
""".strip()


class RecommendationWriter:
    """Creates category recommendation articles from aggregated facts."""

    def generate(
        self,
        category_insight: dict,
        temperature: float = 0.3,
        max_new_tokens: int = 450,
        tone: str = "Professional buying guide",
        use_fallback: bool = True,
    ) -> str:
        """Generate a polished article, falling back gracefully if the model is unavailable.

        FLAN-T5-base is small and frequently drops content under beam search with
        repetition penalties, producing fluent-but-incomplete articles. The
        quality gate below is intentionally strict: the deterministic fallback
        is itself a Wirecutter-style article built from the same structured
        facts, so falling back never hurts the user.
        """
        prompt = build_safe_prompt(category_insight, tone=tone)
        required_markers = ("Best overall", "Bottom line")
        try:
            generator = load_summarizer()
            safe_temperature = max(0.0, min(float(temperature), 1.0))
            generation_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.5,
                "early_stopping": True,
            }
            if safe_temperature > 0:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": safe_temperature,
                        "num_beams": 1,
                    }
                )
            else:
                generation_kwargs.update({"do_sample": False, "num_beams": 4})

            result = generator(prompt, **generation_kwargs)[0]["generated_text"].strip()
            words = result.split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            has_structure = all(marker.lower() in result.lower() for marker in required_markers)
            if len(words) >= 150 and unique_ratio >= 0.35 and has_structure:
                return result
            logger.warning(
                "LLM output failed quality check (words=%d, unique_ratio=%.2f, structure=%s); using deterministic fallback",
                len(words), unique_ratio, has_structure,
            )
            if not use_fallback and result:
                return result
        except Exception as exc:  # pragma: no cover - depends on model availability
            logger.warning("Summary model unavailable, using deterministic article fallback: %s", exc)
            if not use_fallback:
                raise

        return self._fallback_article(category_insight)

    def _fallback_article(self, insight: dict) -> str:
        """A readable backup keeps the product useful even without model downloads."""
        top = insight["top_products"]
        best = top[0]
        alternatives = top[1:]
        alt_text = "\n".join(
            f"- **{item['name']}** is a strong backup pick, with a {item['avg_rating']} average rating and {item['review_count']} reviews."
            for item in alternatives
        ) or "- There was not enough depth in this category to recommend a second pick confidently."

        complaints = ", ".join(insight["complaints"])
        worst = insight["worst_product"]

        return f"""# Best picks in {insight['category_name']}

## Best overall: {best['name']}

{best['name']} rises to the top because it balances a strong average rating of {best['avg_rating']} with a healthy positive sentiment ratio of {best['positive_ratio']}. It also has enough review volume to make the signal more trustworthy than a one-off favorite.

## Also worth considering

{alt_text}

## What to watch out for

The most common concerns in this category are {complaints}. That does not mean every buyer will run into those problems, but they are the issues worth checking before you buy.

## Product to be cautious with

{worst['name']} had the weakest rating profile in this cluster, averaging {worst['avg_rating']} across {worst['review_count']} reviews.

## Bottom line

Start with {best['name']} if you want the safest pick. Look at the runner-up options when price, availability, or a specific feature matters more than the overall score."""
