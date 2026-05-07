from __future__ import annotations

from src.summarization import RecommendationWriter, build_safe_prompt


def _insight() -> dict:
    return {
        "category_name": "Cable",
        "avg_rating": 4.4,
        "review_count": 72,
        "sentiment_ratio": {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
        "top_products": [
            {
                "name": "USB Cable",
                "avg_rating": 4.8,
                "positive_ratio": 0.9,
                "review_count": 30,
                "score": 4.7,
            }
        ],
        "worst_product": {
            "name": "Weak Charger",
            "avg_rating": 2.0,
            "negative_reviews": 4,
            "review_count": 10,
        },
        "complaints": ["charge issues"],
    }


def test_prompt_includes_selected_tone() -> None:
    prompt = build_safe_prompt(_insight(), tone="Technical comparison")

    assert "Tone: Technical comparison." in prompt
    assert "Raw reviews" not in prompt


def test_generate_passes_llm_settings(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_generator(prompt: str, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return [
            {
                "generated_text": (
                    "Best overall " + "word " * 170 + "Bottom line"
                )
            }
        ]

    monkeypatch.setattr("src.summarization.load_summarizer", lambda: fake_generator)

    article = RecommendationWriter().generate(
        _insight(),
        temperature=0.3,
        max_new_tokens=500,
        tone="Professional buying guide",
        use_fallback=False,
    )

    assert "Best overall" in article
    assert calls[0]["temperature"] == 0.3
    assert calls[0]["max_new_tokens"] == 500
    assert calls[0]["do_sample"] is True
    assert calls[0]["num_beams"] == 1
