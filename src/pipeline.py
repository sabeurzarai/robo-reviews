"""End-to-end RoboReviews pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

import pandas as pd

from src.aggregation import aggregate_category_insights, name_categories
from src.clustering import ProductClusterer
from src.config import OUTPUTS_DIR, RAW_DATA_DIR
from src.preprocessing import load_reviews_csv, load_reviews_dir
from src.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


class RoboReviewsPipeline:
    """Coordinates preprocessing, sentiment, clustering, and aggregation."""

    def __init__(self) -> None:
        self.sentiment = SentimentAnalyzer()
        self.clusterer = ProductClusterer()

    def _run(self, df: pd.DataFrame, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        enriched = self.sentiment.label_dataframe(df)
        clustered = self.clusterer.cluster(enriched)
        clustered = name_categories(clustered)
        insights = aggregate_category_insights(clustered)

        clustered_file = output_dir / "clustered_reviews.csv"
        insights_file = output_dir / "category_insights.json"

        clustered.to_csv(clustered_file, index=False)
        insights_file.write_text(json.dumps(insights, indent=2), encoding="utf-8")

        logger.info("Saved clustered reviews to %s", clustered_file)
        logger.info("Saved category insights to %s", insights_file)

        return {
            "clustered_reviews_path": str(clustered_file),
            "category_insights_path": str(insights_file),
            "insights": insights,
        }

    def run_from_csv(self, csv_path: str | Path, output_dir: str | Path = OUTPUTS_DIR) -> dict:
        """Run the analytics pipeline against a single CSV file."""
        return self._run(load_reviews_csv(csv_path), Path(output_dir))

    def run_from_dir(
        self,
        data_dir: str | Path = RAW_DATA_DIR,
        output_dir: str | Path = OUTPUTS_DIR,
    ) -> dict:
        """Merge every review CSV in ``data_dir`` and run the pipeline against the result."""
        return self._run(load_reviews_dir(data_dir), Path(output_dir))
