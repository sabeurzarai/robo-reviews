"""Product clustering based on review text embeddings."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import EMBEDDING_MODEL_NAME, MAX_CLUSTERS, MIN_CLUSTERS

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


class ProductClusterer:
    """Clusters products from product-level text, never from the original category column."""

    def __init__(self, min_clusters: int = MIN_CLUSTERS, max_clusters: int = MAX_CLUSTERS) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def embed_reviews(self, texts: list[str]) -> np.ndarray:
        """Create dense review embeddings."""
        model = load_embedding_model()
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def choose_k(self, embeddings: np.ndarray) -> int:
        """Pick a cluster count using silhouette score when possible."""
        n_samples = len(embeddings)
        min_k = min(2, n_samples)
        if n_samples < min_k + 1:
            return 1

        best_k = min_k
        best_score = -1.0
        for k in range(min_k, min(self.max_clusters, n_samples - 1) + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        logger.info("Selected k=%s for product clustering", best_k)
        return best_k

    def build_product_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create one clustering document per distinct product.

        These documents use only the product name plus the source category text,
        then the resulting product-level labels are mapped back to every review.
        """
        rows: list[dict[str, str | int]] = []
        for product_name, product_df in df.groupby("name", sort=True):
            category_text = (
                product_df["categories"]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            document = " ".join(
                [
                    f"Product name: {product_name}.",
                    f"Product name tokens: {product_name}. {product_name}.",
                    "Source categories:",
                    " ".join(category_text),
                ]
            )
            rows.append(
                {
                    "name": str(product_name),
                    "review_count": int(len(product_df)),
                    "categories": " | ".join(category_text),
                    "cluster_document": document,
                }
            )
        return pd.DataFrame(rows)

    def cluster(self, df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        """Add category_id labels to reviews."""
        if df.empty:
            raise ValueError("Cannot cluster an empty dataset")

        product_docs = self.build_product_documents(df)
        embeddings = self.embed_reviews(product_docs["cluster_document"].astype(str).tolist())
        n_clusters = k or self.choose_k(embeddings)
        n_clusters = max(1, min(n_clusters, len(product_docs)))

        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(embeddings)
        product_docs = product_docs.copy()
        product_docs["category_id"] = labels.astype(int)
        product_label_map = dict(zip(product_docs["name"], product_docs["category_id"]))

        out = df.copy()
        out["category_id"] = out["name"].astype(str).map(product_label_map).astype(int)
        out["category_name"] = out["category_id"].map(lambda value: f"Category {value + 1}")
        return out
