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
        """Pick a cluster count using silhouette score within the brief's [min, max] range.

        The project brief requires 4-10 meta categories, so the search is bounded
        by ``self.min_clusters`` and ``self.max_clusters`` rather than starting at 2.
        Silhouette is computed only within that range.
        """
        n_samples = len(embeddings)
        if n_samples < self.min_clusters + 1:
            return max(1, n_samples - 1)

        lo = self.min_clusters
        hi = min(self.max_clusters, n_samples - 1)
        if lo > hi:
            return hi

        best_k = lo
        best_score = -1.0
        for k in range(lo, hi + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        logger.info("Selected k=%s for product clustering (silhouette=%.3f)", best_k, best_score)
        return best_k

    def build_product_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create one clustering document per distinct product.

        Each document combines the product name with a sample of its review text.
        Reviews carry the strongest semantic signal — what the product *is* and
        what people use it for — so clusters reflect product behaviour rather
        than the upstream `categories` label that preprocessing already collapsed.
        """
        rows: list[dict[str, str | int]] = []
        for product_name, product_df in df.groupby("name", sort=True):
            sample_reviews = (
                product_df["reviews.text"]
                .astype(str)
                .head(10)
                .tolist()
            )
            review_excerpt = " ".join(sample_reviews)[:1500]
            document = f"Product: {product_name}. Reviews: {review_excerpt}"
            rows.append(
                {
                    "name": str(product_name),
                    "review_count": int(len(product_df)),
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
