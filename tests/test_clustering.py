from __future__ import annotations

import numpy as np

from src.clustering import ProductClusterer
from src.config import MAX_CLUSTERS, MIN_CLUSTERS


def test_choose_k_uses_required_range_when_enough_samples() -> None:
    embeddings = np.array(
        [[float(i), 0.0] for i in range(6)]
        + [[float(i), 10.0] for i in range(6)]
    )

    k = ProductClusterer().choose_k(embeddings)

    assert MIN_CLUSTERS <= k <= MAX_CLUSTERS


def test_choose_k_falls_back_for_tiny_inputs() -> None:
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    assert ProductClusterer().choose_k(embeddings) == 2
