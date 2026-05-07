"""Core RoboReviews package.

Bootstraps the HuggingFace model cache to point at the project's `models/`
directory *before* any transformers / sentence-transformers code is imported.
This way downloaded weights persist across runs in a project-controlled
directory (rather than the user's global ~/.cache/huggingface/), and Docker
volume mounts on `models/` keep them across container restarts.

Users can override by setting `HF_HOME` (or `SENTENCE_TRANSFORMERS_HOME`)
externally before launching — `os.environ.setdefault` is a no-op in that case.
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(os.getenv("ROBO_REVIEWS_ROOT", Path(__file__).resolve().parents[1]))
_MODELS_DIR = Path(os.getenv("ROBO_REVIEWS_MODELS_DIR", _PROJECT_ROOT / "models"))
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(_MODELS_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_MODELS_DIR))
