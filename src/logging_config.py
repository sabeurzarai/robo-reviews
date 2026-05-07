"""Small logging helper used by every entrypoint."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging once.

    Uvicorn and Streamlit already add handlers in some situations, so this guard
    keeps local runs from printing every line twice.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
