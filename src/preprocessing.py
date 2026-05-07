"""Dataset loading and cleaning for RoboReviews."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

CATEGORY_INFERENCE_RULES = [
    ("Laptop", ("laptop", "notebook", "macbook", "chromebook")),
    ("Tablet", ("tablet", "kindle", "fire hd", "ipad")),
    ("Cable", ("cable", "charger", "charging", "usb", "adapter", "cord", "hdmi")),
    ("Case Sleeve Bag", ("case", "sleeve", "bag", "cover", "protector", "stand")),
    ("Speaker Audio", ("speaker", "echo", "alexa", "audio", "sound", "dock")),
    ("Remote Streaming TV", ("remote", "streaming", "fire tv", "roku", "tv stick")),
    ("Headphones", ("headphone", "earbud", "earphone", "headset")),
    ("Battery Power", ("battery", "power bank", "powerbank")),
    ("Camera", ("camera", "webcam", "photo", "video")),
    ("Keyboard Mouse", ("keyboard", "mouse", "trackpad")),
]

# Category values that are retailers, promotions, or connectivity specs — not product types.
# These are blanked so infer_category can make a better call from the review text.
_CATEGORY_BLOCKLIST: frozenset[str] = frozenset({
    "frys",
    "amazon",
    "holiday shop",
    "health personal care",
    "digital device 3",
    "voice-enabled smart assistants",
    "e-readers",
})

# Prefixes that identify connectivity / device-spec strings masquerading as categories.
_CATEGORY_BLOCKLIST_PREFIXES: tuple[str, ...] = ("wi-fi", "3g", "4g", "lte")

# Keyword → canonical label for collapsing category path segments.
# Order matters for readability only; conflict detection uses the full set.
_CATEGORY_COLLAPSE_KEYWORDS: list[tuple[str, str]] = [
    ("kindle",     "Tablet"),
    ("ipad",       "Tablet"),
    ("tablet",     "Tablet"),
    ("laptop",     "Laptop"),
    ("notebook",   "Laptop"),
    ("chromebook", "Laptop"),
    ("computer",   "Computer"),
    ("speaker",    "Speaker"),
    ("echo",       "Speaker"),
    ("headphone",  "Headphones"),
    ("earbud",     "Headphones"),
    ("earphone",   "Headphones"),
    ("camera",     "Camera"),
    ("keyboard",   "Keyboard"),
    ("cable",      "Cable"),
    ("charger",    "Cable"),
    ("adapter",    "Cable"),
    ("battery",    "Battery Power"),
    ("power bank", "Battery Power"),
]


def normalize_text(text: str) -> str:
    """Clean review text without destroying useful wording.

    We keep punctuation light rather than over-sanitizing because product complaints
    often depend on phrases like "won't charge" or "too slow".
    """
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_category(name: str, review_text: str) -> str:
    """Infer a single category label from product name and review text.

    Returns the first (highest-priority) rule match so the result is always
    a clean single label, not a compound 'Tablet > Cable' string.
    """
    combined = normalize_text(f"{name} {review_text}")
    for label, keywords in CATEGORY_INFERENCE_RULES:
        if any(keyword in combined for keyword in keywords):
            return label
    return "Uncategorized"


def _collapse_category(segment: str) -> str:
    """Collapse a single category segment to a canonical label.

    Returns:
    - canonical label  — exactly one keyword family matched
    - ""               — ambiguous (2+ families) OR blocklisted junk
    - original segment — no keyword matched and not blocklisted
    """
    lower = segment.lower()
    if lower in _CATEGORY_BLOCKLIST or any(lower.startswith(p) for p in _CATEGORY_BLOCKLIST_PREFIXES):
        return ""
    matched = {canonical for kw, canonical in _CATEGORY_COLLAPSE_KEYWORDS if kw in lower}
    if len(matched) == 1:
        return matched.pop()
    if len(matched) > 1:
        return ""  # ambiguous
    return segment


def normalize_category_path(category: str) -> str:
    """Extract the last segment of a comma-separated path then collapse to a canonical label."""
    parts = [p.strip() for p in str(category).split(",") if p.strip()]
    last = parts[-1] if parts else ""
    return _collapse_category(last) if last else ""


def fill_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise category paths and fill blank or ambiguous cells from product name and review text.

    Order matters: normalise first so ambiguous paths (e.g. "Computers & Tablets")
    are blanked and then re-inferred from the review, just like originally-blank rows.
    """
    out = df.copy()
    out["categories"] = out["categories"].map(normalize_category_path)
    needs_inference = out["categories"].isna() | (out["categories"].astype(str).str.strip() == "")
    if needs_inference.any():
        out.loc[needs_inference, "categories"] = out.loc[needs_inference].apply(
            lambda row: infer_category(row["name"], row["reviews.text"]),
            axis=1,
        )
    return out


def category_normalization_report(df: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    """Describe what fill_missing_categories will do to each row without modifying df.

    Returns a (stats, sample) tuple:
    - stats: counts for blank_inferred, ambiguous_inferred, collapsed, kept
    - sample: up to 12 representative rows showing the change (collapsed + ambiguous only)
    """
    import numpy as np  # local to avoid circular-ish top-level noise

    cats = df["categories"].fillna("").astype(str).str.strip()
    normed = cats.map(normalize_category_path)
    last_segment = cats.str.split(",").str[-1].str.strip()
    last_lower = last_segment.str.lower()

    blank_mask      = cats == ""
    blocklist_mask  = (~blank_mask) & (normed == "") & (
        last_lower.isin(_CATEGORY_BLOCKLIST)
        | last_lower.apply(lambda s: any(s.startswith(p) for p in _CATEGORY_BLOCKLIST_PREFIXES))
    )
    ambiguous_mask  = (~blank_mask) & (~blocklist_mask) & (normed == "")
    collapsed_mask  = (~blank_mask) & (normed != "") & (normed != last_segment)
    kept_mask       = (~blank_mask) & (normed != "") & (normed == last_segment)

    stats: dict[str, int] = {
        "blank_inferred":     int(blank_mask.sum()),
        "blocklist_inferred": int(blocklist_mask.sum()),
        "ambiguous_inferred": int(ambiguous_mask.sum()),
        "collapsed":          int(collapsed_mask.sum()),
        "kept":               int(kept_mask.sum()),
    }

    interesting = df.index[collapsed_mask | ambiguous_mask | blocklist_mask][:12]
    if len(interesting):
        sample = df.loc[interesting, ["name", "categories"]].copy()
        sample["normalized_to"] = normed[interesting]
        sample["action"] = [
            "collapsed" if collapsed_mask[i]
            else "junk → inferred from review" if blocklist_mask[i]
            else "ambiguous → inferred from review"
            for i in interesting
        ]
    else:
        sample = pd.DataFrame(columns=["name", "categories", "normalized_to", "action"])

    return stats, sample


def validate_columns(columns: Iterable[str]) -> None:
    """Fail fast when the input file is not the expected Datafiniti dataset shape."""
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def _clean_review_frame(df: pd.DataFrame, extra_keep: list[str] | None = None) -> pd.DataFrame:
    """Apply the standard cleaning rules to a raw review DataFrame."""
    keep = REQUIRED_COLUMNS + (extra_keep or [])
    df = df[keep].copy()
    df = df.dropna(subset=["name", "reviews.text", "reviews.rating"])
    df = fill_missing_categories(df)
    df = df.drop_duplicates(subset=["name", "reviews.text", "reviews.rating"])
    df["name"] = df["name"].astype(str).str.strip()
    df["reviews.text"] = df["reviews.text"].map(normalize_text)
    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")
    df = df.dropna(subset=["reviews.rating"])
    df = df[df["reviews.rating"].between(1, 5)]
    return df.reset_index(drop=True)


def load_reviews_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load, validate, and clean a single review CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset: {path}")

    logger.info("Loading review data from %s", path)
    df = pd.read_csv(path)
    validate_columns(df.columns)

    df = _clean_review_frame(df)

    # Original categories are intentionally not used for clustering. We keep the
    # column only so the expected source schema remains explicit.
    logger.info("Prepared %s clean reviews", len(df))
    return df


def load_reviews_dir(data_dir: str | Path) -> pd.DataFrame:
    """Merge every review CSV in a directory.

    Files missing the required columns (e.g. ``submissions.csv``) are skipped
    with a warning rather than failing the whole run. The resulting DataFrame
    carries a ``source_file`` column so each row can be traced back to its
    origin, and duplicates across files are removed on the same key as
    ``load_reviews_csv``.
    """
    path = Path(data_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Could not find data directory: {path}")

    csv_files = sorted(p for p in path.glob("*.csv") if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for csv_path in csv_files:
        logger.info("Loading review data from %s", csv_path)
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            validate_columns(df.columns)
        except (ValueError, pd.errors.ParserError) as exc:
            logger.warning("Skipping %s: %s", csv_path.name, exc)
            skipped.append(csv_path.name)
            continue
        df = df[REQUIRED_COLUMNS].copy()
        df["source_file"] = csv_path.name
        frames.append(df)

    if not frames:
        raise ValueError(
            f"No CSV files in {path} match the expected review schema. "
            f"Skipped: {', '.join(skipped) or 'none'}"
        )

    merged_raw = pd.concat(frames, ignore_index=True)
    merged = _clean_review_frame(merged_raw, extra_keep=["source_file"])
    logger.info(
        "Merged %s file(s) into %s clean reviews (skipped: %s)",
        len(frames),
        len(merged),
        ", ".join(skipped) or "none",
    )
    return merged


def dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    """Create a clean DataFrame from API records using the same rules as CSV input."""
    df = pd.DataFrame(records)
    validate_columns(df.columns)
    return _clean_review_frame(df)
