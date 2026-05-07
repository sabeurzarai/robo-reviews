"""Streamlit demo UI for RoboReviews."""
from __future__ import annotations

import datetime
import io
import json
import os
import re
import time
import hashlib
import shutil
from pathlib import Path

import altair as alt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.aggregation import aggregate_category_insights, derive_category_names
from src.clustering import ProductClusterer
from src.config import (
    BLOGPOSTS_DIR,
    DEFAULT_CLUSTERS,
    EMBEDDING_MODEL_NAME,
    FIGURES_DIR,
    MAX_CLUSTERS,
    METRICS_DIR,
    MIN_CLUSTERS,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REQUIRED_COLUMNS,
)
from src.preprocessing import category_normalization_report, fill_missing_categories, infer_category, load_reviews_dir, normalize_text, validate_columns
from src.sentiment import SentimentAnalyzer
from src.summarization import RecommendationWriter, build_safe_prompt

st.set_page_config(page_title="RoboReviews", page_icon="RR", layout="wide")

DEFAULT_API_URL = os.getenv("ROBO_REVIEWS_API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("API URL", DEFAULT_API_URL)


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


st.sidebar.markdown("### Local files")
st.sidebar.write(
    "Drop compatible Amazon review CSVs into `data/raw/`; compatible files are "
    "merged automatically and incompatible schemas are skipped. Blank `categories` "
    "cells are filled during preprocessing from product name and review wording."
)
st.sidebar.markdown("### Model cache")
try:
    cache_label = MODELS_DIR.relative_to(PROJECT_ROOT)
except ValueError:
    cache_label = MODELS_DIR
st.sidebar.caption(f"`{cache_label}` - {_dir_size_mb(MODELS_DIR):,.0f} MB on disk")

_title_col, _restart_col = st.columns([8, 1])
_title_col.title("RoboReviews")
_title_col.caption("Upload reviews, discover product clusters, and generate practical buying advice.")
if _restart_col.button("Restart", type="secondary", use_container_width=True, help="Reset all parameters and pipeline results to defaults"):
    _new_counter = st.session_state.get("_reset_counter", 0) + 1
    st.session_state.clear()
    st.session_state["_reset_counter"] = _new_counter
    st.rerun()

st.session_state.setdefault("_reset_counter", 0)
st.session_state.setdefault("_last_run_ts", None)
st.session_state.setdefault("clustered_df", None)
st.session_state.setdefault("insights", None)
st.session_state.setdefault("articles", {})
st.session_state.setdefault("playground_text", "")
st.session_state.setdefault("heuristic_eval", None)
st.session_state.setdefault("stage_summary", [])

_rc = st.session_state["_reset_counter"]


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@st.cache_resource(show_spinner=False)
def get_clusterer() -> ProductClusterer:
    return ProductClusterer()


@st.cache_resource(show_spinner=False)
def get_writer() -> RecommendationWriter:
    return RecommendationWriter()


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "category"


def _clean_uploaded_frame(raw: pd.DataFrame) -> pd.DataFrame:
    validate_columns(raw.columns)
    df = raw[REQUIRED_COLUMNS + (["source_file"] if "source_file" in raw.columns else [])].copy()
    df = df.dropna(subset=["name", "reviews.text", "reviews.rating"])
    df = fill_missing_categories(df)
    df = df.drop_duplicates(subset=["name", "reviews.text", "reviews.rating"])
    df["name"] = df["name"].astype(str).str.strip()
    df["reviews.text"] = df["reviews.text"].map(normalize_text)
    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")
    df = df.dropna(subset=["reviews.rating"])
    df = df[df["reviews.rating"].between(1, 5)]
    return df.reset_index(drop=True)


def _load_from_disk() -> None:
    insights_path = OUTPUTS_DIR / "category_insights.json"
    clustered_path = OUTPUTS_DIR / "clustered_reviews.csv"
    if insights_path.exists():
        st.session_state.insights = json.loads(insights_path.read_text(encoding="utf-8"))
    if clustered_path.exists():
        st.session_state.clustered_df = pd.read_csv(clustered_path)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(
            f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
        )
        df.to_csv(fallback, index=False)
        return fallback


def _clear_pipeline_cache() -> int:
    cache_dir = OUTPUTS_DIR / "cache"
    if not cache_dir.exists():
        return 0
    file_count = sum(1 for path in cache_dir.rglob("*") if path.is_file())
    shutil.rmtree(cache_dir)
    return file_count


def _embedding_cache_key(texts: list[str], namespace: str = "reviews") -> str:
    hasher = hashlib.sha256()
    hasher.update(namespace.encode("utf-8"))
    hasher.update(EMBEDDING_MODEL_NAME.encode("utf-8"))
    hasher.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        hasher.update(b"\0")
        hasher.update(text.encode("utf-8", errors="replace"))
    return hasher.hexdigest()[:16]


def _embedding_cache_paths(cache_key: str) -> tuple[Path, Path]:
    cache_dir = OUTPUTS_DIR / "cache"
    return (
        cache_dir / f"review_embeddings_{cache_key}.npz",
        cache_dir / f"review_embeddings_{cache_key}.json",
    )


def _load_cached_embeddings(texts: list[str], namespace: str = "reviews") -> tuple[np.ndarray | None, Path, str]:
    cache_key = _embedding_cache_key(texts, namespace=namespace)
    embeddings_path, _ = _embedding_cache_paths(cache_key)
    if not embeddings_path.exists():
        return None, embeddings_path, cache_key

    try:
        with np.load(embeddings_path) as cached:
            embeddings = cached["embeddings"]
    except (OSError, KeyError, ValueError):
        return None, embeddings_path, cache_key

    if len(embeddings) != len(texts):
        return None, embeddings_path, cache_key
    return embeddings, embeddings_path, cache_key


def _build_tfidf_features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9&-]{2,}\b",
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer.get_feature_names_out().tolist()


def _save_cached_embeddings(
    embeddings: np.ndarray,
    embeddings_path: Path,
    cache_key: str,
    row_count: int,
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, embeddings=embeddings)
    _, metadata_path = _embedding_cache_paths(cache_key)
    _write_json(
        metadata_path,
        {
            "cache_key": cache_key,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "row_count": row_count,
            "embedding_shape": list(embeddings.shape),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def _clustering_cache_paths(cache_key: str) -> tuple[Path, Path]:
    cache_dir = OUTPUTS_DIR / "cache"
    return (
        cache_dir / f"review_clusters_{cache_key}.npz",
        cache_dir / f"review_clusters_{cache_key}.json",
    )


def _load_cached_clustering(
    cache_key: str,
    row_count: int,
) -> tuple[np.ndarray | None, int | None, dict[int, float], Path]:
    labels_path, metadata_path = _clustering_cache_paths(cache_key)
    if not labels_path.exists() or not metadata_path.exists():
        return None, None, {}, labels_path

    try:
        with np.load(labels_path) as cached:
            labels = cached["labels"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None, None, {}, labels_path

    if len(labels) != row_count or metadata.get("row_count") != row_count:
        return None, None, {}, labels_path
    scores = {
        int(k): float(v)
        for k, v in metadata.get("silhouette_scores", {}).items()
    }
    return labels.astype(int), int(metadata["best_k"]), scores, labels_path


def _save_cached_clustering(
    labels: np.ndarray,
    best_k: int,
    scores: dict[int, float],
    cache_key: str,
    row_count: int,
    cluster_sizes: dict[str, int],
) -> Path:
    labels_path, metadata_path = _clustering_cache_paths(cache_key)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(labels_path, labels=labels.astype(int))
    _write_json(
        metadata_path,
        {
            "cache_key": cache_key,
            "row_count": row_count,
            "best_k": best_k,
            "min_clusters": 2,
            "max_clusters": MAX_CLUSTERS,
            "silhouette_scores": {str(k): v for k, v in scores.items()},
            "cluster_sizes": cluster_sizes,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return labels_path


def _build_product_documents(df: pd.DataFrame, selected_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for product_name, product_df in df.groupby("name", sort=True):
        category_text = (
            product_df["categories"]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        document_parts: list[str] = []
        if "name" in selected_columns:
            document_parts.extend(
                [
                    f"Product name: {product_name}.",
                    f"Product name tokens: {product_name}. {product_name}.",
                ]
            )
        if "categories" in selected_columns:
            document_parts.extend(["Source categories:", " ".join(category_text)])
        if "reviews.text" in selected_columns:
            review_sample = (
                product_df["reviews.text"]
                .astype(str)
                .drop_duplicates()
                .head(12)
                .tolist()
            )
            document_parts.extend(["Representative reviews:", " ".join(review_sample)])
        if "reviews.rating" in selected_columns:
            document_parts.append(
                f"Average rating: {product_df['reviews.rating'].mean():.2f}."
            )
        if "sentiment" in selected_columns and "sentiment" in product_df.columns:
            sentiment_counts = product_df["sentiment"].value_counts(normalize=True).to_dict()
            document_parts.append(
                "Sentiment mix: "
                + " ".join(f"{label} {share:.2f}" for label, share in sentiment_counts.items())
            )
        document = " ".join(document_parts)
        rows.append(
            {
                "name": str(product_name),
                "review_count": int(len(product_df)),
                "categories": " | ".join(category_text),
                "clustering_columns": ", ".join(selected_columns),
                "cluster_document": document,
            }
        )
    return pd.DataFrame(rows)


def _cluster_diagnostics(product_docs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category_id, cluster_df in product_docs.groupby("category_id"):
        top_categories = (
            cluster_df["categories"]
            .astype(str)
            .str.split("|")
            .explode()
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .value_counts()
            .head(5)
        )
        rows.append(
            {
                "cluster_id": int(category_id),
                "product_count": int(len(cluster_df)),
                "review_count": int(cluster_df["review_count"].sum()),
                "example_products": ", ".join(cluster_df["name"].head(6).tolist()),
                "top_source_categories": ", ".join(top_categories.index.tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id")


def _category_frequency(df: pd.DataFrame) -> pd.Series:
    return (
        df["categories"]
        .astype(str)
        .str.split(r"[>|,]")
        .explode()
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
    )


def _keyword_taxonomy_labels(product_docs: pd.DataFrame) -> tuple[np.ndarray, dict[int, str]]:
    labels_text = [
        infer_category(row["name"], row["cluster_document"])
        for _, row in product_docs.iterrows()
    ]
    unique_labels = sorted(set(labels_text))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    return (
        np.array([label_to_id[label] for label in labels_text], dtype=int),
        {idx: label for label, idx in label_to_id.items()},
    )


def _cluster_with_method(vectors: np.ndarray, k: int, method: str) -> np.ndarray:
    if k <= 1:
        return np.zeros(len(vectors), dtype=int)
    if method == "TF-IDF + Agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(vectors)
    return KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(vectors)


def _show_example(label: str, df: pd.DataFrame, columns: list[str], rows: int = 3) -> None:
    available = [column for column in columns if column in df.columns]
    if not available:
        return
    preview = df.loc[:, available].head(rows).copy()
    for column in preview.select_dtypes(include=["object"]).columns:
        preview[column] = preview[column].astype(str).str.slice(0, 120)
    st.markdown(f"**Example processed data: {label}**")
    st.dataframe(preview, use_container_width=True, hide_index=True)


def _stage_note(data: str, processing: str, output: str) -> None:
    st.markdown(
        f"**Data processed:** {data}\n\n"
        f"**How it was processed:** {processing}\n\n"
        f"**Output:** {output}"
    )


def _config_table(rows: list[tuple[str, object]]) -> None:
    st.markdown("**Configuration**")
    st.dataframe(
        pd.DataFrame(
            [{"setting": setting, "value": str(value)} for setting, value in rows]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "setting": st.column_config.TextColumn(width="medium"),
            "value": st.column_config.TextColumn(width="large"),
        },
    )


def _remember_stage(stage: str, detail: str) -> None:
    summaries = list(st.session_state.get("stage_summary", []))
    summaries = [item for item in summaries if item["stage"] != stage]
    summaries.append({"stage": stage, "detail": detail})
    st.session_state.stage_summary = summaries


def _render_stage_summary() -> None:
    summaries = st.session_state.get("stage_summary", [])
    if not summaries:
        return

    st.info(
        "Showing the latest completed pipeline run. Streamlit reruns the page when "
        "you interact with Insights, so the live status windows are redrawn here as "
        "a persistent summary."
    )
    for item in summaries:
        with st.expander(item["stage"], expanded=False):
            st.write(item["detail"])
            if item["stage"].startswith("Stage 5/7"):
                _render_clustering_figures()


def _make_figures(embeddings: np.ndarray, labels: np.ndarray, best_k: int, scores: dict[int, float]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if scores:
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = list(scores.keys())
        values = [scores[k] for k in ks]
        bars = ax.bar(ks, values, color="#5b8def")
        if best_k in ks:
            bars[ks.index(best_k)].set_color("#22c55e")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette score by k (green = selected)")
        ax.set_xticks(ks)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "silhouette_scores.png", dpi=120)
        plt.close(fig)
    else:
        silhouette_path = FIGURES_DIR / "silhouette_scores.png"
        if silhouette_path.exists():
            silhouette_path.unlink()

    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Product reviews - {best_k} clusters (PCA 2D, KMeans)")
    handles, _ = scatter.legend_elements()
    ax.legend(
        handles,
        [f"Category {i + 1}" for i in range(best_k)],
        title="Cluster",
        loc="best",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_visualization.png", dpi=120)
    plt.close(fig)


def _silhouette_altair_chart(scores: dict[int, float], best_k: int) -> None:
    if not scores:
        return
    chart_data = pd.DataFrame([
        {"k": k, "Silhouette score": v, "Selected": "Selected k" if k == best_k else "Other"}
        for k, v in sorted(scores.items())
    ])
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("k:O", title="k (number of clusters)"),
            y=alt.Y("Silhouette score:Q"),
            color=alt.Color(
                "Selected:N",
                scale=alt.Scale(domain=["Selected k", "Other"], range=["#22c55e", "#5b8def"]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[alt.Tooltip("k:O"), alt.Tooltip("Silhouette score:Q", format=".4f")],
        )
        .properties(title="Silhouette score by k (green = selected)", height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_clustering_figures() -> None:
    st.markdown("**Clustering figures**")
    fig_silhouette = FIGURES_DIR / "silhouette_scores.png"
    fig_cluster = FIGURES_DIR / "cluster_visualization.png"
    left, right = st.columns(2)
    with left:
        if fig_silhouette.exists():
            st.image(str(fig_silhouette), caption="Silhouette score by k", use_container_width=True)
        else:
            st.info("Silhouette chart has not been generated yet.")
    with right:
        if fig_cluster.exists():
            st.image(str(fig_cluster), caption="Cluster visualization (PCA 2D)", use_container_width=True)
        else:
            st.info("Cluster visualization has not been generated yet.")


def _file_row(label: str, path: Path, mime: str, description: str) -> None:
    status, meta, action = st.columns([1, 4, 2])
    rel = path.relative_to(OUTPUTS_DIR.parent) if path.exists() or OUTPUTS_DIR.parent in path.parents else path
    if path.exists():
        status.markdown("ok")
        meta.markdown(f"**{label}** - `{rel}`  \n{description} - {path.stat().st_size / 1024:,.1f} KB")
        action.download_button(
            "Download",
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=f"dl-{path}",
        )
    else:
        status.markdown("missing")
        meta.markdown(f"**{label}** - `{rel}`  \n{description} - not yet generated")
        action.markdown("-")


def _render_playground() -> None:
    st.subheader("Sentiment Playground")
    st.caption(
        "Calls `POST /predict-sentiment`. The endpoint uses a hint-word heuristic, "
        "not a trained classifier."
    )
    with st.expander("Hint words used by the heuristic"):
        st.markdown(
            "**Positive:** great, excellent, love, fast, reliable, perfect, best, easy, happy  \n"
            "**Negative:** bad, broken, slow, terrible, poor, failed, waste, return, "
            "disappointed, worst"
        )

    examples = {
        "Positive": "The setup was easy and the battery life is great.",
        "Negative": "The remote is broken and the screen is terrible.",
        "Neutral": "Arrived on time and matches the description.",
        "Mixed": "Great picture but the sound is terrible.",
    }
    st.markdown("**Quick examples**")
    cols = st.columns(len(examples))
    for col, (label, text) in zip(cols, examples.items()):
        if col.button(label, key=f"ex-{label}"):
            st.session_state.playground_text = text

    text_input = st.text_area(
        "Review text",
        value=st.session_state.get("playground_text", ""),
        height=120,
        key="playground_textarea",
    )
    if st.button("Predict sentiment", type="primary", disabled=not text_input.strip()):
        try:
            with st.spinner("Calling /predict-sentiment..."):
                resp = requests.post(f"{API_URL}/predict-sentiment", json={"text": text_input}, timeout=30)
            if resp.ok:
                label = resp.json()["sentiment"]
                color = {"positive": "green", "negative": "red", "neutral": "gray"}.get(label, "gray")
                st.markdown(f"### Result: :{color}[**{label.upper()}**]")
                with st.expander("Raw response"):
                    st.json(resp.json())
            else:
                st.error(resp.text)
        except requests.exceptions.ConnectionError as exc:
            st.error(f"Cannot reach the API at `{API_URL}`. Is FastAPI running?\n\n{exc}")

    st.markdown("**Heuristic vs rating agreement**")
    st.caption("Scores up to 2,000 loaded reviews and compares the heuristic to the rating-based label.")
    clustered_df = st.session_state.get("clustered_df")
    if st.button("Compute agreement", key="compute-heuristic-agreement"):
        if clustered_df is None or "reviews.text" not in clustered_df.columns:
            st.warning("No clustered reviews in memory yet. Run the pipeline first.")
        else:
            sample_n = min(2000, len(clustered_df))
            sample = clustered_df.sample(sample_n, random_state=42).copy()
            analyzer = get_sentiment_analyzer()
            with st.spinner(f"Scoring {sample_n:,} reviews..."):
                sample["heuristic"] = [analyzer.predict_text(t) for t in sample["reviews.text"].astype(str)]
            agreement = float((sample["heuristic"] == sample["sentiment"]).mean())
            confusion = (
                sample.groupby(["sentiment", "heuristic"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=["negative", "neutral", "positive"], fill_value=0)
                .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
            )
            per_class = (
                sample.groupby("sentiment")
                .apply(lambda g: float((g["heuristic"] == g["sentiment"]).mean()), include_groups=False)
                .rename("agreement")
                .reset_index()
            )
            per_class["agreement"] = per_class["agreement"].map(lambda value: f"{value:.1%}")
            st.session_state.heuristic_eval = {
                "agreement": agreement,
                "n": sample_n,
                "per_class": per_class,
                "confusion": confusion,
            }

    eval_result = st.session_state.get("heuristic_eval")
    if eval_result is not None:
        left, right = st.columns([1, 2])
        with left:
            st.metric("Overall agreement", f"{eval_result['agreement']:.1%}", f"n = {eval_result['n']:,}")
            st.dataframe(eval_result["per_class"], use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Confusion matrix**")
            st.dataframe(eval_result["confusion"], use_container_width=True)


def _render_insights() -> None:
    insights = st.session_state.get("insights")
    clustered_df = st.session_state.get("clustered_df")
    if not insights:
        st.info("Run the pipeline or load the latest artifacts to see insights.")
        return

    st.subheader(f"Insights - {len(insights)} discovered categories")
    for insight in insights:
        cat_id = insight["category_id"]
        cat_name = insight["category_name"]
        with st.expander(f"{cat_name} - {insight['review_count']:,} reviews", expanded=False):
            top_cols = st.columns(3)
            top_cols[0].metric("Average rating", insight["avg_rating"])
            top_cols[1].metric("Reviews", f"{insight['review_count']:,}")
            top_cols[2].metric("Positive", f"{insight['sentiment_ratio']['positive']:.1%}")

            st.markdown("**Top products**")
            st.dataframe(pd.DataFrame(insight["top_products"]), use_container_width=True, hide_index=True)

            st.markdown("**Common complaints**")
            for complaint in insight["complaints"]:
                st.write(f"- {complaint}")

            if clustered_df is not None:
                cat_df = clustered_df[clustered_df["category_id"] == cat_id]
                if not cat_df.empty:
                    products = sorted(cat_df["name"].astype(str).unique())
                    chosen = st.selectbox("Drill into a product", ["-"] + products, key=f"product-{cat_id}")
                    if chosen != "-":
                        rows = cat_df[cat_df["name"] == chosen][["reviews.rating", "sentiment", "reviews.text"]]
                        st.dataframe(rows, use_container_width=True, hide_index=True)

            st.markdown("**Recommendation article**")
            article = st.session_state.articles.get(cat_id)
            col_gen, col_regen, col_dl = st.columns(3)
            gen_clicked = col_gen.button("Generate", key=f"gen-{cat_id}", disabled=bool(article))
            regen_clicked = col_regen.button("Regenerate", key=f"regen-{cat_id}", disabled=not article)
            if article:
                col_dl.download_button(
                    "Download .md",
                    data=article,
                    file_name=f"{_slugify(cat_name)}.md",
                    mime="text/markdown",
                    key=f"dl-article-{cat_id}",
                )
            if gen_clicked or regen_clicked:
                with st.spinner("Writing recommendation article..."):
                    writer = get_writer()
                    article = writer.generate(insight)
                    st.session_state.articles[cat_id] = article
                    BLOGPOSTS_DIR.mkdir(parents=True, exist_ok=True)
                    (BLOGPOSTS_DIR / f"{_slugify(cat_name)}.md").write_text(article, encoding="utf-8")
                    st.rerun()
            if article:
                st.markdown(article)


RUN_HISTORY_PATH = OUTPUTS_DIR / "runs_history.json"


def _save_run_history(record: dict) -> None:
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if RUN_HISTORY_PATH.exists():
        try:
            history = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(record)
    RUN_HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _render_run_history(current_ts: str | None = None) -> None:
    """Compact inline summary used in the Pipeline tab after a run."""
    if not RUN_HISTORY_PATH.exists():
        st.caption("No run history yet. A record is saved automatically at the end of each pipeline run.")
        return
    try:
        history = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Could not read run history file.")
        return
    if not history:
        return

    rows = []
    for i, rec in enumerate(history):
        cfg = rec.get("config", {})
        res = rec.get("results", {})
        sil = res.get("best_silhouette")
        cats = res.get("categories", [])
        rows.append({
            "#": i + 1,
            "": "← this run" if rec.get("timestamp") == current_ts else "",
            "Time": rec.get("timestamp", ""),
            "Method": cfg.get("vector_method", ""),
            "k": res.get("best_k", "?"),
            "k mode": cfg.get("k_mode", ""),
            "Silhouette": round(sil, 3) if sil is not None else "—",
            "Cluster names": " | ".join(c["name"] for c in cats),
            "Reviews": res.get("review_count", ""),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "": st.column_config.TextColumn(width="small"),
            "Time": st.column_config.TextColumn(width="medium"),
            "Method": st.column_config.TextColumn(width="medium"),
            "k": st.column_config.NumberColumn(width="small"),
            "k mode": st.column_config.TextColumn(width="medium"),
            "Silhouette": st.column_config.TextColumn(width="small"),
            "Cluster names": st.column_config.TextColumn(width="large"),
            "Reviews": st.column_config.NumberColumn(width="small"),
        },
    )


def _render_runs_tab() -> None:
    """Full run-history tab with silhouette chart and per-run cluster breakdown."""
    st.subheader("Run history")
    st.caption(
        "Every pipeline run is logged here automatically. "
        "Compare cluster names, silhouette scores, and parameter settings across configurations."
    )

    if not RUN_HISTORY_PATH.exists():
        st.info("No runs recorded yet. Run the Analytics Pipeline to start tracking configurations.")
        return
    try:
        history = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.error("Could not read run history file.")
        return
    if not history:
        st.info("Run history file is empty.")
        return

    last_ts = st.session_state.get("_last_run_ts")

    # ── Top action row ──────────────────────────────────────────────────────
    col_clear, col_dl, _ = st.columns([1, 1, 5])
    if col_clear.button("Clear all history", type="secondary", use_container_width=True):
        RUN_HISTORY_PATH.write_text("[]", encoding="utf-8")
        st.session_state["_last_run_ts"] = None
        st.rerun()

    # ── Build summary rows ──────────────────────────────────────────────────
    rows = []
    for i, rec in enumerate(history):
        cfg = rec.get("config", {})
        res = rec.get("results", {})
        sil = res.get("best_silhouette")
        cats = res.get("categories", [])
        rows.append({
            "#": i + 1,
            "": "← latest" if rec.get("timestamp") == last_ts else "",
            "Time": rec.get("timestamp", ""),
            "Method": cfg.get("vector_method", ""),
            "k": res.get("best_k", "?"),
            "k mode": cfg.get("k_mode", ""),
            "Cluster cols": ", ".join(cfg.get("clustering_columns", [])),
            "Naming cols": ", ".join(cfg.get("naming_columns", [])),
            "Silhouette": round(sil, 3) if sil is not None else None,
            "Cluster names": " | ".join(c["name"] for c in cats),
            "Reviews": res.get("review_count", ""),
            "Products": res.get("product_count", ""),
        })
    df_summary = pd.DataFrame(rows)

    col_dl.download_button(
        "Download CSV",
        df_summary.drop(columns=[""]).to_csv(index=False).encode(),
        file_name="runs_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Section 1: Overview table ───────────────────────────────────────────
    st.markdown("### Overview")
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "": st.column_config.TextColumn(width="small"),
            "Time": st.column_config.TextColumn(width="medium"),
            "Method": st.column_config.TextColumn(width="medium"),
            "k": st.column_config.NumberColumn(width="small"),
            "k mode": st.column_config.TextColumn(width="medium"),
            "Cluster cols": st.column_config.TextColumn(width="medium"),
            "Naming cols": st.column_config.TextColumn(width="medium"),
            "Silhouette": st.column_config.NumberColumn(width="small", format="%.3f"),
            "Cluster names": st.column_config.TextColumn(width="large"),
            "Reviews": st.column_config.NumberColumn(width="small"),
            "Products": st.column_config.NumberColumn(width="small"),
        },
    )

    # ── Section 2: Silhouette comparison chart ──────────────────────────────
    sil_rows = [r for r in rows if r["Silhouette"] is not None]
    if sil_rows:
        st.markdown("### Silhouette score by run")
        st.caption(
            "Higher is better (max 1.0). Silhouette measures how well each product fits its own cluster vs. neighbouring ones. "
            "Values above 0.3 indicate reasonable separation; below 0.2 means clusters heavily overlap. "
            "Note: silhouette alone favours fewer broad clusters — use the cluster names to judge quality too."
        )
        sil_df = pd.DataFrame({
            "run": [f"#{r['#']}  {r['Time']}" for r in sil_rows],
            "silhouette": [r["Silhouette"] for r in sil_rows],
            "method": [r["Method"] for r in sil_rows],
            "k": [str(r["k"]) for r in sil_rows],
        })
        best_sil = sil_df["silhouette"].max()
        sil_df["color"] = sil_df["silhouette"].apply(
            lambda v: "best" if v == best_sil else "other"
        )
        chart = (
            alt.Chart(sil_df)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("silhouette:Q", title="Silhouette score", scale=alt.Scale(domain=[0, max(1.0, best_sil)])),
                y=alt.Y("run:N", sort=None, title=None),
                color=alt.Color(
                    "color:N",
                    scale=alt.Scale(domain=["best", "other"], range=["#2ca02c", "#4c78a8"]),
                    legend=alt.Legend(title=""),
                ),
                tooltip=[
                    alt.Tooltip("run:N", title="Run"),
                    alt.Tooltip("silhouette:Q", title="Silhouette", format=".3f"),
                    alt.Tooltip("method:N", title="Method"),
                    alt.Tooltip("k:N", title="k"),
                ],
            )
            .properties(height=max(120, len(sil_rows) * 44))
        )
        st.altair_chart(chart, use_container_width=True)

    # ── Section 3: Per-run cluster breakdown ────────────────────────────────
    st.markdown("### Per-run cluster breakdown")
    st.caption("Newest run shown first. Expand any run to see per-cluster metrics.")
    for i, rec in enumerate(reversed(history)):
        run_num = len(history) - i
        cfg = rec.get("config", {})
        res = rec.get("results", {})
        cats = res.get("categories", [])
        sil = res.get("best_silhouette")
        ts = rec.get("timestamp", "")
        is_latest = ts == last_ts
        label = (
            f"{'🟢 ' if is_latest else ''}Run #{run_num}  ·  {ts}  ·  "
            f"{cfg.get('vector_method', '')}  ·  k={res.get('best_k', '?')}  ·  "
            f"silhouette={'—' if sil is None else f'{sil:.3f}'}"
        )
        with st.expander(label, expanded=(i == 0)):
            # ── Result metrics ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clusters", res.get("best_k", "?"))
            m2.metric("Reviews", f"{res.get('review_count', 0):,}")
            m3.metric("Products", f"{res.get('product_count', 0):,}")
            m4.metric("Silhouette", f"{sil:.3f}" if sil is not None else "—")

            # ── Full configuration table ────────────────────────────────────
            forced_k_val = cfg.get("forced_k", "?")
            max_k_val = cfg.get("max_k_to_test", "?")
            k_mode = cfg.get("k_mode", "")
            k_detail = (
                f"Force k = {forced_k_val}"
                if k_mode == "Force k"
                else f"Auto by silhouette (max k = {max_k_val})"
            )
            _config_table([
                ("Input source",        cfg.get("input_mode", "—")),
                ("Feature method",      cfg.get("vector_method", "—")),
                ("Clustering columns",  ", ".join(cfg.get("clustering_columns", []))),
                ("Naming columns",      ", ".join(cfg.get("naming_columns", []))),
                ("Cluster count",       k_detail),
                ("Actual k",            res.get("best_k", "?")),
                ("Silhouette",          f"{sil:.3f}" if sil is not None else "— (Keyword taxonomy or too few samples)"),
            ])

            if not cats:
                st.info("No cluster details saved for this run.")
                continue

            total_reviews = sum(c.get("review_count", 0) for c in cats)
            cat_rows = []
            for cat in cats:
                rc = cat.get("review_count", 0)
                pos = cat.get("positive_ratio", 0)
                cat_rows.append({
                    "Cluster": cat["name"],
                    "Top terms": cat.get("top_terms") or "—",
                    "Reviews": rc,
                    "Size %": round(rc / total_reviews * 100, 1) if total_reviews else 0,
                    "Avg ⭐": round(cat.get("avg_rating", 0), 1),
                    "% Positive": round(pos * 100, 1),
                })
            st.dataframe(
                pd.DataFrame(cat_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Cluster": st.column_config.TextColumn(width="medium"),
                    "Top terms": st.column_config.TextColumn(width="large"),
                    "Reviews": st.column_config.NumberColumn(width="small"),
                    "Size %": st.column_config.NumberColumn(width="small", format="%.1f%%"),
                    "Avg ⭐": st.column_config.NumberColumn(width="small", format="%.1f"),
                    "% Positive": st.column_config.NumberColumn(width="small", format="%.1f%%"),
                },
            )


def _render_artifacts() -> None:
    st.subheader("Output artifacts")
    st.caption("Figures are previewed inline in Stage 5; this tab keeps the file list and downloads.")
    st.markdown("### Core outputs")
    _file_row("clustered_reviews.csv", OUTPUTS_DIR / "clustered_reviews.csv", "text/csv", "Reviews with sentiment and cluster labels")
    _file_row("category_insights.json", OUTPUTS_DIR / "category_insights.json", "application/json", "Aggregated category insights")
    st.markdown("### Metrics")
    _file_row("category_names.json", METRICS_DIR / "category_names.json", "application/json", "TF-IDF-derived category names")
    _file_row("silhouette_scores.json", METRICS_DIR / "silhouette_scores.json", "application/json", "KMeans silhouette score per k")
    _file_row("cluster_sizes.json", METRICS_DIR / "cluster_sizes.json", "application/json", "Reviews per discovered cluster")
    _file_row("sentiment_distribution.json", METRICS_DIR / "sentiment_distribution.json", "application/json", "Per-category sentiment shares")
    st.markdown("### Figures")
    _file_row("silhouette_scores.png", FIGURES_DIR / "silhouette_scores.png", "image/png", "Shown inline in Stage 5")
    _file_row("cluster_visualization.png", FIGURES_DIR / "cluster_visualization.png", "image/png", "Shown inline in Stage 5")
    st.markdown("### Blog posts")
    if BLOGPOSTS_DIR.exists() and list(BLOGPOSTS_DIR.glob("*.md")):
        for path in sorted(BLOGPOSTS_DIR.glob("*.md")):
            _file_row(path.name, path, "text/markdown", "Recommendation article")
    else:
        st.info("No generated Markdown articles yet.")


tab_pipeline, tab_insights, tab_playground, tab_runs, tab_artifacts = st.tabs(
    ["Analytics Pipeline", "Insights", "Sentiment Playground", "Run History", "Output Artifacts"]
)

with tab_pipeline:
    st.subheader("Run the analytics pipeline")
    st.write(
        "Pick an input source: a single uploaded CSV or every compatible CSV inside "
        f"`{RAW_DATA_DIR}`."
    )

    input_mode = st.radio(
        "Input source",
        ["Use all CSVs in data/raw", "Upload a single CSV"],
        horizontal=True,
        key=f"input_mode_{_rc}",
    )
    uploaded = st.file_uploader("Upload an Amazon reviews CSV", type=["csv"], key=f"uploaded_{_rc}") if input_mode == "Upload a single CSV" else None

    raw_csvs = sorted(RAW_DATA_DIR.glob("*.csv")) if RAW_DATA_DIR.exists() else []
    n_compatible_csvs = 0
    if input_mode == "Use all CSVs in data/raw":
        if raw_csvs:
            rows = []
            for path in raw_csvs:
                try:
                    header = pd.read_csv(path, nrows=0)
                    ok = all(column in header.columns for column in REQUIRED_COLUMNS)
                except Exception:  # noqa: BLE001
                    ok = False
                if ok:
                    n_compatible_csvs += 1
                rows.append({"file": path.name, "size_MB": round(path.stat().st_size / (1024 * 1024), 1), "compatible": ok})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning(f"No CSV files found in `{RAW_DATA_DIR}`.")

    can_run = uploaded is not None if input_mode == "Upload a single CSV" else bool(raw_csvs)
    st.markdown("### Clustering controls")
    st.caption(
        "These controls affect Stage 4/5. After changing category inference, feature "
        "input columns, feature method, or k settings, choose **Delete cache before processing** before running again."
    )
    clustering_columns = st.multiselect(
        "Input columns for clustering",
        ["name", "categories", "reviews.text", "reviews.rating", "sentiment"],
        default=["categories"],
        key=f"clustering_columns_{_rc}",
        help=(
            "These columns are concatenated into one document per product before clustering.\n\n"
            "• **name + categories** — best for product-type clusters (laptop, tablet, cable…). "
            "The cluster can only separate products that differ in these fields.\n"
            "• **reviews.text** — adds review wording; useful when category labels are generic.\n"
            "• **reviews.rating / sentiment** — pulls quality/opinion into the cluster signal; "
            "use only when you want clusters split by satisfaction, not product type."
        ),
    )
    if not clustering_columns:
        st.warning("Select at least one clustering input column. Defaulting to `name`.")
        clustering_columns = ["name"]
    naming_columns = st.multiselect(
        "Input columns for cluster naming",
        ["name", "categories", "reviews.text"],
        default=["categories"],
        key=f"naming_columns_{_rc}",
        help=(
            "These text columns are used to derive a readable label for each cluster via TF-IDF.\n\n"
            "TF-IDF rewards terms that appear frequently in one cluster but rarely across others — "
            "it highlights what makes each cluster distinctive.\n\n"
            "• **name + categories** — product names and category labels drive the cluster title (e.g. 'Tablet Speaker').\n"
            "• **reviews.text** — review wording contributes; useful when product names are too generic.\n\n"
            "Only text columns are available here — numeric and label columns carry no term signal for naming."
        ),
    )
    if not naming_columns:
        st.warning("Select at least one naming column. Defaulting to `name`.")
        naming_columns = ["name"]
    col_method, col_k_mode, col_k_value, col_k_max = st.columns([2, 2, 1, 1])
    vector_method = col_method.selectbox(
        "Feature method",
        [
            "Keyword taxonomy",
            "TF-IDF terms",
            "TF-IDF + Agglomerative",
            "MiniLM embeddings",
        ],
        index=1,
        key=f"vector_method_{_rc}",
        help=(
            "How product documents are converted to numbers before clustering.\n\n"
            "• **Keyword taxonomy** — rule-based labels (Laptop, Tablet, Cable, Case, Speaker…). "
            "Ignores k and silhouette entirely; number of clusters = number of matched rules.\n"
            "• **TF-IDF terms** — bag-of-words on product/category text → KMeans. "
            "Best general choice; keeps product terms explicit.\n"
            "• **TF-IDF + Agglomerative** — same features but hierarchical clustering instead of KMeans.\n"
            "• **MiniLM embeddings** — semantic sentence embeddings → KMeans. "
            "Better when review text is included and you want meaning over keywords."
        ),
    )
    k_mode = col_k_mode.radio(
        "Cluster count",
        ["Force k", "Auto by silhouette"],
        horizontal=True,
        disabled=vector_method == "Keyword taxonomy",
        key=f"k_mode_{_rc}",
        help=(
            "• **Force k** — cluster into exactly the number you choose. "
            "Use this when you have a clear idea of the product categories (e.g. 6 for laptop/tablet/cable/case/speaker/reader).\n"
            "• **Auto by silhouette** — sweeps k from 2 to Max auto k and picks the mathematically best split. "
            "Often chooses a small k (2–3) because broad groups score well; override with Force k if you want finer categories.\n\n"
            "Disabled for Keyword taxonomy (cluster count is determined by keyword rules)."
        ),
    )
    forced_k = col_k_value.number_input(
        "Forced k",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        disabled=vector_method == "Keyword taxonomy",
        key=f"forced_k_{_rc}",
        help="Exact number of clusters when using Force k. A value of 6 suits laptop/tablet/cable/case/speaker/reader splits.",
    )
    max_k_to_test = col_k_max.number_input(
        "Max auto k",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        disabled=vector_method == "Keyword taxonomy",
        key=f"max_k_{_rc}",
        help="Upper bound of the silhouette sweep when using Auto by silhouette. The app tries every k from 2 to this value and picks the best score.",
    )
    if vector_method == "Keyword taxonomy":
        st.info(
            "Keyword taxonomy ignores k and silhouette. It creates categories directly "
            "from keyword rules, so the number of categories depends on matched labels."
        )

    run_mode = st.radio(
        "Run mode",
        ["Reuse cache", "Delete cache before processing"],
        horizontal=True,
        key=f"run_mode_{_rc}",
        help=(
            "Choose whether Stage 4/5 should reuse cached features/clusters or start fresh."
        ),
    )
    run_clicked = st.button("Process reviews", type="primary", disabled=not can_run)

    slot_progress = st.empty()
    slot_cache_msg = st.empty()
    slot_stage_1 = st.container()
    slot_stage_2 = st.container()
    slot_stage_3 = st.container()
    slot_stage_4 = st.container()
    slot_stage_5 = st.container()
    slot_stage_6 = st.container()
    slot_stage_7 = st.container()
    slot_summary = st.empty()

    if run_clicked and can_run:
        st.session_state.stage_summary = []
        if run_mode == "Delete cache before processing":
            removed = _clear_pipeline_cache()
            slot_cache_msg.info(f"Deleted {removed} cached file(s) from `outputs/cache/` before processing.")
        progress = slot_progress.progress(0, text="Starting...")
        with slot_summary:
            st.markdown("**Selected pipeline parameters**")
            _config_table(
                [
                    ("Input source", input_mode),
                    ("Run mode", run_mode),
                    ("Use cached Stage 4/5 artifacts", "yes" if run_mode == "Reuse cache" else "no"),
                    ("Input columns for clustering", ", ".join(clustering_columns)),
                    ("Input columns for naming", ", ".join(naming_columns)),
                    ("Feature method", vector_method),
                    ("Cluster count mode", k_mode if vector_method != "Keyword taxonomy" else "disabled for Keyword taxonomy"),
                    ("Forced k", forced_k if vector_method != "Keyword taxonomy" and k_mode == "Force k" else "not used"),
                    ("Max auto k", max_k_to_test if vector_method != "Keyword taxonomy" else "not used"),
                ]
            )

        with slot_stage_1:
            with st.status("Stage 1/7 - Data Input", expanded=True) as status:
                _source_label = (
                    f"1 uploaded file — `{uploaded.name}`"
                    if input_mode == "Upload a single CSV"
                    else f"{n_compatible_csvs} compatible CSV file(s) from `{RAW_DATA_DIR.name}/`"
                )
                _stage_note(
                    f"{_source_label}.",
                    (
                        "Required columns are validated. The file is read into a raw DataFrame."
                        if input_mode == "Upload a single CSV"
                        else "Required columns are validated for each file. Incompatible schemas are skipped with a warning; a `source_file` column is added for traceability."
                    ),
                    "A raw DataFrame with original column values, ready for cleaning.",
                )
                _config_table(
                    [
                        ("Input mode", input_mode),
                        ("Raw data directory", RAW_DATA_DIR),
                        ("Required columns", ", ".join(REQUIRED_COLUMNS)),
                        ("Compatible file rule", "CSV must expose all required columns"),
                        ("Skipped file rule", "Invalid schema or parser error"),
                    ]
                )
                if input_mode == "Upload a single CSV":
                    raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                    raw["source_file"] = uploaded.name
                    validate_columns(raw.columns)
                    st.write(f"Read {len(raw):,} raw rows from `{uploaded.name}`.")
                else:
                    raw = None
                    st.write(f"Reading compatible CSVs from `{RAW_DATA_DIR}`.")
                _remember_stage("Stage 1/7 - Data Input", "Input files were read and validated.")
                status.update(label="Stage 1/7 - input ready", state="complete")
        progress.progress(14, text="Input loaded")

        with slot_stage_2:
            with st.status("Stage 2/7 - Preprocessing", expanded=True) as status:
                _stage_note(
                    "Raw rows from " + ("the uploaded CSV." if input_mode == "Upload a single CSV" else f"{n_compatible_csvs} merged CSV file(s)."),
                    "5-step category normalisation: last path segment extracted → keyword-collapsed to canonical label (e.g. 'Kids\\' Tablets' → 'Tablet') → ambiguous multi-family matches blanked → blocklist junk (retailers, promo sections, connectivity specs) removed → all blanks re-inferred from `name` + `reviews.text`. Then: rows missing `name`/`text`/`rating` dropped; duplicates removed on `name + text + rating`; `reviews.text` lowercased, HTML/URLs stripped; `reviews.rating` coerced to numeric, kept in [1, 5].",
                    "A clean DataFrame with all `categories` filled, `reviews.text` normalised, and numeric ratings in [1, 5].",
                )
                _config_table(
                    [
                        ("Columns kept", ", ".join(REQUIRED_COLUMNS) + ", source_file when available"),
                        ("Drop rows missing", "name, reviews.text, reviews.rating"),
                        ("Category step 1 — path extraction", "for comma-separated paths, keep only the last (most specific) segment"),
                        ("Category step 2 — keyword collapse", "single keyword family match → canonical label (e.g. 'Kids\\' Tablets' → 'Tablet')"),
                        ("Category step 3 — ambiguity blanking", "2+ keyword families match (e.g. 'Computers & Tablets') → blank → re-inferred from review"),
                        ("Category step 4 — blocklist removal", "retailer / promo / spec labels (Frys, Holiday Shop, Wi-Fi 3G…) → blank → re-inferred from review"),
                        ("Category step 5 — inference", "blank cells (original or newly blanked) → keyword heuristic on name + reviews.text → first matching rule wins"),
                        ("Inferred category labels", "Laptop, Tablet, Cable, Case Sleeve Bag, Speaker Audio, Remote Streaming TV, Headphones, Battery Power, Camera, Keyboard Mouse, Uncategorized"),
                        ("Duplicate key", "name + reviews.text + reviews.rating"),
                        ("Text cleaning", "lowercase, strip HTML/URLs, normalize whitespace"),
                        ("Rating rule", "numeric only, keep values in [1, 5]"),
                    ]
                )
                if input_mode == "Upload a single CSV":
                    cat_stats, cat_sample = category_normalization_report(
                        raw[["name", "reviews.text", "categories"]].fillna({"categories": ""})
                    )
                    df = _clean_uploaded_frame(raw)
                else:
                    cat_stats, cat_sample = None, None
                    df = load_reviews_dir(RAW_DATA_DIR)

                m1, m2, m3 = st.columns(3)
                m1.metric("Clean reviews", f"{len(df):,}")
                m2.metric("Unique products", f"{df['name'].nunique():,}")
                m3.metric("Unique categories", f"{df['categories'].nunique():,}")

                st.markdown("**Category normalisation**")
                if cat_stats is not None:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Kept as-is", f"{cat_stats['kept']:,}")
                    c2.metric("Collapsed to canonical", f"{cat_stats['collapsed']:,}",
                              help="Last path segment matched a single product-type keyword → replaced with the canonical label (e.g. 'Kids\\' Tablets' → 'Tablet').")
                    c3.metric("Junk → inferred from review", f"{cat_stats['blocklist_inferred']:,}",
                              help="Category was a retailer, promotion, or connectivity spec (e.g. 'Frys', 'Holiday Shop', 'Wi-Fi 3G') → blanked and re-inferred from product name and review text.")
                    c4.metric("Ambiguous → inferred from review", f"{cat_stats['ambiguous_inferred']:,}",
                              help="Last path segment matched 2+ product-type keywords (e.g. 'Computers & Tablets') → blanked and re-inferred from product name and review text.")
                    c5.metric("Blank → inferred from review", f"{cat_stats['blank_inferred']:,}",
                              help="Category was originally empty → inferred from product name and review text.")
                    if not cat_sample.empty:
                        st.markdown("**Sample: collapsed, junk, and ambiguous rows**")
                        st.dataframe(
                            cat_sample.rename(columns={"categories": "original", "normalized_to": "result"}),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "name": st.column_config.TextColumn(width="medium"),
                                "original": st.column_config.TextColumn(width="large"),
                                "result": st.column_config.TextColumn(width="medium"),
                                "action": st.column_config.TextColumn(width="medium"),
                            },
                        )
                else:
                    st.caption("Detailed per-row breakdown is available in upload mode only.")

                if cat_stats is not None:
                    inferred_categories = cat_stats["blank_inferred"] + cat_stats["ambiguous_inferred"]
                else:
                    inferred_categories = int(df["categories"].astype(str).isin([
                        "Laptop", "Tablet", "Cable", "Case Sleeve Bag", "Speaker Audio",
                        "Remote Streaming TV", "Headphones", "Battery Power", "Camera",
                        "Keyboard Mouse", "Uncategorized",
                    ]).sum())

                category_counts = _category_frequency(df)
                st.markdown("**Category frequency after preprocessing**")
                st.dataframe(
                    category_counts.rename_axis("category")
                    .reset_index(name="review_count")
                    .head(50),
                    use_container_width=True,
                    hide_index=True,
                )
                _show_example("clean reviews", df, ["source_file", "name", "categories", "reviews.rating", "reviews.text"])
                if inferred_categories:
                    st.markdown("**Example rows with inferred/simple categories**")
                    inferred_preview = df[
                        df["categories"].astype(str).isin(
                            [
                                "Laptop",
                                "Tablet",
                                "Cable",
                                "Case Sleeve Bag",
                                "Speaker Audio",
                                "Remote Streaming TV",
                                "Headphones",
                                "Battery Power",
                                "Camera",
                                "Keyboard Mouse",
                                "Uncategorized",
                            ]
                        )
                    ][["name", "categories", "reviews.text"]].head(8)
                    st.dataframe(inferred_preview, use_container_width=True, hide_index=True)
                _remember_stage(
                    "Stage 2/7 - Preprocessing",
                    f"Prepared {len(df):,} clean reviews; {inferred_categories:,} row(s) have inferred/simple category labels.",
                )
                status.update(label=f"Stage 2/7 - {len(df):,} clean reviews", state="complete")
        progress.progress(28, text="Preprocessed")

        with slot_stage_3:
            with st.status("Stage 3/7 - Sentiment Analysis", expanded=True) as status:
                _stage_note(
                    f"{len(df):,} clean review rows with numeric ratings in [1, 5].",
                    "Deterministic rule: rating ≤ 2 → negative, 3 → neutral, ≥ 4 → positive. No ML inference — purely rule-based for reproducibility.",
                    "A `sentiment` column appended to every review row.",
                )
                _config_table(
                    [
                        ("Input column", "reviews.rating"),
                        ("Negative rule", "rating <= 2"),
                        ("Neutral rule", "rating == 3"),
                        ("Positive rule", "rating >= 4"),
                        ("Batch model use", "No model inference; deterministic rating mapping"),
                    ]
                )
                analyzer = get_sentiment_analyzer()
                df = analyzer.label_dataframe(df)
                sentiment_counts = df["sentiment"].value_counts().to_dict()
                m1, m2, m3 = st.columns(3)
                m1.metric("Positive", f"{sentiment_counts.get('positive', 0):,}", f"{sentiment_counts.get('positive', 0) / len(df):.1%}")
                m2.metric("Neutral", f"{sentiment_counts.get('neutral', 0):,}", f"{sentiment_counts.get('neutral', 0) / len(df):.1%}")
                m3.metric("Negative", f"{sentiment_counts.get('negative', 0):,}", f"{sentiment_counts.get('negative', 0) / len(df):.1%}")
                _remember_stage("Stage 3/7 - Sentiment Analysis", f"Rating-based sentiment labels: {sentiment_counts}.")
                status.update(label="Stage 3/7 - sentiment labeled", state="complete")
        progress.progress(42, text="Sentiment labeled")

        with slot_stage_4:
            with st.status("Stage 4/7 - Feature Extraction", expanded=True) as status:
                _n_products_s4 = df["name"].nunique()
                _method_desc_s4 = {
                    "Keyword taxonomy": "Keyword rules classify each product directly — no numeric feature matrix is built; clustering happens in Stage 5 by rule match.",
                    "TF-IDF terms": f"TF-IDF bag-of-words (unigrams + bigrams, sublinear_tf, English stopwords) over the selected columns: {', '.join(clustering_columns)}. Produces a sparse term-weight matrix.",
                    "TF-IDF + Agglomerative": f"TF-IDF bag-of-words (unigrams + bigrams, sublinear_tf) over {', '.join(clustering_columns)} — same features as TF-IDF terms; AgglomerativeClustering is applied in Stage 5 instead of KMeans.",
                    "MiniLM embeddings": f"Semantic sentence embeddings via `{EMBEDDING_MODEL_NAME}` over {', '.join(clustering_columns)}. Captures meaning beyond keyword overlap. Embeddings are cached to disk for reuse.",
                }[vector_method]
                _output_desc_s4 = (
                    "Keyword product labels (no numeric matrix)" if vector_method == "Keyword taxonomy"
                    else f"{'Sparse TF-IDF' if vector_method.startswith('TF-IDF') else f'Dense MiniLM ({EMBEDDING_MODEL_NAME})'} matrix of shape ({_n_products_s4:,} products × features)."
                )
                _stage_note(
                    f"{_n_products_s4:,} distinct products; one document per product built from: {', '.join(clustering_columns)}.",
                    _method_desc_s4,
                    _output_desc_s4,
                )
                _config_table(
                    [
                        ("Input columns", ", ".join(clustering_columns)),
                        ("Clustering unit", "one document per distinct product name"),
                        ("Feature method", vector_method),
                        ("MiniLM model", EMBEDDING_MODEL_NAME if vector_method == "MiniLM embeddings" else "not used"),
                        ("Keyword taxonomy", "direct keyword rules" if vector_method == "Keyword taxonomy" else "not used"),
                        ("TF-IDF settings", "word unigrams+bigrams, English stopwords, sublinear_tf" if vector_method.startswith("TF-IDF") or vector_method == "Keyword taxonomy" else "not used"),
                        ("Cache directory", OUTPUTS_DIR / "cache"),
                    ]
                )
                product_docs = _build_product_documents(df, clustering_columns)
                clustering_texts = product_docs["cluster_document"].astype(str).tolist()
                feature_terms: list[str] = []
                column_namespace = _slugify("_".join(clustering_columns))
                if vector_method.startswith("TF-IDF") or vector_method == "Keyword taxonomy":
                    cache_key = _embedding_cache_key(
                        clustering_texts,
                        namespace=f"product-cols-{column_namespace}-{_slugify(vector_method)}-v1",
                    )
                    embeddings, feature_terms = _build_tfidf_features(clustering_texts)
                    st.write(
                        f"Created TF-IDF feature matrix with shape `{embeddings.shape}` "
                        f"from {len(feature_terms):,} terms."
                    )
                else:
                    embeddings, embeddings_path, cache_key = _load_cached_embeddings(
                        clustering_texts,
                        namespace=f"product-cols-{column_namespace}-minilm-v1",
                    )
                    if embeddings is not None:
                        st.write(
                            f"Loaded cached product embeddings `{embeddings_path.name}` "
                            f"with shape `{embeddings.shape}`."
                        )
                    else:
                        clusterer = get_clusterer()
                        embeddings = clusterer.embed_reviews(clustering_texts)
                        _save_cached_embeddings(embeddings, embeddings_path, cache_key, len(product_docs))
                        st.write(
                            f"Created product embeddings with shape `{embeddings.shape}` and saved "
                            f"`{embeddings_path.name}` for future runs."
                        )
                if feature_terms:
                    st.caption(f"Example TF-IDF terms: {', '.join(feature_terms[:30])}")
                _show_example(
                    "product clustering documents",
                    product_docs,
                    ["name", "categories", "clustering_columns", "review_count", "cluster_document"],
                )
                _remember_stage("Stage 4/7 - Feature Extraction", f"{vector_method} feature matrix shape: {embeddings.shape}.")
                status.update(label=f"Stage 4/7 - {embeddings.shape[1]} features", state="complete")
        progress.progress(56, text="Embedded")

        with slot_stage_5:
            with st.status("Stage 5/7 - Clustering", expanded=True) as status:
                if vector_method == "Keyword taxonomy":
                    _cluster_proc_s5 = "Keyword taxonomy assigns a label directly from `infer_category` rules — no distance metric or k needed. Cache lookup is still performed but skipped on a miss."
                    _cluster_out_s5 = f"One cluster label per product from keyword rules, mapped back to all {len(df):,} review rows."
                elif k_mode == "Force k":
                    _algo = "AgglomerativeClustering(linkage='ward')" if vector_method == "TF-IDF + Agglomerative" else f'KMeans(k={forced_k}, random_state=42)'
                    _cluster_proc_s5 = f"Cache checked first. On miss: {_algo} runs at exactly k={forced_k}. Silhouette is still computed for comparison but does not override the forced k."
                    _cluster_out_s5 = f"Exactly {forced_k} cluster labels per product, mapped back to all {len(df):,} review rows."
                else:
                    _algo = "AgglomerativeClustering(linkage='ward')" if vector_method == "TF-IDF + Agglomerative" else "KMeans(random_state=42)"
                    _cluster_proc_s5 = f"Cache checked first. On miss: {_algo} sweeps k=2..{max_k_to_test}; the k with the highest silhouette score is selected."
                    _cluster_out_s5 = f"Best-k cluster labels (from k=2..{max_k_to_test}) per product, mapped back to all {len(df):,} review rows."
                _stage_note(
                    f"{embeddings.shape[0]:,} products × {embeddings.shape[1]} feature dimensions ({vector_method}).",
                    _cluster_proc_s5,
                    _cluster_out_s5 + " Silhouette scores, cluster sizes, PCA figure saved to disk.",
                )
                _config_table(
                    [
                        ("Input columns", ", ".join(clustering_columns)),
                        ("Clustering unit", "one row per distinct product name"),
                        ("Feature text", "Product-level document assembled from selected columns"),
                        ("Feature method", vector_method),
                        ("Embedding model", EMBEDDING_MODEL_NAME if vector_method == "MiniLM embeddings" else "not used"),
                        ("Algorithm", "keyword rules" if vector_method == "Keyword taxonomy" else ("AgglomerativeClustering(linkage='ward')" if vector_method == "TF-IDF + Agglomerative" else 'KMeans(random_state=42, n_init="auto")')),
                        ("k mode", k_mode),
                        ("Forced k", forced_k if k_mode == "Force k" and vector_method != "Keyword taxonomy" else "not used"),
                        ("Auto k range", f"2..{max_k_to_test}" if vector_method != "Keyword taxonomy" else "not used"),
                        ("Selection metric", "keyword label match" if vector_method == "Keyword taxonomy" else ("highest silhouette score" if k_mode == "Auto by silhouette" else "manual business choice")),
                    ]
                )
                n_samples = len(embeddings)
                clustering_cache_key = "_".join(
                    [
                        cache_key,
                        _slugify(vector_method),
                        _slugify(k_mode),
                        _slugify("_".join(clustering_columns)),
                        str(int(forced_k)),
                        str(int(max_k_to_test)),
                    ]
                )
                cached_labels, cached_k, cached_scores, labels_path = _load_cached_clustering(
                    clustering_cache_key,
                    n_samples,
                )
                if cached_labels is not None and cached_k is not None:
                    labels = cached_labels
                    best_k = cached_k
                    scores = cached_scores
                    st.write(
                        f"Loaded cached clustering `{labels_path.name}` "
                        f"with k={best_k}."
                    )
                else:
                    scores: dict[int, float] = {}
                    if vector_method == "Keyword taxonomy":
                        labels, taxonomy_names = _keyword_taxonomy_labels(product_docs)
                        best_k = len(set(labels.tolist()))
                        st.write(f"Keyword taxonomy created {best_k} label(s): {', '.join(taxonomy_names.values())}.")
                    else:
                        min_k = 2
                        max_k = min(int(max_k_to_test), n_samples - 1)
                        if n_samples < min_k + 1:
                            best_k = 1
                            st.write(f"Too few samples for silhouette; using k={best_k}.")
                        else:
                            for k in range(min_k, max_k + 1):
                                labels_k = _cluster_with_method(embeddings, k, vector_method)
                                scores[k] = float(silhouette_score(embeddings, labels_k))
                            if k_mode == "Force k":
                                best_k = max(1, min(int(forced_k), n_samples))
                                st.write(f"Forced k={best_k}. Silhouette scores are shown only for comparison.")
                            else:
                                best_k = max(scores, key=scores.get)
                                st.write(f"Selected k={best_k}.")
                        labels = _cluster_with_method(embeddings, best_k, vector_method)

                product_docs = product_docs.copy()
                product_docs["category_id"] = labels.astype(int)
                product_label_map = dict(zip(product_docs["name"], product_docs["category_id"]))
                df_clustered = df.copy()
                df_clustered["category_id"] = df_clustered["name"].astype(str).map(product_label_map).astype(int)
                df_clustered["category_name"] = df_clustered["category_id"].map(lambda value: f"Category {value + 1}")
                cluster_sizes = df_clustered["category_name"].value_counts().sort_index().to_dict()
                if cluster_sizes:
                    total_reviews = sum(cluster_sizes.values())
                    min_size = min(cluster_sizes.values())
                    max_size = max(cluster_sizes.values())
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Products", f"{len(product_docs):,}")
                    m2.metric("Clusters", str(best_k))
                    m3.metric("Largest cluster", f"{max_size:,}")
                    m4.metric("Smallest cluster", f"{min_size:,}")
                    if min_size < 0.05 * total_reviews:
                        smallest_name = min(cluster_sizes, key=cluster_sizes.get)
                        review_word = "review" if min_size == 1 else "reviews"
                        st.warning(
                            f"Cluster imbalance: **{smallest_name}** has only {min_size:,} {review_word} "
                            f"({min_size / total_reviews:.1%} of total). Consider a higher k or a different feature method."
                        )
                if cached_labels is None:
                    labels_path = _save_cached_clustering(
                        labels,
                        best_k,
                        scores,
                        clustering_cache_key,
                        n_samples,
                        cluster_sizes,
                    )
                    st.write(f"Saved clustering cache `{labels_path.name}` for future runs.")

                _make_figures(embeddings, labels, best_k, scores)
                _silhouette_altair_chart(scores, best_k)
                fig_cluster = FIGURES_DIR / "cluster_visualization.png"
                if fig_cluster.exists():
                    st.markdown("**Cluster visualization (PCA 2D)**")
                    col_img, _ = st.columns([1, 1])
                    with col_img:
                        st.image(str(fig_cluster), use_container_width=True)
                METRICS_DIR.mkdir(parents=True, exist_ok=True)
                _write_json(METRICS_DIR / "silhouette_scores.json", {str(k): v for k, v in scores.items()})
                _write_json(METRICS_DIR / "cluster_sizes.json", cluster_sizes)
                if scores:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"k": k, "silhouette_score": round(score, 4), "selected": k == best_k}
                                for k, score in scores.items()
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.markdown("**Cluster sizes**")
                st.dataframe(pd.DataFrame([{"cluster": k, "review_count": v} for k, v in cluster_sizes.items()]), hide_index=True)
                st.markdown("**Products per cluster**")
                st.dataframe(
                    product_docs.groupby("category_id")
                    .agg(product_count=("name", "size"), reviews=("review_count", "sum"))
                    .reset_index(),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**Cluster diagnostics for tuning**")
                st.dataframe(
                    _cluster_diagnostics(product_docs),
                    use_container_width=True,
                    hide_index=True,
                )
                _remember_stage("Stage 5/7 - Clustering", f"Selected k={best_k}; assigned {len(product_docs):,} products to clusters.")
                status.update(label=f"Stage 5/7 - {best_k} clusters", state="complete")
        progress.progress(70, text="Clustered")

        with slot_stage_6:
            with st.status("Stage 6/7 - Aggregation & Insights", expanded=True) as status:
                _stage_note(
                    f"{len(df_clustered):,} review rows assigned to {best_k} cluster(s) by {vector_method}.",
                    f"TF-IDF derives a readable label for each of the {best_k} clusters from: {', '.join(naming_columns)} (terms frequent in one cluster but rare in others). Products within each cluster are ranked by: 0.5 × avg_rating + 0.3 × positive_ratio + 0.2 × normalized_review_count. Complaint themes extracted from negative and neutral reviews.",
                    f"{best_k} named category insight records. Written to `clustered_reviews.csv`, `category_insights.json`, and `metrics/*.json`.",
                )
                _config_table(
                    [
                        ("Naming columns", ", ".join(naming_columns)),
                        ("Clustering columns", ", ".join(clustering_columns)),
                        ("Naming method", "TF-IDF top terms — rewards terms frequent in this cluster but rare across others"),
                        ("Ranking formula", "0.5 rating + 0.3 positive_ratio + 0.2 normalized review_count"),
                        ("Worst product rule", "lowest avg_rating; ties by most negative_reviews"),
                        ("Complaint source", "negative and neutral reviews.text"),
                        ("Output files", "clustered_reviews.csv, category_insights.json, metrics/*.json"),
                    ]
                )
                derived = derive_category_names(df_clustered, naming_columns=naming_columns)
                naming_rows = []
                for cid in sorted(derived):
                    terms = ", ".join(f"{term} ({weight:.3f})" for term, weight in derived[cid]["top_terms"]) or "-"
                    naming_rows.append({"category_id": cid, "derived_name": derived[cid]["name"], "top_terms": terms})
                st.dataframe(pd.DataFrame(naming_rows), use_container_width=True, hide_index=True)
                df_clustered["category_name"] = df_clustered["category_id"].astype(int).map(lambda cid: derived[int(cid)]["name"])
                insights = aggregate_category_insights(df_clustered)

                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                METRICS_DIR.mkdir(parents=True, exist_ok=True)
                clustered_output_path = _write_csv_with_fallback(
                    df_clustered,
                    OUTPUTS_DIR / "clustered_reviews.csv",
                )
                if clustered_output_path.name != "clustered_reviews.csv":
                    st.warning(
                        "`outputs/clustered_reviews.csv` is locked by another program, "
                        f"so this run was saved as `{clustered_output_path.name}` instead. "
                        "Close the open CSV file before the next run if you want the canonical file overwritten."
                    )
                _write_json(OUTPUTS_DIR / "category_insights.json", insights)
                _write_json(METRICS_DIR / "category_names.json", {str(k): v for k, v in derived.items()})
                sentiment_dist = {
                    str(item["category_id"]): {
                        "category_name": item["category_name"],
                        "review_count": item["review_count"],
                        **item["sentiment_ratio"],
                    }
                    for item in insights
                }
                _write_json(METRICS_DIR / "sentiment_distribution.json", sentiment_dist)
                _remember_stage("Stage 6/7 - Aggregation & Insights", f"Built {len(insights)} category insight records.")
                status.update(label=f"Stage 6/7 - {len(insights)} insights", state="complete")
        progress.progress(85, text="Aggregated")

        st.session_state.clustered_df = df_clustered
        st.session_state.insights = insights
        st.session_state.articles = {}

        with slot_stage_7:
            with st.status("Stage 7/7 - LLM Summarization", expanded=True) as status:
                _stage_note(
                    f"{len(insights)} aggregated category insight dictionaries ({', '.join(i['category_name'] for i in insights)}).",
                    "FLAN-T5 (`google/flan-t5-base`) generates one Markdown recommendation article per category from the insight summary only — raw review text is never sent to the model. Falls back to a deterministic Markdown template if generation fails.",
                    f"{len(insights)} Markdown file(s) written to `{BLOGPOSTS_DIR.name}/`.",
                )
                _config_table(
                    [
                        ("Input data", "aggregated category insight dictionaries"),
                        ("Raw review text sent to LLM", "no"),
                        ("Summary model", "google/flan-t5-base"),
                        ("Fallback", "deterministic Markdown article"),
                        ("Output directory", BLOGPOSTS_DIR),
                    ]
                )
                BLOGPOSTS_DIR.mkdir(parents=True, exist_ok=True)
                writer = get_writer()
                for i, insight in enumerate(insights, start=1):
                    prompt = build_safe_prompt(insight)
                    st.caption(f"Prompt — {insight['category_name']}")
                    st.code(prompt, language="text")
                    article = writer.generate(insight)
                    st.session_state.articles[insight["category_id"]] = article
                    path = BLOGPOSTS_DIR / f"{_slugify(insight['category_name'])}.md"
                    path.write_text(article, encoding="utf-8")
                    st.write(f"{i}/{len(insights)} wrote `{path.name}`")
                _remember_stage("Stage 7/7 - LLM Summarization", f"Wrote {len(insights)} Markdown recommendation articles.")
                status.update(label=f"Stage 7/7 - {len(insights)} articles written", state="complete")
        progress.progress(100, text="Complete")
        slot_summary.success(f"Processed {len(df_clustered):,} reviews into {len(insights)} categories.")

        _run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _run_record = {
            "timestamp": _run_ts,
            "config": {
                "input_mode": input_mode,
                "clustering_columns": clustering_columns,
                "naming_columns": naming_columns,
                "vector_method": vector_method,
                "k_mode": k_mode,
                "forced_k": int(forced_k),
                "max_k_to_test": int(max_k_to_test),
            },
            "results": {
                "best_k": int(best_k),
                "best_silhouette": round(float(scores[best_k]), 4) if scores and best_k in scores else None,
                "review_count": int(len(df_clustered)),
                "product_count": int(df_clustered["name"].nunique()),
                "categories": [
                    {
                        "id": ins["category_id"],
                        "name": ins["category_name"],
                        "top_terms": ", ".join(f"{t} ({w:.3f})" for t, w in (derived.get(ins["category_id"], {}).get("top_terms") or [])[:3]),
                        "review_count": ins["review_count"],
                        "avg_rating": ins["avg_rating"],
                        "positive_ratio": ins["sentiment_ratio"]["positive"],
                    }
                    for ins in insights
                ],
            },
        }
        _save_run_history(_run_record)
        st.session_state["_last_run_ts"] = _run_ts

    elif st.session_state.get("stage_summary"):
        with slot_summary:
            _render_stage_summary()

with tab_insights:
    _render_insights()

with tab_playground:
    _render_playground()

with tab_runs:
    _render_runs_tab()

with tab_artifacts:
    _render_artifacts()
