"""FastAPI application for RoboReviews."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.schemas import (
    ClusterRequest,
    ClusterResponse,
    SentimentRequest,
    SentimentResponse,
    SummaryRequest,
    SummaryResponse,
    UploadResponse,
)
from src.aggregation import aggregate_category_insights, name_categories
from src.clustering import ProductClusterer
from src.config import OUTPUTS_DIR, RAW_DATA_DIR
from src.logging_config import configure_logging
from src.preprocessing import dataframe_from_records, load_reviews_csv, load_reviews_dir
from src.sentiment import SentimentAnalyzer
from src.summarization import RecommendationWriter

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="RoboReviews API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_analyzer = SentimentAnalyzer()
clusterer = ProductClusterer()
writer = RecommendationWriter()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send root visitors to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    """A tiny endpoint that load balancers and humans can both understand."""
    return {"status": "ok", "service": "robo-reviews"}


@app.post("/upload-reviews", response_model=UploadResponse)
async def upload_reviews(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a CSV and store cleaned/enriched outputs on disk."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        df = load_reviews_csv(tmp_path)
        enriched = sentiment_analyzer.label_dataframe(df)
        clustered = clusterer.cluster(enriched)
        clustered = name_categories(clustered)
        insights = aggregate_category_insights(clustered)

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        clustered.to_csv(OUTPUTS_DIR / "clustered_reviews.csv", index=False)
        (OUTPUTS_DIR / "category_insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")

        return UploadResponse(rows_loaded=len(clustered), message="Reviews processed and insights saved.")
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/process-raw-data", response_model=UploadResponse)
def process_raw_data() -> UploadResponse:
    """Merge every CSV under ``data/raw/`` and run the full pipeline against the result."""
    try:
        df = load_reviews_dir(RAW_DATA_DIR)
        enriched = sentiment_analyzer.label_dataframe(df)
        clustered = clusterer.cluster(enriched)
        clustered = name_categories(clustered)
        insights = aggregate_category_insights(clustered)

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        clustered.to_csv(OUTPUTS_DIR / "clustered_reviews.csv", index=False)
        (OUTPUTS_DIR / "category_insights.json").write_text(
            json.dumps(insights, indent=2), encoding="utf-8"
        )

        sources = sorted(clustered["source_file"].unique().tolist()) if "source_file" in clustered else []
        message = (
            f"Reviews processed and insights saved. Merged sources: {', '.join(sources)}"
            if sources
            else "Reviews processed and insights saved."
        )
        return UploadResponse(rows_loaded=len(clustered), message=message)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Raw-data processing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-sentiment", response_model=SentimentResponse)
def predict_sentiment(payload: SentimentRequest) -> SentimentResponse:
    """Predict sentiment for a single review-like sentence."""
    return SentimentResponse(sentiment=sentiment_analyzer.predict_text(payload.text))


@app.post("/cluster-products", response_model=ClusterResponse)
def cluster_products(payload: ClusterRequest) -> ClusterResponse:
    """Cluster posted review records and return category insights."""
    try:
        records = [item.model_dump(by_alias=True) for item in payload.records]
        df = dataframe_from_records(records)
        enriched = sentiment_analyzer.label_dataframe(df)
        clustered = clusterer.cluster(enriched, k=payload.k)
        clustered = name_categories(clustered)
        insights = aggregate_category_insights(clustered)
        return ClusterResponse(
            records_clustered=len(clustered),
            categories_found=len(insights),
            insights=insights,
        )
    except Exception as exc:
        logger.exception("Clustering failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/category-insights")
def category_insights() -> list[dict]:
    """Return the latest saved insights from an upload or pipeline run."""
    path = OUTPUTS_DIR / "category_insights.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No insights found yet. Upload reviews or run the pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/generate-summary", response_model=SummaryResponse)
def generate_summary(payload: SummaryRequest) -> SummaryResponse:
    """Generate a recommendation article from aggregated insights only."""
    try:
        article = writer.generate(
            payload.category_insight,
            temperature=payload.temperature,
            max_new_tokens=payload.max_new_tokens,
            tone=payload.tone,
            use_fallback=payload.use_fallback,
        )
        return SummaryResponse(article=article)
    except Exception as exc:
        logger.exception("Summary generation failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
