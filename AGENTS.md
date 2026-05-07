# RoboReviews — Codex Working Notes

# What this project does

Turns Amazon product reviews into recommendation articles. Cleans the Datafiniti review dataset, derives sentiment from ratings, clusters products into discovered categories (ignoring the original `categories` column), ranks them by a weighted score, mines complaint themes, and generates a per-category recommendation article.

Pipeline: **CSV → preprocess → sentiment → embed → cluster → aggregate → LLM article**

## Tech stack

- **Python 3.11**, FastAPI + Uvicorn (API), Streamlit (demo UI), Pydantic v2 (schemas)
- **ML**: scikit-learn (KMeans, silhouette, PCA, TF-IDF), sentence-transformers (`all-MiniLM-L6-v2`), transformers + torch (`distilbert-base-uncased-finetuned-sst-2-english` for sentiment classification, FLAN-T5 base for summaries)
- **Data**: pandas, numpy, matplotlib (artifact PNGs)
- **Deploy**: Docker + docker-compose → AWS EC2 Ubuntu
- **Tests**: pytest

# Where things live

```
backend/main.py        FastAPI app + service singletons
backend/schemas.py     Pydantic models (ReviewRecord aliases "reviews.text" etc.)
src/config.py          Paths, model names, k bounds, scoring weights — single source of truth
src/preprocessing.py   load_reviews_csv, normalize_text, validate_columns
src/sentiment.py       Rating→label mapping, DistilBERT touch-load
src/clustering.py      MiniLM + KMeans, silhouette-picked k
src/aggregation.py     Score formula, top/worst, complaint mining, TF-IDF cluster naming
src/summarization.py   FLAN-T5 + deterministic Markdown fallback
src/pipeline.py        RoboReviewsPipeline + main() — `python -m src.pipeline --csv …`
notebooks/01_eda.ipynb              EDA: schema, ratings, length, raw-categories sparsity
notebooks/02_evaluation.ipynb       Classifier P/R/F1 + confusion matrix, silhouette curve, ROUGE
streamlit_app/app.py
tests/test_aggregation.py
data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
outputs/                  clustered_reviews.csv, category_insights.json
outputs/blogposts/        per-category recommendation articles (Markdown)
outputs/metrics/          category_names.json, silhouette_scores.json, cluster_sizes.json, sentiment_distribution.json
outputs/figures/          silhouette_scores.png, cluster_visualization.png
models/                   HF cache target
Dockerfile, docker-compose.yml, requirements.txt
```

# How work gets done

## Commands

```bash
pytest                                            # tests
uvicorn backend.main:app --reload                  # API on :8000, /docs
streamlit run streamlit_app/app.py                 # UI on :8501
docker compose up --build                          # full stack (API + UI)
python -m src.pipeline                                        # full pipeline against the canonical Datafiniti CSV
python -m src.pipeline --csv path/to/reviews.csv              # arbitrary single CSV
python -m src.pipeline --from-dir data/raw                    # merge every compatible CSV in a directory
jupyter lab notebooks/                                         # EDA + evaluation notebooks
```

Docker sets `PYTHONPATH=/app` and `ROBO_REVIEWS_ROOT`. Locally, run from project root.

### Env vars

| Var | Default | Purpose |
|---|---|---|
| `ROBO_REVIEWS_ROOT` | project root | Base for `data/`, `outputs/`, `models/` |
| `ROBO_REVIEWS_DATA_DIR` / `OUTPUTS_DIR` / `MODELS_DIR` | derived from ROOT | Per-path overrides |
| `ROBO_REVIEWS_API_URL` | `http://localhost:8000` | Streamlit UI's default API URL. Compose sets it to `http://backend:8000` for the streamlit service. |

## Conventions

- `from __future__ import annotations` at top of every module.
- Imports: stdlib → third-party → `src.*`/`backend.*`, blank line between groups.
- Comments sparse; don't restate code. Tests use plain DataFrames, no fixtures (see [tests/test_aggregation.py](tests/test_aggregation.py)).
- Heavy models lazy-load via `@lru_cache(maxsize=1)`; reuse for any new model loader.
- Service objects (`SentimentAnalyzer`, `ProductClusterer`, `RecommendationWriter`) instantiated once at module import in [backend/main.py](backend/main.py).
- Artifacts → `outputs/`. HF model cache → `models/`. Both `.gitkeep`-only in git.

## Hard rules

1. **Dataset**: only `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`. Ignore `*_May19.csv`, `1429_1.csv`, `submissions.csv`.
2. **Columns required**: `name`, `reviews.text`, `reviews.rating`, `categories` (literal dots).
3. **Categories come from clustering, not the `categories` column** — that column is kept only to validate input shape.
4. **LLM never sees raw review text.** `summarization.build_safe_prompt` accepts only the aggregated insight dict. Preserve this for any new prompt path.
5. **Two label sources** live in [src/sentiment.py](src/sentiment.py) and must not be confused:
   - **Ground truth / batch label**: `sentiment_from_rating` (1–2 negative, 3 neutral, 4–5 positive). Used by `SentimentAnalyzer.label_dataframe` for the aggregation pipeline and as the eval target in [notebooks/02_evaluation.ipynb](notebooks/02_evaluation.ipynb).
   - **Model prediction**: `SentimentAnalyzer.predict_text` / `predict_texts` / `predict_dataframe` run `distilbert-base-uncased-finetuned-sst-2-english` on review text. SST-2 is binary; outputs with confidence below `NEUTRAL_CONFIDENCE_THRESHOLD` (default 0.85, in `config.py`) are remapped to `neutral`.

   Don't pass model predictions into `aggregate_category_insights` — it expects the rating-derived label. Don't evaluate the model against itself either; the whole point of keeping rating labels separate is to avoid leakage when computing precision/recall/F1.

## Scoring (per product, within each cluster)

```
score = 0.5 * avg_rating
      + 0.3 * positive_ratio
      + 0.2 * (review_count / max_review_count_in_category)
```

Review count is **normalized inside the category** before weighting ([src/aggregation.py:63](src/aggregation.py:63)). Worst product = lowest `avg_rating`, ties by most `negative_reviews`. Weights live in `config.py`.

## Clustering

MiniLM normalized embeddings + KMeans (`random_state=42`, `n_init="auto"`). `k` chosen by silhouette over `[MIN_CLUSTERS=4, MAX_CLUSTERS=6]`; `DEFAULT_CLUSTERS=5` falls back for tiny inputs. Backend (`src.clustering.ProductClusterer.cluster`) and the Streamlit Stage 5 inline both use KMeans. The Streamlit pipeline replicates the silhouette-loop inline (instead of calling `ProductClusterer.cluster`) so it can show per-k scores in the status window.

Cluster IDs are integers (`category_id`). Cluster **names** are derived via **TF-IDF over distinct product names** (`derive_category_names` / `name_categories` in [src/aggregation.py](src/aggregation.py)) — each cluster becomes one document built from its unique product names, TF-IDF surfaces terms that are frequent in this cluster but rare across the others, and the top 3 terms (title-cased, with known acronyms uppercased) form the name (e.g. `Speaker Alexa Enabled`, `Fire Tablet Kindle`). The fallback `Category {id+1}` is only used when there are <2 clusters or no token survives the stopword filter. Both API call sites (`/upload-reviews`, `/cluster-products`) and the Streamlit pipeline call `name_categories` between clustering and aggregation. The `pipeline.RoboReviewsPipeline.run_from_csv` script does the same.

## API

| Endpoint | Purpose |
|---|---|
| `GET  /` | Redirects to `/docs` (`include_in_schema=False`) |
| `GET  /health` | Liveness |
| `POST /upload-reviews` | multipart CSV → full pipeline → writes `outputs/clustered_reviews.csv` + `outputs/category_insights.json` |
| `POST /predict-sentiment` | Single text → label from DistilBERT SST-2; low-confidence → `neutral` (see rule 5) |
| `POST /cluster-products` | JSON records → insights, no disk write |
| `GET  /category-insights` | Latest saved `category_insights.json` |
| `POST /generate-summary` | One aggregated insight dict → `{ "article": str }` |

## Streamlit UI

[streamlit_app/app.py](streamlit_app/app.py) — four tabs, runs the pipeline **client-side** (imports `src.preprocessing/sentiment/clustering/aggregation/summarization` directly). Heavy services cached via `@st.cache_resource`.

1. **📥 Process Reviews** — 7-stage pipeline matching the project diagram, runs linearly on **▶ Process reviews**. Each stage opens its own `st.status` window:
   1. Data Input · 2. Preprocessing · 3. Sentiment Analysis · 4. Embedding · 5. Clustering · 6. Aggregation & Insights · 7. LLM Summarization (optional, gated by a checkbox; bulk-writes `outputs/blogposts/*.md`).
   - Stage 5 runs KMeans for each k in `[4, 6]`, picks the best by silhouette, writes `figures/silhouette_scores.png` (selected-k bar in green) + `figures/cluster_visualization.png` (PCA 2D) + `metrics/silhouette_scores.json` + `metrics/cluster_sizes.json`.
   - Stage 6 has two visible sub-steps: (a) **TF-IDF naming** — shows a per-cluster table of top terms with weights and the derived name, then writes `metrics/category_names.json`; (b) **Aggregation** — writes the standard outputs plus `metrics/sentiment_distribution.json`.
2. **📊 Insights** — per-category expander: sentiment distribution chart, rating histogram, top-products table, scatter (rating × volume × positive_ratio), worst product, complaints, product-level drill-down, Generate / Regenerate / Download `.md` for the article.
3. **🧪 Sentiment Playground** — calls `POST /predict-sentiment` with quick example buttons; the underlying classifier is DistilBERT SST-2 with a confidence threshold for the `neutral` bucket (rule 5).
4. **🗂️ Artifacts** — file tree of `outputs/` with ✅/❌ status, sizes, download buttons; PNG previews for the two figures.

`st.set_page_config` **must be the first Streamlit command** in the file (before any sidebar widget) or Streamlit raises `StreamlitSetPageConfigMustBeFirstCommandError`.

## `/generate-summary` article format

Returns JSON `{ "article": str }` with sections: **Best Overall** + "Why we like it", **Runner Up** + "Why it stands out", **Also Consider** + "Best for", **Watch Out For** (worst product) + "Reason", **Common Complaints** (bulleted), **Bottom Line**. Canonical layout in `_fallback_article` ([src/summarization.py](src/summarization.py)) — runs when FLAN-T5 returns <60 words or fails to load. **If the format changes, update prompt and fallback together.**

## Gotchas

- `reviews.text`, `reviews.rating` have literal dots. Pydantic aliases them; pandas uses bracket access.
- `load_reviews_csv` silently drops non-numeric ratings and anything outside `[1, 5]`.
- Complaint extraction uses a hard-coded `COMPLAINT_TERMS` list ([src/aggregation.py:14](src/aggregation.py:14)) — new domain → add terms.
- First run downloads ~500 MB of HF models; EC2 needs ≥4 GB RAM and outbound internet, or pre-baked `models/`.
- `categories` column is required at load but unused downstream — don't "clean it up", `validate_columns` will reject inputs without it.
- TF-IDF naming uses a hand-curated `CATEGORY_NAME_STOPWORDS` set ([src/aggregation.py](src/aggregation.py)). Universal noise (`amazon`, `new`, colors, sizes) is filtered, but distinctive brand/form-factor terms (`fire`, `echo`, `kindle`, `tablet`, `tv`, `hd`) are deliberately left in — TF-IDF naturally penalises terms that appear in many clusters. Add stopwords sparingly: a word that helps name *one* cluster won't hurt the others.
- **Pipeline logic is duplicated**: the Streamlit "Process Reviews" tab inlines the same dropna/dedupe/normalize/rating-filter steps as `src.preprocessing.load_reviews_csv`, **and** runs its own KMeans+silhouette loop (instead of calling `ProductClusterer.cluster`) so it can show per-k scores in the status window. Any change to preprocessing or k-selection logic must be made in **both places** to avoid drift.
- The Streamlit Process tab loads MiniLM, DistilBERT (touch-load), and (if Stage 7 is on) FLAN-T5 in-process — combined ~600 MB. If FastAPI is also running with its services warm, you'll have two copies of each (~1.2 GB total).
- Artifact subdirectories (`outputs/blogposts/`, `outputs/metrics/`, `outputs/figures/`) are created on demand by the Streamlit pipeline. The `/upload-reviews` API endpoint does **not** write them — only `clustered_reviews.csv` and `category_insights.json`. If you need parity, also write artifacts from `backend/main.py`.

## Deployment

FastAPI + Docker → AWS EC2 Ubuntu. Compose mounts `data/`, `outputs/`, `models/` to persist across restarts. Open ports 22, 8000, 8501 (optional 80/443 with Nginx — see [README.md](README.md)).
