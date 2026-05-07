# RoboReviews - Claude Working Notes

# What this project does

Turns Amazon product reviews into recommendation articles. The app loads compatible Amazon review CSVs, cleans and de-duplicates review rows, fills missing `categories` values with a transparent keyword heuristic, derives sentiment from ratings, clusters products into discovered categories, ranks products by a weighted score, mines complaint themes, and generates per-category recommendation articles.

Pipeline: **CSV -> preprocess -> sentiment -> feature extraction -> cluster -> aggregate -> LLM article**

## Tech stack

- **Python 3.11**, FastAPI + Uvicorn, Streamlit, Pydantic v2
- **ML**: scikit-learn (KMeans, silhouette, PCA, TF-IDF), sentence-transformers (`all-MiniLM-L6-v2`), transformers + torch (DistilBERT backbone, FLAN-T5 base)
- **Data**: pandas, numpy, matplotlib
- **Charts**: Altair (ships with Streamlit, no extra install) for the interactive silhouette chart; matplotlib for the PCA cluster visualization saved to disk
- **Deploy**: Docker + docker-compose -> AWS EC2 Ubuntu
- **Tests**: pytest

# Where things live

```text
backend/main.py        FastAPI app + service singletons
backend/schemas.py     Pydantic models (ReviewRecord aliases "reviews.text" etc.)
src/__init__.py        Bootstraps HF_HOME / SENTENCE_TRANSFORMERS_HOME -> MODELS_DIR
src/config.py          Paths, model names, k bounds, scoring weights
src/preprocessing.py   CSV/dir loading, text normalization, category inference, validation
src/sentiment.py       Rating->label mapping, DistilBERT touch-load, heuristic text prediction
src/clustering.py      Product-level documents from name + categories, MiniLM + KMeans
src/aggregation.py     Score formula, top/worst, complaint mining, TF-IDF category naming
src/summarization.py   FLAN-T5 + deterministic Markdown fallback
src/pipeline.py        RoboReviewsPipeline.run_from_csv / run_from_dir
streamlit_app/app.py   Tabbed demo UI and client-side pipeline
tests/test_aggregation.py
tests/test_preprocessing.py
data/raw/              Compatible Amazon review CSVs
outputs/               clustered_reviews.csv, category_insights.json (lock-safe write — falls back to a timestamped sibling if the canonical path is held open)
outputs/runs_history.json  Append-only log of every pipeline run (config + results); read by the Run History tab
outputs/cache/         MiniLM product embeddings + clustering labels (TF-IDF features are NOT cached; recomputed each run)
outputs/blogposts/     Per-category recommendation articles
outputs/metrics/       category_names.json, silhouette_scores.json, cluster_sizes.json, sentiment_distribution.json
outputs/figures/       silhouette_scores.png, cluster_visualization.png
models/                HuggingFace cache target
```

# How work gets done

## Commands

```bash
pytest
uvicorn backend.main:app --reload
streamlit run streamlit_app/app.py
docker compose up --build
python -c "from src.pipeline import RoboReviewsPipeline; RoboReviewsPipeline().run_from_dir()"
python -c "from src.pipeline import RoboReviewsPipeline; RoboReviewsPipeline().run_from_csv('data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')"
```

Docker sets `PYTHONPATH=/app` and `ROBO_REVIEWS_ROOT`. Locally, run from project root.

## Env vars

| Var | Default | Purpose |
|---|---|---|
| `ROBO_REVIEWS_ROOT` | project root | Base for `data/`, `outputs/`, `models/` |
| `ROBO_REVIEWS_DATA_DIR` / `RAW_DATA_DIR` / `OUTPUTS_DIR` / `MODELS_DIR` | derived from root | Per-path overrides |
| `ROBO_REVIEWS_API_URL` | `http://localhost:8000` | Streamlit default API URL |
| `HF_HOME` / `SENTENCE_TRANSFORMERS_HOME` | `MODELS_DIR` | HuggingFace cache location |

## Conventions

- `from __future__ import annotations` at top of every module.
- Imports: stdlib -> third-party -> `src.*`/`backend.*`, blank line between groups.
- Comments sparse; do not restate obvious code.
- Heavy models lazy-load via `@lru_cache(maxsize=1)`.
- Artifacts go under `outputs/`; model weights go under `models/`.

# Data Rules

1. Compatible CSVs must expose `name`, `reviews.text`, `reviews.rating`, and `categories` with literal dots in the review columns.
2. `load_reviews_dir` merges every compatible `*.csv` under `data/raw/`, skips incompatible schemas, and adds `source_file`.
3. Rows are dropped if `name`, `reviews.text`, or `reviews.rating` is missing.
4. `reviews.rating` is coerced to numeric; rows outside `[1, 5]` are dropped.
5. Duplicate reviews are removed by `name + reviews.text + reviews.rating`.
6. `reviews.text` is normalised: lowercased, HTML tags stripped, URLs removed, whitespace collapsed (`normalize_text`).
7. Category values go through a 5-step normalisation pipeline (`fill_missing_categories` → `normalize_category_path` → `_collapse_category`):

   **Step 1 — path extraction** (`normalize_category_path`)
   Amazon CSVs store breadcrumb paths as comma-separated strings. Keep only the last (most specific) segment and discard the rest.
   Example: `Electronics,Fire Tablets,Tablets,Computers & Tablets` → `Computers & Tablets`

   **Step 2 — blocklist removal** (`_collapse_category` + `_CATEGORY_BLOCKLIST`)
   The last segment is checked against a blocklist of known junk values — retailers (`Frys`, `Amazon`), promotional sections (`Holiday Shop`), vague internal labels (`Digital Device 3`, `Voice-Enabled Smart Assistants`, `Health Personal Care`), and connectivity specs that start with `wi-fi`, `3g`, `4g`, or `lte`. Matched segments are replaced with `""` so they fall through to inference.

   **Step 3 — keyword collapse** (`_collapse_category` + `_CATEGORY_COLLAPSE_KEYWORDS`)
   If exactly one keyword family matches the segment (case-insensitive substring), it is replaced with the canonical label.
   Examples: `Kids' Tablets` → `Tablet`, `Bluetooth Speakers` → `Speaker`, `Laptop Computers` → `Laptop`.
   Current canonical labels: `Tablet`, `Laptop`, `Computer`, `Speaker`, `Headphones`, `Camera`, `Keyboard`, `Cable`, `Battery Power`.

   **Step 4 — ambiguity blanking** (`_collapse_category`)
   If two or more keyword families match the segment, the category is ambiguous and cannot be safely collapsed to one label. The segment is replaced with `""` so the row falls through to inference.
   Example: `Computers & Tablets` matches both `computer` and `tablet` → blanked.

   **Step 5 — inference** (`infer_category`)
   Any row whose category is still blank after steps 1–4 (originally blank, blocklisted, or ambiguous) is re-inferred from `name + reviews.text`. The function walks `CATEGORY_INFERENCE_RULES` in priority order and returns the **first** matching label — never a compound `"Tablet > Cable"` string.
   Possible labels: `Laptop`, `Tablet`, `Cable`, `Case Sleeve Bag`, `Speaker Audio`, `Remote Streaming TV`, `Headphones`, `Battery Power`, `Camera`, `Keyboard Mouse`, `Uncategorized`.

# Sentiment

Batch sentiment is deterministic and rating-based:

- 1-2 -> negative
- 3 -> neutral
- 4-5 -> positive

DistilBERT is loaded as the required NLP backbone but is not used as a trained sentiment classifier. `predict_text` uses a small hint-word heuristic for the playground/API single-text endpoint.

# Clustering

The current Streamlit clustering path is product-level:

1. Build one document per distinct `name`.
2. The Streamlit UI lets the user choose clustering input columns from `name`, `categories`, `reviews.text`, `reviews.rating`, and `sentiment`.
3. Feature/clustering method is selectable in the UI:
   - `Keyword taxonomy`: direct keyword rules for explainable labels like Laptop, Tablet, Cable, Case, Speaker, etc.
   - `TF-IDF terms`: product/category terms -> KMeans.
   - `TF-IDF + Agglomerative`: product/category terms -> `AgglomerativeClustering(linkage="ward")`.
   - `MiniLM embeddings`: semantic embeddings -> KMeans.
4. Cluster count is selectable for unsupervised methods:
   - `Force k` (default `6`) for business categories like laptop/tablet/cable/case/speaker.
   - `Auto by silhouette` for mathematically broad groups.
5. `Keyword taxonomy` ignores forced/auto k because labels come directly from rules. In Streamlit, the k controls and silhouette controls are disabled when this method is selected.
6. Product-level cluster labels are mapped back to every review row by product name.

Important: Silhouette often prefers fewer broad clusters, so `Auto by silhouette` can choose `k=2` even when a human wants product-type categories. For recommendation categories, `Keyword taxonomy` or `Force k` plus TF-IDF is usually the better starting point.

The shared `src.clustering.ProductClusterer` also builds product-level documents from `name + categories` and maps product labels back to reviews. The Streamlit UI has extra controls and diagnostics that are not all exposed in the backend/script path. Specifically:

- The Streamlit silhouette sweep uses **`min_k=2`** (hard-coded in `streamlit_app/app.py`), not `MIN_CLUSTERS=4` from `src/config.py`. This intentionally lets `Auto by silhouette` choose a 2-cluster split if that's what the data prefers — the user can override with `Force k`.
- The Streamlit `_build_product_documents` honours user-selected input columns (`name`, `categories`, `reviews.text`, `reviews.rating`, `sentiment`). The backend's `ProductClusterer.build_product_documents` is fixed to `name + categories` only.
- The Streamlit Stage 5 also produces a **cluster diagnostics** table (per cluster: `product_count`, `review_count`, `example_products`, `top_source_categories`) that the backend does not.

## Clustering Diagnostics

Stage 5 shows:

- input columns
- feature method
- embedding/model configuration
- KMeans settings
- k mode and k range
- `st.metric()` cards: products / clusters / largest cluster / smallest cluster
- imbalance warning (`st.warning`) if any cluster is < 5% of total reviews
- interactive Altair silhouette chart (live run) / static PNG (stage summary)
- PCA cluster visualization (static PNG always)
- silhouette scores table
- cluster sizes table
- products per cluster
- example product names and top source categories per cluster

# Caching

Streamlit caches slow Stage 4/5 outputs in `outputs/cache/`. **Only MiniLM embeddings and clustering labels are persisted to disk** — TF-IDF features are recomputed every run because they are cheap, and Keyword taxonomy needs no features at all.

- **Embedding cache** (MiniLM only): `outputs/cache/review_embeddings_{key}.npz` + `.json` metadata. Cache key = SHA-256 prefix over the ordered product clustering documents, the embedding model name, namespace `product-cols-{slug(columns)}-minilm-v1`, and the row count. TF-IDF and Keyword taxonomy still compute a cache key but skip the load/save step (the key feeds Stage 5 only).
- **Clustering label cache**: `outputs/cache/review_clusters_{key}.npz` + `.json` metadata, holding labels, `best_k`, and the silhouette-score-per-k map. Cache key = `{embedding_cache_key}_{slug(method)}_{slug(k_mode)}_{slug(columns)}_{forced_k}_{max_auto_k}` so any control change invalidates the cache.
- Both caches validate row count on load and bail out cleanly on mismatch.
- The standalone **Clear cache** and **Load latest from disk** buttons were removed. The UI uses **Run mode** only for cache behavior:
  - `Reuse cache`
  - `Delete cache before processing` (calls `_clear_pipeline_cache` which `shutil.rmtree`s `outputs/cache/`)
- Deleting cache removes `outputs/cache/` only; it does not delete final outputs, figures, metrics, or blog posts.

# Scoring

Per product within each cluster:

```text
score = 0.5 * avg_rating
      + 0.3 * positive_ratio
      + 0.2 * (review_count / max_review_count_in_category)
```

Review count is normalized inside the category before weighting. Worst product = lowest `avg_rating`, ties by most `negative_reviews`.

# Aggregation and Naming

Cluster IDs are integers in `category_id`.

Stage 6 derives display names via TF-IDF using the **independently chosen** `naming_columns` (the "Input columns for cluster naming" multiselect, separate from the clustering columns). Only text columns (`name`, `categories`, `reviews.text`) are offered in that control — numeric/label columns carry no TF-IDF term signal. Falls back to `["name"]` if the selection is empty. It also computes:

- category average rating
- review count
- sentiment ratio
- top products
- worst product
- complaint themes

Complaint extraction uses a hard-coded `COMPLAINT_TERMS` list in `src/aggregation.py`.

# API

| Endpoint | Purpose |
|---|---|
| `GET /` | Redirects to `/docs` |
| `GET /health` | Liveness |
| `POST /upload-reviews` | Multipart CSV -> full pipeline -> writes `clustered_reviews.csv` and `category_insights.json` |
| `POST /process-raw-data` | Merge every CSV in raw data dir -> full pipeline |
| `POST /predict-sentiment` | Single text -> heuristic sentiment |
| `POST /cluster-products` | JSON records -> insights, no disk write |
| `GET /category-insights` | Latest saved `category_insights.json` |
| `POST /generate-summary` | One insight dict -> `{ "article": str }` |

# Streamlit UI

`streamlit_app/app.py` is tabbed: **Analytics Pipeline**, **Insights**, **Sentiment Playground**. The sidebar shows API URL, raw-files note, and a **Model cache** panel (MODELS_DIR path + on-disk MB via `_dir_size_mb`). `st.set_page_config` must remain the first Streamlit command.

Heavy services are cached via `@st.cache_resource`: `get_sentiment_analyzer`, `get_clusterer`, `get_writer`. Pipeline output is rendered into pre-declared slot containers (`slot_progress`, `slot_stage_1..7`, `slot_summary`) inside the Pipeline tab.

## Tab 1 - Analytics Pipeline

**Restart button** — top-right of the main content area, next to the title. Clears all session state (pipeline results, stage summary, articles) **and** resets every widget to its default value. Widget reset works via a `_reset_counter` stored in session state: incrementing it rotates every widget `key=` (e.g. `clustering_columns_{_rc}`), which forces Streamlit to recreate each widget from its `default=` parameter. `_reset_counter` is the only key preserved after a restart; everything else is cleared.

**Top controls** (always visible):
- Input source radio: `Use all CSVs in data/raw` (shows a per-file compatibility table) or `Upload a single CSV`.
- Clustering controls panel:
  - `Input columns for clustering` multiselect: any subset of `name`, `categories`, `reviews.text`, `reviews.rating`, `sentiment`. Default `categories`. Empty selection silently falls back to `name`.
  - `Input columns for cluster naming` multiselect: any subset of `name`, `categories`, `reviews.text` (text-only; numeric/label columns carry no TF-IDF signal). Default `categories`. Empty selection silently falls back to `name`. **Independent** from the clustering columns — the two controls can differ, allowing separate choices for grouping vs. labelling.
  - `Feature method` selectbox: `Keyword taxonomy`, `TF-IDF terms` (default), `TF-IDF + Agglomerative`, `MiniLM embeddings`.
  - `Cluster count` radio: `Force k` (default 6) or `Auto by silhouette`. Disabled for Keyword taxonomy.
  - `Forced k` and `Max auto k` number inputs (range 2-12). Disabled for Keyword taxonomy.
- `Run mode` radio: `Reuse cache` or `Delete cache before processing`.
- `Process reviews` primary button. Stage 7 always runs — there is no checkbox to skip it.

All interactive controls carry an explicit `key=f"<name>_{_rc}"` so that incrementing `_reset_counter` resets them to defaults.

**Selected pipeline parameters** table renders inside `slot_summary` at run start, summarizing input source, run mode, cache reuse yes/no, clustering columns, naming columns, feature method, and k mode/forced k/max auto k.

**Stages 1-7** each open in their own `st.status` window with a `_stage_note` (data / processing / output) and a `_config_table`. Stage notes are **parameter-aware** — they interpolate actual values from the current run rather than generic boilerplate:

1. **Data Input** — stage note shows filename (upload) or **compatible** file count + directory (multi-file). `n_compatible_csvs` is computed from the same compatibility check that populates the file table; incompatible CSVs are excluded from the count. Validates required columns; for upload mode reads bytes into a DataFrame and stamps `source_file`. Multi-file mode defers actual loading to Stage 2 via `load_reviews_dir`.
2. **Preprocessing** — stage note shows source type and compatible file count (`n_compatible_csvs`). Single-file path uses `_clean_uploaded_frame`; multi-file path calls `load_reviews_dir`. Shows three `st.metric()` cards (clean reviews / unique products / unique categories). In upload mode, `category_normalization_report` is called on the raw data before full cleaning and renders five metrics (kept / collapsed / junk→inferred / ambiguous→inferred / blank→inferred) plus a sample table of collapsed/ambiguous rows. Multi-file mode shows only the three summary metrics. Then shows the **category-frequency** table (top 50, exploded on `>|,`), example clean rows, and a sample of inferred-category rows.
3. **Sentiment Analysis** — stage note shows the exact review count from Stage 2. `analyzer.label_dataframe(df)` then three `st.metric()` cards showing positive / neutral / negative counts with percentage deltas.
4. **Feature Extraction** — stage note shows product count, selected clustering columns, and a method-specific sentence (sparse TF-IDF matrix / dense MiniLM embeddings / keyword labels). Builds product-level documents via `_build_product_documents`. TF-IDF and Keyword taxonomy compute features via `_build_tfidf_features`; MiniLM uses `_load_cached_embeddings` / `_save_cached_embeddings`. Shows feature matrix shape, example TF-IDF terms (first 30), and the product-document preview.
5. **Clustering** — stage note shows feature matrix shape, the exact algorithm string (KMeans/Agglomerative/keyword rules), k choice (forced k value / auto range / keyword rules). Checks `_load_cached_clustering` first. On miss: Keyword taxonomy uses `_keyword_taxonomy_labels`; other methods sweep `k=2..max_auto_k` with `_cluster_with_method`. `Force k` clusters at the chosen k while still computing silhouette for comparison. Shows four `st.metric()` cards (products / clusters / largest cluster / smallest cluster), imbalance warning if any cluster < 5% of reviews, interactive Altair silhouette chart, PCA figure, silhouette-per-k table, cluster-sizes table, products-per-cluster, and the **cluster diagnostics** table.
6. **Aggregation & Insights** — stage note shows review count, cluster count, feature method, and naming columns. `derive_category_names` called with the independently chosen `naming_columns` → renames clusters → `aggregate_category_insights`. Config table shows naming columns and clustering columns side-by-side. Writes `outputs/clustered_reviews.csv` via `_write_csv_with_fallback` (locked file falls back to timestamped sibling with warning). Writes `outputs/category_insights.json`, `metrics/category_names.json`, `metrics/sentiment_distribution.json`.
7. **LLM Summarization** — stage note shows category count and all category names. Always runs unconditionally. For each insight, displays the exact prompt via `st.caption` + `st.code` before calling `RecommendationWriter.generate(insight)`, then writes `outputs/blogposts/{slug}.md`. Articles can also be regenerated on demand from the Insights tab.

**Persistent stage summary**: `_remember_stage(stage, detail)` appends each completed stage into `st.session_state["stage_summary"]`. After the next Streamlit rerun (e.g. user expands an insight), the live `st.status` windows are gone; `_render_stage_summary` then fills `slot_summary` with collapsible expanders so the run remains visible. Stage 5's expander re-renders the silhouette and PCA figures from the saved PNGs via `_render_clustering_figures` (the Altair chart is only shown during the live run; the summary always uses the PNG fallback).

**After Stage 7**, `_save_run_history` appends a record to `outputs/runs_history.json` containing the full config and results snapshot (see Run History tab). A compact `_render_run_history` summary table is also rendered inline in `slot_run_history` (Pipeline tab, below Stage 7), marking the current run with `← this run`. The same slot shows the last-run history when the page is viewed without clicking Run.

**Artifacts panel** is in its own **Output Artifacts** tab via `_render_artifacts` — file rows + download buttons for core outputs, metrics, figures, and blog posts.

## Tab 2 - Insights

`_render_insights()`. Per-category expander showing rating/review/positive metrics, top-products table, complaints list, product drill-down, and Generate / Regenerate / Download `.md` controls. The worst-product JSON block and sentiment bar chart have been removed. Article generation calls `RecommendationWriter.generate` directly (in-process), writes the `.md` to `outputs/blogposts/`, and `st.rerun()`s.

## Tab 3 - Sentiment Playground

`_render_playground()`. Calls `POST /predict-sentiment` with quick-example buttons + hint-words expander. Includes a **Heuristic vs rating agreement** subsection (button-gated): scores up to 2,000 sampled reviews from `st.session_state.clustered_df`, shows overall agreement %, per-class agreement table, and confusion matrix. Result persists across reruns via `st.session_state["heuristic_eval"]`.

## Tab 4 - Run History

`_render_runs_tab()`. See **Run History** section below.

## Tab 5 - Output Artifacts

`_render_artifacts()`. File rows + download buttons for core outputs (`clustered_reviews.csv`, `category_insights.json`), metrics (`category_names.json`, `silhouette_scores.json`, `cluster_sizes.json`, `sentiment_distribution.json`), figures (`silhouette_scores.png`, `cluster_visualization.png`), and blog posts (all `*.md` files in `outputs/blogposts/`).

---

## Tab 4 details

`_render_runs_tab()`. Always visible; reads `outputs/runs_history.json` on every render (no caching — the file may have been updated by the latest run). Three sections:

1. **Overview table** — one row per run: timestamp, method, k, k-mode, clustering cols, naming cols, silhouette score, all cluster names pipe-separated, review count, product count. Sortable. **Download CSV** button exports all rows.

2. **Silhouette score chart** — horizontal Altair bar chart, one bar per run. Best silhouette highlighted green, others blue. Tooltip shows run label, score, method, and k. Caption explains silhouette semantics and its bias toward fewer clusters.

3. **Per-run cluster breakdown** — expandable cards, newest run first, latest run expanded by default and marked 🟢. Each expander shows four `st.metric()` cards (clusters / reviews / products / silhouette), a config caption line, then one card per cluster with: name, top 3 TF-IDF terms, review count, avg rating, and a positive-ratio `st.progress` bar. Cards laid out in rows of up to 4.

**Clear all history** button at the top wipes `runs_history.json` and reruns. The `_last_run_ts` session-state key tracks which run is "current" for the `← latest` marker across reruns.

### Run record schema

```json
{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "config": {
    "input_mode": "...",
    "clustering_columns": [...],
    "naming_columns": [...],
    "vector_method": "...",
    "k_mode": "...",
    "forced_k": 6,
    "max_k_to_test": 6
  },
  "results": {
    "best_k": 6,
    "best_silhouette": 0.312,
    "review_count": 1420,
    "product_count": 85,
    "categories": [
      {
        "id": 0,
        "name": "Cable",
        "top_terms": "cable (1.000), usb (0.800), charger (0.600)",
        "review_count": 72,
        "avg_rating": 4.39,
        "positive_ratio": 0.847
      }
    ]
  }
}
```

# LLM Summary Safety

`summarization.build_safe_prompt` accepts only the aggregated insight dict. Raw review text must not be sent to the summarizer. Keep prompt and deterministic fallback format aligned if the article structure changes.

`load_summarizer` passes `no_repeat_ngram_size=3`, `repetition_penalty=1.5`, `num_beams=4`, and `early_stopping=True` to the pipeline to prevent repetition loops (a known failure mode of FLAN-T5-base). The quality check in `RecommendationWriter.generate` requires both ≥ 60 words **and** a unique-word ratio ≥ 0.15 — either failure falls through to the deterministic Markdown fallback. A repetition loop like "best value, best value…" fails the ratio check even if it is long enough to pass the word count.

# Gotchas

- `reviews.text` and `reviews.rating` have literal dots; use bracket access in pandas.
- Missing `categories` are filled, not dropped, but the column itself is still required in the input schema.
- The Streamlit UI has extra clustering controls that are not all mirrored in the FastAPI endpoints.
- Streamlit reruns on widget interaction; persistent results live in `st.session_state`. The pipeline status windows disappear after a rerun — `stage_summary` is what keeps the run visible.
- Cached clusters can make debugging confusing. Choose **Delete cache before processing** before judging a changed clustering setup. Any change to clustering controls invalidates the cluster cache (the controls feed the key); but TF-IDF feature changes do *not* — features are recomputed every run.
- First model run downloads ~500 MB of HuggingFace weights into `MODELS_DIR` (`models/` by default; configured via `HF_HOME`/`SENTENCE_TRANSFORMERS_HOME` set in `src/__init__.py`). Subsequent runs reuse the local cache. TF-IDF and Keyword-taxonomy feature methods avoid the MiniLM download in Stage 4 entirely. The sidebar's **Model cache** panel shows the current path + on-disk MB.
- HF cache bootstrap **must run before** any `transformers` / `sentence_transformers` import, which is why it lives in `src/__init__.py` (Python runs the package `__init__` on the first `from src.X import Y`). New entrypoints that import transformers before `src` need to set `HF_HOME` manually.
- Streamlit's silhouette sweep starts at **`min_k=2`**, hard-coded in `streamlit_app/app.py` — different from `MIN_CLUSTERS=4` in `src/config.py`. Don't "fix" one to match the other without checking both call sites.
- `outputs/clustered_reviews.csv` is written via `_write_csv_with_fallback`. If the file is held open (e.g. Excel on Windows), the run silently lands at `clustered_reviews_{timestamp}.csv` and shows a warning. The Insights and Artifacts views still read the canonical path, so close the open file and rerun if you want the dashboard to pick up the latest.
- Artifact subdirectories are created on demand.
- `fill_missing_categories` normalises paths **before** inferring — this is intentional. Reverting to the old order (infer then normalise) would mean ambiguous paths like "Computers & Tablets" are never re-inferred from the review text.
- `_collapse_category` uses a set of matched canonical labels to detect ambiguity. Adding a new keyword to `_CATEGORY_COLLAPSE_KEYWORDS` can change whether an existing category becomes ambiguous or collapsed — re-run the pipeline and check the Stage 2 normalization report after any changes to that list.
- The silhouette chart has two rendering paths: during a live pipeline run it is an interactive Altair chart (`_silhouette_altair_chart`); in the persistent stage summary it falls back to the saved PNG via `_render_clustering_figures`. Do not remove the `_make_figures` PNG save — the summary depends on it.
- `_config_table` uses `column_config` with `width="large"` on the value column so long paths and setting values are not truncated. Do not remove this — plain `st.dataframe` will clip values again.
- The **Restart** button uses a `_reset_counter` in session state rather than a plain `st.session_state.clear()`. The counter increments on each restart; every widget has `key=f"<name>_{_rc}"` so changing the counter forces Streamlit to recreate all widgets from their `default=` values. `_reset_counter` is the only key re-seeded after the clear — everything else (pipeline results, stage summary) stays wiped. A plain `clear()` without rotating keys would leave widget values unchanged on the next render.
- `_stage_note` text is interpolated at render time from live variables (`df`, `embeddings`, `clustering_columns`, `vector_method`, `k_mode`, `forced_k`, etc.). Editing parameter-specific sentences in Stage 4/5/6 requires updating both the f-string in `app.py` and any related `_config_table` rows — they should stay consistent.
- `outputs/runs_history.json` is an append-only list — every run adds one record. It is never auto-truncated. Use the **Clear all history** button in the Run History tab to reset it. The Run History tab reads this file fresh on every render; there is no in-memory cache.
- The `_last_run_ts` session-state key is the only bridge between the Pipeline tab and the Run History tab for marking the current run. It is cleared on Restart and on "Clear all history". If it is `None`, no run is marked `← latest`.

# Deployment

FastAPI + Docker -> AWS EC2 Ubuntu. Compose mounts `data/`, `outputs/`, and `models/` to persist across restarts. Open ports 22, 8000, and 8501, with optional 80/443 behind Nginx.
