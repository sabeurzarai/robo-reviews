# RoboReviews — User Guide: Run and Deploy

## Dataset rule (read first)

Use only this file:

```
data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
```

Ignore: `*_May19.csv`, `1429_1.csv`, `submissions.csv`.

Required columns (literal dots in the names):

```
name
reviews.text
reviews.rating
categories
```

Categories are discovered through clustering (4–6 via silhouette). The original `categories` column is kept only to validate input shape — it is **not** used as the category labels.

---

## 1. What this guide covers

How to run RoboReviews locally, in Docker, and on AWS EC2. The project includes:

- Preprocessing + rating-based sentiment labeling
- TF-IDF terms → KMeans product clustering (default; also supports MiniLM embeddings, TF-IDF + Agglomerative, Keyword taxonomy)
- Per-category aggregation, ranking, and complaint mining
- FLAN-T5-base recommendation article generation (with repetition suppression and deterministic Markdown fallback)
- FastAPI backend, Streamlit demo UI, Docker Compose, EC2-friendly deploy

---

## 2. Project structure

```
robo-reviews/
├── backend/
│   ├── main.py            FastAPI app, endpoints, service singletons
│   └── schemas.py         Pydantic v2 models
├── src/
│   ├── config.py          Paths, model names, k bounds, weights
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── clustering.py
│   ├── aggregation.py
│   ├── summarization.py
│   ├── pipeline.py        run_from_csv entrypoint
│   └── logging_config.py
├── streamlit_app/
│   └── app.py
├── tests/
│   └── test_aggregation.py
├── data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
├── outputs/                clustered_reviews.csv, category_insights.json
├── models/                 HF model cache (mounted in Docker)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── ROBO_REVIEWS_USER_GUIDE_DEPLOYMENT.md
```

---

## 3. Prerequisites

- Python 3.11
- Git, pip, virtualenv/venv
- Docker Desktop (recommended)
- AWS account (only if deploying to EC2)
- ≥4 GB free RAM; first run downloads ~500 MB of HF models (MiniLM, DistilBERT, FLAN-T5-base)

---

## 4. Clone and set up locally

```bash
git clone <YOUR_REPO_URL>
cd robo-reviews
```

### Virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins exact versions of: `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic`, `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `transformers`, `torch`, `streamlit`, `requests`, `pytest`.

### Add the dataset

```bash
mkdir -p data/raw
# place the file at:
# data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
```

---

## 5. Run the pipeline as a script

```bash
python -c "from src.pipeline import RoboReviewsPipeline; RoboReviewsPipeline().run_from_csv('data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')"
```

Outputs (see [src/pipeline.py](src/pipeline.py)):

```
outputs/clustered_reviews.csv      reviews + sentiment + category_id/category_name
outputs/category_insights.json     per-category rankings, top/worst, complaints
```

---

## 6. Run the FastAPI backend

### From the terminal

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000/docs` (or `http://127.0.0.1:8000/docs`).

### From PyCharm (Run/Debug Configuration)

1. **Run → Edit Configurations… → +** → **Python**
2. Name it `FastAPI (uvicorn)`
3. Switch the top dropdown from **Script** to **Module name**
4. **Module name**: `uvicorn`
5. **Script parameters**: `backend.main:app --host 127.0.0.1 --port 8000 --reload`
6. **Working directory**: project root, e.g.
   `C:\Users\sisaz\PycharmProjects\IronHackProjects\Labs\[w6_d2_Project]\robo-reviews`
7. **Python interpreter**: the project's `.venv` (PyCharm usually picks this automatically)
8. **Apply → Run** (or **Debug** for breakpoints)

If your PyCharm version doesn't expose **Module name**, use the project's terminal instead:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

After it starts, open `http://127.0.0.1:8000/docs`.

### Endpoints

```
GET  /                      redirects to /docs
GET  /health
POST /upload-reviews        multipart CSV → runs full pipeline, writes outputs/
POST /predict-sentiment     single text → sentiment label (heuristic)
POST /cluster-products      JSON records → insights, no disk write
GET  /category-insights     latest saved insights JSON
POST /generate-summary      one insight dict → recommendation article
```

Health check:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"robo-reviews"}
```

---

## 7. Run the Streamlit UI

### From the terminal

```bash
streamlit run streamlit_app/app.py
```

Open: `http://localhost:8501`.

### From PyCharm (Run/Debug Configuration)

1. **Run → Edit Configurations… → +** → **Python**
2. Name it `Streamlit UI`
3. Switch the top dropdown from **Script** to **Module name**
4. **Module name**: `streamlit`
5. **Script parameters**: `run streamlit_app/app.py`
6. **Working directory**: project root (same as above)
7. **Python interpreter**: the project's `.venv`
8. **Apply → Run**

Fallback if **Module name** isn't available:

```bash
python -m streamlit run streamlit_app/app.py
```

The default API URL is `http://localhost:8000` (overridable via the `ROBO_REVIEWS_API_URL` env var; Docker Compose sets it to `http://backend:8000` automatically). Then:

- Upload the Datafiniti CSV (or use **Use all CSVs in data/raw**) → click **Process reviews**
- The pipeline runs all 7 stages automatically — Stage 7 article generation always runs (no checkbox required)
- Switch to the **Insights** tab to read per-category articles, complaints, and top products
- Expand any category → click **Regenerate** to re-run article generation on demand

**Feature method default**: `TF-IDF terms` is pre-selected. Change to `MiniLM embeddings` for semantic clustering or `Keyword taxonomy` for rule-based explainable labels.

**Compatible CSVs**: The file table marks each CSV as compatible or not. Only compatible files count toward the merged dataset — the Stage 1 and Stage 2 notes show the correct compatible file count, not the total.

### Running both backend + UI in PyCharm

The Streamlit UI calls into the FastAPI backend, so **both processes must be running at the same time**.

1. Click ▶ on **FastAPI (uvicorn)** — wait for `Application startup complete` in the Run window
2. Then click ▶ on **Streamlit UI** in a separate Run tab
3. Open `Alt+8` (**Services** tool window) — both configurations show side-by-side; click either to view its logs without stopping the other
4. Common failure mode: connection refused / `WinError 10061` in Streamlit → the FastAPI process stopped or never started. Restart FastAPI first.

You can also save both into a **Compound** Run Configuration (**Edit Configurations → + → Compound**) so a single ▶ launches them together.

### What to expect on the first `/upload-reviews`

The first time you click **Process reviews** with the full Datafiniti CSV (~95 MB):

1. HuggingFace downloads ~500 MB of models (MiniLM, DistilBERT, FLAN-T5) — **only on first run**, cached afterwards in `models/` (Docker) or your user HF cache (local)
2. MiniLM embeds every review
3. KMeans + silhouette runs over k=4..6
4. Aggregation writes `outputs/clustered_reviews.csv` and `outputs/category_insights.json`

End-to-end: **5–15 minutes on a laptop** the first time, faster on subsequent runs. Streamlit shows a spinner the whole time — don't click Process reviews twice. Watch the uvicorn Run window for download progress and pipeline log lines.

---

## 8. Run with Docker locally

The project ships a working `Dockerfile` and `docker-compose.yml`. Both API and UI build from the same image.

```bash
docker compose up --build
```

Services:

```
FastAPI:   http://localhost:8000/docs
Streamlit: http://localhost:8501
```

Compose mounts `./data`, `./outputs`, and `./models` so uploads, generated insights, and the HF model cache persist across container restarts.

Stop:

```bash
docker compose down
```

---

## 9. AWS EC2 deployment with Docker

> **Actual instance used**: `i-0d417541e6f5ad699`, region `eu-central-1` (Frankfurt), Elastic IP `18.157.233.122`, type `t3.micro`.

### 9.1 Launch instance

1. EC2 → **Launch Instance**
2. Ubuntu Server 22.04 LTS
3. Instance type: `t3.micro` works for demos; use `t3.medium`+ for comfortable LLM inference
4. Key pair: note the name (e.g. `key-0878c9c9551cee5f`) — you may not need the `.pem` if using CloudShell
5. Security group inbound rules — all set to `0.0.0.0/0`:

```
22    SSH        (EC2 Instance Connect / CloudShell)
8000  TCP        FastAPI
8501  TCP        Streamlit
3000  TCP        optional
80    HTTP       optional, for Nginx
443   HTTPS      optional, for Nginx
```

### 9.2 Connect via AWS CloudShell (recommended — no .pem needed)

Open **CloudShell** from the AWS console top bar. Make sure the region matches your instance (e.g. `eu-central-1`).

```bash
aws ec2-instance-connect ssh --instance-id i-0d417541e6f5ad699 --region eu-central-1 --os-user ubuntu
```

> **Important**: always pass `--os-user ubuntu` — without it EC2 Instance Connect defaults to `ec2-user` and gets `Permission denied (publickey)`.

Type `yes` when prompted about the host fingerprint. You are connected when you see `ubuntu@ip-172-31-x-x:~$`.

#### Alternative: connect with .pem from local PowerShell

```powershell
ssh -i "C:\Users\sisaz\Downloads\your-key.pem" ubuntu@18.157.233.122
```

If SSH times out, check that port 22 inbound rule source is `0.0.0.0/0` (not a stale IP).

### 9.3 Install Docker

> **Note**: `docker-compose-plugin` is not available in the default Ubuntu 24.04 apt repos. Install Docker engine first, then Docker Compose as a standalone binary.

```bash
sudo apt update && sudo apt upgrade -y
# A kernel upgrade may appear — reboot if prompted:
sudo reboot
# Reconnect after reboot, then:
sudo apt install -y git docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
exit
```

Reconnect (CloudShell):
```bash
aws ec2-instance-connect ssh --instance-id i-0d417541e6f5ad699 --region eu-central-1 --os-user ubuntu
```

Install Docker Compose standalone binary:
```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose version
# Expected: Docker Compose version v2.24.5
```

### 9.4 Clone the project on EC2

The project is on GitHub at `https://github.com/sabeurzarai/robo-reviews`:

```bash
git clone https://github.com/sabeurzarai/robo-reviews.git robo-reviews
cd robo-reviews
mkdir -p data/raw
```

> **Note**: datasets are excluded from git (too large for GitHub). They are stored in S3 and must be downloaded separately (see 9.5).

### 9.5 Install AWS CLI and download datasets from S3

The datasets are stored in S3 bucket `robo-reviews-data-024820689060-eu-central-1-an`. The EC2 instance must have the `ec2-s3-readonly` IAM role attached (S3ReadOnlyAccess).

**Attach IAM role** (first time only):
1. EC2 Console → select instance → **Aktionen → Sicherheit → IAM-Rolle ändern**
2. Select `ec2-s3-readonly` (has `AmazonS3ReadOnlyAccess` policy)
3. Click **IAM-Rolle aktualisieren**

**Install AWS CLI v2**:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install -y unzip
unzip awscliv2.zip
sudo ./aws/install
aws --version
```

**Download datasets**:
```bash
aws s3 cp s3://robo-reviews-data-024820689060-eu-central-1-an/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv data/raw/
aws s3 cp s3://robo-reviews-data-024820689060-eu-central-1-an/1429_1.csv data/raw/
aws s3 cp s3://robo-reviews-data-024820689060-eu-central-1-an/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv data/raw/
aws s3 cp s3://robo-reviews-data-024820689060-eu-central-1-an/submissions.csv data/raw/
```

### 9.6 Build and run

```bash
cd ~/robo-reviews
docker-compose up --build -d
docker-compose ps
docker-compose logs -f backend
```

Open:

```
http://18.157.233.122:8000/docs   ← FastAPI Swagger
http://18.157.233.122:8501        ← Streamlit UI
```

In the Streamlit sidebar, change the API URL to `http://18.157.233.122:8000`.

### 9.7 Update the deployed app

```bash
cd ~/robo-reviews
git pull
docker-compose down
docker-compose up --build -d
```

---

## 10. EC2 without Docker (alternative)

```bash
sudo apt install -y python3-pip python3-venv git
git clone <YOUR_REPO_URL> robo-reviews
cd robo-reviews
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# in a second session:
streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 8501
```

Use `tmux` or a `systemd` unit to keep them running.

---

## 11. Swagger / API test examples

Open `http://127.0.0.1:8000/docs` and try each endpoint. For each one below: payload to paste into the **Request body** field, plus the equivalent `curl`.

### 11.1 `GET /health`

No body. Click **Try it out → Execute**.

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{ "status": "ok", "service": "robo-reviews" }
```

### 11.2 `POST /upload-reviews`

Multipart form upload — Swagger gives you a file picker. Choose:

```
data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
```

```bash
curl -X POST http://127.0.0.1:8000/upload-reviews \
  -F "file=@data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
```

Expected (counts vary):

```json
{ "rows_loaded": 5897, "message": "Reviews processed and insights saved." }
```

Side effects: writes `outputs/clustered_reviews.csv` and `outputs/category_insights.json`. First call downloads MiniLM + DistilBERT + FLAN-T5 (~500 MB) and can take several minutes.

### 11.3 `POST /predict-sentiment`

Single text → label. Heuristic-only: reacts to a fixed hint-word list. Anything outside that list → `neutral`.

**Positive:**
```json
{ "text": "The setup was easy and the battery life is great." }
```
→ `{ "sentiment": "positive" }`

**Negative:**
```json
{ "text": "The remote is broken and the screen is terrible." }
```
→ `{ "sentiment": "negative" }`

**Neutral / no signal:**
```json
{ "text": "Arrived on time and matches the description." }
```
→ `{ "sentiment": "neutral" }`

**Mixed (tie → neutral):**
```json
{ "text": "Great picture but the sound is terrible." }
```
→ `{ "sentiment": "neutral" }`

```bash
curl -X POST http://127.0.0.1:8000/predict-sentiment \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"The setup was easy and the battery life is great.\"}"
```

To run several inputs, loop in bash:

```bash
for t in "great battery" "broken remote" "arrived on time"; do
  curl -s -X POST http://127.0.0.1:8000/predict-sentiment \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$t\"}"
  echo
done
```

### 11.4 `POST /cluster-products`

Send review records inline (note literal dots in field names — these are `ReviewRecord` aliases). Optional `k` overrides silhouette selection. Tiny payload returns 1 cluster; for real clustering, use `/upload-reviews` with the full CSV.

```json
{
  "records": [
    { "name": "Echo Plus",       "reviews.text": "great sound and easy setup",      "reviews.rating": 5, "categories": "Smart Home" },
    { "name": "Echo Plus",       "reviews.text": "love the alexa integration",       "reviews.rating": 5, "categories": "Smart Home" },
    { "name": "Fire TV Stick",   "reviews.text": "fast streaming and reliable",      "reviews.rating": 4, "categories": "Streaming" },
    { "name": "Fire TV Stick",   "reviews.text": "remote feels broken after a week", "reviews.rating": 2, "categories": "Streaming" },
    { "name": "Kindle Paperwhite","reviews.text": "perfect for reading at night",     "reviews.rating": 5, "categories": "E-Reader" },
    { "name": "Kindle Paperwhite","reviews.text": "battery life is excellent",        "reviews.rating": 5, "categories": "E-Reader" },
    { "name": "Fire Kids Tablet","reviews.text": "screen is slow and apps crash",    "reviews.rating": 1, "categories": "Tablet" },
    { "name": "Fire Kids Tablet","reviews.text": "bad value, had to return",         "reviews.rating": 1, "categories": "Tablet" }
  ],
  "k": 3
}
```

```bash
curl -X POST http://127.0.0.1:8000/cluster-products \
  -H "Content-Type: application/json" \
  -d @cluster_payload.json
```

Expected shape:

```json
{ "records_clustered": 8, "categories_found": 3, "insights": [ { "category_id": 0, ... } ] }
```

### 11.5 `GET /category-insights`

No body. Returns the latest `outputs/category_insights.json`. Run `/upload-reviews` first or you'll get a 404:

```json
{ "detail": "No insights found yet. Upload reviews or run the pipeline first." }
```

```bash
curl http://127.0.0.1:8000/category-insights
```

### 11.6 `POST /generate-summary`

Send **one** aggregated category insight (the kind of object returned by `/category-insights` or `/cluster-products`). Returns a recommendation article. **The LLM only ever sees this aggregated dict — never raw review text.**

```json
{
  "category_insight": {
    "category_id": 0,
    "category_name": "Category 1",
    "avg_rating": 4.56,
    "review_count": 1342,
    "sentiment_ratio": { "positive": 0.922, "neutral": 0.04, "negative": 0.038 },
    "top_products": [
      { "name": "Amazon Echo Plus", "avg_rating": 4.7, "positive_ratio": 0.949, "review_count": 256, "score": 2.803 },
      { "name": "Amazon Fire TV 4K", "avg_rating": 5.0, "positive_ratio": 1.0, "review_count": 3, "score": 2.802 },
      { "name": "Amazon Echo Show", "avg_rating": 4.6, "positive_ratio": 0.927, "review_count": 303, "score": 2.78 }
    ],
    "worst_product": {
      "name": "Fire Kids Edition Tablet",
      "avg_rating": 4.05, "negative_reviews": 2, "review_count": 20
    },
    "complaints": ["app issues", "return issues", "charge issues", "screen issues", "slow issues"]
  }
}
```

Quick chain — pull a real insight from disk and feed it back in:

```bash
curl -s http://127.0.0.1:8000/category-insights \
  | python -c "import json,sys; print(json.dumps({'category_insight': json.load(sys.stdin)[0]}))" \
  | curl -X POST http://127.0.0.1:8000/generate-summary \
      -H "Content-Type: application/json" -d @-
```

Returns:

```json
{ "article": "Best Overall: ...\n\nRunner Up: ...\n\nBottom Line: ..." }
```

The article text follows this structure:

```
Best Overall: <name>
Why we like it: <rating + sentiment + volume rationale>

Runner Up: <name>
Why it stands out: <rationale>

Also Consider: <name>
Best for: <use case>

Watch Out For: <worst product>
Reason: <rating / negative reviews / score>

Common Complaints:
- <complaint>
- <complaint>
- <complaint>

Bottom Line:
<short paragraph>
```

The pipeline uses `no_repeat_ngram_size=3`, `repetition_penalty=1.5`, `num_beams=4`, and `early_stopping=True` to prevent FLAN-T5-base from falling into repetition loops (e.g. "best value, best value…"). The quality check requires both **≥ 60 words** and a **unique-word ratio ≥ 0.15** — either failure triggers `_fallback_article`, which produces a clean deterministic Markdown article with the same sections. Stage 7 also displays the exact prompt sent to the model for each category.

---

## 12. Debugging

```bash
docker compose ps              # container status
docker compose logs -f         # tail all logs
docker compose logs -f backend # tail backend only
docker compose down            # stop
docker compose down && docker system prune -f && docker compose up --build -d  # rebuild clean

free -h                        # memory (EC2)
df -h                          # disk
sudo ss -tulnp                 # open ports
```

### Common issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` locally | `pip install -r requirements.txt` from project root |
| EC2 page won't load | Check security group ports 8000/8501; confirm app binds `0.0.0.0`; `docker compose ps` |
| Model download slow/fails | Pre-bake into `models/` (mounted as a volume), or use a larger instance with more bandwidth |
| Out of memory on EC2 | Move to ≥4 GB instance, or add a 4 GB swap file (see below) |
| `categories` column missing error | Input must include the `categories` column — it's required at load even though clustering ignores it |
| Empty after load | `load_reviews_csv` drops rows with non-numeric ratings or ratings outside `[1, 5]` — check the rating column |

### Swap (if needed)

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

---

## 13. Demo checklist

Before presenting:

- [ ] Dataset placed at `data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`
- [ ] `pytest` passes
- [ ] `/health` returns ok
- [ ] `POST /upload-reviews` with the CSV succeeds; `outputs/category_insights.json` is written
- [ ] Streamlit Insights tab shows category cards with top products, complaints list (worst-product JSON and sentiment bar chart removed)
- [ ] Stage 7 shows a prompt block per category and writes `.md` files to `outputs/blogposts/`
- [ ] `POST /generate-summary` returns a multi-section recommendation article
- [ ] Public URL reachable (if deployed)
- [ ] README and this guide reflect the current state

---

## 14. Command cheatsheet

```bash
# Local
pytest
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
streamlit run streamlit_app/app.py
python -c "from src.pipeline import RoboReviewsPipeline; RoboReviewsPipeline().run_from_csv('data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')"

# Docker
docker compose up --build           # foreground
docker compose up --build -d        # detached
docker compose down

# EC2 update cycle
cd ~/robo-reviews && git pull && docker compose down && docker compose up --build -d
```
