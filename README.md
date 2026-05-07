# RoboReviews

Turn messy Amazon reviews into clear buying advice.

Upload a review CSV → the app cleans it, labels sentiment, finds 4–6 product groups, ranks the best and worst per group, and writes a short recommendation article for each.

## Run it

The fastest way:

```bash
docker compose up --build
```

- Streamlit UI: http://localhost:8501
- API docs: http://localhost:8000/docs

Or locally:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload   # terminal 1
streamlit run streamlit_app/app.py  # terminal 2
```

## Use it

1. Open http://localhost:8501
2. Upload `data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`
3. Click **Process reviews**
4. Pick a category and **Generate** an article

## What's inside

```
CSV → preprocessing → sentiment → MiniLM embeddings → KMeans (k=4–6)
    → ranking → FLAN-T5 article → FastAPI + Streamlit
```

Ranking per category: `score = 0.5·rating + 0.3·positive_ratio + 0.2·review_count` (review count is normalised inside each category). Each category returns top 3, worst 1, and common complaint themes.

## API endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Health check |
| `POST /upload-reviews` | Upload a CSV and run the full pipeline |
| `POST /predict-sentiment` | Sentiment for a single review text |
| `POST /cluster-products` | Cluster a small batch of records inline |
| `GET /category-insights` | Return the latest saved insights |
| `POST /generate-summary` | Generate the article for one category |

The LLM only sees aggregated facts, never raw review text.

## Tests

```bash
pytest
```

## Deploy on EC2

1. Ubuntu instance, open ports 22 / 8000 / 8501
2. `sudo apt install -y docker.io docker-compose-plugin git`
3. Clone the repo, drop the CSV into `data/raw/`
4. `docker compose up --build -d`
5. Visit `http://<your-ip>:8501`

For locked-down hosts, pre-download the Hugging Face models into `models/` so the containers can run offline. ML inference is more comfortable with at least 4 GB RAM.
