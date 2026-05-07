# RoboReviews

RoboReviews turns messy Amazon product reviews into practical buying recommendations. It cleans the Datafiniti review dataset, maps ratings to sentiment, clusters products into discovered categories, ranks the strongest and weakest products, pulls out common complaint themes, and writes recommendation-style summaries from structured insights.

It is built for local demos, Docker runs, and a straightforward AWS EC2 deployment.

## Architecture

```text
CSV upload
  -> preprocessing
  -> rating-based sentiment labels
  -> MiniLM review embeddings
  -> KMeans product clustering, k=4-6
  -> category aggregation and ranking
  -> FLAN-T5 recommendation article
  -> FastAPI + Streamlit
```

The original dataset category column is only used to validate the input shape. Product categories are discovered through clustering, as required.

## Dataset

Use this file:

```text
data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
```

Required columns:

```text
name
reviews.text
reviews.rating
categories
```

Ignore the May19, 1429_1, and submissions files.

## Setup locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Start the API:

```bash
uvicorn backend.main:app --reload
```

Open the API docs:

```text
http://localhost:8000/docs
```

Start Streamlit in another terminal:

```bash
streamlit run streamlit_app/app.py
```

Use `http://localhost:8000` as the API URL in the sidebar.

## Docker

From the project root:

```bash
docker compose up --build
```

Services:

```text
FastAPI:   http://localhost:8000
Docs:      http://localhost:8000/docs
Streamlit: http://localhost:8501
```

The compose file mounts `data/`, `outputs/`, and `models/` so uploads and generated insights survive container restarts.

## API

### GET `/health`

Checks whether the API is running.

Example response:

```json
{
  "status": "ok",
  "service": "robo-reviews"
}
```

### POST `/upload-reviews`

Upload the primary CSV as multipart form data.

```bash
curl -X POST http://localhost:8000/upload-reviews \
  -F "file=@data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"
```

Outputs are written to:

```text
outputs/clustered_reviews.csv
outputs/category_insights.json
```

### POST `/predict-sentiment`

```bash
curl -X POST http://localhost:8000/predict-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"The setup was easy and the battery life is great."}'
```

### POST `/cluster-products`

Posts review records directly and returns clustered insights. This is useful for tests or smaller demos.

### GET `/category-insights`

Returns the most recent saved insights.

### POST `/generate-summary`

Accepts one aggregated category insight object and returns a recommendation article. Raw reviews are never sent to the LLM.

## Ranking logic

For each discovered category, RoboReviews calculates:

```text
avg_rating
sentiment_ratio
review_count
```

Products are ranked with:

```text
score = 0.5 * rating + 0.3 * positive_ratio + 0.2 * review_count
```

In code, review count is normalized inside each category before scoring. That keeps high-volume products from overpowering rating quality.

Each category includes:

```text
Top 3 products
Worst product
Common complaints
```

## Streamlit demo

1. Start Docker with `docker compose up --build`.
2. Open `http://localhost:8501`.
3. Upload the primary Datafiniti CSV.
4. Click **Process reviews**.
5. Load the latest category insights.
6. Generate an article for any category.

## AWS EC2 deployment

A simple EC2 setup is enough for a demo or class deployment.

1. Launch an Ubuntu EC2 instance.
2. Allow inbound ports in the security group:
   - `22` for SSH
   - `8000` for FastAPI
   - `8501` for Streamlit
   - optionally `80` and `443` for Nginx
3. SSH into the instance:

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

4. Install Docker:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin git
sudo usermod -aG docker ubuntu
newgrp docker
```

5. Clone or copy the project:

```bash
git clone <your-repo-url> robo-reviews
cd robo-reviews
```

6. Add the dataset:

```bash
mkdir -p data/raw
scp -i your-key.pem Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv ubuntu@your-ec2-public-ip:~/robo-reviews/data/raw/
```

7. Run the stack:

```bash
docker compose up --build -d
```

8. Check containers:

```bash
docker compose ps
docker compose logs -f backend
```

Visit:

```text
http://your-ec2-public-ip:8000/docs
http://your-ec2-public-ip:8501
```

## Optional Nginx reverse proxy

Install Nginx:

```bash
sudo apt install -y nginx
```

Example API proxy:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Example output

A category insight looks like this:

```json
{
  "category_id": 0,
  "category_name": "Category 1",
  "avg_rating": 4.32,
  "review_count": 842,
  "sentiment_ratio": {
    "positive": 0.81,
    "neutral": 0.09,
    "negative": 0.10
  },
  "top_products": [
    {
      "name": "Example Fire Tablet",
      "avg_rating": 4.6,
      "positive_ratio": 0.88,
      "review_count": 210,
      "score": 2.764
    }
  ],
  "worst_product": {
    "name": "Example Remote",
    "avg_rating": 2.1,
    "negative_reviews": 14,
    "review_count": 35
  },
  "complaints": ["battery issues", "setup issues"]
}
```

A generated article gives a clear best overall pick, alternatives, concerns to watch, and a bottom line. The LLM only receives aggregated facts, not raw review text.

## Notes for production

The first model download can take a while and needs internet access. For locked-down EC2 environments, pre-download Hugging Face models into `models/` or bake them into a private image. For a public demo, keep the EC2 instance type reasonably sized; ML dependencies and model inference are more comfortable with at least 4 GB of RAM.
