FROM python:3.11-slim

# Build tools are needed by a few ML wheels on some platforms. Keeping this slim
# image small matters on EC2 where disk is often the first annoying bottleneck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV ROBO_REVIEWS_ROOT=/app

EXPOSE 8000 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
