FROM python:3.12-slim

# System dependencies for XGBoost and scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Generate data and train models during build (no .pkl files needed in repo)
RUN python build_data_and_models.py && python train_models.py

# Render sets PORT env var; default to 10000
ENV PORT=10000
EXPOSE 10000

# Single worker to fit within 512MB free tier RAM
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120 app:app"]
