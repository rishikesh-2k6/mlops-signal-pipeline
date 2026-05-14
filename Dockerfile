# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# ── Install dependencies first (layer-cached) ─────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ────────────────────────────────────────────────────
COPY run.py          .
COPY config.yaml     .


COPY data.csv .


# ── Run the pipeline; exit code reflects success/failure ─────────────────────
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
