# mlops-signal-pipeline

Python batch job with config-driven signal generation, structured logging, JSON metrics output, and Docker deployment.

## MLOps Batch Signal Processing Pipeline

A production-style MLOps batch processing application that ingests OHLCV market data, computes a rolling-mean-based binary signal, and outputs structured metrics — with full logging, error handling, and Docker support.

---

## Project Structure

```
primetrade/
├── run.py            # Main pipeline entry point
├── config.yaml       # Pipeline configuration
├── data.csv          # Synthetic 10,000-row OHLCV dataset
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition
├── generate_data.py  # (Dev) Script to regenerate data.csv
├── metrics.json      # Output — written on every run
└── run.log           # Output — log file written on every run
```

---

## Requirements

- Python 3.9+
- Docker (for containerised runs)

---

## Local Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset

```bash
python generate_data.py
```

### 3. Run the pipeline

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

The script will:
- Load and validate `config.yaml`
- Load and validate `data.csv`
- Compute a rolling mean on `close` with `window=5`
- Generate a binary signal (1 if `close > rolling_mean` else 0)
- Write `metrics.json` and print it to stdout
- Exit `0` on success, non-zero on failure

---

## Docker Build & Run

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

The container runs the full pipeline and exits. The exit code reflects success (`0`) or failure (non-zero).

### Run with custom output (mount a volume)

```bash
docker run --rm -v "$(pwd)/output:/app/output" mlops-task \
  python run.py --input data.csv --config config.yaml \
                --output output/metrics.json --log-file output/run.log
```

---

## Configuration (`config.yaml`)

| Key       | Type    | Description                              |
|-----------|---------|------------------------------------------|
| `seed`    | int     | NumPy random seed for reproducibility    |
| `window`  | int     | Rolling mean window size                 |
| `version` | string  | Pipeline version tag written to output   |

---

## Output

### `metrics.json` — Success

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.499,
  "latency_ms": 127,
  "seed": 42,
  "status": "success"
}
```

### `metrics.json` — Error

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found in 'data.csv'.",
  "latency_ms": 12
}
```

---

## Signal Logic

| Condition              | Signal |
|------------------------|--------|
| `close > rolling_mean` | `1`    |
| `close ≤ rolling_mean` | `0`    |

The first `window - 1` rows are excluded from signal computation (rolling mean warm-up).

---

## Reproducibility

Given the same `config.yaml` and `data.csv`, every run produces **identical output** — ensured by `numpy.random.seed(seed)` and deterministic pandas operations.

---

## Error Handling

The pipeline gracefully handles:
- Missing config file
- Invalid / missing YAML keys
- Missing input CSV
- Unreadable / corrupt CSV
- Empty dataset
- Missing `close` column

In all error cases, `metrics.json` is **always written** with `"status": "error"` and a descriptive `error_message`.
>>>>>>> b87d49d (ML Engineering Internship - Task 0)
