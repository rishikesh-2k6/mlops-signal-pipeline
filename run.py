"""
MLOps Batch Processing Pipeline
Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
"""

import argparse
import io
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="MLOps Batch Signal Processing Pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to both stdout and a file."""
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def write_metrics(output_path: str, payload: dict, logger: logging.Logger):
    """Write metrics JSON — always called, even on failure."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Metrics written to %s", output_path)
    except Exception as exc:
        logger.error("Failed to write metrics to %s: %s", output_path, exc)


def load_config(config_path: str, logger: logging.Logger) -> dict:
    """Parse YAML config and validate required keys."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found: {config_path}")
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse config YAML '{config_path}': {exc}")

    required_keys = ["seed", "window", "version"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise RuntimeError(
            f"Config '{config_path}' is missing required key(s): {', '.join(missing)}"
        )

    # Validate types
    if not isinstance(cfg["seed"], int):
        raise RuntimeError(f"Config 'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise RuntimeError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str):
        raise RuntimeError(f"Config 'version' must be a string, got: {type(cfg['version']).__name__}")

    logger.info(
        "Config loaded — seed=%s, window=%s, version=%s",
        cfg["seed"], cfg["window"], cfg["version"]
    )
    return cfg


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV that may have every row wrapped in outer double-quotes.

    Some export tools write each row as a single quoted string, e.g.:
        "timestamp,open,high,low,close,volume_btc,volume_usd"
        "2024-01-01 00:00:00,44910.83,..."

    pandas.read_csv treats the whole line as one value in that case.
    This helper detects that pattern and strips the wrapping quotes before
    parsing, then falls back to a normal read for standard CSVs.
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Detect whole-row quoting: header line starts AND ends with '"'
    # and contains commas inside (not just an empty or simple quoted field)
    header = lines[0].strip() if lines else ""
    if header.startswith('"') and header.endswith('"') and "," in header[1:-1]:
        # Strip the outer quote from every non-empty line
        cleaned = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]  # remove exactly the outermost quotes
            cleaned.append(line)
        return pd.read_csv(io.StringIO("\n".join(cleaned)))

    # Standard CSV — let pandas handle it normally
    return pd.read_csv(path)


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load CSV dataset and validate it.

    Only the 'close' column is required; extra columns are ignored.
    Handles both standard CSVs and files where every row is wrapped in
    outer double-quotes (e.g. exports from Excel / Google Sheets).
    """
    try:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        df = _read_csv_robust(path)

    except FileNotFoundError as exc:
        raise RuntimeError(str(exc))
    except PermissionError:
        raise RuntimeError(f"Permission denied — cannot read file: {input_path}")
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"Input file is empty: {input_path}")
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"Failed to parse CSV '{input_path}': {exc}")
    except Exception as exc:
        raise RuntimeError(f"Unexpected error reading '{input_path}': {exc}")

    if df.empty:
        raise RuntimeError(f"Input file loaded but contains no rows: {input_path}")

    if "close" not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise RuntimeError(
            f"Required column 'close' not found in '{input_path}'. "
            f"Available columns: {available}"
        )

    # Ensure close is numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().all():
        raise RuntimeError(
            f"Column 'close' in '{input_path}' contains no valid numeric values."
        )

    logger.info("Dataset loaded — %d rows, %d columns", len(df), len(df.columns))
    return df


def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    """Compute rolling mean on 'close' column."""
    rolling_mean = df["close"].rolling(window=window).mean()
    logger.info(
        "Rolling mean computed — window=%d, valid rows (non-NaN): %d",
        window,
        rolling_mean.notna().sum()
    )
    return rolling_mean


def generate_signal(df: pd.DataFrame, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    """Generate binary signal: 1 if close > rolling_mean else 0. NaN rows excluded."""
    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=df.index)
    signal[valid_mask] = (df["close"][valid_mask] > rolling_mean[valid_mask]).astype(int)
    logger.info(
        "Signal generated — valid signal rows: %d, NaN warm-up rows excluded: %d",
        valid_mask.sum(),
        (~valid_mask).sum()
    )
    return signal


def main():
    start_time = time.time()

    args = parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Job START — %s", time.strftime("%Y-%m-%dT%H:%M:%S"))
    logger.info("Input:  %s", args.input)
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", args.output)
    logger.info("Log:    %s", args.log_file)
    logger.info("=" * 60)

    version = "unknown"  # populated after config load for error payloads

    try:
        # ── 1. Load & validate config ─────────────────────────────────────────
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]

        np.random.seed(seed)

        # ── 2. Load & validate dataset ────────────────────────────────────────
        df = load_dataset(args.input, logger)

        # ── 3. Compute rolling mean ───────────────────────────────────────────
        rolling_mean = compute_rolling_mean(df, window, logger)

        # ── 4. Generate binary signal ─────────────────────────────────────────
        signal = generate_signal(df, rolling_mean, logger)

        # ── 5. Compute metrics ────────────────────────────────────────────────
        valid_signal = signal.dropna()
        rows_processed = int(len(valid_signal))
        signal_rate    = round(float(valid_signal.mean()), 4)
        latency_ms     = int((time.time() - start_time) * 1000)

        logger.info(
            "Metrics — rows_processed=%d, signal_rate=%.4f, latency_ms=%d",
            rows_processed, signal_rate, latency_ms
        )

        payload = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          signal_rate,
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        write_metrics(args.output, payload, logger)
        print(json.dumps(payload, indent=2))

        logger.info("=" * 60)
        logger.info("Job END — status=success")
        logger.info("=" * 60)
        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        error_msg  = str(exc)

        logger.error("Pipeline error: %s", error_msg)
        logger.error("Traceback:\n%s", traceback.format_exc())

        error_payload = {
            "version":       version,
            "status":        "error",
            "error_message": error_msg,
            "latency_ms":    latency_ms,
        }

        write_metrics(args.output, error_payload, logger)
        print(json.dumps(error_payload, indent=2))

        logger.info("=" * 60)
        logger.info("Job END — status=error")
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
