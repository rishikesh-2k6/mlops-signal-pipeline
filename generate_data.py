"""
Generate synthetic 10,000-row OHLCV dataset with numpy seed 42.
Run once: python generate_data.py
"""
import numpy as np
import pandas as pd

SEED   = 42
N_ROWS = 10_000

np.random.seed(SEED)

close  = np.random.uniform(100, 200, N_ROWS)
spread = np.random.uniform(0.5, 3.0, N_ROWS)
high   = close + spread
low    = close - spread
open_  = close + np.random.uniform(-2, 2, N_ROWS)
volume = np.random.randint(100_000, 10_000_000, N_ROWS)

df = pd.DataFrame({
    "open":   np.round(open_, 4),
    "high":   np.round(high,  4),
    "low":    np.round(low,   4),
    "close":  np.round(close, 4),
    "volume": volume,
})

df.to_csv("data.csv", index=False)
print(f"Generated data.csv — {len(df)} rows")
