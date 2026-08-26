"""
build_forecast_windows.py
═════════════════════════
Precompute one 4-week input window per district from the most recent
available scaled data, then save the result to app/weights/latest_windows.pkl.

This script must be run once before starting the FastAPI service, and
re-run whenever the underlying processed/scaled data is refreshed.

Usage (from the denguesense-ai-service project root):
    python scripts/build_forecast_windows.py

    # Or, specify custom paths via env vars:
    SCALED_DATA_DIR=/path/to/scaled  python scripts/build_forecast_windows.py

Output
──────
app/weights/latest_windows.pkl
    dict[int, dict] keyed by rdhs_id (0-25).  Each value is:
    {
        "sequence_input":  np.ndarray shape (4, 27)  float32
        "rdhs_id":         np.ndarray shape (1,)     int32
        "static_features": np.ndarray shape (4,)     float32
        "district_name":   str                        (for debugging)
    }

TODO (weekly retraining job)
─────────────────────────────
When new case/weather data arrives via the Spring Boot report pipeline or
PostGIS updates, the full preprocessing + scaling pipeline must be re-run
first (notebooks 00–05 in dengue_sense_lk/notebooks/), and then this
script re-run to refresh latest_windows.pkl.  The LSTM model itself may
be retrained on the extended dataset as part of that scheduled job.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Add project root to path so we can import app modules ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.lstm_forecaster import SEQ_COLS, STATIC_COLS, LOOKBACK

# ── Paths ─────────────────────────────────────────────────────────────────────
# Default: sibling project's processed data (override with SCALED_DATA_DIR env var)
_DEFAULT_DATA_DIR = (
    PROJECT_ROOT.parent / "dengue_sense_lk" / "data" / "processed" / "scaled"
)
SCALED_DATA_DIR = Path(os.environ.get("SCALED_DATA_DIR", str(_DEFAULT_DATA_DIR)))
OUTPUT_PATH = PROJECT_ROOT / "app" / "weights" / "latest_windows.pkl"


def load_latest_data(data_dir: Path) -> pd.DataFrame:
    """Load and merge all available scaled CSVs, preferring the most recent data.

    Strategy: load test.csv first (2025+), then val.csv (2024), then train.csv
    (2011-2023).  We only need the tail of each district's history — the most
    recent 4 weeks — so loading all three ensures we always have enough rows
    even for districts that appear only in older splits.
    """
    frames = []
    for split in ("test", "val", "train"):
        path = data_dir / f"{split}.csv"
        if path.exists():
            print(f"  Loading {path.name} …", end=" ")
            df = pd.read_csv(path, parse_dates=["week_start_date", "week_end_date"])
            print(f"{len(df):,} rows, years {df['year'].min()}–{df['year'].max()}")
            frames.append(df)
        else:
            print(f"  [SKIP] {path} not found")

    if not frames:
        raise FileNotFoundError(
            f"No scaled CSV files found in {data_dir}.\n"
            "Run the preprocessing notebooks (05_split_and_scaling.ipynb) first."
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined


def build_windows(df: pd.DataFrame) -> dict[int, dict]:
    """Extract the most recent 4-week window per district.

    For each district we sort chronologically, take the last LOOKBACK rows,
    and assemble the three named inputs expected by the LSTM.

    Returns
    -------
    dict[int, dict]
        Keyed by rdhs_id.  Each dict has the numpy arrays ready for
        model.predict(), plus a human-readable "district_name" field.
    """
    windows: dict[int, dict] = {}
    skipped: list[str] = []

    for rdhs_name, group in df.groupby("rdhs"):
        group = group.sort_values(["year", "week_no"]).reset_index(drop=True)

        if len(group) < LOOKBACK:
            print(
                f"  [WARN] {rdhs_name}: only {len(group)} rows "
                f"(need {LOOKBACK}), skipping."
            )
            skipped.append(rdhs_name)
            continue

        # Last LOOKBACK rows ──────────────────────────────────────────────────
        tail = group.iloc[-LOOKBACK:]
        rdhs_id = int(tail["rdhs_id"].iloc[-1])

        # Sequence input: (LOOKBACK, len(SEQ_COLS))
        seq = tail[SEQ_COLS].values.astype(np.float32)            # (4, 27)

        # Static input: snapshot from last row (constant per district)
        static = tail[STATIC_COLS].iloc[-1].values.astype(np.float32)  # (4,)

        # District id: scalar int for the Embedding layer
        rdhs_id_arr = np.array([rdhs_id], dtype=np.int32)         # (1,)

        windows[rdhs_id] = {
            "sequence_input":  seq,
            "rdhs_id":         rdhs_id_arr,
            "static_features": static,
            "district_name":   rdhs_name,
        }

    return windows, skipped


def main() -> None:
    print("=" * 60)
    print("DengueSense - build_forecast_windows.py")
    print("=" * 60)
    print(f"\nScaled data directory : {SCALED_DATA_DIR}")
    print(f"Output path           : {OUTPUT_PATH}\n")

    # Load data
    print("Loading scaled data ...")
    df = load_latest_data(SCALED_DATA_DIR)
    print(
        f"\nCombined dataset: {len(df):,} rows, "
        f"{df['rdhs'].nunique()} distinct districts\n"
    )

    # Validate required columns
    missing_cols = [c for c in SEQ_COLS + STATIC_COLS + ["rdhs", "rdhs_id", "year", "week_no"]
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns missing from scaled CSV: {missing_cols}\n"
            "Re-run the preprocessing pipeline to regenerate the scaled files."
        )

    # Build windows
    print("Building per-district windows ...")
    windows, skipped = build_windows(df)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(windows, OUTPUT_PATH)

    # Summary
    print("\n" + "-" * 60)
    print(f"  Districts with windows : {len(windows)}")
    if skipped:
        print(f"  Skipped (too few rows) : {skipped}")
    print(f"  Saved to               : {OUTPUT_PATH}")
    print("-" * 60)
    print("\nrdhs_id -> district name:")
    for rid, w in sorted(windows.items()):
        tail_info = f"(seq shape {w['sequence_input'].shape})"
        print(f"  {rid:2d}  {w['district_name']:<20} {tail_info}")
    print("\nDone. You can now start the FastAPI service.")


if __name__ == "__main__":
    main()
