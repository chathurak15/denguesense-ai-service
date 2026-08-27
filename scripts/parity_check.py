"""
Parity check: serving feature-engineering vs the ground-truth scaled val.csv.

For each validation (2024) lookback window per district we:
  1. take 8 raw weekly rows (the 4 lookback weeks + 4 preceding weeks for lags)
     from the UNscaled feature-engineered CSV,
  2. run the serving `build_model_inputs` on them,
  3. compare the resulting scaled (4, 27) sequence to the scaled SEQ_COLS values
     already stored in val.csv for the exact same weeks.

Any column whose max abs diff is not ~0 is a train/serve mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.features.feature_engineering import (  # noqa: E402
    LOOKBACK,
    SEQ_COLS,
    build_model_inputs,
    load_temp_zscore_baselines,
)

SIBLING = PROJECT_ROOT.parent / "dengue_sense_lk" / "data" / "processed"
FEAT_CSV = SIBLING / "dengue_2011_2026_feature_engineering_&_encoding.csv"
VAL_CSV = SIBLING / "scaled" / "val.csv"
WEIGHTS = PROJECT_ROOT / "app" / "weights"


def main() -> None:
    feat = pd.read_csv(FEAT_CSV, parse_dates=["week_start_date", "week_end_date"])
    val = pd.read_csv(VAL_CSV, parse_dates=["week_start_date", "week_end_date"])
    feature_scaler = joblib.load(WEIGHTS / "feature_scaler.pkl")
    target_scaler = joblib.load(WEIGHTS / "target_scaler.pkl")
    baselines = load_temp_zscore_baselines(WEIGHTS / "temp_zscore_baselines.json")

    feat = feat.sort_values(["rdhs", "week_start_date"]).reset_index(drop=True)

    # per-column accumulated abs diff
    diffs = {c: [] for c in SEQ_COLS}
    n_windows = 0
    districts = sorted(val["rdhs"].unique())

    for district in districts:
        g = feat[feat["rdhs"] == district].sort_values("week_start_date").reset_index(drop=True)
        # index the val rows for this district by week_start_date
        vg = val[val["rdhs"] == district].set_index("week_start_date")
        # zone one-hots + population_density (raw) from feat
        row0 = g.iloc[0]
        zdry = float(row0["zone_dry_zone"])
        zint = float(row0["zone_intermediate_zone"])
        zwet = float(row0["zone_wet_zone"])
        pop_density = float(row0["population_density"])
        rdhs_id = int(row0["rdhs_id"])

        # a lookback window ends at position p (rows p-3..p). Need p-7..p (8 rows).
        for p in range(7, len(g)):
            lookback_rows = g.iloc[p - 3 : p + 1]
            # only compare windows whose 4 lookback weeks are all in val (2024)
            if not all(d in vg.index for d in lookback_rows["week_start_date"]):
                continue
            hist = g.iloc[p - 7 : p + 1][
                [
                    "week_start_date",
                    "week_end_date",
                    "temp_mean",
                    "temp_max",
                    "temp_min",
                    "rainfall_mm",
                    "humidity_pct",
                    "week_cases",
                    "week_no",
                ]
            ].copy()

            inputs = build_model_inputs(
                hist,
                rdhs_id=rdhs_id,
                district_name=district,
                zone_dry_zone=zdry,
                zone_intermediate_zone=zint,
                zone_wet_zone=zwet,
                population_density=pop_density,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                baselines=baselines,
            )
            seq = inputs.sequence_input[0]  # (4, 27)

            truth = vg.loc[lookback_rows["week_start_date"]][SEQ_COLS].to_numpy(dtype=np.float64)
            for j, c in enumerate(SEQ_COLS):
                diffs[c].append(np.abs(seq[:, j] - truth[:, j]))
            n_windows += 1

    print(f"Compared {n_windows} validation lookback windows across {len(districts)} districts\n")
    print(f"{'column':<26} {'max_abs_diff':>14} {'mean_abs_diff':>14}")
    print("-" * 56)
    worst = []
    for c in SEQ_COLS:
        if not diffs[c]:
            continue
        arr = np.concatenate(diffs[c])
        mx = float(arr.max())
        mn = float(arr.mean())
        worst.append((mx, c, mn))
        flag = "  <-- MISMATCH" if mx > 1e-6 else ""
        print(f"{c:<26} {mx:>14.6e} {mn:>14.6e}{flag}")

    print("\nWorst offenders:")
    for mx, c, mn in sorted(worst, reverse=True)[:8]:
        print(f"  {c:<26} max={mx:.4e} mean={mn:.4e}")


if __name__ == "__main__":
    main()
