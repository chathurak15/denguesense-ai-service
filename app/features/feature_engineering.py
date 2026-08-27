"""
Train-aligned feature engineering for the LSTM dengue forecaster.

This is the **only** place sequence/static features are constructed. The
endpoint must not duplicate any of this logic.

Discovered from dengue_sense_lk notebooks 04 + 05 + 06 (do not guess columns):

    seq_cols (N=27, column order of scaled/train.csv after exclude_cols):
        temp_mean, temp_max, temp_min, rainfall_mm, humidity_pct,
        temp_lag_1w, rain_lag_1w, humid_lag_1w,
        temp_lag_2w, rain_lag_2w, humid_lag_2w,
        temp_lag_3w, rain_lag_3w, humid_lag_3w,
        temp_lag_4w, rain_lag_4w, humid_lag_4w,
        rain_roll4w_sum,
        cases_lag_1w, cases_lag_2w, cases_lag_3w, cases_lag_4w,
        extreme_weather_severity, week_sin, week_cos, temp_zscore,
        week_cases_scaled

    Lags (per district, chronological):
        temp_lag_{k}w  = temp_mean.shift(k)       k = 1..4
        rain_lag_{k}w  = rainfall_mm.shift(k)
        humid_lag_{k}w = humidity_pct.shift(k)
        cases_lag_{k}w = week_cases.shift(k)

    Rolling:
        rain_roll4w_sum = rainfall_mm.rolling(window=4, min_periods=1).sum()
        i.e. current week + previous 3 weeks (NOT shift(1).rolling(4);
        that alternate formula in scripts/fetch_weather.py does not match
        the training CSV).

    week_cases_scaled is produced by target_scaler (same scaler as the
    model output). Lagged case columns are scaled by feature_scaler.

    cumulative_cases was dropped in notebook 02 and is not a model feature.
    It is accepted on the request for the Spring Boot contract only.

Minimum history
---------------
    lookback L = 4
    max lag     = 4  (binding constraint)
    rain_roll4w includes the current week, so it only needs 3 prior weeks
        for a full window.

    Earliest lookback row = history[4] (0-based) given 8 raw weeks [0..7].
        history[4].lag_4w          = history[0]          present
        history[4].rain_roll4w_sum = history[1]+[2]+[3]+[4]  present

    Total raw weeks required = L + 4 = 8.
    The first 4 of the 8 rows have NaN lags; they are discarded. The 4
    lookback rows that enter the model have zero NaNs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# ── Discovered training constants ────────────────────────────────────────────

LOOKBACK: int = 4
FORECAST_HORIZON: int = 4
MIN_HISTORY_WEEKS: int = 8  # LOOKBACK + max_lag (see module docstring)

# Exact seq_cols order printed by notebooks/06_model_training_lstm.ipynb
# (`seq_cols = [c for c in train.columns if c not in exclude_cols]`).
SEQ_COLS: list[str] = [
    "temp_mean",
    "temp_max",
    "temp_min",
    "rainfall_mm",
    "humidity_pct",
    "temp_lag_1w",
    "rain_lag_1w",
    "humid_lag_1w",
    "temp_lag_2w",
    "rain_lag_2w",
    "humid_lag_2w",
    "temp_lag_3w",
    "rain_lag_3w",
    "humid_lag_3w",
    "temp_lag_4w",
    "rain_lag_4w",
    "humid_lag_4w",
    "rain_roll4w_sum",
    "cases_lag_1w",
    "cases_lag_2w",
    "cases_lag_3w",
    "cases_lag_4w",
    "extreme_weather_severity",
    "week_sin",
    "week_cos",
    "temp_zscore",
    "week_cases_scaled",
]

STATIC_COLS: list[str] = [
    "zone_dry_zone",
    "zone_intermediate_zone",
    "zone_wet_zone",
    "population_density",
]

# Columns feature_scaler was .fit() on (notebook 05 numeric_cols, in order).
FEATURE_SCALER_COLS: list[str] = [
    "temp_mean",
    "temp_max",
    "temp_min",
    "rainfall_mm",
    "humidity_pct",
    "temp_lag_1w",
    "temp_lag_2w",
    "temp_lag_3w",
    "temp_lag_4w",
    "rain_lag_1w",
    "rain_lag_2w",
    "rain_lag_3w",
    "rain_lag_4w",
    "humid_lag_1w",
    "humid_lag_2w",
    "humid_lag_3w",
    "humid_lag_4w",
    "rain_roll4w_sum",
    "area_km2",
    "population_total",
    "population_density",
    "cases_lag_1w",
    "cases_lag_2w",
    "cases_lag_3w",
    "cases_lag_4w",
    "week_sin",
    "week_cos",
    "extreme_weather_severity",
    "temp_zscore",
]

# sklearn LabelEncoder on stripped rdhs names (alphabetical) — 26 RDHS units.
RDHS_NAMES: tuple[str, ...] = (
    "Ampara",
    "Anuradhapura",
    "Badulla",
    "Batticaloa",
    "Colombo",
    "Galle",
    "Gampaha",
    "Hambantota",
    "Jaffna",
    "Kalmunai",
    "Kalutara",
    "Kandy",
    "Kegalle",
    "Kilinochchi",
    "Kurunegala",
    "Mannar",
    "Matale",
    "Matara",
    "Monaragala",
    "Mullaitivu",
    "Nuwara Eliya",
    "Polonnaruwa",
    "Puttalam",
    "Ratnapura",
    "Trincomalee",
    "Vavuniya",
)

# Copied from notebooks/04_feature_engineering_and_encoding.ipynb
CLIMATE_ZONE_MAP: dict[str, str] = {
    "Colombo": "wet_zone",
    "Gampaha": "wet_zone",
    "Kalutara": "wet_zone",
    "Kandy": "wet_zone",
    "Galle": "wet_zone",
    "Matara": "wet_zone",
    "Ratnapura": "wet_zone",
    "Kegalle": "wet_zone",
    "Nuwara Eliya": "wet_zone",
    "Matale": "intermediate_zone",
    "Kurunegala": "intermediate_zone",
    "Badulla": "intermediate_zone",
    "Monaragala": "intermediate_zone",
    "Hambantota": "dry_zone",
    "Jaffna": "dry_zone",
    "Kilinochchi": "dry_zone",
    "Mannar": "dry_zone",
    "Vavuniya": "dry_zone",
    "Mullaitivu": "dry_zone",
    "Batticaloa": "dry_zone",
    "Ampara": "dry_zone",
    "Trincomalee": "dry_zone",
    "Puttalam": "dry_zone",
    "Anuradhapura": "dry_zone",
    "Polonnaruwa": "dry_zone",
    "Kalmunai": "dry_zone",
}

# (name, start, end, severity) — inclusive on week_start_date, matching
# `df['week_start_date'].between(start, end)` in notebook 04.
EXTREME_WEATHER_EVENTS: tuple[tuple[str, date, date, int], ...] = (
    ("roanu_2016", date(2016, 5, 14), date(2016, 5, 25), 2),
    ("floods_2017", date(2017, 5, 25), date(2017, 6, 5), 3),
    ("burevi_2020", date(2020, 11, 30), date(2020, 12, 5), 1),
    ("floods_2021", date(2021, 5, 10), date(2021, 6, 10), 1),
    ("floods_2024", date(2024, 5, 10), date(2024, 6, 15), 2),
    ("ditwah_2025", date(2025, 11, 28), date(2025, 12, 31), 3),
)

RAW_HISTORY_COLS: tuple[str, ...] = (
    "week_start_date",
    "week_end_date",
    "temp_mean",
    "temp_max",
    "temp_min",
    "rainfall_mm",
    "humidity_pct",
    "week_cases",
    # Epidemiological week number from the SAME source used in training.
    # week_sin/week_cos and the season used for temp_zscore all depend on this.
    # It is NOT reliably derivable from week_start_date: the training data uses
    # a custom epi numbering that diverges from ISO in 53-week years (e.g. 2016,
    # 2021), so it must be supplied by the caller. See _iso_thursday_week for the
    # best-effort fallback used only when week_no is absent.
    "week_no",
)


@dataclass(frozen=True)
class ModelInputs:
    sequence_input: np.ndarray  # (1, 4, N)
    rdhs_id: np.ndarray  # (1, 1)
    static_features: np.ndarray  # (1, 4)


def get_season(week_no: int, climate_zone: str) -> str:
    """Zone-aware season — identical to notebook 04 `get_season`."""
    if climate_zone == "dry_zone":
        if 40 <= week_no <= 53 or week_no <= 4:
            return "ne_monsoon_peak"
        return "dry_period"
    # wet_zone and intermediate_zone share the same calendar
    if week_no <= 8 or week_no >= 49:
        return "ne_monsoon"
    if 9 <= week_no <= 19:
        return "inter_monsoon_1"
    if 20 <= week_no <= 39:
        return "sw_monsoon"
    return "inter_monsoon_2"


def climate_zone_from_onehots(
    zone_dry_zone: float,
    zone_intermediate_zone: float,
    zone_wet_zone: float,
) -> str:
    if zone_dry_zone == 1.0:
        return "dry_zone"
    if zone_intermediate_zone == 1.0:
        return "intermediate_zone"
    if zone_wet_zone == 1.0:
        return "wet_zone"
    raise ValueError("static_features must be a climate-zone one-hot")


def extreme_weather_severity(week_start: date) -> int:
    severity = 0
    for _name, start, end, sev in EXTREME_WEATHER_EVENTS:
        if start <= week_start <= end:
            severity = max(severity, sev)
    return severity


def load_temp_zscore_baselines(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        (key.split("|", 1)[0], key.split("|", 1)[1]): {
            "mean": float(val["mean"]),
            "std": float(val["std"]),
        }
        for key, val in raw.items()
    }


def _as_date(value: Any) -> date:
    if type(value) is date:
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()


def _iso_thursday_week(week_start: date) -> int:
    """Best-effort epi week number for a 7-day span, keyed on its Thursday.

    ISO weeks are defined by the Thursday of the week, so this reproduces the
    training ``week_no`` for ordinary years regardless of whether the span
    starts on a Monday (serving contract) or Saturday (raw training CSV).

    It CANNOT reproduce the custom epi numbering used in ISO 53-week years
    (2016, 2021 in this dataset), so it is only a fallback: callers should send
    the real ``week_no`` on every history row.
    """
    thursday = week_start + timedelta(days=(3 - week_start.weekday()) % 7)
    return int(thursday.isocalendar()[1])


def _scale_columns(
    scaler: Any,
    columns: Sequence[str],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a fitted MinMaxScaler to a subset of columns (independent per col).

    Uses scaler.min_ / scaler.scale_ so we never have to fabricate unused
    columns such as area_km2 / population_total.
    """
    names = list(getattr(scaler, "feature_names_in_", FEATURE_SCALER_COLS))
    out = frame.copy()
    for col in columns:
        idx = names.index(col)
        out[col] = frame[col].to_numpy(dtype=np.float64) * scaler.scale_[idx] + scaler.min_[idx]
    return out


def engineer_unscaled_features(
    history: pd.DataFrame,
    *,
    district_name: str,
    climate_zone: str,
    baselines: Mapping[tuple[str, str], Mapping[str, float]],
) -> pd.DataFrame:
    """Compute every engineered column on the 8-row raw window (unscaled)."""
    if len(history) != MIN_HISTORY_WEEKS:
        raise ValueError(
            f"history must contain exactly {MIN_HISTORY_WEEKS} rows, got {len(history)}"
        )

    df = history.copy()
    df["week_start_date"] = df["week_start_date"].map(_as_date)
    df = df.sort_values("week_start_date").reset_index(drop=True)

    for lag in range(1, 5):
        df[f"temp_lag_{lag}w"] = df["temp_mean"].shift(lag)
        df[f"rain_lag_{lag}w"] = df["rainfall_mm"].shift(lag)
        df[f"humid_lag_{lag}w"] = df["humidity_pct"].shift(lag)
        df[f"cases_lag_{lag}w"] = df["week_cases"].shift(lag)

    # Current week + previous 3 — matches merge_population_carryover.py / training CSV.
    df["rain_roll4w_sum"] = df["rainfall_mm"].rolling(window=4, min_periods=1).sum()

    # Prefer the epidemiological week_no supplied by the caller (the only value
    # guaranteed to match training in ISO 53-week years). Fall back to the
    # Thursday-of-week ISO number only when it is absent.
    if "week_no" in history.columns and history["week_no"].notna().all():
        df["week_no"] = df["week_no"].astype(int)
    else:
        df["week_no"] = df["week_start_date"].map(_iso_thursday_week)
    df["week_sin"] = np.sin(2.0 * math.pi * df["week_no"] / 52.0)
    df["week_cos"] = np.cos(2.0 * math.pi * df["week_no"] / 52.0)
    df["extreme_weather_severity"] = df["week_start_date"].map(extreme_weather_severity)

    seasons = [get_season(int(w), climate_zone) for w in df["week_no"]]
    zscores: list[float] = []
    for temp, season in zip(df["temp_mean"], seasons):
        stats = baselines.get((district_name, season))
        if stats is None:
            raise ValueError(
                f"No temp_zscore baseline for district={district_name!r} season={season!r}"
            )
        std = stats["std"]
        if std == 0.0 or not math.isfinite(std):
            raise ValueError(
                f"Invalid temp_zscore std for district={district_name!r} season={season!r}"
            )
        zscores.append((float(temp) - stats["mean"]) / std)
    df["temp_zscore"] = zscores
    df["season"] = seasons
    return df


def build_model_inputs(
    history: pd.DataFrame,
    *,
    rdhs_id: int,
    district_name: str,
    zone_dry_zone: float,
    zone_intermediate_zone: float,
    zone_wet_zone: float,
    population_density: float,
    feature_scaler: Any,
    target_scaler: Any,
    baselines: Mapping[tuple[str, str], Mapping[str, float]],
) -> ModelInputs:
    """Turn 8 raw weekly rows + static snapshot into named LSTM inputs.

    Parameters
    ----------
    history:
        DataFrame with RAW_HISTORY_COLS, exactly 8 consecutive weeks.
    """
    # week_no is optional at this layer: when absent we fall back to the
    # Thursday-of-week ISO number (see engineer_unscaled_features). The API
    # schema still requires it so real requests are always exact.
    required = [c for c in RAW_HISTORY_COLS if c != "week_no"]
    missing = [c for c in required if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing columns: {missing}")

    canonical = RDHS_NAMES[rdhs_id] if 0 <= rdhs_id < len(RDHS_NAMES) else district_name.strip()
    climate_zone = climate_zone_from_onehots(
        zone_dry_zone, zone_intermediate_zone, zone_wet_zone
    )

    engineered = engineer_unscaled_features(
        history,
        district_name=canonical,
        climate_zone=climate_zone,
        baselines=baselines,
    )

    engineered["week_cases_scaled"] = target_scaler.transform(
        engineered[["week_cases"]]
    ).ravel()

    feature_cols = [c for c in SEQ_COLS if c != "week_cases_scaled"] + ["population_density"]
    engineered["population_density"] = float(population_density)
    scaled = _scale_columns(feature_scaler, feature_cols, engineered)

    lookback = scaled.iloc[-LOOKBACK:].copy()
    seq = lookback[SEQ_COLS].to_numpy(dtype=np.float32)
    if seq.shape != (LOOKBACK, len(SEQ_COLS)):
        raise ValueError(f"sequence_input shape {seq.shape} != ({LOOKBACK}, {len(SEQ_COLS)})")
    if not np.isfinite(seq).all():
        nan_cols = lookback[SEQ_COLS].columns[lookback[SEQ_COLS].isna().any()].tolist()
        raise ValueError(
            f"NaN/inf in sequence_input columns {nan_cols}. "
            f"8 weeks of history should be sufficient (lookback 4 + lag 4)."
        )

    last = lookback.iloc[-1]
    static = np.array(
        [
            float(zone_dry_zone),
            float(zone_intermediate_zone),
            float(zone_wet_zone),
            float(last["population_density"]),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(static).all():
        raise ValueError("NaN/inf in static_features after scaling")

    return ModelInputs(
        sequence_input=seq[np.newaxis, ...],
        rdhs_id=np.array([[int(rdhs_id)]], dtype=np.int32),
        static_features=static[np.newaxis, ...],
    )
