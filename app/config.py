from pathlib import Path
import json
import os

# Directory roots
APP_DIR: Path = Path(__file__).parent
WEIGHTS_DIR: Path = APP_DIR / "weights"

# Model weight files
CNN_MODEL_PATH: Path = WEIGHTS_DIR / "mobilenetv3_breeding_site_classifier_v1.keras"
LSTM_MODEL_PATH: Path = WEIGHTS_DIR / "_28_loss_0.0039_weights.keras"

# CNN classifier version (classify endpoint)
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "mobilenetv3-breeding-site-v1.0.0")

# Scaler / interval / metadata files
TARGET_SCALER_PATH: Path = WEIGHTS_DIR / "target_scaler.pkl"
FEATURE_SCALER_PATH: Path = WEIGHTS_DIR / "feature_scaler.pkl"
TEMP_ZSCORE_BASELINES_PATH: Path = WEIGHTS_DIR / "temp_zscore_baselines.json"
RESIDUAL_INTERVALS_PATH: Path = WEIGHTS_DIR / "residual_intervals.json"
MODEL_METADATA_PATH: Path = WEIGHTS_DIR / "model_metadata.json"


def load_lstm_model_version() -> str:
    """Read LSTM version from env, then model_metadata.json — never from the route."""
    env_value = os.getenv("LSTM_MODEL_VERSION")
    if env_value:
        return env_value
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"model_metadata.json not found at {MODEL_METADATA_PATH}. "
            "Run  python scripts/export_forecast_artifacts.py --skip-residuals"
        )
    payload = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    version = payload.get("model_version")
    if not version:
        raise ValueError(f"{MODEL_METADATA_PATH} is missing 'model_version'")
    return str(version)
