from pathlib import Path
import os

# Directory roots
APP_DIR: Path = Path(__file__).parent
WEIGHTS_DIR: Path = APP_DIR / "weights"

#Model weight files
CNN_MODEL_PATH: Path = WEIGHTS_DIR / "mobilenetv3_breeding_site_classifier_v1.keras"
LSTM_MODEL_PATH: Path = WEIGHTS_DIR / "_28_loss_0.0039_weights.keras"

# Model Version
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "mobilenetv3-breeding-site-v1.0.0")

#Scaler / precomputed window files
TARGET_SCALER_PATH: Path = WEIGHTS_DIR / "target_scaler.pkl"
FEATURE_SCALER_PATH: Path = WEIGHTS_DIR / "feature_scaler.pkl"
LATEST_WINDOWS_PATH: Path = WEIGHTS_DIR / "latest_windows.pkl"
