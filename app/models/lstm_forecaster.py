"""
LSTM multi-input dengue forecaster — model loading and inference helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf  # noqa: F401

logger = logging.getLogger(__name__)

# Feature metadata
SEQ_COLS: list[str] = [
    "temp_mean", "temp_max", "temp_min", "rainfall_mm", "humidity_pct",
    "temp_lag_1w", "rain_lag_1w", "humid_lag_1w",
    "temp_lag_2w", "rain_lag_2w", "humid_lag_2w",
    "temp_lag_3w", "rain_lag_3w", "humid_lag_3w",
    "temp_lag_4w", "rain_lag_4w", "humid_lag_4w",
    "rain_roll4w_sum",
    "cases_lag_1w", "cases_lag_2w", "cases_lag_3w", "cases_lag_4w",
    "extreme_weather_severity", "week_sin", "week_cos", "temp_zscore",
    "week_cases_scaled",
]

#: Static features fed to the model (one snapshot per district, per window).
STATIC_COLS: list[str] = [
    "zone_dry_zone",
    "zone_intermediate_zone",
    "zone_wet_zone",
    "population_density",
]

LOOKBACK: int = 4          # weeks of history per window
FORECAST_HORIZON: int = 4  # weeks ahead the model predicts


#Loading
def load_lstm_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(
            f"LSTM model not found at {path}. "
            "Place _28_loss_0.0039_weights.keras in app/weights/."
        )
    logger.info("Loading LSTM model from %s …", path)
    
    import keras
    # Patch GlorotUniform for Keras 3 version mismatches
    _orig_gu_init = getattr(keras.initializers.GlorotUniform, '__init__')
    def _patched_gu_init(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        _orig_gu_init(self, seed=seed)
    keras.initializers.GlorotUniform.__init__ = _patched_gu_init
    
    # Patch Embedding for quantization_config
    _orig_emb_init = keras.layers.Embedding.__init__
    def _patched_emb_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        _orig_emb_init(self, *args, **kwargs)
    keras.layers.Embedding.__init__ = _patched_emb_init

    try:
        model = tf.keras.models.load_model(str(path))
    except ValueError:
        logger.warning(
            "load_model() raised ValueError; retrying with compile=False …"
        )
        try:
            model = tf.keras.models.load_model(str(path), compile=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LSTM model (compile=False fallback): {exc}"
            ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load LSTM model: {exc}") from exc

    logger.info("LSTM model loaded ✓  (output shape: %s)", model.output_shape)
    return model


#Inference

def run_forecast(
    model: tf.keras.Model,
    target_scaler,          # sklearn MinMaxScaler fitted on week_cases
    window: dict,
) -> list[float]:
    # Build batched named-input dict (batch size = 1)
    inputs = {
        "sequence_input":  window["sequence_input"][np.newaxis, ...],   # (1,4,27)
        "rdhs_id":         window["rdhs_id"].reshape(1, 1),             # (1,1)
        "static_features": window["static_features"][np.newaxis, ...],  # (1,4)
    }

    # Predict (scaled output, shape (1,4))
    scaled_pred = model.predict(inputs, verbose=0)   # np.ndarray (1, 4)

    # Inverse-transform each week horizon individually
    real_cases: list[float] = []
    for h in range(FORECAST_HORIZON):
        val_scaled = scaled_pred[0, h].reshape(-1, 1)          # (1,1)
        val_real = target_scaler.inverse_transform(val_scaled)  # (1,1)
        real_cases.append(float(val_real[0, 0]))

    return real_cases
