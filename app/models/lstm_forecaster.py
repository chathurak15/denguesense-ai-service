"""
LSTM multi-input dengue forecaster — model loading and inference helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf  # noqa: F401

from app.features.feature_engineering import (
    FORECAST_HORIZON,
    LOOKBACK,
    SEQ_COLS,
    STATIC_COLS,
)

logger = logging.getLogger(__name__)


def load_lstm_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(
            f"LSTM model not found at {path}. "
            "Place _28_loss_0.0039_weights.keras in app/weights/."
        )
    logger.info("Loading LSTM model from %s …", path)

    import keras

    # Patch GlorotUniform for Keras 3 version mismatches
    _orig_gu_init = getattr(keras.initializers.GlorotUniform, "__init__")

    def _patched_gu_init(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        _orig_gu_init(self, seed=seed)

    keras.initializers.GlorotUniform.__init__ = _patched_gu_init

    def _strip_quantization(orig_init):
        def _patched(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            orig_init(self, *args, **kwargs)

        return _patched

    keras.layers.Embedding.__init__ = _strip_quantization(keras.layers.Embedding.__init__)
    keras.layers.Dense.__init__ = _strip_quantization(keras.layers.Dense.__init__)

    try:
        model = tf.keras.models.load_model(str(path))
    except (ValueError, TypeError):
        logger.warning(
            "load_model() failed to deserialize; retrying with compile=False …"
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


def run_forecast(
    model: tf.keras.Model,
    target_scaler,
    sequence_input: np.ndarray,
    rdhs_id: np.ndarray,
    static_features: np.ndarray,
) -> np.ndarray:
    """Run named-input predict and inverse-transform to raw case counts.

    Returns
    -------
    np.ndarray
        Shape (4,) unscaled predicted weekly cases.
    """
    inputs = {
        "sequence_input": sequence_input,
        "rdhs_id": rdhs_id,
        "static_features": static_features,
    }
    scaled_pred = model.predict(inputs, verbose=0)

    real_cases = np.empty(FORECAST_HORIZON, dtype=np.float64)
    for h in range(FORECAST_HORIZON):
        val_scaled = scaled_pred[0, h].reshape(-1, 1)
        val_real = target_scaler.inverse_transform(val_scaled)
        real_cases[h] = float(val_real[0, 0])
    return real_cases
