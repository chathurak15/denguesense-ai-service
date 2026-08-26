"""
CNN breeding-site classifier — model loading and inference helpers.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf  # noqa: F401  — needed for keras registry

logger = logging.getLogger(__name__)

# Model constants
CNN_INPUT_SIZE = (224, 224)   # MobileNetV3Large expected spatial resolution


# Loading

def load_cnn_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(
            f"CNN model not found at {path}. "
            "Place mobilenetv3_breeding_site_classifier_v1.keras in app/weights/."
        )
    logger.info("Loading CNN model from %s …", path)
    try:
        import keras
        # Patch GlorotUniform for Keras 3 version mismatches
        _orig_gu_init = keras.initializers.GlorotUniform.__init__
        def _patched_gu_init(self, seed=None, input_axes=None, output_axes=None, **kwargs):
            _orig_gu_init(self, seed=seed)
        keras.initializers.GlorotUniform.__init__ = _patched_gu_init

        # Patch BatchNormalization for Keras 3 removed arguments
        _orig_bn_init = keras.layers.BatchNormalization.__init__
        def _patched_bn_init(self, *args, **kwargs):
            kwargs.pop('renorm', None)
            kwargs.pop('renorm_clipping', None)
            kwargs.pop('renorm_momentum', None)
            _orig_bn_init(self, *args, **kwargs)
        keras.layers.BatchNormalization.__init__ = _patched_bn_init

        # Patch Dense for quantization_config
        _orig_dense_init = keras.layers.Dense.__init__
        def _patched_dense_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            _orig_dense_init(self, *args, **kwargs)
        keras.layers.Dense.__init__ = _patched_dense_init

        model = tf.keras.models.load_model(str(path), compile=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load CNN model: {exc}") from exc
    logger.info("CNN model loaded ✓  (output shape: %s)", model.output_shape)
    return model


#Business rule
def band_confidence(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH_RISK"
    elif prob >= 0.3:
        return "LOW_RISK"
    else:
        return "INVALID"


#Inference
def classify_image(
    model: tf.keras.Model,
    image_bytes: bytes,
) -> tuple[str, float]:
    # Decode
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    # Resize
    img = img.resize(CNN_INPUT_SIZE, Image.LANCZOS)

    # Convert to float32 array — NO normalisation.
    # MobileNetV3's preprocessing is baked into the model.
    img_array = np.array(img, dtype=np.float32)          # (224, 224, 3), 0-255
    img_array = np.expand_dims(img_array, axis=0)        # (1, 224, 224, 3)

    # Predict
    raw_output = model.predict(img_array, verbose=0)     # shape: (1, 1) or (1,)
    prob = float(np.squeeze(raw_output))                 # scalar in [0, 1]

    risk_label = band_confidence(prob)
    return risk_label, prob
