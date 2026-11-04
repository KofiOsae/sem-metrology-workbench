from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image


def decode_image(data: bytes) -> np.ndarray:
    """Decode an SEM image as an 8-bit RGB array without changing the original."""
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_gray(rgb: np.ndarray) -> np.ndarray:
    # ITU-R BT.601; explicit conversion keeps the core independent of OpenCV.
    return np.clip(
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
        0,
        255,
    ).astype(np.uint8)
