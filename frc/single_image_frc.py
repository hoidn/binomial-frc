"""Split/cutoff helpers for single-image FRC workflows."""

from __future__ import annotations

import numpy as np


def center_crop_even_square(arr: np.ndarray) -> np.ndarray:
    """Center-crop to an even square canvas for split FRC."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    h, w = img.shape
    side = min(h, w)
    if side % 2 == 1:
        side -= 1
    if side < 2:
        raise ValueError(f"Image too small for single-image FRC: {img.shape}")
    h0 = (h - side) // 2
    w0 = (w - side) // 2
    return img[h0:h0 + side, w0:w0 + side]


def split_diagonal_interleaved(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one image into two diagonal interleaved half-images."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square image, got shape {img.shape}")
    if img.shape[0] % 2 != 0:
        raise ValueError(f"Expected even side length, got shape {img.shape}")
    half_a = 0.5 * (img[0::2, 0::2] + img[1::2, 1::2])
    half_b = 0.5 * (img[0::2, 1::2] + img[1::2, 0::2])
    return half_a, half_b


def split_binomial_thinned(
    arr: np.ndarray,
    *,
    rng_seed: int | None = None,
    count_scale: float = 4096.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an image into two half-images via deterministic binomial thinning."""
    img = np.asarray(arr)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"Expected square image, got shape {img.shape}")

    if count_scale <= 0:
        raise ValueError(f"count_scale must be > 0, got {count_scale}")

    rng = np.random.default_rng(0 if rng_seed is None else int(rng_seed))
    mag = np.abs(np.asarray(img, dtype=np.complex128))
    intensity = np.nan_to_num(mag * mag, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.rint(intensity * float(count_scale)).astype(np.int64)
    counts = np.clip(counts, 0, None)

    half_counts_a = rng.binomial(counts, 0.5)
    half_counts_b = counts - half_counts_a
    mag_a = np.sqrt(half_counts_a / float(count_scale))
    mag_b = np.sqrt(half_counts_b / float(count_scale))

    if np.iscomplexobj(img):
        phase = np.angle(img)
        half_a = (mag_a * np.exp(1j * phase)).astype(np.complex64)
        half_b = (mag_b * np.exp(1j * phase)).astype(np.complex64)
    else:
        sign = np.sign(np.asarray(img, dtype=np.float64))
        sign[sign == 0] = 1.0
        half_a = (mag_a * sign).astype(np.float32)
        half_b = (mag_b * sign).astype(np.float32)
    return half_a, half_b


def first_below_threshold(curve: np.ndarray, threshold: float) -> float:
    """Return first index where curve falls below threshold."""
    vals = np.asarray(curve, dtype=np.float64)
    if vals.size == 0:
        return np.nan
    finite = np.isfinite(vals)
    if not finite.any():
        return np.nan
    finite_vals = vals[finite]
    idx = np.where(finite_vals < threshold)[0]
    if len(idx) == 0:
        return float(len(finite_vals))
    return float(idx[0])
