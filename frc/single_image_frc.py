"""Single-image FRC utilities and metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter as gf


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
    """Split an image into two statistically independent Poisson half-images."""
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
    lam = np.clip(intensity * float(count_scale), 0.0, None)

    half_counts_a = rng.poisson(0.5 * lam)
    half_counts_b = rng.poisson(0.5 * lam)
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


def fit_and_remove_plane(phase_img: np.ndarray, reference_phase: np.ndarray | None = None) -> np.ndarray:
    """Fit and remove a plane from a 2D phase image."""
    h, w = phase_img.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    phase_flat = phase_img.flatten()
    a = np.column_stack([x_flat, y_flat, np.ones(len(x_flat))])
    coeffs, _, _, _ = np.linalg.lstsq(a, phase_flat, rcond=None)
    fitted_plane = coeffs[0] * x_coords + coeffs[1] * y_coords + coeffs[2]
    phase_aligned = phase_img - fitted_plane
    if reference_phase is not None:
        ref_coeffs, _, _, _ = np.linalg.lstsq(a, reference_phase.flatten(), rcond=None)
        ref_plane = ref_coeffs[0] * x_coords + ref_coeffs[1] * y_coords + ref_coeffs[2]
        phase_aligned = phase_img - fitted_plane + ref_plane
    return phase_aligned


def trim_image(arr2d: np.ndarray, offset: int) -> np.ndarray:
    """Trim an image by offset/2 on each border."""
    arr = np.asarray(arr2d)
    off = int(offset)
    if off < 0 or (off % 2):
        raise ValueError(f"offset must be a non-negative even integer, got {offset}")
    if off == 0:
        return arr
    half = off // 2
    if arr.shape[0] <= off or arr.shape[1] <= off:
        raise ValueError(f"offset={off} too large for shape {arr.shape}")
    return arr[half:-half, half:-half]


def _phase_align(phase: np.ndarray, phase_align_method: str) -> np.ndarray:
    phase_unwrapped = np.unwrap(np.unwrap(np.asarray(phase, dtype=np.float32), axis=0), axis=1)
    if phase_align_method == "plane":
        return fit_and_remove_plane(phase_unwrapped)
    if phase_align_method == "mean":
        return phase_unwrapped - np.mean(phase_unwrapped)
    raise ValueError(f"Unknown phase_align_method: {phase_align_method}. Use 'plane' or 'mean'.")


def _support_weighted_phase_phasor(
    phase_aligned: np.ndarray,
    amp: np.ndarray,
    support_amp_floor_ratio: float,
) -> np.ndarray:
    amp_np = np.asarray(amp, dtype=np.float32)
    phase_np = np.asarray(phase_aligned, dtype=np.float32)
    if amp_np.shape != phase_np.shape:
        raise ValueError(f"Amplitude/phase shape mismatch: {amp_np.shape} vs {phase_np.shape}")
    max_amp = float(np.max(amp_np)) if amp_np.size else 0.0
    if max_amp <= 0:
        return np.zeros_like(amp_np, dtype=np.complex64)
    floor = max_amp * float(support_amp_floor_ratio)
    support = amp_np >= floor
    phasor = np.zeros_like(amp_np, dtype=np.complex64)
    if np.any(support):
        phasor[support] = np.exp(1j * phase_np[support]).astype(np.complex64)
    return phasor


def _spin_average_2d(x: np.ndarray) -> np.ndarray:
    """Radial ring average mirroring legacy spin_average behavior."""
    nr, nc = np.shape(x)
    nrdc = np.floor(nr / 2) + 1
    ncdc = np.floor(nc / 2) + 1
    r = np.arange(nr) - nrdc + 1
    c = np.arange(nc) - ncdc + 1
    rr, cc = np.meshgrid(r, c)
    index = np.round(np.sqrt(rr**2 + cc**2)) + 1
    maxindex = int(np.max(index))
    output = np.zeros(maxindex, dtype=np.complex128)
    for i in range(maxindex):
        idx = np.where(index == (i + 1))
        if len(idx[0]) == 0:
            output[i] = 0.0
        else:
            output[i] = np.sum(x[idx]) / len(idx[0])
    return output


def single_image_frc_curve(
    image_2d: np.ndarray,
    *,
    frc_sigma: float = 0.0,
    split_mode: str = "spatial",
    rng_seed: int | None = None,
) -> np.ndarray:
    """Compute single-image FRC curve for one real/complex 2D image."""
    canvas = center_crop_even_square(np.asarray(image_2d))
    if split_mode == "spatial":
        half_a, half_b = split_diagonal_interleaved(canvas)
    elif split_mode == "binomial":
        half_a, half_b = split_binomial_thinned(canvas, rng_seed=rng_seed)
    else:
        raise ValueError(f"Unknown split_mode={split_mode!r}; expected 'spatial' or 'binomial'.")

    i1 = np.fft.fftshift(np.fft.fft2(half_a))
    i2 = np.fft.fftshift(np.fft.fft2(half_b))
    c = _spin_average_2d(np.multiply(i1, np.conj(i2)))
    c1 = _spin_average_2d(np.multiply(i1, np.conj(i1)))
    c2 = _spin_average_2d(np.multiply(i2, np.conj(i2)))
    denom = np.sqrt(np.abs(np.multiply(c1, c2)))
    eps = np.finfo(np.float64).eps
    curve = np.asarray(np.real(c) / np.maximum(denom, eps), dtype=np.float64)
    curve = np.clip(curve, -1.0, 1.0)
    if frc_sigma > 0:
        curve = np.asarray(gf(curve, frc_sigma), dtype=np.float64)
    return curve


def _extract_prediction_hw(stitched_obj: np.ndarray) -> np.ndarray:
    pred = np.asarray(stitched_obj)
    if pred.ndim == 4:
        pred = pred[0]
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"Expected a 2D prediction after squeeze, got {pred.shape}")
    return pred


def single_image_frc_metrics(
    stitched_obj: np.ndarray,
    *,
    offset: int = 0,
    split_mode: str = "spatial",
    rng_seed: int | None = None,
    phase_align_method: str = "plane",
    support_amp_floor_ratio: float = 0.05,
    frc_sigma: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Compute no-GT single-image FRC cutoff metrics for amplitude and phase."""
    pred_hw = _extract_prediction_hw(stitched_obj)
    amp = trim_image(np.abs(pred_hw), offset)
    phi = trim_image(np.angle(pred_hw), offset)
    phi_aligned = _phase_align(phi, phase_align_method)

    amp_seed = 0 if rng_seed is None else int(rng_seed)
    phase_seed = amp_seed + 1

    amp_curve = single_image_frc_curve(
        amp,
        frc_sigma=frc_sigma,
        split_mode=split_mode,
        rng_seed=amp_seed,
    )
    amp_cut_50 = first_below_threshold(amp_curve, 0.5)
    amp_cut_1o7 = first_below_threshold(amp_curve, 1.0 / 7.0)

    phase_phasor = _support_weighted_phase_phasor(
        phi_aligned,
        amp,
        support_amp_floor_ratio=support_amp_floor_ratio,
    )
    if np.count_nonzero(np.abs(phase_phasor) > 0) < 4:
        phi_cut_50 = np.nan
        phi_cut_1o7 = np.nan
    else:
        phi_curve = single_image_frc_curve(
            phase_phasor,
            frc_sigma=frc_sigma,
            split_mode=split_mode,
            rng_seed=phase_seed,
        )
        phi_cut_50 = first_below_threshold(phi_curve, 0.5)
        phi_cut_1o7 = first_below_threshold(phi_curve, 1.0 / 7.0)

    return {
        "single_frc50": (amp_cut_50, phi_cut_50),
        "single_frc1over7": (amp_cut_1o7, phi_cut_1o7),
    }
