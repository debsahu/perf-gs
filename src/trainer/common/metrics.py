# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Numpy-only image-quality metrics (PSNR / SSIM); the paper metric (3DGS / SAD-GS) is the
mean of per-image values over the held-out views."""

from __future__ import annotations

import math

import numpy as np


def psnr(rendered: np.ndarray, target: np.ndarray) -> float:
    """Per-image PSNR for data in [0, 1].

    Average the dB across frames; pooling MSE first gives a different, more pessimistic number.
    """
    mse = float(np.mean((rendered - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def psnr_masked(rendered: np.ndarray, target: np.ndarray, keep: np.ndarray | None) -> float | None:
    """PSNR over the pixels a loss mask keeps (``keep``: boolean HxW): the MSE is averaged over
    kept pixels only. None without a mask or with no kept pixel."""
    if keep is None or not keep.any():
        return None
    return psnr(rendered[keep], target[keep])


def lossmask_view(rendered: np.ndarray, target: np.ndarray,
                  keep: np.ndarray | None) -> tuple[float | None, float]:
    """(PSNR over the kept pixels, kept fraction) of one view under ``--masks``. A view with no mask file
    (``keep`` None) keeps every pixel: its whole-image PSNR and coverage 1.0. The PSNR is None when no pixel is kept."""
    if keep is None:
        return psnr(rendered, target), 1.0
    return psnr_masked(rendered, target, keep), float(keep.mean())


def ssim_gaussian_window(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Normalized 1D Gaussian window, matching the 3DGS / SAD-GS SSIM."""
    coords = np.arange(window_size) - window_size // 2
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float64)


# _WIN_1D is applied once per axis; _WIN_2D is the equivalent literal 11x11 window.
_WIN_1D = ssim_gaussian_window(11, 1.5)
_WIN_2D = np.outer(_WIN_1D, _WIN_1D)


def ssim(rendered: np.ndarray, target: np.ndarray) -> float:
    """Per-image SSIM for HxWxC data in [0, 1], faithful to the 3DGS / SAD-GS reference.

    Keep its 11x11 sigma=1.5 window and zero padding so results stay comparable; the
    separable application is the same operator, since zero padding factorizes per axis.
    """
    from scipy.ndimage import correlate1d  # scipy is always present

    r = rendered.astype(np.float64)
    t = target.astype(np.float64)
    if r.ndim == 2:
        r = r[..., None]
        t = t[..., None]

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    w1d = _WIN_1D

    def filt(img2d: np.ndarray) -> np.ndarray:
        # cval=0 'constant' mode == the reference's torch conv2d zero padding, once per axis.
        rows = correlate1d(img2d, w1d, axis=0, mode="constant", cval=0.0)
        return correlate1d(rows, w1d, axis=1, mode="constant", cval=0.0)

    ssim_channels = []
    for c in range(r.shape[-1]):
        x = r[..., c]
        y = t[..., c]
        mu_x = filt(x)
        mu_y = filt(y)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = filt(x * x) - mu_x2
        sigma_y2 = filt(y * y) - mu_y2
        sigma_xy = filt(x * y) - mu_xy
        ssim_map = (((2 * mu_xy + C1) * (2 * sigma_xy + C2)) /
                    ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)))
        ssim_channels.append(ssim_map)
    return float(np.mean(ssim_channels))
