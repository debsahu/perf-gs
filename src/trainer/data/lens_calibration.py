# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Fit cos^4 strength from training tracks. For learned per-camera vignetting, see
Deutsch et al., PPISP (2026), sec. 4.2, https://arxiv.org/abs/2601.18336.
"""
from itertools import combinations, product
from pathlib import Path

import cv2
import numpy as np

from trainer.data.colmap import _find_sparse_dir, _parse_cameras_bin, _parse_images_bin


def _sample_tracks(features, intrinsics, image, keep=None):
    xy, ids = features["xy"], features["id"]
    size = np.array([intrinsics["width"], intrinsics["height"]])
    center = [intrinsics["cx"], intrinsics["cy"]]
    focal = [intrinsics["fx"], intrinsics["fy"]]
    rays = (xy - center) / focal
    log_gain = (-2 / 2.2) * np.log1p(np.sum(rays * rays, axis=1))
    height, width = image.shape[:2]
    pixels = (xy * [width, height] / size - .5).astype(np.float32)
    radius = max(1., width / 720)
    inside = (np.minimum(pixels, [width - 1, height - 1] - pixels) > radius + 2).all(axis=1)
    active = np.flatnonzero(inside)
    if not len(active):
        return ids[:0], log_gain[:0], np.empty(0)
    offsets = np.array(list(product((-1, 0, 1), repeat=2)), np.float32) * radius
    kernel = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], np.float32)
    luma = np.array([.2126, .7152, .0722], np.float32)
    intensities = []
    unmasked = []
    keep_f = None if keep is None else keep.astype(np.float32)
    # OpenCV remap addresses fewer than 32767 output rows at a time.
    for indices in np.array_split(active, (len(active) + 15999) // 16000):
        uv = pixels[indices, None, :] + offsets
        rgb = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
        intensities.append(((rgb[..., :3] @ luma) @ kernel) / 16)
        if keep_f is not None:
            # Every tap of the sampled neighbourhood must be on kept pixels (loss mask).
            taps = cv2.remap(keep_f, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
            unmasked.append((taps > 0.999).all(axis=1))
    intensity = np.concatenate(intensities)
    good = (intensity > .06) & (intensity < .93)
    if keep_f is not None:
        good &= np.concatenate(unmasked)
    kept = active[good]
    order = np.argsort(ids[kept])
    return ids[kept][order], log_gain[kept][order], np.log(intensity[good])[order]


def _contrasts(first, second, rng):
    # Each pair of shared landmarks cancels both exposures and the landmarks' reflectances.
    _, i, j = np.intersect1d(first[0], second[0], return_indices=True)
    if len(i) < 12:
        return None
    radii = first[1][i] - second[1][j]
    values = first[2][i] - second[2][j]
    order = np.argsort(radii)
    half = len(order) // 2
    count = min(half, 32)
    low = rng.choice(order[:half], count, replace=False)
    high = rng.choice(order[-half:], count, replace=False)
    x = radii[high] - radii[low]
    usable = x > .035
    return x[usable], (values[high] - values[low])[usable]


def fit_lens_strength(scene_dir: Path, train_images: dict[str, int], targets,
                      keep_masks=None) -> tuple[float, int]:
    """Return the dimensionless strength and fit support, using native float sRGB targets.
    ``keep_masks[i]`` (boolean, or None) drops landmarks on pixels the loss mask drops."""
    sparse, fmt = _find_sparse_dir(scene_dir)
    if fmt != "bin" or not (sparse / "images.bin").is_file():
        return 0., 0
    names = sorted(train_images)
    samples = [names[i] for i in np.linspace(0, len(names) - 1, min(len(names), 28), dtype=int)]
    records = _parse_images_bin(sparse / "images.bin", set(samples))
    sensors = {r["camera_id"] for r in records if r["name"] in train_images}
    cameras = _parse_cameras_bin(sparse / "cameras.bin")
    if len(cameras) != 1 or sensors != cameras.keys():
        return 0., 0
    intrinsics = cameras[sensors.pop()]
    tracks = {r["name"]: r["tracks"] for r in records if "tracks" in r}
    views = [_sample_tracks(tracks[n], intrinsics, targets[train_images[n]],
                            None if keep_masks is None else keep_masks[train_images[n]])
             for n in samples if n in tracks]
    rng = np.random.default_rng(872)
    pairs = [pair for a, b in combinations(views, 2) if (pair := _contrasts(a, b, rng)) is not None]
    if not pairs:
        return 0., 0
    x = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    if len(x) < 60 or np.sum(x * x) < .1:
        return 0., len(x)
    strength = 0.
    for _ in range(8):
        weight = np.minimum(1., .05 / np.maximum(abs(y - x * strength), 1e-8))
        strength = np.clip((x @ (y * weight)) / (x @ (x * weight)), 0., 2.)
    return float(strength), len(x)
