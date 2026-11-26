# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training targets of a scene: decoded images, linear copies, ground-truth PNGs."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.common.color import srgb_to_linear
from trainer.data.colmap import ColmapDataset, load_colmap_image_native, to_float01
from trainer.data.parallel import default_workers_numbers, map_ordered


def _resident(image: np.ndarray) -> tuple[np.ndarray, int]:
    """The one host copy of a target, and the channel count the float view exposes.

    8-bit images are kept as packed uint8 RGBA (alpha 255 when the source has none):
    exactly the bytes ``TargetImages.pack`` makes from the float decode, since
    ``rint(to_float01(u8) * 255) == u8``. Other dtypes keep the float decode, as before."""
    channels = image.shape[2]
    if image.dtype != np.uint8:
        return to_float01(image), channels
    if channels == 3:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)      # alpha = 255
    return np.ascontiguousarray(image), channels


class _FloatTargets(Sequence):
    """``targets.srgb``: float32 sRGB [0, 1] made per access from the resident copy,
    bit-identical to what ``load_colmap_image`` returns. Nothing is cached."""

    def __init__(self, stored: list[np.ndarray], channels: list[int] | None) -> None:
        self._stored = stored
        self._channels = channels

    def __len__(self) -> int:
        return len(self._stored)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        image = self._stored[index]
        if image.dtype != np.uint8:
            return image
        channels = image.shape[2] if self._channels is None else self._channels[index]
        if channels != image.shape[2]:
            image = np.ascontiguousarray(image[..., :channels])
        return to_float01(image)


class TargetDataset:
    """Owns target loading, lazy linear conversion, and ground-truth images."""

    def __init__(self, scene: ColmapDataset, width: int, height: int, use_alpha: bool) -> None:
        self._scene = scene
        self._width = width
        self._height = height
        self._use_alpha = use_alpha
        # Decoding is startup's long pole and OpenCV releases the GIL, so decode on threads.
        started = time.time()
        workers = default_workers_numbers(len(scene.images))
        loaded = list(map_ordered(
            lambda entry: _resident(load_colmap_image_native(
                entry, scene.image_dir, width, height, load_alpha=use_alpha)),
            scene.images, workers=workers))
        self._srgb = [image for image, _ in loaded]
        self._channels = [channels for _, channels in loaded]
        print(f"[data] loaded {len(self._srgb)} targets at {width}x{height} in "
              f"{time.time() - started:.1f}s ({workers} decode threads, "
              f"{sum(image.nbytes for image in self._srgb) / 1024**3:.2f} GiB resident)", flush=True)

    @property
    def srgb(self) -> Sequence[np.ndarray]:
        return _FloatTargets(self._srgb, getattr(self, "_channels", None))

    @property
    def resident(self) -> list[np.ndarray]:
        """The resident copy of each target: packed uint8 RGBA for 8-bit sources, which
        ``TargetImages.pack`` passes through without copying."""
        return self._srgb

    def host_image_bytes(self) -> dict[str, int]:
        return {"decoded_images_bytes": sum(image.nbytes for image in self._srgb),
                "linear_cache_bytes": 0}

    def linear(self, camera_index: int) -> np.ndarray:
        # Not cached: at 48 Mpx one float linear target is 576 MB.
        return srgb_to_linear(self.srgb[camera_index][..., :3])

    def write_ground_truths(self, cameras_dir: Path) -> None:
        """One ``cam_XX/gt.png`` per camera, skipping those already written."""
        started = time.time()
        # Indices, not targets: map_ordered lists its items up front, and each float target is
        # built inside the worker that writes it (at 48 Mpx one float target is 576 MB).
        written = sum(map_ordered(
            lambda index: self._write_ground_truth(cameras_dir, index),
            range(len(self._srgb))))
        if written:
            print(f"[data] wrote {written} ground-truth PNGs in "
                  f"{time.time() - started:.1f}s", flush=True)

    def _write_ground_truth(self, cameras_dir: Path, camera_index: int) -> int:
        camera_dir = cameras_dir / f"cam_{camera_index:02d}"
        camera_dir.mkdir(parents=True, exist_ok=True)
        path = camera_dir / "gt.png"
        if path.exists():
            return 0
        target = self.srgb[camera_index]
        if self._use_alpha and target.shape[-1] == 4:
            pixels = (target * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGBA").save(path)
        else:
            pixels = (target[..., :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(path)
        return 1
