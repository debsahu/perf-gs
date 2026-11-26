# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training targets of a scene: decoded images, linear copies, ground-truth PNGs."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.common.color import srgb_to_linear
from trainer.data.colmap import (ColmapDataset, load_colmap_image_native, load_colmap_mask,
                                 to_float01)
from trainer.data.parallel import default_workers_numbers, map_ordered


def _resident(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """The one host copy of a target, and the channel count the float view exposes.

    8-bit images are kept as packed uint8 RGBA (alpha 255 when the source has none):
    exactly the bytes ``TargetImages.pack`` makes from the float decode, since
    ``rint(to_float01(u8) * 255) == u8``. Other dtypes keep the float decode, as before.

    A loss mask (uint8, 255 = keep) rides in the alpha channel, so it costs no extra memory;
    the float view still exposes RGB only. Alpha supervision is then off (they are exclusive)."""
    channels = image.shape[2]
    if mask is not None:
        if channels != 3:
            raise ValueError("a loss mask needs an RGB target: alpha supervision is exclusive with masks")
        if image.dtype != np.uint8:
            alpha = mask[..., None].astype(np.float32) * (1.0 / 255.0)
            return np.ascontiguousarray(np.concatenate([to_float01(image), alpha], axis=-1)), 3
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        image[..., 3] = mask
        return np.ascontiguousarray(image), 3
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
        channels = image.shape[2] if self._channels is None else self._channels[index]
        if image.dtype != np.uint8:
            # Only a masked float target stores more channels than it exposes.
            return image if channels == image.shape[2] else np.ascontiguousarray(image[..., :channels])
        if channels != image.shape[2]:
            image = np.ascontiguousarray(image[..., :channels])
        return to_float01(image)


class _KeepMasks(Sequence):
    """``targets.keep_masks``: the boolean keep mask of each target, made per access."""

    def __init__(self, targets: "TargetDataset") -> None:
        self._targets = targets

    def __len__(self) -> int:
        return len(self._targets.resident)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self._targets.keep_mask(index)


class TargetDataset:
    """Owns target loading, lazy linear conversion, and ground-truth images."""

    def __init__(self, scene: ColmapDataset, width: int, height: int, use_alpha: bool,
                 masks_dir: str | Path | None = None, invert_masks: bool = False) -> None:
        self._scene = scene
        self._width = width
        self._height = height
        self._use_alpha = use_alpha
        if masks_dir is not None and use_alpha:
            raise ValueError("loss masks and alpha supervision are exclusive: "
                             "the mask is stored in the target's alpha channel")
        masks = Path(masks_dir) if masks_dir is not None else None
        # Decoding is startup's long pole and OpenCV releases the GIL, so decode on threads.
        started = time.time()
        workers = default_workers_numbers(len(scene.images))

        def load(entry):
            image = load_colmap_image_native(entry, scene.image_dir, width, height, load_alpha=use_alpha)
            if masks is None:
                return _resident(image), None
            mask = load_colmap_mask(entry, masks, width, height, invert_masks)
            keep = None if mask is None else float(np.count_nonzero(mask)) / mask.size
            return _resident(image, mask if mask is not None else np.full(image.shape[:2], 255, np.uint8)), keep

        loaded = list(map_ordered(load, scene.images, workers=workers))
        self._srgb = [image for (image, _), _ in loaded]
        self._channels = [channels for (_, channels), _ in loaded]
        self._masked = masks is not None
        print(f"[data] loaded {len(self._srgb)} targets at {width}x{height} in "
              f"{time.time() - started:.1f}s ({workers} decode threads, "
              f"{sum(image.nbytes for image in self._srgb) / 1024**3:.2f} GiB resident)", flush=True)
        if masks is not None:
            kept = [keep for _, keep in loaded if keep is not None]
            self.mask_count = len(kept)
            if not kept:
                raise FileNotFoundError(
                    f"--masks {masks}: no mask matched any of the {len(loaded)} images "
                    "(looked for <image name>.png and <stem>.png)")
            print(f"[masks] {len(kept)} of {len(loaded)} images have a loss mask in {masks} "
                  f"({'inverted, 255 = drop' if invert_masks else '255 = keep'}); "
                  f"{len(loaded) - len(kept)} without one keep every pixel; mean kept fraction "
                  f"{np.mean(kept):.3f}", flush=True)

    @property
    def srgb(self) -> Sequence[np.ndarray]:
        return _FloatTargets(self._srgb, getattr(self, "_channels", None))

    @property
    def masked(self) -> bool:
        """Whether the targets carry loss masks (in their alpha channel)."""
        return getattr(self, "_masked", False)

    def keep_mask(self, camera_index: int) -> np.ndarray | None:
        """Boolean (H, W) keep mask of a target, or None without masks."""
        if not self.masked:
            return None
        image = self._srgb[camera_index]
        return image[..., 3] >= (128 if image.dtype == np.uint8 else 0.5)

    @property
    def keep_masks(self) -> Sequence[np.ndarray | None] | None:
        """``keep_mask`` per camera, for the metric code; None without masks."""
        return _KeepMasks(self) if self.masked else None

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
