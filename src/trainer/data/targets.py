# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training targets of a scene: decoded images, linear copies, ground-truth PNGs."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.common.color import srgb_to_linear
from trainer.data.colmap import (ColmapDataset, decode_colmap_image, load_colmap_image_native,
                                 load_colmap_mask, read_colmap_image_bytes, to_float01)
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


# Views held decoded beyond the prefetch window, in compressed residency.
DEFAULT_CACHE_VIEWS = 2


class _DecodeCache:
    """Decoded views of a compressed dataset: a bounded LRU plus the pinned prefetch window.

    ``get`` returns a view, decoding it on the calling thread when it is neither cached nor in
    flight; concurrent readers of one view share a single decode. ``prefetch`` pins a window of
    views and decodes the missing ones on a thread pool, in the given order (OpenCV releases the
    GIL). Pinned views are never evicted; at most ``capacity`` unpinned views stay cached."""

    def __init__(self, decode: Callable[[int], tuple], capacity: int, workers: int) -> None:
        self._decode = decode
        self.capacity = max(0, int(capacity))
        self._workers = max(1, int(workers))
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._cached: OrderedDict[int, tuple] = OrderedDict()
        self._flight: dict[int, Future] = {}
        self._pinned: set[int] = set()
        self.hits = self.misses = self.decodes = self.peak_cached = self.waits = 0
        self.wait_s = self.decode_s = 0.0

    def get(self, index: int) -> tuple:
        with self._lock:
            entry = self._cached.get(index)
            if entry is not None:
                self._cached.move_to_end(index)
                self.hits += 1
                return entry
            future = self._flight.get(index)
            owner = future is None
            if owner:
                future = self._flight[index] = Future()
            self.misses += 1
        # A miss means the reader blocks until the view is decoded: that time is what prefetch hides.
        started = time.perf_counter()
        if owner:
            self._run(index, future)
        entry = future.result()
        with self._lock:
            self.waits += 1
            self.wait_s += time.perf_counter() - started
        return entry

    def prefetch(self, indices: Iterable[int]) -> None:
        window = list(dict.fromkeys(int(i) for i in indices))
        submit = []
        with self._lock:
            self._pinned = set(window)
            for index in window:
                if index in self._cached or index in self._flight:
                    continue
                self._flight[index] = future = Future()
                submit.append((index, future))
            self._evict()
        if submit and self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="target-decode")
        for index, future in submit:
            self._pool.submit(self._run, index, future)

    def _run(self, index: int, future: Future) -> None:
        started = time.perf_counter()
        try:
            entry = self._decode(index)
        except BaseException as error:          # surfaces on the reader
            with self._lock:
                self._flight.pop(index, None)
            future.set_exception(error)
            return
        with self._lock:
            self._flight.pop(index, None)
            self.decodes += 1
            self.decode_s += time.perf_counter() - started
            self._cached[index] = entry
            self._cached.move_to_end(index)
            self._evict()
            self.peak_cached = max(self.peak_cached, len(self._cached))
        future.set_result(entry)

    def _evict(self) -> None:
        unpinned = [i for i in self._cached if i not in self._pinned]
        for index in unpinned[:max(0, len(unpinned) - self.capacity)]:
            del self._cached[index]

    def cached_entries(self) -> list[tuple]:
        with self._lock:
            return list(self._cached.values())

    def stats(self) -> dict:
        with self._lock:
            return {"cached": len(self._cached), "peak_cached": self.peak_cached,
                    "capacity": self.capacity, "pinned": len(self._pinned),
                    "hits": self.hits, "misses": self.misses, "decodes": self.decodes,
                    "decode_s": self.decode_s, "waits": self.waits, "wait_s": self.wait_s}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


class _LazyColumn(Sequence):
    """One column of the decoded entry of each view (0 = resident array, 1 = exposed channels),
    decoded on access through the cache."""

    def __init__(self, targets: "TargetDataset", column: int) -> None:
        self._targets = targets
        self._column = column

    def __len__(self) -> int:
        return len(self._targets._encoded)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._targets._entry(index)[self._column]


class CompressedTargetViews(Mapping):
    """The packed uint8 RGBA target of each training view, made on demand from a compressed
    ``TargetDataset``: what the stream residency's dict of packed arrays holds, one view at a time.
    ``prefetch`` decodes the views the sampler will draw next."""

    def __init__(self, targets: "TargetDataset", pack: Callable[[np.ndarray], np.ndarray],
                 indices: Iterable[int] | None = None) -> None:
        if not targets.compressed:
            raise ValueError("CompressedTargetViews needs a compressed TargetDataset")
        self._targets = targets
        self._indices = list(range(len(targets.resident))) if indices is None else list(indices)
        self._keys = set(self._indices)
        targets.set_packer(pack)

    def __getitem__(self, index: int) -> np.ndarray:
        if index not in self._keys:
            raise KeyError(index)
        return self._targets._entry(index)[2]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def prefetch(self, indices: Iterable[int]) -> None:
        self._targets.prefetch(i for i in indices if i in self._keys)

    def resident_nbytes(self) -> int:
        """Bytes of the packed views decoded right now (never decodes)."""
        return self._targets.decoded_nbytes()


class TargetDataset:
    """Owns target loading, lazy linear conversion, and ground-truth images."""

    def __init__(self, scene: ColmapDataset, width: int, height: int, use_alpha: bool,
                 masks_dir: str | Path | None = None, invert_masks: bool = False,
                 residency: str = "decoded", cache_views: int = DEFAULT_CACHE_VIEWS,
                 decode_workers: int | None = None) -> None:
        """``residency``: ``decoded`` holds every target decoded (the stream and gpu image
        residencies); ``compressed`` holds each as its encoded file bytes and decodes on demand
        into the same bytes, with ``cache_views`` decoded views kept beyond the prefetch window."""
        self._scene = scene
        self._width = width
        self._height = height
        self._use_alpha = use_alpha
        if residency not in ("decoded", "compressed"):
            raise ValueError(f"unknown target residency {residency!r}")
        if masks_dir is not None and use_alpha:
            raise ValueError("loss masks and alpha supervision are exclusive: "
                             "the mask is stored in the target's alpha channel")
        masks = Path(masks_dir) if masks_dir is not None else None
        self._masked = masks is not None
        self._invert_masks = invert_masks
        self._compressed = residency == "compressed"
        # Decoding is startup's long pole and OpenCV releases the GIL, so decode on threads.
        started = time.time()
        workers = default_workers_numbers(len(scene.images))
        if self._compressed:
            loaded = self._load_compressed(masks, workers)
            self._cache = _DecodeCache(self._decode, cache_views,
                                       decode_workers or min(4, workers))
            self._srgb = _LazyColumn(self, 0)
            self._channels = _LazyColumn(self, 1)
            print(f"[data] read {len(self._encoded)} compressed targets for {width}x{height} in "
                  f"{time.time() - started:.1f}s ({self._encoded_nbytes() / 1024**3:.2f} GiB "
                  f"encoded resident; decoded on demand, {self._cache.capacity} cached beyond "
                  f"the prefetch window)", flush=True)
        else:
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

    # ------------------------------------------------------------ compressed residency

    def _load_compressed(self, masks: Path | None, workers: int) -> list:
        """Read each image's file bytes; masks are loaded (resized, thresholded) once and kept
        as a lossless PNG of that final form, so decoding one gives the same bytes back."""
        import cv2
        scene, width, height = self._scene, self._width, self._height
        self._packer: Callable[[np.ndarray], np.ndarray] | None = None

        def load(entry):
            data = read_colmap_image_bytes(entry, scene.image_dir)
            if masks is None:
                return (data, None), None
            mask = load_colmap_mask(entry, masks, width, height, self._invert_masks)
            if mask is None:
                return (data, None), None
            ok, png = cv2.imencode(".png", mask)
            if not ok:
                raise OSError(f"failed to encode the loss mask of {entry.image_name}")
            return (data, png), float(np.count_nonzero(mask)) / mask.size

        loaded = list(map_ordered(load, scene.images, workers=workers))
        self._encoded = [encoded for encoded, _ in loaded]
        return [(None, keep) for _, keep in loaded]

    def _decode(self, index: int) -> tuple:
        """(resident, exposed channels, packed) of one view, as the decoded residency holds it."""
        import cv2
        data, mask_png = self._encoded[index]
        entry = self._scene.images[index]
        image = decode_colmap_image(data, entry.image_name, self._width, self._height,
                                    load_alpha=self._use_alpha)
        if self._masked:
            mask = (np.full(image.shape[:2], 255, np.uint8) if mask_png is None
                    else cv2.imdecode(mask_png, cv2.IMREAD_GRAYSCALE))
            resident, channels = _resident(image, mask)
        else:
            resident, channels = _resident(image)
        packer = self._packer
        packed = None if packer is None else packer(resident)
        return resident, channels, packed

    def _entry(self, index: int) -> tuple:
        entry = self._cache.get(index)
        if entry[2] is None and self._packer is not None:
            # Decoded before a packer was set (e.g. by write_ground_truths): pack now.
            entry = (entry[0], entry[1], self._packer(entry[0]))
        return entry

    def _encoded_nbytes(self) -> int:
        return sum(data.nbytes + (0 if mask is None else mask.nbytes) for data, mask in self._encoded)

    @property
    def compressed(self) -> bool:
        """Whether targets are held as encoded bytes and decoded on demand."""
        return getattr(self, "_compressed", False)

    def set_packer(self, pack: Callable[[np.ndarray], np.ndarray]) -> None:
        """The resident -> packed conversion run with each decode (``TargetImages.pack``)."""
        self._packer = pack

    def prefetch(self, indices: Iterable[int]) -> None:
        """Decode these views ahead, in order, and keep them until the next call. No-op when
        every target is already decoded."""
        if self.compressed:
            self._cache.prefetch(indices)

    def cache_stats(self) -> dict:
        return self._cache.stats() if self.compressed else {}

    def decoded_nbytes(self) -> int:
        """Bytes of the views decoded right now (resident plus, when separate, packed)."""
        if not self.compressed:
            return sum(image.nbytes for image in self._srgb)
        total = 0
        for resident, _, packed in self._cache.cached_entries():
            total += resident.nbytes
            if packed is not None and packed is not resident:
                total += packed.nbytes
        return total

    def close(self) -> None:
        if self.compressed:
            self._cache.close()

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
        if self.compressed:
            import cv2
            mask_png = self._encoded[camera_index][1]
            if mask_png is None:
                return np.ones((self._height, self._width), bool)
            return cv2.imdecode(mask_png, cv2.IMREAD_GRAYSCALE) >= 128
        image = self._srgb[camera_index]
        return image[..., 3] >= (128 if image.dtype == np.uint8 else 0.5)

    @property
    def keep_masks(self) -> Sequence[np.ndarray | None] | None:
        """``keep_mask`` per camera, for the metric code; None without masks."""
        return _KeepMasks(self) if self.masked else None

    @property
    def resident(self) -> Sequence[np.ndarray]:
        """The resident copy of each target: packed uint8 RGBA for 8-bit sources, which
        ``TargetImages.pack`` passes through without copying. Compressed: decoded per access."""
        return self._srgb

    def host_image_bytes(self) -> dict[str, int]:
        if self.compressed:
            return {"decoded_images_bytes": 0, "encoded_images_bytes": self._encoded_nbytes(),
                    "decode_cache_bytes": self.decoded_nbytes(), "linear_cache_bytes": 0}
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
        # Compressed: no more decodes in flight than the cache holds.
        workers = max(1, self._cache.capacity) if self.compressed else None
        written = sum(map_ordered(
            lambda index: self._write_ground_truth(cameras_dir, index),
            range(len(self._srgb)), workers=workers, lookahead=1 if self.compressed else 2))
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
