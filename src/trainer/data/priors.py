# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Depth and normal priors: discovery, decoding, the common grid, the load-time sign check, and a
streamed host residency (encoded bytes in RAM, a few views decoded ahead).

On the GPU every prior has one form. Depth: uint16 millimetres, 0 = invalid, 65535 (saturated)
is skipped as invalid. Normal: uint8 codes ``c``, ``n = c / 255 * 2 - 1`` with code 128 decoding to
exactly 0, so ``(128, 128, 128)`` is the invalid sentinel; normals are camera frame (OpenCV: +x
right, +y down, +z forward) and face the camera.

Accepted files (format decided by the magic bytes, never the extension):
  depth   16-bit grayscale PNG (mm, 0 invalid), or float32 TIFF (metres, 0 invalid)
  normal  8-bit RGB PNG, or float32 3-channel TIFF (unit, (0,0,0) invalid)
Normal PNG encodings (``--normal-encoding``):
  slam     code 128 decodes to 0, invalid = (128, 128, 128). The quantized codec of the
           png-quantized prior format (see NOTICE.md).
  spirula  the same affine map, invalid = black (0, 0, 0), no code-128 rule.
Float TIFFs are quantized at load (depth to mm, half to even; normals to the slam codes): one GPU
format and one decode path, at most 0.5 mm and 2/255 per component of loss.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trainer.data.targets import _DecodeCache

PRIOR_SUFFIXES = (".png", ".tiff", ".tif")
NORMAL_ENCODINGS = ("slam", "spirula")
_PNG = b"\x89PNG\r\n\x1a\n"
_TIFF = (b"II*\x00", b"MM\x00*")
DEPTH_MM_MAX = 65535

# Load-time normal check (on up to CHECK_VIEWS evenly spaced views).
CHECK_VIEWS = 8
CHECK_MIN_FACING = 0.99
CHECK_MIN_VALID = 0.5          # below it: a warning, not an error (partially valid maps are legitimate)
CHECK_LENGTH_TOL = 0.02
# No decoded normal may be longer than this. A unit normal quantized to 8 bits stays within ~1.01; a Spirula map
# read as slam decodes each black (invalid) pixel to (-1, -1, -1), length 1.73, which a percentile would miss when
# only a few pixels are black.
CHECK_MAX_LENGTH = 1.1


class PriorError(ValueError):
    """A prior file or set that cannot be used as is."""


# ---------------------------------------------------------------------------- discovery

def find_prior_path(prior_dir: Path, image_name: str) -> Path | None:
    """``<dir>/<image name>.<ext>``, else ``<dir>/<stem>.<ext>``, sub-folders as in ``images/``.
    Two files for one image is an error."""
    name = Path(image_name)
    found: list[Path] = []
    for base in (name.name, name.stem):
        for ext in PRIOR_SUFFIXES:
            candidate = prior_dir / name.parent / f"{base}{ext}"
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
        if found:
            break
    if len(found) > 1:
        raise PriorError(f"more than one prior for {image_name}: {', '.join(str(p) for p in found)}")
    return found[0] if found else None


def match_priors(prior_dir: str | Path, image_names: Sequence[str], kind: str,
                 what: str = "images") -> list[Path | None]:
    """The prior file of each image (None when it has none). Zero matches is an error."""
    prior_dir = Path(prior_dir)
    if not prior_dir.is_dir():
        raise PriorError(f"--{kind}-dir {prior_dir} is not a directory")
    paths = [find_prior_path(prior_dir, n) for n in image_names]
    matched = sum(p is not None for p in paths)
    if matched == 0:
        raise PriorError(f"no {kind} prior in {prior_dir} matches any of the {len(image_names)} {what}")
    if matched < len(image_names):
        print(f"[priors] {kind}: {matched} of {len(image_names)} {what} have a prior; "
              f"the other {len(image_names) - matched} train without the {kind} term")
    else:
        print(f"[priors] {kind}: {matched} of {len(image_names)} {what} matched")
    return paths


def _png_shape(data: bytes) -> tuple[int, int] | None:
    """(height, width) from a PNG's IHDR, without decoding; None for anything else."""
    if data[:8] != _PNG or data[12:16] != b"IHDR" or len(data) < 24:
        return None
    return int.from_bytes(data[20:24], "big"), int.from_bytes(data[16:20], "big")


# ---------------------------------------------------------------------------- decoding

def _sniff(data: bytes) -> str:
    if data[:8] == _PNG:
        return "png"
    if data[:4] in _TIFF:
        return "tiff"
    raise PriorError("not a PNG or TIFF file")


def tiff_orientation(data: bytes) -> int:
    """TIFF tag 274 (Orientation) of the first image, 1 when absent. Classic and BigTIFF."""
    import struct
    order = "<" if data[:2] == b"II" else ">"
    try:
        magic = struct.unpack_from(order + "H", data, 2)[0]
        if magic == 42:
            offset = struct.unpack_from(order + "I", data, 4)[0]
            count = struct.unpack_from(order + "H", data, offset)[0]
            entries, size, value_at = offset + 2, 12, 8
        elif magic == 43:
            offset = struct.unpack_from(order + "Q", data, 8)[0]
            count = struct.unpack_from(order + "Q", data, offset)[0]
            entries, size, value_at = offset + 8, 20, 12
        else:
            return 1
        for k in range(count):
            at = entries + k * size
            tag, kind = struct.unpack_from(order + "HH", data, at)
            if tag == 274:
                return struct.unpack_from(order + ("H" if kind == 3 else "I"), data, at + value_at)[0]
    except struct.error as error:
        raise PriorError(f"malformed TIFF header: {error}") from error
    return 1


def _imdecode(data: bytes, what: str) -> np.ndarray:
    import cv2
    if data[:4] in _TIFF:
        # OpenCV applies the orientation tag even with IMREAD_IGNORE_ORIENTATION: it would turn or transpose the
        # pixels (a 180-degree turn keeps the shape, so the aspect check passes) and never the normal vectors.
        orientation = tiff_orientation(data)
        if orientation != 1:
            raise PriorError(f"{what}: TIFF orientation tag is {orientation}; write the prior without it "
                             "(or with orientation 1), in the image's own pixel layout")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PriorError(f"failed to decode {what}")
    return image


def quantize_depth_m(depth_m: np.ndarray) -> np.ndarray:
    """Metres (0 invalid) to uint16 mm, half to even, saturating at 65535 (then skipped)."""
    d = np.asarray(depth_m, dtype=np.float64)
    bad = ~np.isfinite(d) | (d <= 0.0)
    mm = np.rint(np.where(bad, 0.0, d) * 1000.0)
    return np.clip(mm, 0, DEPTH_MM_MAX).astype(np.uint16)


def encode_normal_codes(normal: np.ndarray) -> np.ndarray:
    """Unit normals ((0,0,0) invalid) to slam uint8 codes: rint((n + 1) / 2 * 255), half to even. A pixel with a
    non-finite component is invalid (the sentinel 128), as a non-finite depth is."""
    n = np.asarray(normal, dtype=np.float32)
    bad = ~np.isfinite(n).all(axis=-1)
    codes = np.rint((np.clip(np.where(bad[..., None], 0.0, n), -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype(np.uint8)
    codes[bad] = 128
    return codes


def decode_normal_codes(codes: np.ndarray) -> np.ndarray:
    """Slam uint8 codes to float normals: code 128 is exactly 0."""
    n = codes.astype(np.float32) * (2.0 / 255.0) - 1.0
    n[codes == 128] = 0.0
    return n


def decode_depth(data: bytes, what: str = "depth prior") -> np.ndarray:
    """Depth file bytes to uint16 mm (0 invalid)."""
    kind = _sniff(data)
    image = _imdecode(data, what)
    if image.ndim != 2:
        raise PriorError(f"{what}: depth must be single-channel, got shape {image.shape}")
    if kind == "png":
        if image.dtype != np.uint16:
            raise PriorError(f"{what}: depth PNG must be 16-bit millimetres, got {image.dtype}")
        return image
    if image.dtype != np.float32:
        raise PriorError(f"{what}: depth TIFF must be float32 metres, got {image.dtype}")
    return quantize_depth_m(image)


def decode_normal(data: bytes, encoding: str, what: str = "normal prior") -> np.ndarray:
    """Normal file bytes to slam uint8 codes (H, W, 3), invalid = (128, 128, 128)."""
    if encoding not in NORMAL_ENCODINGS:
        raise PriorError(f"unknown normal encoding {encoding!r}")
    kind = _sniff(data)
    image = _imdecode(data, what)
    if image.ndim != 3 or image.shape[2] != 3:
        raise PriorError(f"{what}: normals must be 3-channel, got shape {image.shape}")
    image = np.ascontiguousarray(image[..., ::-1])          # OpenCV decodes BGR
    if kind == "tiff":
        if image.dtype != np.float32:
            raise PriorError(f"{what}: normal TIFF must be float32, got {image.dtype}")
        return encode_normal_codes(image)
    if image.dtype != np.uint8:
        raise PriorError(f"{what}: normal PNG must be 8-bit, got {image.dtype}")
    if encoding == "spirula":
        image = image.copy()
        image[(image == 0).all(axis=-1)] = 128            # black = invalid
    return image


# ---------------------------------------------------------------------------- grid

def check_aspect(prior_hw: tuple[int, int], native_wh: tuple[int, int], what: str) -> None:
    """A prior keeps its own grid but must have its image's aspect: |pw*H - ph*W| <= max(W, H),
    with (W, H) the image's own size (views are stretched to the training size per axis, and so
    are their priors)."""
    ph, pw = prior_hw
    w, h = native_wh
    if abs(pw * h - ph * w) > max(w, h):
        raise PriorError(f"{what}: prior is {pw}x{ph} but the image is {w}x{h} (aspect differs)")


def common_grid(shapes: Iterable[tuple[int, int]], image_wh: tuple[int, int]) -> tuple[int, int]:
    """(height, width) of one kind's GPU texture: the training image's grid at the density of the
    densest prior (s = max sqrt(prior pixels / training pixels)), capped at the training size."""
    w, h = image_wh
    shapes = list(shapes)
    densest = max(shapes, key=lambda hw: hw[0] * hw[1])
    s = np.sqrt(densest[0] * densest[1] / (w * h))
    if s >= 1.0:
        return h, w
    grid = max(1, int(round(h * s))), max(1, int(round(w * s)))
    # The densest prior passed the aspect check against its image; keep its own grid when rounding lands a pixel
    # off it, so its file bytes pass through instead of being resized and re-encoded.
    if abs(grid[0] - densest[0]) <= 1 and abs(grid[1] - densest[1]) <= 1:
        return tuple(densest)
    return grid


def nearest_resize(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest resize, sampling at destination pixel centers (never interpolates the sentinels)."""
    h, w = hw
    if image.shape[:2] == (h, w):
        return image
    ys = ((2 * np.arange(h) + 1) * image.shape[0]) // (2 * h)
    xs = ((2 * np.arange(w) + 1) * image.shape[1]) // (2 * w)
    return np.ascontiguousarray(image[ys][:, xs])


def prior_texel(pixel: tuple[int, int], image_wh: tuple[int, int], prior_wh: tuple[int, int]) -> tuple[int, int]:
    """The prior texel the loss shader reads for a training pixel (u, v): nearest at the center.
    Mirrors priorTexel in shaders/losses/geom_loss.slang."""
    (u, v), (w, h), (pw, ph) = pixel, image_wh, prior_wh
    return ((2 * u + 1) * pw) // (2 * w), ((2 * v + 1) * ph) // (2 * h)


# ---------------------------------------------------------------------------- sign check

@dataclass(frozen=True)
class NormalCheck:
    facing: float        # fraction of valid pixels with dot(n, ray) < 0
    valid: float         # valid fraction of the map
    length_p01: float
    length_p99: float
    length_max: float = 0.0

    def passed(self) -> bool:
        """The hard conditions. A low valid fraction is reported separately (``mostly_invalid``)."""
        return (self.valid > 0.0 and self.facing >= CHECK_MIN_FACING
                and abs(self.length_p01 - 1.0) <= CHECK_LENGTH_TOL
                and abs(self.length_p99 - 1.0) <= CHECK_LENGTH_TOL
                and self.length_max <= CHECK_MAX_LENGTH)

    def mostly_invalid(self) -> bool:
        return self.valid < CHECK_MIN_VALID


def measure_normals(codes: np.ndarray, intrinsics: tuple[float, float, float, float],
                    image_wh: tuple[int, int]) -> NormalCheck:
    """Facing and length statistics of one map; rays from the camera scaled to the prior grid.
    The test is the per-pixel view ray, never the optical axis."""
    fx, fy, cx, cy = intrinsics
    ph, pw = codes.shape[:2]
    sx, sy = pw / image_wh[0], ph / image_wh[1]
    n = decode_normal_codes(codes[..., :3]).astype(np.float64)
    length = np.linalg.norm(n, axis=-1)
    valid = length > 0.5
    u, v = np.meshgrid(np.arange(pw) + 0.5, np.arange(ph) + 0.5)
    ray = np.stack([(u / sx - cx) / fx, (v / sy - cy) / fy, np.ones_like(u)], axis=-1)
    if not valid.any():
        return NormalCheck(0.0, 0.0, 0.0, 0.0, 0.0)
    facing = float(np.mean((n * ray).sum(-1)[valid] < 0.0))
    p01, p99 = np.percentile(length[valid], [1, 99])
    return NormalCheck(facing, float(valid.mean()), float(p01), float(p99), float(length[valid].max()))


def check_normal_priors(views: Sequence[tuple[str, np.ndarray, tuple[float, float, float, float]]],
                        image_wh: tuple[int, int]) -> list[NormalCheck]:
    """Hard error unless every checked map faces the camera and is unit length (no decoded normal longer than
    CHECK_MAX_LENGTH). A mostly invalid map is a warning. ``views`` = (name, slam codes, (fx, fy, cx, cy) at the
    image size)."""
    results = []
    for name, codes, intrinsics in views:
        c = measure_normals(codes, intrinsics, image_wh)
        results.append(c)
        print(f"[priors] normal check {name}: facing {c.facing:.6f}, valid {c.valid:.3f}, "
              f"|n| p1/p99/max {c.length_p01:.4f}/{c.length_p99:.4f}/{c.length_max:.4f}")
        if c.mostly_invalid():
            print(f"[priors] WARNING: normal prior {name} is only {c.valid:.1%} valid; its other pixels get no "
                  f"normal term. Expected for partial maps; otherwise check --normal-encoding.")
        if not c.passed():
            hint = ""
            if c.valid == 0.0:
                hint = " The map has no valid pixel. Check --normal-encoding (slam: 128 = invalid; spirula: black)."
            elif c.facing < 0.5:
                hint = " The map points AWAY from the camera; negate every normal (n -> -n) and write it again."
            elif (c.length_max > CHECK_MAX_LENGTH or abs(c.length_p01 - 1.0) > CHECK_LENGTH_TOL
                  or abs(c.length_p99 - 1.0) > CHECK_LENGTH_TOL):
                hint = (" Some decoded normals are not unit length. Check --normal-encoding: a spirula map read as "
                        "slam decodes black (invalid) pixels to (-1, -1, -1).")
            raise PriorError(
                f"normal prior {name} fails the load check (need facing >= {CHECK_MIN_FACING}, "
                f"| |n| - 1 | <= {CHECK_LENGTH_TOL} at p1 and p99, every |n| <= {CHECK_MAX_LENGTH}).{hint}")
    return results


def check_indices(count: int, views: int = CHECK_VIEWS) -> list[int]:
    if count <= views:
        return list(range(count))
    if views <= 1:
        return [0] if views == 1 else []
    return sorted({round(i * (count - 1) / (views - 1)) for i in range(views)})


# ---------------------------------------------------------------------------- dataset

def _encode_png(image: np.ndarray) -> bytes:
    import cv2
    if image.ndim == 3:
        image = image[..., ::-1]
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(image))
    if not ok:
        raise PriorError("PNG encode failed")
    return buf.tobytes()


class PriorDataset:
    """The depth and normal priors of the training views, held as PNG bytes on the common grid.

    ``entry(i)`` returns (uint16 mm (H, W) or None, uint8 RGBA slam codes (H, W, 4) or None),
    decoding through a bounded cache; ``prefetch`` decodes the sampler's next draws on a pool."""

    def __init__(self, image_names: Sequence[str], image_wh: tuple[int, int],
                 intrinsics: Sequence[tuple[float, float, float, float]], *,
                 native_sizes: Sequence[tuple[int, int] | None] | None = None,
                 depth_dir: str | Path | None = None, normal_dir: str | Path | None = None,
                 normal_encoding: str = "slam", cache_views: int = 2, decode_workers: int = 2,
                 check_normals: bool = True, train_indices: Sequence[int] | None = None) -> None:
        """Indexed like ``image_names``. With ``train_indices`` only those views' priors are matched, loaded and
        checked: a held-out view is never trained on, so its prior is never read."""
        self.image_wh = tuple(image_wh)
        n = len(image_names)
        train = list(range(n)) if train_indices is None else sorted(set(int(i) for i in train_indices))
        what = "images" if train_indices is None else "training images"
        self._native = [tuple(s) if s is not None else self.image_wh
                        for s in (native_sizes or [None] * n)]
        self._depth: list[bytes | None] = [None] * n
        self._normal: list[bytes | None] = [None] * n
        self.depth_hw = self.normal_hw = (1, 1)
        self.counts = {"depth": 0, "normal": 0, "images": n}
        def matched(prior_dir, kind) -> list[Path | None]:
            found = match_priors(prior_dir, [image_names[i] for i in train], kind, what)
            paths: list[Path | None] = [None] * n
            for i, path in zip(train, found):
                paths[i] = path
            return paths

        if depth_dir is not None:
            self._depth, self.depth_hw = self._load(matched(depth_dir, "depth"),
                                                    lambda b, p: decode_depth(b, str(p)),
                                                    lambda b: _sniff(b) == "png")
            self.counts["depth"] = sum(b is not None for b in self._depth)
        if normal_dir is not None:
            self._normal, self.normal_hw = self._load(
                matched(normal_dir, "normal"),
                lambda b, p: decode_normal(b, normal_encoding, str(p)),
                lambda b: _sniff(b) == "png" and normal_encoding == "slam")
            self.counts["normal"] = sum(b is not None for b in self._normal)
            if check_normals:
                have = [i for i in range(n) if self._normal[i] is not None]
                picks = [have[i] for i in check_indices(len(have))]
                check_normal_priors([(image_names[i], self._decode_normal(i), intrinsics[i]) for i in picks],
                                    self.image_wh)
        self._cache = _DecodeCache(self._decode, cache_views, decode_workers)

    def _load(self, paths: list[Path | None], decode, keep_bytes) -> tuple[list[bytes | None], tuple[int, int]]:
        """Two passes, so at most one decoded prior is alive: learn each grid, then hold every
        prior as PNG bytes on the common grid (the file's own bytes when nothing changes)."""
        shapes = []
        for i, p in enumerate(paths):
            if p is None:
                continue
            data = p.read_bytes()
            # A PNG's grid is in its header; only a TIFF is decoded to learn it (pass 2 decodes every file).
            shape = _png_shape(data) or decode(data, p).shape[:2]
            check_aspect(shape, self._native[i], str(p))
            shapes.append(shape)
        hw = common_grid(shapes, self.image_wh)
        out: list[bytes | None] = []
        for p in paths:
            if p is None:
                out.append(None)
                continue
            data = p.read_bytes()
            image = decode(data, p)
            if image.shape[:2] == hw and keep_bytes(data):
                out.append(data)
            else:
                out.append(_encode_png(nearest_resize(image, hw)))
        return out, hw

    def _decode_depth(self, i: int) -> np.ndarray | None:
        data = self._depth[i]
        return None if data is None else _imdecode(data, "depth")

    def _decode_normal(self, i: int) -> np.ndarray | None:
        data = self._normal[i]
        if data is None:
            return None
        rgb = _imdecode(data, "normal")[..., ::-1]
        rgba = np.empty((*rgb.shape[:2], 4), np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        return rgba

    def _decode(self, i: int) -> tuple:
        return self._decode_depth(i), self._decode_normal(i)

    def entry(self, i: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._cache.get(int(i))

    def prefetch(self, indices: Iterable[int]) -> None:
        self._cache.prefetch(indices)

    def cache_stats(self) -> dict:
        return self._cache.stats()

    def host_bytes(self) -> dict:
        encoded = sum(len(b) for b in self._depth if b) + sum(len(b) for b in self._normal if b)
        decoded = sum(sum(a.nbytes for a in e if a is not None) for e in self._cache.cached_entries())
        return {"prior_encoded_bytes": encoded, "prior_decoded_bytes": decoded}

    def close(self) -> None:
        self._cache.close()
