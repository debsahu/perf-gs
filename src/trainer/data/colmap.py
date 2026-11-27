# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trainer.data.dataset import Camera

_SH_C0 = 0.28209479177387814

@dataclass
class ColmapImageEntry:
    # image id starts at 1
    image_id: int
    image_name: str
    camera: Camera


@dataclass
class ColmapDataset:
    images: list[ColmapImageEntry]
    image_dir: Path
    initial_points: dict[str, np.ndarray]
    scene_extent: float = 1.0              # bounding sphere radius
    downsample: int = 1


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _parse_cameras(path: Path) -> dict[int, dict]:
    """Return dict of camera_id -> {model, width, height, fx, fy, cx, cy}."""
    cameras: dict[int, dict] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            if model == "SIMPLE_PINHOLE":
                f_, cx, cy = params
                fx = fy = f_
            elif model == "PINHOLE":
                fx, fy, cx, cy = params
            else:
                raise ValueError(f"unsupported COLMAP camera model {model}")
            cameras[cam_id] = {
                "model": model, "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return cameras


def _parse_images(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # Alternating: metadata, feature-points, metadata, feature-points, ...
    for meta in lines[::2]:
        parts = meta.split()
        entries.append({
            "image_id": int(parts[0]),
            "qw": float(parts[1]), "qx": float(parts[2]),
            "qy": float(parts[3]), "qz": float(parts[4]),
            "tx": float(parts[5]), "ty": float(parts[6]), "tz": float(parts[7]),
            "camera_id": int(parts[8]),
            "name": parts[9],
        })
    return entries


def _parse_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions (N,3) float32, colors (N,3) float32 in [0,1])."""
    pts: list[list[float]] = []
    colors: list[list[float]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0])
    positions = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
    rgb = np.array(colors, dtype=np.float32) if colors else np.zeros((0, 3), dtype=np.float32)
    return positions, rgb


def _view_matrix(qw: float, qx: float, qy: float, qz: float,
                 tx: float, ty: float, tz: float) -> np.ndarray:
    R = _quat_wxyz_to_rotmat(np.array([qw, qx, qy, qz]))
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def _projection_matrix(fx: float, fy: float, cx: float, cy: float,
                       width: int, height: int,
                       z_near: float = 0.01, z_far: float = 100.0) -> np.ndarray:
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 * fx / width
    M[1, 1] = 2.0 * fy / height
    M[0, 2] = 2.0 * cx / width - 1.0
    M[1, 2] = 2.0 * cy / height - 1.0
    M[2, 2] = z_far / (z_far - z_near)
    M[2, 3] = -z_far * z_near / (z_far - z_near)
    M[3, 2] = 1.0
    return M


def _srgb_colors_to_sh0(srgb: np.ndarray) -> np.ndarray:
    return ((srgb - 0.5) / _SH_C0).astype(np.float32)


def _find_sparse_dir(scene_dir: Path) -> tuple[Path | None, str]:
    """Tries sparse/0/text/*.txt, then sparse/0/*.txt, then sparse/0/*.bin."""
    d = scene_dir / "sparse" / "0" / "text"
    if (d / "cameras.txt").exists():
        return d, "text"
    d = scene_dir / "sparse" / "0"
    if (d / "cameras.txt").exists():
        return d, "text"
    if (d / "cameras.bin").exists():
        return d, "bin"
    return None, ""


def _parse_cameras_bin(path: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    with path.open("rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, w, h = struct.unpack("<iIQQ", f.read(24))
            # Model IDs: 0=SIMPLE_PINHOLE(3), 1=PINHOLE(4), ...
            if model_id == 0:
                params = struct.unpack("<3d", f.read(24))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                model = "SIMPLE_PINHOLE"
            elif model_id == 1:
                params = struct.unpack("<4d", f.read(32))
                fx, fy, cx, cy = params
                model = "PINHOLE"
            else:
                # Skip unknown models — read param count from model table
                num_params = {2: 4, 3: 5, 4: 5, 5: 12}.get(model_id, 4)
                f.read(num_params * 8)
                continue
            cameras[cam_id] = {
                "model": model, "width": int(w), "height": int(h),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return cameras


def _parse_images_bin(path: Path, track_names: set[str] | None = None) -> list[dict]:
    entries: list[dict] = []
    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            # Image name: null-terminated string.
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            # Only calibration needs registered observations, and only for its sampled train views.
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            tracks = {}
            if track_names is not None and name in track_names:
                dtype = np.dtype([("xy", "<f8", 2), ("id", "<i8")])
                features = np.frombuffer(f.read(num_points2d * 24), dtype)
                tracks["tracks"] = features[features["id"] >= 0]
            else:
                f.seek(num_points2d * 24, 1)
            entries.append({
                "image_id": image_id,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id,
                "name": name,
                **tracks,
            })
    return entries


def _parse_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header = struct.Struct("<8x3d3B8xQ")  # skip the point ID and reprojection error
    pts: list[list[float]] = []
    colors: list[list[float]] = []
    with path.open("rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            x, y, z, r, g, b, track_length = header.unpack(f.read(header.size))
            f.seek(track_length * 8, 1)
            pts.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    positions = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
    rgb = np.array(colors, dtype=np.float32) if colors else np.zeros((0, 3), dtype=np.float32)
    return positions, rgb


def load_colmap(
    scene_dir: Path | str,
    width: int | None = None,
    height: int | None = None,
    downsample: int | None = 1,
    init_opacity: float = 0.1,
) -> ColmapDataset:
    scene_dir = Path(scene_dir)
    if downsample is None:
        # The Mip-NeRF 360 benchmark uses half resolution indoors and quarter outdoors.
        downsample = 4
        path = scene_dir.resolve()
        if path.name.lower() in {"bonsai", "counter", "kitchen", "room"}:
            known_root = path.parent.name.lower() in {"360_v2", "mipnerf360", "mipnerf360_v2"}
            released_scene = (path / "poses_bounds.npy").is_file() and (path / "images_2").is_dir()
            if known_root or released_scene or all((path.parent / s).is_dir() for s in ("bicycle", "garden", "stump")):
                downsample = 2
    if downsample < 1:
        raise ValueError("--downsample must be at least 1")

    if downsample > 1 and (scene_dir / f"images_{downsample}").exists():
        img_dir = scene_dir / f"images_{downsample}"
    else:
        img_dir = scene_dir / "images"

    sparse_dir, fmt = _find_sparse_dir(scene_dir)
    if sparse_dir is None:
        raise FileNotFoundError(
            f"No COLMAP sparse reconstruction found in {scene_dir}/sparse/0/ "
            f"(tried text/*.txt, *.txt, *.bin)")

    if fmt == "text":
        cams = _parse_cameras(sparse_dir / "cameras.txt")
        imgs = _parse_images(sparse_dir / "images.txt")
        pts3d, pts_colors = _parse_points3d(sparse_dir / "points3D.txt")
    else:
        cams = _parse_cameras_bin(sparse_dir / "cameras.bin")
        imgs = _parse_images_bin(sparse_dir / "images.bin")
        pts3d, pts_colors = _parse_points3d_bin(sparse_dir / "points3D.bin")

    imgs = sorted(imgs, key=lambda r: r["name"])

    size_counts: dict[tuple[int, int], int] = {}
    for rec in imgs:
        cam = cams[rec["camera_id"]]
        key = (cam["width"], cam["height"])
        size_counts[key] = size_counts.get(key, 0) + 1
    # Ties break on pixel count, so the choice never depends on dict order.
    native_w, native_h = max(size_counts,
                             key=lambda s: (size_counts[s], s[0] * s[1]))
    uniform = len(size_counts) == 1
    if not uniform:
        spread = ", ".join(f"{w}x{h}x{n}" for (w, h), n in
                           sorted(size_counts.items(), key=lambda kv: -kv[1]))
        print(f"[colmap] {len(size_counts)} native camera sizes in this model "
              f"({spread}) — typical of an undistorted reconstruction. Rendering "
              f"at the most common size ({native_w}x{native_h}); the other "
              f"{len(imgs) - size_counts[(native_w, native_h)]} views are "
              f"resampled on load and keep their own intrinsics.")

    if width is None or height is None:
        # Prefer images_N's real size (the downsampler rounds); mixed-size models use the records.
        disk_w = disk_h = None
        if uniform and img_dir != scene_dir / "images":
            try:
                from PIL import Image
                sample = next(p for p in sorted(img_dir.iterdir())
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
                disk_w, disk_h = Image.open(sample).size
            except (StopIteration, OSError):
                disk_w = disk_h = None
        if width is None:
            width = disk_w if disk_w is not None else native_w // downsample
        if height is None:
            height = disk_h if disk_h is not None else native_h // downsample

    image_entries: list[ColmapImageEntry] = []
    for rec in imgs:
        cam = cams[rec["camera_id"]]
        # Per-camera scale: load_colmap_image stretches each view to the target from its own size.
        sx = width / cam["width"]
        sy = height / cam["height"]
        view = _view_matrix(
            rec["qw"], rec["qx"], rec["qy"], rec["qz"],
            rec["tx"], rec["ty"], rec["tz"],
        )
        proj = _projection_matrix(
            fx=cam["fx"] * sx, fy=cam["fy"] * sy,
            cx=cam["cx"] * sx, cy=cam["cy"] * sy,
            width=width, height=height,
        )
        camera = Camera(
            view_matrix=view.astype(np.float32),
            proj_matrix=proj.astype(np.float32),
            image_size=(width, height),
        )
        image_entries.append(ColmapImageEntry(
            image_id=rec["image_id"], image_name=rec["name"], camera=camera,
        ))

    n = pts3d.shape[0]

    # gsplat-style k-NN scale init: SfM point density varies too much for a uniform scale.
    from scipy.spatial import cKDTree
    k = 3
    if n >= k + 1:
        tree = cKDTree(pts3d)
        dists, _ = tree.query(pts3d, k=k + 1)   # (n, k+1), col 0 is self
        dist_sq = (dists[:, 1:k + 1] ** 2).mean(axis=-1).astype(np.float32)
        dist_sq = np.clip(dist_sq, 1e-7, None)
        knn_dist = np.sqrt(dist_sq)
    else:
        knn_dist = np.full(n, 0.01, dtype=np.float32)
    log_scale_per_point = np.log(knn_dist)

    initial_points = {
        "means": pts3d,
        "log_scales": np.repeat(log_scale_per_point[:, None], 3, axis=1).astype(np.float32),
        "rot_quats": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        # Probabilities; converted to logits on upload.
        "opacities": np.full(n, init_opacity, dtype=np.float32),
        # SH band-0 (DC) coefficients from the sRGB point colors, despite the key's name.
        "color_logits": _srgb_colors_to_sh0(pts_colors),
    }

    # Camera bounding-sphere radius * 1.1 (the 3DGS convention); scales densify thresholds.
    cam_positions = []
    for rec in imgs:
        R = _quat_wxyz_to_rotmat(np.array([rec["qw"], rec["qx"], rec["qy"], rec["qz"]]))
        t = np.array([rec["tx"], rec["ty"], rec["tz"]])
        cam_positions.append(-R.T @ t)
    cam_positions = np.array(cam_positions, dtype=np.float64)
    cam_centroid = cam_positions.mean(axis=0)
    scene_extent = float(np.linalg.norm(cam_positions - cam_centroid, axis=1).max()) * 1.1

    return ColmapDataset(
        images=image_entries,
        image_dir=img_dir,
        initial_points=initial_points,
        scene_extent=scene_extent,
        downsample=downsample,
    )


def load_colmap_image(entry: ColmapImageEntry, image_dir: Path, width: int, height: int,
               load_alpha: bool = False) -> np.ndarray:
    """Read a PNG/JPG as float32 sRGB in [0, 1] at (width, height).

    (H, W, 4) when ``load_alpha`` and the image has alpha, else (H, W, 3). EXIF
    orientation is ignored: rotating would disagree with the COLMAP intrinsics.
    """
    return to_float01(load_colmap_image_native(entry, image_dir, width, height, load_alpha))


def load_colmap_image_native(entry: ColmapImageEntry, image_dir: Path, width: int, height: int,
                             load_alpha: bool = False) -> np.ndarray:
    """``load_colmap_image`` before its float conversion: RGB(A) at (width, height) in the
    source's own integer dtype (uint8 for 8-bit images). Any resize happens here, in that
    dtype, exactly as ``load_colmap_image`` has always done it."""
    import cv2
    path = image_dir / entry.image_name
    flags = _image_flags(load_alpha)
    array = cv2.imread(str(path), flags)
    if array is None and path.exists():
        # cv2.imread uses the ANSI file API on Windows; decode the bytes for non-code-page paths.
        array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if array is None:
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        raise OSError(f"failed to decode image: {path}")
    return _to_rgb_at_size(array, path, width, height, load_alpha)


def read_colmap_image_bytes(entry: ColmapImageEntry, image_dir: Path) -> np.ndarray:
    """The image file of ``entry`` as it is on disk (uint8 bytes), for ``decode_colmap_image``."""
    path = image_dir / entry.image_name
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise OSError(f"image file is empty: {path}")
    return data


def decode_colmap_image(data: np.ndarray, name: str | Path, width: int, height: int,
                        load_alpha: bool = False) -> np.ndarray:
    """``load_colmap_image_native`` from the file's bytes (``read_colmap_image_bytes``):
    the same decoder flags, channel order and resize."""
    import cv2
    try:
        array = cv2.imdecode(data, _image_flags(load_alpha)) if data.size else None
    except cv2.error as error:          # an empty or malformed buffer raises instead of returning None
        raise OSError(f"failed to decode image: {name}: {error}") from error
    if array is None:
        raise OSError(f"failed to decode image: {name}")
    return _to_rgb_at_size(array, name, width, height, load_alpha)


def _image_flags(load_alpha: bool) -> int:
    import cv2
    # IMREAD_COLOR collapses grayscale/alpha sources to 3-channel 8-bit BGR.
    return cv2.IMREAD_IGNORE_ORIENTATION | (cv2.IMREAD_UNCHANGED if load_alpha else cv2.IMREAD_COLOR)


def _to_rgb_at_size(array: np.ndarray, path, width: int, height: int,
                    load_alpha: bool) -> np.ndarray:
    import cv2
    channels = 1 if array.ndim == 2 else array.shape[2]
    if load_alpha and channels == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
    elif channels == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    elif channels == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif channels == 1:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        raise OSError(f"unsupported {channels}-channel image: {path}")

    if (array.shape[1], array.shape[0]) != (width, height):
        # INTER_AREA degenerates to nearest-neighbour when enlarging (off-size undistorted views).
        shrinking = width <= array.shape[1] and height <= array.shape[0]
        array = cv2.resize(array, (width, height),
                           interpolation=cv2.INTER_AREA if shrinking
                           else cv2.INTER_LINEAR)
    return array


def find_mask_path(mask_dir: Path, image_name: str) -> Path | None:
    """The loss mask of ``image_name``: ``<mask_dir>/<image name>.png``, else ``<mask_dir>/<stem>.png``.
    Subfolders of the image name are kept in both forms. None when neither file exists."""
    name = Path(image_name)
    for candidate in (mask_dir / name.parent / f"{name.name}.png",
                      mask_dir / name.parent / f"{name.stem}.png"):
        if candidate.is_file():
            return candidate
    return None


def require_some_mask(mask_dir: Path, image_names) -> None:
    """Raise unless at least one image has a loss mask in ``mask_dir`` (the target dataset's own check)."""
    names = list(image_names)
    if not any(find_mask_path(Path(mask_dir), name) is not None for name in names):
        raise FileNotFoundError(f"--masks {mask_dir}: no mask matched any of the {len(names)} images "
                                "(looked for <image name>.png and <stem>.png)")


def load_mask_file(path: Path, width: int, height: int, invert: bool = False) -> np.ndarray:
    """A loss mask as uint8 (H, W): 255 = keep the pixel, 0 = drop it.

    The file is read as grayscale and inverted first when ``invert`` (for 255 = drop masks).
    Resizing averages the keep fraction of each output pixel (area resampling when shrinking,
    nearest when enlarging) and then thresholds it at one half. A mask is a gate, not a weight:
    a boundary pixel that is mostly dropped content still carries that content's color, so
    giving it a partial weight would keep training on what the mask removes. Ties keep."""
    import cv2
    flags = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
    array = cv2.imread(str(path), flags)
    if array is None and path.exists():
        array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if array is None:
        raise OSError(f"failed to decode mask: {path}")
    keep = array.astype(np.float32) * (1.0 / 255.0)
    if invert:
        keep = 1.0 - keep
    if (keep.shape[1], keep.shape[0]) != (width, height):
        shrinking = width <= keep.shape[1] and height <= keep.shape[0]
        keep = cv2.resize(keep, (width, height),
                          interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_NEAREST)
    return np.where(keep >= 0.5, np.uint8(255), np.uint8(0))


def load_colmap_mask(entry: ColmapImageEntry, mask_dir: Path, width: int, height: int,
                     invert: bool = False) -> np.ndarray | None:
    """``load_mask_file`` for an image's mask, or None when the image has none."""
    path = find_mask_path(Path(mask_dir), entry.image_name)
    return None if path is None else load_mask_file(path, width, height, invert)


def to_float01(array: np.ndarray) -> np.ndarray:
    """The float32 [0, 1] form ``load_colmap_image`` returns, from a decoded integer image.
    For uint8, ``rint(to_float01(u8) * 255) == u8`` on all 256 values, so a packed
    rgba8 target can be made from the uint8 decode directly."""
    import cv2
    # cv2.multiply frees the GIL numpy's cast holds; a bare scalar would only scale channel 0.
    if array.dtype.kind in "ui":
        scale = 1.0 / float(np.iinfo(array.dtype).max)
        array = cv2.multiply(array, (scale,) * 4, dtype=cv2.CV_32F)
    else:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)
