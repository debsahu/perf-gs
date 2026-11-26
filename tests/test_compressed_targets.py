# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Compressed training targets (--image-residency compressed).

Each target stays in host RAM as its encoded file bytes (the loss mask as a lossless PNG of its
final, resized form) and is decoded on demand into exactly the packed array the decoded modes
hold. A small bounded cache serves the decodes; the training loop prefetches the views the
sampler will draw next on a thread pool. CPU only.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from trainer.data.colmap import ColmapDataset, ColmapImageEntry
from trainer.data.dataset import simple_camera
from trainer.data.targets import CompressedTargetViews, TargetDataset
from trainer.frontend.sampling import CameraSampler

W, H = 37, 29          # odd on purpose
DISK = (53, 41)        # larger than the training size: the shrink resize branch


def _write(path: Path, array: np.ndarray, params=()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array, list(params))


def _texture(rng, w, h, c, dtype=np.uint8):
    top = np.iinfo(dtype).max
    return rng.integers(0, top + 1, (h, w, c) if c > 1 else (h, w), dtype=dtype)


def _scene(root: Path, kind: str, size=(W, H), n=6) -> ColmapDataset:
    """Images of one decode branch: jpeg, png (BGR), gray, rgba, png16."""
    rng = np.random.default_rng(11)
    entries = []
    for i in range(n):
        name = f"sub/v{i:02d}." + ("jpg" if kind == "jpeg" else "png")
        if kind == "jpeg":
            _write(root / "images" / name, _texture(rng, *size, 3), (cv2.IMWRITE_JPEG_QUALITY, 90))
        elif kind == "png":
            _write(root / "images" / name, _texture(rng, *size, 3))
        elif kind == "gray":
            _write(root / "images" / name, _texture(rng, *size, 1))
        elif kind == "rgba":
            _write(root / "images" / name, _texture(rng, *size, 4))
        elif kind == "png16":
            _write(root / "images" / name, _texture(rng, *size, 3, np.uint16))
        elif kind == "rgba16":
            _write(root / "images" / name, _texture(rng, *size, 4, np.uint16))
        else:
            raise ValueError(kind)
        camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0))
        entries.append(ColmapImageEntry(i + 1, name, camera))
    return ColmapDataset(entries, root / "images", {})


def _masks(root: Path, scene: ColmapDataset, size=(W, H)) -> Path:
    """Masks for every image but the last (which then keeps every pixel); the first is under
    <image name>.png, the rest under <stem>.png. Not binary, so the threshold is exercised."""
    rng = np.random.default_rng(5)
    for i, entry in enumerate(scene.images[:-1]):
        name = Path(entry.image_name)
        file = f"{name.name}.png" if i == 0 else f"{name.stem}.png"
        _write(root / "masks" / name.parent / file, _texture(rng, *size, 1))
    return root / "masks"


def _pack():
    """TargetImages.pack without a device (it is pure numpy)."""
    from trainer.backend.render.targets import TargetImages
    return TargetImages(None, None, None, W, H).pack


def _pair(scene, *, use_alpha=False, masks=None, invert=False, **kwargs):
    decoded = TargetDataset(scene, W, H, use_alpha=use_alpha, masks_dir=masks, invert_masks=invert)
    compressed = TargetDataset(scene, W, H, use_alpha=use_alpha, masks_dir=masks, invert_masks=invert,
                               residency="compressed", **kwargs)
    return decoded, compressed


CASES = [
    ("jpeg", {}), ("png", {}), ("gray", {}), ("png16", {}),
    ("rgba", {"use_alpha": True}), ("rgba", {}),
    # 16 bits survive the decode only with alpha loading (IMREAD_UNCHANGED); IMREAD_COLOR makes them 8-bit.
    ("png16", {"use_alpha": True}), ("rgba16", {"use_alpha": True}),
    ("jpeg", {"masks": True}), ("png16", {"masks": True}), ("png", {"masks": True, "invert": True}),
]


@pytest.mark.parametrize("disk", [(W, H), DISK], ids=["native", "resized"])
@pytest.mark.parametrize("kind,opts", CASES, ids=[f"{k}-{'-'.join(o) or 'plain'}" for k, o in CASES])
def test_packed_bytes_match_decoded_mode(tmp_path, kind, opts, disk):
    """Every byte a consumer can see is the decoded mode's: resident, packed, float view, keep mask."""
    scene = _scene(tmp_path, kind, size=disk)
    masks = _masks(tmp_path, scene, size=disk) if opts.get("masks") else None
    decoded, compressed = _pair(scene, use_alpha=opts.get("use_alpha", False), masks=masks,
                                invert=opts.get("invert", False))
    assert compressed.compressed and not decoded.compressed
    pack = _pack()
    views = CompressedTargetViews(compressed, pack)
    for i in range(len(scene.images)):
        a, b = decoded.resident[i], compressed.resident[i]
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
        pa, pb = pack(a), views[i]
        assert pa.dtype == pb.dtype == np.uint8 and pa.shape == pb.shape == (H, W, 4)
        assert pa.tobytes() == pb.tobytes()
        fa, fb = decoded.srgb[i], compressed.srgb[i]
        assert fa.shape == fb.shape and fa.tobytes() == fb.tobytes()
        if masks is None:
            assert compressed.keep_mask(i) is None and compressed.keep_masks is None
        else:
            np.testing.assert_array_equal(decoded.keep_mask(i), compressed.keep_mask(i))
    if masks is not None:
        assert compressed.mask_count == decoded.mask_count == len(scene.images) - 1


@pytest.mark.parametrize("kind", ["png16", "rgba16"])
def test_sixteen_bit_fixture_stays_float(tmp_path, kind):
    """The 16-bit cases must reach the float residency, or they test nothing a uint8 case does not."""
    scene = _scene(tmp_path, kind)
    decoded, compressed = _pair(scene, use_alpha=True)
    assert decoded.resident[0].dtype == compressed.resident[0].dtype == np.float32
    assert np.unique(np.rint(decoded.resident[0][..., 0] * 65535) % 257).size > 1   # not an 8-bit image scaled up


def test_fixture_discriminates_decoder_and_resize(tmp_path):
    """The fixture must be able to fail: a different resize kernel or channel order changes bytes,
    and the masks are not all-keep."""
    scene = _scene(tmp_path, "png", size=DISK)
    decoded = TargetDataset(scene, W, H, use_alpha=False)
    raw = cv2.imread(str(scene.image_dir / scene.images[0].image_name), cv2.IMREAD_COLOR)
    linear = cv2.cvtColor(cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    area = cv2.cvtColor(cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    assert decoded.resident[0][..., :3].tobytes() == area.tobytes()
    assert area.tobytes() != linear.tobytes()
    assert area.tobytes() != area[..., ::-1].tobytes()
    masks = _masks(tmp_path, scene, size=DISK)
    masked = TargetDataset(scene, W, H, use_alpha=False, masks_dir=masks)
    keep = masked.keep_mask(0)
    assert 0.2 < keep.mean() < 0.8


def test_mask_rides_in_alpha(tmp_path):
    scene = _scene(tmp_path, "jpeg")
    masks = _masks(tmp_path, scene)
    _, compressed = _pair(scene, masks=masks)
    for i in range(len(scene.images)):
        alpha = compressed.resident[i][..., 3]
        assert set(np.unique(alpha)) <= {0, 255}
        np.testing.assert_array_equal(alpha == 255, compressed.keep_mask(i))
        assert compressed.srgb[i].shape == (H, W, 3)
    assert compressed.keep_mask(len(scene.images) - 1).all()
    assert not compressed.keep_mask(0).all()


def test_holds_encoded_bytes_not_pixels(tmp_path):
    """RAM held per view is the file, not the decoded image."""
    scene = _scene(tmp_path, "jpeg", n=8)
    decoded, compressed = _pair(scene)
    files = sum((scene.image_dir / e.image_name).stat().st_size for e in scene.images)
    held = compressed.host_image_bytes()
    assert held["encoded_images_bytes"] == files
    assert held["decoded_images_bytes"] == 0
    assert decoded.host_image_bytes()["decoded_images_bytes"] == 8 * W * H * 4


def test_startup_decodes_no_image(tmp_path, monkeypatch):
    """Construction, the memory report and len() decode nothing."""
    import trainer.data.targets as targets_module
    calls = []
    real = targets_module.decode_colmap_image
    monkeypatch.setattr(targets_module, "decode_colmap_image",
                        lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    scene = _scene(tmp_path, "jpeg", n=10)
    compressed = TargetDataset(scene, W, H, use_alpha=False, residency="compressed",
                               masks_dir=_masks(tmp_path, scene))
    compressed.host_image_bytes()
    assert len(compressed.resident) == len(compressed.srgb) == 10
    views = CompressedTargetViews(compressed, _pack())
    assert views.resident_nbytes() == 0 and len(views) == 10
    assert calls == []
    _ = compressed.resident[3]
    assert len(calls) == 1


def _counting(tmp_path, n=12, **kwargs):
    scene = _scene(tmp_path, "png", n=n)
    targets = TargetDataset(scene, W, H, use_alpha=False, residency="compressed", **kwargs)
    log = []
    lock = threading.Lock()
    real = targets._cache._decode

    def decode(index):
        with lock:
            log.append(index)
        return real(index)
    targets._cache._decode = decode
    return targets, log


def test_cache_is_bounded_and_counts_hits(tmp_path):
    targets, log = _counting(tmp_path, n=12, cache_views=3)
    for i in range(12):
        _ = targets.resident[i]
        assert targets.cache_stats()["cached"] <= 3
    assert targets.cache_stats()["peak_cached"] == 3
    assert log == list(range(12))
    _ = targets.resident[11]
    assert log == list(range(12))          # a hit, no decode
    _ = targets.resident[0]
    assert log == list(range(12)) + [0]    # evicted, decoded again


def test_prefetch_decodes_ahead_in_order_and_pins(tmp_path):
    targets, log = _counting(tmp_path, n=12, cache_views=2, decode_workers=1)
    targets.prefetch([4, 7, 2, 9])
    deadline = time.time() + 10
    while targets.cache_stats()["cached"] < 4 and time.time() < deadline:
        time.sleep(0.01)
    assert log == [4, 7, 2, 9]              # one worker: submission order
    # pinned views are not evicted by other accesses, even past the LRU capacity
    for i in (0, 1, 3, 5):
        _ = targets.resident[i]
    before = list(log)
    for i in (4, 7, 2, 9):
        _ = targets.resident[i]
    assert log == before
    assert targets.cache_stats()["cached"] <= 4 + 2
    # a new window releases the old pins
    targets.prefetch([1])
    for i in (6, 8, 10, 11):
        _ = targets.resident[i]
    assert targets.cache_stats()["cached"] <= 1 + 2


def test_concurrent_reads_decode_once(tmp_path):
    targets, log = _counting(tmp_path, n=4, cache_views=4)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: targets.resident[2], range(32)))
    assert log == [2]


def test_write_ground_truths_streams(tmp_path):
    """Writing every camera's PNG keeps at most the cache's worth of views decoded or decoding."""
    targets, log = _counting(tmp_path, n=12, cache_views=3)
    live, peak, lock = [0], [0], threading.Lock()
    inner = targets._cache._decode

    def slow(index):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.02)
        try:
            return inner(index)
        finally:
            with lock:
                live[0] -= 1
    targets._cache._decode = slow
    targets.write_ground_truths(tmp_path / "cams")
    assert sorted(log) == list(range(12))
    assert targets.cache_stats()["peak_cached"] <= 3
    assert 1 < peak[0] <= 3                 # parallel, but bounded by the cache
    decoded = TargetDataset(_scene(tmp_path / "d", "png", n=12), W, H, use_alpha=False)
    decoded.write_ground_truths(tmp_path / "dcams")
    for i in range(12):
        a = (tmp_path / "cams" / f"cam_{i:02d}" / "gt.png").read_bytes()
        b = (tmp_path / "dcams" / f"cam_{i:02d}" / "gt.png").read_bytes()
        assert a == b


@pytest.mark.parametrize("mode", ["ours", "sadgs"])
def test_sampler_peek_does_not_change_the_order(mode):
    train, test = list(range(1, 30, 2)), [0, 8]
    plain = CameraSampler(mode, train, test, 3, np.random.default_rng(3))
    peeking = CameraSampler(mode, train, test, 3, np.random.default_rng(3))
    expected = [plain.next() for _ in range(80)]
    got = []
    for step in range(70):
        ahead = peeking.peek(step % 5)
        nxt = peeking.next()
        got.append(nxt)
        if ahead:
            assert ahead[0] == nxt
            assert ahead == expected[step:step + len(ahead)]
    assert got == expected[:70]


def test_training_loop_prefetches_the_sampler_window():
    """_prepare_iteration pins the current view and the next --target-prefetch draws, then selects."""
    from trainer.frontend.app import TrainingLoop
    calls, selected = [], []
    sampler = CameraSampler("ours", list(range(10)), [], 1, np.random.default_rng(1))
    reference = CameraSampler("ours", list(range(10)), [], 1, np.random.default_rng(1))

    class Views(dict):
        def prefetch(self, indices):
            calls.append(list(indices))

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.sampler = sampler
    loop.args = SimpleNamespace(target_prefetch=3)
    loop.target_textures = Views({i: f"t{i}" for i in range(10)})
    loop.runtime = SimpleNamespace(pipeline=SimpleNamespace(targets=SimpleNamespace(select=selected.append)),
                                   queue_limiter=SimpleNamespace(before_submit=lambda: None))
    order = [reference.next() for _ in range(9)]
    for step in range(5):
        camera = sampler.next()
        loop._prepare_iteration(SimpleNamespace(camera_index=camera, step_number=step + 2))
        assert calls[-1] == order[step:step + 4]
        assert selected[-1] == f"t{camera}"


def test_cli_accepts_compressed_residency():
    from trainer.frontend.cli import parse_args
    args = parse_args(["--image-residency", "compressed", "--target-prefetch", "6"])
    assert args.image_residency == "compressed" and args.target_prefetch == 6
    assert parse_args([]).image_residency == "stream" and parse_args([]).target_prefetch == 4
