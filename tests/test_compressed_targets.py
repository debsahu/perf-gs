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


def test_stats_time_decodes_and_training_waits(tmp_path):
    """decode_s sums the decodes; wait_s counts only time a reader blocked on a view not yet decoded."""
    targets, _ = _counting(tmp_path, n=4, cache_views=4, decode_workers=1)
    inner = targets._cache._decode
    targets._cache._decode = lambda i: time.sleep(0.05) or inner(i)
    targets.prefetch([0, 1])
    time.sleep(0.3)                          # both decoded ahead: reading them does not wait
    _ = targets.resident[0], targets.resident[1]
    stats = targets.cache_stats()
    assert stats["wait_s"] < 0.02 and stats["waits"] == 0 and stats["decode_s"] >= 0.1
    _ = targets.resident[2]                  # not prefetched: the reader decodes it and waits
    stats = targets.cache_stats()
    assert stats["waits"] == 1 and stats["wait_s"] >= 0.05


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
    loop.args = SimpleNamespace(target_prefetch=3, iters=100)
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


def test_failed_decode_clears_flight_and_retries(tmp_path):
    """A decode that raises surfaces on the reader, leaves nothing in flight, and the next read decodes again."""
    targets, log = _counting(tmp_path, n=3, cache_views=2)
    inner = targets._cache._decode
    fail = [True]

    def flaky(index):
        if fail[0]:
            fail[0] = False
            raise OSError("disk hiccup")
        return inner(index)
    targets._cache._decode = flaky
    with pytest.raises(OSError, match="disk hiccup"):
        _ = targets.resident[1]
    assert targets._cache._flight == {}
    assert targets.resident[1].shape == (H, W, 4)
    assert targets.cache_stats()["decodes"] == 1


def test_packing_runs_on_the_decode_worker(tmp_path):
    """Prefetched views are packed on the worker thread, so the training thread only binds the result."""
    scene = _scene(tmp_path, "png16")
    targets = TargetDataset(scene, W, H, use_alpha=True, residency="compressed", decode_workers=1)
    pack = _pack()
    threads = []
    views = CompressedTargetViews(targets, lambda image: threads.append(threading.current_thread().name) or pack(image))
    views.prefetch([0, 1])
    deadline = time.time() + 10
    while targets.cache_stats()["cached"] < 2 and time.time() < deadline:
        time.sleep(0.01)
    before = list(threads)
    assert len(before) == 2 and all(t.startswith("target-decode") for t in before), before
    assert views[0].dtype == np.uint8 and views[1].dtype == np.uint8
    assert threads == before                  # reading a prefetched view packs nothing on this thread


def test_close_fails_pending_reads_instead_of_hanging(tmp_path):
    """close() cancels queued decodes; a reader waiting on one gets an error, never a hang."""
    targets, _ = _counting(tmp_path, n=6, cache_views=2, decode_workers=1)
    inner = targets._cache._decode
    gate = threading.Event()
    targets._cache._decode = lambda i: gate.wait(5) and inner(i)
    targets.prefetch([0, 1, 2])             # 0 runs (blocked on the gate), 1 and 2 stay queued
    time.sleep(0.05)
    result = {}

    def read():
        try:
            result["value"] = targets.resident[2]
        except Exception as error:            # noqa: BLE001
            result["error"] = error
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    time.sleep(0.05)
    targets.close()
    reader.join(2)
    gate.set()
    assert not reader.is_alive(), "get() hung after close()"
    assert isinstance(result.get("error"), RuntimeError)
    with pytest.raises(RuntimeError, match="closed"):
        _ = targets.resident[4]


# ---------------------------------------------------------------- review regressions

def test_memory_report_decodes_no_image(tmp_path, monkeypatch):
    """The --run-report memory summary must not walk the compressed views: each read decodes a view."""
    from trainer.frontend.app import TrainingApplication
    import trainer.frontend.app as app_module
    targets, log = _counting(tmp_path, n=6, cache_views=0)
    views = CompressedTargetViews(targets, _pack())
    stats = SimpleNamespace(to_numpy=lambda: np.zeros(2, np.uint32))
    pipeline = SimpleNamespace(buffers=SimpleNamespace(sort=SimpleNamespace(stats=stats)), max_splats=1,
                               max_sort_pairs=1)
    monkeypatch.setattr(app_module, "allocated_bytes", lambda _buffers: {"x": 0})
    monkeypatch.setattr(app_module, "query_video_memory", lambda _name: None)
    app = SimpleNamespace(_pipeline=pipeline, _stream_targets=False, _targets=targets, target_textures=views,
                          structure_textures={}, args=SimpleNamespace(run_report=True),
                          device=SimpleNamespace(info=SimpleNamespace(adapter_name="x")))
    summary = TrainingApplication._report_memory(app)
    assert log == []
    assert summary["gpu_view_cache"]["training_images_bytes"] == 0
    assert summary["host_ram"]["packed_training_images_bytes"] == 0
    _ = views[2]
    assert log == [2]                       # the control: one read is one decode


def test_float_view_decodes_once_per_read(tmp_path):
    """targets.srgb[i] reads the cached entry once. Reading it twice (array, then channel count) decoded a cold view
    twice at cache_views=0 and recorded a hit that never happened."""
    targets, log = _counting(tmp_path, n=4, cache_views=0)
    _ = targets.srgb[1]
    assert log == [1]
    assert targets.cache_stats()["hits"] == 0 and targets.cache_stats()["misses"] == 1


def test_views_decoded_before_the_packer_are_packed_once(tmp_path):
    """--auto-lens reads targets before CompressedTargetViews sets the packer: those cached views are packed on first
    use and the packed array is kept, not re-packed on every access."""
    targets, log = _counting(tmp_path, n=4, cache_views=4)
    _ = targets.srgb[0]                     # decoded before any packer
    pack, calls = _pack(), []
    views = CompressedTargetViews(targets, lambda image: calls.append(1) or pack(image))
    first = views[0]
    assert views[0] is first and len(calls) == 1 and log == [0]
    _ = targets.srgb[0], targets.resident[0]
    assert len(calls) == 1


def test_a_new_packer_repacks_cached_views(tmp_path):
    targets, _ = _counting(tmp_path, n=2, cache_views=2)
    CompressedTargetViews(targets, lambda image: np.zeros(1, np.uint8))
    _ = targets.resident[0]
    views = CompressedTargetViews(targets, _pack())
    assert views[0].shape == (H, W, 4)


def test_prefetch_normalizes_and_checks_indices(tmp_path):
    targets, log = _counting(tmp_path, n=5, cache_views=0, decode_workers=1)
    targets.prefetch([-1])
    _ = targets.resident[4]                 # pinned under its real index: no second decode
    assert log == [4] and targets.cache_stats()["pinned"] == 1
    with pytest.raises(IndexError):
        targets.prefetch([5])
    with pytest.raises(TypeError):
        targets.prefetch([1.5])
    assert targets.cache_stats()["pinned"] == 1         # a refused window leaves the previous one


def test_compressed_views_have_unique_in_range_keys(tmp_path):
    targets, _ = _counting(tmp_path, n=4)
    views = CompressedTargetViews(targets, _pack(), [2, 0, 2])
    assert list(views) == [2, 0] and len(views) == 2
    with pytest.raises(IndexError):
        CompressedTargetViews(targets, _pack(), [4])


def test_cached_views_are_read_only(tmp_path):
    """The cache hands out the arrays it keeps: a caller writing into one would corrupt every later read."""
    targets, _ = _counting(tmp_path, n=2, cache_views=2)
    views = CompressedTargetViews(targets, _pack())
    for array in (targets.resident[0], views[0]):
        assert not array.flags.writeable
    scene = _scene(tmp_path / "f", "png16")
    for residency in ("decoded", "compressed"):
        floats = TargetDataset(scene, W, H, use_alpha=True, residency=residency)
        image = floats.srgb[0]
        assert image.dtype == np.float32
        with pytest.raises(ValueError):
            image[0, 0, 0] = 2.0


def test_training_starts_with_fresh_cache_stats(tmp_path):
    """Startup reads (auto-lens, ground truths, probes) must not be reported as 'training waited': the counters
    restart when the training loop begins. Cached views survive the reset."""
    from trainer.frontend.app import TrainingLoop
    targets, _ = _counting(tmp_path, n=3, cache_views=3)
    _ = targets.srgb[0], targets.resident[0], targets.resident[1]
    assert targets.cache_stats()["misses"] == 2
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.targets = targets
    loop.args = SimpleNamespace(wait_for_viewer=False)
    loop.runtime = SimpleNamespace(viewer=SimpleNamespace(set_training_active=lambda _on: None),
                                   reporter=SimpleNamespace(start=lambda _t: None))
    loop._begin_training_loop()
    stats = targets.cache_stats()
    assert (stats["hits"], stats["misses"], stats["decodes"], stats["waits"]) == (0, 0, 0, 0)
    assert stats["wait_s"] == stats["decode_s"] == 0.0 and stats["cached"] == 2


class _HookLock:
    """A lock that runs ``after_release`` once, right after its first release: a deterministic interleaving point."""

    def __init__(self, after_release):
        self._lock, self._hook = threading.Lock(), after_release

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *exc):
        self._lock.release()
        hook, self._hook = self._hook, None
        if hook is not None:
            hook()


def test_close_racing_prefetch_leaves_no_pool(tmp_path):
    """close() landing between prefetch's locked section and its pool creation must not leave a new pool behind."""
    targets, _ = _counting(tmp_path, n=3)
    cache = targets._cache
    cache._lock = _HookLock(lambda: cache.close())
    cache.prefetch([0, 1])
    assert cache._pool is None
    with pytest.raises(RuntimeError, match="closed"):
        cache.get(0)


def test_a_failed_submit_fails_the_window_instead_of_hanging(tmp_path, monkeypatch):
    """If the pool refuses a decode, the views already marked in flight must fail, not leave readers waiting."""
    from concurrent.futures import ThreadPoolExecutor
    targets, _ = _counting(tmp_path, n=3, decode_workers=1)

    def refuse(self, *args, **kwargs):
        raise RuntimeError("cannot schedule new futures")
    monkeypatch.setattr(ThreadPoolExecutor, "submit", refuse)
    with pytest.raises(RuntimeError, match="schedule"):
        targets.prefetch([0, 1])
    monkeypatch.undo()
    result = {}
    reader = threading.Thread(target=lambda: result.update(view=targets.resident[1]), daemon=True)
    reader.start()
    reader.join(5)
    assert not reader.is_alive(), "a read hung on a never-submitted decode"
    assert result["view"].shape == (H, W, 4)


def test_close_clears_the_pins_and_skips_queued_decodes(tmp_path):
    targets, log = _counting(tmp_path, n=4, decode_workers=1)
    cache = targets._cache
    with cache._lock:
        cache._pinned = {0, 1}
    cache.close()
    assert cache.stats()["pinned"] == 0
    from concurrent.futures import Future
    future = Future()
    cache._run(3, future)                   # a decode the pool started after close()
    assert log == [] and isinstance(future.exception(), RuntimeError)


@pytest.mark.parametrize("path", ["start-training", "training-loop", "capture"])
def test_cli_run_closes_its_targets(path):
    """The decode pool and the pinned views are released when a CLI run ends, whatever ends it."""
    from trainer.frontend.app import TrainingApplication
    from trainer.frontend.profiling.instrumentation import CaptureFinished
    closed = []
    targets = SimpleNamespace(close=lambda: closed.append(1))

    def build():
        if path == "capture":
            raise CaptureFinished()
        raise RuntimeError("boom")
    app = SimpleNamespace(targets=targets, build_training_loop=build,
                          _training_runtime=SimpleNamespace(artifacts=SimpleNamespace(close_all=lambda: None)))
    app._close_targets = lambda: TrainingApplication._close_targets(app)
    app._train_and_write_artifacts = lambda hold: TrainingApplication._train_and_write_artifacts(app, hold)
    if path == "start-training":
        def prepare():
            raise RuntimeError("setup failed")
        app.prepare_training_context = prepare
        app._run_training_and_post = lambda: None
        with pytest.raises(RuntimeError, match="setup failed"):
            TrainingApplication.start_training(app)
    elif path == "training-loop":
        with pytest.raises(RuntimeError, match="boom"):
            TrainingApplication._run_training_and_post(app)
    else:
        TrainingApplication._run_training_and_post(app)
    assert closed == [1]


def test_hub_run_that_fails_to_build_closes_its_targets():
    """A hub run whose setup fails after the targets were loaded must not leave their decode pool running."""
    from trainer.frontend.hub import TrainingHub
    import trainer.frontend.hub as hub_module
    closed = []
    noop = lambda *_a, **_k: None                                    # noqa: E731
    state = SimpleNamespace(reset_for_new_run=noop, training_initializing=noop, begin_activity=noop, log=noop,
                            training_failed=noop, notify=noop)
    viewer = SimpleNamespace(state=state, clear_stop_request=noop, suspend_rendering=noop, resume_rendering=noop)

    def begin():
        raise RuntimeError("setup failed")
    hub = TrainingHub.__new__(TrainingHub)
    hub.app = SimpleNamespace(_viewer=viewer, viewer=viewer, _begin_repeat_run=begin,
                              _close_targets=lambda: closed.append(1))
    hub._hub_ensure_stbn = lambda _setup: True
    real = hub_module.report_exception
    hub_module.report_exception = noop
    try:
        hub._hub_train_scene("scene", {})
    finally:
        hub_module.report_exception = real
    assert closed == [1]


def test_target_prefetch_is_validated():
    from trainer.config.validation import ConfigError, validate_training_config
    from trainer.frontend.cli import parse_args
    for value in ("-1", "9"):
        args = parse_args(["--image-residency", "compressed", "--target-prefetch", value])
        with pytest.raises(ConfigError, match="target-prefetch"):
            validate_training_config(args, n_cams=8, n_train=8, n_test=1)
    validate_training_config(parse_args(["--target-prefetch", "8"]), n_cams=8, n_train=8, n_test=1)


def test_final_steps_prefetch_only_the_views_left_to_train():
    from trainer.frontend.app import TrainingLoop
    calls = []
    sampler = CameraSampler("ours", list(range(10)), [], 1, np.random.default_rng(1))

    class Views(dict):
        def prefetch(self, indices):
            calls.append(list(indices))

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.sampler = sampler
    loop.args = SimpleNamespace(target_prefetch=4, iters=10)
    loop.target_textures = Views({i: f"t{i}" for i in range(10)})
    loop.runtime = SimpleNamespace(pipeline=SimpleNamespace(targets=SimpleNamespace(select=lambda _t: None)),
                                   queue_limiter=SimpleNamespace(before_submit=lambda: None))
    for step, ahead in ((7, 3), (9, 1), (10, 0)):
        loop._prepare_iteration(SimpleNamespace(camera_index=sampler.next(), step_number=step))
        assert len(calls[-1]) == 1 + ahead, (step, calls[-1])


def test_empty_image_file_is_an_os_error(tmp_path):
    from trainer.data.colmap import decode_colmap_image, read_colmap_image_bytes
    scene = _scene(tmp_path, "png", n=1)
    (scene.image_dir / scene.images[0].image_name).write_bytes(b"")
    with pytest.raises(OSError, match="empty"):
        read_colmap_image_bytes(scene.images[0], scene.image_dir)
    with pytest.raises(OSError):
        decode_colmap_image(np.frombuffer(b"not an image", np.uint8), "x.png", W, H)


def test_lazy_column_takes_integer_like_indices(tmp_path):
    targets, _ = _counting(tmp_path, n=3)
    assert targets.resident[np.int64(1)].shape == (H, W, 4)
    with pytest.raises(TypeError):
        _ = targets.resident[1.0]


def test_corrupt_stored_mask_is_an_os_error(tmp_path):
    scene = _scene(tmp_path, "png", n=2)
    targets = TargetDataset(scene, W, H, use_alpha=False, residency="compressed", masks_dir=_masks(tmp_path, scene))
    data, _ = targets._encoded[0]
    targets._encoded[0] = (data, np.frombuffer(b"broken", np.uint8))
    with pytest.raises(OSError, match="mask"):
        targets.keep_mask(0)
    with pytest.raises(OSError, match="mask"):
        _ = targets.resident[0]
