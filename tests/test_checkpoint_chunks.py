# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Raster checkpoints split into chunks, each within one 4 GiB storage buffer.

A pipeline forced to small chunks must write and read exactly what the single buffer holds: chunk c is the slice
[c * chunk_buckets, (c + 1) * chunk_buckets) of the single buffer's buckets. The GPU tests force chunks of a few
buckets on a small frame, so every boundary case runs: a tile whose buckets straddle two chunks, and a per-bucket
backward workgroup whose two warps read different chunks."""

from __future__ import annotations

import numpy as np
import pytest
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import create_device, create_slang_session
from trainer.backend.compute.execution import (
    RADIX_PART_SIZE,
    RASTER_BUCKET_SIZE,
    TILE_AREA,
)
from trainer.backend.memory import buffers
from trainer.backend.memory.buffers import (
    CHECKPOINT_CHUNK_BUCKETS,
    MAX_CHECKPOINT_CHUNKS,
    SORT_PAIR_DEVICE_BYTES,
    STORAGE_BUFFER_RANGE,
    checkpoint_chunk_buckets,
    checkpoint_defines,
    max_raster_buckets,
    sort_pairs_ceiling,
)
from trainer.config.config import GeometryConfig, LossConfig
from trainer.config.validation import ConfigError, validate_training_config
from trainer.data.dataset import simple_camera
from trainer.frontend.cli import build_parser

W, H = 64, 48               # 8 x 6 = 48 tiles
TILES = 48
PAIRS = 2 * RADIX_PART_SIZE  # 4,096 pairs: 128 + 48 = 176 buckets allocated
CHUNK = 36                  # 176 buckets -> 5 chunks (36, 36, 36, 36, 32)
SORT_ARRAYS = 1_073_739_776  # (2^32 - 1) // 4, rounded down to RADIX_PART_SIZE


# ------------------------------------------------------------------------------------------- layout and ceiling

def test_chunk_layout_sums_to_the_single_buffer() -> None:
    sizes = checkpoint_chunk_buckets(PAIRS, TILES, CHUNK)
    assert sizes == [36, 36, 36, 36, 32]
    assert sum(sizes) == max_raster_buckets(PAIRS, TILES) == 176
    # One chunk whenever it fits: the single buffer, byte for byte the old allocation.
    assert checkpoint_chunk_buckets(PAIRS, TILES) == [176]
    # the old single-buffer ceiling at full resolution is still one chunk
    assert checkpoint_chunk_buckets(244_410_368, 750_750) == [8_388_574]
    assert checkpoint_defines(PAIRS, TILES) == {}
    assert checkpoint_defines(PAIRS, TILES, CHUNK) == {"CHECKPOINT_CHUNKS": "5", "CHECKPOINT_CHUNK_BUCKETS": "36u"}


def test_chunk_layout_at_full_resolution() -> None:
    """402,653,184 pairs at 5993 x 8005 (750,750 tiles): two chunks, each within 4 GiB, which one buffer was not."""
    sizes = checkpoint_chunk_buckets(402_653_184, 750_750)
    assert len(sizes) == 2 and sizes[0] == CHECKPOINT_CHUNK_BUCKETS
    assert all(n * TILE_AREA * 8 <= STORAGE_BUFFER_RANGE for n in sizes)
    assert sum(sizes) * TILE_AREA * 8 > STORAGE_BUFFER_RANGE      # one buffer would not fit


def test_chunk_layout_refuses_bad_chunks() -> None:
    with pytest.raises(ValueError, match="raster checkpoint chunks"):
        checkpoint_chunk_buckets(PAIRS, TILES, 35)                # 6 chunks
    for bad in (0, CHECKPOINT_CHUNK_BUCKETS + 1):
        with pytest.raises(ValueError, match="checkpoint chunk of"):
            checkpoint_chunk_buckets(PAIRS, TILES, bad)


def test_every_chunk_passes_the_per_buffer_guard_and_one_oversized_buffer_does_not(monkeypatch) -> None:
    """Allocation at 402M pairs: every checkpoint chunk goes through the 4 GiB guard and passes; a chunk size past
    the range would be refused by the same guard."""
    seen = []
    original = buffers.check_storage_range

    def record(label, size):
        seen.append((label, size))
        original(label, size)
    monkeypatch.setattr(buffers, "check_storage_range", record)
    counter = buffers._AllocationCounter(4, 4)
    buffers._allocate_checkpoints(counter, 402_653_184, 750_750, CHECKPOINT_CHUNK_BUCKETS)
    chunks = [(label, size) for label, size in seen if label.startswith("raster_checkpoints")]
    assert [label for label, _ in chunks] == ["raster_checkpoints[0]", "raster_checkpoints[1]"]
    assert all(size <= STORAGE_BUFFER_RANGE for _, size in chunks)
    assert sum(size for _, size in chunks) == max_raster_buckets(402_653_184, 750_750) * TILE_AREA * 8
    with pytest.raises(ValueError, match="4 GiB"):
        buffers.check_storage_range("raster_checkpoints[0]", (CHECKPOINT_CHUNK_BUCKETS + 1) * TILE_AREA * 8)


def test_ceiling_is_the_smallest_bound() -> None:
    # spec-minimum dispatch width: the old ceiling, unchanged
    assert sort_pairs_ceiling(65_535, 0) == (134_215_680, sort_pairs_ceiling(65_535, 0)[1])
    assert "dispatch width" in sort_pairs_ceiling(65_535, 0)[1]
    # a wide device at full resolution: the sort's key arrays, not the checkpoints (they split)
    pairs, bound = sort_pairs_ceiling(1 << 30, 750_750)
    assert pairs == SORT_ARRAYS and "key and value arrays" in bound
    assert SORT_ARRAYS * 4 <= STORAGE_BUFFER_RANGE < (SORT_ARRAYS + RADIX_PART_SIZE) * 4
    # a huge frame: five checkpoint chunks bind first, and the ceiling is tight against them
    pairs, bound = sort_pairs_ceiling(1 << 30, 9_000_000)
    assert "checkpoint chunks" in bound and pairs < SORT_ARRAYS
    assert len(checkpoint_chunk_buckets(pairs, 9_000_000)) == MAX_CHECKPOINT_CHUNKS
    with pytest.raises(ValueError):
        checkpoint_chunk_buckets(pairs + RADIX_PART_SIZE, 9_000_000)
    # a known device budget: 8 GiB at SORT_PAIR_DEVICE_BYTES per pair
    pairs, bound = sort_pairs_ceiling(1 << 30, 750_750, memory_bytes=8 * 2**30)
    assert pairs == int(8 * 2**30 // SORT_PAIR_DEVICE_BYTES) // RADIX_PART_SIZE * RADIX_PART_SIZE
    assert "device budget" in bound and pairs < SORT_ARRAYS
    # the 32-bit bucket index and the live-bucket list are bounds too, but never the tightest ones today
    assert all(p % RADIX_PART_SIZE == 0 for p in (pairs, SORT_ARRAYS))


def test_sort_pair_device_bytes_is_the_allocation_slope(gpu_device) -> None:
    """The memory bound's bytes per pair equal what the allocator actually adds per pair."""
    from trainer.backend.render.pipeline import TrainingPipeline
    module = TrainingPipeline(gpu_device, 16, 16, max_splats=16).shared_data_module
    sizes = {"max_splats": 16, "width": 16, "height": 16, "sh_rest_floats": 0}
    low, high = 64 * RADIX_PART_SIZE, 128 * RADIX_PART_SIZE
    lo = buffers.estimate_allocation_bytes(module, max_sort_pairs=low, **sizes)
    hi = buffers.estimate_allocation_bytes(module, max_sort_pairs=high, **sizes)
    assert SORT_PAIR_DEVICE_BYTES == 40.75
    assert hi - lo == SORT_PAIR_DEVICE_BYTES * (high - low)


def _args(pairs: int):
    return build_parser().parse_args(["--scene", "unused", "--no-viewer", "--max-sort-pairs", str(pairs)])


def test_refusal_names_the_bound() -> None:
    limit, bound = sort_pairs_ceiling(1 << 30, 750_750, memory_bytes=8 * 2**30)
    validate_training_config(_args(limit), n_cams=8, n_train=8, n_test=1,
                             max_sort_pairs_limit=limit, max_sort_pairs_bound=bound)
    with pytest.raises(ConfigError) as caught:
        validate_training_config(_args(limit + 1), n_cams=8, n_train=8, n_test=1,
                                 max_sort_pairs_limit=limit, max_sort_pairs_bound=bound)
    message = str(caught.value)
    assert f"{limit:,}" in message and f"{limit + 1:,}" in message
    assert "8.00 GiB device budget" in message


def test_app_passes_frame_tiles_and_device_budget(tmp_path, monkeypatch) -> None:
    """_prepare_scene must hand the scene's frame tiles and the --vram-budget-gib budget to the ceiling: 0.5 GiB
    left after the reserve holds about 13M pairs, so 20M is refused for the device budget."""
    from types import SimpleNamespace

    from _colmap_testkit import write_colmap_scene

    from trainer.backend.memory.buffers import frame_tiles
    from trainer.frontend.app import TrainingApplication
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("trainer.frontend.app.dispatch_groups_x", lambda _device: 1 << 30)
    monkeypatch.setattr("trainer.backend.memory.budget.query_video_memory", lambda _name: None)
    calls = []
    real = buffers.sort_pairs_ceiling

    def spy_ceiling(groups, tiles, memory):
        calls.append((tiles, memory))
        return real(groups, tiles, memory)
    monkeypatch.setattr("trainer.frontend.app.sort_pairs_ceiling", spy_ceiling)
    scene = write_colmap_scene(tmp_path / "scene", views=[(f"{i}.png", 128, 128, 100.0) for i in range(2)])
    args = build_parser().parse_args(["--scene", str(scene), "--no-viewer", "--max-sort-pairs", "20000000",
                                      "--width", "128", "--height", "128", "--no-write-ground-truths",
                                      "--no-stereo-init", "--vram-budget-gib", "1"])
    app = TrainingApplication(args)
    app.device = SimpleNamespace(info=SimpleNamespace(adapter_name="test"))
    with pytest.raises(ConfigError, match="device budget"):
        app._prepare_scene(scene)
    assert calls == [(frame_tiles(128, 128), 2**30 - 2**29)]


# ------------------------------------------------------------------------------------------- GPU equivalence

@pytest.fixture(scope="module")
def device():
    try:
        dev = create_device(backend="vulkan")
    except RuntimeError as exc:
        pytest.skip(f"vulkan unavailable: {exc}")
    yield dev
    dev.close()


def _scene(n: int = 160, seed: int = 5):
    rng = np.random.default_rng(seed)
    means = np.column_stack([rng.uniform(-0.5, 0.5, (n, 2)), rng.uniform(2.0, 4.0, n)]).astype(np.float32)
    return {"means": means,
            "log_scales": np.log(rng.uniform(0.15, 0.4, (n, 3))).astype(np.float32),
            "rot_quats": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            "opacities": rng.uniform(0.2, 0.6, n).astype(np.float32),
            "color_logits": rng.normal(size=(n, 3)).astype(np.float32)}


def _target() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([xx / W, yy / H, 0.5 * np.ones_like(xx)], axis=-1)


def _camera():
    return simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                         fovy_degrees=60.0)


def _pipeline(device, chunk: int, defines: dict[str, str], per_bucket: bool) -> TestPipeline:
    session = create_slang_session(device, defines) if defines else None
    geometry = GeometryConfig(depth_normal_weight=0.05) if defines.get("GEOM_PRIORS") else None
    p = TestPipeline(device, W, H, max_splats=256, max_sort_pairs=PAIRS, slang_session=session,
                     geometry_config=geometry, loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
                     use_flat_bucket_backward=per_bucket, checkpoint_chunk_buckets=chunk)
    p.targets.set_image(_target())
    p.set_primitive_from_numpy(**_scene())
    return p


def _run(p: TestPipeline):
    p.forward_and_loss(_camera())
    with p.commands.recording("backward"):
        p.run_backward()
    p.runtime.wait_for_idle()
    r = p.buffers.raster
    used = int(r.bucket_offset_total.to_numpy().reshape(-1)[0])
    checkpoints = np.concatenate([c.to_numpy().reshape(-1, 2) for c in r.raster_checkpoints])[:used * TILE_AREA]
    return {"image": r.output_image.to_numpy().copy(), "checkpoints": checkpoints, "used": used,
            "grads": {k: v.copy() for k, v in p.read_gradients().items()},
            "offsets": r.bucket_offsets.to_numpy().reshape(-1).copy(),
            "counts": r.bucket_count.to_numpy().reshape(-1).copy(),
            "live": p.buffers.backward.live_buckets.to_numpy().reshape(-1, 2).copy(),
            "live_total": int(p.buffers.backward.live_bucket_total.to_numpy().reshape(-1)[0]),
            "chunks": len(r.raster_checkpoints),
            "sort_overflows": int(p.buffers.sort.stats.to_numpy().reshape(-1)[1])}


DEFINES = {"plain": {}, "geometry": {"GEOM_PRIORS": "1"}, "loss-mask": {"LOSS_MASK": "1"}}


def _straddles(run: dict, chunk: int, per_bucket: bool) -> bool:
    """Whether ``chunk`` separates the layouts on this frame: buckets used in at least three chunks, a tile whose
    buckets straddle a boundary, and (per-bucket backward) a workgroup whose two warps read different chunks."""
    offsets, counts = run["offsets"], run["counts"]
    if run["used"] <= 2 * chunk or len(checkpoint_chunk_buckets(PAIRS, TILES, chunk)) > MAX_CHECKPOINT_CHUNKS:
        return False
    if not any(counts[t] > 1 and offsets[t] // chunk != (offsets[t] + counts[t] - 1) // chunk for t in range(TILES)):
        return False
    if per_bucket:
        live = run["live"][:run["live_total"]]
        chunk_of = (offsets[live[:, 0]] + live[:, 1]) // chunk
        warps = chunk_of[: len(chunk_of) // 2 * 2].reshape(-1, 2)        # NUM_WARPS = 2 live slots per group
        if not np.any(warps[:, 0] != warps[:, 1]):
            return False
    return True


@pytest.mark.parametrize("per_bucket", [True, False], ids=["per-bucket", "per-splat"])
@pytest.mark.parametrize("variant", list(DEFINES))
def test_chunked_checkpoints_match_the_single_buffer(device, variant: str, per_bucket: bool) -> None:
    single = _run(_pipeline(device, CHECKPOINT_CHUNK_BUCKETS, DEFINES[variant], per_bucket))
    assert single["chunks"] == 1 and single["sort_overflows"] == 0
    # The chunk size comes from this frame's own layout, so the fixture always separates the two layouts.
    chunk = next((c for c in range(CHUNK, 64) if _straddles(single, c, per_bucket)), None)
    assert chunk is not None, "no chunk size separates the layouts on this frame"
    chunked = _run(_pipeline(device, chunk, DEFINES[variant], per_bucket))
    assert chunked["chunks"] == len(checkpoint_chunk_buckets(PAIRS, TILES, chunk)) >= 3
    assert chunked["used"] == single["used"] and chunked["sort_overflows"] == 0

    assert np.any(single["checkpoints"][2 * chunk * TILE_AREA:] != 0)     # data lives past the second boundary
    assert np.array_equal(chunked["checkpoints"], single["checkpoints"])
    assert np.array_equal(chunked["image"], single["image"])
    assert any(np.abs(g).max() > 0 for g in single["grads"].values())
    for name, grad in single["grads"].items():
        assert np.array_equal(chunked["grads"][name], grad), name


def test_viewer_keeps_its_one_element_stub_with_a_forced_chunk(device) -> None:
    """The viewer never splits: its checkpoints are a stub whatever the chunk size asked for."""
    p = TestPipeline(device, W, H, max_splats=256, max_sort_pairs=PAIRS, viewer_only=True,
                     checkpoint_chunk_buckets=CHUNK)
    assert [c.storage.size for c in p.buffers.raster.raster_checkpoints] == [8]
    p.set_gaussians(**_scene())
    p.render(_camera())
    assert p.read_output_image()[..., 3].max() > 0.2


def test_buckets_per_pair_constant_matches_the_shader_bucket() -> None:
    assert RASTER_BUCKET_SIZE == 32 and TILE_AREA == 64
    assert CHECKPOINT_CHUNK_BUCKETS == STORAGE_BUFFER_RANGE // (TILE_AREA * 8) == 8_388_607
