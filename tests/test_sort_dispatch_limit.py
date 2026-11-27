# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

# --max-sort-pairs is bounded by the DEVICE's dispatch limit, not by the Vulkan-spec minimum of 65,535 groups.
# Fixtures come from conftest.py.
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from trainer.backend.compute.execution import RADIX_PART_SIZE
from trainer.backend.compute.device import dispatch_groups_x
from trainer.backend.memory.buffers import max_sort_pairs_limit
from trainer.config.validation import SPEC_MAX_SORT_PAIRS, ConfigError, validate_training_config
from trainer.frontend.cli import build_parser

OLD_CEILING = 65_535 * 2_048          # 134,215,680


def _args(pairs: int):
    return build_parser().parse_args(["--scene", "unused", "--no-viewer", "--max-sort-pairs", str(pairs)])


def _validate(pairs: int, groups: int | None = None) -> None:
    kw = {} if groups is None else {"max_sort_pairs_limit": max_sort_pairs_limit(groups)}
    validate_training_config(_args(pairs), n_cams=8, n_train=8, n_test=1, **kw)


def test_config_default_ceiling_is_the_backend_spec_ceiling() -> None:
    """config/ imports no backend module, so its default ceiling is a literal: pin it to the backend's value."""
    assert SPEC_MAX_SORT_PAIRS == max_sort_pairs_limit() == OLD_CEILING


def test_default_ceiling_is_unchanged() -> None:
    _validate(OLD_CEILING)
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        _validate(OLD_CEILING + 1)


def test_device_limit_raises_the_ceiling() -> None:
    _validate(200_000_000, groups=1 << 30)                     # above the old ceiling, inside the checkpoint range
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        _validate(200_000_000, groups=65_535)


def test_ceiling_also_bounded_by_a_4gib_storage_buffer() -> None:
    # each of the six pair buffers holds capacity * 4 bytes and must stay addressable by a uint32 byte offset
    # (raster_checkpoints, below, is the tighter bound, so this one is not tight)
    cap = max_sort_pairs_limit(1 << 31)
    assert cap % RADIX_PART_SIZE == 0 and cap * 4 <= 2**32 - 1
    _validate(cap, groups=1 << 31)
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        _validate(cap + 1, groups=1 << 31)


def test_dispatch_groups_x_reads_the_device_and_falls_back() -> None:
    dev = SimpleNamespace(info=SimpleNamespace(limits=SimpleNamespace(max_compute_dispatch_thread_groups=(7, 1, 1))))
    assert dispatch_groups_x(dev) == 7
    assert dispatch_groups_x(None) == 65_535


def test_app_validates_against_its_device(gpu_device, tmp_path, monkeypatch) -> None:
    """_prepare_scene must pass the real device's X limit: a capacity above the old ceiling loads on a device that
    allows it and is refused when the same device reports the spec minimum."""
    from _colmap_testkit import write_colmap_scene
    from trainer.frontend.app import TrainingApplication
    if dispatch_groups_x(gpu_device) < 2 * 65_535:
        pytest.skip("device reports the spec-minimum dispatch width")
    monkeypatch.chdir(tmp_path)
    scene = write_colmap_scene(tmp_path / "scene", views=[(f"{i}.png", 16, 16, 20.0) for i in range(4)])
    pairs = 2 * OLD_CEILING
    args = build_parser().parse_args(["--scene", str(scene), "--no-viewer", "--max-sort-pairs", str(pairs),
                                      "--width", "16", "--height", "16", "--no-write-ground-truths"])
    app = TrainingApplication(args)
    app.device = gpu_device
    app._prepare_scene(scene)                                    # accepted
    monkeypatch.setattr("trainer.frontend.app.dispatch_groups_x", lambda _device: 65_535)
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        app._prepare_scene(scene)


@pytest.mark.skipif(os.environ.get("PERFGS_BIG_SORT") != "1", reason="allocates ~4 GB; set PERFGS_BIG_SORT=1")
def test_radix_sort_beyond_65535_groups(radix_kernels) -> None:
    """The sort itself is correct past 65,535 groups along X (the old ceiling), on a device that allows it."""
    from test_radix_sort import _radix_sort
    device, kernels = radix_kernels
    if dispatch_groups_x(device) <= 65_535 + 8:
        pytest.skip("device dispatch width does not exceed 65,535 groups")
    n = OLD_CEILING + 5 * RADIX_PART_SIZE + 17
    rng = np.random.default_rng(65_535)
    keys = rng.integers(0, 2**32, size=n, dtype=np.uint32)
    values = np.arange(n, dtype=np.uint32)                       # identity values: checks the permutation too
    got_lo, _, got_v = _radix_sort(device, kernels, keys, np.zeros(n, np.uint32), values, passes=4, key_words=1)
    order = np.argsort(keys, kind="stable")
    assert np.array_equal(got_v, order.astype(np.uint32))
    assert np.array_equal(got_lo, keys[order])
    assert n > OLD_CEILING and int(got_v[-1]) != 0


from test_radix_sort import radix_kernels  # noqa: E402,F401  (module fixture reused)


# raster_checkpoints is the largest buffer that grows with --max-sort-pairs: one uint2 per tile pixel per bucket,
# (ceil(pairs / RASTER_BUCKET_SIZE) + num_tiles) * TILE_AREA * 8 bytes, about 16 bytes per pair. It must stay inside a
# 4 GiB - 1 storage-buffer range too, which bounds the pairs ~4x tighter than the 4-byte pair arrays do.
RANGE = 2**32 - 1


def _checkpoint_bytes(pairs: int, num_tiles: int) -> int:
    from trainer.backend.compute.execution import RASTER_BUCKET_SIZE, TILE_AREA
    return (-(-pairs // RASTER_BUCKET_SIZE) + num_tiles) * TILE_AREA * 8


def test_ceiling_keeps_raster_checkpoints_inside_a_4gib_buffer() -> None:
    cap = max_sort_pairs_limit(1 << 31)
    assert 0 < _checkpoint_bytes(cap, 0) <= RANGE, cap
    # tight: one more radix part would not fit, so the bound is the checkpoint buffer and not something smaller
    assert _checkpoint_bytes(cap + RADIX_PART_SIZE, 0) > RANGE
    assert cap > OLD_CEILING                                    # the device limit still lifts the old ceiling


@pytest.mark.parametrize("num_tiles", [1, 4_096, 750_000])
def test_ceiling_counts_the_frame_tiles(num_tiles: int) -> None:
    cap = max_sort_pairs_limit(1 << 31, num_tiles=num_tiles)
    assert _checkpoint_bytes(cap, num_tiles) <= RANGE < _checkpoint_bytes(cap + RADIX_PART_SIZE, num_tiles)
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        validate_training_config(_args(cap + 1), n_cams=8, n_train=8, n_test=1, max_sort_pairs_limit=cap)


def test_allocation_refuses_checkpoints_past_4gib(gpu_device) -> None:
    """The allocator itself refuses an oversized checkpoint buffer, for callers that skip validation (viewer, API)."""
    from trainer.backend.memory.buffers import estimate_allocation_bytes
    from trainer.backend.render.pipeline import TrainingPipeline
    module = TrainingPipeline(gpu_device, 16, 16, max_splats=16).shared_data_module
    sizes = dict(max_splats=16, width=16, height=16, sh_rest_floats=0)
    fits = max_sort_pairs_limit(1 << 31, num_tiles=4)
    assert estimate_allocation_bytes(module, max_sort_pairs=fits, **sizes) > _checkpoint_bytes(fits, 4)
    with pytest.raises(ValueError, match="raster_checkpoints"):
        estimate_allocation_bytes(module, max_sort_pairs=402_653_184, **sizes)
    # the viewer binds a one-element checkpoint stub, so its sort capacity is bounded by the pair arrays only
    estimate_allocation_bytes(module, viewer_only=True, max_sort_pairs=402_653_184, **sizes)


def test_default_sort_pairs_stay_inside_the_checkpoint_range() -> None:
    from trainer.backend.render.pipeline import default_sort_pairs
    pairs = default_sort_pairs(200_000_000, num_tiles=10_000)    # 6 per splat would be 1.2 G pairs
    assert _checkpoint_bytes(pairs, 10_000) <= RANGE


@pytest.mark.parametrize("groups", [(0, 1, 1), (-5, 1, 1), ("wide", 1, 1), (None, 1, 1)],
                         ids=["zero", "negative", "not-a-number", "none"])
def test_dispatch_groups_x_falls_back_on_a_bad_device_value(groups) -> None:
    """A device that reports no usable width must not set the sort ceiling to 0 or crash."""
    dev = SimpleNamespace(info=SimpleNamespace(limits=SimpleNamespace(max_compute_dispatch_thread_groups=groups)))
    assert dispatch_groups_x(dev) == 65_535


def test_every_storage_buffer_is_refused_past_4gib(gpu_device) -> None:
    """Every storage buffer, not only raster_checkpoints, must fit one 4 GiB - 1 range: sh.adam_rest at SH degree 3
    with fp32 moments passes it at ~11.9M splats, and an index past the range wraps silently on NVIDIA."""
    from trainer.backend.memory.buffers import estimate_allocation_bytes
    from trainer.backend.render.pipeline import TrainingPipeline
    module = TrainingPipeline(gpu_device, 16, 16, max_splats=16).shared_data_module
    sizes = dict(width=16, height=16, max_sort_pairs=1024, sh_rest_floats=45)
    estimate_allocation_bytes(module, max_splats=11_000_000, **sizes)
    with pytest.raises(ValueError, match="adam_rest"):
        estimate_allocation_bytes(module, max_splats=12_000_000, **sizes)


def test_the_device_allocator_refuses_a_buffer_past_4gib(monkeypatch) -> None:
    """The guard sits in the allocator, so no buffer reaches the device oversized (none is allocated here)."""
    from trainer.backend.memory import buffers
    made = []
    monkeypatch.setattr(buffers, "_make_labeled_ndbuffer", lambda *a, **k: made.append(a[2]))
    alloc = buffers._Alloc(None, 1, 1)
    alloc.typed_buffer("ok", "uint", (2**32 - 1) // 4)
    with pytest.raises(ValueError, match="too_big"):
        alloc.typed_buffer("too_big", "uint", 2**30)
    with pytest.raises(ValueError, match="too_big_i64"):
        alloc.i64_buffer("too_big_i64", 2**29)
    assert made == [((2**32 - 1) // 4,)]


def test_app_ceiling_counts_the_scene_frame_tiles(tmp_path, monkeypatch) -> None:
    """At 128 x 128 (256 tiles) the ceiling is 8,192 pairs under the tile-less one: a capacity between the two must be
    refused, which it is only if _prepare_scene passes the frame's tiles."""
    from _colmap_testkit import write_colmap_scene
    from trainer.backend.memory.buffers import frame_tiles
    from trainer.frontend.app import TrainingApplication
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("trainer.frontend.app.dispatch_groups_x", lambda _device: 1 << 30)
    scene = write_colmap_scene(tmp_path / "scene", views=[(f"{i}.png", 128, 128, 100.0) for i in range(2)])
    tileless = max_sort_pairs_limit(1 << 30, num_tiles=0)
    framed = max_sort_pairs_limit(1 << 30, num_tiles=frame_tiles(128, 128))
    assert framed < tileless
    args = build_parser().parse_args(["--scene", str(scene), "--no-viewer", "--max-sort-pairs", str(tileless),
                                      "--width", "128", "--height", "128", "--no-write-ground-truths",
                                      "--no-stereo-init"])
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        TrainingApplication(args)._prepare_scene(scene)
