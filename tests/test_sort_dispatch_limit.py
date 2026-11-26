# compute/perfgs sort_dispatch_limit.patch: --max-sort-pairs is bounded by the DEVICE's dispatch limit, not by the
# Vulkan-spec minimum of 65,535 groups. Runs in perf-gs's own harness (fixtures from conftest.py).
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from trainer.backend.compute.execution import RADIX_PART_SIZE
from trainer.config.validation import ConfigError, dispatch_groups_x, max_sort_pairs_limit, validate_training_config
from trainer.frontend.cli import build_parser

OLD_CEILING = 65_535 * 2_048          # 134,215,680


def _args(pairs: int):
    return build_parser().parse_args(["--scene", "unused", "--no-viewer", "--max-sort-pairs", str(pairs)])


def _validate(pairs: int, groups: int | None = None) -> None:
    kw = {} if groups is None else {"max_dispatch_groups": groups}
    validate_training_config(_args(pairs), n_cams=8, n_train=8, n_test=1, **kw)


def test_default_ceiling_is_unchanged() -> None:
    _validate(OLD_CEILING)
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        _validate(OLD_CEILING + 1)


def test_device_limit_raises_the_ceiling() -> None:
    _validate(402_653_184, groups=1 << 30)                     # a full-res run's value
    with pytest.raises(ConfigError, match="max-sort-pairs"):
        _validate(402_653_184, groups=65_535)


def test_ceiling_also_bounded_by_a_4gib_storage_buffer() -> None:
    # each of the six pair buffers holds capacity * 4 bytes and must stay addressable by a uint32 byte offset
    cap = max_sort_pairs_limit(1 << 31)
    assert cap % RADIX_PART_SIZE == 0 and cap * 4 <= 2**32 - 1 < (cap + RADIX_PART_SIZE) * 4
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
