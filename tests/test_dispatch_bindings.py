# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Every recorded dispatch must bind exactly the entry point's parameters.

Slang leaves an unbound entry-point parameter at zero without an error, so a forgotten buffer
argument silently reads a zero buffer. CommandBatch now checks each kernel's keyword arguments
against its reflected parameters (system values such as SV_DispatchThreadID excluded) and raises
on a missing or an unknown one. Vulkan tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainer.backend.compute.gpu import BindingError, entry_point_bindings
from test_structure_densify import _count, _setup
from test_deterministic_densify import _densify_structure, _population


def test_system_values_are_not_required(gpu_device) -> None:
    p = _setup(gpu_device, max_splats=64)
    lib = p.structure_densifier._kernel_library
    required, _ = entry_point_bindings(lib.kernel("densify/split_anisotropic.slang", "splitAnisotropic"))
    assert "group_id" not in required and "gtid" not in required
    assert {"gaussians", "child_offset", "selected_parents", "max_splats"} <= required
    required, _ = entry_point_bindings(lib.kernel("common/scan.slang", "scanBlock"))
    assert required == {"in_buf", "out_buf", "block_sums", "n"}       # dtid, gid and lid are system values
    required, _ = entry_point_bindings(lib.kernel("mip/compute_3d_filter.slang", "clearMipFilterTail"))
    assert "thread_id" not in required                                  # an SV_GroupThreadID named like a buffer


@pytest.mark.parametrize("indirect", [False, True], ids=["dispatch", "indirect"])
@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_a_wrong_binding_raises(gpu_device, monkeypatch, indirect, fault) -> None:
    """Drop (or add) one argument of a real densify dispatch: the recording must refuse it."""
    p = _setup(gpu_device, max_splats=600)
    _population(p, 200, seed=1)
    commands = p.structure_densifier._commands
    name = "split_anisotropic" if indirect else "list_split_parents"
    method = "dispatch_indirect" if indirect else "dispatch"
    original = getattr(commands, method)

    def broken(kernel, marker, *args, **kwargs):
        if marker == name:
            if fault == "missing":
                kwargs.pop("selected_parents")
            else:
                kwargs["not_a_parameter"] = 1
        return original(kernel, marker, *args, **kwargs)
    monkeypatch.setattr(commands, method, broken)
    with pytest.raises(BindingError, match="selected_parents" if fault == "missing" else "not_a_parameter"):
        _densify_structure(p)


def test_the_densify_path_binds_cleanly(gpu_device) -> None:
    """The unmodified recording passes the check (it runs on every dispatch of the whole suite too)."""
    p = _setup(gpu_device, max_splats=600)
    _population(p, 200, seed=1)
    _densify_structure(p, rank_split_families=True)
    assert _count(p) > 200
    assert np.isfinite(p.buffers.primitive.values.to_numpy().view(np.float32)).all()


# ---------------------------------------------------------------- review regressions (no GPU)

class _FakeKernel:
    """A kernel-shaped object: the reflection check_bindings reads, nothing else."""

    collected = 0

    def __init__(self, params):
        import slangpy as spy
        from types import SimpleNamespace as NS
        scalar = spy.TypeReflection.Kind.scalar
        struct = spy.TypeReflection.Kind.resource
        parameters = [NS(name=name, type_layout=NS(kind=scalar if system else struct, size=0 if system else 8))
                      for name, system in params]
        self.program = NS(layout=NS(entry_points=[NS(name="fake", parameters=parameters)]))

    def __del__(self):
        type(self).collected += 1


def test_binding_cache_does_not_pin_every_kernel():
    """The cache held a strong reference to every kernel (and so to its program and device) forever."""
    import gc
    from trainer.backend.compute import gpu
    gpu._BINDINGS.clear()
    _FakeKernel.collected = 0
    bound = getattr(gpu, "BINDING_CACHE_SIZE", 1024)
    n = bound + 300
    for _ in range(n):
        entry_point_bindings(_FakeKernel([("buf", False)]))
    gc.collect()
    assert _FakeKernel.collected >= n - bound, (_FakeKernel.collected, n)
    assert len(gpu._BINDINGS) <= bound


def test_hot_reload_refreshes_the_reflected_parameters(monkeypatch):
    """After reload_all_programs the same kernel object can take different parameters: the pipeline's hot-reload
    hook must drop the cached ones."""
    from types import SimpleNamespace
    from trainer.backend.compute import gpu
    from trainer.backend.render import pipeline as pipeline_module
    kernel = _FakeKernel([("old", False)])
    assert entry_point_bindings(kernel)[0] == {"old"}
    kernel.program.layout.entry_points[0].parameters[0].name = "new"
    monkeypatch.setattr(pipeline_module, "_module_for_reflection", lambda *_a: None)
    pipeline_module.TrainingPipeline.on_hot_reload(SimpleNamespace(device=None, slang_session=None))
    assert entry_point_bindings(kernel)[0] == {"new"}
    gpu.check_bindings(kernel, "m", {"new": 1})


def test_a_system_value_or_an_indirect_thread_count_is_not_a_binding():
    from trainer.backend.compute import gpu
    kernel = _FakeKernel([("dtid", True), ("buf", False)])
    gpu.check_bindings(kernel, "m", {"buf": 1, "thread_count": [1, 1, 1]})
    with pytest.raises(BindingError, match="dtid"):
        gpu.check_bindings(kernel, "m", {"buf": 1, "dtid": 0})
    with pytest.raises(BindingError, match="thread_count"):
        gpu.check_bindings(kernel, "m", {"buf": 1, "thread_count": [1, 1, 1]}, indirect=True)


def test_every_dispatch_binds_with_masks_priors_and_stbn_compiled_in() -> None:
    """LOSS_MASK, GEOM_PRIORS and STBN_ENABLED together change the arguments of the loss, forward and both blend
    backwards: a full training step (dense STBN, every geometry term on) must bind every dispatch exactly."""
    from trainer.backend.compute.device import create_device
    from trainer.config.config import GeometryConfig, LossConfig, StbnConfig
    from _pipeline_testkit import TestPipeline
    from trainer.data.dataset import simple_camera
    device = create_device(backend="vulkan", defines={"LOSS_MASK": "1", "GEOM_PRIORS": "1", "STBN_ENABLED": "1"})
    try:
        for per_bucket in (False, True):
            p = TestPipeline(device=device, width=16, height=16, max_splats=64,
                             loss_config=LossConfig(ssim_weight=0.2),
                             geometry_config=GeometryConfig(depth_weight=1.0, normal_weight=0.2,
                                                            depth_normal_weight=0.05, flatten_weight=1.0),
                             stbn_config=StbnConfig(backward_compacted=True), use_flat_bucket_backward=per_bucket)
            assert p.loss_mask and p.geom_priors
            rng = np.random.default_rng(1)
            n = 16
            p.set_primitive_from_numpy(
                means=np.column_stack([rng.uniform(-0.5, 0.5, (n, 2)), np.full(n, 3.0)]).astype(np.float32),
                log_scales=np.log(np.full((n, 3), 0.2, np.float32)),
                rot_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
                opacities=np.full(n, 0.7, np.float32), color_logits=np.zeros((n, 3), np.float32))
            rgba = np.ones((16, 16, 4), np.float32)
            rgba[:, :8, 3] = 0.0
            p.targets.set_image(rgba)
            p.geometry.set_prior_grids((16, 16), (16, 16))
            normal = np.zeros((16, 16, 4), np.uint8)
            normal[..., :2], normal[..., 3] = 128, 255
            p.geometry.select(np.full((16, 16), 3000, np.uint16), normal)
            cam = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0))
            p.train_step(cam)                                # every dispatch passes the binding guard
            p.runtime.wait_for_idle()
            assert p.geometry.read_stats()["V_d"] > 0
    finally:
        device.close()
