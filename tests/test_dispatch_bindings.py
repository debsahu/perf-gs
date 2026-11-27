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
