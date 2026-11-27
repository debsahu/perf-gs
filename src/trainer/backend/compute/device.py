# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Device creation: a SlangPy device (Vulkan primary, D3D12 cross-check) with the
packaged or source ``shaders/`` directory on the Slang include path."""

from __future__ import annotations

from pathlib import Path
from importlib.resources import files
from typing import Literal

import slangpy as spy

Backend = Literal["vulkan", "d3d12", "metal"]

_REPO_ROOT = Path(__file__).resolve().parents[4]
SHADERS_DIR = Path(str(files("trainer").joinpath("shaders")))
if not SHADERS_DIR.is_dir():
    SHADERS_DIR = _REPO_ROOT / "shaders"  # editable/source checkout

_BACKEND_MAP: dict[Backend, spy.DeviceType] = {
    "vulkan": spy.DeviceType.vulkan,
    "d3d12": spy.DeviceType.d3d12,
    "metal": spy.DeviceType.metal,
}


SPEC_MIN_DISPATCH_GROUPS = 65_535


def dispatch_groups_x(device) -> int:
    """The device's maximum compute dispatch width along X; the Vulkan-spec minimum when unknown or not positive."""
    try:
        groups = int(device.info.limits.max_compute_dispatch_thread_groups[0])
    except (AttributeError, TypeError, IndexError, ValueError):
        return SPEC_MIN_DISPATCH_GROUPS
    return groups if groups > 0 else SPEC_MIN_DISPATCH_GROUPS


def define_enabled(defines: dict[str, str] | None, name: str) -> bool:
    """Whether Slang's ``#if name`` is true under ``defines``: absent or "0" is off, any other integer is on.
    ``bool(value)`` would read the string "0" as on."""
    value = (defines or {}).get(name)
    if value is None:
        return False
    try:
        return int(str(value).strip(), 0) != 0
    except ValueError as exc:
        raise ValueError(f"shader define {name}={value!r} is not an integer") from exc


def create_device(
    backend: Backend = "vulkan",
    enable_debug: bool = False,
    shader_debug_info: bool = False,
    defines: dict[str, str] | None = None,
) -> spy.Device:
    """Create a SlangPy device with ``shaders/`` on the include path.

    ``shader_debug_info`` adds source-line mapping for Nsight without lowering optimization;
    ``defines`` are preprocessor macros injected into every shader compile.
    """
    if backend not in _BACKEND_MAP:
        raise ValueError(f"unknown backend {backend!r}; expected one of {list(_BACKEND_MAP)}")

    if not shader_debug_info and not defines:
        return spy.create_device(
            type=_BACKEND_MAP[backend],
            enable_debug_layers=enable_debug,
            include_paths=[SHADERS_DIR],
        )

    # spy.create_device() hides compiler_options; replicate its include paths and hot reload.
    compiler_options: dict = {
        "include_paths": [spy.SHADER_PATH, str(SHADERS_DIR)],
    }
    if shader_debug_info:
        compiler_options["debug_info"] = spy.SlangDebugInfoLevel.standard
    if defines:
        compiler_options["defines"] = dict(defines)
    return spy.Device(
        type=_BACKEND_MAP[backend],
        enable_debug_layers=enable_debug,
        enable_hot_reload=True,
        compiler_options=compiler_options,
    )


def create_slang_session(device: spy.Device, defines: dict[str, str] | None) -> spy.SlangSession:
    """A session compiling like the device's own, but with ``defines`` as its macros."""
    # Copied field by field: desc.compiler_options is the device's live object, not a copy.
    own = device.slang_session.desc.compiler_options
    return device.create_slang_session(compiler_options=spy.SlangCompilerOptions({
        "include_paths": own.include_paths,
        "debug_info": own.debug_info,
        "defines": dict(defines or {}),
    }))
