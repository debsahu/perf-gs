# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The STBN table on the GPU and the per-step mask parameters the kernels take."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

from trainer.backend.compute.execution import TILE_AREA, TILE_HEIGHT, TILE_WIDTH
from trainer.backend.render import stbn_table
from trainer.backend.render.stbn_table import StbnSpec
from trainer.config.config import StbnConfig, StbnVariables

_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access


@dataclass(frozen=True)
class StbnMask:
    """The uploaded table; only the slice and per-tile budget change per step."""

    table: spy.Buffer
    dims: tuple[int, int]       # (width, height) in pixels
    slices: int

    @classmethod
    def from_table(cls, device: spy.Device, packed: np.ndarray, spec: StbnSpec,
                   label: str = "stbn_table") -> StbnMask:
        if packed.shape != (spec.voxels,) or packed.dtype != np.uint32:
            raise ValueError(
                f"packed table must be uint32[{spec.voxels}], got "
                f"{packed.dtype}{packed.shape}")
        from trainer.backend.memory.buffers import check_storage_range
        check_storage_range(label, packed.nbytes)
        buffer = device.create_buffer(size=packed.nbytes, usage=_USAGE, label=label)
        buffer.copy_from_numpy(np.ascontiguousarray(packed).view(np.uint8))
        return cls(table=buffer, dims=(spec.width, spec.height), slices=spec.slices)

    @classmethod
    def build(cls, device: spy.Device, spec: StbnSpec,
              cache_dir: Path | str | None = None) -> StbnMask:
        return cls.from_table(device, stbn_table.load_or_build(spec, cache_dir), spec)

    @classmethod
    def disabled(cls, device: spy.Device) -> StbnMask:
        """A one-word placeholder: the kernels never read it, but must bind something."""
        buffer = device.create_buffer(size=4, usage=_USAGE, label="stbn_table_disabled")
        buffer.copy_from_numpy(np.zeros(1, dtype=np.uint32).view(np.uint8))
        return cls(table=buffer, dims=(TILE_WIDTH, TILE_HEIGHT), slices=1)

    @classmethod
    def for_config(cls, device: spy.Device, config: StbnConfig) -> StbnMask:
        if not config.enabled:
            return cls.disabled(device)
        if config.select_mode == stbn_table.SELECT_GLOBAL and int(config.warmup_iters) > 0:
            raise ValueError(
                "--stbn-warmup-iters > 0 is incompatible with "
                "--stbn-select-mode global: a global threshold fixes the "
                "selected set when the table is built, so the density cannot "
                "ramp at runtime and the 1/rho weight would no longer match the "
                "mask. Use --stbn-warmup-iters 0, or "
                "--stbn-select-mode tile_rank if you need the ramp.")
        spec = StbnSpec(source=config.source, path=config.file_path, seed=config.seed,
                        select_mode=config.select_mode,
                        fraction=float(config.pixel_fraction))
        mask = cls.build(device, spec, stbn_table.default_cache_dir() / "cache")
        print(f"[stbn] {spec.source} mask {spec.width}x{spec.height}x{spec.slices}"
              f", {stbn_table.pixels_per_tile(config.pixel_fraction)}"
              f"/{TILE_AREA} pixels per tile at rho={config.pixel_fraction:g}"
              f"{' (compacted backward)' if config.backward_compacted else ''}")
        return mask

    def shader_kwargs(self, slice_index: int, pixels_per_tile: int) -> dict:
        """The four ``stbn_*`` parameters every masked kernel takes."""
        return {
            "stbn_table": self.table,
            "stbn_dims": spy.uint2(self.dims[0], self.dims[1]),
            "stbn_slice": int(slice_index) % max(self.slices, 1),
            "stbn_tile_pixels": int(pixels_per_tile),
        }

    def step_kwargs(self, variables: StbnVariables | None) -> dict:
        """This step's parameters; no variables means every pixel (the dense path)."""
        if variables is None or variables.pixels_per_tile <= 0:
            return self.shader_kwargs(0, TILE_AREA)
        return self.shader_kwargs(variables.slice_index,
                                  min(int(variables.pixels_per_tile), TILE_AREA))
