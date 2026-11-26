# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Per-view structure-tensor field (Sxx, Sxy, Syy) for SAD-GS densification: a finest-scale
Sobel tensor re-integrated per DoG band, weighted by band energy^gamma * frequency^2."""

from __future__ import annotations

import math

import numpy as np
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import GpuRuntime

NUM_LEVELS = 4
SIGMA_BASE = 1.5        # sigma_l = SIGMA_BASE ** l
RHO_FACTOR = 3.0        # integration scale rho_l = RHO_FACTOR * sigma_l
GAMMA = 3.0             # band-energy sharpness
# Frequency 1 / sigma_l, unscaled: SADGS's eta > 1 threshold assumes it (wavelength ~0.25x).
OMEGA_CALIBRATION = 1.0
# Energy floor of the aggregation denominator: flat regions fall to ~0 (no split).
MIN_ENERGY = 1e-6

_SHADER = "structure/structure_tensor.slang"
_TRUNCATE = 4.0         # scipy.ndimage.gaussian_filter's default kernel truncation
_ENTRIES = ("blurH", "blurV", "sobelTensor", "clearMaxTrace", "reduceMaxTrace",
            "normalizeBase", "dogBand", "clearAccum", "accumulateLevel", "finalize")
_SCENE_CHUNK = 16       # views per submit when building a whole scene
_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access


def _blur_radius(sigma: float) -> int:
    return int(_TRUNCATE * float(sigma) + 0.5)


class StructureTensor:
    """GPU builder of the field for one resolution; reuse it for every view.

    Not thread-safe: the scratch textures are shared across calls.
    """

    def __init__(self, device: spy.Device, width: int, height: int,
                 num_levels: int = NUM_LEVELS, sigma_base: float = SIGMA_BASE,
                 rho_factor: float = RHO_FACTOR, gamma: float = GAMMA) -> None:
        self.device = device
        self.runtime = GpuRuntime(device)
        self.width = int(width)
        self.height = int(height)
        self.num_levels = int(num_levels)
        self.sigma_base = float(sigma_base)
        self.rho_factor = float(rho_factor)
        self.gamma = float(gamma)
        self._dim = spy.uint2(self.width, self.height)
        self._threads = [self.width, self.height, 1]
        self._k = {name: device.create_compute_kernel(
            device.load_program(str(SHADERS_DIR / _SHADER), [name])) for name in _ENTRIES}
        self.image = self._texture("img")           # input sRGB in .xyz
        self._scratch = self._texture("scratchA")   # separable-blur intermediate
        self._smoothed = self._texture("scratchB")
        self._base = self._texture("base")          # normalized finest-scale tensor
        self._integrated = self._texture("int")     # base tensor re-integrated per level
        self._band = self._texture("band")          # DoG band response in .x
        self._cascade = (self._texture("smooth0"), self._texture("smooth1"))
        self._accum = self._texture("accum")        # (weighted tensor.xyz, weight.w)
        self.output = self._texture("out")
        self._max_trace = spy.NDBuffer(
            device=device, dtype="uint", shape=(1,),
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)

    def _texture(self, label: str) -> spy.Texture:
        return self.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=self.width, height=self.height, usage=_TEXTURE_USAGE, label=f"st_{label}")

    def _dispatch(self, encoder, name: str, **kwargs) -> None:
        self._k[name].dispatch(command_encoder=encoder, thread_count=self._threads, **kwargs)

    def _blur(self, encoder, src: spy.Texture, dst: spy.Texture, sigma: float) -> None:
        """Separable Gaussian (scipy ``mode='nearest'``); in place is fine."""
        radius = _blur_radius(sigma)
        self._dispatch(encoder, "blurH", src=src, dst=self._scratch,
                       dim=self._dim, sigma=float(sigma), radius=int(radius))
        self._dispatch(encoder, "blurV", src=self._scratch, dst=dst,
                       dim=self._dim, sigma=float(sigma), radius=int(radius))

    def record_into(self, encoder, src: spy.Texture, dst: spy.Texture) -> None:
        """Record the field of the sRGB image in ``src`` into ``dst``; no submit."""
        self._blur(encoder, src, self._smoothed, self.sigma_base)
        self._dispatch(encoder, "sobelTensor", smoothed=self._smoothed,
                       tensorOut=self._base, dim=self._dim)
        self._blur(encoder, self._base, self._base, 1.0)
        self._k["clearMaxTrace"].dispatch(command_encoder=encoder, thread_count=[1, 1, 1],
                                          maxTrace=self._max_trace.storage)
        self._dispatch(encoder, "reduceMaxTrace", tensorIn=self._base,
                       maxTrace=self._max_trace.storage, dim=self._dim)
        self._dispatch(encoder, "normalizeBase", tensor=self._base,
                       maxTrace=self._max_trace.storage, dim=self._dim)

        self._dispatch(encoder, "clearAccum", accum=self._accum, dim=self._dim)
        current, sigma_accum = src, 0.0
        for level in range(self.num_levels):
            target_sigma = self.sigma_base ** level
            sigma_inc = math.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
            freq_sq = (OMEGA_CALIBRATION / target_sigma) ** 2
            following = self._cascade[level % 2]
            self._blur(encoder, current, following, sigma_inc)
            self._dispatch(encoder, "dogBand", cur=current, nxt=following,
                           bandOut=self._band, dim=self._dim)
            if level > 0:
                self._blur(encoder, self._band, self._band, target_sigma * 2.0)
            self._blur(encoder, self._base, self._integrated, self.rho_factor * target_sigma)
            self._dispatch(encoder, "accumulateLevel", band=self._band,
                           intTensor=self._integrated, accum=self._accum, dim=self._dim,
                           gamma=float(self.gamma), freq_sq=float(freq_sq))
            current, sigma_accum = following, target_sigma
        self._dispatch(encoder, "finalize", accum=self._accum, source=src, fieldOut=dst,
                       dim=self._dim, min_energy=float(MIN_ENERGY))

    def compute(self, image_srgb: np.ndarray) -> np.ndarray:
        """``(H, W, 3)`` float32 (Sxx, Sxy, Syy) of an ``(H, W, >=3)`` raw sRGB image."""
        image = np.asarray(image_srgb, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"expected (H, W, >=3) image, got {image.shape}")
        height, width = image.shape[:2]
        if (width, height) != (self.width, self.height):
            raise ValueError(
                f"image {width}x{height} != instance {self.width}x{self.height}; "
                "build a StructureTensor per resolution")
        rgba = np.zeros((height, width, 4), dtype=np.float32)
        rgba[..., :3] = image[..., :3]
        self.image.copy_from_numpy(np.ascontiguousarray(rgba))
        batch = self.runtime.command_batch("structure_tensor")
        self.record_into(batch.encoder, self.image, self.output)
        self.runtime.submit(batch)
        self.runtime.wait_for_idle()
        return np.ascontiguousarray(self.output.to_numpy()[..., :3].astype(np.float32))

    def build_textures(self, targets: dict[int, spy.Texture]) -> dict[int, spy.Texture]:
        """One rgba16_float field texture per target texture, all on the GPU."""
        fields = {index: self.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba16_float,
            width=self.width, height=self.height, usage=_TEXTURE_USAGE,
            label=f"structure_tensor_{index}") for index in targets}
        indices = list(targets)
        for start in range(0, len(indices), _SCENE_CHUNK):
            batch = self.runtime.command_batch("structure_tensor_scene")
            for index in indices[start:start + _SCENE_CHUNK]:
                self.record_into(batch.encoder, targets[index], fields[index])
            self.runtime.submit(batch)
            self.runtime.wait_for_idle()
        return fields
