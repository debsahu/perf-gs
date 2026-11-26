# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Ground-truth target images: upload, selection, and the streamed structure texture."""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.backend.compute.gpu import CommandRecorder

_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access


def srgb01_to_u8(rgba: np.ndarray) -> np.ndarray:
    """Quantize [0, 1] sRGB to contiguous uint8 (round to nearest); uint8 passes through."""
    if rgba.dtype == np.uint8:
        return np.ascontiguousarray(rgba)
    u8 = np.clip(np.rint(rgba.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(u8)


class TargetImages:
    """The rgba8_unorm (8-bit sRGB) target the next loss compares against.

    A pre-uploaded texture is bound by reference; a packed host image streams into the
    owned texture inside the next recorded step.
    """

    def __init__(self, device: spy.Device, commands: CommandRecorder,
                 owned: spy.Texture, width: int, height: int) -> None:
        self._device = device
        self._commands = commands
        self._owned = owned
        self.width = width
        self.height = height
        self.active: spy.Texture = owned
        self._pending: np.ndarray | None = None
        self._structure: spy.Texture | None = None

    def set_image(self, rgb: np.ndarray) -> None:
        """Upload ``(H, W, 3)`` sRGB into the owned target and make it active."""
        assert rgb.shape == (self.height, self.width, 3), (
            f"target shape {rgb.shape} != ({self.height}, {self.width}, 3)")
        self._owned.copy_from_numpy(srgb01_to_u8(self._with_alpha(rgb)))
        self.active = self._owned

    def pack(self, image: np.ndarray) -> np.ndarray:
        """CPU half of :meth:`create` (pure numpy, so it can run on a thread pool)."""
        if image.ndim == 3 and image.shape[2] == 4:
            assert image.shape[:2] == (self.height, self.width)
            return srgb01_to_u8(image)
        assert image.shape == (self.height, self.width, 3), (
            f"target shape {image.shape} != ({self.height}, {self.width}, 3)")
        return srgb01_to_u8(self._with_alpha(image))

    def upload(self, packed: np.ndarray) -> spy.Texture:
        """GPU half of :meth:`create`; must run on the device's thread."""
        texture = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba8_unorm,
            width=self.width, height=self.height,
            usage=_TEXTURE_USAGE,
            label="target_image_cached",
        )
        texture.copy_from_numpy(packed)
        return texture

    def create(self, image: np.ndarray) -> spy.Texture:
        """A new target texture from ``(H, W, 3)`` RGB or ``(H, W, 4)`` RGBA."""
        return self.upload(self.pack(image))

    def select(self, target: spy.Texture | np.ndarray) -> None:
        """Bind a pre-uploaded texture, or stream a packed image with the next loss."""
        if isinstance(target, np.ndarray):
            self._pending = target
            self.active = self._owned
        else:
            self._pending = None
            self.active = target

    def flush(self) -> None:
        """Upload a streamed target into the owned texture (inside the active batch)."""
        if self._pending is not None:
            self._commands.upload_texture(self._owned, self._pending)
            self._pending = None

    def structure_texture(self, builder) -> spy.Texture:
        """Structure field of the streamed active target, derived on the GPU by
        ``builder.record_into(encoder, source, destination)``."""
        if self._structure is None:
            self._structure = self._device.create_texture(
                type=spy.TextureType.texture_2d, format=spy.Format.rgba16_float,
                width=self.width, height=self.height, usage=_TEXTURE_USAGE,
                label="structure_tensor_streamed")
        builder.record_into(self._commands.batch.encoder, self.active, self._structure)
        return self._structure

    def _with_alpha(self, rgb: np.ndarray) -> np.ndarray:
        # Opaque is 255 in uint8, 1.0 in float: np.ones would give a uint8 target alpha 1.
        opaque = 255 if rgb.dtype == np.uint8 else 1
        return np.concatenate([rgb, np.full((self.height, self.width, 1), opaque, rgb.dtype)], axis=-1)
