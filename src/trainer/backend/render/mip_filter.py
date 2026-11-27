# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Mip-Splatting 3D smoothing filter: per-splat size from the training cameras."""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import MipResources, PrimitiveStorage
from trainer.data.dataset import Camera

_CAMERA_FLOATS = 36   # view 4x4, proj 4x4, image size, 2 pad (CameraData)


class MipFilter:
    KERNELS = {
        "prepare_args": ("mip/compute_3d_filter.slang", "prepareMipFilterArgs"),
        "clear_tail": ("mip/compute_3d_filter.slang", "clearMipFilterTail"),
        "compute": ("mip/compute_3d_filter.slang", "computeMip3DFilter"),
    }

    def __init__(self, device: spy.Device, commands: CommandRecorder,
                 kernels: KernelLibrary, *, primitive: PrimitiveStorage,
                 resources: MipResources, capacity: int) -> None:
        self._device = device
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._primitive = primitive
        self._resources = resources
        self._capacity = capacity
        self.cameras: spy.Buffer | None = None
        self.num_cameras = 0
        self.filter_coef = 0.0

    def set_cameras(self, cameras: list[Camera]) -> None:
        """Upload the training cameras and the sqrt(0.2) / max_focal coefficient."""
        data = np.zeros((len(cameras), _CAMERA_FLOATS), dtype=np.float32)
        max_focal = 0.0
        for i, camera in enumerate(cameras):
            data[i, 0:16] = camera.view_matrix.flatten()
            data[i, 16:32] = camera.proj_matrix.flatten()
            data[i, 32:34] = np.array(camera.image_size, dtype=np.float32)
            # proj[0][0] = 2 fx / W
            fx = abs(float(camera.proj_matrix[0, 0])) * float(camera.image_size[0]) * 0.5
            max_focal = max(max_focal, fx)
        from trainer.backend.memory.buffers import check_storage_range
        check_storage_range("mip_cameras", data.nbytes)
        buffer = self._device.create_buffer(
            size=data.nbytes,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="mip_cameras",
        )
        buffer.copy_from_numpy(np.ascontiguousarray(data.reshape(-1).view(np.uint8)))
        self.cameras = buffer
        self.num_cameras = len(cameras)
        self.filter_coef = float(np.sqrt(0.2) / max(max_focal, 1e-8))

    def compute(self) -> None:
        """Recompute every live splat's filter size; run after each count change."""
        if self.cameras is None:
            raise RuntimeError("MipFilter.compute() requires set_cameras(...) first")
        primitive, r = self._primitive, self._resources
        commands = self._commands
        with commands.recording("mip_filter"), commands.marker_group("mip_filter"):
            commands.dispatch(self._k.prepare_args, "mip_filter_prepare_args",
                thread_count=[1, 1, 1],
                num_gaussians_counter=primitive.active_count.storage,
                dispatch_args=r.dispatch_args.storage,
                previous_count=r.previous_count.storage,
                max_splats=self._capacity)
            commands.dispatch_indirect(self._k.clear_tail, "mip_filter_clear",
                r.dispatch_args, offset=3 * 4,
                previous_count=r.previous_count.storage,
                filter_3D=primitive.filter_3d.storage)
            commands.dispatch_indirect(self._k.compute, "compute_3d_filter",
                r.dispatch_args,
                gaussians=primitive.values.storage,
                num_gaussians_counter=primitive.active_count.storage,
                cameras=self.cameras,
                filter_3D=primitive.filter_3d.storage,
                num_cameras=self.num_cameras,
                filter_coef=self.filter_coef,
                max_splats=self._capacity)
