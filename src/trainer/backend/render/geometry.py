# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Depth-prior, normal-prior, depth-normal and flatten losses on the GPU.

``GeometryForward`` writes each splat's camera-frame (normal, depth) and is composited by the
forward into ``aux_image`` (shader define GEOM_PRIORS). ``GeometryTerms`` turns that composite
and the priors into dL/d(N, S) for the blend backward, then chains the per-splat seeds to the
means and rotations, and adds the flatten term. Blend weights are constants for these terms.
"""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.backend.compute.device import define_enabled
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.config.config import GeometryConfig

_TEX_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
_BUF_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
_BUF_COPY = _BUF_USAGE | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination

GEOM_STATS_SLOTS = 17
SLOT = {"V_d": 0, "V_n": 1, "V_dn": 2, "sum_x": 3, "sum_y": 4, "sum_rx": 5, "sum_ry": 6, "sum_rxx": 7,
        "sum_rxy": 8, "loss_d": 9, "loss_n": 10, "loss_dn": 11, "depth_on": 14, "skipped_frames": 15,
        "depth_frames": 16}
_COUNT_SLOTS = {"V_d", "V_n", "V_dn", "depth_on", "skipped_frames", "depth_frames"}
_MEAN_SLOTS = {"sum_x", "sum_y"}          # summed at SSI_MEAN_SCALE (geom_loss.slang)
SSI_MEAN_SCALE = 16777216.0
SLOT_SSI_A, SLOT_SSI_B = 12, 13
DEPTH_SPACES = ("disparity", "metric", "ssi")
DEPTH_SOURCES = ("center", "plane")
NORMAL_LOSSES = ("l1", "cos")
SEED_NAMES = ("nx", "ny", "nz", "s")
FIXED_SCALE = 65536.0
# Flatten stops below this smallest-axis scale (metres): see geom_backward.slang.
FLATTEN_FLOOR_M = 1e-5


def compiled_with_geometry(session: spy.SlangSession) -> bool:
    return define_enabled(session.desc.compiler_options.defines, "GEOM_PRIORS")


def geometry_estimate_bytes(config: GeometryConfig | None, *, capacity: int, width: int, height: int) -> int:
    """Device bytes GeometryForward + GeometryTerms allocate for a training pipeline (no GPU work), the prior
    textures at their largest (the training size). 0 without the image terms."""
    if config is None or not config.image_terms:
        return 0
    cap, pixels = max(1, int(capacity)), int(width) * int(height)
    forward = 16 * cap + 16 * pixels                        # geom_data, aux_image
    terms = 2 * 16 * pixels + len(SEED_NAMES) * 8 * cap + 8 * GEOM_STATS_SLOTS   # resolved, dL_dAux, seeds, stats
    priors = 2 * pixels + 4 * pixels                        # r16_uint depth, rgba8_uint normal
    return forward + terms + priors


def resource_bytes(owner) -> dict[str, int]:
    """Device bytes of every buffer and texture attribute of ``owner`` (a dict attribute is summed)."""
    out = {}
    for name, value in vars(owner).items():
        values = list(value.values()) if isinstance(value, dict) else [value]
        size = sum(v.memory_usage.device for v in values if isinstance(v, (spy.Buffer, spy.Texture)))
        if size:
            out[name] = size
    return out


def _texture(device: spy.Device, fmt: spy.Format, width: int, height: int, label: str) -> spy.Texture:
    return device.create_texture(type=spy.TextureType.texture_2d, format=fmt, width=width,
                                 height=height, usage=_TEX_USAGE, label=label)


def _buffer(device: spy.Device, size: int, label: str) -> spy.Buffer:
    return device.create_buffer(size=size, usage=_BUF_COPY, label=label)


class GeometryForward:
    """Per-splat features and the composite texture the forward writes under GEOM_PRIORS."""

    KERNELS = {"geom_features": ("rasterize/tile_binned/geom_features.slang", "geomFeatures")}

    def __init__(self, device: spy.Device, kernels: KernelLibrary, *, width: int, height: int,
                 capacity: int, plane_depth: bool = False, stub: bool = False) -> None:
        """``stub``: the forward kernel's arguments only, one element each, never composited (the viewer: it
        never reads the composite, so it must not pay 16 B per splat and per pixel for it)."""
        self._k = kernels.load(self.KERNELS)
        self.capacity = 1 if stub else capacity
        self.plane_depth = bool(plane_depth)
        self.stub = bool(stub)
        # Whether the forward composites (N, S): off for a stub and before --geom-from (GeometryTerms.active).
        self.composite = not stub
        # Tests write geom_data themselves and skip the feature kernel.
        self.skip_features = False
        self.geom_data = _buffer(device, 16 * max(1, self.capacity), "geom_data")
        self.aux_image = _texture(device, spy.Format.rgba32_float, 1 if stub else width, 1 if stub else height,
                                  "geom_aux_image")

    def record(self, commands: CommandRecorder, *, gaussians, active_count, camera,
               cluster_visible) -> None:
        if self.skip_features or not self.composite:
            return
        commands.dispatch(self._k.geom_features, "geom_features",
            thread_count=[self.capacity, 1, 1],
            gaussians=gaussians, num_gaussians_counter=active_count, camera=camera,
            cluster_visible=cluster_visible, geom_data=self.geom_data,
            plane_depth=int(self.plane_depth), max_splats=self.capacity)

    def forward_kwargs(self) -> dict:
        return {"geom_data": self.geom_data, "aux_image": self.aux_image, "aux_on": int(self.composite)}

    def resource_bytes(self) -> dict[str, int]:
        return resource_bytes(self)


class GeometryTerms:
    """The loss passes, the backward seeds, the per-gaussian chain and the flatten term."""

    LOSS_KERNELS = {
        "clear_stats": ("losses/geom_loss.slang", "geomClearStats"),
        "resolve": ("losses/geom_loss.slang", "geomResolve"),
        "ssi_moments": ("losses/geom_loss.slang", "geomSsiMoments"),
        "grad": ("losses/geom_loss.slang", "geomGrad"),
    }
    BACKWARD_KERNELS = {"geom_backward": ("rasterize/tile_binned/geom_backward.slang", "geomBackward")}

    def __init__(self, device: spy.Device, commands: CommandRecorder, kernels: KernelLibrary, *,
                 config: GeometryConfig, forward: GeometryForward | None, width: int, height: int,
                 capacity: int, loss_mask: bool) -> None:
        if config.depth_source not in DEPTH_SOURCES:
            raise ValueError(f"depth source {config.depth_source!r} not in {DEPTH_SOURCES}")
        if config.depth_space not in DEPTH_SPACES:
            raise ValueError(f"depth space {config.depth_space!r} not in {DEPTH_SPACES}")
        if config.normal_loss not in NORMAL_LOSSES:
            raise ValueError(f"normal loss {config.normal_loss!r} not in {NORMAL_LOSSES}")
        if config.image_terms and forward is None:
            raise RuntimeError("depth/normal/depth-normal weights need the GEOM_PRIORS shaders")
        self.device = device
        self.config = config
        self._commands = commands
        self.forward = forward
        self.width, self.height, self.capacity = width, height, capacity
        self.loss_mask = loss_mask
        self.image_terms = forward is not None
        self._k = kernels.load({**(self.LOSS_KERNELS if self.image_terms else {}), **self.BACKWARD_KERNELS})
        self._active = True              # the three image terms (--geom-from gates them)
        self._grad_zeroed = False        # dL_dAux holds zeros since the terms went inactive
        self._has_depth = self._has_normal = False
        self._pending_depth = self._pending_normal = None
        if self.image_terms:
            if forward.plane_depth != (config.depth_source == "plane"):
                raise RuntimeError("the forward's depth source does not match the loss's")
            self.resolved = _texture(device, spy.Format.rgba32_float, width, height, "geom_resolved")
            self.dL_dAux = _texture(device, spy.Format.rgba32_float, width, height, "geom_dL_dAux")
            self.seeds = {n: _buffer(device, 8 * max(1, capacity), f"grad_geom_{n}") for n in SEED_NAMES}
            for b in self.seeds.values():
                b.copy_from_numpy(np.zeros(2 * max(1, capacity), dtype=np.uint32))
            self.stats = _buffer(device, 8 * GEOM_STATS_SLOTS, "geom_stats")
            self.stats.copy_from_numpy(np.zeros(2 * GEOM_STATS_SLOTS, dtype=np.uint32))
            self.set_prior_grids((1, 1), (1, 1))

    @property
    def active(self) -> bool:
        """Whether the image terms run this step. Off (before --geom-from) the forward composites nothing and the
        loss passes do not run; dL_dAux reads zero for the backward."""
        return self._active

    @active.setter
    def active(self, on: bool) -> None:
        self._active = bool(on)
        if self.forward is not None:
            self.forward.composite = self._active
        if self._active:
            self._grad_zeroed = False

    # ------------------------------------------------------------------ priors

    def set_prior_grids(self, depth_hw: tuple[int, int], normal_hw: tuple[int, int]) -> None:
        """Owned prior textures at each kind's common grid (height, width)."""
        self.depth_hw, self.normal_hw = tuple(depth_hw), tuple(normal_hw)
        self.depth_prior = _texture(self.device, spy.Format.r16_uint, depth_hw[1], depth_hw[0], "geom_depth_prior")
        self.normal_prior = _texture(self.device, spy.Format.rgba8_uint, normal_hw[1], normal_hw[0],
                                     "geom_normal_prior")

    def select(self, depth_mm: np.ndarray | None, normal_codes: np.ndarray | None) -> None:
        """This step's priors: uint16 millimetres (H, W) and gray128-coded uint8 normals (H, W, 4),
        either None for a view without one. Uploaded inside the next recorded loss."""
        if depth_mm is not None:
            assert depth_mm.dtype == np.uint16 and depth_mm.shape == self.depth_hw, (depth_mm.shape, self.depth_hw)
        if normal_codes is not None:
            assert normal_codes.dtype == np.uint8 and normal_codes.shape == (*self.normal_hw, 4), (
                normal_codes.shape, self.normal_hw)
        self._has_depth = depth_mm is not None
        self._has_normal = normal_codes is not None
        self._pending_depth, self._pending_normal = depth_mm, normal_codes

    def _flush(self) -> None:
        if self._pending_depth is not None:
            self._commands.upload_texture(self.depth_prior, np.ascontiguousarray(self._pending_depth))
            self._pending_depth = None
        if self._pending_normal is not None:
            self._commands.upload_texture(self.normal_prior, np.ascontiguousarray(self._pending_normal))
            self._pending_normal = None

    # ------------------------------------------------------------------ loss

    def _frame_kwargs(self, output_image: spy.Texture, camera, loss_mask: dict) -> dict:
        c = self.config
        return dict(
            aux_image=self.forward.aux_image, output_image=output_image, camera=camera,
            depth_prior=self.depth_prior, normal_prior=self.normal_prior,
            geom_resolved=self.resolved, geom_stats=self.stats,
            frame_dim=spy.uint2(self.width, self.height),
            plane_depth=int(c.depth_source == "plane"), min_alpha=float(c.min_alpha),
            units_per_meter=float(c.units_per_meter),
            has_depth=int(self._has_depth), has_normal=int(self._has_normal),
            **(loss_mask if self.loss_mask else {}))

    def weights(self) -> tuple[float, float, float]:
        c = self.config
        if not self.active:
            return 0.0, 0.0, 0.0
        return float(c.depth_weight), float(c.normal_weight), float(c.depth_normal_weight)

    def record_loss(self, *, output_image: spy.Texture, camera, loss_mask: dict) -> None:
        """Pass R then pass G: writes dL_dAux. ``loss_mask`` is ``{"loss_mask": target}``."""
        if not self.image_terms:
            return
        commands = self._commands
        threads = [self.width, self.height, 1]
        if not self.active:
            # The blend backward still reads dL_dAux: zero it once (a weightless grad pass writes zeros and touches
            # no counter), then nothing until the terms switch on. The priors are not uploaded either.
            if not self._grad_zeroed:
                frame = self._frame_kwargs(output_image, camera, loss_mask)
                frame.update(has_depth=0, has_normal=0)
                commands.dispatch(self._k.grad, "geom_grad_zero", thread_count=threads, dL_dAux=self.dL_dAux,
                                  depth_space=0, normal_cos=0, w_depth=0.0, w_normal=0.0, w_dn=0.0, **frame)
                self._grad_zeroed = True
            return
        self._flush()
        w_d, w_n, w_dn = self.weights()
        frame = self._frame_kwargs(output_image, camera, loss_mask)
        with commands.marker_group("geom_loss"):
            commands.dispatch(self._k.clear_stats, "geom_clear_stats", thread_count=[16, 1, 1],
                              geom_stats=self.stats, clear_all=0)
            commands.dispatch(self._k.resolve, "geom_resolve", thread_count=threads,
                              dn_on=int(w_dn > 0.0), **frame)
            if self.config.depth_space == "ssi" and w_d > 0.0 and self._has_depth:
                commands.dispatch(self._k.ssi_moments, "geom_ssi_moments", thread_count=threads,
                                  geom_resolved=self.resolved, depth_prior=self.depth_prior,
                                  geom_stats=self.stats, frame_dim=frame["frame_dim"],
                                  units_per_meter=frame["units_per_meter"], has_depth=1,
                                  **(loss_mask if self.loss_mask else {}))
            commands.dispatch(self._k.grad, "geom_grad", thread_count=threads,
                              dL_dAux=self.dL_dAux,
                              depth_space=DEPTH_SPACES.index(self.config.depth_space),
                              normal_cos=int(self.config.normal_loss == "cos"),
                              w_depth=w_d, w_normal=w_n, w_dn=w_dn, **frame)

    # ------------------------------------------------------------------ backward

    def blend_kwargs(self) -> dict:
        """Extra arguments of the blend backward kernels under GEOM_PRIORS."""
        if not self.image_terms:
            return {}
        return {"dL_dAux": self.dL_dAux, **{f"grad_geom_{n}": b for n, b in self.seeds.items()}}

    def flatten_coefficient(self) -> float:
        """w * 3WH / U (the shader divides by the live count): this trainer's photometric loss is a
        sum over 3WH values, so a mean-over-splats term is scaled by 3WH."""
        c = self.config
        if c.flatten_weight <= 0.0:
            return 0.0
        return float(c.flatten_weight) * 3.0 * self.width * self.height / float(c.units_per_meter)

    def record_backward(self, *, gaussians, active_count, camera, cluster_visible, grads: dict) -> None:
        """After projection_backward: chain the seeds and add flatten (the only writer then)."""
        flatten = self.flatten_coefficient()
        if not self.image_terms and flatten <= 0.0:
            return
        kwargs = dict(gaussians=gaussians, num_gaussians_counter=active_count, camera=camera,
                      cluster_visible=cluster_visible,
                      grad_log_scale_x=grads["log_scale_x"].storage,
                      grad_log_scale_y=grads["log_scale_y"].storage,
                      grad_log_scale_z=grads["log_scale_z"].storage,
                      flatten_coef=flatten,
                      flatten_floor_units=FLATTEN_FLOOR_M * float(self.config.units_per_meter),
                      max_splats=self.capacity)
        if self.image_terms:
            kwargs.update({f"grad_geom_{n}": b for n, b in self.seeds.items()})
            kwargs.update({f"grad_{n}": grads[n].storage for n in
                           ("mean_x", "mean_y", "mean_z", "rot_w", "rot_x", "rot_y", "rot_z")})
            kwargs["plane_depth"] = int(self.config.depth_source == "plane")
        self._commands.dispatch(self._k.geom_backward, "geom_backward",
                                thread_count=[self.capacity, 1, 1], **kwargs)

    # ------------------------------------------------------------------ readback

    def read_stats(self) -> dict:
        """The last step's counts and loss sums (call after the GPU is idle)."""
        if not self.image_terms:
            return {}
        data = self.stats.to_numpy().tobytes()
        raw = np.frombuffer(data, dtype=np.int64)
        out = {}
        for name, slot in SLOT.items():
            v = int(raw[slot])
            out[name] = v if name in _COUNT_SLOTS else v / (SSI_MEAN_SCALE if name in _MEAN_SLOTS else FIXED_SCALE)
        words = np.frombuffer(data, dtype=np.float32)
        out["ssi_a"], out["ssi_b"] = float(words[2 * SLOT_SSI_A]), float(words[2 * SLOT_SSI_B])
        return out

    def read_seeds(self) -> dict[str, np.ndarray]:
        return {n: np.frombuffer(b.to_numpy().tobytes(), dtype=np.int64) / FIXED_SCALE
                for n, b in self.seeds.items()}

    def resource_bytes(self) -> dict[str, int]:
        return resource_bytes(self)

    def log_line(self, stats: dict, skipped_frames_total: int | None = None) -> str:
        """The step's mean of each term as the loss optimizes it, or '-' for a term that did not run (weight 0,
        a skipped scale-shift fit, no supervised pixel). The l1 normal error is a per-component mean."""
        c = self.config
        def mean(key, count, weight, ran=True, per=1.0):
            n = stats.get(count, 0)
            return f"{stats.get(key, 0.0) / (per * n):.5f}" if (n and weight > 0.0 and ran) else "-"
        skipped = stats.get("skipped_frames", 0) if skipped_frames_total is None else skipped_frames_total
        depth = mean("loss_d", "V_d", c.depth_weight, ran=bool(stats.get("depth_on", 1)))
        normal = mean("loss_n", "V_n", c.normal_weight, per=3.0 if c.normal_loss == "l1" else 1.0)
        dn = mean("loss_dn", "V_dn", c.depth_normal_weight)
        return (f"[geom] depth {depth} ({stats.get('V_d', 0)}) | normal {normal} ({stats.get('V_n', 0)}) | "
                f"dn {dn} ({stats.get('V_dn', 0)}) | skipped frames {skipped}")
