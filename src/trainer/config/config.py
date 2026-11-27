# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Responsibility-specific immutable configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainer.config.config_types import DensificationMode, SplatKernel


QUALITY_STEPS = 3500


@dataclass(frozen=True)
class RasterConfig:
    """Projection, binning, culling, and Mip-Splatting policy."""

    z_near_cull: float = 0.01
    cluster_cull: bool = True
    mip_enabled: bool = False
    kernel_size: float = 0.3


@dataclass(frozen=True)
class AppearanceConfig:

    sh_degree: int = 0
    band_interval: int = 1000

    @property
    def enabled(self) -> bool:
        return self.sh_degree > 0


@dataclass(frozen=True)
class CompositorConfig:
    """Front-to-back extinction/compositing policy."""
    extinction_hardness: float = 1.0


@dataclass(frozen=True)
class DensificationProperties:
    structure_tensor_texture: Any = None
    structure_kwargs: dict[str, Any] | None = None
    # Sum each splat's blend weight into the densify window (blend-weight pruning).
    track_blend_weight: bool = False
    # Accumulate the per-step screen gradient densification thresholds on.
    track_screen_grad: bool = True
            

@dataclass(frozen=True)
class LossConfig:
    """Image-space objective weights owned by :class:`ImageLoss`."""

    alpha_weight: float = 0.0
    ssim_weight: float = 0.2
    l2_weight: float = 0.0


@dataclass(frozen=True)
class RegularizationConfig:
    """Per-primitive opacity/scale regularization weights."""
    opacity_weight: float = 0.0
    scale_weight: float = 0.0


@dataclass(frozen=True)
class StbnConfig:
    """Spatiotemporal-blue-noise pixel subsampling policy (off at 1.0).

    ``backward_compacted`` records whether STBN_ENABLED was compiled in; when off, the
    loss still zeroes unsampled pixels.
    """

    # Steady-state fraction rho of pixels each backward supervises.
    pixel_fraction: float = 1.0
    # Iterations spent ramping the density from 1.0 down to pixel_fraction.
    warmup_iters: int = 0
    backward_compacted: bool = False
    source: str = "nvidia"
    file_path: str = ""
    seed: int = 0
    # Random slice per step instead of the per-visit temporal walk (the ablation baseline).
    independent: bool = False
    # "global" (NVIDIA's point set) or "tile_rank" (k per 8x8 tile; degrades the blue mask).
    select_mode: str = "global"

    @property
    def enabled(self) -> bool:
        return self.pixel_fraction < 1.0


@dataclass(frozen=True)
class IterationEvents:
    iteration: int
    densify: bool
    opacity_reset: bool
    structure: bool
    preview: bool
    render_all: bool
    log: bool
    eval: bool
    psnr_curve: bool
    morton: bool
    mip_after_densify: bool   # recompute the 3D filter inside the densify block
    mip_periodic: bool         # standalone periodic recompute past densify_to
    blend_prune: bool = False  # this densify tick also prunes by blend weight

@dataclass(frozen=True)
class StbnVariables:
    """This step's STBN mask slice and per-tile pixel budget.

    ``slice_index`` is the camera's visit count plus a per-camera phase, not the iteration:
    the mask's temporal axis tracks successive evaluations of the same image.
    """
    slice_index: int = 0
    pixels_per_tile: int = 0    # 0 => resolved to TILE_AREA (disabled) downstream


@dataclass(frozen=True)
class IterationContextualVariables:
    hardness_value: float
    sh_degree: int
    densification_variables: DensificationProperties | None = None
    contextual_stbn: StbnVariables | None = None


@dataclass
class AdamLRs:
    mean: float = 1.6e-4
    scale: float = 5e-3
    rot: float = 1e-3
    opacity: float = 2.5e-2
    color: float = 2.5e-3  # the SH0 learning rate
    # 6x 3DGS's 0.0025/20, which leaves bands 1-3 undertrained in a 3000-step run.
    sh_rest: float = 0.015 / 20.0


@dataclass(frozen=True)
class ShaderSpecialization:
    """Compile-time shader policy: the device's defines, or a hub run's own Slang session."""

    backward_pixel_stride: int = 1
    # Alpha-over-only compositor when the run's hardness stays <= 1.
    hardness_is_one: bool = False
    # The backward's compacted-visit loop over the STBN-selected pixels.
    stbn_enabled: bool = False
    # Diagnostic (--stbn-max-fraction): staged arrays sized to this tile fraction; 0 = full tile.
    stbn_staged_fraction: float = 0.0
    adam_bf16: bool = False
    splat_kernel: SplatKernel = SplatKernel.GAUSSIAN
    splat_profile_power: float = 1.0
    splat_profile_onset: float = 0.0
    # Per-image loss masks ride in the target's alpha (--masks): LOSS_MASK gates every per-pixel
    # training signal by it and turns alpha supervision off.
    loss_mask: bool = False
    # Depth / normal / depth-normal losses (GEOM_PRIORS): the forward composites per-splat normal
    # and depth beside color, and the backward carries their gradient.
    geom_priors: bool = False


@dataclass(frozen=True)
class ScheduleConfig:
    """The per-iteration schedule knobs ``TrainingSchedule`` turns into events."""

    iterations: int
    densify_from: int
    densify_to: int
    densify_every: int
    reset_every: int
    densify_mode: DensificationMode
    sad_structure_every: int
    sad_accum_every: int
    preview_every: int
    log_every: int
    render_all_every: int
    eval_every: int
    psnr_curve_enabled: bool
    psnr_curve_every: int
    morton_every: int
    stream_every: int
    mip_enabled: bool
    # Extra eval checkpoints, union-ed with the eval_every cadence.
    eval_iters: tuple[int, ...] = ()
    # Blend-weight pruning on densify ticks, in pixels per megapixel (--prune-blend-weight); 0 = off.
    prune_blend_weight: float = 0.2
    # First pruning tick; 0 = the last densify tick, a single cleanup pass.
    prune_blend_from: int = 0
    # Previews also render every camera; --no-evaluation vetoes that.
    render_all_on_preview: bool = True
    until_convergence: bool = False

    @property
    def optimization_steps(self) -> int:
        return QUALITY_STEPS if self.until_convergence else self.iterations

    @property
    def blend_prune_start(self) -> int:
        return self.prune_blend_from or self.densify_to


@dataclass
class DensifyConfig:
    """Shared simple-densify and opacity-reset thresholds."""
    # Prune floor; must stay below t_opacity_reset_value or a reset prunes every splat.
    t_opacity: float = 0.005
    t_big_scale: float = 0.1        # prune if max_scale > this (world-space floater removal)
    t_opacity_reset_value: float = 0.01  # reset opacity to this value (clamp mode)
    # True: scale opacity by reset_decay (SADGS, keeps the ordering); False: clamp (gsplat).
    reset_relative: bool = False
    reset_decay: float = 0.1


@dataclass
class StructureConfig:
    """Structure-aware densification (SAD-GS, arXiv 2604.28016), with SADGS's gates.

    split = high_ratio > tau_split and the gradient is high; prune = low_ratio >
    tau_prune; per-axis split factor n_k = ceil(eta_max_k ** split_power).
    """
    n_scoring_views: int = 10        # views sampled per offline scoring step
    tau_split: float = 0.8
    tau_prune: float = 0.8
    grad_threshold: float = 1e-5     # SADGS is_grad_high floor
    split_power: float = 0.5         # 0.5 = sqrt (paper)
    # Per-axis cap on n_k. SADGS uses 8 (max_clones_per_axis); 2 trains 17% faster for 0.04 dB.
    max_split_factor: int = 2
    # Child scale sigma_k / n_k**power: 1.0 for Mip-NeRF 360, 1.2 for T&T / Deep Blending (SADGS).
    ks_scale_power: float = 1.0
    t_opacity: float = 0.005
    opacity_cap: float = 0.8         # post-densify cap on every survivor (SADGS)
    # Clone (max scale <= this) vs anisotropic split, in world units (SADGS dense * extent).
    dense_scale: float = 0.0
    detailed_diagnostics: bool = False  # full eta readback only when requested
    rank_split_families: bool = False  # allocate scarce slots by screen gradient per child


@dataclass(frozen=True)
class GeometryConfig:
    """Depth-prior, normal-prior, depth-normal and flatten losses (all off by default).

    Weights are Brush-equivalent: each term is a mean over its supervised pixels (flatten: over
    live splats), scaled so a weight means what it means beside a per-pixel-mean L1."""

    depth_weight: float = 0.0
    normal_weight: float = 0.0
    depth_normal_weight: float = 0.0
    flatten_weight: float = 0.0
    depth_source: str = "center"        # center | plane
    depth_space: str = "disparity"      # disparity | metric | ssi
    normal_loss: str = "l1"             # l1 | cos
    normal_encoding: str = "slam"       # slam | spirula
    min_alpha: float = 0.5
    start_iter: int = 0                 # first step the three image terms are on
    units_per_meter: float = 1.0
    depth_dir: str | None = None
    normal_dir: str | None = None

    @property
    def image_terms(self) -> bool:
        """Whether any per-pixel term is on (the GEOM_PRIORS shaders are then required)."""
        return max(self.depth_weight, self.normal_weight, self.depth_normal_weight) > 0.0

    @property
    def enabled(self) -> bool:
        return self.image_terms or self.flatten_weight > 0.0
