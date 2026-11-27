# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The one place parsed CLI arguments become the run's config objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from trainer.config.config import (
    AdamLRs,
    AppearanceConfig,
    DensifyConfig,
    GeometryConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
    ScheduleConfig,
    ShaderSpecialization,
    StbnConfig,
    StructureConfig,
)
from trainer.config.config_types import DensificationMode, LearningRateRecipe, SplatKernel

if TYPE_CHECKING:
    import argparse

# The quality recipe: measured +0.37 dB at SH0 for ~11% more training time.
QUALITY_OPACITY_LR_FACTOR = 0.7
QUALITY_SCALE_ROTATION_DECAY = (3.0, 0.5)


def shader_specialization(args: argparse.Namespace) -> ShaderSpecialization:
    """Compile-time policy; needed before the scene, when the device is created."""
    fraction = float(args.stbn_pixel_fraction)
    enabled = fraction < 1.0
    # A warmup starts at full density, which a shrunken stage could not hold.
    staged = 0.0 if int(args.stbn_warmup_iters) > 0 else float(args.stbn_max_fraction or 0.0)
    return ShaderSpecialization(
        backward_pixel_stride=args.backward_pixel_stride,
        hardness_is_one=float(args.extinction_hardness) <= 1.0,
        stbn_enabled=enabled,
        stbn_staged_fraction=(max(staged, fraction) if enabled and staged else 0.0),
        adam_bf16=args.adam_moments == "bf16",
        splat_kernel=SplatKernel(args.splat_kernel),
        splat_profile_power=float(args.splat_profile_power),
        splat_profile_onset=float(args.splat_profile_onset),
        loss_mask=bool(getattr(args, "masks", None)),
        geom_priors=geometry_config(args).image_terms,
    )


def geometry_config(args: argparse.Namespace) -> GeometryConfig:
    """The depth / normal / depth-normal / flatten settings; every flag defaults to off."""
    g = getattr
    return GeometryConfig(
        depth_weight=float(g(args, "depth_loss_weight", 0.0) or 0.0),
        normal_weight=float(g(args, "normal_loss_weight", 0.0) or 0.0),
        depth_normal_weight=float(g(args, "depth_normal_weight", 0.0) or 0.0),
        flatten_weight=float(g(args, "flatten_loss_weight", 0.0) or 0.0),
        depth_source=str(g(args, "depth_source", "center")),
        depth_space=str(g(args, "depth_loss_space", "disparity")),
        normal_loss=str(g(args, "normal_loss", "l1")),
        normal_encoding=str(g(args, "normal_encoding", "gray128")),
        min_alpha=float(g(args, "geom_min_alpha", 0.5)),
        start_iter=int(g(args, "geom_from", 0)),
        units_per_meter=float(g(args, "geom_units_per_meter", 1.0)),
        depth_dir=g(args, "depth_dir", None),
        normal_dir=g(args, "normal_dir", None),
    )


@dataclass(frozen=True)
class RunConfig:
    """Every config object of one run, resolved from its arguments and scene extent."""

    densify_mode: DensificationMode
    schedule: ScheduleConfig
    raster: RasterConfig
    loss: LossConfig
    regularization: RegularizationConfig
    appearance: AppearanceConfig
    stbn: StbnConfig
    densify: DensifyConfig
    structure: StructureConfig
    base_lrs: AdamLRs
    position_lr_max_steps: int
    # Scale and rotation LR multipliers at the first and the last step.
    scale_rotation_lr_decay: tuple[float, float]
    geometry: GeometryConfig = GeometryConfig()

    @classmethod
    def from_args(cls, args: argparse.Namespace, scene_extent: float) -> RunConfig:
        mode = DensificationMode(args.densify_mode)
        loss, base_lrs = _loss_and_lrs(args, mode)
        base_lrs, scale_rotation_lr_decay = _apply_recipe(
            base_lrs, LearningRateRecipe(args.lr_recipe))
        schedule = _schedule(args, mode)
        return cls(
            densify_mode=mode,
            schedule=schedule,
            raster=RasterConfig(
                z_near_cull=args.z_near_cull,
                cluster_cull=args.cluster_cull,
                mip_enabled=args.mip_splatting,
                kernel_size=(args.mip_kernel_size if args.mip_splatting
                             else RasterConfig.kernel_size),
            ),
            loss=loss,
            regularization=RegularizationConfig(opacity_weight=args.opacity_reg,
                                                scale_weight=args.scale_reg),
            appearance=AppearanceConfig(sh_degree=args.sh_degree,
                                        band_interval=_sh_band_interval(args, schedule.optimization_steps)),
            stbn=_stbn(args),
            densify=DensifyConfig(
                t_opacity=args.t_opacity,
                t_big_scale=args.t_big_scale * scene_extent,
                t_opacity_reset_value=args.opacity_reset_value,
                reset_relative=args.sad_relative_reset,
                reset_decay=args.sad_reset_decay,
            ),
            structure=StructureConfig(
                n_scoring_views=args.n_scoring_views,
                tau_split=args.sad_tau_split,
                tau_prune=args.sad_tau_prune,
                grad_threshold=args.sad_grad_threshold,
                split_power=args.sad_split_power,
                max_split_factor=args.sad_max_split_factor,
                ks_scale_power=args.sad_ks_scale_power,
                t_opacity=args.sad_structure_min_opacity,
                opacity_cap=args.sad_opacity_cap,
                dense_scale=args.grow_scale3d * scene_extent,
                rank_split_families=args.preset in ("quality", "quality+"),
                detailed_diagnostics=bool(args.densify_diag),
            ),
            base_lrs=base_lrs,
            position_lr_max_steps=(schedule.optimization_steps if args.position_lr_max_steps <= 0
                                   else args.position_lr_max_steps),
            scale_rotation_lr_decay=scale_rotation_lr_decay,
            geometry=geometry_config(args),
        )


def end_of_run_cadence(every: int | None, iters: int) -> int:
    """A pass's cadence in iterations; None (the default) runs it once, on the last step."""
    return iters if every is None else every


def _schedule(args: argparse.Namespace, mode: DensificationMode) -> ScheduleConfig:
    return ScheduleConfig(
        iterations=args.iters,
        densify_from=args.densify_from,
        densify_to=args.densify_to,
        densify_every=args.densify_every,
        reset_every=args.reset_every,
        densify_mode=mode,
        sad_structure_every=args.sad_structure_every,
        sad_accum_every=args.sad_accum_every,
        preview_every=end_of_run_cadence(args.preview_every, args.iters),
        log_every=args.log_every,
        render_all_every=args.render_all_every,
        eval_every=end_of_run_cadence(args.eval_every, args.iters) if args.evaluation else 0,
        psnr_curve_enabled=bool(args.psnr_curve),
        psnr_curve_every=args.psnr_curve_every,
        morton_every=args.morton_every,
        stream_every=args.stream_every,
        mip_enabled=bool(args.mip_splatting),
        eval_iters=tuple(sorted(set(args.eval_iters or ()))) if args.evaluation else (),
        prune_blend_weight=float(args.prune_blend_weight),
        prune_blend_from=int(args.prune_blend_from),
        render_all_on_preview=bool(args.evaluation),
        until_convergence=args.preset == "quality+",
    )


def _loss_and_lrs(args: argparse.Namespace,
                  mode: DensificationMode) -> tuple[LossConfig, AdamLRs]:
    """SAD-GS runs train with SADGS's loss and learning rates unless told otherwise."""
    if mode == DensificationMode.STRUCTURE_AWARE and args.sad_match_loss_lrs:
        print(f"[SAD] matching SADGS loss + LRs: lambda_l2={args.sad_lambda_l2}, "
              f"lambda_ssim={args.lambda_ssim}, scale_lr=0.01, rot_lr=0.002, "
              f"opacity_lr=0.05, sh0_lr=0.0025")
        return (LossConfig(alpha_weight=args.w_alpha, ssim_weight=args.lambda_ssim,
                           l2_weight=args.sad_lambda_l2),
                AdamLRs(scale=0.01, rot=0.002, opacity=0.05, color=0.0025))
    return (LossConfig(alpha_weight=args.w_alpha, ssim_weight=args.lambda_ssim,
                       l2_weight=args.lambda_l2),
            AdamLRs())


def _apply_recipe(lrs: AdamLRs, recipe: LearningRateRecipe) -> tuple[AdamLRs, tuple[float, float]]:
    if recipe == LearningRateRecipe.QUALITY:
        return (replace(lrs, opacity=lrs.opacity * QUALITY_OPACITY_LR_FACTOR),
                QUALITY_SCALE_ROTATION_DECAY)
    return lrs, (1.0, 1.0)


def _sh_band_interval(args: argparse.Namespace, optimization_steps: int) -> int:
    """0 = auto: the last SH band turns on by half the run.

    The 3DGS constant (1000) assumes 30k iterations; a 3k run would barely train degree 3.
    """
    if args.sh_band_interval > 0:
        return args.sh_band_interval
    if args.sh_degree <= 0:
        return max(1, optimization_steps)
    return max(1, optimization_steps // (2 * args.sh_degree))


def _stbn(args: argparse.Namespace) -> StbnConfig:
    # backward_compacted must match whether STBN_ENABLED was compiled in.
    return StbnConfig(
        pixel_fraction=float(args.stbn_pixel_fraction),
        warmup_iters=int(args.stbn_warmup_iters),
        backward_compacted=shader_specialization(args).stbn_enabled,
        source=str(args.stbn_source),
        file_path=str(args.stbn_file),
        seed=int(args.stbn_seed),
        independent=bool(args.stbn_independent),
        select_mode=str(args.stbn_select_mode),
    )
