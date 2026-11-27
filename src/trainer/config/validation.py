# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Fail-fast checks of the parsed arguments, before any GPU allocation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from trainer.config.config_types import DensificationMode

if TYPE_CHECKING:
    import argparse


class ConfigError(ValueError):
    """An invalid CLI/config combination, reported before anything is allocated."""


# The radix sort's --max-sort-pairs ceiling at the Vulkan-spec minimum dispatch width: 65,535 groups of 2,048 keys.
# The frontend passes the device's own ceiling (trainer.backend.memory.buffers.max_sort_pairs_limit); this layer
# imports no backend module, so the default is a literal that tests/test_sort_dispatch_limit.py pins to the backend's.
SPEC_MAX_SORT_PAIRS = 65_535 * 2_048


def validate_training_config(args: argparse.Namespace, *, n_cams: int, n_train: int,
                             n_test: int, max_sort_pairs_limit: int = SPEC_MAX_SORT_PAIRS) -> None:
    """Reject combinations that would crash mid-run or silently do nothing.

    Runs once the scene and holdout split are known; the CLI defaults always pass.
    """
    mode = DensificationMode(args.densify_mode)

    masks = getattr(args, "masks", None)
    if masks and float(getattr(args, "w_alpha", 0.0)) > 0.0:
        raise ConfigError("--masks and --w-alpha > 0 are exclusive: the loss mask is stored in the "
                          "target's alpha channel, so alpha supervision would train against the mask.")
    if getattr(args, "invert_masks", False) and not masks:
        raise ConfigError("--invert-masks needs --masks.")
    validate_geometry(args)

    if args.iters < 1 and not (args.preset == "quality+" and args.iters == 0):
        raise ConfigError(f"--iters must be >= 1, got {args.iters}.")
    if args.max_splats < 1:
        raise ConfigError(f"--max-splats must be >= 1, got {args.max_splats}.")
    prefetch = getattr(args, "target_prefetch", 0)
    if prefetch < 0:
        raise ConfigError(f"--target-prefetch must be >= 0, got {prefetch}.")
    # A window past the training views is clamped to them by the loop (the defaults always pass).
    if not 0.0 < args.final_splat_fraction <= 1.0:
        raise ConfigError("--final-splat-fraction must be in (0, 1].")
    # The radix sort dispatches one group per 2,048 keys along a single GPU dimension (65,535 groups is the
    # Vulkan-spec minimum), and the raster checkpoints must fit one storage buffer: the caller passes the ceiling
    # for its device and frame size when it knows them.
    if args.max_sort_pairs is not None and not 1 <= args.max_sort_pairs <= max_sort_pairs_limit:
        raise ConfigError(f"--max-sort-pairs must be between 1 and {max_sort_pairs_limit:,}.")

    # densify_every is a modulo divisor of the schedule, not a "disable" value.
    if args.densify_every < 1:
        raise ConfigError(
            f"--densify-every must be >= 1 (it divides the densify schedule); "
            f"got {args.densify_every}. Use --densify-mode none to disable "
            f"densification.")
    if args.densify_from < 0:
        raise ConfigError(f"--densify-from must be >= 0, got {args.densify_from}.")
    if args.densify_from > args.densify_to:
        raise ConfigError(
            f"--densify-from ({args.densify_from}) must be <= --densify-to "
            f"({args.densify_to}); otherwise the densify window never opens.")
    profile_power = float(args.splat_profile_power)
    if profile_power < 1.0:
        raise ConfigError(
            f"--splat-profile-power must be >= 1, got {profile_power}: below 1 "
            f"the falloff has a cusp at every splat's center.")
    blend_weight = float(args.prune_blend_weight)
    if blend_weight < 0.0:
        raise ConfigError(
            f"--prune-blend-weight must be >= 0 (0 = off), got {blend_weight}.")

    # Modulo divisors only on the structure-aware path.
    if mode == DensificationMode.STRUCTURE_AWARE:
        if args.sad_structure_every < 1:
            raise ConfigError(
                f"--sad-structure-every must be >= 1 for --densify-mode "
                f"structure_aware, got {args.sad_structure_every}.")
        if args.sad_accum_every < 1:
            raise ConfigError(
                f"--sad-accum-every must be >= 1 for --densify-mode "
                f"structure_aware, got {args.sad_accum_every}.")
        if args.sad_max_split_factor < 1:
            raise ConfigError(
                f"--sad-max-split-factor must be >= 1, got "
                f"{args.sad_max_split_factor}.")

    stbn_fraction = float(args.stbn_pixel_fraction)
    if stbn_fraction <= 0.0 or stbn_fraction > 1.0:
        raise ConfigError(
            f"--stbn-pixel-fraction must be in (0, 1], got {stbn_fraction}. "
            f"1.0 = supervise every pixel (the feature off).")
    if stbn_fraction < 1.0:
        if int(args.backward_pixel_stride) > 1:
            raise ConfigError(
                f"--stbn-pixel-fraction {stbn_fraction} and "
                f"--backward-pixel-stride {args.backward_pixel_stride} both "
                f"subsample the backward's pixels; use one or the other.")
        if args.stbn_source == "file" and not args.stbn_file:
            raise ConfigError("--stbn-source file requires --stbn-file <path.npy>.")

    stride = max(args.eval_holdout_stride, 1)
    if n_train < 1:
        raise ConfigError(
            f"empty training split: {n_cams} cameras with holdout stride "
            f"{stride} left 0 train views.")
    if args.eval_holdout and n_test < 1:
        raise ConfigError(
            f"--eval is on but the holdout split left 0 test views "
            f"({n_cams} cameras, stride {stride}).")


def validate_geometry(args: argparse.Namespace) -> None:
    """The depth / normal / depth-normal / flatten flags: refuse what would crash or do nothing."""
    g = getattr
    weights = {"--depth-loss-weight": float(g(args, "depth_loss_weight", 0.0) or 0.0),
               "--normal-loss-weight": float(g(args, "normal_loss_weight", 0.0) or 0.0),
               "--depth-normal-weight": float(g(args, "depth_normal_weight", 0.0) or 0.0),
               "--flatten-loss-weight": float(g(args, "flatten_loss_weight", 0.0) or 0.0)}
    for flag, value in weights.items():
        if not value >= 0.0:
            raise ConfigError(f"{flag} must be >= 0, got {value}.")
        if not math.isfinite(value):
            raise ConfigError(f"{flag} must be finite, got {value}.")
    for weight, folder, flag in (("--depth-loss-weight", "depth_dir", "--depth-dir"),
                                 ("--normal-loss-weight", "normal_dir", "--normal-dir")):
        has_dir = bool(g(args, folder, None))
        if weights[weight] > 0.0 and not has_dir:
            raise ConfigError(f"{weight} > 0 needs {flag}.")
        if has_dir and weights[weight] <= 0.0:
            raise ConfigError(f"{flag} is set but {weight} is 0: the priors would be decoded for nothing.")
    image_terms = max(weights["--depth-loss-weight"], weights["--normal-loss-weight"],
                      weights["--depth-normal-weight"]) > 0.0
    if image_terms and float(g(args, "stbn_pixel_fraction", 1.0)) < 1.0:
        raise ConfigError("depth/normal losses are not supported with STBN (--stbn-pixel-fraction < 1): "
                          "their gradient needs every pixel's neighbours.")
    units = float(g(args, "geom_units_per_meter", 1.0))
    if not (0.0 < units and math.isfinite(units)):
        raise ConfigError(f"--geom-units-per-meter must be finite and > 0, got {units}.")
    if not 0.0 < float(g(args, "geom_min_alpha", 0.5)) <= 1.0:
        raise ConfigError("--geom-min-alpha must be in (0, 1].")
    geom_from = int(g(args, "geom_from", 0))
    if geom_from < 0:
        raise ConfigError("--geom-from must be >= 0.")
    iters = int(g(args, "iters", 0) or 0)
    if image_terms and iters > 0 and geom_from > iters:
        raise ConfigError(f"--geom-from {geom_from} is past --iters {iters}: the depth/normal terms would never run.")
