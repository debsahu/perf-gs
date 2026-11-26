# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Fail-fast checks of the parsed arguments, before any GPU allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainer.config.config_types import DensificationMode

if TYPE_CHECKING:
    import argparse


class ConfigError(ValueError):
    """An invalid CLI/config combination, reported before anything is allocated."""


SPEC_MIN_DISPATCH_GROUPS = 65_535


def dispatch_groups_x(device) -> int:
    """The device's maximum compute dispatch width along X; the Vulkan-spec minimum when unknown."""
    try:
        return int(device.info.limits.max_compute_dispatch_thread_groups[0])
    except (AttributeError, TypeError, IndexError):
        return SPEC_MIN_DISPATCH_GROUPS


def max_sort_pairs_limit(max_dispatch_groups: int = SPEC_MIN_DISPATCH_GROUPS) -> int:
    """Largest --max-sort-pairs: the radix sort dispatches one group per RADIX_PART_SIZE keys along X, and each of
    its six pair buffers (capacity x 4 bytes) must stay within a 4 GiB storage-buffer range.

    The 4 GiB is hard-coded: slangpy 0.42 exposes max_compute_dispatch_thread_groups but not the device's
    maxStorageBufferRange (4 GiB - 1 on MoltenVK and on a Pascal desktop GPU). This is an addressing bound, not a memory
    budget: six buffers at the cap need ~24 GiB, so a device without that much fails loudly at allocation."""
    from trainer.backend.compute.execution import RADIX_PART_SIZE
    groups = min(int(max_dispatch_groups), (2**32 - 1) // (4 * RADIX_PART_SIZE))
    return groups * RADIX_PART_SIZE


def validate_training_config(args: argparse.Namespace, *, n_cams: int, n_train: int,
                             n_test: int, max_dispatch_groups: int = SPEC_MIN_DISPATCH_GROUPS) -> None:
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

    if args.iters < 1 and not (args.preset == "quality+" and args.iters == 0):
        raise ConfigError(f"--iters must be >= 1, got {args.iters}.")
    if args.max_splats < 1:
        raise ConfigError(f"--max-splats must be >= 1, got {args.max_splats}.")
    if not 0.0 < args.final_splat_fraction <= 1.0:
        raise ConfigError("--final-splat-fraction must be in (0, 1].")
    # The radix sort dispatches one group per 2,048 keys along a single GPU dimension: 65,535 groups is the
    # Vulkan-spec minimum, but the caller passes the device's own limit when it knows it.
    max_pairs = max_sort_pairs_limit(max_dispatch_groups)
    if args.max_sort_pairs is not None and not 1 <= args.max_sort_pairs <= max_pairs:
        raise ConfigError(f"--max-sort-pairs must be between 1 and {max_pairs:,}.")

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
