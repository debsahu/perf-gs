# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from PIL import Image

from trainer.common.metrics import lossmask_view, psnr as psnr_image, ssim as ssim_image
from trainer.frontend.run_control import Checkpoint, uninterrupted

if TYPE_CHECKING:
    import argparse

    from trainer.backend.render.pipeline import TrainingPipeline

SCHEMA_VERSION = 3

# Render-timing samples: enough to average out a hitch, negligible on a 3k-iteration run.
DEFAULT_FPS_FRAMES = 60
FPS_WARMUP_FRAMES = 5



def device_info(pipeline: "TrainingPipeline") -> dict:

    info: dict[str, Any] = {}
    try:
        device = pipeline.device
    except Exception:                                      # noqa: BLE001
        return {"error": "device unavailable"}

    raw = getattr(device, "info", None)
    for field in ("adapter_name", "api_name", "type", "timestamp_frequency",
                  "optix_version"):
        try:
            value = getattr(raw, field, None)
            info[field] = value if isinstance(
                value, (int, float, str, type(None))) else str(value)
        except Exception:                                  # noqa: BLE001
            info[field] = None

    limits = getattr(raw, "limits", None)
    if limits is not None:
        picked = {}
        for field in ("max_compute_dispatch_thread_groups_x",
                      "max_compute_thread_group_size",
                      "max_shader_visible_samplers",
                      "constant_buffer_offset_alignment"):
            try:
                value = getattr(limits, field, None)
                if value is not None:
                    picked[field] = (value if isinstance(value, (int, float, str))
                                     else str(value))
            except Exception:                              # noqa: BLE001
                pass
        if picked:
            info["limits"] = picked

    try:
        info["supported_shader_model"] = str(device.supported_shader_model)
    except Exception:                                      # noqa: BLE001
        pass
    try:
        # Feature list pins what the backend actually offered this run.
        info["features"] = sorted(str(f) for f in device.features)
    except Exception:                                      # noqa: BLE001
        pass
    return info


def gpu_allocation(pipeline: "TrainingPipeline") -> dict:
    """The trainer's own device allocations by field, as slangpy reports them.
    Machine-independent (``--max-splats``, frame size, SH degree); excludes driver overhead."""
    by_field = pipeline.allocated_bytes()
    total = sum(by_field.values())
    return {
        "total_bytes": total,
        "total_mb": round(total / (1024 * 1024), 2),
        "by_field": dict(sorted(by_field.items(), key=lambda kv: -kv[1])),
        "note": ("trainer-owned allocations walked from GaussianBuffers; excludes "
                 "driver/runtime overhead. Capacity, not use: per-splat buffers are "
                 "sized from --max-splats, so divide by max_splats for bytes-per-splat."),
    }


def stbn_asset_fingerprint(args: "argparse.Namespace") -> dict:
    """Identity of the STBN asset this run read, pinned by content hash rather than filename."""
    from trainer.backend.compute.execution import TILE_AREA
    from trainer.backend.render import stbn_table

    source = str(getattr(args, "stbn_source", ""))
    record: dict[str, Any] = {
        "source": source,
        "pixel_fraction": float(getattr(args, "stbn_pixel_fraction", 1.0)),
        "warmup_iters": int(getattr(args, "stbn_warmup_iters", 0)),
        "independent": bool(getattr(args, "stbn_independent", False)),
        "stbn_seed": int(getattr(args, "stbn_seed", 0)),
        "tile_pixels": stbn_table.pixels_per_tile(
            float(getattr(args, "stbn_pixel_fraction", 1.0))),
        "tile_area": TILE_AREA,
    }
    if source == stbn_table.SOURCE_WHITE:
        record["asset"] = "generated (per-tile permutations from stbn_seed)"
        return record

    override = str(getattr(args, "stbn_file", "") or "")
    path = (Path(override) if override
            else stbn_table.nvidia_volume_path())
    record["asset_path"] = str(path)
    try:
        data = path.read_bytes()
        record["asset_sha256"] = hashlib.sha256(data).hexdigest()
        record["asset_bytes"] = len(data)
        record["asset_name"] = path.name
    except OSError as error:
        record["asset_error"] = f"{type(error).__name__}: {error}"
    return record


def process_environment() -> dict:
    """Interpreter and numeric-stack versions; numpy/scipy drift can move SSIM's last digits."""
    versions: dict[str, str] = {}
    for module in ("numpy", "scipy", "slangpy", "PIL"):
        try:
            versions[module] = str(__import__(module).__version__)
        except Exception:                                  # noqa: BLE001
            versions[module] = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
    }


# The forward stops a pixel's walk once its transmittance drops below this.
_EARLY_OUT_TRANSMITTANCE = 1e-4


def _overdraw(pipeline: "TrainingPipeline", alpha: np.ndarray) -> dict:
    """Blend-loop work of the frame just rendered: per-pixel walk length (what the backward
    replays), tile-splat pairs sorted, and the share of pixels that went opaque early."""
    tile_counts, walk, _, _ = pipeline.read_bin_debug()
    return {
        "walk_mean": float(walk.mean()),
        "walk_p99": float(np.percentile(walk, 99)),
        "tile_pairs": int(tile_counts.sum()),
        "early_out_fraction": float(np.mean(alpha > 1.0 - _EARLY_OUT_TRANSMITTANCE)),
    }


def evaluate_test_views(
    pipeline: "TrainingPipeline",
    scene,
    test_indices: Sequence[int],
    targets_srgb: Sequence[np.ndarray],
    *,
    render_dir: Path | None,
    with_lpips: bool = False,
    progress: Callable[[int, int], None] = lambda _d, _t: None,
    checkpoint: Checkpoint = uninterrupted,
    keep_masks: Sequence[np.ndarray | None] | None = None,
) -> tuple[list[dict], dict]:
    """Per-view PSNR/SSIM over the held-out split, optionally dumping renders.
    Metrics use the float render, not the PNG, so the 8-bit round trip cannot move a score."""
    checkpoint()
    pipeline.runtime.wait_for_idle()
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    lpips_ready = False
    if with_lpips:
        from trainer.common import perceptual
        lpips_ready = perceptual.load()

    per_view: list[dict] = []
    total = max(1, len(test_indices))
    for done, cam_i in enumerate(test_indices, start=1):
        progress(done, total)
        checkpoint()
        entry = scene.images[cam_i]
        pipeline.render(entry.camera)
        image = pipeline.read_output_image()
        rendered = np.clip(image[..., :3], 0.0, 1.0)
        target = targets_srgb[cam_i][..., :3]

        record = {
            "cam_idx": int(cam_i),
            "image_name": entry.image_name,
            "psnr_srgb": psnr_image(rendered, target),
            "ssim_srgb": ssim_image(rendered, target),
            "height": int(rendered.shape[0]),
            "width": int(rendered.shape[1]),
            **_overdraw(pipeline, image[..., 3]),
        }
        if keep_masks is not None:
            record["psnr_srgb_lossmask"], record["lossmask_coverage"] = lossmask_view(
                rendered, target, keep_masks[cam_i])
        # Same float render as PSNR/SSIM, so all three metrics describe one image.
        record["lpips"] = (perceptual.lpips(rendered, target)
                                if lpips_ready else None)
        if render_dir is not None:
            name = f"cam_{cam_i:04d}.png"
            Image.fromarray((rendered * 255.0 + 0.5).astype(np.uint8),
                            mode="RGB").save(render_dir / name)
            record["render_png"] = name
        per_view.append(record)

    checkpoint()

    def _mean(key: str) -> float | None:
        values = [r[key] for r in per_view if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    def _std(key: str) -> float | None:
        values = [r[key] for r in per_view if r.get(key) is not None]
        return float(np.std(values)) if values else None

    summary = {
        # Paper metric: the mean of per-image scores over the held-out split.
        "psnr_srgb": _mean("psnr_srgb"),
        "ssim_srgb": _mean("ssim_srgb"),
        "lpips": _mean("lpips"),
        # Spread across views of one run, not a seed-to-seed uncertainty.
        "psnr_srgb_std_views": _std("psnr_srgb"),
        "ssim_srgb_std_views": _std("ssim_srgb"),
        "lpips_std_views": _std("lpips"),
        "overdraw": {key: _mean(key) for key in
                     ("walk_mean", "walk_p99", "tile_pairs", "early_out_fraction")},
        "n_test_views": len(per_view),
        "n_lpips_views": sum(1 for r in per_view
                             if r.get("lpips") is not None),
        "renders_dir": (str(render_dir) if render_dir is not None else None),
    }
    if keep_masks is not None:
        # Over the --masks kept pixels only; psnr_srgb counts every pixel.
        summary["psnr_srgb_lossmask"] = _mean("psnr_srgb_lossmask")
        summary["lossmask_coverage"] = _mean("lossmask_coverage")
    if with_lpips:
        from trainer.common import perceptual
        summary["lpips_backend"] = perceptual.availability()
    return per_view, summary


def measure_render_fps(pipeline: "TrainingPipeline", scene,
                       cam_idx: int, frames: int, *,
                       checkpoint: Checkpoint = uninterrupted) -> dict:
    """Steady-state render FPS for one held-out camera.
    Serialized on purpose: each frame waits for the GPU, so this measures per-frame latency."""
    if frames <= 0:
        return {"frames": 0, "fps": None, "ms_per_frame": None}
    camera = scene.images[cam_idx].camera
    for _ in range(FPS_WARMUP_FRAMES):
        checkpoint()
        pipeline.render(camera)
    pipeline.runtime.wait_for_idle()

    elapsed = 0.0
    for _ in range(frames):
        checkpoint()
        started = time.perf_counter()
        pipeline.render(camera)
        pipeline.runtime.wait_for_idle()
        elapsed += time.perf_counter() - started
    checkpoint()

    per_frame = elapsed / frames
    return {
        "frames": int(frames),
        "cam_idx": int(cam_idx),
        "fps": (1.0 / per_frame) if per_frame > 0 else None,
        "ms_per_frame": per_frame * 1e3,
        "total_s": elapsed,
        "mode": "serialized (wait_for_idle per frame)",
    }


def build_report(
    *,
    pipeline: "TrainingPipeline",
    scene_context,
    args: "argparse.Namespace",
    run_name: str,
    result,
    per_view: list[dict],
    quality: dict,
    render: dict,
    loop_timing: dict,
    process_started_at: float,
) -> dict:
    """Assemble the full record. Pure aggregation — no GPU work."""
    wall = float(getattr(result, "wall_time_seconds", 0.0) or 0.0)
    observation = float(loop_timing.get("observation_s", 0.0) or 0.0)
    completed = int(getattr(result, "iterations", 0) or 0)
    active = max(0.0, wall - observation)

    n_gaussians = None
    try:
        n_gaussians = pipeline.primitives.count()
    except Exception:                                      # noqa: BLE001
        pass

    return {
        "schema": SCHEMA_VERSION,
        "run_name": run_name,
        "scene_path": str(args.scene),
        "iters_requested": int(args.iters),
        "iters_completed": completed,
        "timing": {
            # Frozen at the last iteration with the GPU drained (same as test_psnr.json's).
            "train_wall_s": wall,
            # The number sparse-vs-dense speed claims must use.
            "train_active_s": active,
            # In-window observation cost: the only part subtracted from train_wall_s.
            "observation_s": observation,
            # Includes the post-freeze final sweep.
            "observation_total_s": float(
                loop_timing.get("observation_total_s", observation) or 0.0),
            "iters_per_s_active": (completed / active) if active > 0 else None,
            "ms_per_iter_active": (active * 1e3 / completed) if completed else None,
            "process_wall_s": time.time() - process_started_at,
            "note": ("train_active_s = train_wall_s - observation_s; "
                     "observation_s is every mid-loop eval/psnr-curve/log pass. "
                     "Runs with no mid-loop evaluation have observation_s == 0."),
        },
        "quality": {**quality, "per_view": per_view},
        "model": {
            "n_gaussians": n_gaussians,
            "sh_degree": int(args.sh_degree),
            "max_splats": int(args.max_splats),
        },
        "render": render,
        "memory": gpu_allocation(pipeline),
        "device": device_info(pipeline),
        "scene": {
            "width": int(scene_context.width),
            "height": int(scene_context.height),
            "downsample": scene_context.colmapScene.downsample,
            "n_cams": int(scene_context.camera_count),
            "n_train": len(scene_context.train_indices),
            "n_test": len(scene_context.test_indices),
            "holdout_stride": int(scene_context.holdout_stride),
            "eval_holdout": bool(args.eval_holdout),
            "split_rule": (
                f"test = cameras where (name-sorted index) % "
                f"{scene_context.holdout_stride} == 0"
                if args.eval_holdout else "no holdout (train == test == all)"),
            "test_indices": [int(i) for i in scene_context.test_indices],
            "n_init_points": int(scene_context.initial_gaussian_count),
            "extent": float(scene_context.extent),
        },
        "stbn": stbn_asset_fingerprint(args),
        "config": {
            "densify_mode": str(getattr(args, "densify_mode", "")),
            "mip_splatting": bool(args.mip_splatting),
            "seed": int(args.seed),
            "sample_mode": str(args.sample_mode),
            "lambda_ssim": float(args.lambda_ssim),
            "backend": str(args.backend),
            "flat_bucket_backward": bool(args.flat_bucket_backward),
            "backward_pixel_stride": int(args.backward_pixel_stride),
        },
        "argv": list(sys.argv),
        "cli_args": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                         else str(v))
                     for k, v in vars(args).items()},
        "environment": process_environment(),
    }


def write_run_report(report: dict, run_dir: Path) -> Path:
    path = run_dir / "run_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
