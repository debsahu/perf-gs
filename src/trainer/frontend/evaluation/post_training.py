# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Post-training analysis: final per-camera metrics, ``final_psnr.json`` and PLY export.
Refreshes the Mip-Splatting 3D filter first so metrics reflect final positions."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from trainer.common.color import srgb_to_linear as _srgb_to_linear
from trainer.backend.export import save_ply
from trainer.common.metrics import psnr as _psnr_image, psnr_masked as _psnr_masked, ssim as _ssim_image
from trainer.frontend.evaluation.evaluation import render_all_cameras
from trainer.frontend.evaluation.psnr_plot import write_psnr_curve
from trainer.frontend.run_control import Checkpoint, uninterrupted

if TYPE_CHECKING:
    import argparse

    from trainer.backend.render.pipeline import TrainingPipeline


def run_post_training(
    *,
    pipeline: "TrainingPipeline",
    scene,
    args: "argparse.Namespace",
    run_dir: Path,
    cams_dir: Path,
    targets_srgb: Sequence[np.ndarray],
    targets_linear: Callable[[int], np.ndarray],
    test_indices: Sequence[int],
    stride: int,
    n_cams: int,
    train_wall_s: float,
    completed_iters: int,
    t_start: float,
    psnr_curve: list[dict] | None = None,
    progress: Callable[[str, float], None] = lambda _label, _fraction: None,
    checkpoint: Checkpoint = uninterrupted,
    keep_masks: Sequence[np.ndarray | None] | None = None,
) -> None:

    checkpoint()
    if args.profile_run:
        print("[profile-run] finished; skipping final metrics / PLY export.")
        return

    metrics = args.compute_final_metrics and args.evaluation
    if args.compute_final_metrics and not args.evaluation:
        print("[post-training] --no-evaluation: skipping final metrics.")


    if args.psnr_curve:
        write_psnr_curve(psnr_curve or [], args=args, run_dir=run_dir,
                         holdout_stride=stride, n_eval_cams=len(test_indices))

    if args.mip_splatting and metrics:
        progress("refreshing the 3D filter", 0.2)
        checkpoint()
        pipeline.mip_filter.compute()

    per_cam: list[dict] = []
    total_mse_lin = 0.0
    total_mse_srgb = 0.0
    total_mse_srgb_mask = 0.0
    n_mask_used = 0

    if metrics:
        print("computing final per-camera metrics...")
        progress("rendering every camera", 0.25)
        render_all_cameras(pipeline, scene, completed_iters, cams_dir,
                           checkpoint=checkpoint)

        total = max(1, len(scene.images))
        for cam_i, entry in enumerate(scene.images):
            # 0.35 -> 0.9 of the pass: the only part long enough to be worth interpolating.
            progress(f"final metrics, camera {cam_i + 1}/{total}",
                     0.35 + 0.55 * (cam_i / total))
            checkpoint()
            pipeline.render(entry.camera)
            # The render is sRGB (INRIA/gsplat/SADGS convention); sRGB PSNR is the paper metric.
            r_srgb = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
            t_srgb = targets_srgb[cam_i][..., :3]
            mse_srgb = float(np.mean((r_srgb - t_srgb) ** 2))

            r_lin = _srgb_to_linear(r_srgb)
            t_lin = targets_linear(cam_i)
            mse_lin = float(np.mean((r_lin - t_lin) ** 2))

            mask = (t_srgb.sum(-1) > 0.05) | (r_srgb.sum(-1) > 0.05)
            if mask.sum() > 0:
                mse_srgb_mask = float(np.mean((r_srgb[mask] - t_srgb[mask]) ** 2))
                total_mse_srgb_mask += mse_srgb_mask
                n_mask_used += 1
            else:
                mse_srgb_mask = 0.0

            total_mse_lin += mse_lin
            total_mse_srgb += mse_srgb
            per_cam.append({
                "cam_idx": cam_i, "image_name": entry.image_name,
                "in_test_split": cam_i in set(test_indices),
                "mse_linear": mse_lin,
                "psnr_linear": _psnr_image(r_lin, t_lin),
                "mse_srgb": mse_srgb,
                "psnr_srgb": _psnr_image(r_srgb, t_srgb),
                "ssim_srgb": _ssim_image(r_srgb, t_srgb),
                "mse_srgb_masked": mse_srgb_mask,
                "psnr_srgb_masked": (_psnr_image(r_srgb[mask], t_srgb[mask])
                                    if mask.sum() > 0 else 0.0),
            })
            if keep_masks is not None:
                # Over the loss mask's kept pixels; None when a view keeps no pixel.
                per_cam[-1]["psnr_srgb_lossmask"] = _psnr_masked(r_srgb, t_srgb, keep_masks[cam_i])
                per_cam[-1]["lossmask_coverage"] = float(keep_masks[cam_i].mean())

            checkpoint()
        test_set = set(test_indices)
        test_cams = [c for c in per_cam if c["cam_idx"] in test_set]
        train_cams = [c for c in per_cam if c["cam_idx"] not in test_set]

        def mean_metric(cams, key):
            return float(np.mean([c[key] for c in cams])) if cams else 0.0

        def db_of_mean_mse(total_mse, n): return -10.0 * math.log10(max(total_mse / max(n, 1), 1e-12))

        summary = {
            # Headline paper metrics: mean of per-image PSNR/SSIM over the test split.
            "psnr_srgb": mean_metric(test_cams, "psnr_srgb"),
            "ssim_srgb": mean_metric(test_cams, "ssim_srgb"),
            "psnr_linear": mean_metric(test_cams, "psnr_linear"),
            "psnr_srgb_masked": mean_metric(test_cams, "psnr_srgb_masked"),
            "eval_holdout": bool(args.eval_holdout),
            "eval_holdout_stride": stride,
            "n_test_cams": len(test_cams),
            "n_train_cams": len(train_cams),
            # Same paper metric, other splits (train / all cams).
            "train_psnr_srgb": mean_metric(train_cams, "psnr_srgb"),
            "train_ssim_srgb": mean_metric(train_cams, "ssim_srgb"),
            "all_psnr_srgb": mean_metric(per_cam, "psnr_srgb"),
            "all_ssim_srgb": mean_metric(per_cam, "ssim_srgb"),
            # dB of mean MSE (pessimistic pooling): not the paper metric, hence the suffix.
            "test_psnr_srgb_db_of_mean_mse": db_of_mean_mse(
                sum(c["mse_srgb"] for c in test_cams), len(test_cams)),
            "all_psnr_srgb_db_of_mean_mse": db_of_mean_mse(total_mse_srgb, n_cams),
            "all_psnr_linear_db_of_mean_mse": db_of_mean_mse(total_mse_lin, n_cams),
            "per_cam": per_cam,
            "n_cams": n_cams,
            "iters": completed_iters,
            # Training-loop wall-clock, captured with the GPU drained (excludes this pass).
            "train_wall_s": train_wall_s,
            "train_it_per_s": completed_iters / max(train_wall_s, 1e-9),
            "final_n_gaussians": pipeline.primitives.count(),
        }
        if keep_masks is not None:
            def mean_lossmask(cams):
                values = [c["psnr_srgb_lossmask"] for c in cams if c["psnr_srgb_lossmask"] is not None]
                return float(np.mean(values)) if values else None
            summary["psnr_srgb_lossmask"] = mean_lossmask(test_cams)
            summary["train_psnr_srgb_lossmask"] = mean_lossmask(train_cams)
            summary["lossmask_coverage"] = mean_metric(test_cams, "lossmask_coverage")
        (run_dir / "final_psnr.json").write_text(json.dumps(summary, indent=2))

        if args.eval_holdout:
            print(f"TEST PSNR sRGB (paper):  {summary['psnr_srgb']:6.2f} dB  "
                f"<- mean of per-image dB over {summary['n_test_cams']} held-out views "
                f"(compare to SAD-GS/3DGS table)")
            print(f"TEST SSIM sRGB (paper):  {summary['ssim_srgb']:6.4f}     "
                f"<- mean of per-image SSIM, {summary['n_test_cams']} held-out views")
            print(f"train PSNR/SSIM sRGB:    {summary['train_psnr_srgb']:6.2f} dB / "
                f"{summary['train_ssim_srgb']:.4f}  (mean per-image, {summary['n_train_cams']} train views)")
        else:
            print(f"PSNR sRGB (paper):       {summary['psnr_srgb']:6.2f} dB  "
                f"<- mean of per-image dB over all {n_cams} views (no holdout)")
            print(f"SSIM sRGB (paper):       {summary['ssim_srgb']:6.4f}     "
                f"<- mean of per-image SSIM over all {n_cams} views")
        print(f"PSNR sRGB masked:        {summary['psnr_srgb_masked']:6.2f} dB  <- honest silhouette (test split)")
        if summary.get("psnr_srgb_lossmask") is not None:
            print(f"PSNR sRGB loss-masked:   {summary['psnr_srgb_lossmask']:6.2f} dB  <- kept pixels of "
                  f"--masks only (test split, coverage {100.0 * summary['lossmask_coverage']:.1f}%)")
        print(f"PSNR linear:             {summary['psnr_linear']:6.2f} dB  (test split, mean per-image)")
        print(f"  [legacy dB-of-mean-MSE, all cams: {summary['all_psnr_srgb_db_of_mean_mse']:.2f} dB — NOT the paper metric]")
        print(f"final gaussian count:    {summary['final_n_gaussians']}")
        print(f"run dir: {run_dir}")

    if metrics or args.save_ply:
        ply_path = run_dir / "output.ply"
        progress("writing the .ply", 0.92)
        checkpoint()
        save_ply(pipeline, ply_path)
        print(f"saved ply:               {ply_path}")
        print(f"done in {time.time() - t_start:.1f}s")
