# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The loaded scene of a run: COLMAP data, resolution, and the train/test split."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from trainer.backend.densify.morton import morton_bounds
from trainer.data.colmap import ColmapDataset, ColmapImageEntry, load_colmap
from trainer.data.dataset import Camera


@dataclass(frozen=True)
class CameraSet:
    """Named dataset poses under one label, as the viewer's camera picker lists them."""

    label: str
    cameras: tuple[tuple[str, Camera], ...]


@dataclass(frozen=True)
class SceneContext:
    colmapScene: ColmapDataset
    width: int
    height: int
    camera_count: int
    initial_gaussian_count: int
    extent: float
    train_indices: list[int]
    test_indices: list[int]
    holdout_stride: int
    morton_min: np.ndarray
    morton_max: np.ndarray

    def camera_sets(self) -> tuple[CameraSet, CameraSet]:
        return split_camera_sets(self.colmapScene.images, self.test_indices, self.train_indices)


def split_camera_sets(images: Sequence[ColmapImageEntry], test_indices: Sequence[int],
                      train_indices: Sequence[int]) -> tuple[CameraSet, CameraSet]:
    """The held-out eval views, then the training views."""
    def named(indices: Sequence[int]) -> tuple[tuple[str, Camera], ...]:
        return tuple((images[i].image_name, images[i].camera) for i in indices)
    return CameraSet("eval", named(test_indices)), CameraSet("train", named(train_indices))


def holdout_split(camera_count: int, stride: int,
                  holdout: bool) -> tuple[list[int], list[int]]:
    """``(test, train)`` indices: every ``stride``-th camera is held out (3DGS
    llffhold); without holdout every camera is both trained and evaluated."""
    if not holdout:
        return list(range(camera_count)), list(range(camera_count))
    return ([i for i in range(camera_count) if i % stride == 0],
            [i for i in range(camera_count) if i % stride != 0])


def load_scene_context(args: argparse.Namespace, prepared: ColmapDataset | None = None) -> SceneContext:
    if prepared is None:
        colmap_scene = load_colmap(args.scene, width=args.width, height=args.height,
                                  downsample=args.downsample, init_opacity=args.init_opacity)
    else:
        # Stereo appends by replacing arrays in this dictionary. Each run starts from the imported cloud.
        colmap_scene = replace(prepared, initial_points=prepared.initial_points.copy())
    camera_count = len(colmap_scene.images)
    initial_count = colmap_scene.initial_points["means"].shape[0]
    width, height = colmap_scene.images[0].camera.image_size
    print(f"loaded {camera_count} cameras, {initial_count} init points, "
          f"scene_extent={colmap_scene.scene_extent:.3f}")

    stride = max(args.eval_holdout_stride, 1)
    test_indices, train_indices = holdout_split(camera_count, stride, bool(args.eval_holdout))
    if getattr(args, "stereo_init", False) and initial_count < args.max_splats:
        from trainer.data.stereo import DEFAULT_SEED_LIMIT, seed_from_stereo
        limit = min(DEFAULT_SEED_LIMIT, args.max_splats - initial_count)
        masks = getattr(args, "masks", None)
        initial_count += seed_from_stereo(colmap_scene, train_indices, limit=limit,
                                          masks_dir=Path(masks) if masks else None,
                                          invert_masks=bool(getattr(args, "invert_masks", False)))
    if args.eval_holdout:
        print(f"[eval] holdout protocol ON (stride={stride}): "
              f"{len(train_indices)} train / {len(test_indices)} test cameras. "
              f"Training samples train views only; PSNR reported on test views "
              f"as mean-of-per-image dB (paper metric).")
    else:
        print(f"[eval] holdout protocol OFF: training on all {camera_count} cameras, "
              f"reporting over all cameras (internal metric).")
    morton_min, morton_max = morton_bounds(colmap_scene.initial_points["means"])
    print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}")

    return SceneContext(
        colmapScene=colmap_scene,
        width=width,
        height=height,
        camera_count=camera_count,
        initial_gaussian_count=initial_count,
        extent=colmap_scene.scene_extent,
        train_indices=train_indices,
        test_indices=test_indices,
        holdout_stride=stride,
        morton_min=morton_min,
        morton_max=morton_max,
    )
