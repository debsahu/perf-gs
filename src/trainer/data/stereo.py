# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Add colored Gaussian seeds from calibrated stereo matches in the training split."""

from collections.abc import Sequence
from functools import cache

import cv2
import numpy as np

from trainer.data.colmap import ColmapDataset, _srgb_colors_to_sh0, load_colmap_image, load_colmap_mask
from trainer.data.parallel import map_ordered

DEFAULT_SEED_LIMIT = 256_000


def _intrinsics(projection, width, height):
    # COLMAP measures pixel centers at half-integers, OpenCV at integers.
    return np.array([[projection[0, 0] * width / 2, 0, (projection[0, 2] + 1) * width / 2 - 0.5],
                     [0, projection[1, 1] * height / 2, (projection[1, 2] + 1) * height / 2 - 0.5],
                     [0, 0, 1]], dtype=np.float64)


def _matched_pixels(left, right):
    gray, other = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (left, right))
    height, width = gray.shape
    span = min(96, ((width - 5) // 16) * 16)
    matcher = cv2.StereoSGBM_create(numDisparities=span, blockSize=5, P1=200, P2=800, disp12MaxDiff=1,
                                     preFilterCap=31, uniquenessRatio=10, speckleWindowSize=40, speckleRange=2)
    forward = matcher.compute(gray, other).astype(np.float32) / 16
    reverse = matcher.compute(np.fliplr(other), np.fliplr(gray))[:, ::-1].astype(np.float32) / 16
    y, x = np.mgrid[3:height-3:3, 3:width-3:3]
    disparity = forward[y, x]
    right_x = np.clip(np.rint(x - disparity).astype(int), 0, width - 1)
    keep = (disparity > 3) & (abs(disparity - reverse[y, right_x]) < 1.2) & (gray[y, x] > 3)
    keep &= np.max(abs(left[y, x].astype(np.float32) - right[y, right_x]), axis=2) < 32
    return x[keep], y[keep], disparity[keep]


def seed_from_stereo(dataset: ColmapDataset, train_indices: Sequence[int], limit: int = DEFAULT_SEED_LIMIT,
                     masks_dir=None, invert_masks: bool = False) -> int:
    """Append up to `limit` seeds supported by three training views. Return the number added.
    With loss masks (``masks_dir``), a seed whose projection lands on a dropped pixel of any of
    its three views is discarded: no masked content is seeded."""
    entries = [dataset.images[i] for i in train_indices]
    points = dataset.initial_points
    if len(entries) < 3 or not len(points["means"]) or limit <= 0:
        return 0
    width, height = entries[0].camera.image_size
    size = (672, max(96, round(672 * height / width)))
    rotations, translations = np.split(np.array([e.camera.view_matrix[:3] for e in entries], np.float64), [3], axis=2)
    centers = -(rotations.transpose(0, 2, 1) @ translations)[:, :, 0]
    intrinsics = [_intrinsics(entry.camera.proj_matrix, *size) for entry in entries]
    extent = max(dataset.scene_extent, 1e-6)

    @cache
    def image(index):
        return np.rint(load_colmap_image(entries[index], dataset.image_dir, *size) * 255).astype(np.uint8)

    @cache
    def keep(index):
        if masks_dir is None:
            return None
        mask = load_colmap_mask(entries[index], masks_dir, *size, invert_masks)
        return None if mask is None else mask > 0

    def kept(index, world):
        mask = keep(index)
        if mask is None:
            return np.ones(len(world), bool)
        local = (world - centers[index]) @ rotations[index].T
        k = intrinsics[index]
        uv = np.rint(local[:, :2] / np.maximum(local[:, 2:], 1e-6) * [k[0, 0], k[1, 1]] + k[:2, 2])
        inside = (local[:, 2] > 0) & (uv >= 0).all(axis=1) & (uv < size).all(axis=1)
        ok = np.ones(len(world), bool)   # outside the view: no mask there
        ui, vi = uv[inside].astype(int).T
        ok[inside] = mask[vi, ui]
        return ok

    def rectify(index, rotation, projection, axis):
        maps = cv2.initUndistortRectifyMap(intrinsics[index], None, rotation, projection, size, cv2.CV_32FC1)
        return cv2.remap(image(index), *maps, cv2.INTER_LINEAR).swapaxes(0, axis)  # SGBM searches along rows.

    def triangulate(anchor):
        offsets = (centers - centers[anchor]) @ rotations[anchor].T
        lateral = np.max(abs(offsets[:, :2]), axis=1)
        alignment = rotations[:, 2] @ rotations[anchor, 2]
        candidates = np.flatnonzero((lateral > 0.01 * extent) & (lateral > abs(offsets[:, 2])) & (alignment > 0.86))
        if len(candidates) < 2:
            return None
        # Two nearby, transverse training baselines: one for stereo, one for color verification.
        cost = abs(lateral[candidates] / extent - 0.06) + 2 * (1 - alignment[candidates])
        partner, check = candidates[np.argsort(cost)[:2]]
        axis = np.argmax(abs(offsets[partner, :2]))
        a, b = (anchor, partner) if offsets[partner, axis] > 0 else (partner, anchor)
        baseline = rotations[b] @ (centers[a] - centers[b])
        ra, rb, pa, pb, *_ = cv2.stereoRectify(intrinsics[a], None, intrinsics[b], None, size,
                                                rotations[b] @ rotations[a].T, baseline[:, None], alpha=-1)
        axis = np.argmax(abs(pb[:2, 3]))
        if not np.isfinite(pa).all() or not np.isfinite(pb).all() or pb[axis, 3] >= 0:
            return None
        left, right = rectify(a, ra, pa, axis), rectify(b, rb, pb, axis)
        u, v, d = _matched_pixels(left, right)
        z = -pb[axis, 3] / d
        x, y = (u, v) if axis == 0 else (v, u)
        rectified = np.stack([(x - pa[0, 2]) * z / pa[0, 0], (y - pa[1, 2]) * z / pa[1, 1], z], axis=1)
        world = (rectified @ ra) @ rotations[a] + centers[a]
        # Occluded or inconsistent matches fail the third camera's color reprojection.
        third = (world - centers[check]) @ rotations[check].T
        k = intrinsics[check]
        uv = np.rint(third[:, :2] / np.maximum(third[:, 2:], 1e-6) * [k[0, 0], k[1, 1]] + k[:2, 2])
        seen = (third[:, 2] > 0) & (uv > 0).all(axis=1) & (uv < np.subtract(size, 1)).all(axis=1) & (z < 20 * extent)
        tx, ty = uv[seen].astype(int).T
        rgb = left[v, u]
        seen[seen] = np.mean(abs(image(check)[ty, tx].astype(np.float32) - rgb[seen]), axis=1) < 18
        if masks_dir is not None:
            seen &= kept(a, world) & kept(b, world) & kept(check, world)
        return world[seen], rgb[seen], z[seen] * 1.5 / pa[0, 0]

    anchors = np.linspace(0, len(entries) - 1, min(28, len(entries)), dtype=int)
    batches = [batch for batch in map_ordered(triangulate, anchors) if batch is not None and len(batch[0])]
    if not batches:
        return 0
    xyz, rgb, footprint = (np.concatenate(group) for group in zip(*batches))
    sample = np.random.default_rng(0).choice(len(xyz), min(limit, len(xyz)), replace=False)
    xyz, rgb = xyz[sample].astype(np.float32), rgb[sample]
    radius = np.maximum(footprint[sample], 1e-5)
    additional = dict(means=xyz, log_scales=np.repeat(np.log(radius)[:, None], 3, axis=1),
                      rot_quats=np.tile([1, 0, 0, 0], (len(xyz), 1)),
                      opacities=np.full(len(xyz), 0.1),
                      color_logits=_srgb_colors_to_sh0(rgb.astype(np.float32) / 255))
    points.update({k: np.concatenate([points[k], v.astype(np.float32)]) for k, v in additional.items()})
    return len(xyz)
