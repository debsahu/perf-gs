"""Per-image loss masks (--masks / --invert-masks).

The mask rides in the target's alpha (255 = keep). With LOSS_MASK compiled in, a dropped pixel gives
no color gradient (L1/L2 and SSIM), no blend weight, no removal-cost sample and no structure-aware
eta sample; alpha supervision is off. Without --masks nothing changes. CPU tests cover the loader
(naming, polarity, resize, counts), the auto-lens and stereo-seed filters, the metrics and the
config wiring; GPU tests (Vulkan) cover the shaders against an unmasked device.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.ndimage import correlate1d

from _pipeline_testkit import TestPipeline
from test_stereo_init import plane_scene  # noqa: F401  (fixture)
from trainer.config.config import (DensificationProperties, IterationContextualVariables,
                                   LossConfig, ShaderSpecialization)
from trainer.config.validation import ConfigError, validate_training_config
from trainer.data.colmap import (ColmapDataset, ColmapImageEntry, find_mask_path,
                                 load_mask_file)
from trainer.data.dataset import simple_camera
from trainer.data.targets import TargetDataset

SIZE = 32
HALF = SIZE // 2


# ---------------------------------------------------------------- loader (CPU)

def _png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def _scene(tmp_path: Path, names: list[str], size=(SIZE, SIZE)) -> ColmapDataset:
    rng = np.random.default_rng(3)
    entries = []
    for i, name in enumerate(names):
        _png(tmp_path / "images" / name, rng.integers(0, 256, (size[1], size[0], 3), np.uint8))
        camera = simple_camera(width=size[0], height=size[1], eye=(0.0, 0.0, 0.0),
                               target=(0.0, 0.0, 1.0))
        entries.append(ColmapImageEntry(i + 1, name, camera))
    return ColmapDataset(entries, tmp_path / "images", {})


def _left_half(value_left: int, value_right: int, w=SIZE, h=SIZE) -> np.ndarray:
    mask = np.full((h, w), value_right, np.uint8)
    mask[:, : w // 2] = value_left
    return mask


def test_mask_names_full_name_then_stem_and_subfolders(tmp_path):
    _png(tmp_path / "a.jpg.png", _left_half(255, 0))
    _png(tmp_path / "b.png", _left_half(255, 0))
    _png(tmp_path / "sub" / "c.jpg.png", _left_half(255, 0))
    _png(tmp_path / "d.png", _left_half(255, 0))
    _png(tmp_path / "d.jpg.png", _left_half(0, 255))
    assert find_mask_path(tmp_path, "a.jpg") == tmp_path / "a.jpg.png"
    assert find_mask_path(tmp_path, "b.jpg") == tmp_path / "b.png"
    assert find_mask_path(tmp_path, "sub/c.jpg") == tmp_path / "sub" / "c.jpg.png"
    assert find_mask_path(tmp_path, "c.jpg") is None          # the subfolder is part of the name
    assert find_mask_path(tmp_path, "d.jpg") == tmp_path / "d.jpg.png"   # full name wins
    assert find_mask_path(tmp_path, "e.jpg") is None


@pytest.mark.parametrize("invert", [False, True], ids=["keep255", "drop255"])
def test_polarity(tmp_path, invert):
    """The file is 255 on its left half. 255 = keep by default; --invert-masks makes it drop."""
    scene = _scene(tmp_path, ["v.png"])
    _png(tmp_path / "masks" / "v.png", _left_half(255, 0))
    targets = TargetDataset(scene, SIZE, SIZE, use_alpha=False,
                            masks_dir=tmp_path / "masks", invert_masks=invert)
    keep = targets.keep_mask(0)
    expect_left = not invert
    assert keep[:, :HALF].all() == expect_left and (~keep[:, :HALF]).all() == (not expect_left)
    assert keep[:, HALF:].all() == (not expect_left)
    alpha = targets.resident[0][..., 3]
    assert set(np.unique(alpha)) == {0, 255}
    np.testing.assert_array_equal(alpha == 255, keep)


def test_mask_rides_in_alpha_and_rgb_is_unchanged(tmp_path):
    """Same bytes in RGB and in resident memory as without masks; the float view stays RGB."""
    scene = _scene(tmp_path, ["v.png", "w.png"])
    _png(tmp_path / "masks" / "v.png", _left_half(0, 255))
    plain = TargetDataset(scene, SIZE, SIZE, use_alpha=False)
    masked = TargetDataset(scene, SIZE, SIZE, use_alpha=False, masks_dir=tmp_path / "masks")
    for i in range(2):
        a, b = plain.resident[i], masked.resident[i]
        assert a.dtype == b.dtype == np.uint8 and a.shape == b.shape == (SIZE, SIZE, 4)
        np.testing.assert_array_equal(a[..., :3], b[..., :3])
        assert masked.srgb[i].shape == (SIZE, SIZE, 3)
        np.testing.assert_array_equal(plain.srgb[i], masked.srgb[i])
    assert sum(x.nbytes for x in plain.resident) == sum(x.nbytes for x in masked.resident)
    assert (masked.resident[0][:, :HALF, 3] == 0).all() and (masked.resident[0][:, HALF:, 3] == 255).all()
    assert plain.keep_masks is None and not plain.masked


def test_missing_mask_keeps_every_pixel_and_is_counted(tmp_path, capsys):
    scene = _scene(tmp_path, ["v.png", "w.png", "x.png"])
    _png(tmp_path / "masks" / "w.png", _left_half(0, 255))
    targets = TargetDataset(scene, SIZE, SIZE, use_alpha=False, masks_dir=tmp_path / "masks")
    assert targets.mask_count == 1
    assert targets.keep_mask(0).all() and targets.keep_mask(2).all()
    assert not targets.keep_mask(1)[:, :HALF].any()
    out = capsys.readouterr().out
    assert "1 of 3 images have a loss mask" in out and "2 without one keep every pixel" in out


def test_no_mask_matching_any_image_is_an_error(tmp_path):
    scene = _scene(tmp_path, ["v.png"])
    _png(tmp_path / "masks" / "other.png", _left_half(0, 255))
    with pytest.raises(FileNotFoundError, match="no mask matched any"):
        TargetDataset(scene, SIZE, SIZE, use_alpha=False, masks_dir=tmp_path / "masks")


def test_masks_and_alpha_supervision_are_exclusive(tmp_path):
    scene = _scene(tmp_path, ["v.png"])
    _png(tmp_path / "masks" / "v.png", _left_half(0, 255))
    with pytest.raises(ValueError, match="exclusive"):
        TargetDataset(scene, SIZE, SIZE, use_alpha=True, masks_dir=tmp_path / "masks")
    from trainer.frontend.cli import parse_args
    args = parse_args(["--masks", str(tmp_path / "masks"), "--w-alpha", "1.0"])
    with pytest.raises(ConfigError, match="exclusive"):
        validate_training_config(args, n_cams=8, n_train=7, n_test=1)
    validate_training_config(parse_args(["--masks", "m"]), n_cams=8, n_train=7, n_test=1)
    validate_training_config(parse_args(["--w-alpha", "1.0"]), n_cams=8, n_train=7, n_test=1)
    with pytest.raises(ConfigError, match="needs --masks"):
        validate_training_config(parse_args(["--invert-masks"]), n_cams=8, n_train=7, n_test=1)


def test_resize_shrink_averages_then_thresholds(tmp_path):
    """2x shrink: each output pixel is a 2x2 block. 1 or 2 kept of 4 -> keep only from one half up."""
    block = np.array([[0, 0, 255, 0, 255, 0, 255, 255],
                      [0, 0, 0, 0, 255, 0, 255, 255]], np.uint8)     # kept: 0/4, 1/4, 2/4, 4/4
    mask = np.tile(block, (2, 1))
    _png(tmp_path / "m.png", mask)
    out = load_mask_file(tmp_path / "m.png", 4, 2)
    np.testing.assert_array_equal(out, np.array([[0, 0, 255, 255]] * 2, np.uint8))
    # 3 of 4 kept: still keep; inverted, 1 of 4 kept: drop
    _png(tmp_path / "n.png", np.array([[255, 255], [255, 0]], np.uint8))
    assert load_mask_file(tmp_path / "n.png", 1, 1)[0, 0] == 255
    assert load_mask_file(tmp_path / "n.png", 1, 1, invert=True)[0, 0] == 0
    # a nearest/first-pixel resize would read the top-left texel of the 1/4 block (kept) instead
    assert mask[0, 2] == 255 and load_mask_file(tmp_path / "m.png", 4, 2)[0, 1] == 0


def test_resize_enlarge_is_nearest_and_binary(tmp_path):
    mask = np.array([[255, 0, 255], [0, 255, 0]], np.uint8)
    _png(tmp_path / "m.png", mask)
    out = load_mask_file(tmp_path / "m.png", 9, 4)
    assert set(np.unique(out)) <= {0, 255}
    np.testing.assert_array_equal(out, np.kron(mask // 255, np.ones((2, 3), np.uint8)) * 255)
    # 2x2 -> 3x3: the center samples the source at (0.5, 0.5). Thresholded bilinear reads a 0.5 tie
    # (kept); nearest reads the top-left texel (dropped).
    _png(tmp_path / "c.png", np.array([[0, 255], [255, 0]], np.uint8))
    assert load_mask_file(tmp_path / "c.png", 3, 3)[1, 1] == 0


def test_resize_matches_the_image_size(tmp_path):
    """A mask at a different resolution than its image is resized to the training size."""
    scene = _scene(tmp_path, ["v.png"], size=(SIZE * 2, SIZE * 2))
    _png(tmp_path / "masks" / "v.png", _left_half(0, 255, w=SIZE // 2, h=SIZE // 2))
    targets = TargetDataset(scene, SIZE, SIZE, use_alpha=False, masks_dir=tmp_path / "masks")
    keep = targets.keep_mask(0)
    assert keep.shape == (SIZE, SIZE)
    assert not keep[:, :HALF].any() and keep[:, HALF:].all()


# ---------------------------------------------------------------- metrics, lens, stereo, config (CPU)

def test_psnr_masked_scores_kept_pixels_only():
    from trainer.common.metrics import psnr, psnr_masked
    r = np.zeros((4, 4, 3))
    t = np.zeros((4, 4, 3))
    t[:, :2] = 1.0                      # error only on the left half
    keep = np.zeros((4, 4), bool)
    keep[:, 2:] = True
    assert psnr_masked(r, t, keep) == 99.0
    assert psnr(r, t) == pytest.approx(3.0103, abs=1e-3)
    assert psnr_masked(r, t, None) is None and psnr_masked(r, t, np.zeros((4, 4), bool)) is None


def test_auto_lens_skips_landmarks_on_dropped_pixels():
    from trainer.data.lens_calibration import _sample_tracks
    h, w = 160, 240
    image = np.full((h, w, 3), 0.5, np.float32)
    xs = np.array([40.0, 60.0, 180.0, 200.0])
    features = {"xy": np.stack([xs, np.full(4, 80.0)], axis=1), "id": np.arange(4)}
    intrinsics = {"width": w, "height": h, "cx": w / 2, "cy": h / 2, "fx": 200.0, "fy": 200.0}
    keep = np.ones((h, w), bool)
    keep[:, : w // 2] = False
    assert list(_sample_tracks(features, intrinsics, image)[0]) == [0, 1, 2, 3]
    assert list(_sample_tracks(features, intrinsics, image, keep)[0]) == [2, 3]


def test_stereo_seeds_skip_dropped_pixels(plane_scene, tmp_path):
    from trainer.data.stereo import seed_from_stereo
    scene = plane_scene
    base = {k: v.copy() for k, v in scene.initial_points.items()}
    assert seed_from_stereo(scene, [0, 2, 3], limit=4096) > 0
    plain_x = scene.initial_points["means"][1:, 0].copy()
    scene.initial_points.clear()
    scene.initial_points.update({k: v.copy() for k, v in base.items()})
    # Drop the left half of every training image (the plane's world x < eye_x there).
    masks = tmp_path / "masks"
    for i in (0, 2, 3):
        name = scene.images[i].image_name
        h, w = cv2.imread(str(tmp_path / name)).shape[:2]
        _png(masks / name, _left_half(0, 255, w=w, h=h))
    added = seed_from_stereo(scene, [0, 2, 3], limit=4096, masks_dir=masks)
    masked_x = scene.initial_points["means"][1:, 0]
    assert added > 0 and len(masked_x) == added
    # Without masks many seeds lie left of the rightmost camera; with them none may.
    assert (plain_x < 0.2).mean() > 0.3
    assert masked_x.min() > 0.2


def test_shader_define_and_spec():
    from trainer.config.resolve import shader_specialization
    from trainer.frontend.cli import parse_args
    from trainer.frontend.factory import shader_defines
    assert shader_specialization(parse_args([])).loss_mask is False
    assert shader_specialization(parse_args(["--masks", "m"])).loss_mask is True
    assert "LOSS_MASK" not in (shader_defines(ShaderSpecialization()) or {})
    assert (shader_defines(ShaderSpecialization(loss_mask=True)) or {}).get("LOSS_MASK") == "1"


# ---------------------------------------------------------------- shaders (GPU)

def _device(masked: bool):
    from trainer.backend.compute.device import create_device
    try:
        return create_device(backend="vulkan", defines={"LOSS_MASK": "1"} if masked else None)
    except Exception as exc:  # pragma: no cover - device availability
        pytest.skip(f"vulkan unavailable: {exc}")


def _camera():
    return simple_camera(width=SIZE, height=SIZE, eye=(0.0, 0.0, 0.0),
                         target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), fovy_degrees=60.0)


def _splats(x, y, z, scale, opacity):
    n = len(x)
    rng = np.random.default_rng(5)
    return {"means": np.stack([x, y, z], axis=1).astype(np.float32),
            "log_scales": np.log(np.full((n, 3), scale, np.float32)),
            "rot_quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
            "opacities": np.full(n, opacity, np.float32),
            "color_logits": rng.normal(0.0, 1.0, (n, 3)).astype(np.float32)}


def _cover_scene():
    g = np.linspace(-0.8, 0.8, 5)
    xx, yy = np.meshgrid(g, g)
    return _splats(xx.ravel(), yy.ravel(), np.full(xx.size, 2.5), 0.25, 0.7)


def _target_rgb() -> np.ndarray:
    return np.random.default_rng(11).integers(0, 256, (SIZE, SIZE, 3), np.uint8)


def _keep_left() -> np.ndarray:
    keep = np.zeros((SIZE, SIZE), bool)
    keep[:, :HALF] = True
    return keep


def _rgba(rgb: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.concatenate([rgb, np.where(keep, 255, 0).astype(np.uint8)[..., None]], axis=-1)


def _gradient(masked: bool, loss: LossConfig, rgb, keep, scene=None, **pipeline_kwargs):
    p = TestPipeline(device=_device(masked), width=SIZE, height=SIZE, max_splats=64,
                     loss_config=loss, **pipeline_kwargs)
    assert p.loss_mask == masked
    p.set_primitive_from_numpy(**(scene or _cover_scene()))
    p.targets.set_image(_rgba(rgb, keep) if masked else rgb)
    p.forward_and_loss(_camera())
    p.runtime.wait_for_idle()
    return p, p.buffers.loss.image_gradient.to_numpy().astype(np.float64)


@pytest.mark.parametrize("l2", [0.0, 0.5], ids=["l1", "l1+l2"])
def test_l1_masked_pixels_have_zero_gradient_and_kept_pixels_are_unchanged(l2):
    rgb, keep = _target_rgb(), _keep_left()
    loss = LossConfig(ssim_weight=0.0, alpha_weight=0.0, l2_weight=l2)
    _, plain = _gradient(False, loss, rgb, keep)
    _, masked = _gradient(True, loss, rgb, keep)
    assert np.count_nonzero(plain[~keep, :3]) > 0.9 * (~keep).sum() * 3   # the fixture can tell
    assert (masked[~keep] == 0.0).all()
    np.testing.assert_array_equal(masked[keep, :3], plain[keep, :3])
    assert (masked[..., 3] == 0.0).all()


@pytest.mark.parametrize("ssim", [0.0, 0.2], ids=["l1", "ssim"])
def test_without_masks_target_alpha_does_not_gate_the_color_gradient(ssim):
    """No --masks: an RGBA target's alpha (here 0 on half the image) must not mask anything."""
    rgb, keep = _target_rgb(), _keep_left()
    loss = LossConfig(ssim_weight=ssim, alpha_weight=0.0)
    p = TestPipeline(device=_device(False), width=SIZE, height=SIZE, max_splats=64, loss_config=loss)
    p.set_primitive_from_numpy(**_cover_scene())
    gradients = []
    for target in (rgb, _rgba(rgb, keep)):
        p.targets.set_image(target)
        p.forward_and_loss(_camera())
        p.runtime.wait_for_idle()
        gradients.append(p.buffers.loss.image_gradient.to_numpy().astype(np.float64))
    np.testing.assert_array_equal(gradients[0], gradients[1])
    assert np.count_nonzero(gradients[1][~keep, :3]) > 0.9 * (~keep).sum() * 3


def test_masked_run_has_no_alpha_supervision():
    """--w-alpha is refused with --masks, but the shader also ignores it under LOSS_MASK."""
    rgb, keep = _target_rgb(), _keep_left()
    for ssim in (0.0, 0.2):
        _, masked = _gradient(True, LossConfig(ssim_weight=ssim, alpha_weight=1.0), rgb, keep)
        assert (masked[..., 3] == 0.0).all()
        _, plain = _gradient(False, LossConfig(ssim_weight=ssim, alpha_weight=1.0),
                             rgb, keep)   # unmasked: target alpha 1 vs render alpha < 1
        assert np.count_nonzero(plain[..., 3]) > 0


def test_ssim_masked_pixels_have_zero_gradient_and_far_pixels_are_unchanged():
    rgb, keep = _target_rgb(), _keep_left()
    loss = LossConfig(ssim_weight=0.2, alpha_weight=0.0)
    _, plain = _gradient(False, loss, rgb, keep)
    _, masked = _gradient(True, loss, rgb, keep)
    assert (masked[~keep] == 0.0).all()
    # Beyond the 5 px window radius from every dropped pixel nothing changes.
    far = np.zeros_like(keep)
    far[:, : HALF - 5] = True
    # Same math; the LOSS_MASK build may contract floating point differently (measured ~1e-5 rel).
    np.testing.assert_allclose(masked[far, :3], plain[far, :3], rtol=1e-4, atol=1e-6)
    near = keep & ~far
    assert np.abs(masked[near, :3] - plain[near, :3]).max() > 1e-2   # the boundary band does change
    assert np.count_nonzero(plain[~keep, :3]) > 0


def _ssim_loss(x: np.ndarray, y: np.ndarray, centers: np.ndarray) -> float:
    """sum over window centers in ``centers`` and channels of (1 - SSIM), 11x11 sigma 1.5, zero pad."""
    k = np.exp(-np.arange(-5, 6) ** 2 / (2 * 1.5 ** 2))
    k /= k.sum()

    def f(a):
        return correlate1d(correlate1d(a, k, axis=0, mode="constant"), k, axis=1, mode="constant")
    total = 0.0
    for c in range(3):
        a, b = x[..., c], y[..., c]
        mx, my = f(a), f(b)
        sx, sy, sxy = f(a * a) - mx * mx, f(b * b) - my * my, f(a * b) - mx * my
        s = ((2 * mx * my + 1e-4) * (2 * sxy + 9e-4)) / ((mx * mx + my * my + 1e-4) * (sx + sy + 9e-4))
        total += float(np.sum((1.0 - s)[centers]))
    return total


def test_ssim_gradient_matches_kept_window_loss_at_the_boundary():
    """With lambda_ssim = 1 the shader's dL/dC is the SSIM term alone. Near the boundary it must be the
    finite difference of the loss summed over KEPT window centers, not over all of them."""
    rgb, keep = _target_rgb(), _keep_left()
    loss = LossConfig(ssim_weight=1.0, alpha_weight=0.0)
    p, masked = _gradient(True, loss, rgb, keep)
    _, plain = _gradient(False, loss, rgb, keep)
    x = p.buffers.raster.output_image.to_numpy()[..., :3].astype(np.float64)
    y = rgb.astype(np.float64) / 255.0
    every = np.ones_like(keep)
    eps = 1e-4
    separated = 0
    for py, px in [(5, HALF - 1), (16, HALF - 1), (26, HALF - 2), (9, HALF - 2)]:
        for c in range(3):
            xp, xm = x.copy(), x.copy()
            xp[py, px, c] += eps
            xm[py, px, c] -= eps
            fd_kept = (_ssim_loss(xp, y, keep) - _ssim_loss(xm, y, keep)) / (2 * eps)
            fd_all = (_ssim_loss(xp, y, every) - _ssim_loss(xm, y, every)) / (2 * eps)
            assert masked[py, px, c] == pytest.approx(fd_kept, rel=2e-2, abs=2e-3)
            assert plain[py, px, c] == pytest.approx(fd_all, rel=2e-2, abs=2e-3)
            separated += abs(fd_kept - fd_all) > 10 * (2e-2 * abs(fd_kept) + 2e-3)
    assert separated >= 6, "fixture cannot tell kept-window SSIM from all-window SSIM"


def _track(p: TestPipeline, camera) -> None:
    contextual = IterationContextualVariables(
        hardness_value=1.0, sh_degree=0,
        densification_variables=DensificationProperties(track_blend_weight=True))
    with p.commands.recording("blend_weight"):
        p.rasterize(camera, contextual)
        p.record_backward(contextual)
    p.runtime.wait_for_idle()


@pytest.mark.parametrize("per_bucket", [True, False], ids=["per_bucket", "per_splat"])
def test_blend_weight_counts_kept_pixels_only(per_bucket):
    camera, keep = _camera(), _keep_left()
    one = _splats([0.0], [0.0], [2.0], 0.3, 0.6)
    results = {}
    for masked in (False, True):
        p = TestPipeline(device=_device(masked), width=SIZE, height=SIZE, max_splats=16,
                         use_flat_bucket_backward=per_bucket)
        p.set_primitive_from_numpy(**one)
        p.render(camera)
        alpha = p.read_output_image()[..., 3].astype(np.float64)
        p.targets.set_image(_rgba(_target_rgb(), keep) if masked else _target_rgb())
        p.densify_stats.clear()
        _track(p, camera)
        raw = p.buffers.densification.blend_weight.to_numpy().tobytes()
        results[masked] = (np.frombuffer(raw, np.int64)[:1] / 65536.0)[0], alpha
    (plain, alpha), (masked, _) = results[False], results[True]
    assert plain == pytest.approx(alpha.sum(), rel=1e-3)
    assert masked == pytest.approx(alpha[keep].sum(), rel=1e-3)
    assert alpha[~keep].sum() > 0.2 * alpha.sum()    # the splat straddles the boundary


def test_removal_cost_skips_dropped_pixels():
    from trainer.backend.densify.removal import accumulate_removal_cost
    camera = _camera()
    two = _splats([-0.5, 0.5], [0.0, 0.0], [2.0, 2.0], 0.12, 0.8)
    scores = {}
    for masked in (False, True):
        p = TestPipeline(device=_device(masked), width=SIZE, height=SIZE, max_splats=16)
        p.set_primitive_from_numpy(**_splats([-0.5], [0.0], [2.0], 0.12, 0.8))
        p.render(camera)
        first = p.read_output_image()[..., 3]
        keep = np.ones((SIZE, SIZE), bool)
        keep[first > 1e-3] = False           # drop exactly the first splat's footprint
        p.set_primitive_from_numpy(**two)
        p.targets.set_image(_rgba(_target_rgb(), keep) if masked else _target_rgb())
        p.densify_stats.clear()
        for step in range(16):
            with p.commands.recording("removal"):
                p.rasterize(camera)
                accumulate_removal_cost(p, step, 1.0)
            p.runtime.wait_for_idle()
        scores[masked] = np.frombuffer(p.buffers.densification.blend_weight.to_numpy().tobytes(),
                                       np.int64)[:2]
    assert (scores[False] > 0).all()
    assert scores[True][0] == 0 and scores[True][1] == scores[False][1]


def _field(p, keep_value: float):
    import slangpy as spy
    field = np.zeros((SIZE, SIZE, 4), np.float32)
    field[..., 0] = 0.25
    field[..., 3] = keep_value
    tex = p.device.create_texture(type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
                                  width=SIZE, height=SIZE,
                                  usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access)
    tex.copy_from_numpy(field)
    return tex


@pytest.mark.parametrize(("masked", "keep_value", "observed"),
                         [(True, 0.0, 0), (True, 1.0, 1), (False, 0.0, 1)],
                         ids=["masked-drop", "masked-keep", "unmasked-ignores-w"])
def test_structure_score_skips_dropped_samples(masked, keep_value, observed):
    p = TestPipeline(device=_device(masked), width=SIZE, height=SIZE, max_splats=16)
    p.set_primitive_from_numpy(**_splats([0.0], [0.0], [3.0], 0.3, 0.9))
    p.structure_densifier.clear_scores()
    camera = _camera()
    with p.commands.recording("score"):
        p.rasterize(camera)
        p.structure_densifier.accumulate_scores(_field(p, keep_value), p.rasterizer.camera_buffer, pass_idx=0)
    p.runtime.wait_for_idle()
    n_total = np.frombuffer(p.buffers.eta.n_total.to_numpy().tobytes(), np.uint32)[0]
    assert n_total == observed


def test_structure_field_carries_the_target_alpha():
    from trainer.backend.densify.structure_tensor import StructureTensor
    p = TestPipeline(device=_device(True), width=SIZE, height=SIZE, max_splats=16)
    keep = _keep_left()
    texture = p.targets.create(_rgba(_target_rgb(), keep))
    field = StructureTensor(p.device, SIZE, SIZE).build_textures({0: texture})[0].to_numpy()
    np.testing.assert_array_equal(field[..., 3].astype(np.float32) == 1.0, keep)
    assert (field[~keep, 3] == 0).all()
