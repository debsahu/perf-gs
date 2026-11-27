# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Depth and normal prior loading: discovery, formats, encodings, the sign check, the grid, the
streamed residency, and the CLI refusals (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from trainer.data import priors as P

FIX = Path(__file__).with_name("fixtures")
FORMATS = FIX / "prior_formats"


def _write_png(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image[..., ::-1] if image.ndim == 3 else image)
    return path


def _toward_normals(w, h, intr, rng=None, tilt=(0.3, -0.2, -1.0)):
    """A facing normal field for a pinhole camera: a plane n = normalize(tilt), flipped per pixel
    to face its own ray."""
    fx, fy, cx, cy = intr
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    r = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    n = np.broadcast_to(np.array(tilt, float), r.shape).copy()
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n *= -np.sign((n * r).sum(-1, keepdims=True))
    return n


# ------------------------------------------------------------------------------ discovery

def test_discovery_full_name_then_stem_and_subfolders(tmp_path):
    d = tmp_path / "pri"
    _write_png(d / "a.jpg.png", np.zeros((4, 4), np.uint16))
    _write_png(d / "sub" / "b.png", np.zeros((4, 4), np.uint16))
    assert P.find_prior_path(d, "a.jpg") == d / "a.jpg.png"
    assert P.find_prior_path(d, "sub/b.jpg") == d / "sub" / "b.png"
    assert P.find_prior_path(d, "c.jpg") is None


def test_two_priors_for_one_image_is_an_error(tmp_path):
    d = tmp_path / "pri"
    _write_png(d / "a.png", np.zeros((4, 4), np.uint16))
    (d / "a.tiff").write_bytes(b"II*\x00")
    with pytest.raises(P.PriorError, match="more than one prior"):
        P.find_prior_path(d, "a.jpg")


def test_zero_matches_is_an_error_and_partial_is_counted(tmp_path, capsys):
    d = tmp_path / "pri"
    _write_png(d / "a.png", np.zeros((4, 4), np.uint16))
    with pytest.raises(P.PriorError, match="matches any"):
        P.match_priors(d, ["x.jpg", "y.jpg"], "depth")
    paths = P.match_priors(d, ["a.jpg", "y.jpg"], "depth")
    assert paths[0] is not None and paths[1] is None
    assert "1 of 2 images have a prior" in capsys.readouterr().out


# ------------------------------------------------------------------------------ formats

def test_depth_formats_decode_to_millimetres():
    """uint16 PNG passes through; float32 TIFF (plain and deflate + predictor 3) quantizes to mm."""
    ref = np.load(FORMATS / "depth_reference.npy")
    want = P.quantize_depth_m(ref)
    assert (want == 0).any() and (want > 0).sum() > 100
    for name in ("depth_png-quantized.png", "depth_tiff-f32.tiff", "depth_tiff-f32-deflate.tiff"):
        got = P.decode_depth((FORMATS / name).read_bytes(), name)
        assert got.dtype == np.uint16 and np.array_equal(got, want), name


def test_normal_formats_decode_to_slam_codes():
    ref = np.load(FORMATS / "normal_reference.npy")
    want = P.encode_normal_codes(ref)
    assert (want[0, 0] == 128).all()                         # the invalid (0,0,0) sentinel
    for name in ("normal_png-quantized.png", "normal_tiff-f32.tiff", "normal_tiff-f32-deflate.tiff"):
        got = P.decode_normal((FORMATS / name).read_bytes(), "slam", name)
        assert got.dtype == np.uint8 and np.array_equal(got, want), name


def test_depth_quantization_saturates_and_drops_invalid():
    got = P.quantize_depth_m(np.array([[0.0, -1.0, np.nan, 0.0004, 0.0005, 70.0, 1.2345]]))
    assert got.tolist() == [[0, 0, 0, 0, 0, 65535, 1234]]       # 0.5 mm rounds half to even


def test_a_non_prior_file_is_refused(tmp_path):
    with pytest.raises(P.PriorError, match="not a PNG or TIFF"):
        P.decode_depth(b"\xff\xd8\xff\xe0 jpeg")


def test_eight_bit_depth_png_is_refused(tmp_path):
    p = _write_png(tmp_path / "d.png", np.full((4, 4), 7, np.uint8))
    with pytest.raises(P.PriorError, match="16-bit"):
        P.decode_depth(p.read_bytes())


# ------------------------------------------------------------------------------ T11 encodings

def _private_fixtures() -> Path:
    """Crops of real priors are not shipped in this repository; a caller that has them names their folder in
    PERFGS_PRIVATE_FIXTURES (it holds real_priors.json and the files it lists). Skipped otherwise."""
    import os
    folder = os.environ.get("PERFGS_PRIVATE_FIXTURES")
    if not folder or not (Path(folder) / "real_priors.json").is_file():
        pytest.skip("real prior crops not available (PERFGS_PRIVATE_FIXTURES)")
    return Path(folder)


def _real():
    folder = _private_fixtures()
    meta = json.loads((folder / "real_priors.json").read_text())
    return meta, (folder / meta["normals_spirula"]["file"]).read_bytes()


def test_spirula_black_is_invalid_and_slam_128_is_invalid():
    codes = np.array([[[0, 0, 0], [128, 128, 128], [200, 30, 128]]], np.uint8)
    png = cv2.imencode(".png", codes[..., ::-1])[1].tobytes()
    spirula = P.decode_normal(png, "spirula")
    slam = P.decode_normal(png, "slam")
    n_sp = P.decode_normal_codes(spirula)
    n_sl = P.decode_normal_codes(slam)
    valid = lambda n: np.linalg.norm(n, axis=-1) > 0.5            # noqa: E731
    assert valid(n_sp).tolist() == [[False, False, True]]
    assert valid(n_sl).tolist() == [[True, False, True]]         # slam reads black as (-1,-1,-1)
    assert np.linalg.norm(n_sl[0, 0]) == pytest.approx(np.sqrt(3), rel=1e-6)


def test_real_spirula_map_passes_as_spirula_and_fails_read_as_slam():
    """T11: the encoding flag matters on real data, and the load check catches the wrong one."""
    meta, data = _real()
    K = meta["normals_spirula"]["intrinsics"]
    intr = (K["fx"], K["fy"], K["cx"], K["cy"])
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    wh = (raw.shape[1], raw.shape[0])
    good = P.measure_normals(P.decode_normal(data, "spirula"), intr, wh)
    assert good.passed(), good
    # the crop is fully valid, so add the black rejected pixels a real map carries at its border
    holed = raw.copy()
    holed[:, :24] = 0
    png = cv2.imencode(".png", holed)[1].tobytes()
    right = P.measure_normals(P.decode_normal(png, "spirula"), intr, wh)
    wrong = P.measure_normals(P.decode_normal(png, "slam"), intr, wh)
    assert right.passed(), right
    assert not wrong.passed() and wrong.valid > right.valid, (wrong, right)


# ------------------------------------------------------------------------------ T12 sign check

def _check(codes, intr, wh):
    return P.check_normal_priors([("view", codes, intr)], wh)


def test_away_from_camera_map_is_refused():
    intr, wh = (20.0, 20.0, 16.0, 12.0), (32, 24)
    n = _toward_normals(*wh, intr)
    _check(P.encode_normal_codes(n), intr, wh)
    with pytest.raises(P.PriorError, match="AWAY from the camera"):
        _check(P.encode_normal_codes(-n), intr, wh)


def test_sign_check_uses_the_pixel_ray_not_the_optical_axis():
    """T12: a grazing, off-axis map whose normals face the camera along their rays but have
    n.z > 0 on most pixels passes; an n.z test would refuse it."""
    intr, wh = (8.0, 8.0, -20.0, 12.0), (32, 24)                # principal point far left: wide, off-axis
    n = _toward_normals(*wh, intr, tilt=(-1.0, 0.1, 0.3))
    assert np.mean(n[..., 2] > 0) > 0.5                        # the fixture can tell the two rules apart
    _check(P.encode_normal_codes(n), intr, wh)


def test_mostly_invalid_map_warns_but_loads(capsys):
    """A partially valid map is legitimate: a warning, not an error; a map with no valid pixel is refused."""
    intr, wh = (20.0, 20.0, 16.0, 12.0), (32, 24)
    codes = P.encode_normal_codes(_toward_normals(*wh, intr))
    codes[:, :20] = 128
    (c,) = _check(codes, intr, wh)
    assert c.valid < P.CHECK_MIN_VALID and "WARNING" in capsys.readouterr().out
    codes[:] = 128
    with pytest.raises(P.PriorError, match="no valid pixel"):
        _check(codes, intr, wh)


def _spirula_png_with_black(wh, intr, fraction, rng):
    """A camera-facing map in spirula encoding (black = invalid) with ``fraction`` of its pixels black."""
    codes = P.encode_normal_codes(_toward_normals(*wh, intr))
    black = rng.random(codes.shape[:2]) < fraction
    codes[black] = 0
    return codes, black


def test_spirula_map_with_few_black_pixels_is_refused_as_slam():
    """B1: under 1% black (p99 cannot see it), read as slam: the (-1,-1,-1) pixels are longer than 1.1."""
    rng = np.random.default_rng(4)
    intr, wh = (20.0, 20.0, 16.0, 12.0), (64, 48)
    codes, black = _spirula_png_with_black(wh, intr, 0.006, rng)
    assert 0 < black.mean() < 0.01
    png = cv2.imencode(".png", codes[..., ::-1])[1].tobytes()
    as_slam = P.measure_normals(P.decode_normal(png, "slam"), intr, wh)
    assert abs(as_slam.length_p99 - 1.0) <= P.CHECK_LENGTH_TOL        # the percentile test alone would pass it
    with pytest.raises(P.PriorError, match="spirula map read as slam"):
        _check(P.decode_normal(png, "slam"), intr, wh)
    _check(P.decode_normal(png, "spirula"), intr, wh)


def test_dataset_decodes_spirula_black_to_the_slam_sentinel(tmp_path):
    """B2: a spirula PNG already on the common grid is re-encoded, not kept as file bytes, so its black
    pixels reach the GPU as (128, 128, 128)."""
    rng = np.random.default_rng(5)
    intr, wh = (20.0, 20.0, 16.0, 12.0), (32, 24)
    names = ["a.jpg", "b.jpg"]
    blacks = []
    for n in names:
        codes, black = _spirula_png_with_black(wh, intr, 0.004, rng)
        blacks.append(black)
        _write_png(tmp_path / "normal" / f"{Path(n).stem}.png", codes)
    ds = P.PriorDataset(names, wh, [intr] * 2, normal_dir=tmp_path / "normal", normal_encoding="spirula")
    assert ds.normal_hw == (24, 32)                                   # on the common grid: the keep-bytes path
    for i, black in enumerate(blacks):
        assert black.any()
        _, n = ds.entry(i)
        assert (n[black][:, :3] == 128).all()
        assert not (n[..., :3] == 0).all(-1).any()
    ds.close()


# ------------------------------------------------------------------------------ T13 size policy

def test_prior_texel_is_nearest_at_the_pixel_center():
    """798x1064 onto 1512x2016: the corners and the center hit the expected texels."""
    image, prior = (1512, 2016), (798, 1064)
    assert P.prior_texel((0, 0), image, prior) == (0, 0)
    assert P.prior_texel((1511, 2015), image, prior) == (797, 1063)
    assert P.prior_texel((1511, 0), image, prior) == (797, 0)
    assert P.prior_texel((0, 2015), image, prior) == (0, 1063)
    assert P.prior_texel((756, 1008), image, prior) == (399, 532)
    # floor((u + 0.5) * pw / W) everywhere on a row
    u = np.arange(1512)
    got = [P.prior_texel((int(x), 0), image, prior)[0] for x in u]
    assert got == np.floor((u + 0.5) * 798 / 1512).astype(int).tolist()


def test_aspect_mismatch_is_refused():
    P.check_aspect((1064, 798), (6048, 8064), "ok")
    P.check_aspect((798, 1064), (8064, 6048), "landscape")
    with pytest.raises(P.PriorError, match="aspect differs"):
        P.check_aspect((798, 1064), (6048, 8064), "rotated")


def test_common_grid_is_the_densest_prior_capped_at_the_image():
    assert P.common_grid([(1064, 798)], (1512, 2016)) == (1064, 798)
    assert P.common_grid([(1064, 798), (532, 399)], (1512, 2016)) == (1064, 798)
    assert P.common_grid([(1064, 798), (798, 1064)], (1512, 2016)) == (1064, 798)   # a landscape view
    assert P.common_grid([(4000, 3000)], (1512, 2016)) == (2016, 1512)


def test_resize_is_nearest_and_never_smears_the_sentinel():
    rng = np.random.default_rng(0)
    depth = rng.integers(1000, 3000, (10, 8)).astype(np.uint16)
    depth[rng.random((10, 8)) < 0.3] = 0
    up = P.nearest_resize(depth, (25, 20))
    assert set(np.unique(up)) <= set(np.unique(depth))           # no interpolated codes
    assert np.isclose((up == 0).mean(), (depth == 0).mean(), atol=0.06)
    bilinear = cv2.resize(depth, (20, 25), interpolation=cv2.INTER_LINEAR)
    assert not set(np.unique(bilinear)) <= set(np.unique(depth))   # the discriminator works


# ------------------------------------------------------------------------------ dataset, T14

def _dataset_dir(tmp_path, names, wh=(32, 24), grid=(16, 12)):
    intr = (20.0, 20.0, wh[0] / 2, wh[1] / 2)
    normals = P.encode_normal_codes(_toward_normals(grid[0], grid[1], (intr[0] * grid[0] / wh[0],) * 2
                                                    + (grid[0] / 2, grid[1] / 2)))
    for i, name in enumerate(names):
        _write_png(tmp_path / "depth" / f"{Path(name).stem}.png", np.full(grid[::-1], 1000 + i, np.uint16))
        _write_png(tmp_path / "normal" / f"{Path(name).stem}.png", normals)
    return intr


def test_dataset_holds_png_bytes_and_decodes_on_demand(tmp_path):
    names = [f"img{i}.jpg" for i in range(5)]
    intr = _dataset_dir(tmp_path, names)
    ds = P.PriorDataset(names, (32, 24), [intr] * 5, depth_dir=tmp_path / "depth",
                        normal_dir=tmp_path / "normal")
    assert ds.counts == {"depth": 5, "normal": 5, "images": 5}
    assert ds.depth_hw == (12, 16) and ds.normal_hw == (12, 16)
    assert ds.cache_stats()["decodes"] == 0                     # nothing decoded at startup
    d, n = ds.entry(3)
    assert d.dtype == np.uint16 and d.shape == (12, 16) and (d == 1003).all()
    assert n.dtype == np.uint8 and n.shape == (12, 16, 4) and (n[..., 3] == 255).all()
    assert ds.host_bytes()["prior_encoded_bytes"] > 0
    ds.close()


def test_training_loop_prefetches_priors_in_sampler_order(tmp_path):
    """T14: the loop pins the current view and the next --target-prefetch draws of the priors,
    in the sampler's order, and selects this view's priors; the cache decodes and hits."""
    from trainer.frontend.app import TrainingLoop
    from trainer.frontend.sampling import CameraSampler
    names = [f"img{i}.jpg" for i in range(10)]
    intr = _dataset_dir(tmp_path, names)
    ds = P.PriorDataset(names, (32, 24), [intr] * 10, depth_dir=tmp_path / "depth",
                        normal_dir=tmp_path / "normal", decode_workers=1)
    windows, selected = [], []
    original = ds.prefetch
    ds.prefetch = lambda idx: (windows.append(list(idx)), original(idx))[1]
    sampler = CameraSampler("ours", list(range(10)), [], 1, np.random.default_rng(1))
    reference = CameraSampler("ours", list(range(10)), [], 1, np.random.default_rng(1))
    geometry = SimpleNamespace(select=lambda d, n: selected.append(int(d[0, 0]) - 1000), active=None)
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.sampler, loop.priors, loop.geom_from = sampler, ds, 3
    loop.args = SimpleNamespace(target_prefetch=3)
    loop.target_textures = {i: i for i in range(10)}
    loop.runtime = SimpleNamespace(
        pipeline=SimpleNamespace(targets=SimpleNamespace(select=lambda t: None), geometry=geometry),
        queue_limiter=SimpleNamespace(before_submit=lambda: None))
    order = [reference.next() for _ in range(12)]
    actives = []
    for step in range(8):
        camera = sampler.next()
        loop._prepare_iteration(SimpleNamespace(camera_index=camera, step_number=step + 1))
        assert windows[-1] == order[step:step + 4]
        actives.append(geometry.active)
    assert selected == order[:8]
    assert actives == [False, False, True, True, True, True, True, True]   # --geom-from 3
    stats = ds.cache_stats()
    assert stats["decodes"] > 0 and stats["hits"] > 0
    ds.close()


def test_dataset_runs_the_sign_check(tmp_path):
    names = ["a.jpg", "b.jpg"]
    intr = _dataset_dir(tmp_path, names)
    flipped = 255 - cv2.imread(str(tmp_path / "normal" / "b.png"))
    cv2.imwrite(str(tmp_path / "normal" / "b.png"), flipped)
    with pytest.raises(P.PriorError, match="AWAY"):
        P.PriorDataset(names, (32, 24), [intr] * 2, normal_dir=tmp_path / "normal")


def test_landscape_view_prior_is_stretched_like_its_photo(tmp_path):
    """A landscape view of a portrait run: its prior passes the aspect check against its own size
    and lands on the common grid."""
    names = ["p.jpg", "l.jpg"]
    _write_png(tmp_path / "depth" / "p.png", np.full((8, 6), 1500, np.uint16))
    _write_png(tmp_path / "depth" / "l.png", np.full((6, 8), 2500, np.uint16))
    ds = P.PriorDataset(names, (12, 16), [(10.0, 10.0, 6.0, 8.0)] * 2, native_sizes=[(60, 80), (80, 60)],
                        depth_dir=tmp_path / "depth")
    assert ds.depth_hw == (8, 6)
    assert ds.entry(1)[0].shape == (8, 6) and (ds.entry(1)[0] == 2500).all()
    with pytest.raises(P.PriorError, match="aspect"):
        P.PriorDataset(names, (12, 16), [(10.0, 10.0, 6.0, 8.0)] * 2, native_sizes=[(60, 80), (60, 80)],
                       depth_dir=tmp_path / "depth")


# ------------------------------------------------------------------------------ CLI and refusals

def _args(*argv):
    from trainer.frontend.cli import parse_args
    return parse_args(list(argv))


def test_cli_defaults_are_off():
    from trainer.config.resolve import geometry_config, shader_specialization
    args = _args()
    g = geometry_config(args)
    assert not g.enabled and g.depth_source == "center" and g.depth_space == "disparity"
    assert g.normal_loss == "l1" and g.normal_encoding == "slam" and g.min_alpha == 0.5
    assert not shader_specialization(args).geom_priors


@pytest.mark.parametrize("argv,message", [
    (["--depth-loss-weight", "1"], "needs --depth-dir"),
    (["--normal-loss-weight", "0.2"], "needs --normal-dir"),
    (["--depth-dir", "d"], "decoded for nothing"),
    (["--normal-dir", "n"], "decoded for nothing"),
    (["--depth-normal-weight", "0.05", "--stbn-pixel-fraction", "0.5"], "STBN"),
    (["--flatten-loss-weight", "-1"], ">= 0"),
    (["--geom-units-per-meter", "0", "--flatten-loss-weight", "1"], "units-per-meter"),
])
def test_refusals(argv, message):
    from trainer.config.validation import ConfigError, validate_geometry
    with pytest.raises(ConfigError, match=message):
        validate_geometry(_args(*argv))


def test_flags_reach_the_config_and_the_define():
    from trainer.config.resolve import geometry_config, shader_specialization
    args = _args("--depth-dir", "d", "--depth-loss-weight", "1", "--depth-source", "plane",
                 "--depth-loss-space", "ssi", "--normal-loss", "cos", "--geom-from", "100")
    g = geometry_config(args)
    assert (g.depth_weight, g.depth_source, g.depth_space, g.normal_loss, g.start_iter) == (
        1.0, "plane", "ssi", "cos", 100)
    assert shader_specialization(args).geom_priors
    assert not shader_specialization(_args("--flatten-loss-weight", "1")).geom_priors


def test_pipeline_without_the_define_refuses_the_weights():
    from trainer.frontend.app import require_geometry
    with pytest.raises(RuntimeError, match="GEOM_PRIORS"):
        require_geometry(_args("--depth-normal-weight", "0.05"), SimpleNamespace(geom_priors=False), None)
    require_geometry(_args("--flatten-loss-weight", "1"), SimpleNamespace(geom_priors=False), None)
