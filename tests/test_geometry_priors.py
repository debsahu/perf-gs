# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""GPU tests of the depth-prior, normal-prior, depth-normal and flatten losses.

Every comparison first asserts that the set it compares over is non-empty. The loss gradient is
checked against central finite differences of an independent float64 reference
(tests/geom_reference.py); the backward against weights read back from the forward; the
feature chain against host finite differences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import geom_reference as ref
from _pipeline_testkit import TestPipeline
from trainer.backend.compute.device import create_device
from trainer.backend.render.geometry import FIXED_SCALE, SEED_NAMES
from trainer.config.config import GeometryConfig, LossConfig
from trainer.data.dataset import simple_camera

W = H = 16
FIXTURE = Path(__file__).with_name("fixtures") / "normals_from_depth_slanted_plane.json"


# One device per define set for the whole module: a device per test held ~15 GB of host memory on a
# 16 GB Linux box and let the next test file be OOM-killed. Created before any test starts, so the
# per-test device cleanup in conftest.py leaves them alone.
_DEFINE_SETS = ((), ("GEOM_PRIORS",), ("GEOM_PRIORS", "LOSS_MASK"), ("GEOM_PRIORS", "STRIDE2"))
_DEFINE_VALUES = {"GEOM_PRIORS": ("GEOM_PRIORS", "1"), "LOSS_MASK": ("LOSS_MASK", "1"),
                  "STRIDE2": ("BACKWARD_PIXEL_STRIDE", "2u")}
_DEVICES: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _shared_devices():
    for key in _DEFINE_SETS:
        try:
            _DEVICES[key] = create_device(backend="vulkan",
                                          defines=dict(_DEFINE_VALUES[k] for k in key) or None)
        except Exception as exc:   # noqa: BLE001
            _DEVICES[key] = exc
    yield
    for device in _DEVICES.values():
        if not isinstance(device, Exception):
            device.close()
    _DEVICES.clear()


@pytest.fixture(autouse=True)
def _collect_pipelines():
    """Free each test's pipelines (and their GPU memory) before the next test allocates its own."""
    yield
    import gc
    gc.collect()


def _device(**defines):
    key = tuple(k for k in ("GEOM_PRIORS", "LOSS_MASK", "STRIDE2") if defines.get(k))
    device = _DEVICES.get(key)
    if device is None or isinstance(device, Exception):
        pytest.skip(f"vulkan unavailable: {device}")
    return device


def _camera(w=W, h=H, fov=70.0):
    return simple_camera(width=w, height=h, eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0),
                         up=(0.0, 1.0, 0.0), fovy_degrees=fov)


def _pipeline(device, config: GeometryConfig, w=W, h=H, **kwargs) -> TestPipeline:
    return TestPipeline(device=device, width=w, height=h, max_splats=16, geometry_config=config,
                        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0), **kwargs)


# ------------------------------------------------------------------------------------ fixtures

def _synthetic_frame(rng, plane: bool, w=W, h=H, fov=70.0):
    """A smooth tilted surface seen by a wide camera, composited with alpha in [0.3, 0.95]."""
    cam = _camera(w, h, fov)
    rays = ref.rays_for(cam.proj_matrix.astype(np.float64), w, h)
    u, v = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    z = 2.0 + 0.4 * np.sin(2.1 * u) + 0.3 * v * v
    n = np.stack([0.35 * np.sin(1.7 * v) + 0.2, -0.25 + 0.1 * u, -np.ones_like(u)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n *= -np.sign(np.sum(n * rays, -1, keepdims=True))          # face the camera along the ray
    A = rng.uniform(0.3, 0.95, (h, w))
    N = A[..., None] * (n + 0.08 * rng.normal(size=n.shape))
    if plane:
        S = np.sum(N * rays, -1) * z * (1 + 0.05 * rng.normal(size=z.shape))
    else:
        S = A * z * (1 + 0.05 * rng.normal(size=z.shape))
    F = np.concatenate([N, S[..., None]], -1).astype(np.float32)
    # The last row is empty background (nothing rendered): A = 0, F = 0. Every term must skip it before
    # dividing, so its depth (0/0) never reaches a gradient.
    F[-1] = 0.0
    A[-1] = 0.0
    return cam, rays, F, A.astype(np.float32), z, n


def _priors(rng, z, n, pw=11, ph=11):
    """Coarser prior grids at a non-integer ratio (11 onto 16, so sampling at the pixel center and at
    its corner pick different texels), with every invalid sentinel present."""
    rows, cols = (np.arange(ph) * z.shape[0]) // ph, (np.arange(pw) * z.shape[1]) // pw
    zs, ns = z[rows][:, cols], n[rows][:, cols]
    depth = np.rint(zs * 1000 * (1 + 0.1 * rng.normal(size=zs.shape))).astype(np.uint16)
    depth[0, 0] = 0
    depth[1, 2] = 65535
    g = ns + 0.15 * rng.normal(size=ns.shape)
    g /= np.linalg.norm(g, axis=-1, keepdims=True)
    codes = np.rint((g + 1) / 2 * 255).astype(np.uint8)
    codes[2, 3] = 128
    codes[4, 5] = 0                      # decodes to (-1,-1,-1), length 1.73: not a target
    rgba = np.concatenate([codes, np.full((ph, pw, 1), 255, np.uint8)], -1)
    return depth, rgba


def _run_loss(p: TestPipeline, cam, F, A, depth, normal, mask=None):
    p.rasterizer.upload_camera(cam)
    out = np.zeros((p.height, p.width, 4), np.float32)
    out[..., 3] = A
    p.buffers.raster.output_image.copy_from_numpy(out)
    p.geometry_forward.aux_image.copy_from_numpy(np.ascontiguousarray(F))
    p.geometry.set_prior_grids(depth.shape if depth is not None else (1, 1),
                               normal.shape[:2] if normal is not None else (1, 1))
    p.geometry.select(depth, normal)
    rgba = np.ones((p.height, p.width, 4), np.float32)
    if mask is not None:
        rgba[..., 3] = mask
    p.targets.set_image(rgba)
    p.geometry.record_loss(output_image=p.buffers.raster.output_image, camera=p.rasterizer.camera_buffer,
                           loss_mask={"loss_mask": p.targets.active})
    p.runtime.wait_for_idle()
    return p.geometry.dL_dAux.to_numpy().astype(np.float64), p.geometry.read_stats()


def _reference_frame(rays, F, A, depth, normal, mask, w=W, h=H):
    Dg = ref.depth_codes_to_m(ref.sample_prior(depth, w, h)) if depth is not None else np.zeros((h, w))
    g = ref.normal_codes_to_n(ref.sample_prior(normal, w, h)) if normal is not None else np.zeros((h, w, 3))
    return ref.Frame(F=F.astype(np.float64), A=A.astype(np.float64), rays=rays, Dg=Dg, g=g,
                     mask=np.ones((h, w)) if mask is None else mask.astype(np.float64))


def _rel_err(gpu, fd):
    return float(np.abs(gpu - fd).max() / np.abs(fd).max())


# ------------------------------------------------------------------------------------ T1, T9, T10

T1_TOL = 1e-2
CASES = [(src, space, nl) for src in ("center", "plane") for space in ("disparity", "metric", "ssi")
         for nl in ("l1", "cos")]


@pytest.mark.parametrize("source,space,normal_loss", CASES)
def test_loss_gradient_matches_reference(source, space, normal_loss):
    """T1: dL_dAux equals central differences of the float64 reference, all three terms on."""
    rng = np.random.default_rng(7)
    plane = source == "plane"
    cam, rays, F, A, z, n = _synthetic_frame(rng, plane)
    depth, normal = _priors(rng, z, n)
    cfg = GeometryConfig(depth_weight=1.0, normal_weight=0.2, depth_normal_weight=0.05,
                         depth_source=source, depth_space=space, normal_loss=normal_loss)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg)
    gpu, stats = _run_loss(p, cam, F, A, depth, normal)
    s = ref.Settings(plane=plane, space=space, normal_cos=normal_loss == "cos")
    fd, sets = ref.fd_gradient(_reference_frame(rays, F, A, depth, normal, None), s)
    # T10: the compared sets are non-empty, and the GPU supervised exactly the reference's pixels.
    assert sets["depth"].sum() > 0 and sets["normal"].sum() > 0 and sets["dn"].sum() > 0
    assert (stats["V_d"], stats["V_n"], stats["V_dn"]) == (
        int(sets["depth"].sum()), int(sets["normal"].sum()), int(sets["dn"].sum()))
    assert np.isfinite(gpu).all()
    err = _rel_err(gpu, fd)
    assert err < T1_TOL, f"max|gpu - fd| / max|fd| = {err:.3e}"


def test_prior_grid_separates_center_and_corner_sampling():
    """T13: at 11 onto 16, the center rule and the corner rule pick different texels somewhere."""
    u = np.arange(W)
    assert (np.floor((u + 0.5) * 11 / W) != np.floor(u * 11 / W)).any()


def test_fixture_separates_the_coverage_denominator():
    """M1 is blind when A ~ 1: the synthetic frame keeps the supervised alpha well below 1."""
    rng = np.random.default_rng(7)
    _, rays, F, A, z, n = _synthetic_frame(rng, False)
    depth, normal = _priors(rng, z, n)
    sets = ref.supervision(_reference_frame(rays, F, A, depth, normal, None), ref.Settings())
    a = A[sets["depth"]]
    assert a.size > 0 and a.min() < 0.7 and a.max() > 0.85


def test_fixture_is_wide_enough_for_the_ray():
    """M4 (optical axis for the ray) is blind at a narrow field of view."""
    rng = np.random.default_rng(7)
    _, rays, *_ = _synthetic_frame(rng, True)
    off_axis = np.degrees(np.arctan(np.linalg.norm(rays[..., :2], axis=-1)))
    assert off_axis.max() > 40.0


def test_invalid_priors_give_zero_gradient_and_no_count():
    """T9: an all-invalid frame (depth 0 and 65535, normal 128) gives no gradient, no count, no NaN."""
    rng = np.random.default_rng(3)
    cam, rays, F, A, z, n = _synthetic_frame(rng, False)
    depth = np.zeros((8, 8), np.uint16)
    depth[::2] = 65535
    normal = np.full((8, 8, 4), 128, np.uint8)
    cfg = GeometryConfig(depth_weight=1.0, normal_weight=0.2)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg)
    assert (A == 0).any()                      # empty pixels: their rendered depth is 0/0
    gpu, stats = _run_loss(p, cam, F, A, depth, normal)
    assert np.isfinite(gpu).all()
    assert stats["V_d"] == 0 and stats["V_n"] == 0
    assert np.count_nonzero(gpu) == 0


def test_invalid_prior_texels_are_skipped_among_valid_ones():
    """T9: the sentinels in an otherwise valid prior are absent from the counts."""
    rng = np.random.default_rng(7)
    cam, rays, F, A, z, n = _synthetic_frame(rng, False)
    depth, normal = _priors(rng, z, n)
    cfg = GeometryConfig(depth_weight=1.0, normal_weight=0.2)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg)
    gpu, stats = _run_loss(p, cam, F, A, depth, normal)
    frame = _reference_frame(rays, F, A, depth, normal, None)
    sentinel = ref.sample_prior(depth, W, H)
    bad_d = (sentinel == 0) | (sentinel == 65535)
    sampled = ref.sample_prior(normal, W, H)[..., :3]
    bad_n = (sampled == 128).all(-1) | (sampled == 0).all(-1)
    assert (sampled == 0).all(-1).any()
    assert bad_d.any() and bad_n.any()
    covered = (A >= 0.5)
    assert stats["V_d"] == int(((~bad_d) & covered).sum())
    assert stats["V_n"] == int(((~bad_n) & covered).sum())
    assert frame.Dg[bad_d].max() == 0.0


# ------------------------------------------------------------------------------------ T8 masks

def test_mask_drops_pixels_and_the_depth_normal_stencil():
    """T8: dropped pixels get dL_dAux == 0 and leave every count; a term whose forward-difference
    stencil crosses the mask edge is skipped; the gradient still matches the reference."""
    rng = np.random.default_rng(11)
    cam, rays, F, A, z, n = _synthetic_frame(rng, False)
    A = np.maximum(A, 0.55).astype(np.float32)            # every pixel covered: the mask alone gates
    depth, normal = _priors(rng, z, n)
    mask = np.ones((H, W), np.float32)
    mask[:, 9] = 0.0                                    # a column: its left neighbours' stencils cross it
    mask[12:, :] = 0.0
    cfg = GeometryConfig(depth_weight=1.0, normal_weight=0.2, depth_normal_weight=0.05)
    p = _pipeline(_device(GEOM_PRIORS=True, LOSS_MASK=True), cfg)
    gpu, stats = _run_loss(p, cam, F, A, depth, normal, mask)
    fd, sets = ref.fd_gradient(_reference_frame(rays, F, A, depth, normal, mask), ref.Settings())
    dropped = mask == 0
    assert dropped.any() and np.count_nonzero(gpu[dropped]) == 0
    edge = np.zeros_like(dropped)
    edge[:, 8] = True                                   # kept, with its +u neighbour dropped
    edge &= ~dropped
    assert edge.any() and not sets["dn"][edge].any()
    assert stats["V_dn"] == int(sets["dn"].sum()) > 0
    assert stats["V_d"] == int(sets["depth"].sum()) > 0
    assert _rel_err(gpu, fd) < T1_TOL


# ------------------------------------------------------------------------------------ T6, T7

def _fixture():
    data = json.loads(FIXTURE.read_text())
    assert data["schema_version"] == "2.0" and len(data["cases"]) == 2
    return data


def _normal_from_depth_gpu(device, depth, K, offset=0.0):
    import slangpy as spy
    from trainer.backend.compute.kernels import KernelLibrary
    h, w = depth.shape
    k = KernelLibrary(device).kernel("losses/geom_loss.slang", "geomNormalFromDepthTest")
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    d = device.create_buffer(size=4 * h * w, usage=usage)
    d.copy_from_numpy(depth.astype(np.float32).ravel())
    out = device.create_buffer(size=16 * h * w, usage=usage)
    out.copy_from_numpy(np.zeros(4 * h * w, np.float32))
    k.dispatch(thread_count=[w, h, 1], depth=d, normal_out=out, dim=spy.uint2(w, h),
               intrinsics=spy.float4(K["fx"], K["fy"], K["cx"], K["cy"]), sample_offset=offset)
    return out.to_numpy().view(np.float32).reshape(h, w, 4)[..., :3].astype(np.float64)


@pytest.mark.parametrize("case", [0, 1])
def test_normal_from_depth_matches_the_fixture(case):
    """T6: the shader's depth-graph normal on both fixture cases (integer sample coordinates)."""
    c = _fixture()["cases"][case]
    depth, expected = np.array(c["depth"]), np.array(c["expected_normal"])
    got = _normal_from_depth_gpu(_device(), depth, c["intrinsics"])
    interior = (slice(0, depth.shape[0] - 1), slice(0, depth.shape[1] - 1))
    assert got[interior].size > 0 and np.abs(expected[interior]).sum() > 0
    assert np.abs(got[interior] - expected[interior]).max() < 1e-5


def test_fixture_grazing_case_has_teeth():
    """T7: case 2 has n.z > 0 on every interior pixel while facing the camera along the ray."""
    c = _fixture()["cases"][1]
    n = np.array(c["expected_normal"])[:-1, :-1]
    K = c["intrinsics"]
    u, v = np.meshgrid(np.arange(n.shape[1]), np.arange(n.shape[0]))
    r = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u, float)], -1)
    assert n.size > 0 and (n[..., 2] > 0).all() and ((n * r).sum(-1) < 0).all()


# ------------------------------------------------------------------------------------ T2, T3, T5

def _scene(opacity_scale=0.45):
    means = np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]], np.float32)
    log_scales = np.log(np.array([[0.3, 0.3, 0.1], [0.2, 0.05, 0.2], [0.25, 0.25, 0.08]], np.float32))
    rot = np.array([[1.0, 0.0, 0.0, 0.0], [0.9659, 0.0, 0.2588, 0.0], [0.9239, 0.0, 0.0, 0.3827]], np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    return {"means": means, "log_scales": log_scales, "rot_quats": rot,
            "opacities": (np.array([0.8, 0.6, 0.7]) * opacity_scale).astype(np.float32),
            "color_logits": np.array([[1.5, -1.5, -1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5]], np.float32)}


def _write_geom(p, geom):
    buf = np.zeros((p.max_splats, 4), np.float32)
    buf[:len(geom)] = geom
    p.geometry_forward.geom_data.copy_from_numpy(buf.ravel())


def _forward(p, cam):
    p.forward_and_loss(cam)
    p.runtime.wait_for_idle()


def _blend_only(p):
    """The blend backward alone, with the dense (disabled) STBN arguments the pipeline itself passes: the
    live-bucket compaction always takes them, and the compacted backward does when STBN_ENABLED is compiled in."""
    tracking = {"blend_weight_accum": p.buffers.densification.blend_weight.storage,
                "track_blend_weight": 0, "track_screen_grad": 1, **p.loss_mask_kwargs()}
    stbn = p._stbn_kwargs(None)
    p.backward._record_blend(stbn, stbn if p.backward._stbn_compacted else {}, tracking, 1.0)
    p.runtime.wait_for_idle()


def _backward_setup(per_bucket, rng, stride2=False):
    cfg = GeometryConfig(depth_normal_weight=0.05)
    p = _pipeline(_device(GEOM_PRIORS=True, STRIDE2=stride2), cfg, use_flat_bucket_backward=per_bucket)
    cam = _camera(fov=60.0)
    p.targets.set_image(np.ones((H, W, 3), np.float32))
    p.set_primitive_from_numpy(**_scene())
    p.geometry_forward.skip_features = True
    weights = []
    for i in range(3):                                  # one-hot features read the weights back
        onehot = np.zeros((3, 4), np.float32)
        onehot[i, 0] = 1.0
        _write_geom(p, onehot)
        _forward(p, cam)
        weights.append(p.geometry_forward.aux_image.to_numpy()[..., 0].astype(np.float64))
    geom = rng.normal(size=(3, 4)).astype(np.float32)
    _write_geom(p, geom)
    _forward(p, cam)
    aux = p.geometry_forward.aux_image.to_numpy().astype(np.float64)
    G = rng.normal(size=(H, W, 4)).astype(np.float32)
    p.buffers.loss.image_gradient.copy_from_numpy(np.zeros((H, W, 4), np.float32))
    p.geometry.dL_dAux.copy_from_numpy(G)
    p.optimizer.zero_grad()
    _blend_only(p)
    return p, np.stack(weights), geom, aux, G.astype(np.float64)


@pytest.mark.parametrize("per_bucket", [False, True])
def test_backward_seeds_are_detached_weight_sums(per_bucket):
    """T2: seeds = sum_p dL_dAux_p * w_ip with w read back by rendering one-hot features; the
    composite itself is sum_i w_i f_i."""
    rng = np.random.default_rng(5)
    p, w, geom, aux, G = _backward_setup(per_bucket, rng)
    covered = w.sum(0) > 0
    assert covered.sum() > 20 and 0.1 < w.sum(0)[covered].max() < 0.95
    assert np.abs(aux - np.einsum("ihw,ic->hwc", w, geom)).max() < 1e-5
    seeds = p.geometry.read_seeds()
    expected = np.einsum("ihw,hwc->ic", w, G)
    got = np.stack([seeds[n][:3] for n in SEED_NAMES], -1)
    assert np.abs(expected).max() > 0.1
    assert np.abs(got - expected).max() < 1e-3 + 1e-4 * np.abs(expected).max()


@pytest.mark.parametrize("per_bucket", [False, True])
def test_geometry_gradient_never_touches_the_blend_weights(per_bucket):
    """T3: with dL_dC = 0, every photometric seed and gradient is exactly zero."""
    rng = np.random.default_rng(5)
    p, *_ = _backward_setup(per_bucket, rng)
    assert np.abs(np.stack(list(p.geometry.read_seeds().values()))).max() > 0     # the aux path ran
    grads = p.read_gradients()
    for name in ("opacity", "color_r", "color_g", "color_b"):
        assert np.count_nonzero(grads[name]) == 0, name
    for name, buf in p.buffers.gradients.seeds.items():
        assert np.count_nonzero(np.frombuffer(buf.to_numpy().tobytes(), np.int64)) == 0, name


def test_per_splat_and_per_bucket_backward_agree():
    """T5: both blend backward kernels produce the same geometry seeds."""
    a, *_ = _backward_setup(False, np.random.default_rng(5))
    b, *_ = _backward_setup(True, np.random.default_rng(5))
    sa, sb = a.geometry.read_seeds(), b.geometry.read_seeds()
    assert max(np.abs(sa[n][:3]).max() for n in SEED_NAMES) > 0.1
    for n in SEED_NAMES:
        assert np.abs(sa[n][:3] - sb[n][:3]).max() < 1e-3, n


# ------------------------------------------------------------------------------------ T4

def _chain_scene():
    """Off-axis splats whose facing sign disagrees with -sign(n.z) for at least one of them."""
    means = np.array([[1.6, 0.3, 2.0], [-0.4, -0.9, 3.0], [0.2, 0.1, 2.5]], np.float32)
    log_scales = np.log(np.array([[0.2, 0.3, 0.02], [0.03, 0.2, 0.3], [0.2, 0.04, 0.25]], np.float32))
    # Splat 0: its disc normal is world (-1, 0, 0.3) (a rotation of z about y by -73.3 deg), camera
    # (1, 0, 0.3) for this camera: it faces the camera along the splat's ray but has n.z > 0.
    half = 0.5 * np.arctan2(-1.0, 0.3)
    rot = np.array([[np.cos(half), 0.0, np.sin(half), 0.0], [0.60, -0.50, 0.30, 0.55],
                    [0.95, 0.10, -0.20, 0.20]], np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    return {"means": means, "log_scales": log_scales, "rot_quats": rot,
            "opacities": np.array([0.5, 0.5, 0.5], np.float32),
            "color_logits": np.zeros((3, 3), np.float32)}


@pytest.mark.parametrize("source", ["center", "plane"])
def test_feature_chain_matches_host_finite_differences(source):
    """T4: geom_features against the host feature, and geom_backward against host FD on the mean
    and the quaternion, with the axis and the facing sign detached."""
    plane = source == "plane"
    cfg = GeometryConfig(depth_normal_weight=0.05, depth_source=source)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg)
    cam = _camera(fov=60.0)
    scene = _chain_scene()
    p.targets.set_image(np.ones((H, W, 3), np.float32))
    p.set_primitive_from_numpy(**scene)
    _forward(p, cam)
    view = cam.view_matrix.astype(np.float64)
    got = p.geometry_forward.geom_data.to_numpy().view(np.float32).reshape(-1, 4)[:3].astype(np.float64)
    ks = [int(np.argmin(s)) for s in scene["log_scales"]]
    sgns = [ref.facing_sign(scene["means"][i].astype(np.float64), scene["rot_quats"][i].astype(np.float64), view, ks[i])
            for i in range(3)]
    want = np.stack([ref.feature(scene["means"][i].astype(np.float64), scene["rot_quats"][i].astype(np.float64),
                                 view, ks[i], sgns[i], plane) for i in range(3)])
    # the fixture separates the ray rule from an n.z rule on at least one splat
    unsigned = [want[i, :3] * sgns[i] for i in range(3)]
    nz_rule = [1.0 if unsigned[i][2] <= 0 else -1.0 for i in range(3)]
    assert sum(nz_rule[i] != sgns[i] for i in range(3)) >= 1
    assert np.abs(got - want).max() < 1e-5
    # backward: seeds dL/df, host FD of dot(seed, f)
    rng = np.random.default_rng(2)
    seed = rng.normal(size=(3, 4))
    p.optimizer.zero_grad()                              # zeroes the parameter gradients and the geometry seeds
    for c, name in enumerate(SEED_NAMES):
        fixed = np.zeros(p.max_splats, np.int64)
        fixed[:3] = np.trunc(seed[:, c] * FIXED_SCALE).astype(np.int64)
        p.geometry.seeds[name].copy_from_numpy(fixed.view(np.uint32))
        seed[:, c] = fixed[:3] / FIXED_SCALE
    p.geometry.record_backward(gaussians=p.buffers.primitive.values.storage,
                               active_count=p.buffers.primitive.active_count.storage,
                               camera=p.rasterizer.camera_buffer,
                               cluster_visible=p.buffers.raster.cluster_visible.storage,
                               grads=p.buffers.gradients.params)
    p.runtime.wait_for_idle()
    grads = p.read_gradients()
    gpu = np.stack([np.stack([grads[n][i] for n in ("mean_x", "mean_y", "mean_z", "rot_w", "rot_x", "rot_y", "rot_z")])
                    for i in range(3)])
    eps = 1e-6
    fd = np.zeros_like(gpu)
    for i in range(3):
        x0 = np.concatenate([scene["means"][i], scene["rot_quats"][i]]).astype(np.float64)
        for j in range(7):
            xp, xm = x0.copy(), x0.copy()
            xp[j] += eps
            xm[j] -= eps
            fp = ref.feature(xp[:3], xp[3:], view, ks[i], sgns[i], plane)
            fm = ref.feature(xm[:3], xm[3:], view, ks[i], sgns[i], plane)
            fd[i, j] = np.dot(seed[i], fp - fm) / (2 * eps)
    assert np.abs(fd).max() > 0.1
    assert np.abs(gpu - fd).max() / np.abs(fd).max() < 2e-3
    # the seeds are consumed
    assert all(np.count_nonzero(v[:3]) == 0 for v in p.geometry.read_seeds().values())


# ------------------------------------------------------------------------------------ T16 flatten

def test_flatten_is_on_the_smallest_axis_only_and_stops_at_the_floor():
    """T16: grad log s_k = w * 3WH / (n * U) * s_k on k = argmin only; none below 1e-5 m * U."""
    w_f, U = 1.0, 2.0
    cfg = GeometryConfig(flatten_weight=w_f, units_per_meter=U)
    p = _pipeline(_device(), cfg)                    # flatten alone needs no GEOM_PRIORS
    assert p.geometry is not None and not p.geometry.image_terms
    scene = _chain_scene()
    scene["log_scales"][2] = np.log([0.2, 1e-6, 0.3])      # below the floor (1e-5 m * U = 2e-5)
    p.set_primitive_from_numpy(**scene)
    _forward(p, _camera())
    p.optimizer.zero_grad()
    p.geometry.record_backward(gaussians=p.buffers.primitive.values.storage,
                               active_count=p.buffers.primitive.active_count.storage,
                               camera=p.rasterizer.camera_buffer,
                               cluster_visible=p.buffers.raster.cluster_visible.storage,
                               grads=p.buffers.gradients.params)
    p.runtime.wait_for_idle()
    g = p.read_gradients()
    got = np.stack([g["log_scale_x"], g["log_scale_y"], g["log_scale_z"]], -1)[:3]
    n_live = 3
    for i in range(2):
        k = int(np.argmin(scene["log_scales"][i]))
        s_k = float(np.exp(scene["log_scales"][i][k]))
        expected = w_f * 3 * W * H / (n_live * U) * s_k
        assert got[i, k] == pytest.approx(expected, rel=1e-3, abs=2 / FIXED_SCALE)
        assert expected > 0.1
        others = [a for a in range(3) if a != k]
        assert np.count_nonzero(got[i, others]) == 0
    assert np.count_nonzero(got[2]) == 0


# ------------------------------------------------------------------------------------ T15

def _scaling_ratio(size, terms):
    """|g_geometry| / |g_photometric| on the 3-splat scene at size x size."""
    rng = np.random.default_rng(1)
    cfg = GeometryConfig(**terms)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg, w=size, h=size)
    cam = _camera(size, size, 60.0)
    target = np.ones((size, size, 3), np.float32) * 0.9
    p.targets.set_image(target)
    p.set_primitive_from_numpy(**_scene(1.0))
    if "depth_weight" in terms:
        p.geometry.set_prior_grids((8, 8), (1, 1))
        prior = np.full((8, 8), 2300, np.uint16)
        prior[:, :4] = 2600
    out = []
    for active in (False, True):
        p.geometry.active = active
        if "depth_weight" in terms:
            p.geometry.select(prior, None)
        p.set_primitive_from_numpy(**_scene(1.0))
        p.forward_and_loss(cam)
        p.optimizer.zero_grad()
        p.run_backward()
        p.runtime.wait_for_idle()
        g = p.read_gradients()
        out.append(np.concatenate([g[n][:3] for n in ("mean_x", "mean_y", "mean_z")]))
    photo, total = out
    del rng
    return np.linalg.norm(total - photo) / np.linalg.norm(photo)


@pytest.mark.parametrize("terms", [{"depth_normal_weight": 0.05}, {"depth_weight": 1.0}])
def test_weights_are_resolution_independent(terms):
    """T15: doubling the image area at a fixed scene keeps the geometry-to-photometric ratio."""
    r1, r2 = _scaling_ratio(32, terms), _scaling_ratio(64, terms)
    assert r1 > 1e-3 and r2 > 1e-3
    assert abs(r2 / r1 - 1.0) < 0.02, (r1, r2)


# ------------------------------------------------------------------------------------ forward, viewer

def test_forward_with_geometry_is_bit_identical_to_without():
    """The geometry blend writes exactly the color, alpha, counts and checkpoints of the default one."""
    cam = _camera(fov=60.0)
    outputs = []
    for on in (False, True):
        cfg = GeometryConfig(depth_normal_weight=0.05) if on else GeometryConfig()
        p = _pipeline(_device(GEOM_PRIORS=on), cfg)
        p.targets.set_image(np.ones((H, W, 3), np.float32))
        p.set_primitive_from_numpy(**_scene())
        _forward(p, cam)
        r = p.buffers.raster
        outputs.append((r.output_image.to_numpy().copy(), r.gaussian_count_per_pixel.to_numpy().copy(),
                        np.concatenate([c.to_numpy().reshape(-1) for c in r.raster_checkpoints])))
    (a_img, a_cnt, a_cp), (b_img, b_cnt, b_cp) = outputs
    assert a_img[..., 3].max() > 0.1 and a_cnt.max() > 0
    assert np.array_equal(a_img, b_img) and np.array_equal(a_cnt, b_cnt) and np.array_equal(a_cp, b_cp)


def test_viewer_pipeline_renders_with_the_define_on():
    """R11: a render-only pipeline binds the geometry arguments when GEOM_PRIORS is compiled in (a one-element
    stub it never composites into: the viewer does not read it)."""
    from trainer.backend.render.pipeline import TrainingPipeline
    p = TrainingPipeline(_device(GEOM_PRIORS=True), W, H, max_splats=16, viewer_only=True)
    assert p.geometry_forward is not None and p.geometry_forward.stub
    p.geometry_forward.aux_image.copy_from_numpy(np.full((1, 1, 4), 9.0, np.float32))
    p.set_gaussians(**{k: v for k, v in _scene().items()})
    p.render(_camera(fov=60.0))
    img = p.read_output_image()
    assert img[..., 3].max() > 0.1
    assert (p.geometry_forward.aux_image.to_numpy() == 9.0).all()


def test_train_steps_run_and_the_terms_change_the_result():
    """A few full training steps with every term on stay finite and differ from the same steps
    without the terms (the positive control of the no-flag identity tests)."""
    def run(cfg, on):
        p = _pipeline(_device(GEOM_PRIORS=on), cfg)
        p.targets.set_image(np.ones((H, W, 3), np.float32) * 0.8)
        p.set_primitive_from_numpy(**_scene())
        if on:
            p.geometry.set_prior_grids((8, 8), (8, 8))
            n = np.zeros((8, 8, 4), np.uint8)
            n[..., :3] = (140, 120, 20)
            n[..., 3] = 255
            p.geometry.select(np.full((8, 8), 2400, np.uint16), n)
        for _ in range(5):
            if on:
                p.geometry.select(np.full((8, 8), 2400, np.uint16), n)
            p.train_step(_camera(fov=60.0))
        p.runtime.wait_for_idle()
        values = p.buffers.primitive.values.to_numpy()
        return np.frombuffer(values.tobytes(), np.float32), (p.geometry.read_stats() if on else {})
    base, _ = run(GeometryConfig(), False)
    geo, stats = run(GeometryConfig(depth_weight=1.0, normal_weight=0.2, depth_normal_weight=0.05,
                                    flatten_weight=1.0), True)
    assert np.isfinite(geo).all()
    assert stats["V_d"] > 0 and stats["V_n"] > 0 and stats["V_dn"] > 0
    assert not np.array_equal(base, geo)


# ------------------------------------------------------------------------------------ static identity

# Measured on the base commit (branch compressed-targets, 3ca200e) with the probe below: the default
# shader defines and the dispatch markers of one train_step of the 3-splat scene.
BASE_DEFINES = {'COMPOSITOR_HARDNESS_ONE': '1', 'ADAM_MOMENTS_BF16': '1'}
BASE_DISPATCHES = ["clear_textures", "clear_tile_ranges", "build_cluster_aabb", "cull_clusters_frustum", "bin_count", "radix_prepare_dispatch", "radix_init_global_hist", "radix_upsweep", "radix_scan", "radix_downsweep", "radix_upsweep", "radix_scan", "radix_downsweep", "radix_upsweep", "radix_scan", "radix_downsweep", "radix_upsweep", "radix_scan", "radix_downsweep", "gather_tile_counts", "scan_block", "scan_block_sums", "add_block_offsets", "pair_dispatch_args", "bin_emit", "radix_prepare_dispatch", "radix_init_global_hist", "radix_upsweep", "radix_scan", "radix_downsweep", "identify_tile_ranges", "bucket_count", "scan_block", "scan_block_sums", "add_block_offsets", "forward", "ssim_maps", "ssim_grad", "tile_max_processed", "scan_block", "scan_block_sums", "add_block_offsets", "scatter_live_buckets", "bucket_prepare_dispatch", "backward_persbucket", "projection_backward", "accumulate_iter", "adam_prepare_args", "adam"]


def _identity_probe(argv):
    from trainer.backend.compute import gpu
    from trainer.config.resolve import RunConfig, shader_specialization
    from trainer.frontend.cli import build_parser
    from trainer.frontend.factory import shader_defines
    args = build_parser().parse_args(["--scene", "x", *argv])
    defines = shader_defines(shader_specialization(args))
    names = []
    orig_d, orig_i = gpu.CommandBatch.dispatch, gpu.CommandBatch.dispatch_indirect
    def d(self, kernel, marker, **kw):
        names.append(marker)
        return orig_d(self, kernel, marker, **kw)
    def i(self, kernel, marker, args_buf, offset=0, **kw):
        names.append(marker)
        return orig_i(self, kernel, marker, args_buf, offset, **kw)
    gpu.CommandBatch.dispatch, gpu.CommandBatch.dispatch_indirect = d, i
    try:
        device = create_device(backend="vulkan", defines=defines)
        config = RunConfig.from_args(args, 1.0).geometry
        p = TestPipeline(device=device, width=16, height=16, max_splats=16, loss_config=LossConfig(),
                         geometry_config=config)
        p.targets.set_image(np.ones((16, 16, 3), np.float32))
        p.set_primitive_from_numpy(
            means=np.array([[0, 0, 3], [0.2, 0.1, 2.5], [-0.15, -0.1, 2]], np.float32),
            log_scales=np.log(np.full((3, 3), 0.25, np.float32)), rot_quats=np.array([[1, 0, 0, 0]] * 3, np.float32),
            opacities=np.array([0.8, 0.6, 0.7], np.float32), color_logits=np.zeros((3, 3), np.float32))
        names.clear()
        p.train_step(simple_camera(width=16, height=16, eye=(0, 0, 0), target=(0, 0, 1), up=(0, 1, 0)))
        p.runtime.wait_for_idle()
    finally:
        gpu.CommandBatch.dispatch, gpu.CommandBatch.dispatch_indirect = orig_d, orig_i
    return defines, names


def test_no_flag_defines_and_dispatches_equal_the_base():
    """5.2 static: without a geometry flag, the defines and one step's dispatches are the base's."""
    defines, names = _identity_probe([])
    assert defines == BASE_DEFINES
    assert names == BASE_DISPATCHES


def test_geometry_flags_change_the_defines_and_dispatches():
    """Positive control of the identity test: the probe sees the change when a flag is on."""
    defines, names = _identity_probe(["--depth-normal-weight", "0.05", "--flatten-loss-weight", "1"])
    assert defines == {**BASE_DEFINES, "GEOM_PRIORS": "1"}
    assert names != BASE_DISPATCHES
    for marker in ("geom_features", "geom_clear_stats", "geom_resolve", "geom_grad", "geom_backward"):
        assert marker in names, marker
    flat_defines, flat_names = _identity_probe(["--flatten-loss-weight", "1"])
    assert flat_defines == BASE_DEFINES and "geom_backward" in flat_names and "geom_resolve" not in flat_names


def test_depth_graph_normal_faces_the_camera_on_real_depth():
    """F-real: on a real depth crop, at pixel centers, in float32, every depth-graph normal faces
    the camera along its own ray (dot(dP/dv x dP/du, r) = -z(u+1,v) z(u,v+1) / (fx fy) < 0)."""
    import cv2
    import os
    folder = Path(os.environ.get("PERFGS_PRIVATE_FIXTURES", "/nonexistent"))
    if not (folder / "real_priors.json").is_file():
        pytest.skip("real prior crops not available (PERFGS_PRIVATE_FIXTURES)")
    meta = json.loads((folder / "real_priors.json").read_text())["depth_mm"]
    depth = cv2.imread(str(folder / meta["file"]), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    K = meta["intrinsics"]
    n = _normal_from_depth_gpu(_device(), depth, K, offset=0.5)[:-1, :-1]
    u, v = np.meshgrid(np.arange(depth.shape[1] - 1) + 0.5, np.arange(depth.shape[0] - 1) + 0.5)
    r = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u)], -1)
    assert (depth > 0).all() and n.size > 0
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-4)
    assert ((n * r).sum(-1) < 0).all()


@pytest.mark.parametrize("per_bucket", [False, True])
def test_backward_seeds_with_pixel_stride_are_unbiased(per_bucket):
    """BACKWARD_PIXEL_STRIDE 2: each pass visits half the pixels and scales by 2, so over both phases the seeds
    sum to twice the full-pixel sums."""
    rng = np.random.default_rng(5)
    p, w, geom, aux, G = _backward_setup(per_bucket, rng, stride2=True)
    _blend_only(p)                                       # the second phase; seeds are not consumed in between
    seeds = p.geometry.read_seeds()
    expected = 2.0 * np.einsum("ihw,hwc->ic", w, G)
    got = np.stack([seeds[n][:3] for n in SEED_NAMES], -1)
    assert np.abs(expected).max() > 0.1
    assert np.abs(got - expected).max() < 2e-3 + 1e-4 * np.abs(expected).max()


def test_ssi_fit_recovers_the_slope_at_long_range():
    """At 40-100 m the disparities are ~0.01-0.025 and their squares fall below the fixed-point step; the
    relative moments keep the fit: a and b of y = a x + b recovered within 1%."""
    rng = np.random.default_rng(9)
    w = h = 64
    cam = _camera(w, h, 60.0)
    a_true, b_true = 0.8, -0.004
    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    D = 40.0 + 60.0 * (0.6 * u + 0.4 * v)                # rendered depth, 40 to 100 m
    x = (1.0 / D - b_true) / a_true                      # prior disparity: 1/D = a x + b exactly
    depth = np.rint(1000.0 / x).astype(np.uint16)        # 27 to 57 m, in mm
    assert depth.max() < 65535 and depth.min() > 20000
    A = np.full((h, w), 0.9, np.float32)
    F = np.zeros((h, w, 4), np.float32)
    F[..., 2] = -0.9
    F[..., 3] = A * D
    cfg = GeometryConfig(depth_weight=1.0, depth_space="ssi")
    p = _pipeline(_device(GEOM_PRIORS=True), cfg, w=w, h=h)
    gpu, stats = _run_loss(p, cam, F, A, depth, None)
    assert stats["V_d"] == w * h and stats["skipped_frames"] == 0
    assert stats["ssi_a"] == pytest.approx(a_true, rel=0.01)
    assert stats["ssi_b"] == pytest.approx(b_true, rel=0.01)
    assert np.isfinite(gpu).all()
    del rng


def test_inverted_ssi_fit_skips_the_frame():
    """a <= 0 (prior and render ordered oppositely) is not a fit: the frame's depth term is skipped and counted."""
    w = h = 32
    cam = _camera(w, h, 60.0)
    u, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    D = 2.0 + u
    depth = np.rint(1000.0 * (5.0 - 2.0 * u)).astype(np.uint16)   # nearer where the render is farther
    A = np.full((h, w), 0.9, np.float32)
    F = np.zeros((h, w, 4), np.float32)
    F[..., 2] = -0.9
    F[..., 3] = A * D
    p = _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(depth_weight=1.0, depth_space="ssi"), w=w, h=h)
    gpu, stats = _run_loss(p, cam, F, A, depth, None)
    assert stats["V_d"] > 0 and stats["skipped_frames"] == 1
    assert np.count_nonzero(gpu) == 0


def test_flatten_with_the_per_pixel_backward_is_refused():
    """The per-pixel reference backward never runs geom_backward, so flatten would be silently ignored."""
    with pytest.raises(RuntimeError, match="per-splat backward"):
        _pipeline(_device(), GeometryConfig(flatten_weight=1.0), use_persplat_backward=False,
                  use_flat_bucket_backward=False)


def _ssi_frame(D, depth_mm, units_per_meter=1.0):
    w, h = D.shape[1], D.shape[0]
    A = np.full((h, w), 0.9, np.float32)
    F = np.zeros((h, w, 4), np.float32)
    F[..., 2] = -0.9
    F[..., 3] = A * D
    cfg = GeometryConfig(depth_weight=1.0, depth_space="ssi", units_per_meter=units_per_meter)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg, w=w, h=h)
    return _run_loss(p, _camera(w, h, 60.0), F, A, depth_mm, None)


def test_ssi_fit_refuses_a_narrow_depth_spread():
    """Depths over 98-100 m (2%) cannot pin a slope and an offset: the frame is skipped and counted."""
    w = h = 32
    u, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    D = 98.0 + 2.0 * u
    x = (1.0 / D + 0.0078) / 0.8                          # a = 0.8, b = -0.0078: priors ~ 44-45 m
    mm = np.rint(1000.0 / x)
    assert 1000 < mm.min() and mm.max() < 65535            # in the uint16 mm range (no wrap)
    depth = mm.astype(np.uint16)
    rel_var = np.var(x) / np.mean(x) ** 2
    assert 0 < rel_var < 1e-3                             # the fixture really sits under the gate
    gpu, stats = _ssi_frame(D, depth)
    assert stats["V_d"] == w * h and stats["skipped_frames"] == 1
    assert np.count_nonzero(gpu) == 0


def test_ssi_means_resolve_small_disparities():
    """Pins the 2^24 scale of the disparity means. Two depth plateaus whose disparities (scene units per metre
    0.01: y = 1e-4 and 2.5e-4 per m) round by up to ~7% at a 2^16 scale, where b then misses by 1.1%; a and b
    must come out within 0.3% (measured: 0.01% and 0.05%)."""
    w = h = 32
    a_true, b_true, U = 0.008, -4e-5, 0.01
    D = np.where(np.arange(w)[None, :] < w // 2, 100.0, 40.0) * np.ones((h, 1))     # scene units
    y = U / D
    x = (y - b_true) / a_true                             # prior disparity, 1/m: 57 m and 27.6 m
    mm = np.rint(1000.0 / x)
    assert 1000 < mm.min() and mm.max() < 65535
    depth = mm.astype(np.uint16)
    coarse = np.abs(np.rint(y * 65536) / 65536 - y) / y   # what a 2^16 scale would make of each mean
    assert coarse.max() > 0.05
    gpu, stats = _ssi_frame(D, depth, units_per_meter=U)
    assert stats["V_d"] == w * h and stats["skipped_frames"] == 0
    assert stats["ssi_a"] == pytest.approx(a_true, rel=0.003)
    assert stats["ssi_b"] == pytest.approx(b_true, rel=0.003)
    assert np.isfinite(gpu).all()


# ------------------------------------------------------------------------------------ review regressions

def _scaled_case(rng, plane: bool, depth_scale: float, units_per_meter: float, **cfg):
    """The T1 frame with every rendered depth multiplied by ``depth_scale`` (scene units), its priors in metres
    (depth / units_per_meter), and the GPU and float64-reference results of all three terms."""
    cam, rays, F, A, z, n = _synthetic_frame(rng, plane)
    F = F.copy()
    F[..., 3] *= depth_scale
    depth, normal = _priors(rng, z * depth_scale / units_per_meter, n)
    config = GeometryConfig(depth_weight=1.0, normal_weight=0.2, depth_normal_weight=0.05,
                            depth_source="plane" if plane else "center", units_per_meter=units_per_meter, **cfg)
    p = _pipeline(_device(GEOM_PRIORS=True), config)
    gpu, stats = _run_loss(p, cam, F, A, depth, normal)
    s = ref.Settings(plane=plane, U=units_per_meter)
    fd, sets = ref.fd_gradient(_reference_frame(rays, F, A, depth, normal, None), s)
    return gpu, stats, fd, sets


@pytest.mark.parametrize("plane", [True, False], ids=["plane", "center"])
@pytest.mark.parametrize("depth_scale,units", [(1e4, 1000.0), (3e-4, 1e-3)], ids=["mm-scene-20m", "km-scene-0.6m"])
def test_depth_validity_range_is_in_metres(plane, depth_scale, units):
    """The 1 mm .. 10 km validity range of a rendered depth is in metres, not scene units: 20 m in a millimetre
    scene is 20,000 units, and 0.6 m in a kilometre scene is 0.0006 units; both are valid depths."""
    gpu, stats, fd, sets = _scaled_case(np.random.default_rng(7), plane, depth_scale, units)
    assert sets["depth"].sum() > 0 and sets["dn"].sum() > 0
    assert (stats["V_d"], stats["V_dn"]) == (int(sets["depth"].sum()), int(sets["dn"].sum()))
    assert np.isfinite(gpu).all() and _rel_err(gpu, fd) < T1_TOL


def test_center_depth_below_a_millimetre_is_not_supervised():
    """Center depth had no range check: a splat a fraction of a millimetre from the camera entered the disparity
    sums (U / D) unbounded, which can overflow the int64 fixed point. It is now invalid, as in plane mode."""
    rng = np.random.default_rng(7)
    cam, rays, F, A, z, n = _synthetic_frame(rng, plane=False)
    F = F.copy()
    F[..., 3] *= 2e-4                                   # D ~ 0.4 mm at U = 1
    depth, normal = _priors(rng, z, n)                  # the priors still say ~2 m
    p = _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(depth_weight=1.0, depth_normal_weight=0.05))
    gpu, stats = _run_loss(p, cam, F, A, depth, None)
    assert stats["V_d"] == 0 and stats["V_dn"] == 0
    assert np.count_nonzero(gpu) == 0
    assert ref.resolve(F.astype(np.float64), A.astype(np.float64), rays, ref.Settings())[0].max() == 0.0


def test_fixed_point_disparity_sums_fit_the_bound_the_range_sets():
    """1 mm (the nearest valid depth) is a disparity of 1000 / m; at SSI_MEAN_SCALE over 24 Mpx the sum stays in
    int64. The range check is what makes this bound hold for center depth too."""
    from trainer.backend.render.geometry import SSI_MEAN_SCALE
    assert 1000.0 * SSI_MEAN_SCALE * 24e6 < 2**63


@pytest.mark.parametrize("term", ["depth", "normal", "dn"])
@pytest.mark.parametrize("off_center", [False, True], ids=["centred", "off-centre"])
def test_each_term_alone_matches_the_reference(term, off_center):
    """T1 sums the three terms, so a 15% error in the smallest could hide under its tolerance: each term alone,
    also with an off-centre principal point."""
    rng = np.random.default_rng(11)
    cam, rays, F, A, z, n = _synthetic_frame(rng, plane=False)
    if off_center:
        cam.proj_matrix[0, 2], cam.proj_matrix[1, 2] = 0.12, -0.08
        rays = ref.rays_for(cam.proj_matrix.astype(np.float64), W, H)
    depth, normal = _priors(rng, z, n)
    weights = {"depth": (1.0, 0.0, 0.0), "normal": (0.0, 0.2, 0.0), "dn": (0.0, 0.0, 0.05)}[term]
    cfg = GeometryConfig(depth_weight=weights[0], normal_weight=weights[1], depth_normal_weight=weights[2])
    p = _pipeline(_device(GEOM_PRIORS=True), cfg)
    gpu, _ = _run_loss(p, cam, F, A, depth if weights[0] else None, normal if weights[1] else None)
    s = ref.Settings(w_d=weights[0], w_n=weights[1], w_dn=weights[2])
    fd, sets = ref.fd_gradient(_reference_frame(rays, F, A, depth if weights[0] else None,
                                                normal if weights[1] else None, None), s)
    assert sets[term].sum() > 0 and np.abs(fd).max() > 0
    assert _rel_err(gpu, fd) < 2e-3


def test_cos_normal_error_is_never_negative():
    """A decoded 8-bit normal is up to ~1% off unit length; 1 - dot(nh, g) then went below 0 when the render
    matched the prior's direction. The prior is normalised before use."""
    rng = np.random.default_rng(3)
    cam, rays, F, A, z, n = _synthetic_frame(rng, plane=False)
    _, normal = _priors(rng, z, n, pw=W, ph=H)
    g = ref.normal_codes_to_n(normal)
    raw = normal[..., :3].astype(np.float64) / 255.0 * 2.0 - 1.0
    assert np.abs(np.linalg.norm(raw, axis=-1) - 1.0).max() > 2e-3            # the codes are off unit length
    F = F.copy()
    F[..., :3] = (A[..., None] * np.where(np.linalg.norm(g, axis=-1, keepdims=True) > 0, g, n)).astype(np.float32)
    p = _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(normal_weight=1.0, normal_loss="cos"))
    _, stats = _run_loss(p, cam, F, A, None, normal)
    assert stats["V_n"] > 0
    assert stats["loss_n"] >= -1e-4 and abs(stats["loss_n"]) / stats["V_n"] < 1e-4


def test_geom_priors_define_zero_is_off():
    """Slang reads GEOM_PRIORS="0" as off; the pipeline must too, or it binds geometry arguments the kernels lack."""
    from trainer.backend.render.pipeline import TrainingPipeline
    device = create_device(backend="vulkan", defines={"GEOM_PRIORS": "0"})
    try:
        p = TrainingPipeline(device, W, H, max_splats=16)
        assert p.geom_priors is False and p.geometry_forward is None
        p.set_gaussians(**_scene())
        p.render(_camera(fov=60.0))
        assert p.read_output_image()[..., 3].max() > 0.1
    finally:
        device.close()


def test_stbn_config_is_refused_with_the_image_terms(monkeypatch):
    """The aux seeds carry no STBN unbiasing scale, so a subsampled run would underweight them silently. The CLI
    refuses it; the pipeline must too, for API callers. (The STBN table load is stubbed: the asset is not needed
    to show the pipeline accepts the combination.)"""
    from trainer.backend.render import pipeline as pipeline_module
    from trainer.backend.render.stbn_mask import StbnMask
    from trainer.config.config import StbnConfig
    monkeypatch.setattr(pipeline_module.StbnMask, "for_config", classmethod(lambda cls, device, _c: StbnMask.disabled(device)))
    _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(flatten_weight=1.0), stbn_config=StbnConfig(pixel_fraction=0.5))
    with pytest.raises(RuntimeError, match="STBN"):
        _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(depth_normal_weight=0.05),
                  stbn_config=StbnConfig(pixel_fraction=0.5))


def test_stbn_subsampled_step_is_refused_with_the_image_terms():
    from trainer.config.config import StbnVariables
    p = _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(depth_normal_weight=0.05))
    p.set_primitive_from_numpy(**_scene())
    p.targets.set_image(np.ones((H, W, 3), np.float32))
    with pytest.raises(RuntimeError, match="STBN"):
        p.forward_and_loss(_camera(fov=60.0), stbn=StbnVariables(slice_index=0, pixels_per_tile=16))


def test_viewer_allocates_no_geometry_and_renders_the_same_colour():
    """The viewer never reads the geometry composite: under GEOM_PRIORS it binds a one-element stub and the
    forward skips the composite, and its colour is the plain viewer's, bit for bit."""
    from trainer.backend.render.pipeline import TrainingPipeline
    images = []
    for on in (False, True):
        p = TrainingPipeline(_device(GEOM_PRIORS=on), W, H, max_splats=16, viewer_only=True)
        if on:
            g = p.geometry_forward
            assert g.geom_data.size == 16 and (g.aux_image.width, g.aux_image.height) == (1, 1)
        p.set_gaussians(**_scene())
        p.render(_camera(fov=60.0))
        images.append(p.read_output_image().copy())
    assert images[0][..., 3].max() > 0.1 and np.array_equal(images[0], images[1])


def test_viewer_resize_releases_the_old_geometry_first(monkeypatch):
    from trainer.backend.render import pipeline as pipeline_module
    from trainer.backend.render.pipeline import TrainingPipeline
    p = TrainingPipeline(_device(GEOM_PRIORS=True), W, H, max_splats=16, viewer_only=True)
    seen = []
    real = pipeline_module.GeometryForward

    def spy(*args, **kwargs):
        seen.append(p.geometry_forward)
        return real(*args, **kwargs)
    monkeypatch.setattr(pipeline_module, "GeometryForward", spy)
    p.resize(32)
    assert seen == [None]


def _resource_bytes(*owners) -> int:
    import slangpy as spy
    total = 0
    for owner in owners:
        for value in list(vars(owner).values()) + list(getattr(owner, "seeds", {}).values()):
            if isinstance(value, (spy.Buffer, spy.Texture)):
                total += value.memory_usage.device
    return total


def test_geometry_memory_is_reported_and_planned():
    """GeometryForward and GeometryTerms sit outside pipeline.buffers: the run report and the VRAM plan must count
    them (~385 MB at 4M splats / 1600x1200)."""
    from trainer.backend.render.pipeline import estimate_pipeline_bytes
    from trainer.frontend.evaluation.run_report import gpu_allocation
    cfg = GeometryConfig(depth_weight=1.0, normal_weight=0.2, depth_normal_weight=0.05)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg)
    geometry = _resource_bytes(p.geometry_forward, p.geometry)
    assert geometry > 0
    from trainer.backend.memory.budget import allocated_bytes
    reported = gpu_allocation(p)["total_bytes"]
    assert reported >= sum(allocated_bytes(p.buffers).values()) + geometry
    shape = dict(max_splats=16, width=W, height=H, sh_degree=0)
    planned = (estimate_pipeline_bytes(p.shared_data_module, geometry_config=cfg, **shape)
               - estimate_pipeline_bytes(p.shared_data_module, **shape))
    assert 0.8 * geometry <= planned <= 1.25 * geometry, (planned, geometry)


def test_geometry_before_geom_from_composites_nothing():
    """Before --geom-from the image terms are off: no feature pass, no composite, no loss passes; dL_dAux reads
    zero for the backward, and the counters are not touched."""
    cfg = GeometryConfig(depth_normal_weight=0.05)
    p = _pipeline(_device(GEOM_PRIORS=True), cfg)
    p.set_primitive_from_numpy(**_scene())
    p.targets.set_image(np.ones((H, W, 3), np.float32))
    sentinel = np.full((H, W, 4), 7.0, np.float32)
    p.geometry_forward.aux_image.copy_from_numpy(sentinel)
    p.geometry.dL_dAux.copy_from_numpy(sentinel)
    p.geometry.stats.copy_from_numpy(np.full(2 * 17, 3, np.uint32)[:p.geometry.stats.size // 4])
    before = p.geometry.stats.to_numpy().copy()
    p.geometry.active = False
    _forward(p, _camera(fov=60.0))
    assert np.array_equal(p.geometry_forward.aux_image.to_numpy(), sentinel)       # nothing composited
    assert np.count_nonzero(p.geometry.dL_dAux.to_numpy()) == 0                     # the backward reads zero
    assert np.array_equal(p.geometry.stats.to_numpy(), before)
    p.geometry.active = True
    _forward(p, _camera(fov=60.0))
    assert not np.array_equal(p.geometry_forward.aux_image.to_numpy(), sentinel)   # the control


def test_zero_grad_clears_the_geometry_seeds():
    p = _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(depth_normal_weight=0.05))
    for b in p.geometry.seeds.values():
        b.copy_from_numpy(np.full(2 * p.max_splats, 5, np.uint32))
    p.optimizer.zero_grad()
    p.runtime.wait_for_idle()
    assert all(np.count_nonzero(v) == 0 for v in p.geometry.read_seeds().values())


def test_skipped_depth_frame_logs_no_depth_loss():
    """A frame whose scale-shift fit was skipped still counts its depth pixels; the log must say '-', not 0."""
    w = h = 32
    u, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    D = 2.0 + u
    depth = np.rint(1000.0 * (5.0 - 2.0 * u)).astype(np.uint16)      # inverted: the fit is skipped
    A = np.full((h, w), 0.9, np.float32)
    F = np.zeros((h, w, 4), np.float32)
    F[..., 2] = -0.9
    F[..., 3] = A * D
    p = _pipeline(_device(GEOM_PRIORS=True), GeometryConfig(depth_weight=1.0, depth_space="ssi"), w=w, h=h)
    _, stats = _run_loss(p, _camera(w, h, 60.0), F, A, depth, None)
    assert stats["V_d"] > 0 and stats["depth_on"] == 0
    assert "depth - (" in p.geometry.log_line(stats)


def test_geometry_buffers_pass_the_storage_range_guard(monkeypatch):
    """The geometry buffers are created outside the allocator; they take the same 4 GiB storage-buffer guard."""
    from trainer.backend.memory import buffers
    device = _device(GEOM_PRIORS=True)
    from trainer.backend.render.geometry import GeometryForward
    p = _pipeline(device, GeometryConfig(depth_normal_weight=0.05))
    monkeypatch.setattr(buffers, "STORAGE_BUFFER_RANGE", 64)
    GeometryForward(device, p.kernels, width=W, height=H, capacity=4)    # 64 B of geom_data: fits
    with pytest.raises(ValueError, match="geom_data"):
        GeometryForward(device, p.kernels, width=W, height=H, capacity=16)
