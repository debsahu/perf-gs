# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""VRAM budget planning, streamed targets, sort-pair accounting, viewer growth."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import slangpy as spy
from _colmap_testkit import write_colmap_scene
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import create_device
from trainer.backend.densify.structure_tensor import StructureTensor
from trainer.backend.memory import budget
from trainer.backend.platform.dxgi import query_video_memory
from trainer.backend.render.pipeline import TrainingPipeline, default_sort_pairs, estimate_pipeline_bytes
from trainer.config.config import AdamLRs, LossConfig
from trainer.data.dataset import simple_camera
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser
from trainer.frontend.viewer.viewer import Viewer

GIB = budget.GIB
CAMERA = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                       target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), fovy_degrees=60.0)


@pytest.fixture(params=["vulkan", "d3d12"])
def device(request):
    try:
        device = create_device(backend=request.param)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"backend unavailable: {error}")
    yield device
    device.close()


def _three_splats(device, **kwargs) -> TestPipeline:
    pipeline = TestPipeline(device=device, width=16, height=16, max_splats=16,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
                            **kwargs)
    pipeline.set_primitive_from_numpy(
        means=np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]], np.float32),
        log_scales=np.log(np.full((3, 3), 0.3, np.float32)),
        rot_quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (3, 1)),
        opacities=np.array([0.8, 0.6, 0.7], np.float32),
        color_logits=np.array([[1.5, -1.5, -1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5]],
                              np.float32))
    pipeline.optimizer.zero_grad()
    return pipeline


def test_preset_sort_workspace() -> None:
    parser = build_parser()
    for preset, expected in (("ultrafast", 16_776_960), ("balanced", 16_776_960), ("quality", 24_000_000)):
        assert default_sort_pairs(parser.parse_args(["--preset", preset]).max_splats) == expected


def test_fit_capacity_is_the_largest_that_fits() -> None:
    def cost(capacity: int) -> int:
        return 1000 + 10 * capacity
    assert budget.fit_capacity(500, 10_000, cost) == 500
    assert budget.fit_capacity(5000, 10_000, cost) == 900
    assert budget.fit_capacity(5000, 999, cost) == 0


def test_available_bytes_reserves_headroom(monkeypatch) -> None:
    monkeypatch.setattr(budget, "query_video_memory", lambda name: (16 * GIB, GIB))
    assert budget.available_bytes(0, "gpu") == 16 * GIB - GIB - 16 * GIB // 10
    # An explicit budget replaces DXGI's but still subtracts what is in use.
    assert budget.available_bytes(8, "gpu") == 8 * GIB - GIB - 8 * GIB // 10
    monkeypatch.setattr(budget, "query_video_memory", lambda name: None)
    assert budget.available_bytes(0, "gpu") is None
    assert budget.available_bytes(4, "gpu") == 4 * GIB - GIB // 2


@pytest.mark.skipif(sys.platform != "win32", reason="DXGI is Windows-only")
def test_query_video_memory_reports_this_process() -> None:
    measured = query_video_memory("no such adapter")   # falls back to the largest GPU
    assert measured is not None
    assert measured[0] > 0 and measured[1] >= 0


def test_allocated_bytes_cover_the_planned_allocation(device) -> None:
    pipeline = _three_splats(device)
    owned = budget.allocated_bytes(pipeline.buffers)
    assert owned["raster.output_image"] >= 16 * 16 * 16
    planned = estimate_pipeline_bytes(pipeline.shared_data_module, max_splats=16,
                                      width=16, height=16, sh_degree=0,
                                      max_sort_pairs=pipeline.max_sort_pairs)
    assert sum(owned.values()) >= planned


def test_streamed_target_matches_resident_texture(device) -> None:
    target = np.zeros((16, 16, 3), np.float32)
    target[..., 0] = 0.5
    lrs = AdamLRs(mean=1e-2, scale=1e-2, rot=1e-3, opacity=1e-2, color=1e-2)
    results = []
    for streamed in (False, True):
        pipeline = _three_splats(device)
        packed = pipeline.targets.pack(target)
        pipeline.targets.select(
            packed if streamed else pipeline.targets.upload(packed))
        pipeline.train_step(CAMERA, lrs=lrs)
        device.wait_for_idle()
        results.append(pipeline.buffers.primitive.values.to_numpy().copy())
    np.testing.assert_array_equal(results[0], results[1])


def test_streamed_structure_fields_match_resident_texture(device) -> None:
    target = np.random.default_rng(0).uniform(0.0, 1.0, (16, 16, 3)).astype(np.float32)
    builder = StructureTensor(device, 16, 16)
    results = []
    for mode in ("resident", "derived"):
        pipeline = _three_splats(device)
        packed = pipeline.targets.pack(target)
        target_texture = pipeline.targets.upload(packed)
        field = device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba16_float,
            width=16, height=16,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access)
        batch = builder.runtime.command_batch("field")
        builder.record_into(batch.encoder, target_texture, field)
        builder.runtime.submit(batch)
        builder.runtime.wait_for_idle()
        structure = field if mode == "resident" else builder
        pipeline.targets.select(target_texture if mode == "resident" else packed)
        pipeline.train_step(CAMERA, structure_tensor=structure)
        device.wait_for_idle()
        eta = pipeline.buffers.eta
        results.append(np.stack([buffer.to_numpy() for buffer in eta.components.values()]))
    assert results[0].any()
    np.testing.assert_array_equal(results[0], results[1])


def test_sort_stats_count_demand_and_dropped_pairs(device) -> None:
    roomy = _three_splats(device, max_sort_pairs=1024)
    roomy.render(CAMERA)
    peak, dropped = roomy.buffers.sort.stats.to_numpy().view(np.uint32)
    assert peak > 2 and dropped == 0
    tight = _three_splats(device, max_sort_pairs=2)
    tight.render(CAMERA)
    assert list(tight.buffers.sort.stats.to_numpy().view(np.uint32)) == [peak, 1]


def test_viewer_grows_to_the_snapshot(device) -> None:
    train = _three_splats(device)
    viewer = Viewer(device, width=16, height=16, max_splats=1)
    viewer.update_gaussians(train)
    device.wait_for_idle()
    assert viewer.max_splats == 3
    shown = viewer._pipeline.buffers.primitive.active_count.to_numpy().view(np.uint32)
    assert shown[0] == 3
    viewer._pipeline.render(CAMERA)      # the resized buffers rasterize
    assert viewer._pipeline.read_output_image()[..., 3].max() > 0


def _visible_scene(root, size: int = 16, views: int = 4):
    """A COLMAP text model the cameras actually see: an 8 x 8 grid of points 3 units ahead of four views a small
    baseline apart, and textured photos (write_colmap_scene's points project outside a 16 px frame)."""
    import cv2
    sparse, images_dir = root / "sparse" / "0", root / "images"
    sparse.mkdir(parents=True)
    images_dir.mkdir()
    rng = np.random.default_rng(4)
    cameras, images = ["# camera list"], ["# image list"]
    for i in range(views):
        cameras.append(f"{i + 1} PINHOLE {size} {size} {size} {size} {size / 2} {size / 2}")
        images.append(f"{i + 1} 1 0 0 0 {0.1 * i} 0 0 {i + 1} {i}.png")
        images.append("0.0 0.0 -1")
        photo = cv2.GaussianBlur(rng.integers(0, 256, (size, size, 3)).astype(np.uint8), (3, 3), 0)
        assert cv2.imwrite(str(images_dir / f"{i}.png"), photo)
    rows = ["# 3D points"]
    g = np.linspace(-1.0, 1.0, 8)
    for k, (x, y) in enumerate((x, y) for x in g for y in g):
        r, gg, b = rng.integers(0, 256, 3)
        rows.append(f"{k} {x} {y} 3.0 {r} {gg} {b} 0.5 0")
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (sparse / "images.txt").write_text("\n".join(images) + "\n")
    (sparse / "points3D.txt").write_text("\n".join(rows) + "\n")
    return root


def _train_scene(device, tmp_path, monkeypatch, residency: str, masks: bool = False, iters: int = 12,
                 visible: bool = False, densify: bool = True) -> np.ndarray:
    """A short structure-aware run (densify every step) in one image residency; the final splat buffer. With
    ``masks`` every view drops a different quarter of its pixels (--masks: a LOSS_MASK device of its own)."""
    import cv2
    name = f"{residency}{'_masked' if masks else ''}"
    scene = (_visible_scene(tmp_path / name) if visible
             else write_colmap_scene(tmp_path / name, views=[(f"{i}.png", 16, 16, 20.0) for i in range(4)]))
    argv = [
        "--scene", str(scene), "--no-viewer", "--iters", str(iters), "--max-splats", "256" if visible else "64",
        "--width", "16", "--height", "16", "--eval-holdout-stride", "2",
        "--no-evaluation", "--no-write-ground-truths", "--preview-every", "0",
        "--densify-mode", "structure_aware" if densify else "none", "--densify-from", "0", "--densify-to", str(iters),
        "--densify-every", "1", "--sad-accum-every", "1", "--sad-structure-every", "2",
        "--image-residency", residency, "--run-report", "--run-name", name,
    ]
    if masks:
        mask_dir = tmp_path / f"{name}_masks"
        mask_dir.mkdir()
        for i in range(4):
            keep = np.full((16, 16), 255, np.uint8)
            keep[(i // 2) * 8:(i // 2 + 1) * 8, (i % 2) * 8:(i % 2 + 1) * 8] = 0
            assert cv2.imwrite(str(mask_dir / f"{i}.png"), keep)
        argv += ["--masks", str(mask_dir)]
    args = build_parser().parse_args(argv)
    own_device = None
    if masks:
        from trainer.frontend.factory import create_training_device
        own_device = create_training_device(args)
        device = own_device
    monkeypatch.setattr("trainer.frontend.app.create_training_device", lambda _args: device)
    # Earlier tests' devices count against this process's DXGI budget.
    monkeypatch.setattr("trainer.frontend.app.available_bytes", lambda *_: None)
    monkeypatch.setattr("trainer.frontend.factory.TrainingPipeline",
                        lambda **kwargs: TrainingPipeline(max_sort_pairs=512, **kwargs))
    app = TrainingApplication(args)
    app.prepare_training_context()
    try:
        assert app._stream_targets == (residency in ("stream", "compressed"))
        assert app.pipeline.loss_mask == masks
        app.pipeline.runtime.wait_for_idle()
        _train_scene.seed = app.pipeline.buffers.primitive.values.to_numpy().copy()
        app.build_training_loop().launch_training_loop()
        app.pipeline.runtime.wait_for_idle()
        memory = app._report_memory()
        ram, cache = memory["host_ram"], memory["gpu_view_cache"]
        assert ram["linear_cache_bytes"] == 0
        packed_bytes = 2 * 16 * 16 * 4
        if residency == "compressed":
            assert ram["decoded_images_bytes"] == 0 and ram["encoded_images_bytes"] > 0
            assert memory["target_cache"]["decodes"] > 0
        else:
            assert ram["decoded_images_bytes"] == 4 * 16 * 16 * 4      # u8_targets.patch: uint8 RGBA, not float RGB
        if residency == "stream":
            # the packed uint8 RGBA targets ARE the decoded arrays (one host copy): counted once, in decoded
            assert all(np.shares_memory(app.target_textures[i], app._targets.resident[i]) for i in app.target_textures)
            assert len(app.target_textures) * 16 * 16 * 4 == packed_bytes
        if residency in ("stream", "compressed"):
            assert ram["packed_training_images_bytes"] == 0
            assert cache == {"training_images_bytes": 0, "structure_fields_bytes": 0}
        else:
            assert ram["packed_training_images_bytes"] == 0
            assert cache["training_images_bytes"] >= packed_bytes
            assert cache["structure_fields_bytes"] >= 2 * packed_bytes
        if visible:
            assert float(app.pipeline.buffers.raster.output_image.to_numpy()[..., 3].max()) > 0.5   # it renders
        count = int(app.pipeline.buffers.primitive.active_count.to_numpy().view(np.uint32)[0])
        _train_scene.count = count
        return app.pipeline.buffers.primitive.values.to_numpy()[:count].copy()
    finally:
        app.runtime.artifacts.close_all()
        if own_device is not None:
            own_device.close()


def test_streamed_training_matches_resident(device, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resident = _train_scene(device, tmp_path, monkeypatch, "gpu")
    streamed = _train_scene(device, tmp_path, monkeypatch, "stream")
    np.testing.assert_array_equal(resident, streamed)


def test_viewer_writes_no_raster_checkpoints_and_renders_the_same(device) -> None:
    """Only the backward reads raster_checkpoints, and the viewer has none: it binds a one-element stub, the forward
    skips the writes, and the image is the training pipeline's, bit for bit."""
    from trainer.data.dataset import simple_camera
    camera = simple_camera(width=32, height=24, eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), fovy_degrees=60.0)
    rng = np.random.default_rng(3)
    n = 40
    scene = dict(means=np.column_stack([rng.uniform(-0.8, 0.8, (n, 2)), rng.uniform(2.0, 4.0, n)]).astype(np.float32),
                 log_scales=np.log(rng.uniform(0.05, 0.3, (n, 3))).astype(np.float32),
                 rot_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
                 opacities=rng.uniform(0.3, 0.9, n).astype(np.float32),
                 color_logits=rng.normal(size=(n, 3)).astype(np.float32))
    images, flags = [], []
    for viewer in (False, True):
        p = TrainingPipeline(device, 32, 24, max_splats=64, viewer_only=viewer)
        p.set_gaussians(**scene)
        # A write past a one-element stub is dropped by robust buffer access, so the image alone cannot show it:
        # check the forward is told not to write.
        original = p.rasterizer._commands.dispatch

        def dispatch(kernel, marker, **kwargs):
            if marker == "forward":
                flags.append(kwargs["write_checkpoints"])
            return original(kernel, marker, **kwargs)
        p.rasterizer._commands.dispatch = dispatch
        p.render(camera)
        images.append(p.read_output_image().copy())
        if viewer:
            assert p.buffers.raster.raster_checkpoints.storage.size == 8
    assert images[0][..., 3].max() > 0.2 and np.array_equal(images[0], images[1])
    assert flags == [1, 0]


def test_viewer_forward_never_writes_its_checkpoint_stub(device) -> None:
    """The shader gate itself: a frame-covering render reaches bucket 0 of tile 0, pixel 0, which is checkpoint
    index 0, the stub's only element. With the write gate in blend_forward.slang the sentinel survives; without it
    the viewer writes there (and on Metal past the stub)."""
    camera = simple_camera(width=32, height=24, eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), fovy_degrees=60.0)
    p = TrainingPipeline(device, 32, 24, max_splats=64, viewer_only=True)
    p.set_gaussians(means=np.array([[0.0, 0.0, 2.0]], np.float32), log_scales=np.log(np.full((1, 3), 3.0, np.float32)),
                    rot_quats=np.array([[1, 0, 0, 0]], np.float32), opacities=np.array([0.5], np.float32),
                    color_logits=np.zeros((1, 3), np.float32))
    stub = p.buffers.raster.raster_checkpoints
    sentinel = np.array([[0xDEADBEEF, 0xCAFEF00D]], np.uint32)
    stub.storage.copy_from_numpy(sentinel)
    p.render(camera)
    image = p.read_output_image()
    assert image[0, 0, 3] > 0.05                         # pixel 0 of tile 0 is covered: bucket 0 is written there
    np.testing.assert_array_equal(stub.storage.to_numpy().view(np.uint32).reshape(1, 2), sentinel)


def test_small_device_buffers_pass_the_storage_range_guard(device, monkeypatch) -> None:
    """The mip-filter cameras and the STBN table are created outside the allocator; they take the same guard."""
    from trainer.backend.memory import buffers
    from trainer.backend.render.stbn_mask import StbnMask
    p = TrainingPipeline(device, 16, 16, max_splats=16)
    monkeypatch.setattr(buffers, "STORAGE_BUFFER_RANGE", 16)
    with pytest.raises(ValueError, match="mip_cameras"):
        p.mip_filter.set_cameras([simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0))])
    spec = type("Spec", (), {"voxels": 8, "width": 2, "height": 2, "slices": 2})()
    with pytest.raises(ValueError, match="stbn_table"):
        StbnMask.from_table(device, np.zeros(8, np.uint32), spec)


@pytest.mark.parametrize("masks", [False, True], ids=["plain", "masked"])
def test_compressed_training_matches_stream(device, tmp_path, monkeypatch, masks) -> None:
    """A real run in compressed residency (decode cache, prefetch window, packer, upload) trains the same splats as
    the stream residency, bit for bit. Densification is off: before deterministic densify, two runs of one residency
    already differ once it runs, so a difference would not be the residency's."""
    monkeypatch.chdir(tmp_path)
    run = dict(masks=masks, visible=True, iters=40, densify=False)
    streamed = _train_scene(device, tmp_path, monkeypatch, "stream", **run)
    seed = _train_scene.seed
    compressed = _train_scene(device, tmp_path, monkeypatch, "compressed", **run)
    assert not np.array_equal(streamed, seed[:len(streamed)])     # the run trained: the splats moved
    np.testing.assert_array_equal(streamed, compressed)
