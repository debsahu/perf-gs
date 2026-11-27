# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Deterministic densification: child slots come from prefix sums in parent-index order.

Checked against a CPU reference of the slot layout (structure-aware and simple), the stated
cap-full policy (lowest parent indices first; ranked cutoff-bin upgrades in index order), an
order-independence test that feeds the split kernel a permuted parent list (a stand-in for a
different workgroup order), and repeated runs on a large population. Vulkan tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from test_radix_sort import _radix_sort, radix_kernels  # noqa: F401  (fixture)
from test_structure_densify import _count, _read_gaussians, _setup
from trainer.config.config import DensifyConfig, StructureConfig

SPACING = 10.0          # parents far apart, so a child's owner is its nearest parent


def _u32(buf, n=None):
    a = np.frombuffer(buf.to_numpy().tobytes(), dtype=np.uint32)
    return a if n is None else a[:n]


def _population(p, n, seed=0, *, clone_fraction=0.3, keep_fraction=0.2, eta_choices=(0.5, 1.5, 2.5, 4.0)):
    """n splats on a line, each triggered (or not) with random per-axis eta: a mix of keeps,
    clones (small scale) and splits with different grids."""
    rng = np.random.default_rng(seed)
    small = rng.random(n) < clone_fraction
    scales = np.where(small, 0.02, 0.3).astype(np.float32)
    p.set_primitive_from_numpy(
        means=np.column_stack([np.arange(n) * SPACING, np.zeros(n), np.full(n, 3.0)]).astype(np.float32),
        log_scales=np.log(np.repeat(scales[:, None], 3, axis=1)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full(n, 0.7, np.float32),
        color_logits=rng.normal(size=(n, 3)).astype(np.float32))
    triggered = rng.random(n) >= keep_fraction
    b, cap = p.buffers, p.max_splats

    def full(values, dtype=np.uint32):
        a = np.zeros(cap, dtype)
        a[:n] = values
        return a
    total = full(np.where(triggered, 10, 0))
    b.eta.n_total.copy_from_numpy(total)
    b.eta.components["n_high"].copy_from_numpy(total)
    b.eta.components["n_low"].copy_from_numpy(np.zeros(cap, np.uint32))
    for axis in ("max_x", "max_y", "max_z"):
        b.eta.components[axis].copy_from_numpy(full((rng.choice(eta_choices, n) * 65536).astype(np.uint32)))
    b.densification.visibility_count.copy_from_numpy(full(np.ones(n)))
    b.densification.grad_mean_2d_mag.copy_from_numpy(full(np.where(triggered, 65536, 0), np.int64))
    return _read_gaussians(p, n).copy()


def _structure_reference(parents, actions, factors, cap, ks):
    """The expected buffer after a structure densify with no prune: rows by slot, and the live count."""
    n = len(parents)
    out = {i: parents[i].copy() for i in range(n)}
    tail = n
    for i in range(n):
        a = actions[i]
        if a == 2:                                      # clone
            if tail < cap:
                out[tail] = parents[i].copy()
            tail += 1
        elif a == 1:                                    # split
            f = np.maximum(factors[i], 1)
            total = int(np.prod(f))
            if total > 1:
                headroom = cap - tail if tail < cap else 0
                count = 0 if headroom == 0 else 1 + min(total - 1, headroom)
                sep = np.exp(parents[i, 3:6]) / f * 3.4641016151377544
                for o in range(count):
                    idx = np.array([o % f[0], (o // f[0]) % f[1], o // (f[0] * f[1])])
                    child = parents[i].copy()
                    child[3:6] = parents[i, 3:6] - ks * np.log(f.astype(np.float32))
                    child[:3] = parents[i, :3] + (idx - 0.5 * (f - 1)) * sep
                    out[i if o == 0 else tail + o - 1] = child
            tail += total - 1
    live = min(tail, cap)
    return np.stack([out[j] for j in range(live)]), live


def _densify_structure(p, **cfg):
    cfg.setdefault("cap_order", "index")                # the CPU references below are index order
    return p.structure_densifier.densify(StructureConfig(opacity_cap=0, dense_scale=0.05, **cfg))


@pytest.mark.parametrize("cap", [4000, 1500, 1000], ids=["roomy", "cap-full", "cap-at-parents"])
def test_structure_slots_match_cpu_reference(gpu_device, cap) -> None:
    """Every slot holds the child the index-order reference puts there, cap-full included:
    lowest parent indices get their children first, the boundary family is truncated, and a
    split with no room keeps its parent intact."""
    n = 1000
    p = _setup(gpu_device, max_splats=cap)
    parents = _population(p, n, seed=1)
    _densify_structure(p)
    actions = _u32(p.buffers.densification.actions, n)
    factors = np.stack([_u32(getattr(p.buffers.structure, f"split_factor_{a}"), n) for a in "xyz"], axis=1)
    want, live = _structure_reference(parents, actions, factors, cap, StructureConfig().ks_scale_power)
    assert {1, 2} <= set(actions.tolist()) and len(set(map(tuple, factors[actions == 1]))) > 3
    assert _count(p) == live
    got = _read_gaussians(p, live)
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-5)
    if cap < 4000:
        assert live == cap                              # the budget really bound


def test_reference_discriminates_order(gpu_device) -> None:
    """The fixture can fail: two families swapped in the tail no longer match the reference."""
    n = 1000
    p = _setup(gpu_device, max_splats=4000)
    parents = _population(p, n, seed=1)
    _densify_structure(p)
    actions = _u32(p.buffers.densification.actions, n)
    factors = np.stack([_u32(getattr(p.buffers.structure, f"split_factor_{a}"), n) for a in "xyz"], axis=1)
    want, live = _structure_reference(parents, actions, factors, 4000, StructureConfig().ks_scale_power)
    swapped = want.copy()
    swapped[[n, n + 1]] = swapped[[n + 1, n]]
    assert not np.allclose(swapped, want, atol=2e-5)


def _one_axis(p) -> None:
    for axis in ("max_y", "max_z"):                     # each family then costs one child
        p.buffers.eta.components[axis].copy_from_numpy(np.zeros(p.max_splats, np.uint32))


@pytest.mark.parametrize("n,room", [(6000, 3000), (20000, 7001)])
def test_ranked_cutoff_admits_lowest_indices(gpu_device, n, room) -> None:
    """n identical parents in one priority bin over many workgroups, room for `room` children:
    exactly parents 0..room-1 split, in that order, and the rest keep their parent."""
    p = _setup(gpu_device, max_splats=n + room)
    parents = _population(p, n, seed=2, clone_fraction=0, keep_fraction=0, eta_choices=(1.5,))
    _one_axis(p)
    _densify_structure(p, rank_split_families=True)
    actions = _u32(p.buffers.densification.actions, n)
    assert (actions[:room] == 1).all() and (actions[room:] == 0).all(), \
        (int((actions == 1).sum()), np.flatnonzero(actions[:room] != 1)[:5], np.flatnonzero(actions[room:] == 1)[:5])
    assert _count(p) == n + room
    owners = np.rint(_read_gaussians(p, n + room)[n:, 0] / SPACING).astype(int)
    np.testing.assert_array_equal(owners, np.arange(room))
    np.testing.assert_array_equal(_read_gaussians(p, n)[room:], parents[room:])


def test_ranked_cutoff_is_all_or_nothing_per_parent(gpu_device) -> None:
    """Six identical (2,2,1) families: level 1 costs 1 each, level 2 costs 2 each. Room for all six level-1
    upgrades plus 3 slots: the cutoff bin is level 2 with 3 slots left, so parent 0 (0 + 2 <= 3) gets its
    second axis, parent 1 (2 + 2 > 3) does not, nor any later parent."""
    n = 6
    p = _setup(gpu_device, max_splats=n + n + 3)
    _population(p, n, seed=2, clone_fraction=0, keep_fraction=0, eta_choices=(1.5,))
    p.buffers.eta.components["max_z"].copy_from_numpy(np.zeros(p.max_splats, np.uint32))
    _densify_structure(p, rank_split_families=True)
    f = np.stack([_u32(getattr(p.buffers.structure, f"split_factor_{a}"), n) for a in "xyz"], axis=1)
    assert f.prod(axis=1).tolist() == [4, 2, 2, 2, 2, 2], f.tolist()


def _trigger(p, mask) -> None:
    """Retrigger exactly the splats in mask (one-axis eta 1.5), clearing every other accumulator."""
    cap, n = p.max_splats, len(mask)

    def full(values, dtype=np.uint32):
        a = np.zeros(cap, dtype)
        a[:n] = values
        return a
    b = p.buffers
    total = full(np.where(mask, 10, 0))
    b.eta.n_total.copy_from_numpy(total)
    b.eta.components["n_high"].copy_from_numpy(total)
    b.eta.components["n_low"].copy_from_numpy(np.zeros(cap, np.uint32))
    for axis in ("max_x", "max_y", "max_z"):
        b.eta.components[axis].copy_from_numpy(full(np.full(n, int(1.5 * 65536), np.uint32)))
    b.densification.visibility_count.copy_from_numpy(full(np.ones(n)))
    b.densification.grad_mean_2d_mag.copy_from_numpy(full(np.where(mask, 65536, 0), np.int64))


def test_ranked_second_tick_ignores_stale_cutoff_scratch(gpu_device) -> None:
    """Tick 1 leaves nonzero offsets in the scratch the cutoff cost reuses. Tick 2 triggers parents
    1, 3, 5, 7 with room for 3: stale entries of the non-parents must not eat the cutoff budget."""
    n, room = 8, 3
    p = _setup(gpu_device, max_splats=n + room)
    _population(p, n, seed=2, clone_fraction=0, keep_fraction=0, eta_choices=(1.5,))
    _one_axis(p)
    _densify_structure(p, rank_split_families=True)
    assert _count(p) == n + room
    _population(p, n, seed=2, clone_fraction=0, keep_fraction=0, eta_choices=(1.5,))
    _trigger(p, np.arange(n) % 2 == 1)
    _one_axis(p)
    _densify_structure(p, rank_split_families=True)
    assert _u32(p.buffers.densification.actions, n).tolist() == [0, 1, 0, 1, 0, 1, 0, 0]
    assert _count(p) == n + room


def _simple_population(p, n, seed=0):
    rng = np.random.default_rng(seed)
    big = rng.random(n) < 0.5
    scales = np.where(big, 0.5, 0.01).astype(np.float32)
    p.set_primitive_from_numpy(
        means=np.column_stack([np.arange(n) * SPACING, np.zeros(n), np.full(n, 3.0)]).astype(np.float32),
        log_scales=np.log(np.repeat(scales[:, None], 3, axis=1)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full(n, 0.7, np.float32), color_logits=np.zeros((n, 3), np.float32))
    hot = rng.random(n) < 0.7
    cap = p.max_splats
    vis = np.zeros(cap, np.uint32)
    vis[:n] = 1
    grad = np.zeros(cap, np.int64)
    grad[:n] = np.where(hot, 65536 * 1000, 0)
    p.buffers.densification.visibility_count.copy_from_numpy(vis)
    p.buffers.densification.grad_mean_2d_mag.copy_from_numpy(grad)
    return _read_gaussians(p, n).copy(), np.flatnonzero(hot), big


@pytest.mark.parametrize("cap", [2000, 1200], ids=["roomy", "cap-full"])
def test_simple_slots_match_cpu_reference(gpu_device, cap) -> None:
    """Tail slot pre + k belongs to the k-th split/clone parent in index order; past the cap a
    split keeps its parent and a clone is dropped."""
    n = 1000
    p = _setup(gpu_device, max_splats=cap)
    parents, hot, big = _simple_population(p, n, seed=3)
    p.simple_densifier.run(DensifyConfig(cap_order="index"), grad_thresh=0.0002, scale_thresh=0.05, allow_scale_prune=False)
    actions = _u32(p.buffers.densification.actions, n)
    assert big[hot].any() and (~big[hot]).any()         # both splits and clones requested
    expected_owner = hot[:max(0, cap - n)]
    live = n + len(expected_owner)
    assert _count(p) == live
    got = _read_gaussians(p, live)
    owners = np.rint(got[n:, 0] / SPACING).astype(int)
    np.testing.assert_array_equal(owners, expected_owner)
    fit, dropped = hot[:len(expected_owner)], hot[len(expected_owner):]
    np.testing.assert_array_equal(actions[fit], np.where(big[fit], 1, 2))
    # past the cap a split falls back to KEEP with its parent intact; a clone is dropped
    np.testing.assert_array_equal(actions[dropped], np.where(big[dropped], 0, 2))
    np.testing.assert_array_equal(got[dropped[big[dropped]]], parents[dropped[big[dropped]]])
    if cap < n + len(hot):
        assert big[dropped].any() and (~big[dropped]).any()


def _buffer(device, values):
    return device.create_buffer(size=values.nbytes, struct_size=4, data=values,
                                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)


@pytest.mark.parametrize("order", ["reversed", "shuffled"])
def test_split_output_ignores_parent_list_order(gpu_device, monkeypatch, order) -> None:
    """Feeding the split kernel the same parents in another order (so its workgroups run in
    another order) leaves every slot unchanged."""
    n, cap = 1000, 4000

    def run(permute):
        p = _setup(gpu_device, max_splats=cap)
        _population(p, n, seed=4)
        if permute is not None:
            commands = p.structure_densifier._commands
            original = commands.dispatch_indirect

            def dispatch_indirect(kernel, name, args, **kwargs):
                if name == "split_anisotropic":
                    kwargs["selected_parents"] = permute
                return original(kernel, name, args, **kwargs)
            monkeypatch.setattr(commands, "dispatch_indirect", dispatch_indirect)
        _densify_structure(p)
        monkeypatch.undo()
        return p, _read_gaussians(p, _count(p)).copy()

    p, reference = run(None)
    actions = _u32(p.buffers.densification.actions, n)
    parents = np.flatnonzero((actions == 1) | (actions == 2)).astype(np.uint32)
    rng = np.random.default_rng(5)
    listed = parents[::-1].copy() if order == "reversed" else rng.permutation(parents)
    assert not np.array_equal(listed, parents)
    full = np.zeros(cap, np.uint32)
    full[:len(listed)] = listed
    _, permuted = run(_buffer(gpu_device, full))
    assert permuted.shape == reference.shape
    np.testing.assert_array_equal(permuted, reference)


def test_large_population_repeats_bit_identically(gpu_device) -> None:
    """Many workgroups racing over 20,000 parents give the same bytes every time."""
    n, cap = 20_000, 60_000
    p = _setup(gpu_device, max_splats=cap)
    hashes = set()
    for _ in range(8):
        _population(p, n, seed=6)
        _densify_structure(p)
        count = _count(p)
        values = p.buffers.primitive.values.to_numpy().view(np.uint8).reshape(cap, -1)[:count]
        hashes.add((count, values.tobytes()))
    assert len(hashes) == 1
    assert next(iter(hashes))[0] == cap                 # the cap bound, so the boundary family was cut


def _owner_check(p, n, cap):
    """Tail slot k must hold a child of the k-th unit of growth in parent-index order."""
    actions = _u32(p.buffers.densification.actions, n)
    fx, fy, fz = (_u32(getattr(p.buffers.structure, f"split_factor_{a}"), n) for a in "xyz")
    grid = np.maximum(fx, 1).astype(np.int64) * np.maximum(fy, 1) * np.maximum(fz, 1)
    growth = np.where(actions == 2, 1, np.where(actions == 1, grid - 1, 0))
    total = int(growth.sum())
    live = min(n + total, cap)
    assert _count(p) == live, (_count(p), live)
    owners_expected = np.repeat(np.arange(n), growth)[: live - n]
    owners_got = np.rint(_read_gaussians(p, live)[n:live, 0] / SPACING).astype(np.int64)
    bad = np.flatnonzero(owners_got != owners_expected)
    assert bad.size == 0, f"{bad.size} tail slots hold another parent's child; first {bad[:5]}"
    return total


@pytest.mark.parametrize("cap,keep_fraction", [(3_500_000, 0.2), (2_000_000, 0.85)], ids=["cap-binds", "roomy"])
def test_three_level_scan_layout(gpu_device, cap, keep_fraction) -> None:
    """1.15M splats: the prefix sum needs three levels and parents sit past index 1,048,576. The roomy case
    triggers fewer parents instead of a bigger cap, so it fits an 8 GB GPU."""
    n = 1_150_000
    p = _setup(gpu_device, max_splats=cap)
    _population(p, n, seed=11, keep_fraction=keep_fraction)
    _densify_structure(p)
    total = _owner_check(p, n, cap)
    assert (n + total > cap) == (cap == 3_500_000)
    actions = _u32(p.buffers.densification.actions, n)
    assert np.flatnonzero((actions == 1) | (actions == 2)).max() > 1_048_576


@pytest.mark.parametrize("cap", [4000, 1500], ids=["roomy", "cap-full"])
def test_sh_structure_layout_matches_reference(gpu_device, cap) -> None:
    n = 1000
    p = _setup(gpu_device, max_splats=cap, sh_degree=3)
    parents = _population(p, n, seed=1)
    _densify_structure(p)
    actions = _u32(p.buffers.densification.actions, n)
    factors = np.stack([_u32(getattr(p.buffers.structure, f"split_factor_{a}"), n) for a in "xyz"], axis=1)
    want, live = _structure_reference(parents, actions, factors, cap, StructureConfig().ks_scale_power)
    assert _count(p) == live
    np.testing.assert_allclose(_read_gaussians(p, live), want, rtol=0, atol=2e-5)


@pytest.mark.parametrize("cap", [2000, 1200], ids=["roomy", "cap-full"])
def test_sh_simple_layout_matches_reference(gpu_device, cap) -> None:
    n = 1000
    p = _setup(gpu_device, max_splats=cap, sh_degree=3)
    _, hot, _ = _simple_population(p, n, seed=3)
    p.simple_densifier.run(DensifyConfig(cap_order="index"), grad_thresh=0.0002, scale_thresh=0.05, allow_scale_prune=False)
    expected_owner = hot[:max(0, cap - n)]
    live = n + len(expected_owner)
    assert _count(p) == live
    owners = np.rint(_read_gaussians(p, live)[n:, 0] / SPACING).astype(int)
    np.testing.assert_array_equal(owners, expected_owner)


def test_stale_actions_past_pre_count_are_not_parents(gpu_device) -> None:
    """Tick 1 on 1,000 triggered splats leaves split/clone actions at 300..999; tick 2 has 300 splats."""
    cap = 6000
    p = _setup(gpu_device, max_splats=cap)
    _population(p, 1000, seed=21, keep_fraction=0.0)
    _densify_structure(p)
    stale = _u32(p.buffers.densification.actions, 1000)[300:]
    assert ((stale == 1) | (stale == 2)).any()
    _population(p, 300, seed=22)
    _densify_structure(p)
    _owner_check(p, 300, cap)


# ---------------------------------------------------------------- overflow

def test_prefix_scan_saturates_instead_of_wrapping(gpu_device) -> None:
    """Offsets past 0xFFFFFFFF stay 0xFFFFFFFF, across a block boundary too."""
    from trainer.backend.compute.execution import SCAN_BLOCK_SIZE
    p = _setup(gpu_device, max_splats=4 * SCAN_BLOCK_SIZE)
    n = 3 * SCAN_BLOCK_SIZE
    values = np.zeros(p.max_splats, np.uint32)
    values[0], values[1], values[5] = 0xF0000000, 0x20000000, 7
    values[SCAN_BLOCK_SIZE + 3] = 1
    buf = p.buffers.densification.child_offset
    buf.copy_from_numpy(values)
    scan = p.structure_densifier._scan
    with p.commands.recording("scan_test"):
        scan.scan(n, buf, buf, p.buffers.densification.post_count)
    p.runtime.wait_for_idle()
    out = _u32(buf, n)
    assert out[0] == 0 and out[1] == 0xF0000000
    assert (out[2:] == 0xFFFFFFFF).all()
    assert _u32(p.buffers.densification.post_count)[0] == 0xFFFFFFFF


def test_huge_split_grid_does_not_wrap(gpu_device) -> None:
    """A 2048 x 2048 x 1024 grid is exactly 2**32 children: a wrapping product reads as a 0-child grid
    and skips the split. Saturated, the family fills the cap."""
    cap = 600
    p = _setup(gpu_device, max_splats=cap)
    _population(p, 1, seed=1, clone_fraction=0, keep_fraction=0)
    for axis, eta in (("max_x", 2048), ("max_y", 2048), ("max_z", 1024)):
        e = np.zeros(cap, np.uint32)
        e[0] = eta * 65536
        p.buffers.eta.components[axis].copy_from_numpy(e)
    _densify_structure(p, max_split_factor=2048, split_power=1.0)
    factors = [int(_u32(getattr(p.buffers.structure, f"split_factor_{a}"), 1)[0]) for a in "xyz"]
    assert factors == [2048, 2048, 1024]
    assert _count(p) == cap
    assert np.isfinite(_read_gaussians(p, cap)).all()


def test_tail_offset_past_the_cap_writes_nothing(gpu_device, monkeypatch) -> None:
    """A parent whose offset is near 0xFFFFFFFF must not wrap pre_count + offset onto a live slot."""
    n, cap = 200, 4000
    p = _setup(gpu_device, max_splats=cap)
    parents = _population(p, n, seed=7, clone_fraction=0, keep_fraction=0)
    poisoned = np.full(cap, 0xFFFFFF40, np.uint32)       # pre_count + offset would wrap to slot 8
    commands = p.structure_densifier._commands
    original = commands.dispatch_indirect

    def dispatch_indirect(kernel, name, args, **kwargs):
        if name == "split_anisotropic":
            kwargs["child_offset"] = _buffer(gpu_device, poisoned)
        return original(kernel, name, args, **kwargs)
    monkeypatch.setattr(commands, "dispatch_indirect", dispatch_indirect)
    _densify_structure(p)
    got = _read_gaussians(p, n)
    owners = np.rint(got[:, 0] / SPACING).astype(int)
    np.testing.assert_array_equal(owners, np.arange(n))  # no parent's slot holds another parent's child
    np.testing.assert_array_equal(got, parents)          # no room anywhere, so every parent is intact


# ---------------------------------------------------------------- radix sort ties

def test_radix_sort_ties_keep_input_order_and_repeat(radix_kernels) -> None:
    """Few distinct 64-bit keys over many partitions: equal keys keep their input order, every time."""
    device, kernels = radix_kernels
    n = 300_000
    rng = np.random.default_rng(3)
    keys_hi = rng.integers(0, 3, size=n, dtype=np.uint32)
    keys_lo = rng.integers(0, 5, size=n, dtype=np.uint32)
    values = np.arange(n, dtype=np.uint32)
    order = np.argsort((keys_hi.astype(np.uint64) << np.uint64(32)) | keys_lo, kind="stable")
    for _ in range(3):
        _, _, got_v = _radix_sort(device, kernels, keys_lo, keys_hi, values)
        np.testing.assert_array_equal(got_v, values[order])


# ---------------------------------------------------------------- cap order: hash thinning

def _pcg(v):
    v = np.asarray(v, dtype=np.uint64) & 0xFFFFFFFF
    state = (v * 747796405 + 2891336453) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF
    return (word >> 22) ^ word


def _hash_bins(n, tick, seed):
    return (_pcg(np.arange(n, dtype=np.uint64) ^ _pcg(np.uint64(tick) ^ _pcg(np.uint64(seed)))) >> 24).astype(np.int64)


def _hash_admitted(cost, budget, tick, seed):
    """CPU reference of the hash-order admission: lowest hash bins whole, the cutoff bin in index order."""
    bins = _hash_bins(len(cost), tick, seed)
    hist = np.bincount(bins, weights=cost, minlength=256).astype(np.int64)
    slots, cutoff = int(budget), 0
    while cutoff < 256 and hist[cutoff] <= slots:
        slots -= hist[cutoff]
        cutoff += 1
    admitted = (cost > 0) & (bins < cutoff)
    prefix = 0
    for i in np.flatnonzero((cost > 0) & (bins == cutoff)):
        if prefix + cost[i] <= slots:
            admitted[i] = True
        prefix += cost[i]
    return admitted | (cost == 0)


def _growth(p, n, actions_before=None):
    actions = _u32(p.buffers.densification.actions, n) if actions_before is None else actions_before
    fx, fy, fz = (_u32(getattr(p.buffers.structure, f"split_factor_{a}"), n) for a in "xyz")
    grid = np.maximum(fx, 1).astype(np.int64) * np.maximum(fy, 1) * np.maximum(fz, 1)
    return np.where(actions == 2, 1, np.where(actions == 1, grid - 1, 0))


@pytest.mark.parametrize("seed,tick", [(0, 300), (7, 1900)])
def test_hash_thinning_matches_cpu_reference(gpu_device, seed, tick) -> None:
    """Unranked, the cap binding hard: exactly the parents the CPU hash order admits keep their children,
    laid out in index order; the rest keep their parent; the budget is never exceeded."""
    n, cap = 4000, 7000
    roomy = _setup(gpu_device, max_splats=40_000)
    _population(roomy, n, seed=5)
    _densify_structure(roomy, cap_order="index")
    cost = _growth(roomy, n)                             # what each parent asks for, before any thinning
    assert n + cost.sum() > cap
    p = _setup(gpu_device, max_splats=cap)
    parents = _population(p, n, seed=5)
    p.structure_densifier.densify(StructureConfig(opacity_cap=0, dense_scale=0.05, cap_order="hash",
                                                  cap_seed=seed), tick=tick)
    want = _hash_admitted(cost, cap - n, tick, seed)
    actions = _u32(p.buffers.densification.actions, n)
    got = (actions == 1) | (actions == 2) | (cost == 0)
    np.testing.assert_array_equal(got, want)
    kept_growth = int(cost[want & (cost > 0)].sum())
    assert kept_growth <= cap - n and cap - n - kept_growth <= cost.max()     # leftover < one family
    assert _count(p) == n + kept_growth
    owners = np.rint(_read_gaussians(p, n + kept_growth)[n:, 0] / SPACING).astype(int)
    np.testing.assert_array_equal(owners, np.repeat(np.arange(n), np.where(want, cost, 0)))
    dropped = np.flatnonzero(~want)
    np.testing.assert_array_equal(_read_gaussians(p, n)[dropped], parents[dropped])


def test_hash_reference_discriminates(gpu_device) -> None:
    """The reference can fail: another seed or tick admits a different set, and index order is not it."""
    cost = np.ones(5000, np.int64)
    a = _hash_admitted(cost, 1500, 300, 0)
    assert a.sum() == 1500
    assert not np.array_equal(a, _hash_admitted(cost, 1500, 400, 0))
    assert not np.array_equal(a, _hash_admitted(cost, 1500, 300, 1))
    assert not np.array_equal(a, np.arange(5000) < 1500)


def _decile_shares(indices, n):
    return np.bincount(np.minimum(indices * 10 // n, 9), minlength=10) / max(len(indices), 1)


@pytest.mark.parametrize("mode", ["structure", "simple"])
def test_hash_survivors_spread_over_the_index_range(gpu_device, mode) -> None:
    """Index order is Morton order after a reorder. A quarter of 20,000 clone requests fit: index order
    keeps the first 25% of the range, hash order keeps every tenth of the range nearly equally."""
    n, room = 20_000, 5_000
    shares = {}
    for order in ("index", "hash"):
        p = _setup(gpu_device, max_splats=n + room)
        if mode == "structure":
            _population(p, n, seed=8, clone_fraction=1.0, keep_fraction=0.0)
            _densify_structure(p, cap_order=order)
            kept = np.rint(_read_gaussians(p, _count(p))[n:, 0] / SPACING).astype(int)
        else:
            _simple_population(p, n, seed=8)
            grad = np.zeros(p.max_splats, np.int64)
            grad[:n] = 65536 * 1000
            p.buffers.densification.grad_mean_2d_mag.copy_from_numpy(grad)
            p.simple_densifier.run(DensifyConfig(cap_order=order), grad_thresh=0.0002, scale_thresh=10.0,
                                   allow_scale_prune=False)            # every splat asks to clone
            owners = np.rint(_read_gaussians(p, _count(p))[n:, 0] / SPACING).astype(int)
            kept = owners
        assert len(kept) == room
        shares[order] = _decile_shares(kept, n)
    assert shares["index"][:2].sum() > 0.79 and shares["index"][3:].sum() == 0
    assert shares["hash"].min() > 0.08 and shares["hash"].max() < 0.12, shares["hash"]


def test_hash_ranked_cutoff_is_all_or_nothing_and_spread(gpu_device) -> None:
    """Ranked, hash order inside the cutoff bin: exactly `room` single-child families split, spread over
    the range; and at a cost of two per parent no family is cut."""
    n, room = 6000, 3000
    p = _setup(gpu_device, max_splats=n + room)
    _population(p, n, seed=2, clone_fraction=0, keep_fraction=0, eta_choices=(1.5,))
    _one_axis(p)
    _densify_structure(p, rank_split_families=True, cap_order="hash")
    actions = _u32(p.buffers.densification.actions, n)
    split = np.flatnonzero(actions == 1)
    assert len(split) == room and set(np.unique(actions)) == {0, 1}
    assert _decile_shares(split, n).min() > 0.08
    assert _count(p) == n + room
    q = _setup(gpu_device, max_splats=6 + 6 + 3)
    _population(q, 6, seed=2, clone_fraction=0, keep_fraction=0, eta_choices=(1.5,))
    q.buffers.eta.components["max_z"].copy_from_numpy(np.zeros(q.max_splats, np.uint32))
    _densify_structure(q, rank_split_families=True, cap_order="hash")
    f = np.stack([_u32(getattr(q.buffers.structure, f"split_factor_{a}"), 6) for a in "xyz"], axis=1)
    assert sorted(f.prod(axis=1).tolist()) == [2, 2, 2, 2, 2, 4], f.tolist()


def test_hash_thinning_repeats_and_never_exceeds_the_cap(gpu_device) -> None:
    """20,000 mixed parents over a hard cap, eight times: one result, at most the cap."""
    n, cap = 20_000, 40_000
    p = _setup(gpu_device, max_splats=cap)
    results = set()
    for _ in range(8):
        _population(p, n, seed=6)
        p.structure_densifier.densify(StructureConfig(opacity_cap=0, dense_scale=0.05, cap_order="hash"), tick=500)
        count = _count(p)
        assert n < count <= cap
        values = p.buffers.primitive.values.to_numpy().view(np.uint8).reshape(cap, -1)[:count]
        results.add((count, values.tobytes()))
    assert len(results) == 1


# ---------------------------------------------------------------- review regressions

def _set_axis_eta(p, cap, values):
    for k, axis in enumerate(("max_x", "max_y", "max_z")):
        e = np.zeros(cap, np.uint32)
        for i, v in enumerate(values):
            e[i] = min(int(v[k] * 65536), 0xFFFFFFFF)
        p.buffers.eta.components[axis].copy_from_numpy(e)


def test_overbudget_is_reported_when_the_requested_sum_would_wrap(gpu_device) -> None:
    """Two 2048 x 1024 x 1024 splits request 2**31 - 1 children each and two clones 1 each: 2**32 in all, which
    wrapped to 0 and reported 'within budget' at a 600-splat cap. Each request is capped at the cap, as the slot
    allocation caps it, and the sum saturates."""
    cap = 600
    p = _setup(gpu_device, max_splats=cap)
    _population(p, 4, seed=1, clone_fraction=0, keep_fraction=0)
    scales = np.log(np.array([[0.3] * 3, [0.3] * 3, [0.02] * 3, [0.02] * 3], np.float32))
    g = _read_gaussians(p, 4)
    p.set_primitive_from_numpy(means=g[:, :3].copy(), log_scales=scales,
                               rot_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1)),
                               opacities=np.full(4, 0.7, np.float32), color_logits=np.zeros((4, 3), np.float32))
    _set_axis_eta(p, cap, [(2048, 1024, 1024), (2048, 1024, 1024), (2, 2, 2), (2, 2, 2)])
    result = _densify_structure(p, max_split_factor=2048, split_power=1.0)
    actions = _u32(p.buffers.structure.action, 4) if hasattr(p.buffers.structure, "action") else None
    factors = [[int(_u32(getattr(p.buffers.structure, f"split_factor_{a}"), 2)[i]) for a in "xyz"] for i in range(2)]
    assert factors == [[2048, 1024, 1024]] * 2 and (actions is None or list(actions[2:]) == [2, 2])
    requested = result["requested_children"]
    assert result["overbudget"] is True
    assert requested == 2 * cap + 2                     # min(2**31 - 1, cap) per split, 1 per clone


def test_child_grid_index_does_not_divide_by_a_wrapped_product(gpu_device) -> None:
    """With 65536 x 65536 x 1 factors, n.x * n.y wraps to 0 in 32 bits and ordinal / (n.x * n.y) divides by zero:
    the children's z index came out as garbage. n.z = 1, so every child must sit at the parent's z."""
    cap = 3000
    p = _setup(gpu_device, max_splats=cap)
    parent = _population(p, 1, seed=1, clone_fraction=0, keep_fraction=0)[0]
    _set_axis_eta(p, cap, [(65535.99, 65535.99, 1.0)])
    _densify_structure(p, max_split_factor=65536, split_power=1.0)
    factors = [int(_u32(getattr(p.buffers.structure, f"split_factor_{a}"), 1)[0]) for a in "xyz"]
    assert factors == [65536, 65536, 1]
    count = _count(p)
    children = _read_gaussians(p, count)
    assert count == cap and np.isfinite(children).all()
    np.testing.assert_allclose(children[:, 2], parent[2], atol=1e-4)


def test_priority_bin_puts_non_positive_and_nan_scores_last(gpu_device) -> None:
    """asuint(score) >> 20 has the sign bit set for a negative score and a high exponent for NaN: both landed in
    the top bin, ahead of every real score."""
    import slangpy as spy
    from trainer.backend.compute.device import SHADERS_DIR
    source = ('#include "densify/rank_families.slang"\n'
              'uint bin(float score) { return priorityBin(score); }\n')
    module = spy.Module.load_from_source(gpu_device, "priority_bin_probe", source)
    del SHADERS_DIR
    scores = [-1.0, -1e-30, 0.0, float("nan"), 1e-3, 1.0, 1e3, float("inf")]
    bins = [int(module.bin(s)) for s in scores]
    assert bins[:4] == [0, 0, 0, 0], bins
    assert 0 < bins[4] < bins[5] < bins[6] <= 255


def test_requested_children_saturate_past_32_bits(gpu_device) -> None:
    """70,000 parents at a 70,000 cap, each capped at the cap: 4.9e9 children, past 2**32 even after the per-parent
    cap. The report saturates at 0xFFFFFFFF instead of wrapping to a smaller number."""
    cap = 70_000
    p = _setup(gpu_device, max_splats=cap)
    _population(p, cap, seed=1, clone_fraction=0, keep_fraction=0)
    _set_axis_eta(p, cap, [(2048, 1024, 1024)] * cap)
    assert cap * cap > 2**32
    result = _densify_structure(p, max_split_factor=2048, split_power=1.0)
    assert result["requested_children"] == 0xFFFFFFFF
    assert result["overbudget"] is True
