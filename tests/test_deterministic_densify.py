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


def test_ranked_cutoff_admits_lowest_indices(gpu_device) -> None:
    """Six identical parents in one priority bin, room for three children: parents 0, 1, 2 split."""
    n, room = 6, 3
    p = _setup(gpu_device, max_splats=n + room)
    parents = _population(p, n, seed=2, clone_fraction=0, keep_fraction=0, eta_choices=(1.5,))
    b = p.buffers                                       # one axis only: each family costs one child
    for axis in ("max_y", "max_z"):
        b.eta.components[axis].copy_from_numpy(np.zeros(p.max_splats, np.uint32))
    _densify_structure(p, rank_split_families=True)
    actions = _u32(b.densification.actions, n)
    assert actions.tolist() == [1, 1, 1, 0, 0, 0]
    assert _count(p) == n + room
    owners = np.rint(_read_gaussians(p, n + room)[n:, 0] / SPACING).astype(int)
    assert owners.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(_read_gaussians(p, n)[3:], parents[3:])


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
    p.simple_densifier.run(DensifyConfig(), grad_thresh=0.0002, scale_thresh=0.05, allow_scale_prune=False)
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
