# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Which parents get their children when a densify tick asks for more than the cap leaves room for.

``hash`` (default): parents are admitted in ascending order of a hash of (index, tick, seed), all
or nothing per parent, so the survivors are a spatially uniform subset. ``index``: the lowest
indices first, which after a Morton reorder all sit in one part of the scene. Both are
deterministic and never exceed the budget. Kernels: ``densify/allocate_children.slang``.
"""

from __future__ import annotations

from trainer.backend.compute.algorithms import PrefixScan
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import DensificationResources

_ALLOCATE = "densify/allocate_children.slang"
HASH_BINS = 256
HASH_REMAINING = 2 * HASH_BINS + 1
CAP_ORDERS = ("hash", "index")


class CapOrder:
    KERNELS = {
        "clear_uint": ("optimizers/grad_utils.slang", "clearUintBuffer"),
        "write_cost": (_ALLOCATE, "writeParentCost"),
        "write_headroom": (_ALLOCATE, "writeHeadroom"),
        "copy_uint": (_ALLOCATE, "copyUint"),
        "histogram": (_ALLOCATE, "hashHistogram"),
        "cutoff": (_ALLOCATE, "hashCutoff"),
        "cutoff_cost": (_ALLOCATE, "hashCutoffCost"),
        "hash_admit": (_ALLOCATE, "hashAdmit"),
        "index_admit": (_ALLOCATE, "indexAdmit"),
        "drop": (_ALLOCATE, "dropUnadmitted"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary,
                 resources: DensificationResources, scan: PrefixScan, capacity: int) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._r = resources
        self._scan = scan
        self._capacity = capacity

    def admit(self, order: str, cost, budget, budget_index: int, admit, tick: int, seed: int) -> None:
        """admit[i] = 1 for each parent (cost[i] > 0) inside budget[budget_index], in ``order``."""
        if order not in CAP_ORDERS:
            raise ValueError(f"unknown cap order {order!r}")
        k, commands, cap, hist = self._k, self._commands, self._capacity, self._r.cap_histogram
        work = self._scan.resources.work
        if order == "index":
            self._scan.scan(cap, cost, work)
            commands.dispatch(k.index_admit, "cap_index_admit", thread_count=[cap, 1, 1],
                cost=cost.storage, cutoff_offset=work.storage, budget=budget.storage,
                admit=admit.storage, budget_index=budget_index, max_splats=cap)
            return
        hashing = dict(tick=int(tick) & 0xFFFFFFFF, seed=int(seed) & 0xFFFFFFFF, max_splats=cap)
        commands.dispatch(k.clear_uint, "cap_clear_histogram", thread_count=[2 * HASH_BINS, 1, 1],
            buffer=hist.storage, count=2 * HASH_BINS)
        if budget is not hist or budget_index != HASH_REMAINING:
            commands.dispatch(k.copy_uint, "cap_copy_budget", thread_count=[1, 1, 1],
                src=budget.storage, dst=hist.storage, src_index=budget_index, dst_index=HASH_REMAINING)
        commands.dispatch(k.histogram, "cap_hash_histogram", thread_count=[cap, 1, 1],
            cost=cost.storage, histogram=hist.storage, **hashing)
        commands.dispatch(k.cutoff, "cap_hash_cutoff", thread_count=[1, 1, 1], histogram=hist.storage)
        commands.dispatch(k.cutoff_cost, "cap_hash_cutoff_cost", thread_count=[cap, 1, 1],
            cost=cost.storage, histogram=hist.storage, scratch=admit.storage, **hashing)
        self._scan.scan(cap, admit, work)
        commands.dispatch(k.hash_admit, "cap_hash_admit", thread_count=[cap, 1, 1],
            cost=cost.storage, histogram=hist.storage, cutoff_offset=work.storage,
            admit=admit.storage, **hashing)

    def thin(self, order: str, tick: int, seed: int, *, simple: bool, factors=None) -> None:
        """Unranked: drop (action -> keep) the parents outside the free slots, in ``order``. With
        ``index`` nothing is dropped here: the allocation already keeps the lowest indices."""
        if order == "index":
            return
        r, cap = self._r, self._capacity
        fx, fy, fz = factors if factors is not None else (r.actions, r.actions, r.actions)
        self._commands.dispatch(self._k.write_cost, "cap_parent_cost", thread_count=[cap, 1, 1],
            action=r.actions.storage, pre_count_buf=r.pre_count.storage,
            split_factor_x=fx.storage, split_factor_y=fy.storage, split_factor_z=fz.storage,
            cost=r.child_offset.storage, simple=1 if simple else 0, max_splats=cap)
        self._commands.dispatch(self._k.write_headroom, "cap_headroom", thread_count=[1, 1, 1],
            pre_count_buf=r.pre_count.storage, histogram=r.cap_histogram.storage, max_splats=cap)
        self.admit(order, r.child_offset, r.cap_histogram, HASH_REMAINING, r.valid_mask, tick, seed)
        self._commands.dispatch(self._k.drop, "cap_drop_unadmitted", thread_count=[cap, 1, 1],
            cost=r.child_offset.storage, admit=r.valid_mask.storage, action=r.actions.storage,
            max_splats=cap)
