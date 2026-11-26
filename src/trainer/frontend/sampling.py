# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Which camera, and which STBN mask slice and budget, each training step uses."""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from trainer.backend.render.stbn_table import pixels_per_tile
from trainer.config.config import StbnConfig, StbnVariables
from trainer.frontend.schedule import StbnDensitySchedule


class CameraSampler:
    """``ours``: uniform draws with replacement. ``sadgs``: SADGS's shuffled epochs."""

    def __init__(self, mode: str, train_indices, test_indices, seed: int,
                 rng: np.random.Generator) -> None:
        self.mode = mode
        self._rng = rng
        self._train_arr = np.asarray(train_indices, dtype=np.int64)
        self._ahead: deque[int] = deque()     # drawn by peek(), not yet returned by next()
        if mode == "sadgs":
            self._py = random.Random(seed)
            train = list(train_indices)
            test = list(test_indices)
            # SADGS shuffles both splits at scene load; the test shuffle consumes the stream too.
            self._py.shuffle(train)
            self._py.shuffle(test)
            self._base = list(train)
            self._stack = list(self._base)
            print(f"[sample-mode=sadgs] replicating SADGS random-shuffle sampling "
                  f"(seed={seed}, {len(self._base)} train cams, batch_size=1, "
                  f"first epoch in load-shuffled order, reshuffle on refill)")

    def next(self) -> int:
        """The train-camera index to optimize this iteration."""
        if self._ahead:
            return self._ahead.popleft()
        return self._draw()

    def peek(self, count: int) -> list[int]:
        """The next ``count`` indices ``next()`` will return, without consuming them. They are
        drawn now, in order, from the same stream, so peeking never changes the sequence."""
        while len(self._ahead) < count:
            self._ahead.append(self._draw())
        return list(self._ahead)[:max(0, count)]

    def _draw(self) -> int:
        if self.mode == "sadgs":
            if not self._stack:
                self._stack.extend(self._base)
                self._py.shuffle(self._stack)
            return self._stack.pop(0)
        return int(self._train_arr[self._rng.integers(0, self._train_arr.shape[0])])


class StbnSampler:
    """The mask slice and per-tile budget of each sparse step.
    Slices follow each camera's visit count plus a per-camera phase, so cameras don't march in
    lockstep; ``independent`` draws random slices from its own RNG (the temporal ablation)."""

    def __init__(self, config: StbnConfig, slices: int) -> None:
        self.density = StbnDensitySchedule(target_fraction=config.pixel_fraction,
                                           warmup=config.warmup_iters)
        self.slices = max(int(slices), 1)
        self.seed = int(config.seed)
        self.independent = config.independent
        self.visits: dict[int, int] = {}
        self.phases: dict[int, int] = {}
        self._rng = np.random.default_rng(self.seed)

    def variables(self, camera_index: int, step: int) -> StbnVariables:
        budget = pixels_per_tile(self.density.at(step))
        if self.independent:
            return StbnVariables(slice_index=int(self._rng.integers(self.slices)),
                                 pixels_per_tile=budget)
        if camera_index not in self.phases:
            mixed = int(camera_index) * 2654435761 + self.seed * 40503
            self.phases[camera_index] = mixed % self.slices
        visit = self.visits.get(camera_index, 0)
        self.visits[camera_index] = visit + 1
        return StbnVariables(slice_index=(visit + self.phases[camera_index]) % self.slices,
                             pixels_per_tile=budget)
