# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Densification controller: executes one densify tick per call, keeping its ordering invariants
(tracking clears before Morton, eta carry on filler ticks, mip recompute after count change)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trainer.config.config_types import DensificationMode

if TYPE_CHECKING:
    import argparse

    from trainer.backend.render.pipeline import TrainingPipeline
    from trainer.frontend.schedule import ExtinctionHardnessSchedule, IterationEvents
    from trainer.frontend.queue_limiter import QueueLimiter


@dataclass
class DensifyOutcome:
    """Result of one densify tick, reported by :meth:`TrainingReporter.report_densify`."""

    iteration: int
    count: int
    kind: str
    stats: dict


class DensificationController:
    """Owns simple/structure densification plus bookkeeping."""

    def __init__(
        self,
        *,
        pipeline: "TrainingPipeline",
        args: "argparse.Namespace",
        densify_mode: DensificationMode,
        densify_cfg,
        structure_cfg,
        extent: float,
        image_width: int,
        image_height: int,
        morton_world_min,
        morton_world_max,
        extinction_hardness_schedule: "ExtinctionHardnessSchedule",
        queue_limiter: "QueueLimiter",
    ) -> None:
        self._pipeline = pipeline
        self._args = args
        self._densify_mode = densify_mode
        self._densify_cfg = densify_cfg
        self._structure_cfg = structure_cfg
        self._extent = extent
        self._image_width = image_width
        self._image_height = image_height
        self._morton_world_min = morton_world_min
        self._morton_world_max = morton_world_max
        self._extinction_hardness_schedule = extinction_hardness_schedule
        self._queue_limiter = queue_limiter

    def run(self, it: int, ev: "IterationEvents", cur_n_gaussians: int) -> DensifyOutcome:
        """Execute one densify tick and return its outcome; reporting is the caller's job."""
        pipeline = self._pipeline
        args = self._args
        mode = self._densify_mode
        densify_cfg = self._densify_cfg
        structure_cfg = self._structure_cfg
        extent = self._extent
        morton_world_min = self._morton_world_min
        morton_world_max = self._morton_world_max
        extinction_hardness_schedule = self._extinction_hardness_schedule
        queue_limiter = self._queue_limiter
        do_structure_step = ev.structure

        # The first densify starts the extinction-hardness ramp (held at 1.0 until then).
        extinction_hardness_schedule.notify_first_densification(it)

        stats = {}

        sad_carry_eta = (mode == DensificationMode.STRUCTURE_AWARE
                         and not do_structure_step)
        # Per megapixel, so the floor is the same image fraction at every resolution.
        min_blend_weight = (args.prune_blend_weight * self._image_width * self._image_height / 1e6
                            if ev.blend_prune else 0.0)

        if do_structure_step:
            stats = pipeline.structure_densifier.densify(structure_cfg, min_blend_weight, tick=it)
        elif mode == DensificationMode.STRUCTURE_AWARE:
            allow_scale_prune = it > args.reset_every
            stats = pipeline.simple_densifier.run(
                densify_cfg,
                grad_thresh=args.grad_thresh,
                scale_thresh=extent * args.grow_scale3d,
                prune_big_scale=extent * args.prune_scale3d,
                allow_scale_prune=allow_scale_prune,
                carry_eta=True,
                min_blend_weight=min_blend_weight,
                tick=it,
            )
        elif mode == DensificationMode.SIMPLE:
            allow_scale_prune = it > args.reset_every
            stats = pipeline.simple_densifier.run(
                densify_cfg,
                grad_thresh=args.grad_thresh,
                scale_thresh=extent * args.grow_scale3d,
                prune_big_scale=extent * args.prune_scale3d,
                allow_scale_prune=allow_scale_prune,
                min_blend_weight=min_blend_weight,
                tick=it,
            )
        if pipeline.last_submit_id:
            queue_limiter.record(pipeline.last_submit_id)
        commands = pipeline.commands
        with commands.recording("densify_tracking"), commands.marker_group("densify_tracking"):
            pipeline.densify_stats.clear()
            if mode == DensificationMode.STRUCTURE_AWARE and not sad_carry_eta:
                pipeline.structure_densifier.clear_scores()

        pipeline.morton.reorder(morton_world_min, morton_world_max,
                             carry_eta=sad_carry_eta)
        if pipeline.last_submit_id:
            queue_limiter.record(pipeline.last_submit_id)

        if ev.mip_after_densify:
            pipeline.mip_filter.compute()
            if pipeline.last_submit_id:
                queue_limiter.record(pipeline.last_submit_id)

        if stats.get("final_count") is not None:
            cur_n_gaussians = stats["final_count"]
        kind = "structure" if do_structure_step else (
            "absgs" if mode == DensificationMode.STRUCTURE_AWARE else "densify")

        return DensifyOutcome(
            iteration=it, count=cur_n_gaussians, kind=kind, stats=stats)
