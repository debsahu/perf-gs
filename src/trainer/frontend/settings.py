# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Training knobs the CLI and the hub panel both expose, each declared once as a :class:`Setting`.
:class:`TrainingSetup` is the panel's store, applied to the args right before a run is built."""
from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import Any

from trainer.config.config import (
    QUALITY_STEPS,
    CompositorConfig,
    ScheduleConfig,
    ShaderSpecialization,
    StbnConfig,
)
from trainer.config.config_types import DensificationMode, LearningRateRecipe, SplatKernel

# Widget kinds, derived from a Setting's own declaration (see :attr:`Setting.kind`).
INT = "int"
FLOAT = "float"
BOOL = "bool"
CHOICE = "choice"

# Panel groups in display order (explicit, rather than sorting SETTINGS by group).
GROUPS: tuple[tuple[str, str], ...] = (
    ("schedule", "SCHEDULE"),
    ("densify", "DENSIFICATION"),
    ("loss", "LOSS"),
    ("appearance", "APPEARANCE"),
    ("perf", "PERFORMANCE"),
    ("eval", "EVALUATION"),
    ("stream", "VIEWER STREAM"),
)


@dataclass(frozen=True)
class Setting:
    setting_name: str
    caption: str
    group: str
    type: type = int
    default: Any = 0
    choices: tuple[Any, ...] = ()
    help: str = ""
    # Panel-only: argparse validates no ranges, so the drag domain lives here.
    minimum: float = 0.0
    maximum: float = 0.0
    speed: float = 1.0
    fmt: str = ""
    note: str = ""
    # The panel's description: what the knob does and what it costs, in about two lines.
    about: str = ""

    @property
    def flag(self) -> str:
        """``densify_from`` -> ``--densify-from``: the dest is the flag."""
        return "--" + self.setting_name.replace("_", "-")

    @property
    def label(self) -> str:
        return self.caption

    @property
    def kind(self) -> str:
        if self.choices:
            return CHOICE
        if self.type is bool:
            return BOOL
        return INT if self.type is int else FLOAT

    def add_to(self, parser: argparse.ArgumentParser) -> argparse.Action:
        if self.kind == BOOL:
            return parser.add_argument(self.flag, default=self.default,
                                       action=argparse.BooleanOptionalAction,
                                       help=self.help or None)
        return parser.add_argument(self.flag, type=self.type, default=self.default,
                                   choices=list(self.choices) or None,
                                   help=self.help or None)

    def coerce(self, value: Any) -> Any:
        """Clamp/convert an incoming value into this setting's domain."""
        if self.kind == BOOL:
            return bool(value)
        if self.kind == CHOICE:
            return value if value in self.choices else self.choices[0]
        if self.kind == INT:
            return int(min(self.maximum, max(self.minimum, int(value))))
        return float(min(self.maximum, max(self.minimum, float(value))))

    def choice_index(self, value: Any) -> int:
        return self.choices.index(value) if value in self.choices else 0

    def choice_labels(self) -> list[str]:
        return [str(choice) for choice in self.choices]

    def display(self, value: Any) -> str:
        if self.kind == BOOL:
            return "on" if value else "off"
        if self.kind == FLOAT:
            return (self.fmt or "%.3f") % float(value)
        return f"{int(value):,}" if self.kind == INT else str(value)


PRESET = "preset"
# Each preset's settings; explicit flags (or later panel edits) still override them.
PRESETS: dict[str, dict[str, Any]] = {
    "ultrafast": {"max_splats": 1_850_000, "sh_band_interval": 500,
                  "stbn_pixel_fraction": 0.25, "image_residency": "gpu",
                  "densify_to": 1600, "grad_thresh": 0.00060,
                  "prune_blend_from": 650, "prune_blend_weight": 0.25},
    "balanced": {},
    "quality": {"max_splats": 4_000_000, "sh_degree": 3, "iters": QUALITY_STEPS,
                "final_splat_fraction": 0.75},
}
PRESETS["quality+"] = {**PRESETS["quality"], "iters": 0}


SETTINGS: tuple[Setting, ...] = (
    Setting(PRESET, "preset", "perf", str, default="balanced", choices=tuple(PRESETS),
            about="Ultrafast samples fewer gradients. Quality fits a larger model. "
                  "Quality+ refines it until training-view improvement levels off.",
            help="training speed and detail: ultrafast, balanced (default), quality, or quality+. "
                 "Explicit flags override preset settings."),
    # The preset resolves its position LR horizon; an explicit CLI override wins.
    Setting("iters", "iterations", "schedule", int, default=3000,
            about="Training steps. For Quality+, 0 trains until convergence and a positive value caps the run.",
            minimum=0, maximum=200_000, speed=25),
    Setting("lr_recipe", "learning-rate recipe", "schedule", str,
            about="The short recipe cools geometry and appearance updates toward the end of training.",
            default=LearningRateRecipe.SHORT.value,
            choices=tuple(r.value for r in LearningRateRecipe),
            help="per-parameter learning-rate plan. 'quality' starts the scale and "
                 "rotation LRs 3x higher, decays them log-linearly to 0.5x by the "
                 "last step and lowers the opacity LR to 0.7x: sharper results for "
                 "6-12%% more training time. The short recipe cools appearance and rotation LRs, then ramps L2."),
    Setting("densify_mode", "strategy", "densify", str,
            about="How splats are added and removed: simple = 3DGS clone/split, "
                  "structure_aware = SAD-GS splits along image structure, none = fixed set.",
            default=DensificationMode.STRUCTURE_AWARE.value,
            choices=tuple(m.value for m in DensificationMode),
            help="densification strategy the loop runs on a densify tick "
                 "(default: structure_aware). Modes: 'none' = no densification; "
                 "'simple' = reference 3DGS threshold-based clone/split; "
                 "'structure_aware' = SAD-GS anisotropic grid-split "
                 "(arXiv 2604.28016) with an interleaved AbsGS gradient filler on "
                 "the non-structure densify ticks."),
    Setting("grad_thresh", "gradient threshold", "densify", float, default=0.00058,
            minimum=0.0, maximum=0.05, speed=0.00001, fmt="%.6f",
            about="Minimum accumulated 2D gradient for clone/split densification. Lower values grow more splats."),
    Setting("densify_from", "first densify iter", "densify", int, default=100,
            about="First iteration that adds or removes splats.",
            minimum=0, maximum=200_000, speed=25),
    Setting("densify_to", "last densify iter", "densify", int, default=1900,
            about="Last iteration that proposes new splats. A final trim, if enabled, follows growth.",
            minimum=0, maximum=200_000, speed=25),
    Setting("densify_every", "densify interval", "densify", int, default=100,
            about="Iterations between densification steps.",
            minimum=1, maximum=10_000, speed=5),
    Setting("reset_every", "opacity reset interval", "densify", int, default=1200,
            about="Iterations between opacity resets, which pull every splat's opacity "
                  "down so the unneeded ones fade and get pruned.",
            help="opacity-reset interval. Big-scale prune is disabled until after "
                 "the first reset.",
            minimum=0, maximum=200_000, speed=25, note="0 = never reset"),
    Setting("max_splats", "splat capacity", "densify", int, default=2_600_000,
            about="Most splats the model may grow to. GPU memory for this many is "
                  "reserved up front, and it is lowered if it does not fit.",
            minimum=100_000, maximum=100_000_000, speed=50_000),
    Setting("final_splat_fraction", "fraction to keep after growth", "densify", float, default=1.0,
            help="fraction of live splats to retain after growth, scored by their effect on rendered color. "
                 "Unobserved splats are kept. 1 disables the final trim.",
            about="After growth, keep the splats whose removal changes the training images most. "
                  "1 keeps them all.",
            minimum=0.01, maximum=1.0, speed=0.01, fmt="%.2f"),
    Setting("lambda_ssim", "SSIM weight", "loss", float, default=0.3,
            about="Weight of the structural-similarity (SSIM) term in the loss; the "
                  "rest is per-pixel error.",
            help="SSIM loss weight (3DGS default 0.2); 0 = pure L2",
            minimum=0.0, maximum=1.0, speed=0.005, fmt="%.3f",
            note="0 = pure L1/L2"),
    Setting("log_every", "log every N iters", "loss", int, default=0,
            about="How often the loss graph and the log sample loss and PSNR. Each "
                  "sample briefly stalls the GPU.",
            help="sample the loss (and the single-camera PSNR) every N iters. Each "
                 "sample re-renders the sampled camera and reads the frame back to "
                 "the host, so this is the cadence of a GPU sync, not a free "
                 "counter: it feeds the viewer's LOSS graph and the log lines. "
                 "0 = off (no samples, empty graph).",
            minimum=0, maximum=10_000, speed=10, note="0 = off"),

    Setting("sh_degree", "SH degree", "appearance", int, default=2,
            about="View-dependent colour (spherical harmonics) for shine and "
                  "reflections. SH3: about +0.3 dB, but ~40% slower and more memory.",
            choices=(0, 1, 2, 3),
            help="Spherical-harmonics degree for view-dependent color (classic "
                 "3DGS). 0 = view-independent SH band-0 (DC) only, the "
                 "original bit-identical path. 1-3 enables higher bands, stored in "
                 "a separate features_rest buffer allocated only when >0, activated "
                 "one band every --sh-band-interval iters (oneupSHdegree)."),
    Setting("sh_band_interval", "SH band interval", "appearance", int, default=0,
            about="Iterations between switching on each extra SH band.",
            help="Iterations between activating successive SH bands (classic 3DGS "
                 "oneupSHdegree). Only used with --sh-degree > 0. "
                 "0 (default) = AUTO: scale to --iters so the last band turns on "
                 "by iters//2 (interval = iters // (2*sh_degree)); the classic "
                 "constant 1000 assumes a 30k-iter run and starves the top bands "
                 "on short runs. Pass a positive value to override.",
            minimum=0, maximum=20_000, speed=10, note="0 = auto from iterations"),
    Setting("mip_splatting", "Mip-Splatting", "appearance", bool, default=True,
            about="Anti-aliasing: avoids artifacts when the model is viewed closer or "
                  "farther away than the training photos.",
            help="Enable Mip-Splatting (Yu et al., CVPR 2024): a per-gaussian 3D "
                 "smoothing filter (from the TRAIN cameras) folded into the "
                 "projected covariance, plus the 2D Mip antialiasing filter with "
                 "differentiable opacity compensation. Targets the high-frequency "
                 "detail gap vs SADGS (which uses the same 3D filter). Default on; "
                 "pass --no-mip-splatting for the bit-identical vanilla renderer."),
    Setting("extinction_hardness", "extinction hardness k", "perf", float,
            about="Above 1, each splat hides more of what is behind it, so rendering "
                  "stops sooner: faster, at some loss of quality.",
            default=CompositorConfig.extinction_hardness,
            help="Compositor extinction hardness k (a shader UNIFORM, so "
                 "no recompile). 1.0 (default) = standard alpha-over "
                 "(bit-identical). k>1 hardens the per-splat transmittance "
                 "to (1-alpha)^k so T decays faster => stronger occlusion "
                 "culling (earlier early-out in forward AND backward) at a "
                 "quality tradeoff. Beer-Lambert-style; requires the "
                 "per-splat backward (the default).",
            minimum=1.0, maximum=16.0, speed=0.05, fmt="%.2f",
            note="1 = standard alpha-over"),
    Setting("extinction_hardness_warmup", "hardness warmup steps", "perf", int, default=500,
            about="Steps over which the hardness ramps up from 1, starting at the "
                  "first densification.",
            help="Steps to ramp the extinction hardness from 1.0 up to "
                 "--extinction-hardness AFTER the first densification. "
                 "k is held at 1.0 (standard alpha-over) until the first "
                 "densify fires, then log-linearly ramps 1 -> target over "
                 "this many steps (like the mean-LR schedule), so "
                 "extinction only sharpens once geometry starts forming. "
                 "0 = jump straight to the target after the first densify; "
                 "<0 = disable the schedule (constant target from step 0, "
                 "the pre-schedule behavior). No-op when "
                 "--extinction-hardness <= 1.",
            minimum=-1, maximum=20_000, speed=10, note="-1 = no ramp"),
    Setting("backward_pixel_stride", "backward pixel stride", "perf", int,
            about="Gradients from 1 in N pixels per step, rotating: a faster backward "
                  "pass, noisier gradients. Cannot be combined with STBN.",
            default=ShaderSpecialization.backward_pixel_stride,
            choices=(1, 2, 4, 8, 16),
            help="Stochastic pixel-subsampling stride for the backward "
                 "rasterizer (BACKWARD_PIXEL_STRIDE shader define). "
                 "1 = exact full-pixel backward; N>1 visits 1/N of tile "
                 "pixels per iter (unbiased SGD estimator, rotating "
                 "phase) for a faster backward at some gradient-noise "
                 "cost. Injected as a compile-time define.",
            note="1 = exact backward"),
    Setting("stbn_pixel_fraction", "STBN pixel fraction", "perf", float,
            about="Share of pixels that drive each step (a blue-noise pattern). "
                  "0.5: 6% faster, -0.04 dB; 0.25: 13% faster, -0.24 dB.",
            default=StbnConfig.pixel_fraction,
            help="Fraction rho of pixels each backward supervises, chosen by a "
                 "spatiotemporal blue-noise mask (NVIDIA STBN, re-ranked per "
                 "tile so EXACTLY rho*64 of every 8x8 tile is sampled). 1.0 "
                 "(default) = every pixel, feature off and bit-identical. Below "
                 "1.0 the surviving per-pixel terms are scaled by 1/rho, so the "
                 "gradient stays an unbiased estimate of the full-image one and "
                 "the LR needs no retuning. The mask advances per CAMERA VISIT, "
                 "so repeat visits of one view walk the mask's temporal axis "
                 "instead of resampling the same pixels. Requires the asset: "
                 "python scripts/data/download_stbn.py (non-commercial license). "
                 "Mutually exclusive with --backward-pixel-stride > 1.",
            minimum=0.02, maximum=1.0, speed=0.01, fmt="%.3f",
            note="1 = every pixel (off)"),
    Setting("stbn_warmup_iters", "STBN warmup steps", "perf", int,
            about="Steps that start from every pixel and ramp down to the share "
                  "above, which protects thin structures early on.",
            default=StbnConfig.warmup_iters,
            help="Steps spent ramping the STBN pixel fraction from 1.0 down to "
                 "--stbn-pixel-fraction (log-linear, so it passes through the "
                 "intermediate densities). Sparse supervision is hardest on the "
                 "thin structures early densification is trying to create, so "
                 "starting dense is the cheap hedge. 0 (default) = the target "
                 "fraction from step 1. No-op at --stbn-pixel-fraction 1.0.",
            minimum=0, maximum=20_000, speed=10, note="0 = no ramp"),
    Setting("image_residency", "training images", "perf", str, default="stream",
            about="Streaming saves GPU memory. Keeping all training images on the GPU saves transfer time. "
                  "Compressed keeps only the image files in host RAM and decodes a few views ahead.",
            choices=("stream", "gpu", "compressed"), help="stream photos from host RAM and compute structure "
            "fields when needed, keep all training photos and their structure fields on the GPU, or "
            "'compressed': stream, but hold each photo in host RAM as its encoded file and decode it "
            "on a background thread a few steps before it is trained on (same bytes as 'stream')."),
    Setting("adam_moments", "Adam moments", "perf", str, default="bf16",
            about="Precision of the optimizer state. bf16: the same quality with "
                  "0.7-1.7 GiB less memory, and slightly faster.",
            choices=("fp32", "bf16"),
            help="storage precision of the Adam moments. 'bf16' (stochastic "
                 "rounding) halves optimizer memory: 56 B/splat at SH0, "
                 "236 B at SH3."),
    Setting("prune_blend_weight", "blend-weight prune floor", "perf", float,
            about="Densify steps remove splats that barely show in any view (under "
                  "this many pixels of blend weight per megapixel): fewer splats, faster.",
            default=ScheduleConfig.prune_blend_weight,
            help="on densify ticks, prune splats whose blend weight (alpha*T "
                 "summed over a view's pixels), averaged over the views they "
                 "appeared in since the last tick, is below this many pixels "
                 "per megapixel of the training image. Such splats barely change "
                 "any image. 0 = off.",
            minimum=0.0, maximum=50.0, speed=0.01, fmt="%.2f", note="0 = off"),
    Setting("prune_blend_from", "blend-weight prune from iter", "perf", int,
            about="Iteration where that pruning starts; 0 = one pass at the last densify "
                  "step (12% faster, same quality). From 1000: 27% faster, -0.04 dB.",
            default=900,
            help="first densify tick that applies --prune-blend-weight; 0 "
                 "= only the last densify tick (--densify-to). The "
                 "tick right after an opacity reset is always skipped: the "
                 "reset itself depresses every blend weight.",
            minimum=0, maximum=200_000, speed=25, note="0 = last densify step"),
    Setting("splat_kernel", "splat kernel", "perf", str,
            about="How a splat's footprint becomes opacity; the density kernels grow "
                  "solid cores. opaque_core (power 3, onset 2): 9% faster, -0.3 dB.",
            default=ShaderSpecialization.splat_kernel.value,
            choices=tuple(k.value for k in SplatKernel),
            help="how a splat's footprint G becomes alpha. 'gaussian' "
                 "(default) = opacity * G. 'density' treats it as an optical "
                 "depth: alpha = 1 - (1 - opacity)^G, the same peak alpha, but "
                 "opaque splats grow fully opaque cores. 'log_density' = "
                 "1 - exp(-exp(logit) * G), whose core grows faster with the "
                 "logit. 'opaque_core' = log_density whose falloff power "
                 "rises from 1 to --splat-profile-power as the splat turns "
                 "opaque."),
    Setting("splat_profile_power", "splat profile power", "perf", float,
            about="Above 1, flattens each splat's top and sharpens its edge, "
                  "shrinking its footprint.",
            default=ShaderSpecialization.splat_profile_power,
            help="radial falloff power p >= 1 of every splat kernel: "
                 "G = exp(-m^p), m = half the squared Mahalanobis "
                 "distance. 1 (default) = gaussian; > 1 flattens the top "
                 "and steepens the edge, shrinking the footprint.",
            minimum=1.0, maximum=16.0, speed=0.05, fmt="%.2f", note="1 = gaussian falloff"),
    Setting("splat_profile_onset", "opaque-core onset", "perf", float,
            about="How opaque a splat must become before opaque_core starts to "
                  "flatten it; higher = later.",
            default=ShaderSpecialization.splat_profile_onset,
            help="opaque_core only: the power ramps with sigmoid(logit - "
                 "onset), so a splat flattens only once its opacity logit "
                 "passes the onset. 0 (default) = ramp with the opacity "
                 "itself.",
            minimum=-20.0, maximum=20.0, speed=0.05, fmt="%.2f", note="opaque_core only"),
    Setting("evaluation", "Evaluate image quality", "eval", bool, default=False,
            about="Render and score the camera set after training. Export is available either way.",
            help="Run an evaluation sweep on the last step. Off by default. "
                 "Individual metric flags also opt in. --no-evaluation disables sweeps, "
                 "held-out reports and final metrics."),
    Setting("stream_every", "snapshot every N iters", "stream", int, default=10,
            about="How often this viewer gets a fresh copy of the splats while training.",
            help="push a gaussian snapshot to the in-process viewer every "
                 "N iters. Now a zero-copy GPU->GPU blit, so a fine cadence "
                 "is cheap (was coarse when it required a GPU->CPU readback).",
            minimum=1, maximum=1_000, speed=1),
)

SETTINGS_MAP: dict[str, Setting] = {s.setting_name: s for s in SETTINGS}


def preset_values(name: str) -> dict[str, Any]:
    """Restore the common recipe and apply the selected tier's overrides."""
    touched = {"lr_recipe", "adam_moments", "lambda_ssim", "densify_from", "prune_blend_weight",
               "splat_kernel", "splat_profile_power", "splat_profile_onset"}
    touched.update(dest for bundle in PRESETS.values() for dest in bundle)
    return {dest: SETTINGS_MAP[dest].default for dest in touched} | PRESETS[name]


def group_settings(group: str) -> list[Setting]:
    """The settings of one group, in table order."""
    return [s for s in SETTINGS if s.group == group]


def validate() -> list[str]:
    """Every way a row can be wrong or unmounted, as a list of complaints."""
    from trainer.frontend.cli import build_parser  # local: cli.py imports us

    mounted = set(vars(build_parser().parse_args([])))
    problems: list[str] = []
    for setting in SETTINGS:
        if setting.setting_name not in mounted:
            problems.append(f"{setting.setting_name}: in SETTINGS but never added to the parser - add `BY_DEST[{setting.setting_name!r}].add_to(parser)`")
        if setting.kind in (INT, FLOAT):
            if not setting.minimum <= setting.default <= setting.maximum:
                problems.append(f"{setting.setting_name}: default {setting.default} is "
                                f"outside the panel range "
                                f"[{setting.minimum}, {setting.maximum}]")
            if setting.speed <= 0:
                problems.append(f"{setting.setting_name}: drag speed must be positive")
        elif setting.kind == CHOICE and setting.default not in setting.choices:
            problems.append(f"{setting.setting_name}: default {setting.default!r} is not "
                            f"one of {setting.choices}")
        if setting.group not in dict(GROUPS):
            problems.append(f"{setting.setting_name}: unknown group {setting.group!r}")
    return problems


class TrainingSetup:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {s.setting_name: s.coerce(s.default) for s in SETTINGS}
        if values:
            for dest, value in values.items():
                self.set(dest, value)

    @classmethod
    def from_args(cls, args: Any) -> TrainingSetup:
        setup = cls()
        for setting in SETTINGS:
            setup.set(setting.setting_name, getattr(args, setting.setting_name))
        return setup

    def get(self, dest: str) -> Any:
        with self._lock:
            return self._values[dest]

    def set(self, dest: str, value: Any) -> Any:
        """Store a coerced value (out-of-range input is clamped, never rejected)."""
        setting = SETTINGS_MAP[dest]
        coerced = setting.coerce(value)
        with self._lock:
            if dest == "iters" and self._values[PRESET] != "quality+":
                coerced = max(100, coerced)
            self._values[dest] = coerced
        return coerced

    def index_of(self, dest: str) -> int:
        """Combo index of a CHOICE setting's current value."""
        return SETTINGS_MAP[dest].choice_index(self.get(dest))

    def set_index(self, dest: str, index: int) -> Any:
        """Store a CHOICE setting from a combo index; choosing a preset also applies its settings."""
        choices = SETTINGS_MAP[dest].choices
        value = self.set(dest, choices[int(index) % len(choices)])
        if dest == PRESET:
            for key, preset_value in preset_values(value).items():
                self.set(key, preset_value)
        return value

    def values(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def apply_to(self, args: Any) -> list[str]:
        changes: list[str] = []
        for setting in SETTINGS:
            new = self.get(setting.setting_name)
            old = getattr(args, setting.setting_name, None)
            if old != new:
                changes.append(f"{setting.setting_name}: "
                               f"{setting.display(old) if old is not None else '-'}"
                               f" -> {setting.display(new)}")
            setattr(args, setting.setting_name, new)
        return changes
