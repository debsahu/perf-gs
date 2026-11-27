# Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

"""Float64 numpy reference of the depth / normal / depth-normal losses and the per-splat feature,
written from the formulas (not from the shaders), for the GPU gradient tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PLANE_MIN_DOT, PLANE_MIN_DEPTH, PLANE_MAX_DEPTH = 0.05, 1e-3, 1e4
SATURATED = 65535


@dataclass
class Frame:
    """One frame's inputs: the composite F = (N, S) (H, W, 4), alpha A (H, W), rays (H, W, 3),
    per-pixel prior depth in metres (0 invalid) and prior normal ((0,0,0) invalid), keep mask."""
    F: np.ndarray
    A: np.ndarray
    rays: np.ndarray
    Dg: np.ndarray
    g: np.ndarray
    mask: np.ndarray


@dataclass
class Settings:
    plane: bool = False
    space: str = "disparity"
    normal_cos: bool = False
    w_d: float = 1.0
    w_n: float = 0.2
    w_dn: float = 0.05
    min_alpha: float = 0.5
    U: float = 1.0


def rays_for(proj: np.ndarray, w: int, h: int) -> np.ndarray:
    fx, fy = proj[0, 0] * w / 2, proj[1, 1] * h / 2
    cx, cy = (1 + proj[0, 2]) * w / 2, (1 + proj[1, 2]) * h / 2
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)


def sample_prior(prior: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest at the pixel center: texel = floor((u + 0.5) * pw / W)."""
    ph, pw = prior.shape[:2]
    ys = np.floor((np.arange(h) + 0.5) * ph / h).astype(int)
    xs = np.floor((np.arange(w) + 0.5) * pw / w).astype(int)
    return prior[ys][:, xs]


def depth_codes_to_m(codes: np.ndarray) -> np.ndarray:
    d = codes.astype(np.float64) / 1000.0
    d[(codes == 0) | (codes >= SATURATED)] = 0.0
    return d


def normal_codes_to_n(codes: np.ndarray) -> np.ndarray:
    n = codes[..., :3].astype(np.float64) / 255.0 * 2.0 - 1.0
    n[codes[..., :3] == 128] = 0.0
    l2 = np.sum(n * n, -1)
    n[(l2 < 0.49) | (l2 > 1.69)] = 0.0
    return n


def resolve(F, A, rays, s: Settings):
    """Rendered depth (0 invalid) and normal (0 invalid)."""
    N, S = F[..., :3], F[..., 3]
    n2 = np.sum(N * N, -1)
    cover = A >= s.min_alpha
    nh = np.where((cover & (n2 > 1e-12))[..., None], N / np.sqrt(np.maximum(n2, 1e-300))[..., None], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if s.plane:
            den = np.sum(N * rays, -1)
            D = S / den
            ok = cover & (np.abs(den) >= PLANE_MIN_DOT) & (D >= PLANE_MIN_DEPTH) & (D <= PLANE_MAX_DEPTH)
        else:
            D = S / A
            ok = cover & np.isfinite(D) & (D > 0)
    return np.where(ok, D, 0.0), nh


def dn_normal(D0, D1, D2, r0, r1, r2):
    P0 = D0[..., None] * r0
    c = np.cross(D2[..., None] * r2 - P0, D1[..., None] * r1 - P0)
    return c, c / np.sqrt(np.maximum(np.sum(c * c, -1), 1e-24))[..., None]


def supervision(f: Frame, s: Settings):
    """The frozen supervised sets and the scale-shift fit, at the given composite."""
    D, nh = resolve(f.F, f.A, f.rays, s)
    m = f.mask > 0
    depth = (f.Dg > 0) & m & (D > 0)
    normal = (np.sum(f.g * f.g, -1) > 0) & m & (np.sum(nh * nh, -1) > 0)
    dn = np.zeros_like(m)
    c, _ = dn_normal(D[:-1, :-1], D[:-1, 1:], D[1:, :-1], f.rays[:-1, :-1], f.rays[:-1, 1:], f.rays[1:, :-1])
    dn[:-1, :-1] = ((D[:-1, :-1] > 0) & (D[:-1, 1:] > 0) & (D[1:, :-1] > 0)
                    & (np.sum(nh[:-1, :-1] ** 2, -1) > 0) & m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1]
                    & (np.linalg.norm(c, axis=-1) > 1e-12))
    ab = (1.0, 0.0)
    if s.space == "ssi" and depth.any():
        x = 1.0 / f.Dg[depth]
        y = s.U / D[depth]
        vxx = np.mean(x * x) - np.mean(x) ** 2
        if depth.sum() >= 16 and vxx / np.mean(x) ** 2 > 1e-3:      # relative variance gate
            a = (np.mean(x * y) - np.mean(x) * np.mean(y)) / vxx
            ab = (a, np.mean(y) - a * np.mean(x))
        else:
            depth = np.zeros_like(depth)
    return {"depth": depth, "normal": normal, "dn": dn, "ab": ab}


def loss(F: np.ndarray, f: Frame, s: Settings, sets: dict) -> float:
    """The frame's geometry loss at composite F, on the frozen sets."""
    H, W = F.shape[:2]
    scale = 3.0 * W * H
    N, S = F[..., :3], F[..., 3]
    nh = N / np.sqrt(np.maximum(np.sum(N * N, -1), 1e-24))[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        D = S / np.sum(N * f.rays, -1) if s.plane else S / f.A
    total = 0.0
    dmask = sets["depth"]
    if s.w_d > 0 and dmask.any():
        Dm, Dg = D[dmask] / s.U, f.Dg[dmask]
        if s.space == "metric":
            r = Dm - Dg
        elif s.space == "ssi":
            a, b = sets["ab"]
            r = 1 / Dm - (a / Dg + b)
        else:
            r = 1 / Dm - 1 / Dg
        total += s.w_d * scale / dmask.sum() * np.abs(r).sum()
    nmask = sets["normal"]
    if s.w_n > 0 and nmask.any():
        if s.normal_cos:
            total += s.w_n * scale / nmask.sum() * (1 - np.sum(nh * f.g, -1))[nmask].sum()
        else:
            total += s.w_n * scale / (3 * nmask.sum()) * np.abs(nh - f.g)[nmask].sum()
    q = sets["dn"]
    if s.w_dn > 0 and q.any():
        _, nd = dn_normal(D[:-1, :-1], D[:-1, 1:], D[1:, :-1], f.rays[:-1, :-1], f.rays[:-1, 1:],
                          f.rays[1:, :-1])
        term = 1 - np.sum(nd * nh[:-1, :-1], -1)
        total += s.w_dn * scale / q.sum() * term[q[:-1, :-1]].sum()
    return float(total)


def fd_gradient(f: Frame, s: Settings, eps: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Central finite differences of the loss with respect to every component of F."""
    sets = supervision(f, s)
    F = f.F.astype(np.float64)
    grad = np.zeros_like(F)
    H, W = F.shape[:2]
    for y in range(H):
        for x in range(W):
            for k in range(4):
                old = F[y, x, k]
                F[y, x, k] = old + eps
                lp = loss(F, f, s, sets)
                F[y, x, k] = old - eps
                lm = loss(F, f, s, sets)
                F[y, x, k] = old
                grad[y, x, k] = (lp - lm) / (2 * eps)
    return grad, sets


# ------------------------------------------------------------------------- per-splat feature

def quat_matrix(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z), normalized here."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def feature(mean, quat, view, k, sgn, plane):
    mean_cam = view[:3, :3] @ mean + view[:3, 3]
    n = sgn * (view[:3, :3] @ quat_matrix(quat)[:, k])
    return np.concatenate([n, [np.dot(n, mean_cam) if plane else mean_cam[2]]])


def facing_sign(mean, quat, view, k):
    mean_cam = view[:3, :3] @ mean + view[:3, 3]
    n = view[:3, :3] @ quat_matrix(quat)[:, k]
    return 1.0 if np.dot(n, mean_cam) <= 0 else -1.0
