// Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

#ifndef TRAINER_GEOM_PRIORS_SLANG
#define TRAINER_GEOM_PRIORS_SLANG

// GEOM_PRIORS = 1 when the run has a depth, normal or depth-normal loss. The forward then
// composites two more per-splat quantities beside color, with the same blend weights:
//   n_i  the splat's unit normal in the camera frame (its smallest-scale axis, facing the camera)
//   s_i  its depth: the camera-frame z of the mean ("center"), or the plane offset dot(n_i, mean)
// into aux_image = (N, S) = (sum_i w_i n_i, sum_i w_i s_i). The blend weights are treated as
// constants by these terms: they move means and rotations, never opacity or 2D footprints.
// Off by default, and then every shader compiles exactly as without this header.
#ifndef GEOM_PRIORS
#define GEOM_PRIORS 0
#endif

#include "math.slang"

// 64-bit fixed-point add from two 32-bit atomics: add the low word, carry into the high word.
// Exact and order-independent modulo 2^64. Used for the geometry seeds on every backend,
// so this code never needs a native 64-bit atomic.
void geomAtomicAdd(RWStructuredBuffer<uint2> b, uint i, int64_t v)
{
    uint64_t u = uint64_t(v);
    uint lo = uint(u & 0xffffffffull);
    uint hi = uint(u >> 32);
    uint old;
    InterlockedAdd(b[i].x, lo, old);
    uint carry = ((old + lo) < old) ? 1u : 0u;
    InterlockedAdd(b[i].y, hi + carry);
}

int64_t geomLoad(uint2 v)
{
    return int64_t((uint64_t(v.y) << 32) | uint64_t(v.x));
}

static const float GEOM_FIXED_SCALE = 65536.0;
int64_t geomToFixed(float value) { return int64_t(value * GEOM_FIXED_SCALE); }
float geomFromFixed(int64_t value) { return float(value) * (1.0 / GEOM_FIXED_SCALE); }

// Index of the smallest of the three log scales; ties go to the lowest index.
uint geomMinAxis(float3 log_scale)
{
    uint k = 0u;
    if (log_scale.y < log_scale.x) k = 1u;
    if (log_scale.z < ((k == 0u) ? log_scale.x : log_scale.y)) k = 2u;
    return k;
}

float3 geomUnitAxis(uint k)
{
    return float3(k == 0u ? 1.0 : 0.0, k == 1u ? 1.0 : 0.0, k == 2u ? 1.0 : 0.0);
}

// Per-splat geometry feature (n.xyz, s). k (the smallest-scale axis) and sgn (the facing sign)
// are detached: they are chosen outside and passed in, so no gradient flows through the choice.
// plane = 0: s = camera-frame z of the mean. plane = 1: s = dot(n, mean_cam), negative when the
// normal faces the camera.
[Differentiable]
float4 geomFeature(float3 mean, float4 rot_quat, no_diff float4x4 world_to_view,
                   no_diff uint k, no_diff float sgn, no_diff uint plane)
{
    float3 mean_cam = mul(world_to_view, float4(mean, 1.0)).xyz;
    float3 axis_world = mul(matrixFromQuaternion(rot_quat), geomUnitAxis(k));
    float3 n = sgn * mul(float3x3(world_to_view), axis_world);
    float s = (plane != 0u) ? dot(n, mean_cam) : mean_cam.z;
    return float4(n, s);
}

// The facing sign of the splat's axis: the normal must point back along the splat's own view ray,
// dot(n, mean_cam) < 0. The ray through the mean is the test, never the optical axis.
float geomFacingSign(float3 mean, float4 rot_quat, float4x4 world_to_view, uint k)
{
    float3 mean_cam = mul(world_to_view, float4(mean, 1.0)).xyz;
    float3 n = mul(float3x3(world_to_view), mul(matrixFromQuaternion(rot_quat), geomUnitAxis(k)));
    return (dot(n, mean_cam) <= 0.0) ? 1.0 : -1.0;
}

// View ray through the pixel-space point uv (the pixel center is integer + 0.5), z = 1.
float3 geomPixelRay(float2 uv, float4x4 proj, float2 image_size)
{
    float2 center = (1.0 + float2(proj[0][2], proj[1][2])) * (0.5 * image_size);
    float2 focal = float2(proj[0][0], proj[1][1]) * (0.5 * image_size);
    return float3((uv - center) / focal, 1.0);
}

// Prior normal codes (uint8 per component) to a unit vector. Code 128 decodes to exactly 0, so
// (128, 128, 128) is the invalid sentinel (0, 0, 0). The loader converts every encoding to this one.
float3 geomDecodeNormal(uint3 code)
{
    float3 n = float3(code) * (2.0 / 255.0) - 1.0;
    return float3(code.x == 128u ? 0.0 : n.x, code.y == 128u ? 0.0 : n.y, code.z == 128u ? 0.0 : n.z);
}

#endif // TRAINER_GEOM_PRIORS_SLANG
