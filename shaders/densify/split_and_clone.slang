// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Densify phase B: SPLIT replaces slot i with child-1 and appends child-2 (scale / 1.6, jittered; 3DGS §5.2);
// CLONE appends an exact copy. Children get zeroed Adam state so clone twins don't take identical steps.

#include "../common/types.slang"
#include "../common/math.slang"
#include "sh_carry.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

static const float SPLIT_SCALE_DIVISOR = 1.6;

// Cheap deterministic-per-seed uniform[0,1).
uint _wang_hash(uint x)
{
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

float _rand01(uint seed)
{
    return float(_wang_hash(seed)) / 4294967295.0;
}

// Box-Muller: two uniforms -> two standard-normal samples.
float2 _randn2(uint seed)
{
    float u1 = max(_rand01(seed + 1u), 1e-7);
    float u2 = _rand01(seed + 7919u);
    float r = sqrt(-2.0 * log(u1));
    float a = 6.28318530718 * u2;
    return float2(r * cos(a), r * sin(a));
}

// Offset drawn from N(0, Σ), Σ = R S^2 R^T being the parent's world-space covariance.
float3 _sample_parent_cov_offset(float3 log_scale, float4 rot_quat, uint seed)
{
    float2 a = _randn2(seed);
    float2 b = _randn2(seed + 73856093u);
    // Three independent N(0,1) samples (fourth discarded).
    float3 z = float3(a.x, a.y, b.x);

    float3 scale = exp(log_scale);
    // R*S*z ~ N(0, R S S R^T) = N(0, Σ) for z ~ N(0, I).
    float3x3 R = matrixFromQuaternion(rot_quat);
    float3 scaled = float3(scale.x * z.x, scale.y * z.y, scale.z * z.z);
    return mul(R, scaled);
}

Gaussian3D _child_for_split(Gaussian3D parent, uint seed)
{
    Gaussian3D child = parent;
    float3 offset = _sample_parent_cov_offset(parent.log_scale, parent.rot_quat, seed);
    child.mean = parent.mean + offset;
    child.log_scale = parent.log_scale - log(SPLIT_SCALE_DIVISOR);
    return child;
}

// Zero-init cast, so new AdamState fields need no edit here.
AdamState _zero_adam()
{
    return (AdamState)0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splitAndClone(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    StructuredBuffer<uint> child_offset,   // exclusive prefix of the parents' growth (allocate_children.slang)
    RWStructuredBuffer<uint> action,
    // Pre-dispatch count snapshot (snapshotCount), so bumping the counter can't race the bounds check.
    StructuredBuffer<uint> pre_count_buf,
    uniform uint max_splats)
{
    uint i = dtid.x;
    uint pre_count = pre_count_buf[0];
    if (i >= pre_count) return;

    uint a = action[i];
    if (a == ACTION_KEEP || a == ACTION_PRUNE) return;

    Gaussian3D parent = gaussians[i];
    AdamState parent_adam = adam_state[i];

    if (a == ACTION_SPLIT)
    {
        Gaussian3D c1 = _child_for_split(parent, i * 2u);
        Gaussian3D c2 = _child_for_split(parent, i * 2u + 1u);

        uint tail = pre_count + child_offset[i];
        if (tail >= max_splats)
        {
            action[i] = ACTION_KEEP;
            return;
        }
        gaussians[tail] = c2;
        adam_state[tail] = _zero_adam();
        gaussians[i] = c1;
        adam_state[i] = _zero_adam();
    }
    else if (a == ACTION_CLONE)
    {
        Gaussian3D clone = parent;

        uint tail = pre_count + child_offset[i];
        if (tail < max_splats)
        {
            gaussians[tail] = clone;
            adam_state[tail] = _zero_adam();
        }
    }
}

// splitAndClone plus the SH rest carry; must replace it, not run alongside (same tail slots).
[shader("compute")]
[numthreads(256, 1, 1)]
void splitAndCloneSH(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    StructuredBuffer<uint> child_offset,   // exclusive prefix of the parents' growth (allocate_children.slang)
    RWStructuredBuffer<uint> action,
    StructuredBuffer<uint> pre_count_buf,
    RWStructuredBuffer<uint> features_rest,
    RWStructuredBuffer<uint> adam_rest,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint i = dtid.x;
    uint pre_count = pre_count_buf[0];
    if (i >= pre_count) return;

    uint a = action[i];
    if (a == ACTION_KEEP || a == ACTION_PRUNE) return;

    Gaussian3D parent = gaussians[i];

    if (a == ACTION_SPLIT)
    {
        Gaussian3D c1 = _child_for_split(parent, i * 2u);
        Gaussian3D c2 = _child_for_split(parent, i * 2u + 1u);

        uint tail = pre_count + child_offset[i];
        if (tail >= max_splats)
        {
            action[i] = ACTION_KEEP;
            return;
        }
        gaussians[tail] = c2;
        adam_state[tail] = _zero_adam();
        shInheritRest(tail, i, rest_floats, max_splats, features_rest, adam_rest);
        // Child-1 overwrites the parent slot: keep its features_rest, fresh Adam.
        gaussians[i] = c1;
        adam_state[i] = _zero_adam();
        shZeroRestAdam(i, rest_floats, max_splats, adam_rest);
    }
    else if (a == ACTION_CLONE)
    {
        Gaussian3D clone = parent;

        uint tail = pre_count + child_offset[i];
        if (tail < max_splats)
        {
            gaussians[tail] = clone;
            adam_state[tail] = _zero_adam();
            shInheritRest(tail, i, rest_floats, max_splats, features_rest, adam_rest);
        }
    }
}
