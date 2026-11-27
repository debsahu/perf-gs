// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// SAD-GS grid splitting: one workgroup per selected parent, lanes emit children
// cooperatively. Child 0 replaces the parent; the rest go to the parent's tail range,
// pre_count + child_offset[parent] (allocate_children.slang), whatever order groups run in.

#include "../common/types.slang"
#include "../common/math.slang"
#include "sh_carry.slang"

static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint SPLIT_GROUP_SIZE = 32u;
static const uint SPLIT_DISPATCH_WIDTH = 65535u;

groupshared Gaussian3D s_parent;
groupshared uint3 s_factors;
groupshared uint s_tail;

struct ChildGrid
{
    Gaussian3D child;
    float3x3 rotation;
    float3 separation;
    uint3 factors;
};

ChildGrid prepareChildGrid(Gaussian3D parent, uint3 factors, float ks_scale_power)
{
    float3 nf = float3(factors);
    ChildGrid grid;
    Gaussian3D child = parent;
    child.log_scale = parent.log_scale
                    - ks_scale_power * float3(log(nf.x), log(nf.y), log(nf.z));
    grid.child = child;
    grid.rotation = matrixFromQuaternion(parent.rot_quat);
    // sqrt(12) spacing preserves the parent's mixture covariance when p == 1.
    grid.separation = (exp(parent.log_scale) / nf) * 3.4641016151377544;
    grid.factors = factors;
    return grid;
}

Gaussian3D gridChild(ChildGrid grid, uint ordinal)
{
    uint3 n = grid.factors;
    uint3 idx = uint3(ordinal % n.x, (ordinal / n.x) % n.y, ordinal / (n.x * n.y));
    float3 coord = float3(idx) - 0.5 * float3(n - uint3(1u));
    Gaussian3D child = grid.child;
    child.mean += mul(grid.rotation, coord * grid.separation);
    return child;
}

// Snapshot the parent before child 0 overwrites its slot; look up its tail range.
void prepareParent(uint i, uint lane, uint act,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> child_offset,
    uint pre_count,
    StructuredBuffer<uint> split_factor_x,
    StructuredBuffer<uint> split_factor_y,
    StructuredBuffer<uint> split_factor_z)
{
    if (lane == 0u)
    {
        s_parent = gaussians[i];
        s_factors = act == ACTION_CLONE ? uint3(1u) : uint3(
            max(split_factor_x[i], 1u), max(split_factor_y[i], 1u), max(split_factor_z[i], 1u));
        s_tail = pre_count + child_offset[i];
    }
    GroupMemoryBarrierWithGroupSync();
}

uint survivingChildren(uint total, uint max_splats)
{
    uint headroom = s_tail < max_splats ? max_splats - s_tail : 0u;
    // A split needs a second slot. Keep the parent intact if the budget is full.
    return headroom == 0u ? 0u : 1u + min(total - 1u, headroom);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void splitAnisotropic(
    uint3 group_id: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    StructuredBuffer<uint> child_offset,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> split_factor_x,
    StructuredBuffer<uint> split_factor_y,
    StructuredBuffer<uint> split_factor_z,
    StructuredBuffer<uint> pre_count_buf,
    StructuredBuffer<uint> selected_parents,
    StructuredBuffer<uint> selected_count,
    uniform float ks_scale_power,
    uniform uint max_splats)
{
    uint selected = group_id.x + group_id.y * SPLIT_DISPATCH_WIDTH;
    if (selected >= selected_count[0]) return;
    uint i = selected_parents[selected];
    if (i >= min(pre_count_buf[0], max_splats)) return;
    uint lane = gtid.x;
    uint act = action[i];
    if (act != ACTION_SPLIT && act != ACTION_CLONE) return;
    prepareParent(i, lane, act, gaussians, child_offset, pre_count_buf[0],
                  split_factor_x, split_factor_y, split_factor_z);
    if (act == ACTION_CLONE)
    {
        if (lane == 0u && s_tail < max_splats)
        {
            gaussians[s_tail] = s_parent;
            adam_state[s_tail] = (AdamState)0;
        }
        return;
    }
    uint total = s_factors.x * s_factors.y * s_factors.z;
    if (total <= 1u) return;
    uint count = survivingChildren(total, max_splats);
    if (lane >= count) return;
    ChildGrid grid = prepareChildGrid(s_parent, s_factors, ks_scale_power);
    for (uint ordinal = lane; ordinal < count; ordinal += SPLIT_GROUP_SIZE)
    {
        uint dst = ordinal == 0u ? i : s_tail + ordinal - 1u;
        gaussians[dst] = gridChild(grid, ordinal);
        adam_state[dst] = (AdamState)0;
    }
}

// Same grid/clone behavior, with inherited SH coefficients and fresh moments.
[shader("compute")]
[numthreads(32, 1, 1)]
void splitAnisotropicSH(
    uint3 group_id: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    StructuredBuffer<uint> child_offset,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> split_factor_x,
    StructuredBuffer<uint> split_factor_y,
    StructuredBuffer<uint> split_factor_z,
    StructuredBuffer<uint> pre_count_buf,
    StructuredBuffer<uint> selected_parents,
    StructuredBuffer<uint> selected_count,
    RWStructuredBuffer<uint> features_rest,
    RWStructuredBuffer<uint> adam_rest,
    uniform float ks_scale_power,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint selected = group_id.x + group_id.y * SPLIT_DISPATCH_WIDTH;
    if (selected >= selected_count[0]) return;
    uint i = selected_parents[selected];
    if (i >= min(pre_count_buf[0], max_splats)) return;
    uint lane = gtid.x;
    uint act = action[i];
    if (act != ACTION_SPLIT && act != ACTION_CLONE) return;
    prepareParent(i, lane, act, gaussians, child_offset, pre_count_buf[0],
                  split_factor_x, split_factor_y, split_factor_z);

    if (act == ACTION_CLONE)
    {
        if (lane == 0u && s_tail < max_splats)
        {
            gaussians[s_tail] = s_parent;
            adam_state[s_tail] = (AdamState)0;
            shInheritRest(s_tail, i, rest_floats, max_splats, features_rest, adam_rest);
        }
        return;
    }
    uint total = s_factors.x * s_factors.y * s_factors.z;
    if (total <= 1u) return;
    uint count = survivingChildren(total, max_splats);
    if (lane >= count) return;
    ChildGrid grid = prepareChildGrid(s_parent, s_factors, ks_scale_power);
    for (uint ordinal = lane; ordinal < count; ordinal += SPLIT_GROUP_SIZE)
    {
        uint dst = ordinal == 0u ? i : s_tail + ordinal - 1u;
        gaussians[dst] = gridChild(grid, ordinal);
        adam_state[dst] = (AdamState)0;
        if (ordinal == 0u)
            shZeroRestAdam(dst, rest_floats, max_splats, adam_rest);
        else
            shInheritRest(dst, i, rest_floats, max_splats, features_rest, adam_rest);
    }
}
