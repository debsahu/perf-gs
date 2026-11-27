// Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

// Order-independent child slots for densification. A parent's children go to
// pre_count + (exclusive prefix sum of the growth of every lower-index parent),
// so the layout depends only on the parents' indices and growth, never on the
// order workgroups run in. The prefix sums come from PrefixScan between these kernels.

#include "child_slots.slang"

static const uint ACTION_SPLIT = 1u;
static const uint ACTION_CLONE = 2u;

bool isParent(uint i, StructuredBuffer<uint> action, StructuredBuffer<uint> pre_count_buf, uint max_splats)
{
    if (i >= min(pre_count_buf[0], max_splats)) return false;
    uint a = action[i];
    return a == ACTION_SPLIT || a == ACTION_CLONE;
}

// Growth of a structure-aware parent: 1 for a clone, factors.x*y*z - 1 for a split, at most
// max_splats (more could never be placed, and it keeps one parent from saturating the scan).
uint structureGrowth(uint i, uint a, StructuredBuffer<uint> fx, StructuredBuffer<uint> fy,
                     StructuredBuffer<uint> fz, uint max_splats)
{
    if (a == ACTION_CLONE) return 1u;
    return min(gridSize(fx[i], fy[i], fz[i]) - 1u, max_splats);
}

// Pass 1: 1 per parent, 0 everywhere else up to max_splats (the scan input of the parent list).
[shader("compute")]
[numthreads(256, 1, 1)]
void markParents(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> pre_count_buf,
    RWStructuredBuffer<uint> scratch,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    scratch[i] = isParent(i, action, pre_count_buf, max_splats) ? 1u : 0u;
}

// Pass 2 (after scanning the marks into parent_rank): list each parent at its rank, in index
// order, and replace the marks with each parent's growth (the scan input of the tail offsets).
[shader("compute")]
[numthreads(256, 1, 1)]
void listParentsAndGrowth(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> pre_count_buf,
    StructuredBuffer<uint> split_factor_x,
    StructuredBuffer<uint> split_factor_y,
    StructuredBuffer<uint> split_factor_z,
    StructuredBuffer<uint> parent_rank,
    RWStructuredBuffer<uint> selected_parents,
    RWStructuredBuffer<uint> scratch,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (!isParent(i, action, pre_count_buf, max_splats))
    {
        scratch[i] = 0u;
        return;
    }
    selected_parents[parent_rank[i]] = i;
    scratch[i] = structureGrowth(i, action[i], split_factor_x, split_factor_y, split_factor_z, max_splats);
}

// Simple densifier: every split or clone adds exactly one tail child.
[shader("compute")]
[numthreads(256, 1, 1)]
void markSimpleGrowth(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> pre_count_buf,
    RWStructuredBuffer<uint> scratch,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    scratch[i] = isParent(i, action, pre_count_buf, max_splats) ? 1u : 0u;
}

// The live count after the split: pre_count plus the requested children, capped at max_splats
// (children past it are never written), so the sum cannot overflow.
[shader("compute")]
[numthreads(1, 1, 1)]
void finishAllocation(
    StructuredBuffer<uint> pre_count_buf,
    StructuredBuffer<uint> growth_total,
    RWStructuredBuffer<uint> num_gaussians_counter,
    uniform uint max_splats)
{
    uint pre = min(pre_count_buf[0], max_splats);
    num_gaussians_counter[0] = pre + min(growth_total[0], max_splats - pre);
}
