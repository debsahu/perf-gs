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

// ---------------------------------------------------------------- cap-full thinning (--cap-order hash)
//
// When the requested children do not fit, keep a spatially uniform subset of parents instead of the
// lowest indices (index order follows the Morton order, so lowest-index-first clusters survivors in one
// corner of the scene). Each parent gets a hash of (index, tick, seed); parents are admitted in ascending
// hash order until the budget is spent, all or nothing per parent. That is exact selection by hash rank:
// a 256-bin histogram of the cost by the hash's top byte picks the cutoff bin; bins below it are admitted
// whole, and inside the cutoff bin parents are admitted in index order while their prefix fits (the
// leftover is at most one parent's cost). The admitted total never exceeds the budget.

static const uint HASH_BINS = 256u;
static const uint HASH_CUTOFF = 2u * HASH_BINS, HASH_REMAINING = HASH_CUTOFF + 1u;

uint pcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint capHashBin(uint i, uint tick, uint seed)
{
    return pcgHash(i ^ pcgHash(tick ^ pcgHash(seed))) >> 24u;
}

// Each parent's cost (its growth) for the unranked thinning; 0 for everything else.
[shader("compute")]
[numthreads(256, 1, 1)]
void writeParentCost(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> pre_count_buf,
    StructuredBuffer<uint> split_factor_x,
    StructuredBuffer<uint> split_factor_y,
    StructuredBuffer<uint> split_factor_z,
    RWStructuredBuffer<uint> cost,
    uniform uint simple,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (!isParent(i, action, pre_count_buf, max_splats)) { cost[i] = 0u; return; }
    cost[i] = simple != 0u ? 1u
            : structureGrowth(i, action[i], split_factor_x, split_factor_y, split_factor_z, max_splats);
}

// The unranked budget: the free slots after the live splats.
[shader("compute")]
[numthreads(1, 1, 1)]
void writeHeadroom(
    StructuredBuffer<uint> pre_count_buf,
    RWStructuredBuffer<uint> histogram,
    uniform uint max_splats)
{
    histogram[HASH_REMAINING] = max_splats - min(pre_count_buf[0], max_splats);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void hashHistogram(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> cost,
    RWStructuredBuffer<uint> histogram,
    uniform uint tick, uniform uint seed, uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats || cost[i] == 0u) return;
    uint bin = capHashBin(i, tick, seed), old;
    InterlockedAdd(histogram[bin], cost[i], old);
    if (old > 0xFFFFFFFFu - cost[i]) InterlockedOr(histogram[HASH_BINS + bin], 1u);
}

// One thread: the first hash bin that does not fit the budget (HASH_BINS when all fit) and what is left.
[shader("compute")]
[numthreads(1, 1, 1)]
void hashCutoff(RWStructuredBuffer<uint> histogram)
{
    uint slots = histogram[HASH_REMAINING];
    uint bin = 0u;
    while (bin < HASH_BINS && histogram[HASH_BINS + bin] == 0u && histogram[bin] <= slots)
    {
        slots -= histogram[bin];
        ++bin;
    }
    histogram[HASH_CUTOFF] = bin;
    histogram[HASH_REMAINING] = slots;
}

// The cutoff bin's costs, in index order (the scan input of the in-bin prefix).
[shader("compute")]
[numthreads(256, 1, 1)]
void hashCutoffCost(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> cost,
    StructuredBuffer<uint> histogram,
    RWStructuredBuffer<uint> scratch,
    uniform uint tick, uniform uint seed, uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    bool in_bin = cost[i] != 0u && capHashBin(i, tick, seed) == histogram[HASH_CUTOFF];
    scratch[i] = in_bin ? cost[i] : 0u;
}

// admit[i] = 1 when parent i is admitted: its hash bin is below the cutoff, or it is in the cutoff bin and
// the costs of the lower-index cutoff-bin parents plus its own fit what is left. admit may alias scratch.
[shader("compute")]
[numthreads(256, 1, 1)]
void hashAdmit(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> cost,
    StructuredBuffer<uint> histogram,
    StructuredBuffer<uint> cutoff_offset,
    RWStructuredBuffer<uint> admit,
    uniform uint tick, uniform uint seed, uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    uint c = cost[i];
    uint bin = capHashBin(i, tick, seed), cutoff = histogram[HASH_CUTOFF];
    bool ok = c == 0u || bin < cutoff
           || (bin == cutoff && cutoff_offset[i] <= histogram[HASH_REMAINING] - min(c, histogram[HASH_REMAINING])
               && c <= histogram[HASH_REMAINING]);
    admit[i] = ok ? 1u : 0u;
}

// Index order (--cap-order index) for the ranked cutoff bin: admitted while the prefix fits.
[shader("compute")]
[numthreads(256, 1, 1)]
void indexAdmit(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> cost,
    StructuredBuffer<uint> cutoff_offset,
    StructuredBuffer<uint> budget,
    RWStructuredBuffer<uint> admit,
    uniform uint budget_index, uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    uint c = cost[i], rem = budget[budget_index];
    admit[i] = (c == 0u || (c <= rem && cutoff_offset[i] <= rem - c)) ? 1u : 0u;
}

// Unranked thinning: a parent that was not admitted keeps its slot unchanged (its split or clone is dropped).
[shader("compute")]
[numthreads(256, 1, 1)]
void dropUnadmitted(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> cost,
    StructuredBuffer<uint> admit,
    RWStructuredBuffer<uint> action,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (cost[i] != 0u && admit[i] == 0u) action[i] = 0u;
}

// dst[dst_index] = src[src_index] (one thread).
[shader("compute")]
[numthreads(1, 1, 1)]
void copyUint(StructuredBuffer<uint> src, RWStructuredBuffer<uint> dst, uniform uint src_index, uniform uint dst_index)
{
    dst[dst_index] = src[src_index];
}
