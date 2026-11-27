// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "../common/fixed_point.slang"

// The first 256 counters hold requested slots. The next 256 flag counter overflow.
static const uint BINS = 256u;
static const uint CUTOFF = 2u * BINS, REMAINING = CUTOFF + 1u;

// Eight buckets per power of two keep ranking inexpensive at a densification tick.
uint priorityBin(float score)
{
    return uint(clamp(int(asuint(score) >> 20u) - (127 - 16) * 8, 0, int(BINS) - 1));
}

// Upgrades in the cutoff bin share the slots left after every higher bin. They go to parents
// in index order: a parent gets all of its cutoff-bin upgrades when the cutoff costs of every
// lower-index parent plus its own fit (cutoff_offset is their exclusive prefix sum), else none
// and it stops there. Pass 1 (apply = 1) measures each parent's cutoff cost; pass 2 grants.
[shader("compute")]
[numthreads(256, 1, 1)]
void rankFamilies(uint3 tid : SV_DispatchThreadID, StructuredBuffer<uint> pre_count,
    RWStructuredBuffer<uint> histogram, RWStructuredBuffer<uint> action,
    RWStructuredBuffer<uint> factor_x, RWStructuredBuffer<uint> factor_y, RWStructuredBuffer<uint> factor_z,
    StructuredBuffer<uint> eta_x, StructuredBuffer<uint> eta_y, StructuredBuffer<uint> eta_z,
    StructuredBuffer<uint> views, StructuredBuffer<int64_t> gradient,
    RWStructuredBuffer<uint> cutoff_cost, StructuredBuffer<uint> cutoff_offset,
    uniform float gradient_scale, uniform uint capacity, uniform uint apply)
{
    uint i = tid.x;
    if (i >= min(capacity, pre_count[0])) return;
    uint act = action[i];
    if (act != 1u && act != 2u) return;
    uint3 factors = act == 2u ? uint3(2u, 1u, 1u) : uint3(factor_x[i], factor_y[i], factor_z[i]);
    uint3 eta = uint3(eta_x[i], eta_y[i], eta_z[i]);
    float evidence = fromFixed(gradient[i]) / max(float(views[i]), 1.0) / gradient_scale;
    uint3 granted = uint3(1u);
    uint size = 1u;
    uint cut = 0u;
    bool cutoff_fits = apply == 2u && cutoff_offset[i] + cutoff_cost[i] <= histogram[REMAINING];
    for (uint level = 0u; level < 3u; ++level)
    {
        // Each upgrade adds one complete axis, strongest structure signal first.
        uint axis = 3u;
        for (uint a = 0u; a < 3u; ++a)
            if (factors[a] > 1u && (axis == 3u || eta[a] > eta[min(axis, 2u)])) axis = a;
        if (axis == 3u || factors[min(axis, 2u)] > capacity / size) break;
        uint next_size = size * factors[axis];
        uint cost = next_size - size;
        uint bin = priorityBin(evidence / float(next_size));
        if (apply == 0u)
        {
            uint old;
            InterlockedAdd(histogram[bin], cost, old);
            if (old > 0xffffffffu - cost) InterlockedOr(histogram[BINS + bin], 1u);
        }
        else
        {
            uint cutoff = histogram[CUTOFF];
            if (bin < cutoff) break;
            if (bin == cutoff)
            {
                if (apply == 1u) cut += cost;
                else if (!cutoff_fits) break;
            }
            granted[axis] = factors[axis];
        }
        size = next_size;
        factors[axis] = 1u;
    }
    if (apply == 1u)
    {
        cutoff_cost[i] = cut;
        return;
    }
    if (apply != 0u)
    {
        if (size == 1u) action[i] = 0u;
        if (act == 1u)
        {
            factor_x[i] = granted.x;
            factor_y[i] = granted.y;
            factor_z[i] = granted.z;
        }
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void chooseCutoff(RWStructuredBuffer<uint> histogram, StructuredBuffer<uint> pre_count, uniform uint capacity)
{
    uint slots = capacity - min(capacity, pre_count[0]);
    uint bin = BINS - 1u;
    while (bin > 0u && histogram[BINS + bin] == 0u && histogram[bin] <= slots)
    {
        slots -= histogram[bin];
        --bin;
    }
    histogram[CUTOFF] = bin;
    histogram[REMAINING] = slots;
}
