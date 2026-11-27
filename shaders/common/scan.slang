// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Exclusive prefix scan of a uint buffer (out[0] = 0) that also writes the total sum.
// Sums saturate at 0xFFFFFFFF instead of wrapping (saturating addition is still associative), so a
// huge total can never wrap an offset back into the live range.
// Level-agnostic: the host chains 2 levels (n <= SCAN_BLOCK_SIZE^2) or 3 (n <= SCAN_BLOCK_SIZE^3).

#ifndef TRAINER_SCAN_SLANG
#define TRAINER_SCAN_SLANG

#include "common/execution.slang"
// scanBlockSums is a single workgroup, so it scans at most SCAN_BLOCK_SIZE block sums.
static const uint SCAN_MAX_BLOCKS = 1024;

groupshared uint _scan_lds[SCAN_BLOCK_SIZE];

uint _scan_add(uint a, uint b)
{
    return a > 0xFFFFFFFFu - b ? 0xFFFFFFFFu : a + b;
}

// In-place exclusive Hillis-Steele scan of _scan_lds[0..count); returns the total.
uint _scan_lds_inplace(uint lid, uint count)
{
    for (uint offset = 1u; offset < count; offset <<= 1u)
    {
        GroupMemoryBarrierWithGroupSync();
        uint add_val = 0u;
        if (lid >= offset && lid < count) add_val = _scan_lds[lid - offset];
        GroupMemoryBarrierWithGroupSync();
        if (lid < count) _scan_lds[lid] = _scan_add(_scan_lds[lid], add_val);
    }
    GroupMemoryBarrierWithGroupSync();

    // Convert inclusive -> exclusive by shifting right 1.
    uint total = (count > 0u) ? _scan_lds[count - 1u] : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint my_prev = (lid == 0u || lid >= count) ? 0u : _scan_lds[lid - 1u];
    GroupMemoryBarrierWithGroupSync();
    if (lid < count) _scan_lds[lid] = my_prev;
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[shader("compute")]
[numthreads(SCAN_BLOCK_SIZE, 1, 1)]
void scanBlock(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint lid: SV_GroupIndex,
    StructuredBuffer<uint> in_buf,
    RWStructuredBuffer<uint> out_buf,
    RWStructuredBuffer<uint> block_sums,
    uniform uint n)
{
    uint global_idx = gid.x * SCAN_BLOCK_SIZE + lid;
    uint block_start = gid.x * SCAN_BLOCK_SIZE;
    uint block_count = min(SCAN_BLOCK_SIZE, (n > block_start) ? (n - block_start) : 0u);

    _scan_lds[lid] = (global_idx < n) ? in_buf[global_idx] : 0u;

    uint total = _scan_lds_inplace(lid, block_count);

    if (global_idx < n) out_buf[global_idx] = _scan_lds[lid];
    if (lid == 0u) block_sums[gid.x] = total;
}

// Single workgroup: scans num_blocks <= SCAN_MAX_BLOCKS block sums in place.
[shader("compute")]
[numthreads(SCAN_BLOCK_SIZE, 1, 1)]
void scanBlockSums(
    uint3 dtid: SV_DispatchThreadID,
    uint lid: SV_GroupIndex,
    RWStructuredBuffer<uint> block_sums,
    RWStructuredBuffer<uint> total_out,
    uniform uint num_blocks)
{
    _scan_lds[lid] = (lid < num_blocks) ? block_sums[lid] : 0u;
    uint total = _scan_lds_inplace(lid, num_blocks);
    if (lid < num_blocks) block_sums[lid] = _scan_lds[lid];
    if (lid == 0u) total_out[0] = total;
}

[shader("compute")]
[numthreads(SCAN_BLOCK_SIZE, 1, 1)]
void addBlockOffsets(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint lid: SV_GroupIndex,
    RWStructuredBuffer<uint> out_buf,
    StructuredBuffer<uint> block_sums,
    uniform uint n)
{
    uint global_idx = gid.x * SCAN_BLOCK_SIZE + lid;
    if (global_idx >= n) return;
    out_buf[global_idx] = _scan_add(out_buf[global_idx], block_sums[gid.x]);
}

#endif // TRAINER_SCAN_SLANG
