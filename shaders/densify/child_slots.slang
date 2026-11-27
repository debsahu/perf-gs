// Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

// Overflow-safe arithmetic for densify child slots, shared by the allocation and split kernels.

#ifndef TRAINER_CHILD_SLOTS_SLANG
#define TRAINER_CHILD_SLOTS_SLANG

// x * y, saturating at 0xFFFFFFFF.
uint saturatingMul(uint x, uint y)
{
    return (y != 0u && x > 0xFFFFFFFFu / y) ? 0xFFFFFFFFu : x * y;
}

// Children in a split grid of these factors, saturating (never wraps).
uint gridSize(uint fx, uint fy, uint fz)
{
    return saturatingMul(saturatingMul(max(fx, 1u), max(fy, 1u)), max(fz, 1u));
}

// Tail slot of a parent's first new child, max_splats when it starts past the cap (no overflow).
uint tailStart(uint pre_count, uint offset, uint max_splats)
{
    uint pre = min(pre_count, max_splats);
    return offset >= max_splats - pre ? max_splats : pre + offset;
}

#endif // TRAINER_CHILD_SLOTS_SLANG
