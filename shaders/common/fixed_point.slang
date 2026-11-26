// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_FIXED_POINT_SLANG
#define TRAINER_FIXED_POINT_SLANG

// Fixed-point atomic gradient accumulation (int64, no overflow).
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

int64_t toFixed(float value) { return int64_t(value * FIXED_SCALE); }
float fromFixed(int64_t value) { return float(value) * FIXED_INV_SCALE; }

// MoltenVK patch: Metal has no 64-bit atomics. Same 8-byte layout as int64_t (little endian);
// add the low word atomically, derive the carry from the returned old value, add it to the high word.
struct I64Atomic { uint lo; uint hi; };
void atomicAdd64(RWStructuredBuffer<I64Atomic> b, uint i, int64_t v)
{
    uint64_t u = uint64_t(v);
    uint lo = uint(u & 0xffffffffull);
    uint hi = uint(u >> 32);
    uint old;
    InterlockedAdd(b[i].lo, lo, old);
    uint carry = ((old + lo) < old) ? 1u : 0u;
    InterlockedAdd(b[i].hi, hi + carry);
}
int64_t loadI64(I64Atomic a) { return int64_t((uint64_t(a.hi) << 32) | uint64_t(a.lo)); }

#endif // TRAINER_FIXED_POINT_SLANG
