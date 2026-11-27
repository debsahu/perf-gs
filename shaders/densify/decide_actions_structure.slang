// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "../common/types.slang"
#include "../common/math.slang"
#include "blend_weight.slang"
#include "child_slots.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

uint saturatingAdd(uint a, uint b) { return a > 0xFFFFFFFFu - b ? 0xFFFFFFFFu : a + b; }

uint per_axis_split_factor(float eta_axis, float split_power, uint max_split_factor)
{
    float e = max(eta_axis, 0.0);
    float raw = ceil(pow(e, split_power));
    return clamp(uint(raw), 1u, max_split_factor);
}

// Propose each Gaussian's action and per-axis split factors before applying the slot budget.
[shader("compute")]
[numthreads(256, 1, 1)]
void decideActionsStructure(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> eta_n_total,
    StructuredBuffer<uint> eta_n_high,
    StructuredBuffer<uint> eta_n_low,
    StructuredBuffer<uint> eta_max_x,
    StructuredBuffer<uint> eta_max_y,
    StructuredBuffer<uint> eta_max_z,
    StructuredBuffer<int64_t> grad_mean_2d_mag_accum,
    StructuredBuffer<uint> visibility_count_accum,
    StructuredBuffer<int64_t> blend_weight_accum,
    RWStructuredBuffer<uint> action,
    RWStructuredBuffer<uint> split_factor_x,
    RWStructuredBuffer<uint> split_factor_y,
    RWStructuredBuffer<uint> split_factor_z,
    RWStructuredBuffer<uint> densify_children_sum,  // atomic: total requested tail growth (for overbudget log)
    uniform float tau_split, uniform float tau_prune,
    uniform float grad_threshold_internal,
    uniform float split_power, uniform uint max_split_factor,
    uniform float t_opacity, uniform float dense_scale,
    uniform float min_blend_weight,
    uniform uint max_splats)
{
    uint gaussian_idx = dtid.x;
    if (gaussian_idx >= max_splats) return;

    if (gaussian_idx >= num_gaussians_counter[0]) return;

    uint n_total_u = eta_n_total[gaussian_idx];
    bool seen = n_total_u > 0u;
    float inv_total = seen ? (1.0 / float(max(n_total_u, 1u))) : 0.0;

    float high_ratio = float(eta_n_high[gaussian_idx]) * inv_total;
    float low_ratio  = float(eta_n_low[gaussian_idx])  * inv_total;

    // Eta maxima are uint(eta*FIXED_SCALE): decode with a plain divide, not fromFixed (int64 grad convention).
    float eta_x = float(eta_max_x[gaussian_idx]) / FIXED_SCALE;
    float eta_y = float(eta_max_y[gaussian_idx]) / FIXED_SCALE;
    float eta_z = float(eta_max_z[gaussian_idx]) / FIXED_SCALE;

    uint vis = visibility_count_accum[gaussian_idx];
    float avg_grad = 0.0;
    if (vis > 0u)
        avg_grad = fromFixed(grad_mean_2d_mag_accum[gaussian_idx]) / float(vis);
    bool is_grad_high = avg_grad >= grad_threshold_internal;

    Gaussian3D g = gaussians[gaussian_idx];
    float opacity = sigmoid(g.opacity);
    float3 scale = exp(g.log_scale);
    float max_scale = max(scale.x, max(scale.y, scale.z));

    uint out_action = ACTION_KEEP;
    uint nkx = 1u, nky = 1u, nkz = 1u;

    bool is_prune = ((low_ratio > tau_prune) && seen) || (opacity < t_opacity)
        || isUnderweight(blend_weight_accum[gaussian_idx], vis, min_blend_weight);
    bool triggered = (high_ratio > tau_split) && is_grad_high && seen;

    if (is_prune)
    {
        out_action = ACTION_PRUNE;
    }
    else if (triggered)
    {
        nkx = per_axis_split_factor(eta_x, split_power, max_split_factor);
        nky = per_axis_split_factor(eta_y, split_power, max_split_factor);
        nkz = per_axis_split_factor(eta_z, split_power, max_split_factor);
        uint prod = gridSize(nkx, nky, nkz);   // saturating: a 2**32 grid must not read as 0

        bool is_small = (max_scale <= dense_scale);
        if (is_small || prod <= 1u)
        {
            out_action = ACTION_CLONE;
            nkx = 1u; nky = 1u; nkz = 1u;
        }
        else
        {
            out_action = ACTION_SPLIT;
        }
    }

    action[gaussian_idx] = out_action;
    split_factor_x[gaussian_idx] = nkx;
    split_factor_y[gaussian_idx] = nky;
    split_factor_z[gaussian_idx] = nkz;

    // The requested growth only feeds the overbudget report: an integer sum, so its order
    // does not matter. The parent list is built in index order by allocate_children.slang.
    // Each request is capped at max_splats, as structureGrowth caps it (more could never be placed). The sum must
    // not wrap into "within budget": every add is positive, so the true sum reaches 2**32 exactly when some add
    // wraps the running counter, and that add sets word 1, which the host reads as a saturated 0xFFFFFFFF. One
    // atomic add per wave, as before (a compare-exchange loop on the one word doubled the tick). The wave sum is
    // taken in 16-bit halves so it cannot wrap either (a wave of 64 lanes sums at most 64 * 0xFFFF per half).
    uint requested = out_action == ACTION_SPLIT ? min(gridSize(nkx, nky, nkz) - 1u, max_splats)
                   : (out_action == ACTION_CLONE ? 1u : 0u);
    uint wave_hi = WaveActiveSum(requested >> 16u);
    uint wave_lo = WaveActiveSum(requested & 0xFFFFu);
    if (WaveIsFirstLane() && (wave_hi | wave_lo) != 0u)
    {
        uint add = wave_hi >= 0x10000u ? 0xFFFFFFFFu : saturatingAdd(wave_hi << 16u, wave_lo);
        uint before;
        InterlockedAdd(densify_children_sum[0], add, before);
        if (before > 0xFFFFFFFFu - add || add == 0xFFFFFFFFu)
            InterlockedOr(densify_children_sum[1], 1u);
    }
}
