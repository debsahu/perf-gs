// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// L1 (+ optional L2) photometric loss gradient with optional alpha supervision, in sRGB as in 3DGS.
// Sum-over-pixels convention (no 1/N), matching the backward rasterizer's accumulation.
// LOSS_MASK: target.a is a per-pixel keep mask (1 keep, 0 drop), so the color gradient is gated by
// it and alpha supervision is off (the host refuses --w-alpha with --masks).

#include "../common/stbn.slang"
#include "../common/loss_mask.slang"

[shader("compute")]
[numthreads(16, 16, 1)]
void computeLossGrad(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float4> target_image,
    RWTexture2D<float4> dL_dC,
    // STBN sparse supervision; stbn_tile_pixels == TILE_AREA disables it.
    StructuredBuffer<uint> stbn_table,
    uniform uint2 stbn_dims,
    uniform uint stbn_slice,
    uniform uint stbn_tile_pixels,
    uniform uint2 frame_dim,
    uniform float w_alpha,
    uniform float lambda_l2)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    // dL_dC is rewritten every iteration, so unsampled pixels must write zero, not skip.
    if (!stbnSelected(stbn_table, p, stbn_dims, stbn_slice, stbn_tile_pixels))
    {
        dL_dC[p] = float4(0.0);
        return;
    }
    // Unbiasing scale applied here so every backward variant inherits it.
    float stbn_scale = stbnWeight(stbn_tile_pixels);

    float4 rendered = output_image[p];
    float4 target = target_image[p];

    float3 C = rendered.rgb;
    float3 T = target.rgb;   // sRGB, compared directly (no linearization)

    float3 diff = C - T;
    float3 grad_rgb = sign(diff);                    // L1
    if (lambda_l2 > 0.0)
        grad_rgb += lambda_l2 * 2.0 * diff;          // + lambda_l2 * d/dC (C-T)^2

    float grad_alpha = 0.0;
#if LOSS_MASK
    // Select, not multiply: NaN * 0 is NaN, and nothing keeps a dropped pixel's render finite.
    grad_rgb = target.a > 0.0 ? grad_rgb * target.a : float3(0.0);
#else
    if (w_alpha > 0.0)
    {
        float A_rendered = rendered.a;
        float A_target = target.a;
        grad_alpha = w_alpha * sign(A_rendered - A_target);
    }
#endif

    dL_dC[p] = float4(grad_rgb * stbn_scale, grad_alpha * stbn_scale);
}
