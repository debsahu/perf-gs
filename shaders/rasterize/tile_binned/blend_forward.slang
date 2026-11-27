// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_BLEND_FORWARD_SLANG
#define TRAINER_BLEND_FORWARD_SLANG

#include "../../core/interfaces.slang"
#include "../../core/compositor.slang"
#include "checkpoint.slang"
#include "common/execution.slang"

static const float BLEND_T_EARLY_BREAK = 1e-4;
static const float BLEND_ALPHA_MAX = 0.99;
static const float BLEND_ALPHA_MIN = 1.0 / 255.0;

// Shared front-to-back tile blend (compositing, checkpoints, early-out) for any IPrimitiveSource, over
// N pixels of one tile that share each primitive load.
// write_checkpoints = false (a render-only viewer) skips the checkpoint writes: nothing reads them, so the
// buffer may be a one-element stub. Colour, transmittance and counts are unchanged.
void blendForwardImpl<S : IPrimitiveSource, let N : int>(
    S src,
    uint t_start,
    uint primitive_count,
    no_diff float2 pixel_center[N],
    bool valid_pixel[N],
    uint pixel_in_tile[N],
    uint tile_bucket_base,
    StructuredBuffer<uint> sorted_primitive_ids,
    RWStructuredBuffer<uint2> raster_checkpoints,
    float extinction_hardness,
    out float3 color[N],
    out float transmittance[N],
    out uint processed[N],
    bool write_checkpoints)
{
    Compositor compositor;
    compositor.hardness = extinction_hardness;

    // Pixels still blending: in the frame and not yet saturated.
    bool active[N];
    [unroll]
    for (int j = 0; j < N; ++j)
    {
        color[j] = float3(0.0);
        transmittance[j] = 1.0;
        processed[j] = 0u;
        active[j] = valid_pixel[j];
    }

    for (uint local_index = 0u; local_index < primitive_count; ++local_index)
    {
        bool bucket_start = (local_index % RASTER_BUCKET_SIZE) == 0u;
        uint bucket = tile_bucket_base + local_index / RASTER_BUCKET_SIZE;
        let primitive = src.primitiveAt(sorted_primitive_ids[t_start + local_index]);

        bool any_active = false;
        [unroll]
        for (int j = 0; j < N; ++j)
        {
            if (!active[j])
                continue;
            if (bucket_start && write_checkpoints)
                raster_checkpoints[checkpointIndex(bucket, pixel_in_tile[j])] =
                    packCheckpoint(transmittance[j], color[j]);

            float alpha = min(primitive.coverage(pixel_center[j]), BLEND_ALPHA_MAX);
            if (alpha >= BLEND_ALPHA_MIN)
            {
                color[j] += primitive.color() * compositor.weight(alpha) * transmittance[j];
                transmittance[j] *= compositor.transmittance(alpha);
                processed[j] = local_index + 1u;
                active[j] = transmittance[j] >= BLEND_T_EARLY_BREAK;
            }
            any_active = any_active || active[j];
        }
        if (!any_active)
            break;
    }
}

// blendForwardImpl writing every checkpoint (the backward reads them).
void blendForward<S : IPrimitiveSource, let N : int>(
    S src,
    uint t_start,
    uint primitive_count,
    no_diff float2 pixel_center[N],
    bool valid_pixel[N],
    uint pixel_in_tile[N],
    uint tile_bucket_base,
    StructuredBuffer<uint> sorted_primitive_ids,
    RWStructuredBuffer<uint2> raster_checkpoints,
    float extinction_hardness,
    out float3 color[N],
    out float transmittance[N],
    out uint processed[N])
{
    blendForwardImpl<S, N>(src, t_start, primitive_count, pixel_center, valid_pixel, pixel_in_tile,
                           tile_bucket_base, sorted_primitive_ids, raster_checkpoints, extinction_hardness,
                           color, transmittance, processed, true);
}

#endif // TRAINER_BLEND_FORWARD_SLANG
