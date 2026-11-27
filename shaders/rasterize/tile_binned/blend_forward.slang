// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_BLEND_FORWARD_SLANG
#define TRAINER_BLEND_FORWARD_SLANG

#include "../../core/interfaces.slang"
#include "../../core/compositor.slang"
#include "checkpoint.slang"
#include "common/execution.slang"
#include "../../common/geom_priors.slang"

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
    RWStructuredBuffer<uint2> raster_checkpoints[CHECKPOINT_CHUNKS],
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
                storeCheckpoint(raster_checkpoints, bucket, pixel_in_tile[j],
                    packCheckpoint(transmittance[j], color[j]));

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
    RWStructuredBuffer<uint2> raster_checkpoints[CHECKPOINT_CHUNKS],
    float extinction_hardness,
    out float3 color[N],
    out float transmittance[N],
    out uint processed[N])
{
    blendForwardImpl<S, N>(src, t_start, primitive_count, pixel_center, valid_pixel, pixel_in_tile,
                           tile_bucket_base, sorted_primitive_ids, raster_checkpoints, extinction_hardness,
                           color, transmittance, processed, true);
}

#if GEOM_PRIORS
// blendForward plus the geometry composite (common/geom_priors.slang): the same loop, thresholds,
// checkpoints and early break, with aux[j] += geom_data[id] * weight(alpha) * T beside the color.
// Kept as a separate function so the default path compiles exactly as before; a test pins that
// both write the same color, transmittance, counts and checkpoints.
void blendForwardAux<S : IPrimitiveSource, let N : int>(
    S src,
    uint t_start,
    uint primitive_count,
    no_diff float2 pixel_center[N],
    bool valid_pixel[N],
    uint pixel_in_tile[N],
    uint tile_bucket_base,
    StructuredBuffer<uint> sorted_primitive_ids,
    RWStructuredBuffer<uint2> raster_checkpoints[CHECKPOINT_CHUNKS],
    float extinction_hardness,
    StructuredBuffer<float4> geom_data,
    out float3 color[N],
    out float transmittance[N],
    out uint processed[N],
    out float4 aux[N],
    bool write_checkpoints)
{
    Compositor compositor;
    compositor.hardness = extinction_hardness;

    bool active[N];
    [unroll]
    for (int j = 0; j < N; ++j)
    {
        color[j] = float3(0.0);
        transmittance[j] = 1.0;
        processed[j] = 0u;
        aux[j] = float4(0.0);
        active[j] = valid_pixel[j];
    }

    for (uint local_index = 0u; local_index < primitive_count; ++local_index)
    {
        bool bucket_start = (local_index % RASTER_BUCKET_SIZE) == 0u;
        uint bucket = tile_bucket_base + local_index / RASTER_BUCKET_SIZE;
        uint primitive_id = sorted_primitive_ids[t_start + local_index];
        let primitive = src.primitiveAt(primitive_id);
        float4 geom = geom_data[primitive_id];

        bool any_active = false;
        [unroll]
        for (int j = 0; j < N; ++j)
        {
            if (!active[j])
                continue;
            if (bucket_start && write_checkpoints)
                storeCheckpoint(raster_checkpoints, bucket, pixel_in_tile[j],
                    packCheckpoint(transmittance[j], color[j]));

            float alpha = min(primitive.coverage(pixel_center[j]), BLEND_ALPHA_MAX);
            if (alpha >= BLEND_ALPHA_MIN)
            {
                float weight = compositor.weight(alpha) * transmittance[j];
                color[j] += primitive.color() * compositor.weight(alpha) * transmittance[j];
                aux[j] += geom * weight;
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
#endif // GEOM_PRIORS

#endif // TRAINER_BLEND_FORWARD_SLANG
