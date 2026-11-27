// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Tile-binned forward rasterizer, front to back: one warp per tile with two pixels per lane (rows r and
// r + TILE_HEIGHT / 2), so each splat load serves two pixels; a workgroup holds two vertically adjacent
// tiles. Writes each pixel's early-break stop index so the backward replays the same entries.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/eval.slang"
#include "../../primitives/gaussian/blend.slang"
#include "blend_forward.slang"
#include "common/execution.slang"

static const uint PIXELS_PER_THREAD = 2u;
static const uint TILE_ROWS_PER_PASS = TILE_HEIGHT / PIXELS_PER_THREAD;
static const uint TILES_PER_GROUP = 2u;

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_ROWS_PER_PASS * TILES_PER_GROUP, 1)]
void forwardRasterize(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    // Compact blend record packed by bin_count, gathered identically by the backward.
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,   // the value array after sort
    StructuredBuffer<CameraData> camera,
    // Ragged (CSR) (T, C_in_front) checkpoints, one per RASTER_BUCKET_SIZE splats, for the per-splat backward.
    StructuredBuffer<uint> bucket_offsets,
    RWStructuredBuffer<uint2> raster_checkpoints[CHECKPOINT_CHUNKS],
    RWTexture2D<float4> output_image,
    RWTexture2D<uint> gaussian_count_per_pixel,
#if GEOM_PRIORS
    // Per-splat (n, s) from geom_features.slang, and their composite (N, S) per pixel. aux_on = 0 skips the
    // composite (the viewer, and steps before --geom-from): geom_data and aux_image are then never read or written.
    StructuredBuffer<float4> geom_data,
    RWTexture2D<float4> aux_image,
    uniform uint aux_on,
#endif
    uniform float extinction_hardness,
    // 0 for a render-only pipeline: no backward reads the checkpoints, and raster_checkpoints is a stub.
    uniform uint write_checkpoints)
{
    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint2 tile_dim = uint2(TILE_WIDTH, TILE_HEIGHT);
    uint2 num_tiles = (frame_dim_px + tile_dim - 1u) / tile_dim;
    uint2 tile = uint2(gid.x, gid.y * TILES_PER_GROUP + gtid.y / TILE_ROWS_PER_PASS);
    if (tile.y >= num_tiles.y) return;

    uint tile_idx = tile.y * num_tiles.x + tile.x;
    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0;
    uint tile_bucket_base = bucket_offsets[tile_idx];

    uint2 pixel[PIXELS_PER_THREAD];
    float2 pixel_center[PIXELS_PER_THREAD];
    bool valid_pixel[PIXELS_PER_THREAD];
    uint pixel_in_tile[PIXELS_PER_THREAD];   // row-major slot in each bucket's checkpoint block
    [unroll]
    for (uint j = 0u; j < PIXELS_PER_THREAD; ++j)
    {
        uint2 in_tile = uint2(gtid.x, gtid.y % TILE_ROWS_PER_PASS + j * TILE_ROWS_PER_PASS);
        pixel[j] = tile * tile_dim + in_tile;
        pixel_center[j] = float2(pixel[j]) + 0.5;
        valid_pixel[j] = pixel[j].x < frame_dim_px.x && pixel[j].y < frame_dim_px.y;
        pixel_in_tile[j] = in_tile.y * TILE_WIDTH + in_tile.x;
    }

    GaussianSource src;
    src.blend_data = blend_data;

    float3 C[PIXELS_PER_THREAD];
    float T[PIXELS_PER_THREAD];
    uint processed[PIXELS_PER_THREAD];
#if GEOM_PRIORS
    float4 aux[PIXELS_PER_THREAD];
    if (aux_on != 0u)
        blendForwardAux<GaussianSource, PIXELS_PER_THREAD>(
            src, t_start, tile_count, pixel_center, valid_pixel,
            pixel_in_tile, tile_bucket_base, sorted_gaussian_ids,
            raster_checkpoints, extinction_hardness, geom_data, C, T, processed, aux, write_checkpoints != 0u);
    else
        blendForwardImpl<GaussianSource, PIXELS_PER_THREAD>(
            src, t_start, tile_count, pixel_center, valid_pixel,
            pixel_in_tile, tile_bucket_base, sorted_gaussian_ids,
            raster_checkpoints, extinction_hardness, C, T, processed, write_checkpoints != 0u);
#else
    blendForwardImpl<GaussianSource, PIXELS_PER_THREAD>(
        src, t_start, tile_count, pixel_center, valid_pixel,
        pixel_in_tile, tile_bucket_base, sorted_gaussian_ids,
        raster_checkpoints, extinction_hardness, C, T, processed, write_checkpoints != 0u);
#endif

    [unroll]
    for (uint j = 0u; j < PIXELS_PER_THREAD; ++j)
    {
        if (!valid_pixel[j])
            continue;
        // The backward recovers final transmittance as T = 1 - output_image.a.
        output_image[pixel[j]] = float4(C[j] * camera[0].lensGain(pixel[j]), 1.0 - T[j]);
        gaussian_count_per_pixel[pixel[j]] = processed[j];
#if GEOM_PRIORS
        if (aux_on != 0u)
            aux_image[pixel[j]] = aux[j];   // no lens gain: geometry is not color
#endif
    }
}
