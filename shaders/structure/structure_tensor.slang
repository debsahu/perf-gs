// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

static const float ST_EPS = 1e-8;

[shader("compute")]
[numthreads(16, 16, 1)]
void blurH(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> src,
    RWTexture2D<float4> dst,
    uniform uint2 dim,
    uniform float sigma,
    uniform int radius)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;

    float inv2s2 = 1.0 / (2.0 * sigma * sigma);
    float4 acc = float4(0.0);
    float wsum = 0.0;
    int W = int(dim.x);
    for (int i = -radius; i <= radius; ++i)
    {
        int x = clamp(int(p.x) + i, 0, W - 1);   // mode="nearest"
        float w = exp(-float(i * i) * inv2s2);
        acc += w * src[uint2(uint(x), p.y)];
        wsum += w;
    }
    dst[p] = acc / wsum;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void blurV(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> src,
    RWTexture2D<float4> dst,
    uniform uint2 dim,
    uniform float sigma,
    uniform int radius)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;

    float inv2s2 = 1.0 / (2.0 * sigma * sigma);
    float4 acc = float4(0.0);
    float wsum = 0.0;
    int H = int(dim.y);
    for (int i = -radius; i <= radius; ++i)
    {
        int y = clamp(int(p.y) + i, 0, H - 1);   // mode="nearest"
        float w = exp(-float(i * i) * inv2s2);
        acc += w * src[uint2(p.x, uint(y))];
        wsum += w;
    }
    dst[p] = acc / wsum;
}

// Di Zenzo multi-channel structure tensor of the pre-smoothed image: edge-clamped 3x3 Sobel
// per channel, summed over RGB.
[shader("compute")]
[numthreads(16, 16, 1)]
void sobelTensor(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> smoothed,
    RWTexture2D<float4> tensorOut,   // (Sxx, Sxy, Syy, 0)
    uniform uint2 dim)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;

    int W = int(dim.x), H = int(dim.y);
    int xm = clamp(int(p.x) - 1, 0, W - 1);
    int xp = clamp(int(p.x) + 1, 0, W - 1);
    int ym = clamp(int(p.y) - 1, 0, H - 1);
    int yp = clamp(int(p.y) + 1, 0, H - 1);
    int xc = int(p.x), yc = int(p.y);

    float3 tl = smoothed[uint2(uint(xm), uint(ym))].xyz;
    float3 tm = smoothed[uint2(uint(xc), uint(ym))].xyz;
    float3 tr = smoothed[uint2(uint(xp), uint(ym))].xyz;
    float3 ml = smoothed[uint2(uint(xm), uint(yc))].xyz;
    float3 mr = smoothed[uint2(uint(xp), uint(yc))].xyz;
    float3 bl = smoothed[uint2(uint(xm), uint(yp))].xyz;
    float3 bm = smoothed[uint2(uint(xc), uint(yp))].xyz;
    float3 br = smoothed[uint2(uint(xp), uint(yp))].xyz;

    // Sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]] (right - left).
    float3 ix = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    // Sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]] (bottom - top).
    float3 iy = (bl + 2.0 * bm + br) - (tl + 2.0 * tm + tr);

    tensorOut[p] = float4(dot(ix, ix), dot(ix, iy), dot(iy, iy), 0.0);
}

// Max-of-trace reduction; trace >= 0, so InterlockedMax on its IEEE bits (asuint) is the true max.
[shader("compute")]
[numthreads(1, 1, 1)]
void clearMaxTrace(uint3 dtid: SV_DispatchThreadID, RWStructuredBuffer<uint> maxTrace)
{
    if (dtid.x == 0u) maxTrace[0] = 0u;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void reduceMaxTrace(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> tensorIn,
    RWStructuredBuffer<uint> maxTrace,
    uniform uint2 dim)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;
    float tr = max(tensorIn[p].x + tensorIn[p].z, 0.0);
    uint prev;
    InterlockedMax(maxTrace[0], asuint(tr), prev);
}

// SADGS batch-max normalization of the base tensor, in place.
[shader("compute")]
[numthreads(16, 16, 1)]
void normalizeBase(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> tensor,
    RWStructuredBuffer<uint> maxTrace,
    uniform uint2 dim)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;
    float maxt = asfloat(maxTrace[0]) + 1e-6;
    float4 v = tensor[p];
    tensor[p] = float4(v.xyz / maxt, 0.0);
}

// DoG band response: L2 norm over channels of (current_smooth - next_smooth).
[shader("compute")]
[numthreads(16, 16, 1)]
void dogBand(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> cur,
    RWTexture2D<float4> nxt,
    RWTexture2D<float4> bandOut,   // response in .x
    uniform uint2 dim)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;
    float3 d = cur[p].xyz - nxt[p].xyz;
    bandOut[p] = float4(sqrt(dot(d, d)), 0.0, 0.0, 0.0);
}

// Per-level accumulation (SADGS Eq. 5): num.xyz += band^gamma * freq_sq * the trace-normalized
// base tensor re-integrated by rho_l; den.w += band^gamma.
[shader("compute")]
[numthreads(16, 16, 1)]
void clearAccum(uint3 dtid: SV_DispatchThreadID, RWTexture2D<float4> accum, uniform uint2 dim)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;
    accum[p] = float4(0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void accumulateLevel(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> band,
    RWTexture2D<float4> intTensor,
    RWTexture2D<float4> accum,     // (num.xyz, den.w)
    uniform uint2 dim,
    uniform float gamma,
    uniform float freq_sq)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;

    float3 t = intTensor[p].xyz;             // (Sxx_i, Sxy_i, Syy_i)
    float tr = t.x + t.z + ST_EPS;
    float3 n = t / tr;                        // trace-normalized orientation
    float weight = pow(band[p].x, gamma);     // E_l^gamma

    float4 a = accum[p];
    a.xyz += n * (weight * freq_sq);
    a.w += weight;                            // denominator: no freq factor
    accum[p] = a;
}

// Final aggregation: S-bar = num / (den + MIN_ENERGY). The source's alpha is carried into .w:
// with --masks it is the keep mask, which the structure-aware score reads (.xyz never depends on it).
[shader("compute")]
[numthreads(16, 16, 1)]
void finalize(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> accum,
    RWTexture2D<float4> source,
    RWTexture2D<float4> fieldOut,
    uniform uint2 dim,
    uniform float min_energy)
{
    uint2 p = dtid.xy;
    if (p.x >= dim.x || p.y >= dim.y) return;
    float4 a = accum[p];
    fieldOut[p] = float4(a.xyz / (a.w + min_energy), source[p].a);
}
