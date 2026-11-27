// Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

// Depth-prior, normal-prior and depth-normal consistency losses on the geometry composite
// aux_image = (N, S) (common/geom_priors.slang). Two full-screen passes:
//   geomResolve  per pixel: rendered depth D and normal nh -> geom_resolved = (nh, D), zeros where
//                invalid; counts of supervised pixels and the scale-shift sums, summed exactly
//                (integers and 64-bit fixed point) into geom_stats.
//   geomGrad     per pixel: dL/d(N, S) -> dL_dAux, by bwd_diff of the pixel's share of the loss.
// Every term is a mean over its supervised pixels, scaled by 3WH so that a weight means what it
// means beside a per-pixel-mean photometric L1 (this trainer sums its photometric loss over pixels).
//
// Rendered depth: center D = S / A (A = 1 - T, a constant here); plane D = S / dot(N, r) (the ray
// through the pixel center meets the composited plane). Residuals are in metres: D_m = D / U with
// U = scene units per metre. Invalid pixels are skipped before any division, never multiplied by
// zero afterwards.

#include "../common/types.slang"
#include "../common/geom_priors.slang"
#include "../common/loss_mask.slang"

// geom_stats slots (uint2 = one 64-bit value each). Slots below GS_PER_STEP are cleared every step.
static const uint GS_VD = 0u;    // supervised depth pixels (count)
static const uint GS_VN = 1u;    // supervised normal pixels (count)
static const uint GS_VDN = 2u;   // supervised depth-normal pixels (count)
static const uint GS_SX = 3u;    // sums over depth pixels at SSI_MEAN_SCALE: x = 1 / prior depth,
static const uint GS_SY = 4u;    //   y = 1 / rendered depth (both 1/m); their means centre the moments below
static const uint GS_RX = 5u;    // scale-shift moments of the relative disparities (fixed point, ssi only):
static const uint GS_RY = 6u;    //   rx = x / mean(x) - 1, ry = y / mean(y) - 1: O(1) at any range, so
static const uint GS_RXX = 7u;   //   fixed point keeps the variance even at 100 m, where x^2 itself is
static const uint GS_RXY = 8u;   //   below the fixed-point step
static const uint GS_LD = 9u;    // sum of |depth residual| (fixed point, for logging)
static const uint GS_LN = 10u;   // sum of the normal error (L1 per pixel summed over xyz, or 1 - cos)
static const uint GS_LDN = 11u;  // sum of 1 - cos(depth normal, rendered normal)
static const uint GS_SSI_A = 12u; // the frame's fitted (a, b) as float bits in .x (logging and tests)
static const uint GS_SSI_B = 13u;
static const uint GS_PER_STEP = 14u;
static const uint GS_SKIPPED = 14u;      // frames whose scale-shift fit was degenerate (cumulative)
static const uint GS_DEPTH_FRAMES = 15u; // frames with the depth term evaluated (cumulative)
static const uint GEOM_STATS_SLOTS = 16u;

static const uint DEPTH_SPACE_DISPARITY = 0u;
static const uint DEPTH_SPACE_METRIC = 1u;
static const uint DEPTH_SPACE_SSI = 2u;

static const uint DEPTH_CODE_SATURATED = 65535u;   // >= 65.535 m: skipped as invalid

// Plane-depth validity (as PGSR-style renderers use): enough coverage, not edge-on, in range.
static const float PLANE_MIN_DOT = 0.05;
static const float PLANE_MIN_DEPTH = 1e-3;
static const float PLANE_MAX_DEPTH = 1e4;
static const float NORMAL_MIN_LEN2 = 1e-12;
static const float CROSS_MIN_LEN = 1e-12;
static const uint SSI_MIN_PIXELS = 16u;
static const float SSI_MEAN_SCALE = 16777216.0;   // 2^24: x up to 1000 (1 mm) over 24 Mpx stays in int64

int64_t statFixed(float value, float scale) { return int64_t(round(value * scale)); }

struct GeomFrame
{
    uint2 frame_dim;
    uint plane_depth;
    float min_alpha;
    float units_per_meter;
    uint has_depth;
    uint has_normal;
};

// Rendered depth (0 = invalid) and normal ((0,0,0) = invalid) of pixel q, from the composite.
void resolvePixel(uint2 q, RWTexture2D<float4> aux_image, RWTexture2D<float4> output_image,
                  CameraData cam, GeomFrame f, out float D, out float3 nh)
{
    float4 F = aux_image[q];
    float A = output_image[q].a;
    float3 r = geomPixelRay(float2(q) + 0.5, cam.proj_matrix, cam.image_size);
    D = 0.0;
    nh = float3(0.0);
    if (!(A >= f.min_alpha))
        return;
    float n2 = dot(F.xyz, F.xyz);
    if (n2 > NORMAL_MIN_LEN2)
        nh = F.xyz * rsqrt(n2);
    if (f.plane_depth != 0u)
    {
        float den = dot(F.xyz, r);
        if (abs(den) >= PLANE_MIN_DOT)
        {
            float d = F.w / den;
            if (d >= PLANE_MIN_DEPTH && d <= PLANE_MAX_DEPTH)
                D = d;
        }
    }
    else
    {
        float d = F.w / A;
        if (d > 0.0 && !isinf(d) && !isnan(d))
            D = d;
    }
}

float maskAt(uint2 q
#if LOSS_MASK
             , RWTexture2D<float4> loss_mask
#endif
             )
{
#if LOSS_MASK
    return loss_mask[q].a >= 0.5 ? 1.0 : 0.0;
#else
    return 1.0;
#endif
}

// Prior grid texel of a training pixel: nearest at the pixel center, in integers.
uint2 priorTexel(uint2 p, uint2 frame_dim, uint2 prior_dim)
{
    return uint2(((2u * p.x + 1u) * prior_dim.x) / (2u * frame_dim.x),
                 ((2u * p.y + 1u) * prior_dim.y) / (2u * frame_dim.y));
}

// Prior depth in metres at pixel p, 0 when invalid (code 0 or saturated).
float priorDepth(uint2 p, Texture2D<uint> depth_prior, GeomFrame f)
{
    if (f.has_depth == 0u) return 0.0;
    uint w, h;
    depth_prior.GetDimensions(w, h);
    uint code = depth_prior.Load(int3(int2(priorTexel(p, f.frame_dim, uint2(w, h))), 0));
    if (code == 0u || code >= DEPTH_CODE_SATURATED) return 0.0;
    return float(code) * 1e-3;
}

// Prior normal at pixel p, (0,0,0) when invalid.
float3 priorNormal(uint2 p, Texture2D<uint4> normal_prior, GeomFrame f)
{
    if (f.has_normal == 0u) return float3(0.0);
    uint w, h;
    normal_prior.GetDimensions(w, h);
    uint4 code = normal_prior.Load(int3(int2(priorTexel(p, f.frame_dim, uint2(w, h))), 0));
    float3 g = geomDecodeNormal(code.xyz);
    // A target must be (near) unit length: 0.7 <= |g| <= 1.3. Anything else is a sentinel or a misread
    // encoding (a black pixel of a spirula map read as slam decodes to (-1, -1, -1), length 1.73).
    float l2 = dot(g, g);
    return (l2 >= 0.49 && l2 <= 1.69) ? g : float3(0.0);
}

// Whether the depth-normal term centered at q is supervised, from the resolved values at q,
// q + (1, 0), q + (0, 1). Same predicate in both passes.
bool dnValid(float D0, float D1, float D2, float3 nh0, float m0, float m1, float m2,
             float3 r0, float3 r1, float3 r2)
{
    if (!(D0 > 0.0 && D1 > 0.0 && D2 > 0.0)) return false;
    if (dot(nh0, nh0) == 0.0) return false;
    if (m0 * m1 * m2 == 0.0) return false;
    float3 P0 = D0 * r0;
    float3 c = cross(D2 * r2 - P0, D1 * r1 - P0);
    return length(c) > CROSS_MIN_LEN;
}

// Depth-normal consistency at q: 1 - cos between the normal of the depth graph (forward
// differences, dP/dv x dP/du, which faces the camera for any positive depth graph) and nh.
[Differentiable]
float dnTerm(float D0, float D1, float D2, float3 nh, no_diff float3 r0, no_diff float3 r1,
             no_diff float3 r2)
{
    float3 P0 = D0 * r0;
    float3 c = cross(D2 * r2 - P0, D1 * r1 - P0);
    float3 nd = c * rsqrt(max(dot(c, c), 1e-24));
    return 1.0 - dot(nd, nh);
}

float3 normalFromDepth(float D0, float D1, float D2, float3 r0, float3 r1, float3 r2)
{
    float3 P0 = D0 * r0;
    float3 c = cross(D2 * r2 - P0, D1 * r1 - P0);
    return c * rsqrt(max(dot(c, c), 1e-24));
}

// Test entry: normalFromDepth over a depth image at the given sample coordinates (pixel + offset).
[shader("compute")]
[numthreads(16, 16, 1)]
void geomNormalFromDepthTest(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<float> depth,
    RWStructuredBuffer<float4> normal_out,
    uniform uint2 dim,
    uniform float4 intrinsics,   // fx, fy, cx, cy
    uniform float sample_offset)
{
    uint2 q = dtid.xy;
    if (q.x + 1u >= dim.x || q.y + 1u >= dim.y) return;
    float3 r[3];
    uint2 at[3] = { q, q + uint2(1u, 0u), q + uint2(0u, 1u) };
    float d[3];
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        float2 uv = float2(at[i]) + sample_offset;
        r[i] = float3((uv.x - intrinsics.z) / intrinsics.x, (uv.y - intrinsics.w) / intrinsics.y, 1.0);
        d[i] = depth[at[i].y * dim.x + at[i].x];
    }
    normal_out[q.y * dim.x + q.x] = float4(normalFromDepth(d[0], d[1], d[2], r[0], r[1], r[2]), 0.0);
}

struct GeomPixelCtx
{
    uint plane;
    float A;
    float U;
    float3 r0;
    bool need_D;
    // depth
    bool depth_on;
    uint depth_space;
    float kd;
    float Dg;
    float ssi_a;
    float ssi_b;
    // normal
    bool normal_on;
    uint normal_cos;
    float kn;
    float3 g;
    // depth-normal: the term at p (self), at p - (1,0) (left) and at p - (0,1) (up)
    float kdn;
    bool dn_self;
    float D_u; float D_v; float3 r_u; float3 r_v;
    bool dn_left;
    float D_l; float D_lv; float3 nh_l; float3 r_l; float3 r_lv;
    bool dn_up;
    float D_t; float D_tu; float3 nh_t; float3 r_t; float3 r_tu;
};

[Differentiable]
float depthResidual(float D, no_diff GeomPixelCtx c)
{
    float Dm = D / c.U;
    if (c.depth_space == DEPTH_SPACE_METRIC)
        return Dm - c.Dg;
    if (c.depth_space == DEPTH_SPACE_SSI)
        return 1.0 / Dm - (c.ssi_a / c.Dg + c.ssi_b);
    return 1.0 / Dm - 1.0 / c.Dg;
}

[Differentiable]
float normalError(float3 nh, no_diff GeomPixelCtx c)
{
    if (c.normal_cos != 0u)
        return 1.0 - dot(nh, c.g);
    return abs(nh.x - c.g.x) + abs(nh.y - c.g.y) + abs(nh.z - c.g.z);
}

// The part of the frame's geometry loss that depends on this pixel's composite F = (N, S).
[Differentiable]
float geomPixelLoss(float4 F, no_diff GeomPixelCtx c)
{
    float3 N = F.xyz;
    float3 nh = N * rsqrt(max(dot(N, N), 1e-24));
    float D = 0.0;
    if (c.need_D)
        D = (c.plane != 0u) ? F.w / dot(N, c.r0) : F.w / c.A;
    float L = 0.0;
    if (c.depth_on)
        L += c.kd * abs(depthResidual(D, c));
    if (c.normal_on)
        L += c.kn * normalError(nh, c);
    if (c.dn_self)
        L += c.kdn * dnTerm(D, c.D_u, c.D_v, nh, c.r0, c.r_u, c.r_v);
    if (c.dn_left)
        L += c.kdn * dnTerm(c.D_l, D, c.D_lv, c.nh_l, c.r_l, c.r0, c.r_lv);
    if (c.dn_up)
        L += c.kdn * dnTerm(c.D_t, c.D_tu, D, c.nh_t, c.r_t, c.r_tu, c.r0);
    return L;
}

// Exact group sums of per-thread int64 values, then one 64-bit add per slot into geom_stats.
static const uint GEOM_GROUP = 256u;
static const uint GEOM_REDUCE_SLOTS = 5u;
groupshared int64_t gs_reduce[GEOM_REDUCE_SLOTS][GEOM_GROUP];

void groupReduceAdd(uint tid, int64_t values[GEOM_REDUCE_SLOTS], uint first_slot, uint slot_count,
                    RWStructuredBuffer<uint2> geom_stats)
{
    for (uint k = 0u; k < GEOM_REDUCE_SLOTS; ++k)
        gs_reduce[k][tid] = values[k];
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = GEOM_GROUP / 2u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
        {
            for (uint k = 0u; k < GEOM_REDUCE_SLOTS; ++k)
                gs_reduce[k][tid] += gs_reduce[k][tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u)
    {
        for (uint k = 0u; k < slot_count; ++k)
        {
            int64_t v = gs_reduce[k][0];
            if (v != int64_t(0))
                geomAtomicAdd(geom_stats, first_slot + k, v);
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void geomClearStats(uint3 dtid: SV_DispatchThreadID, RWStructuredBuffer<uint2> geom_stats,
                    uniform uint clear_all)
{
    uint i = dtid.x + dtid.y * 16u;
    if (i < GEOM_STATS_SLOTS && (i < GS_PER_STEP || clear_all != 0u))
        geom_stats[i] = uint2(0u);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void geomResolve(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<float4> aux_image,
    RWTexture2D<float4> output_image,
    StructuredBuffer<CameraData> camera,
    Texture2D<uint> depth_prior,
    Texture2D<uint4> normal_prior,
#if LOSS_MASK
    RWTexture2D<float4> loss_mask,
#endif
    RWTexture2D<float4> geom_resolved,
    RWStructuredBuffer<uint2> geom_stats,
    uniform uint2 frame_dim,
    uniform uint plane_depth,
    uniform float min_alpha,
    uniform float units_per_meter,
    uniform uint has_depth,
    uniform uint has_normal,
    uniform uint dn_on)
{
    uint tid = gtid.y * 16u + gtid.x;
    uint2 p = dtid.xy;
    bool inside = p.x < frame_dim.x && p.y < frame_dim.y;
    GeomFrame f = { frame_dim, plane_depth, min_alpha, units_per_meter, has_depth, has_normal };
    CameraData cam = camera[0];

    int64_t values[GEOM_REDUCE_SLOTS];
    for (uint k = 0u; k < GEOM_REDUCE_SLOTS; ++k)
        values[k] = int64_t(0);

    if (inside)
    {
        float D; float3 nh;
        resolvePixel(p, aux_image, output_image, cam, f, D, nh);
        geom_resolved[p] = float4(nh, D);
#if LOSS_MASK
        float m = maskAt(p, loss_mask);
#else
        float m = maskAt(p);
#endif
        float Dg = priorDepth(p, depth_prior, f);
        if (Dg > 0.0 && m > 0.0 && D > 0.0)
        {
            float x = 1.0 / Dg;
            float y = units_per_meter / D;
            values[GS_VD] = int64_t(1);
            values[GS_SX] = statFixed(x, SSI_MEAN_SCALE);
            values[GS_SY] = statFixed(y, SSI_MEAN_SCALE);
        }
        float3 g = priorNormal(p, normal_prior, f);
        if (dot(g, g) > 0.0 && m > 0.0 && dot(nh, nh) > 0.0)
            values[GS_VN] = int64_t(1);
        if (dn_on != 0u && p.x + 1u < frame_dim.x && p.y + 1u < frame_dim.y)
        {
            uint2 pu = p + uint2(1u, 0u), pv = p + uint2(0u, 1u);
            float Du, Dv; float3 nu, nv;
            resolvePixel(pu, aux_image, output_image, cam, f, Du, nu);
            resolvePixel(pv, aux_image, output_image, cam, f, Dv, nv);
#if LOSS_MASK
            float mu = maskAt(pu, loss_mask), mv = maskAt(pv, loss_mask);
#else
            float mu = 1.0, mv = 1.0;
#endif
            float3 r0 = geomPixelRay(float2(p) + 0.5, cam.proj_matrix, cam.image_size);
            float3 ru = geomPixelRay(float2(pu) + 0.5, cam.proj_matrix, cam.image_size);
            float3 rv = geomPixelRay(float2(pv) + 0.5, cam.proj_matrix, cam.image_size);
            if (dnValid(D, Du, Dv, nh, m, mu, mv, r0, ru, rv))
                values[GS_VDN] = int64_t(1);
        }
    }
    groupReduceAdd(tid, values, 0u, GEOM_REDUCE_SLOTS, geom_stats);
}

float statMean(RWStructuredBuffer<uint2> geom_stats, uint slot, float count, float scale)
{
    return float(geomLoad(geom_stats[slot])) / (scale * max(count, 1.0));
}

// ssi only, between geomResolve and geomGrad: moments of the relative disparities over the depth pixels,
// centred on the means geomResolve summed.
[shader("compute")]
[numthreads(16, 16, 1)]
void geomSsiMoments(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<float4> geom_resolved,
    Texture2D<uint> depth_prior,
#if LOSS_MASK
    RWTexture2D<float4> loss_mask,
#endif
    RWStructuredBuffer<uint2> geom_stats,
    uniform uint2 frame_dim,
    uniform float units_per_meter,
    uniform uint has_depth)
{
    uint tid = gtid.y * 16u + gtid.x;
    uint2 p = dtid.xy;
    GeomFrame f = { frame_dim, 0u, 0.0, units_per_meter, has_depth, 0u };
    float vd = float(geomLoad(geom_stats[GS_VD]));
    float mx = statMean(geom_stats, GS_SX, vd, SSI_MEAN_SCALE);
    float my = statMean(geom_stats, GS_SY, vd, SSI_MEAN_SCALE);
    int64_t values[GEOM_REDUCE_SLOTS];
    for (uint k = 0u; k < GEOM_REDUCE_SLOTS; ++k)
        values[k] = int64_t(0);
    if (p.x < frame_dim.x && p.y < frame_dim.y && mx > 0.0 && my > 0.0)
    {
        float D = geom_resolved[p].w;
#if LOSS_MASK
        float m = maskAt(p, loss_mask);
#else
        float m = maskAt(p);
#endif
        float Dg = priorDepth(p, depth_prior, f);
        if (Dg > 0.0 && m > 0.0 && D > 0.0)
        {
            float rx = (1.0 / Dg) / mx - 1.0;
            float ry = (units_per_meter / D) / my - 1.0;
            values[0] = geomToFixed(rx);
            values[1] = geomToFixed(ry);
            values[2] = geomToFixed(rx * rx);
            values[3] = geomToFixed(rx * ry);
        }
    }
    groupReduceAdd(tid, values, GS_RX, 4u, geom_stats);
}

// Resolved values of pixel q (zeros out of frame), its mask and its ray.
void readResolved(int2 q, RWTexture2D<float4> geom_resolved, CameraData cam, uint2 frame_dim,
#if LOSS_MASK
                  RWTexture2D<float4> loss_mask,
#endif
                  out float D, out float3 nh, out float m, out float3 r)
{
    r = geomPixelRay(float2(q) + 0.5, cam.proj_matrix, cam.image_size);
    if (q.x < 0 || q.y < 0 || uint(q.x) >= frame_dim.x || uint(q.y) >= frame_dim.y)
    {
        D = 0.0; nh = float3(0.0); m = 0.0;
        return;
    }
    float4 v = geom_resolved[uint2(q)];
    D = v.w; nh = v.xyz;
#if LOSS_MASK
    m = maskAt(uint2(q), loss_mask);
#else
    m = 1.0;
#endif
}

float statValue(RWStructuredBuffer<uint2> geom_stats, uint slot)
{
    return float(geomLoad(geom_stats[slot])) * (1.0 / GEOM_FIXED_SCALE);
}

float statCount(RWStructuredBuffer<uint2> geom_stats, uint slot)
{
    return float(geomLoad(geom_stats[slot]));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void geomGrad(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<float4> aux_image,
    RWTexture2D<float4> output_image,
    StructuredBuffer<CameraData> camera,
    Texture2D<uint> depth_prior,
    Texture2D<uint4> normal_prior,
#if LOSS_MASK
    RWTexture2D<float4> loss_mask,
#endif
    RWTexture2D<float4> geom_resolved,
    RWStructuredBuffer<uint2> geom_stats,
    RWTexture2D<float4> dL_dAux,
    uniform uint2 frame_dim,
    uniform uint plane_depth,
    uniform float min_alpha,
    uniform float units_per_meter,
    uniform uint has_depth,
    uniform uint has_normal,
    uniform uint depth_space,
    uniform uint normal_cos,
    // Brush-equivalent weights (0 = term off).
    uniform float w_depth,
    uniform float w_normal,
    uniform float w_dn)
{
    uint tid = gtid.y * 16u + gtid.x;
    uint2 p = dtid.xy;
    bool inside = p.x < frame_dim.x && p.y < frame_dim.y;
    GeomFrame f = { frame_dim, plane_depth, min_alpha, units_per_meter, has_depth, has_normal };
    CameraData cam = camera[0];
    float scale = 3.0 * float(frame_dim.x) * float(frame_dim.y);

    float Vd = statCount(geom_stats, GS_VD);
    float Vn = statCount(geom_stats, GS_VN);
    float Vdn = statCount(geom_stats, GS_VDN);

    // Scale-shift fit of a / Dg + b to the rendered 1 / D, a constant for the gradient.
    bool depth_frame = has_depth != 0u && w_depth > 0.0 && Vd > 0.0;
    float ssi_a = 1.0, ssi_b = 0.0;
    if (depth_frame && depth_space == DEPTH_SPACE_SSI)
    {
        // y ~ a x + b with x, y in relative form: a = (my / mx) cov(rx, ry) / var(rx), b = my - a mx.
        float inv = 1.0 / Vd;
        float mx = statMean(geom_stats, GS_SX, Vd, SSI_MEAN_SCALE);
        float my = statMean(geom_stats, GS_SY, Vd, SSI_MEAN_SCALE);
        float mrx = statValue(geom_stats, GS_RX) * inv, mry = statValue(geom_stats, GS_RY) * inv;
        float var_r = statValue(geom_stats, GS_RXX) * inv - mrx * mrx;
        float cov_r = statValue(geom_stats, GS_RXY) * inv - mrx * mry;
        float a = (mx > 0.0 && var_r > 0.0) ? (my / mx) * cov_r / var_r : 0.0;
        // Degenerate (too few pixels, no spread) or inverted (a <= 0: prior and render disagree in order)
        // fits skip the frame's depth term; they are counted.
        if (Vd < float(SSI_MIN_PIXELS) || !(var_r > 1e-8) || !(a > 0.0) || isinf(a) || isnan(a))
            depth_frame = false;
        else
        {
            ssi_a = a;
            ssi_b = my - a * mx;
        }
    }
    if (dtid.x == 0u && dtid.y == 0u && has_depth != 0u && w_depth > 0.0)
    {
        if (depth_frame) geomAtomicAdd(geom_stats, GS_DEPTH_FRAMES, int64_t(1));
        else geomAtomicAdd(geom_stats, GS_SKIPPED, int64_t(1));
        geom_stats[GS_SSI_A] = uint2(asuint(ssi_a), 0u);
        geom_stats[GS_SSI_B] = uint2(asuint(ssi_b), 0u);
    }

    int64_t values[GEOM_REDUCE_SLOTS];
    for (uint k = 0u; k < GEOM_REDUCE_SLOTS; ++k)
        values[k] = int64_t(0);

    if (inside)
    {
        GeomPixelCtx c = {};
        c.plane = plane_depth;
        c.A = output_image[p].a;
        c.U = units_per_meter;
        float4 F = aux_image[p];

        float D, m; float3 nh, r0;
#if LOSS_MASK
        readResolved(int2(p), geom_resolved, cam, frame_dim, loss_mask, D, nh, m, r0);
#else
        readResolved(int2(p), geom_resolved, cam, frame_dim, D, nh, m, r0);
#endif
        c.r0 = r0;

        float Dg = priorDepth(p, depth_prior, f);
        c.depth_on = depth_frame && Dg > 0.0 && m > 0.0 && D > 0.0;
        c.depth_space = depth_space;
        c.kd = c.depth_on ? w_depth * scale / Vd : 0.0;
        c.Dg = c.depth_on ? Dg : 1.0;
        c.ssi_a = ssi_a;
        c.ssi_b = ssi_b;

        float3 g = priorNormal(p, normal_prior, f);
        c.normal_on = w_normal > 0.0 && Vn > 0.0 && dot(g, g) > 0.0 && m > 0.0 && dot(nh, nh) > 0.0;
        c.normal_cos = normal_cos;
        c.kn = c.normal_on ? w_normal * scale / (normal_cos != 0u ? Vn : 3.0 * Vn) : 0.0;
        c.g = g;

        bool dn = w_dn > 0.0 && Vdn > 0.0;
        c.kdn = dn ? w_dn * scale / Vdn : 0.0;
        if (dn)
        {
            float Du, Dv, Dl, Dlv, Dt, Dtu, mu, mv, ml, mlv, mt, mtu;
            float3 nu, nv, nl, nlv, nt, ntu, ru, rv, rl, rlv, rt, rtu;
            int2 q = int2(p);
#if LOSS_MASK
            readResolved(q + int2(1, 0), geom_resolved, cam, frame_dim, loss_mask, Du, nu, mu, ru);
            readResolved(q + int2(0, 1), geom_resolved, cam, frame_dim, loss_mask, Dv, nv, mv, rv);
            readResolved(q + int2(-1, 0), geom_resolved, cam, frame_dim, loss_mask, Dl, nl, ml, rl);
            readResolved(q + int2(-1, 1), geom_resolved, cam, frame_dim, loss_mask, Dlv, nlv, mlv, rlv);
            readResolved(q + int2(0, -1), geom_resolved, cam, frame_dim, loss_mask, Dt, nt, mt, rt);
            readResolved(q + int2(1, -1), geom_resolved, cam, frame_dim, loss_mask, Dtu, ntu, mtu, rtu);
#else
            readResolved(q + int2(1, 0), geom_resolved, cam, frame_dim, Du, nu, mu, ru);
            readResolved(q + int2(0, 1), geom_resolved, cam, frame_dim, Dv, nv, mv, rv);
            readResolved(q + int2(-1, 0), geom_resolved, cam, frame_dim, Dl, nl, ml, rl);
            readResolved(q + int2(-1, 1), geom_resolved, cam, frame_dim, Dlv, nlv, mlv, rlv);
            readResolved(q + int2(0, -1), geom_resolved, cam, frame_dim, Dt, nt, mt, rt);
            readResolved(q + int2(1, -1), geom_resolved, cam, frame_dim, Dtu, ntu, mtu, rtu);
#endif
            c.dn_self = dnValid(D, Du, Dv, nh, m, mu, mv, r0, ru, rv);
            c.D_u = Du; c.D_v = Dv; c.r_u = ru; c.r_v = rv;
            c.dn_left = dnValid(Dl, D, Dlv, nl, ml, m, mlv, rl, r0, rlv);
            c.D_l = Dl; c.D_lv = Dlv; c.nh_l = nl; c.r_l = rl; c.r_lv = rlv;
            c.dn_up = dnValid(Dt, Dtu, D, nt, mt, mtu, m, rt, rtu, r0);
            c.D_t = Dt; c.D_tu = Dtu; c.nh_t = nt; c.r_t = rt; c.r_tu = rtu;
            if (c.dn_self)
                values[GS_LDN - GS_LD] = geomToFixed(dnTerm(D, Du, Dv, nh, r0, ru, rv));
        }
        c.need_D = c.depth_on || c.dn_self || c.dn_left || c.dn_up;

        float4 grad = float4(0.0);
        if (c.need_D || c.normal_on)
        {
            var dpF = diffPair(F);
            bwd_diff(geomPixelLoss)(dpF, c, 1.0);
            grad = dpF.d;
            if (c.depth_on)
                values[GS_LD - GS_LD] = geomToFixed(abs(depthResidual(D, c)));
            if (c.normal_on)
                values[GS_LN - GS_LD] = geomToFixed(normalError(nh, c));
        }
        dL_dAux[p] = grad;
    }
    groupReduceAdd(tid, values, GS_LD, 3u, geom_stats);
}
