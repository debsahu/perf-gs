// Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

// Per-gaussian backward of the geometry features, after projection_backward and before the
// regularization pass. Under GEOM_PRIORS it reads (and resets) the tile-summed seeds
// dL/d(n, s) the blend backward wrote and chains them to the mean and the rotation through
// geomFeature. With a flatten weight it adds the min-axis scale term. It is the only writer of
// these gradients while it runs, so plain read-modify-write is safe (as in projection_backward).

#include "../../common/types.slang"
#include "../../common/geom_priors.slang"

#include "common/execution.slang"

// No flatten gradient once the smallest axis is below this many scene units per metre: the Mip
// compensation sqrt(prod s^2 / prod(s^2 + f^2)) underflows in float32 as one axis goes to zero.
static const float FLATTEN_FLOOR_M = 1e-5;

[shader("compute")]
[numthreads(256, 1, 1)]
void geomBackward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    StructuredBuffer<uint> cluster_visible,
#if GEOM_PRIORS
    RWStructuredBuffer<uint2> grad_geom_nx,
    RWStructuredBuffer<uint2> grad_geom_ny,
    RWStructuredBuffer<uint2> grad_geom_nz,
    RWStructuredBuffer<uint2> grad_geom_s,
    RWStructuredBuffer<int64_t> grad_mean_x,
    RWStructuredBuffer<int64_t> grad_mean_y,
    RWStructuredBuffer<int64_t> grad_mean_z,
    RWStructuredBuffer<int64_t> grad_rot_w,
    RWStructuredBuffer<int64_t> grad_rot_x,
    RWStructuredBuffer<int64_t> grad_rot_y,
    RWStructuredBuffer<int64_t> grad_rot_z,
    uniform uint plane_depth,
#endif
    RWStructuredBuffer<int64_t> grad_log_scale_x,
    RWStructuredBuffer<int64_t> grad_log_scale_y,
    RWStructuredBuffer<int64_t> grad_log_scale_z,
    // Flatten: w * 3WH / U; the shader divides by the live splat count (a mean over splats, as the
    // regularization pass does). 0 disables it. floor_units = FLATTEN_FLOOR_M * U.
    uniform float flatten_coef,
    uniform float flatten_floor_units,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    uint live = min(num_gaussians_counter[0], max_splats);
    if (gid >= live) return;
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    Gaussian3D g = gaussians[gid];
    uint k = geomMinAxis(g.log_scale);

#if GEOM_PRIORS
    int64_t seed_nx = geomLoad(grad_geom_nx[gid]), seed_ny = geomLoad(grad_geom_ny[gid]);
    int64_t seed_nz = geomLoad(grad_geom_nz[gid]), seed_s = geomLoad(grad_geom_s[gid]);
    if ((seed_nx | seed_ny | seed_nz | seed_s) != int64_t(0))
    {
        grad_geom_nx[gid] = uint2(0u); grad_geom_ny[gid] = uint2(0u);
        grad_geom_nz[gid] = uint2(0u); grad_geom_s[gid] = uint2(0u);
        float4 dL_df = float4(geomFromFixed(seed_nx), geomFromFixed(seed_ny),
                              geomFromFixed(seed_nz), geomFromFixed(seed_s));
        float4x4 world_to_view = camera[0].view_matrix;
        float sgn = geomFacingSign(g.mean, g.rot_quat, world_to_view, k);
        var dp_mean = diffPair(g.mean);
        var dp_quat = diffPair(g.rot_quat);
        bwd_diff(geomFeature)(dp_mean, dp_quat, world_to_view, k, sgn, plane_depth, dL_df);
        grad_mean_x[gid] = grad_mean_x[gid] + geomToFixed(dp_mean.d.x);
        grad_mean_y[gid] = grad_mean_y[gid] + geomToFixed(dp_mean.d.y);
        grad_mean_z[gid] = grad_mean_z[gid] + geomToFixed(dp_mean.d.z);
        // rot_quat is (w, x, y, z), stored in .xyzw.
        grad_rot_w[gid] = grad_rot_w[gid] + geomToFixed(dp_quat.d.x);
        grad_rot_x[gid] = grad_rot_x[gid] + geomToFixed(dp_quat.d.y);
        grad_rot_y[gid] = grad_rot_y[gid] + geomToFixed(dp_quat.d.z);
        grad_rot_z[gid] = grad_rot_z[gid] + geomToFixed(dp_quat.d.w);
    }
#endif

    if (flatten_coef > 0.0)
    {
        float lsk = (k == 0u) ? g.log_scale.x : ((k == 1u) ? g.log_scale.y : g.log_scale.z);
        float sk = exp(lsk);
        if (sk >= flatten_floor_units)
        {
            // d/d(log s) of s is s.
            int64_t d = geomToFixed(flatten_coef / float(max(live, 1u)) * sk);
            if (k == 0u)      grad_log_scale_x[gid] = grad_log_scale_x[gid] + d;
            else if (k == 1u) grad_log_scale_y[gid] = grad_log_scale_y[gid] + d;
            else              grad_log_scale_z[gid] = grad_log_scale_z[gid] + d;
        }
    }
}
