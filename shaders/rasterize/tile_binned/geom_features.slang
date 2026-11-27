// Copyright (c) 2026 Debashish Sahu. BSD 3-Clause License, as the rest of this repository (see LICENSE).

// Per-splat geometry features for the depth and normal losses (common/geom_priors.slang), one
// thread per gaussian, after bin_count: geom_data[gid] = (n, s) in the camera frame.

#include "../../common/types.slang"
#include "../../common/geom_priors.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void geomFeatures(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    StructuredBuffer<uint> cluster_visible,
    RWStructuredBuffer<float4> geom_data,
    uniform uint plane_depth,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    if (gid >= min(num_gaussians_counter[0], max_splats)) return;
    // Culled splats emit no tile pair, so the forward never reads their slot.
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;
    Gaussian3D g = gaussians[gid];
    float4x4 world_to_view = camera[0].view_matrix;
    uint k = geomMinAxis(g.log_scale);
    float sgn = geomFacingSign(g.mean, g.rot_quat, world_to_view, k);
    geom_data[gid] = geomFeature(g.mean, g.rot_quat, world_to_view, k, sgn, plane_depth);
}
