"""Turn a solid into a point cloud: dots scattered evenly over its surface."""

# A point cloud is what a 3D scanner produces, and it is used to measure Chamfer distance.

from __future__ import annotations

import cadquery as cq
import numpy as np

from d2b.pipeline.mesh import make_mesh

# Number of points in a saved point cloud.
POINT_COUNT = 2048


# Scatters points over a solid's surface; returns the points and the way the surface faces there.
def sample_points(
    shape: cq.Shape, count: int = POINT_COUNT, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    points, triangles = make_mesh(shape)
    corners = points[triangles]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    # The cross product of two sides points out of a triangle; its length is twice the area.
    outward = np.cross(b - a, c - a)
    lengths = np.linalg.norm(outward, axis=1)
    # Slivers with no area are dropped, since they have no direction and get no points.
    keep = lengths > 0.0
    a, b, c, outward, lengths = a[keep], b[keep], c[keep], outward[keep], lengths[keep]
    areas = lengths / 2.0
    # Bigger triangles get more points, so the points are spread evenly by area.
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(areas), size=count, p=areas / areas.sum())
    # Pick a random spot inside each chosen triangle.
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    sampled = a[chosen] + u[:, None] * (b - a)[chosen] + v[:, None] * (c - a)[chosen]
    # Every point takes the direction of the triangle it landed on, scaled to length 1.
    normals = outward[chosen] / lengths[chosen, None]
    return sampled, normals
