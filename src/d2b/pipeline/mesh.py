"""Turn a solid into a mesh: its surface as a set of small flat triangles."""

# A mesh loses the exact surfaces a solid has, but it is what most 3D tools work with.

from __future__ import annotations

import cadquery as cq
import numpy as np

# Triangles are kept no further from the true surface than this fraction of the part's size.
MESH_TOLERANCE = 1e-3


# Splits a solid's surface into triangles; returns the corner points and the triangles.
def make_mesh(shape: cq.Shape) -> tuple[np.ndarray, np.ndarray]:
    # A smaller tolerance follows curved surfaces more closely, with more triangles.
    tolerance = shape.BoundingBox().DiagonalLength * MESH_TOLERANCE
    points, triangles = shape.tessellate(tolerance)
    # Each triangle is three numbers: the positions of its corners in the list of points.
    return np.array([p.toTuple() for p in points]), np.array(triangles)


# Works out the volume enclosed by a mesh, to check that it matches the solid.
def mesh_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    a, b, c = (points[triangles[:, i]] for i in range(3))
    # Each triangle and the origin form a small pyramid; their signed volumes add up.
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
