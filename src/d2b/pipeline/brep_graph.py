"""Turn a solid into a B-Rep graph: one node per face, joined where two faces share an edge."""

# The graph keeps what a mesh loses: which faces are flat or round, and how they connect.

from __future__ import annotations

import cadquery as cq
import numpy as np
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

# The kinds of surface a face can be; a face's kind is stored as its position in this list.
SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "other"]

# The kinds of curve an edge can be; stored the same way.
CURVE_TYPES = ["line", "circle", "other"]


# Converts a geometry-engine type name such as "PLANE" to its position in a list of kinds.
def _kind(name: str, kinds: list[str]) -> int:
    name = name.lower()
    return kinds.index(name) if name in kinds else kinds.index("other")


# Builds the face-adjacency graph of a solid, as a dictionary of arrays.
def make_graph(shape: cq.Shape) -> dict[str, np.ndarray]:
    # Number the faces, so each one can be referred to by its position.
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape.wrapped, TopAbs_FACE, face_map)
    faces = [cq.Face(face_map.FindKey(i)) for i in range(1, face_map.Extent() + 1)]

    # For every edge, ask the geometry engine which faces it belongs to.
    owners = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape.wrapped, TopAbs_EDGE, TopAbs_FACE, owners)

    links, link_types, link_lengths = [], [], []
    for i in range(1, owners.Extent() + 1):
        touching = sorted({face_map.FindIndex(face) - 1 for face in owners.FindFromIndex(i)})
        # An edge on only one face is a seam, where a round face meets itself; it joins nothing.
        if len(touching) < 2:
            continue
        edge = cq.Edge(owners.FindKey(i))
        # An edge normally joins two faces; if it joins more, every pair is linked.
        for first in range(len(touching)):
            for second in range(first + 1, len(touching)):
                links.append((touching[first], touching[second]))
                link_types.append(_kind(edge.geomType(), CURVE_TYPES))
                link_lengths.append(edge.Length())

    return {
        # One row per face.
        "face_type": np.array([_kind(f.geomType(), SURFACE_TYPES) for f in faces], dtype=np.int8),
        "face_area": np.array([f.Area() for f in faces]),
        "face_centre": np.array([f.Center().toTuple() for f in faces]),
        "face_normal": np.array([f.normalAt().toTuple() for f in faces]),
        # One row per shared edge: the two faces it joins, its kind of curve, and its length.
        "links": np.array(links, dtype=np.int32).reshape(-1, 2),
        "link_type": np.array(link_types, dtype=np.int8),
        "link_length": np.array(link_lengths),
    }
