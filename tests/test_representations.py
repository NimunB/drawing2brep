"""The mesh, point cloud and B-Rep graph of simple solids come out as expected."""

# Boxes and cylinders are used because their correct answers can be worked out by hand.

import math

import cadquery as cq
import numpy as np
import pytest

from d2b.eval.geometry import to_shape
from d2b.pipeline.brep_graph import CURVE_TYPES, SURFACE_TYPES, make_graph
from d2b.pipeline.mesh import make_mesh, mesh_volume
from d2b.pipeline.pointcloud import sample_points


# Helper: a box 2 wide, 3 deep and 4 tall, centred on the origin.
def box():
    return to_shape(cq.Workplane("XY").box(2, 3, 4))


# Helper: a cylinder of radius 1 and height 2, standing on the origin.
def cylinder():
    return to_shape(cq.Workplane("XY").circle(1).extrude(2))


# Helper: how many other faces each face is joined to.
def neighbour_counts(graph):
    return np.bincount(graph["links"].ravel(), minlength=len(graph["face_type"]))


# A box's flat faces are followed exactly, so its mesh encloses exactly the box's volume.
def test_mesh_of_a_box_has_the_right_volume():
    points, triangles = make_mesh(box())
    # A box needs only its 8 corners, and two triangles for each of its 6 faces.
    assert len(points) >= 8
    assert len(triangles) == 12
    assert mesh_volume(points, triangles) == pytest.approx(24.0)


# A cylinder's round side is approximated, so the mesh volume is close but slightly under.
def test_mesh_of_a_cylinder_is_close_to_the_true_volume():
    points, triangles = make_mesh(cylinder())
    true_volume = math.pi * 1**2 * 2
    assert mesh_volume(points, triangles) == pytest.approx(true_volume, rel=0.01)
    assert mesh_volume(points, triangles) < true_volume


# Every point of the cloud lies on the surface of the box.
def test_point_cloud_lies_on_the_surface():
    points, _ = sample_points(box(), count=500)
    assert points.shape == (500, 3)
    # On this box, a surface point is at x = 1, y = 1.5 or z = 2 (or the negative of one).
    on_a_face = np.isclose(np.abs(points), [1.0, 1.5, 2.0]).any(axis=1)
    assert on_a_face.all()


# Each point's direction has length 1 and points away from the inside of the box.
def test_point_cloud_directions_point_outward():
    points, normals = sample_points(box(), count=500)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # The box is centred on the origin, so "outward" means away from the origin.
    assert (np.einsum("ij,ij->i", points, normals) > 0).all()


# The same seed gives the same points, so results can be repeated.
def test_point_cloud_is_repeatable():
    assert np.array_equal(sample_points(box(), seed=3)[0], sample_points(box(), seed=3)[0])


# A box has 6 flat faces joined by 12 straight edges, and each face touches 4 others.
def test_graph_of_a_box():
    graph = make_graph(box())
    assert len(graph["face_type"]) == 6
    assert (graph["face_type"] == SURFACE_TYPES.index("plane")).all()
    assert len(graph["links"]) == 12
    assert (graph["link_type"] == CURVE_TYPES.index("line")).all()
    assert (neighbour_counts(graph) == 4).all()


# The face areas in the graph add up to the solid's whole surface area.
def test_graph_face_areas_add_up():
    graph = make_graph(box())
    # Two faces each of 2x3, 2x4 and 3x4.
    assert graph["face_area"].sum() == pytest.approx(2 * (6 + 8 + 12))


# A cylinder has one round face and two flat ends, joined by two circles.
def test_graph_of_a_cylinder():
    graph = make_graph(cylinder())
    kinds = sorted(SURFACE_TYPES[k] for k in graph["face_type"])
    assert kinds == ["cylinder", "plane", "plane"]
    # The seam where the round face meets itself joins nothing, so it is not a link.
    assert len(graph["links"]) == 2
    assert (graph["link_type"] == CURVE_TYPES.index("circle")).all()
    assert graph["link_length"] == pytest.approx([2 * math.pi, 2 * math.pi])


# Drilling a hole through a plate adds one round face, joined to the top and the bottom.
def test_graph_gains_a_face_for_a_hole():
    plate = cq.Workplane("XY").box(4, 4, 1)
    holed = to_shape(plate.faces(">Z").workplane().hole(1.0))
    graph = make_graph(holed)
    assert len(graph["face_type"]) == 7
    assert (graph["face_type"] == SURFACE_TYPES.index("cylinder")).sum() == 1
    # 12 edges of the plate, plus the two circles where the hole meets the top and bottom.
    assert len(graph["links"]) == 14
