"""The drawing code projects solids correctly and produces a usable image."""

# These tests use simple boxes and cylinders, so they run without the dataset.

import cadquery as cq
import numpy as np
import pytest

from d2b.eval.geometry import to_shape
from d2b.pipeline.drawing import IMAGE_SIZE, VIEWS, format_dimension, project, render_drawing


# Helper: a box that is 4 wide (x), 2 deep (y) and 1 tall (z).
def box():
    return to_shape(cq.Workplane("XY").box(4, 2, 1))


# Helper: the same box with a hole drilled halfway down from the top face.
def box_with_blind_hole():
    return to_shape(cq.Workplane("XY").box(4, 2, 1).faces(">Z").workplane().hole(0.5, 0.5))


# Helper: projects a solid in one named view.
def view_of(shape, name):
    eye, right = VIEWS[name]
    return project(shape, eye, right, tolerance=1e-3)


# Helper: the width and height covered by a view's visible edges.
def extent(view):
    points = np.vstack(view.visible)
    return tuple(points.max(axis=0) - points.min(axis=0))


# Each view shows the two dimensions it should: front is width by height, and so on.
def test_views_show_the_right_dimensions():
    assert extent(view_of(box(), "front")) == pytest.approx((4, 1))
    assert extent(view_of(box(), "top")) == pytest.approx((4, 2))
    assert extent(view_of(box(), "right")) == pytest.approx((2, 1))


# Helper: True for each hidden point that lies inside the box's outline, not on its border.
def hidden_points_inside_outline(view):
    if not view.hidden:
        return np.array([], dtype=bool)
    points = np.vstack(view.hidden)
    # In the front view the box spans -2..2 across and -0.5..0.5 up.
    return (np.abs(points[:, 0]) < 1.99) & (np.abs(points[:, 1]) < 0.49)


# A plain box hides only its back edges, which sit exactly behind the front ones.
def test_plain_box_hides_nothing_inside_its_outline():
    assert not hidden_points_inside_outline(view_of(box(), "front")).any()


# A hole that cannot be seen from the front shows up there as hidden (dashed) edges.
def test_hole_is_hidden_from_the_front():
    assert hidden_points_inside_outline(view_of(box_with_blind_hole(), "front")).any()


# The same hole is plainly visible from above, as a circle of many short pieces.
def test_hole_is_visible_from_the_top():
    plain = sum(len(line) for line in view_of(box(), "top").visible)
    holed = sum(len(line) for line in view_of(box_with_blind_hole(), "top").visible)
    assert holed > plain


# The finished image has the expected size and actually contains dark lines.
def test_drawing_is_the_right_size_and_not_blank():
    image = render_drawing(box())
    assert image.size == (IMAGE_SIZE, IMAGE_SIZE)
    pixels = np.asarray(image.convert("L"))
    # Mostly white paper, with some dark ink on it.
    assert (pixels < 128).mean() > 0.001
    assert (pixels > 250).mean() > 0.8


# Drawing the same part twice gives exactly the same image.
def test_drawing_is_repeatable():
    first = np.asarray(render_drawing(box_with_blind_hole()))
    second = np.asarray(render_drawing(box_with_blind_hole()))
    assert (first == second).all()


# Dimensions are written with 4 significant figures and no padding zeros.
def test_dimension_format():
    assert format_dimension(0.55142) == "0.5514"
    assert format_dimension(12.5) == "12.5"
    assert format_dimension(25.0) == "25"


# A flat shape with no thickness cannot be drawn as a part, and is rejected.
def test_flat_shape_is_rejected():
    flat = cq.Workplane("XY").rect(2, 2).val()
    with pytest.raises(ValueError, match="no thickness"):
        render_drawing(flat)
