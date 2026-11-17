"""The verifier scores identical solids as 1 and known overlaps exactly."""

import cadquery as cq
import numpy as np
import pytest

from d2b.eval.geometry import (
    chamfer_distance,
    compare,
    evaluate,
    iou,
    load_step,
    sample_surface,
    size_error,
    to_shape,
)


# Helper: builds a box of the given size, optionally moved away from the origin.
def box(x, y, z, offset=(0.0, 0.0, 0.0)):
    return to_shape(cq.Workplane("XY").box(x, y, z).translate(offset))


# Two identical boxes must score a perfect 1.
def test_identical_solids_score_one():
    assert iou(box(2, 3, 4), box(2, 3, 4)) == pytest.approx(1.0)


# Two half-overlapping cubes must score exactly 1/3, an answer we can work out by hand.
def test_half_overlap():
    # Two 2x2x2 cubes shifted by 1: they share 4, and together cover 8 + 8 - 4 = 12.
    assert iou(box(2, 2, 2), box(2, 2, 2, offset=(1, 0, 0))) == pytest.approx(1 / 3)


# Two boxes that do not touch must score 0.
def test_disjoint_solids_score_zero():
    # The second cube is moved 5 away, so there is a gap between them.
    assert iou(box(1, 1, 1), box(1, 1, 1, offset=(5, 0, 0))) == 0.0


# A part that is slightly the wrong size must fail the 0.99 pass mark.
def test_wrong_dimension_fails_threshold():
    # Candidate is 2.1 tall; the true part is 2.0 tall (5% too tall).
    result = compare(cq.Workplane("XY").box(2, 2, 2.1), box(2, 2, 2))
    # It is still a proper solid...
    assert result.valid
    # ...but it does not match closely enough to be kept.
    assert result.iou < 0.99
    # And the volume is off by exactly 5%.
    assert result.volume_error == pytest.approx(0.05)


# A loaded STEP file must come back in centimetres, not millimetres.
def test_step_is_loaded_in_centimetres(tmp_path):
    # CadQuery writes STEP in millimetres, so a 10 mm cube is a 1 cm cube.
    path = tmp_path / "cube.step"
    cq.exporters.export(cq.Workplane("XY").box(10, 10, 10), str(path))
    # A 1 cm cube has volume 1; without the unit fix this would be 1,000.
    assert load_step(path).Volume() == pytest.approx(1.0)


# The point cloud has the requested number of points, and they all lie on the surface.
def test_point_cloud_lies_on_the_surface():
    points = sample_surface(box(2, 2, 2), count=500)
    assert points.shape == (500, 3)
    # On a cube from -1 to 1, every surface point has at least one coordinate at -1 or 1.
    assert np.allclose(np.abs(points).max(axis=1), 1.0)


# Two identical solids have surfaces that coincide, so their Chamfer distance is zero.
def test_chamfer_is_zero_for_identical_solids():
    assert chamfer_distance(box(2, 3, 4), box(2, 3, 4)) == pytest.approx(0.0, abs=1e-9)


# Moving a solid further away makes the Chamfer distance larger.
def test_chamfer_grows_with_distance():
    near = chamfer_distance(box(2, 2, 2), box(2, 2, 2, offset=(0.1, 0, 0)))
    far = chamfer_distance(box(2, 2, 2), box(2, 2, 2, offset=(1.0, 0, 0)))
    assert 0.0 < near < far


# The size error reports the worst of the three overall dimensions.
def test_size_error_reports_the_worst_dimension():
    # Height is 2.2 against 2.0 (10% off); width and depth are exact.
    assert size_error(box(2, 2, 2.2), box(2, 2, 2)) == pytest.approx(0.1)


# A solid with a hole has more faces than a plain one, which flags a different structure.
def test_missing_hole_changes_the_structure():
    plain = cq.Workplane("XY").box(4, 4, 1)
    holed = to_shape(cq.Workplane("XY").box(4, 4, 1).faces(">Z").workplane().hole(1.0))
    result = evaluate(plain, holed)
    # The overlap is still high, so IoU alone would not reveal the missing hole.
    assert result.iou > 0.9
    assert result.face_count == 6
    assert result.face_count_ref == 7
    assert not result.same_structure


# A perfect candidate scores perfectly on every measurement.
def test_evaluate_scores_a_perfect_match():
    result = evaluate(cq.Workplane("XY").box(2, 3, 4), box(2, 3, 4))
    assert result.valid
    assert result.iou == pytest.approx(1.0)
    assert result.chamfer == pytest.approx(0.0, abs=1e-9)
    assert result.size_error == pytest.approx(0.0, abs=1e-6)
    assert result.same_structure


# Helper: a 2 x 2 x 1 block made by joining two halves without merging the seam between them.
def block_in_two_halves():
    left = cq.Workplane("XY").box(1, 2, 1).translate((-0.5, 0, 0))
    right = cq.Workplane("XY").box(1, 2, 1).translate((0.5, 0, 0))
    return left.union(right, clean=False)


# Face counts are compared with the structure reference when one is given.
def test_structure_reference_is_used_for_counts():
    # The same block built in one piece has 6 faces; built in two halves it has more.
    one_piece = box(2, 2, 1)
    # Same shape, so the overlap is perfect, but the counts differ against the one-piece solid.
    against_truth = evaluate(block_in_two_halves(), one_piece)
    assert against_truth.iou == pytest.approx(1.0, abs=1e-3)
    assert against_truth.face_count > against_truth.face_count_ref
    assert not against_truth.same_structure
    # Compared with a solid built the same way, the structure matches.
    against_label = evaluate(
        block_in_two_halves(), one_piece, structure_reference=block_in_two_halves()
    )
    assert against_label.same_structure


# A correct part built in the wrong place scores low as placed, but perfectly once aligned.
def test_aligned_iou_ignores_position():
    result = evaluate(box(2, 2, 2, offset=(5, 0, 0)), box(2, 2, 2))
    assert result.iou == 0.0
    assert result.iou_aligned == pytest.approx(1.0)
    assert result.chamfer == pytest.approx(0.0, abs=1e-9)


# Aligning does not hide a wrong size.
def test_aligned_iou_still_catches_wrong_size():
    result = evaluate(box(2, 2, 3, offset=(5, 0, 0)), box(2, 2, 2))
    assert result.iou_aligned == pytest.approx(2 / 3)
