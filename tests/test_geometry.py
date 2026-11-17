""The verifier scores identical solids as 1 and known overlaps exactly."""

import cadquery as cq
import pytest

from d2b.eval.geometry import compare, iou, load_step, to_shape


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
