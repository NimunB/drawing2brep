"""The verifier scores identical solids as 1 and known overlaps exactly."""

import cadquery as cq
import pytest

from d2b.eval.geometry import compare, iou, load_step, to_shape


def box(x, y, z, offset=(0.0, 0.0, 0.0)):
    return to_shape(cq.Workplane("XY").box(x, y, z).translate(offset))


def test_identical_solids_score_one():
    assert iou(box(2, 3, 4), box(2, 3, 4)) == pytest.approx(1.0)


def test_half_overlap():
    # Two 2x2x2 cubes shifted by 1: intersection 4, union 12.
    assert iou(box(2, 2, 2), box(2, 2, 2, offset=(1, 0, 0))) == pytest.approx(1 / 3)


def test_disjoint_solids_score_zero():
    assert iou(box(1, 1, 1), box(1, 1, 1, offset=(5, 0, 0))) == 0.0


def test_wrong_dimension_fails_threshold():
    result = compare(cq.Workplane("XY").box(2, 2, 2.1), box(2, 2, 2))
    assert result.valid
    assert result.iou < 0.99
    assert result.volume_error == pytest.approx(0.05)


def test_step_is_loaded_in_centimetres(tmp_path):
    # CadQuery writes STEP in millimetres, so a 10 mm cube is a 1 cm cube.
    path = tmp_path / "cube.step"
    cq.exporters.export(cq.Workplane("XY").box(10, 10, 10), str(path))
    assert load_step(path).Volume() == pytest.approx(1.0)
