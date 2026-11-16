"""CadQuery and its OpenCascade kernel install and produce correct geometry."""

import cadquery as cq
import pytest


def test_box_volume():
    box = cq.Workplane("XY").box(2.0, 3.0, 4.0)
    assert box.val().Volume() == pytest.approx(24.0)


def test_cut_reduces_volume():
    plate = cq.Workplane("XY").box(10.0, 10.0, 1.0)
    holed = plate.faces(">Z").workplane().hole(2.0)
    assert holed.val().isValid()
    assert holed.val().Volume() < plate.val().Volume()


def test_step_round_trip(tmp_path):
    path = tmp_path / "box.step"
    cq.exporters.export(cq.Workplane("XY").box(1.0, 2.0, 3.0), str(path))
    loaded = cq.importers.importStep(str(path))
    assert loaded.val().Volume() == pytest.approx(6.0)
