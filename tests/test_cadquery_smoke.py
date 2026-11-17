"""CadQuery and its OpenCascade kernel install and produce correct geometry."""

# If these fail, the problem is the installation on this machine, not our own code.

import cadquery as cq
import pytest


# CadQuery can build a box and report its volume correctly.
def test_box_volume():
    box = cq.Workplane("XY").box(2.0, 3.0, 4.0)
    # 2 x 3 x 4 = 24.
    assert box.val().Volume() == pytest.approx(24.0)


# Cutting a hole removes material, so the volume must go down.
def test_cut_reduces_volume():
    plate = cq.Workplane("XY").box(10.0, 10.0, 1.0)
    # Drill a hole of diameter 2 through the top face.
    holed = plate.faces(">Z").workplane().hole(2.0)
    # The result is still a proper solid.
    assert holed.val().isValid()
    assert holed.val().Volume() < plate.val().Volume()


# A solid saved to a STEP file and loaded back must be unchanged.
def test_step_round_trip(tmp_path):
    # tmp_path is a throwaway folder that pytest creates and deletes for us.
    path = tmp_path / "box.step"
    cq.exporters.export(cq.Workplane("XY").box(1.0, 2.0, 3.0), str(path))
    loaded = cq.importers.importStep(str(path))
    # 1 x 2 x 3 = 6.
    assert loaded.val().Volume() == pytest.approx(6.0)
