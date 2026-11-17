"""The sandbox runs good programs, and stops or refuses bad ones without crashing."""

# Each test hands the sandbox a small program as text, the way a model would.

import pytest

from d2b.eval.sandbox import check_program, run_untrusted

GOOD_PROGRAM = """
import cadquery as cq

result = cq.Workplane("XY").box(2, 3, 4)
"""


# A normal program runs and gives back the solid it built.
def test_good_program_returns_its_solid():
    outcome = run_untrusted(GOOD_PROGRAM)
    assert outcome.status == "ok"
    assert outcome.shape.Volume() == pytest.approx(24.0)


# A program that never finishes is stopped at the time limit.
def test_endless_program_is_stopped():
    outcome = run_untrusted("while True:\n    pass\n", timeout=5.0)
    assert outcome.status == "timeout"
    assert outcome.shape is None


# A program that crashes is reported as an error, with the reason.
def test_crashing_program_is_reported():
    outcome = run_untrusted("import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1 / 0)\n")
    assert outcome.status == "error"
    assert "ZeroDivisionError" in outcome.detail


# A program that never defines `result` is reported, not treated as a success.
def test_program_without_result_is_reported():
    outcome = run_untrusted("import cadquery as cq\npart = cq.Workplane('XY').box(1, 1, 1)\n")
    assert outcome.status == "no_result"


# A program with a typo that is not valid Python is refused before it runs.
def test_invalid_python_is_rejected():
    outcome = run_untrusted("result = cq.Workplane('XY'.box(1, 1, 1\n")
    assert outcome.status == "rejected"
    assert "syntax error" in outcome.detail


# Programs that try to reach outside (other modules, files, hidden names) are refused.
@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = None\n",
        "import subprocess\n",
        "from pathlib import Path\n",
        "data = open('/etc/passwd').read()\n",
        "exec('import os')\n",
        "x = ().__class__\n",
        "import cadquery as cq\ncq.exporters.export(None, 'out.step')\n",
    ],
)
def test_reaching_outside_is_rejected(code):
    assert check_program(code) is not None
    assert run_untrusted(code).status == "rejected"


# The kind of program our converter writes passes the check untouched.
def test_converter_style_program_is_allowed():
    code = """
import cadquery as cq

# Step 1: new body
plane_1 = cq.Plane(origin=(0.0, 0.0, 0.0), xDir=(1.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
part_1 = (
    cq.Workplane(plane_1)
    .polyline([(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]).close()
    .moveTo(2.0, 1.0).circle(0.5)
    .extrude(1.0)
)
result = part_1
"""
    assert check_program(code) is None
    assert run_untrusted(code).status == "ok"
