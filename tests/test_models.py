"""The prompt helpers, the cost sum and the grading of a reply all behave as expected."""

# No test here calls a model, so none of them costs anything.

import json

import cadquery as cq
import pytest

from d2b.eval.run import grade_reply, score_group
from d2b.models.baselines import choose_pairs, cost_in_dollars
from d2b.models.prompt import extract_program

BOX_PROGRAM = 'import cadquery as cq\n\nresult = cq.Workplane("XY").box(40, 20, 10)\n'


# Helper: wraps a program in the fenced code block a model is asked to reply with.
def as_reply(program):
    return f"```python\n{program}```"


# Helper: writes a 4 x 2 x 1 cm box as a part's true solid, and returns the dataset folder.
def write_truth(tmp_path, name="box"):
    (tmp_path / "reconstruction").mkdir()
    # STEP files are in millimetres, so 40 x 20 x 10 mm is 4 x 2 x 1 cm.
    solid = cq.Workplane("XY").box(40, 20, 10)
    cq.exporters.export(solid, str(tmp_path / "reconstruction" / f"{name}.step"))
    return tmp_path


# The program is taken from inside the code block, without the surrounding text.
def test_extract_program_from_a_code_block():
    reply = "Here is the program:\n```python\nimport cadquery as cq\nresult = 1\n```\nDone."
    assert extract_program(reply) == "import cadquery as cq\nresult = 1\n"


# When a reply has several code blocks, the last one is used.
def test_extract_program_takes_the_last_block():
    reply = "```python\nfirst = 1\n```\nActually:\n```python\nsecond = 2\n```"
    assert extract_program(reply) == "second = 2\n"


# A reply with no code block gives nothing, which is graded as a failure.
def test_extract_program_without_a_block():
    assert extract_program("I cannot read this drawing.") is None


# Cost is tokens times price per million: 1M input at $4 plus 0.5M output at $20.
def test_cost_in_dollars():
    assert cost_in_dollars("claude-opus-5-5", 1_000_000, 500_000) == pytest.approx(14.0)


# A run on 5 parts uses the same first 5 as a run on 10, so runs can be extended.
def test_smaller_subset_is_a_prefix_of_a_larger_one(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps({"name": f"part{i}"}) + "\n" for i in range(50)))
    assert choose_pairs(pairs, 5) == choose_pairs(pairs, 10)[:5]


# A correct program in the right size is graded as a perfect match.
def test_grade_reply_scores_a_correct_program(tmp_path):
    root = write_truth(tmp_path)
    program = 'import cadquery as cq\n\nresult = cq.Workplane("XY").box(4, 2, 1)\n'
    record = grade_reply(as_reply(program), "box", program, root)
    assert record["status"] == "ok"
    assert record["iou_aligned"] == pytest.approx(1.0)
    assert record["same_structure"]


# A program in the wrong units builds a part 10 times too big, and scores badly.
def test_grade_reply_scores_a_wrong_size_low(tmp_path):
    root = write_truth(tmp_path)
    label = 'import cadquery as cq\n\nresult = cq.Workplane("XY").box(4, 2, 1)\n'
    record = grade_reply(as_reply(BOX_PROGRAM), "box", label, root)
    assert record["status"] == "ok"
    assert record["iou_aligned"] < 0.01
    assert record["size_error"] == pytest.approx(9.0)


# A reply with no program, or with a forbidden program, is recorded as a failure.
def test_grade_reply_records_failures(tmp_path):
    root = write_truth(tmp_path)
    assert grade_reply("no code here", "box", BOX_PROGRAM, root)["status"] == "no_program"
    assert grade_reply(as_reply("import os\n"), "box", BOX_PROGRAM, root)["status"] == "rejected"


# In a group's score, a reply that gave no valid solid counts as an overlap of 0.
def test_score_group_counts_failures_as_zero():
    records = [
        {"iou_aligned": 1.0},
        {"iou_aligned": 0.95},
        {"status": "error"},
        {"iou_aligned": 0.5},
    ]
    count, mean, close = score_group(records)
    assert count == 4
    assert mean == pytest.approx((1.0 + 0.95 + 0.0 + 0.5) / 4)
    # Two of the four are above 0.9.
    assert close == pytest.approx(0.5)
