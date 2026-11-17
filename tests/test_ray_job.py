"""The dataset job reads labels, draws parts and writes the training pairs correctly."""

# These tests call the job's pieces directly, without starting Ray or needing the dataset.

import json

import cadquery as cq

from d2b.pipeline.ray_job import draw_part, load_labels, write_pairs


# Helper: writes records to a file, one JSON object per line.
def write_lines(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


# Helper: reads a file of one JSON object per line.
def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# Only programs that passed the check are loaded as labels.
def test_load_labels_keeps_only_verified_programs(tmp_path):
    records = [
        {"name": "a", "status": "ok", "program": "code a"},
        {"name": "b", "status": "mismatch", "program": "code b"},
        {"name": "c", "status": "unsupported", "program": None},
    ]
    write_lines(tmp_path / "verify.jsonl", records)
    assert load_labels(tmp_path / "verify.jsonl") == {"a": "code a"}


# A pair is written only for a part that has both a program and a drawing.
def test_write_pairs_needs_a_program_and_a_drawing(tmp_path):
    labels = {"a": "code a", "b": "code b"}
    # Part "b" has a program but its drawing failed; part "c" has a drawing but no program.
    count = write_pairs(tmp_path / "pairs.jsonl", ["c", "b", "a"], labels, drawn={"a", "c"})
    assert count == 1
    expected = {"name": "a", "image": "drawings/a.png", "program": "code a"}
    assert read_lines(tmp_path / "pairs.jsonl") == [expected]


# Drawing a real STEP file saves an image and reports success.
def test_draw_part_saves_an_image(tmp_path):
    (tmp_path / "reconstruction").mkdir()
    (tmp_path / "out" / "drawings").mkdir(parents=True)
    step = tmp_path / "reconstruction" / "box.step"
    cq.exporters.export(cq.Workplane("XY").box(40, 20, 10), str(step))
    result = draw_part("box", tmp_path, tmp_path / "out")
    assert result["status"] == "ok"
    assert (tmp_path / "out" / "drawings" / "box.png").exists()


# A part whose file is missing is reported as failed, and does not stop the job.
def test_draw_part_reports_a_missing_file(tmp_path):
    result = draw_part("nothing_here", tmp_path, tmp_path)
    assert result["status"] == "failed"
    assert result["name"] == "nothing_here"
