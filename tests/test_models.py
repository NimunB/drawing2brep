"""The prompt helpers, the cost sum and the grading of a reply all behave as expected."""

# No test here calls a model, so none of them costs anything.

import json
from pathlib import Path

import cadquery as cq
import pytest

from d2b.eval.run import grade_reply, score_group
from d2b.models.baselines import choose_pairs, cost_in_dollars
from d2b.models.finetune import (
    IGNORE,
    build_training_messages,
    find_answer_start,
    format_answer,
    keep_short,
    learning_rate_scale,
    mask_labels,
    split_validation,
)
from d2b.models.prompt import SYSTEM_PROMPT, USER_PROMPT, extract_program
from d2b.models.qwen import batches, build_messages, default_out, pending_pairs

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


# Qwen is sent the same instructions and request as Claude, with a marker for the image.
def test_qwen_gets_the_same_prompt_as_claude():
    system, user = build_messages()
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert user["content"] == [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]


# A run that was stopped only asks about the drawings it has no reply for yet.
def test_pending_pairs_skips_finished_drawings(tmp_path):
    pairs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    out = tmp_path / "replies.jsonl"
    # Before anything is saved, every drawing is still to do.
    assert pending_pairs(pairs, out) == pairs
    out.write_text(json.dumps({"name": "b", "reply": "..."}) + "\n")
    assert pending_pairs(pairs, out) == [{"name": "a"}, {"name": "c"}]


# Drawings are grouped into batches of the given size, with a smaller last batch.
def test_batches_splits_a_list():
    assert batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert batches([], 2) == []


# A training answer is in the form the prompt asks for, so the grader can read it back.
def test_training_answer_reads_back_as_the_program():
    program = "import cadquery as cq\nresult = cq.Workplane().box(1, 2, 3)\n"
    assert extract_program(format_answer(program)) == program


# A training conversation is the baseline's conversation with the answer added as a last turn.
def test_training_messages_add_the_answer_to_the_baseline_prompt():
    messages = build_training_messages("result = 1")
    assert messages[:-1] == build_messages()
    assert messages[-1] == {"role": "assistant", "content": format_answer("result = 1")}


# The held-back pairs are the same on every run and share nothing with the training pairs.
def test_split_validation_is_repeatable_and_separate():
    pairs = [{"name": f"part_{i}"} for i in range(20)]
    train, validation = split_validation(pairs, count=5)
    assert len(train) == 15 and len(validation) == 5
    assert not {p["name"] for p in train} & {p["name"] for p in validation}
    # The order the pairs arrive in does not change who is held back.
    assert split_validation(pairs[::-1], count=5)[1] == validation


# Pairs whose answer is over the limit are left out; one exactly at the limit is kept.
def test_keep_short_drops_long_answers():
    pairs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    assert keep_short(pairs, lengths=[10, 21, 20], limit=20) == [{"name": "a"}, {"name": "c"}]


# The answer starts just after the last marker, even if the marker's tokens appear earlier too.
def test_find_answer_start_uses_the_last_marker():
    tokens = [7, 8, 1, 2, 5, 5, 1, 2, 9, 9]
    assert find_answer_start(tokens, marker=[1, 2]) == 8
    with pytest.raises(ValueError):
        find_answer_start(tokens, marker=[3, 4])


# Only the answer's tokens are training targets; the prompt and the padding are ignored.
def test_mask_labels_keeps_only_the_answer():
    tokens = [7, 8, 1, 2, 9, 4, 0, 0]
    assert mask_labels(tokens, marker=[1, 2], padding=0) == [IGNORE] * 4 + [9, 4, IGNORE, IGNORE]


# The learning rate rises during warm-up, peaks at full strength, and ends at zero.
def test_learning_rate_scale_warms_up_then_decays():
    scales = [learning_rate_scale(step, total_steps=100, warmup_steps=10) for step in range(101)]
    assert scales[0] == pytest.approx(0.1)
    assert max(scales) == pytest.approx(1.0)
    assert scales[100] == pytest.approx(0.0)
    assert all(a >= b for a, b in zip(scales[10:], scales[11:], strict=False))


# Replies from a fine-tuned model go to a file named after the run, not over the baseline's.
def test_default_out_names_the_file_after_the_run():
    assert default_out("Qwen/Qwen2.5-VL-3B-Instruct", None) == Path(
        "outputs/baseline_Qwen2.5-VL-3B-Instruct.jsonl"
    )
    adapter = Path("outputs/finetune/first/checkpoint-300")
    assert default_out("Qwen/Qwen2.5-VL-3B-Instruct", adapter) == Path(
        "outputs/finetuned_first_checkpoint-300.jsonl"
    )
