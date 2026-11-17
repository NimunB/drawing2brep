"""Grade a model's saved replies: run each program safely and measure the solid it builds."""

# Run with: uv run python -m d2b.eval.run outputs/baseline_claude-opus-5-5.jsonl

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import Counter
from pathlib import Path

import numpy as np

from d2b.data.verify import run_program
from d2b.eval.geometry import evaluate, load_step
from d2b.eval.sandbox import run_untrusted
from d2b.models.prompt import extract_program

DEFAULT_ROOT = Path("data/raw/r1.0.1")
DEFAULT_PAIRS = Path("data/processed/pairs_test.jsonl")

# A part counts as "close" when its aligned overlap with the true solid is above this.
CLOSE_IOU = 0.9

# A part counts as "thin" when its smallest side is under this fraction of its largest.
THIN_RATIO = 0.05


# Grades one reply against the true solid; returns a flat record of every measurement.
def grade_reply(reply: str, name: str, label_program: str, root: Path) -> dict:
    record = {"name": name, "status": "", "detail": ""}
    program = extract_program(reply)
    if program is None:
        return record | {"status": "no_program", "detail": "the reply had no code block"}

    # The model's program is untrusted, so it runs in the sandbox.
    outcome = run_untrusted(program)
    if outcome.status != "ok":
        return record | {"status": outcome.status, "detail": outcome.detail}

    truth = load_step(root / "reconstruction" / f"{name}.step")
    # Face and edge counts are compared with the solid our own label program builds.
    result = evaluate(outcome.shape, truth, structure_reference=run_program(label_program))
    if not result.valid:
        return record | {"status": "invalid", "detail": "the solid is not a proper closed solid"}
    return record | {
        "status": "ok",
        "iou": result.iou,
        "iou_aligned": result.iou_aligned,
        "volume_error": result.volume_error,
        "size_error": result.size_error,
        "chamfer_normalised": result.chamfer_normalised,
        "same_structure": result.same_structure,
    }


# Describes a part so results can be grouped: its number of steps, and whether it is hard to draw.
def describe_part(name: str, label_program: str, root: Path) -> dict:
    properties = json.loads((root / "reconstruction" / f"{name}.json").read_text())["properties"]
    box = properties["bounding_box"]
    sides = sorted(box["max_point"][axis] - box["min_point"][axis] for axis in "xyz")
    return {
        # Each build step in a label program starts with a "# Step" comment.
        "steps": label_program.count("# Step "),
        "thin": sides[0] / sides[2] < THIN_RATIO,
        "multi_body": properties["body_count"] > 1,
    }


# Scores one group of replies: (how many, mean aligned IoU, share above the "close" mark).
def score_group(records: list[dict]) -> tuple[int, float, float]:
    # A reply with no valid solid counts as an overlap of 0.
    scores = np.array([r.get("iou_aligned", 0.0) for r in records])
    return len(records), float(scores.mean()), float((scores > CLOSE_IOU).mean())


# Prints the results split by kind of part, to show where a model does well or badly.
def print_breakdown(records: list[dict]) -> None:
    groups = {
        "1 step": [r for r in records if r["steps"] == 1],
        "2 steps": [r for r in records if r["steps"] == 2],
        "3 or more steps": [r for r in records if r["steps"] >= 3],
        "thin parts": [r for r in records if r["thin"]],
        "not thin": [r for r in records if not r["thin"]],
        "more than one body": [r for r in records if r["multi_body"]],
        "one body": [r for r in records if not r["multi_body"]],
    }
    print(f"\n{'group':<20} {'parts':>6} {'mean IoU':>9} {'IoU > ' + str(CLOSE_IOU):>10}")
    for label, group in groups.items():
        if group:
            count, mean, close = score_group(group)
            print(f"{label:<20} {count:>6} {mean:>9.3f} {close:>10.1%}")


# Grades one reply and adds the part's description; takes one tuple so it can run in a pool.
def grade_task(task: tuple[str, str, str, Path]) -> dict:
    reply, name, label_program, root = task
    return grade_reply(reply, name, label_program, root) | describe_part(name, label_program, root)


# Prints the summary table: how many replies gave a solid, and how close those solids were.
def print_summary(records: list[dict]) -> None:
    total = len(records)
    good = [r for r in records if r["status"] == "ok"]
    print(f"\nreplies graded: {total}")
    for status, count in Counter(r["status"] for r in records).most_common():
        print(f"  {status:<11} {count:>5}  {count / total:6.1%}")
    if not good:
        return
    # A reply that gave no valid solid counts as an overlap of 0 in the averages over all replies.
    aligned = np.array([r["iou_aligned"] for r in good])
    print(f"\nvalid solid:             {len(good) / total:.1%}")
    print(f"mean IoU (aligned):      {aligned.sum() / total:.3f}   (over all replies)")
    close = (aligned > CLOSE_IOU).sum() / total
    placed = sum(r["iou"] for r in good) / total
    print(f"IoU above {CLOSE_IOU} (aligned): {close:.1%}   (over all replies)")
    print(f"mean IoU as placed:      {placed:.3f}   (over all replies)")
    print("among valid solids only:")
    print(f"  median Chamfer distance: {np.median([r['chamfer_normalised'] for r in good]):.4f}")
    print(f"  median volume error:     {np.median([r['volume_error'] for r in good]):.1%}")
    print(f"  median size error:       {np.median([r['size_error'] for r in good]):.1%}")
    print(f"  same faces and edges:    {np.mean([r['same_structure'] for r in good]):.1%}")


# Command-line entry point: grade every reply in a file and save the measurements.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replies", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    labels = {}
    for line in args.pairs.read_text().splitlines():
        pair = json.loads(line)
        labels[pair["name"]] = pair["program"]

    replies = [json.loads(line) for line in args.replies.read_text().splitlines()]
    tasks = [(r["reply"], r["name"], labels[r["name"]], args.root) for r in replies]
    # Each reply is graded independently, so they are spread across CPU cores.
    with multiprocessing.Pool(args.workers) as pool:
        records = pool.map(grade_task, tasks, chunksize=4)

    out = args.replies.with_name(args.replies.stem + "_graded.jsonl")
    out.write_text("".join(json.dumps(record) + "\n" for record in records))
    print_summary(records)
    print_breakdown(records)
    print(f"\nmeasurements saved to {out}")


if __name__ == "__main__":
    main()
