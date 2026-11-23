"""Convert designs to CadQuery, run each program, and keep those that match the true solid."""

# Run with: uv run python -m d2b.data.verify --subset single (the share that passes is the yield).

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import Counter
from pathlib import Path

from d2b.data.to_cadquery import UnsupportedError, convert, design_centre
from d2b.eval.geometry import compare, load_step

# A converted program is kept only if its solid overlaps the true solid this closely.
IOU_THRESHOLD = 0.99

DEFAULT_ROOT = Path("data/raw/r1.0.1")


# Runs a CadQuery program and returns the solid it stores in `result`.
def run_program(code: str):
    # These programs come from our own converter, so running them in-process is safe.
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102
    return namespace["result"]


# Converts, runs and scores one design; returns a small record of what happened.
def check_design(json_path: Path) -> dict:
    name = json_path.stem
    record = {"name": name, "status": "", "detail": "", "iou": None, "program": None}
    design = json.loads(json_path.read_text())

    # Step 1: translate the recorded steps into a program that builds the part on the origin.
    # A drawing does not show where a part sits, so its position is removed from the program.
    centre = design_centre(design)
    record["centre"] = list(centre)
    try:
        record["program"] = convert(design, centre)
    except UnsupportedError as error:
        return record | {"status": "unsupported", "detail": str(error)}

    # Step 2: run the program to get our solid, then compare it with the true solid.
    try:
        # The true solid is moved by the same amount, so the two are compared in the same place.
        truth = load_step(json_path.with_suffix(".step")).translate(tuple(-c for c in centre))
        result = compare(run_program(record["program"]), truth)
    except Exception as error:  # noqa: BLE001
        # The program crashed or built geometry the engine could not handle.
        return record | {"status": "error", "detail": type(error).__name__}

    # Step 3: keep it only if it is a proper solid that matches closely enough.
    record["iou"] = result.iou
    record["method"] = result.method
    if not result.valid:
        return record | {"status": "invalid"}
    if result.iou <= IOU_THRESHOLD:
        return record | {"status": "mismatch"}
    return record | {"status": "ok"}


# Counts the extrude steps in a design file.
def count_extrudes(json_path: Path) -> int:
    entities = json.loads(json_path.read_text())["entities"]
    return sum(e["type"] == "ExtrudeFeature" for e in entities.values())


# Lists the design files to check, according to the chosen split and subset.
def select_designs(root: Path, split: str, subset: str, limit: int | None) -> list[Path]:
    names = json.loads((root / "train_test.json").read_text())
    chosen = names["train"] + names["test"] if split == "all" else names[split]
    paths = [root / "reconstruction" / f"{name}.json" for name in sorted(chosen)]
    # "single" keeps only the simplest designs: those made with one extrude.
    if subset == "single":
        paths = [p for p in paths if count_extrudes(p) == 1]
    return paths[:limit] if limit else paths


# Prints the yield and a breakdown of why designs failed.
def print_report(records: list[dict]) -> None:
    total = len(records)
    statuses = Counter(r["status"] for r in records)
    print(f"\ndesigns checked: {total}")
    for status, count in statuses.most_common():
        print(f"  {status:<12} {count:>6}  {count / total:6.1%}")
    print(f"\nyield (IoU > {IOU_THRESHOLD}): {statuses['ok'] / total:.1%}")

    # How often the slower sampling fallback was needed to score a design.
    sampled = sum(r.get("method") == "sampled" for r in records)
    print(f"scored by sampling fallback: {sampled}")

    # Break the failures down by their specific reason.
    reasons = Counter(
        f"{r['status']}: {r['detail']}" for r in records if r["status"] in ("unsupported", "error")
    )
    if reasons:
        print("\nreasons:")
        for reason, count in reasons.most_common(15):
            print(f"  {count:>6}  {reason}")


# Command-line entry point: check the chosen designs in parallel and save the records.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--subset", choices=["single", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, default=Path("outputs/verify.jsonl"))
    args = parser.parse_args()

    paths = select_designs(args.root, args.split, args.subset, args.limit)
    # Each design is independent, so they are checked on several CPU cores at once.
    with multiprocessing.Pool(args.workers) as pool:
        records = list(pool.imap_unordered(check_design, paths, chunksize=16))

    # One line per design, so failures can be inspected afterwards.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in sorted(records, key=lambda r: r["name"]):
            handle.write(json.dumps(record) + "\n")

    print_report(records)
    print(f"\nrecords written to {args.out}")


if __name__ == "__main__":
    main()
