"""Make every representation of each verified part, and save the training pairs."""

# Run with: uv run python -m d2b.pipeline.ray_job (Ray spreads the parts across CPU cores).

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

# Stops Ray starting each worker through `uv`, which made start-up take over a minute.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import numpy as np  # noqa: E402
import ray  # noqa: E402

from d2b.eval.geometry import load_step  # noqa: E402
from d2b.pipeline.brep_graph import make_graph  # noqa: E402
from d2b.pipeline.drawing import save_drawing  # noqa: E402
from d2b.pipeline.mesh import make_mesh  # noqa: E402
from d2b.pipeline.pointcloud import sample_points  # noqa: E402

DEFAULT_ROOT = Path("data/raw/r1.0.1")
DEFAULT_LABELS = Path("outputs/verify.jsonl")
DEFAULT_OUT = Path("data/processed")

# Parts are handed to workers in groups of this size, to keep the overhead per part small.
BATCH_SIZE = 32


# Reads the verified programs from the yield script's output, as {part name: program text}.
def load_labels(path: Path) -> dict[str, str]:
    labels = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            # Only programs that rebuilt the true solid are used as labels.
            if record["status"] == "ok":
                labels[record["name"]] = record["program"]
    return labels


# Saves a part's mesh, point cloud and B-Rep graph together in one compressed file.
def save_representations(solid, path: Path) -> None:
    mesh_points, mesh_triangles = make_mesh(solid)
    cloud_points, cloud_normals = sample_points(solid)
    np.savez_compressed(
        path,
        mesh_points=mesh_points.astype(np.float32),
        mesh_triangles=mesh_triangles.astype(np.int32),
        cloud_points=cloud_points.astype(np.float32),
        cloud_normals=cloud_normals.astype(np.float32),
        **make_graph(solid),
    )


# Makes every representation of one part from its true solid; returns what happened.
def process_part(name: str, root: Path, out_dir: Path, representations: bool) -> dict:
    try:
        # Everything comes from the dataset's solid, never from our converted program.
        solid = load_step(root / "reconstruction" / f"{name}.step")
        save_drawing(solid, out_dir / "drawings" / f"{name}.png")
        if representations:
            save_representations(solid, out_dir / "representations" / f"{name}.npz")
    except Exception as error:  # noqa: BLE001
        return {"name": name, "status": "failed", "detail": f"{type(error).__name__}: {error}"}
    return {"name": name, "status": "ok", "detail": ""}


# Ray task: processes a group of parts on one worker.
@ray.remote
def process_batch(names: list[str], root: Path, out_dir: Path, representations: bool) -> list[dict]:
    return [process_part(name, root, out_dir, representations) for name in names]


# Writes one file of pairs for a split: each line has a part's drawing path and its program.
def write_pairs(path: Path, names: list[str], labels: dict[str, str], drawn: set[str]) -> int:
    count = 0
    with path.open("w") as handle:
        for name in sorted(names):
            # A part needs both a verified program and a drawing to be a training pair.
            if name in labels and name in drawn:
                pair = {"name": name, "image": f"drawings/{name}.png", "program": labels[name]}
                handle.write(json.dumps(pair) + "\n")
                count += 1
    return count


# Command-line entry point: draw all verified parts in parallel, then save the pairs.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int, default=None)
    # Drawings are always made; the mesh, point cloud and graph are made only when asked for.
    parser.add_argument("--representations", action="store_true")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    names = sorted(labels)[: args.limit] if args.limit else sorted(labels)
    # Workers run in other processes, so they are given full paths.
    root, out_dir = args.root.resolve(), args.out.resolve()
    (out_dir / "drawings").mkdir(parents=True, exist_ok=True)
    (out_dir / "representations").mkdir(parents=True, exist_ok=True)

    # Start Ray on this machine with a fixed number of worker processes.
    ray.init(num_cpus=args.workers, include_dashboard=False, logging_level="ERROR")
    started = time.time()
    batches = [names[i : i + BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]
    futures = [process_batch.remote(b, root, out_dir, args.representations) for b in batches]
    # ray.get waits for every worker to finish and collects their results.
    results = [record for batch in ray.get(futures) for record in batch]
    seconds = time.time() - started
    ray.shutdown()

    drawn = {r["name"] for r in results if r["status"] == "ok"}
    failed = [r for r in results if r["status"] != "ok"]

    # The dataset's official split decides which pairs are for training and which for testing.
    split = json.loads((args.root / "train_test.json").read_text())
    chosen = set(names)
    counts = {
        part: write_pairs(
            out_dir / f"pairs_{part}.jsonl", [n for n in split[part] if n in chosen], labels, drawn
        )
        for part in ("train", "test")
    }

    # Failures are saved with their reason so they can be looked at afterwards.
    with (out_dir / "drawing_failures.jsonl").open("w") as handle:
        for record in failed:
            handle.write(json.dumps(record) + "\n")

    print(f"parts drawn: {len(drawn)} of {len(names)}")
    print(f"failed:      {len(failed)}")
    for reason, count in Counter(r["detail"].split(":")[0] for r in failed).most_common():
        print(f"  {count:>5}  {reason}")
    print(f"pairs:       {counts['train']} train, {counts['test']} test")
    rate = len(names) / seconds
    print(f"time:        {seconds:.1f} s on {args.workers} workers ({rate:.0f} parts/s)")
    print(f"written to   {args.out}")


if __name__ == "__main__":
    main()
