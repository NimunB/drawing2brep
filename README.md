# drawing2brep

Engineers describe parts with flat technical drawings, but design software needs a 3D model that can be edited. This project turns a drawing of a part into an editable CAD program, checks the result against the true geometry, and (planned) lets you change it with requests like "make it 5 mm thicker". It is trained and tested on [Fusion 360 Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset), Autodesk Research's dataset of real designs with their construction history.

> **Status:** in development. This README describes the planned design. Results are added as they are measured.

## What it does

1. Converts each dataset design into a CadQuery program and verifies it reproduces the original solid.
2. Produces every representation of each design with a Ray pipeline: orthographic drawing with hidden lines, mesh, point cloud, and B-Rep face-adjacency graph.
3. Fine-tunes a vision-language model to write the CadQuery program from the drawing.
4. Executes the generated program in a sandbox and measures the solid against the ground truth.

## Design

- **The output is a CadQuery program,** not a mesh. It is readable, editable, and runs without a CAD application.
- **Convert, then verify.** A converted program is kept only if its solid matches the original (IoU above 0.99). The yield is reported.
- **Official split.** 6,900 training and 1,725 test designs, so results are comparable with published work.
- **Drawings carry overall dimensions.** Without them scale cannot be recovered. Metrics are reported both absolute and scale-normalised.
- **Baselines before training.** Off-the-shelf models are benchmarked first on the same test set.
- **Failure taxonomy.** Each failure is classed as wrong dimension, missing feature or invalid solid.

## Evaluation

| Model | Valid solid | Mean IoU | IoU > 0.9 | Chamfer distance | Volume error |
| --- | --- | --- | --- | --- | --- |
| Frontier model, zero-shot | | | | | |
| Qwen2.5-VL, zero-shot | | | | | |
| Qwen2.5-VL, fine-tuned | | | | | |
| + GRPO with geometric reward | | | | | |
| + repair loop | | | | | |

## Roadmap

MVP

- [ ] Sequence-to-CadQuery converter with verification
- [ ] Ray pipeline for drawings, meshes, point clouds and B-Rep graphs
- [ ] Geometric evaluator
- [ ] Zero-shot baselines
- [ ] LoRA fine-tune
- [ ] Failure taxonomy
- [ ] Demo: upload a drawing, see the solid

After the MVP

- [ ] Full test set and few-shot baselines
- [ ] GRPO with a geometric reward
- [ ] Execute, check, repair loop
- [ ] UV-Net-style B-Rep encoder for retrieval and similarity scoring
- [ ] WebGL viewer with MCP tools for natural-language edits, with approval before each change
- [ ] Round trip through Fusion 360's Python API
- [ ] Released benchmark

## Quickstart

To be added with the first working version.
