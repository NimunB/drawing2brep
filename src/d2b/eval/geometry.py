"""Geometric comparison of solids: load ground truth, compute exact volumetric IoU.

All lengths are in centimetres, the unit of the Fusion 360 Gallery JSON. STEP files are
read in millimetres by OpenCascade, so `load_step` scales them back to centimetres.
"""

# This file only measures two solids that already exist. It converts and generates nothing.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq

MM_PER_CM = 10.0


# Unwraps whatever a CadQuery program returns into one plain solid we can measure.
def to_shape(obj: cq.Workplane | cq.Shape) -> cq.Shape:
    """Collapse a Workplane (possibly holding several bodies) into one Shape."""
    # Already a plain solid (e.g. from load_step): nothing to unwrap.
    if isinstance(obj, cq.Shape):
        return obj
    # A CadQuery program returns a Workplane wrapper; pull the solid bodies out of it.
    solids = [s for v in obj.vals() if isinstance(v, cq.Shape) for s in v.Solids()]
    # The program ran but built no solid (e.g. only a 2D sketch).
    if not solids:
        raise ValueError("no solids in workplane")
    # Several separate bodies are grouped so they can be measured as one part.
    return solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)


# Reads a part's true solid (the correct answer) from the dataset's .step file.
def load_step(path: str | Path) -> cq.Shape:
    """Load a ground-truth STEP file as a single shape in centimetres."""
    # The .step file is the correct answer from the dataset. We only read it, never write it.
    # Scale mm -> cm; without this the true solid is 1,000x too large in volume.
    return to_shape(cq.importers.importStep(str(path))).scale(1.0 / MM_PER_CM)


# Holds the results of comparing one candidate solid with the true solid.
@dataclass(frozen=True)
class Comparison:
    valid: bool  # is the candidate a proper closed solid with some volume?
    volume: float  # candidate's volume, cm^3
    volume_ref: float  # true solid's volume, cm^3
    iou: float  # overlap score, 0 to 1; the main number

    # Returns how far off the candidate's volume is, as a fraction (0.05 = 5% off).
    @property
    def volume_error(self) -> float:
        """Relative volume error against the reference."""
        return abs(self.volume - self.volume_ref) / self.volume_ref


# Scores how well two solids overlap: shared volume / total covered volume (1 = identical).
def iou(a: cq.Shape, b: cq.Shape) -> float:
    """Exact volumetric intersection over union, from a B-Rep boolean."""
    va, vb = a.Volume(), b.Volume()
    # An empty or broken solid scores 0 instead of dividing by zero.
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    # Volume of the region that is inside both solids.
    inter = sum(s.Volume() for s in a.intersect(b).Solids())
    # Adding both volumes counts the shared region twice, so subtract it once.
    union = va + vb - inter
    # Clamp to 0..1 to absorb tiny numerical errors from the geometry engine.
    return max(0.0, min(1.0, inter / union)) if union > 0.0 else 0.0


# Main entry point: checks a candidate solid against the true solid and returns all scores.
def compare(candidate: cq.Workplane | cq.Shape, reference: cq.Shape) -> Comparison:
    # candidate = solid built by running a CadQuery program (ours or the model's).
    # reference = true solid from load_step.
    # No alignment is done: a correct shape in the wrong position scores low.
    shape = to_shape(candidate)
    return Comparison(
        valid=shape.isValid() and shape.Volume() > 0.0,
        volume=shape.Volume(),
        volume_ref=reference.Volume(),
        iou=iou(shape, reference),
    )
