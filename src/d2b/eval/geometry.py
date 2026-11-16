"""Geometric comparison of solids: load ground truth, compute exact volumetric IoU.

All lengths are in centimetres, the unit of the Fusion 360 Gallery JSON. STEP files are
read in millimetres by OpenCascade, so `load_step` scales them back to centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq

MM_PER_CM = 10.0


def to_shape(obj: cq.Workplane | cq.Shape) -> cq.Shape:
    """Collapse a Workplane (possibly holding several bodies) into one Shape."""
    if isinstance(obj, cq.Shape):
        return obj
    solids = [s for v in obj.vals() if isinstance(v, cq.Shape) for s in v.Solids()]
    if not solids:
        raise ValueError("no solids in workplane")
    return solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)


def load_step(path: str | Path) -> cq.Shape:
    """Load a ground-truth STEP file as a single shape in centimetres."""
    return to_shape(cq.importers.importStep(str(path))).scale(1.0 / MM_PER_CM)


@dataclass(frozen=True)
class Comparison:
    valid: bool
    volume: float
    volume_ref: float
    iou: float

    @property
    def volume_error(self) -> float:
        """Relative volume error against the reference."""
        return abs(self.volume - self.volume_ref) / self.volume_ref


def iou(a: cq.Shape, b: cq.Shape) -> float:
    """Exact volumetric intersection over union, from a B-Rep boolean."""
    va, vb = a.Volume(), b.Volume()
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    inter = sum(s.Volume() for s in a.intersect(b).Solids())
    union = va + vb - inter
    return max(0.0, min(1.0, inter / union)) if union > 0.0 else 0.0


def compare(candidate: cq.Workplane | cq.Shape, reference: cq.Shape) -> Comparison:
    shape = to_shape(candidate)
    return Comparison(
        valid=shape.isValid() and shape.Volume() > 0.0,
        volume=shape.Volume(),
        volume_ref=reference.Volume(),
        iou=iou(shape, reference),
    )
