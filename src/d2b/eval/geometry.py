"""The verifier: measures how closely two solids match (lengths are in centimetres)."""

# This file only measures two solids that already exist. It converts and generates nothing.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.spatial import cKDTree

MM_PER_CM = 10.0

# Solids are treated as touching when they are within this distance (cm) of each other.
FUZZY_TOLERANCE = 1e-4

# Two measurements of shared volume must agree to within this fraction to be trusted.
AGREEMENT = 1e-3

# Number of random points used when falling back to sampling.
SAMPLE_POINTS = 20000

# Number of points scattered on each solid's surface to make its point cloud.
SURFACE_POINTS = 2048

# A much denser cloud used as the target when measuring Chamfer distance.
DENSE_POINTS = 30000

# Surfaces are split into triangles no further from the true surface than this fraction of size.
MESH_TOLERANCE = 1e-3


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
    # The engine loads STEP files in mm but our programs are in cm, so scale by 1/10.
    return to_shape(cq.importers.importStep(str(path))).scale(1.0 / MM_PER_CM)


# Holds the results of comparing one candidate solid with the true solid.
@dataclass(frozen=True)
class Comparison:
    valid: bool  # is the candidate a proper closed solid with some volume?
    volume: float  # candidate's volume, cm^3
    volume_ref: float  # true solid's volume, cm^3
    iou: float  # overlap score, 0 to 1; the main number
    method: str  # how the IoU was measured: "exact", "sampled" or "empty"

    # Returns how far off the candidate's volume is, as a fraction (0.05 = 5% off).
    @property
    def volume_error(self) -> float:
        """Relative volume error against the reference."""
        return abs(self.volume - self.volume_ref) / self.volume_ref


# Adds up the volume of every solid body inside a shape.
def _total_volume(shape: cq.Shape) -> float:
    return sum(solid.Volume() for solid in shape.Solids())


# Measures the shared volume exactly, two independent ways; None if they disagree.
def _exact_shared_volume(a: cq.Shape, b: cq.Shape, va: float, vb: float) -> float | None:
    # The engine can silently get near-identical solids wrong, so two answers must agree.
    try:
        # Way 1: build the region inside both solids and measure it.
        direct = _total_volume(a.intersect(b, tol=FUZZY_TOLERANCE))
        # Way 2: merge the solids; the merged volume is va + vb minus the shared part.
        via_merge = va + vb - _total_volume(a.fuse(b, tol=FUZZY_TOLERANCE))
    except Exception:  # noqa: BLE001
        return None
    smaller = min(va, vb)
    if abs(direct - via_merge) > AGREEMENT * smaller:
        return None
    # The shared part can never be negative or bigger than the smaller solid.
    if direct < -AGREEMENT * smaller or direct > (1.0 + AGREEMENT) * smaller:
        return None
    return max(0.0, min(direct, smaller))


# True/False for each point: is it inside the shape?
def _points_inside(shape: cq.Shape, points: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    # A shape may hold several bodies; a point counts if it is inside any of them.
    for solid in shape.Solids():
        inside |= np.array([solid.isInside(cq.Vector(*p), 1e-6) for p in points])
    return inside


# Estimates IoU by scattering random points and counting which solids contain them.
def _sampled_iou(a: cq.Shape, b: cq.Shape) -> float:
    # Slower and approximate, so it is only the fallback when the exact method fails.
    box_a, box_b = a.BoundingBox(), b.BoundingBox()
    low = np.minimum([box_a.xmin, box_a.ymin, box_a.zmin], [box_b.xmin, box_b.ymin, box_b.zmin])
    high = np.maximum([box_a.xmax, box_a.ymax, box_a.zmax], [box_b.xmax, box_b.ymax, box_b.zmax])
    # A fixed seed makes the score identical every time it is run.
    points = np.random.default_rng(0).uniform(low, high, (SAMPLE_POINTS, 3))
    in_a, in_b = _points_inside(a, points), _points_inside(b, points)
    in_either = int((in_a | in_b).sum())
    return float((in_a & in_b).sum() / in_either) if in_either else 0.0


# Scores the overlap and also says which method produced the score.
def iou_with_method(a: cq.Shape, b: cq.Shape) -> tuple[float, str]:
    va, vb = a.Volume(), b.Volume()
    # An empty or broken solid scores 0 instead of dividing by zero.
    if va <= 0.0 or vb <= 0.0:
        return 0.0, "empty"
    shared = _exact_shared_volume(a, b, va, vb)
    if shared is None:
        return _sampled_iou(a, b), "sampled"
    # Adding both volumes counts the shared region twice, so subtract it once.
    return shared / (va + vb - shared), "exact"


# Scores how well two solids overlap: shared volume / total covered volume (1 = identical).
def iou(a: cq.Shape, b: cq.Shape) -> float:
    """Volumetric intersection over union."""
    return iou_with_method(a, b)[0]


# Main entry point: checks a candidate solid against the true solid and returns all scores.
def compare(candidate: cq.Workplane | cq.Shape, reference: cq.Shape) -> Comparison:
    # candidate = solid built by running a program; reference = true solid from load_step.
    shape = to_shape(candidate)
    score, method = iou_with_method(shape, reference)
    return Comparison(
        valid=shape.isValid() and shape.Volume() > 0.0,
        volume=shape.Volume(),
        volume_ref=reference.Volume(),
        iou=score,
        method=method,
    )


# The full set of measurements used to grade a candidate solid against the true solid.
@dataclass(frozen=True)
class Evaluation:
    valid: bool  # is the candidate a proper closed solid with some volume?
    iou: float  # overlap score, 0 to 1, with the candidate where its program put it
    iou_aligned: float  # overlap score after sliding the candidate onto the true solid
    method: str  # how the IoU was measured: "exact", "sampled" or "empty"
    volume_error: float  # relative volume error (0.05 = 5% off)
    chamfer: float  # average distance between the surfaces after aligning, in cm
    chamfer_normalised: float  # the same, as a fraction of the true solid's diagonal
    size_error: float  # worst relative error in overall width, depth or height
    face_count: int  # number of faces on the candidate
    face_count_ref: int  # number of faces on the solid used as the structure reference
    edge_count: int  # number of edges on the candidate
    edge_count_ref: int  # number of edges on the solid used as the structure reference

    # True when the candidate has the same number of faces and edges as the structure reference.
    @property
    def same_structure(self) -> bool:
        return self.face_count == self.face_count_ref and self.edge_count == self.edge_count_ref


# Length of the diagonal of a solid's bounding box: a single number for its overall size.
def _diagonal(shape: cq.Shape) -> float:
    return shape.BoundingBox().DiagonalLength


# Scatters points evenly over a solid's surface; this is its point cloud.
def sample_surface(shape: cq.Shape, count: int = SURFACE_POINTS, seed: int = 0) -> np.ndarray:
    # Split the surface into triangles (a mesh), fine enough to follow the curves.
    vertices, triangles = shape.tessellate(_diagonal(shape) * MESH_TOLERANCE)
    points = np.array([v.toTuple() for v in vertices])
    corners = points[np.array(triangles)]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    # Bigger triangles get more points, so the points are spread evenly by area.
    areas = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2.0
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(areas), size=count, p=areas / areas.sum())
    # Pick a random spot inside each chosen triangle.
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return a[chosen] + u[:, None] * (b - a)[chosen] + v[:, None] * (c - a)[chosen]


# Average distance from each point on one surface to the nearest point on the other.
def chamfer_distance(a: cq.Shape, b: cq.Shape) -> float:
    # Each solid gets a dense cloud; its first points serve as the sparse cloud to measure from.
    dense_a, dense_b = sample_surface(a, DENSE_POINTS), sample_surface(b, DENSE_POINTS)
    # Measuring to a dense cloud keeps the gaps between points from inflating the distance.
    a_to_b = cKDTree(dense_b).query(dense_a[:SURFACE_POINTS])[0]
    b_to_a = cKDTree(dense_a).query(dense_b[:SURFACE_POINTS])[0]
    # Measured both ways and averaged, so neither solid is favoured.
    return float((a_to_b.mean() + b_to_a.mean()) / 2.0)


# Worst relative error in overall width, depth or height.
def size_error(candidate: cq.Shape, reference: cq.Shape) -> float:
    box, ref = candidate.BoundingBox(), reference.BoundingBox()
    sizes = np.array([box.xlen, box.ylen, box.zlen])
    ref_sizes = np.array([ref.xlen, ref.ylen, ref.zlen])
    return float(np.max(np.abs(sizes - ref_sizes) / ref_sizes))


# Slides a candidate so the centre of its bounding box sits on the reference's centre.
def align_centres(candidate: cq.Shape, reference: cq.Shape) -> cq.Shape:
    # A drawing does not show where the part sits in space, so position is not graded.
    return candidate.translate(reference.BoundingBox().center - candidate.BoundingBox().center)


# Grades a candidate against the true solid with every measurement. Used to score models.
def evaluate(
    candidate: cq.Workplane | cq.Shape,
    reference: cq.Shape,
    structure_reference: cq.Workplane | cq.Shape | None = None,
) -> Evaluation:
    shape = to_shape(candidate)
    basic = compare(shape, reference)
    # Shape and size are graded with the position difference removed.
    aligned = align_centres(shape, reference)
    chamfer = chamfer_distance(aligned, reference)
    # Face counts depend on how a solid was built, so they are compared with a solid built
    # the same way (the label program's solid) when one is given.
    structure = to_shape(structure_reference) if structure_reference is not None else reference
    return Evaluation(
        valid=basic.valid,
        iou=basic.iou,
        iou_aligned=iou(aligned, reference),
        method=basic.method,
        volume_error=basic.volume_error,
        chamfer=chamfer,
        chamfer_normalised=chamfer / _diagonal(reference),
        size_error=size_error(shape, reference),
        face_count=len(shape.Faces()),
        face_count_ref=len(structure.Faces()),
        edge_count=len(shape.Edges()),
        edge_count_ref=len(structure.Edges()),
    )
