"""Turn a 3D solid into a technical drawing: front, top and right views on one image."""

# The drawing is the model's input. It is computed from the solid; nothing is drawn by hand.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepLib import BRepLib
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from PIL import Image, ImageDraw, ImageFont

# Width and height of the finished image, in pixels.
IMAGE_SIZE = 768

# The image is drawn this many times larger, then shrunk, which smooths the lines.
SUPERSAMPLE = 3

# Empty space kept around the views, in pixels: left and bottom hold the dimensions.
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 96, 28, 44, 70

# Space between neighbouring views, in pixels.
VIEW_GAP = 56

# Line widths, in pixels.
VISIBLE_WIDTH, HIDDEN_WIDTH, DIMENSION_WIDTH = 2.0, 1.0, 1.0

# Length of each dash and each gap in a hidden line, in pixels.
DASH, DASH_GAP = 7.0, 5.0

# Colours: black edges, grey hidden edges and labels, blue dimensions.
BLACK, GREY, BLUE, WHITE = (17, 17, 17), (95, 95, 95), (29, 78, 216), (255, 255, 255)

# Curves are approximated by straight pieces no further than this fraction of the part's size.
CURVE_TOLERANCE = 5e-4

# Each view: its label, the direction the viewer looks from, and which way is "right" on paper.
VIEWS = {
    "front": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "top": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}


# One flat view of a part: its visible edges and its hidden edges, as lists of 2D points.
@dataclass(frozen=True)
class View:
    visible: list[np.ndarray]
    hidden: list[np.ndarray]


# Turns the edges of a projected shape into lists of 2D points.
def _polylines(compound, tolerance: float) -> list[np.ndarray]:
    lines = []
    if compound.IsNull():
        return lines
    # The projected curves need their 3D form rebuilt before they can be measured.
    BRepLib.BuildCurves3d_s(compound, 1e-6)
    for edge in cq.Shape(compound).Edges():
        curve = BRepAdaptor_Curve(edge.wrapped)
        # Sample points along the edge, closely enough that curves look smooth.
        sampler = GCPnts_QuasiUniformDeflection(
            curve, tolerance, curve.FirstParameter(), curve.LastParameter()
        )
        if not sampler.IsDone() or sampler.NbPoints() < 2:
            continue
        points = [sampler.Value(i + 1) for i in range(sampler.NbPoints())]
        lines.append(np.array([(p.X(), p.Y()) for p in points]))
    return lines


# Projects a solid onto a flat plane as seen from one direction, sorting edges by visibility.
def project(shape: cq.Shape, eye: tuple, right: tuple, tolerance: float) -> View:
    # The geometry engine works out which edges are blocked from view by the part itself.
    algo = HLRBRep_Algo()
    algo.Add(shape.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(), gp_Dir(*eye), gp_Dir(*right))))
    algo.Update()
    algo.Hide()
    edges = HLRBRep_HLRToShape(algo)
    # Sharp edges, smooth edges and silhouette outlines all count as drawable lines.
    visible = [edges.VCompound(), edges.Rg1LineVCompound(), edges.OutLineVCompound()]
    hidden = [edges.HCompound(), edges.OutLineHCompound()]
    return View(
        visible=[line for c in visible for line in _polylines(c, tolerance)],
        hidden=[line for c in hidden for line in _polylines(c, tolerance)],
    )


# Formats a dimension to 4 significant figures, such as 0.5514 or 12.5.
def format_dimension(value: float) -> str:
    return f"{value:.4g}"


# Draws a line made of dashes along a list of pixel points.
def _draw_dashed(draw: ImageDraw.ImageDraw, points: np.ndarray, colour, width: int) -> None:
    dash, gap = DASH * SUPERSAMPLE, DASH_GAP * SUPERSAMPLE
    # `offset` carries the dash pattern across the corners of the line.
    offset = 0.0
    for start, end in zip(points[:-1], points[1:], strict=True):
        length = float(np.linalg.norm(end - start))
        if length < 1e-9:
            continue
        direction = (end - start) / length
        position = -offset
        while position < length:
            a, b = max(position, 0.0), min(position + dash, length)
            if b > a:
                p, q = start + direction * a, start + direction * b
                draw.line([tuple(p), tuple(q)], fill=colour, width=width)
            position += dash + gap
        offset = (length + offset) % (dash + gap)


# Draws a dimension line with end ticks and its number, either horizontal or vertical.
def _draw_dimension(draw, font, start, end, value: float) -> None:
    width = round(DIMENSION_WIDTH * SUPERSAMPLE)
    tick = 6 * SUPERSAMPLE
    text = format_dimension(value)
    draw.line([start, end], fill=BLUE, width=width)
    if abs(start[1] - end[1]) < 1e-6:
        # Horizontal: ticks point up and down, and the number sits below the line.
        for x in (start[0], end[0]):
            draw.line([(x, start[1] - tick), (x, start[1] + tick)], fill=BLUE, width=width)
        anchor, position = "mt", ((start[0] + end[0]) / 2, start[1] + 9 * SUPERSAMPLE)
    else:
        # Vertical: ticks point left and right, and the number sits to the left of the line.
        for y in (start[1], end[1]):
            draw.line([(start[0] - tick, y), (start[0] + tick, y)], fill=BLUE, width=width)
        anchor, position = "rm", (start[0] - 10 * SUPERSAMPLE, (start[1] + end[1]) / 2)
    draw.text(position, text, fill=BLUE, font=font, anchor=anchor)


# Draws one view's edges on the image, given where its lower-left corner goes.
def _draw_view(draw, view: View, origin, low, scale: float) -> None:
    # Maps a point in centimetres to a pixel; the paper's y axis points down, so it is flipped.
    def to_pixels(points: np.ndarray) -> np.ndarray:
        shifted = (points - np.array(low)) * scale
        return np.column_stack([origin[0] + shifted[:, 0], origin[1] - shifted[:, 1]])

    # Hidden edges go first so that visible edges are drawn over them.
    for line in view.hidden:
        _draw_dashed(draw, to_pixels(line), GREY, round(HIDDEN_WIDTH * SUPERSAMPLE))
    for line in view.visible:
        pixels = [tuple(p) for p in to_pixels(line)]
        draw.line(pixels, fill=BLACK, width=round(VISIBLE_WIDTH * SUPERSAMPLE), joint="curve")


# Renders the full drawing of a solid: three views, their labels and the overall dimensions.
def render_drawing(shape: cq.Shape, size: int = IMAGE_SIZE) -> Image.Image:
    box = shape.BoundingBox()
    width, depth, height = box.xlen, box.ylen, box.zlen
    if min(width, depth, height) <= 0.0:
        raise ValueError("solid has no thickness in one direction")

    # Project the part from the front, the top and the right.
    tolerance = max(width, depth, height) * CURVE_TOLERANCE
    views = {name: project(shape, eye, right, tolerance) for name, (eye, right) in VIEWS.items()}

    # One scale for all three views, so that they line up and sizes can be compared.
    s = SUPERSAMPLE
    usable_w = size - MARGIN_LEFT - MARGIN_RIGHT - VIEW_GAP
    usable_h = size - MARGIN_TOP - MARGIN_BOTTOM - 2 * VIEW_GAP
    scale = min(usable_w / (width + depth), usable_h / (depth + height)) * s

    # Centre the block of views in the space left after the margins.
    block_w = (width + depth) * scale + VIEW_GAP * s
    block_h = (depth + height) * scale + VIEW_GAP * s
    left = MARGIN_LEFT * s + ((size - MARGIN_LEFT - MARGIN_RIGHT) * s - block_w) / 2
    top = (MARGIN_TOP + VIEW_GAP / 2) * s
    top += ((size - MARGIN_TOP - MARGIN_BOTTOM - VIEW_GAP / 2) * s - block_h) / 2

    # Standard layout: top view above the front view, right view beside the front view.
    top_bottom = top + depth * scale
    front_bottom = top_bottom + VIEW_GAP * s + height * scale
    right_left = left + width * scale + VIEW_GAP * s

    image = Image.new("RGB", (size * s, size * s), WHITE)
    draw = ImageDraw.Draw(image)
    _draw_view(draw, views["top"], (left, top_bottom), (box.xmin, box.ymin), scale)
    _draw_view(draw, views["front"], (left, front_bottom), (box.xmin, box.zmin), scale)
    _draw_view(draw, views["right"], (right_left, front_bottom), (box.ymin, box.zmin), scale)

    # A small label above each view says which one it is.
    label_font = ImageFont.load_default(size=13 * s)
    for name, x, y in (
        ("TOP", left, top_bottom - depth * scale),
        ("FRONT", left, front_bottom - height * scale),
        ("RIGHT", right_left, front_bottom - height * scale),
    ):
        draw.text((x, y - 8 * s), name, fill=GREY, font=label_font, anchor="ls")

    # Overall width, height and depth; without them the part's real size cannot be known.
    number_font = ImageFont.load_default(size=18 * s)
    below, beside = front_bottom + 26 * s, left - 26 * s
    _draw_dimension(draw, number_font, (left, below), (left + width * scale, below), width)
    _draw_dimension(
        draw, number_font, (beside, front_bottom), (beside, front_bottom - height * scale), height
    )
    _draw_dimension(
        draw, number_font, (beside, top_bottom), (beside, top_bottom - depth * scale), depth
    )

    # Shrink to the final size; averaging the extra pixels gives smooth lines.
    return image.resize((size, size), Image.LANCZOS)


# Renders a solid's drawing and saves it as a PNG file.
def save_drawing(shape: cq.Shape, path: str | Path, size: int = IMAGE_SIZE) -> None:
    render_drawing(shape, size).save(path)
