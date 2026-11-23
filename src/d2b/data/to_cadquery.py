"""Convert one Fusion 360 Gallery design (its JSON build steps) into a CadQuery program."""

# The output is plain Python text. Running it defines a variable called `result`.

from __future__ import annotations

import math

# Two points closer than this (in cm) are treated as the same point when joining curves.
JOIN_TOLERANCE = 1e-4

# Decimal places kept for every number written into the program.
DECIMALS = 4

# The Fusion 360 name of each extrude operation, and the comment we write for it.
OPERATIONS = {
    "NewBodyFeatureOperation": "new body",
    "JoinFeatureOperation": "join",
    "CutFeatureOperation": "cut",
    "IntersectFeatureOperation": "intersect",
}


# Raised when a design uses something the converter cannot translate yet.
class UnsupportedError(Exception):
    pass


# Formats one number for the program: rounded, and never "-0.0".
def _num(value: float) -> str:
    return repr(round(value, DECIMALS) + 0.0)


# Formats a 2D or 3D point as "(x, y)" or "(x, y, z)".
def _tuple(values) -> str:
    return "(" + ", ".join(_num(v) for v in values) + ")"


# Reads a JSON point or vector as a plain (x, y, z) tuple.
def _xyz(point: dict) -> tuple[float, float, float]:
    return (point["x"], point["y"], point["z"])


# Reads a JSON point that lies on the sketch plane as a 2D (x, y) pair.
def _xy(point: dict) -> tuple[float, float]:
    # Sketch geometry is flat, so z must be zero in the sketch's own coordinates.
    if abs(point["z"]) > JOIN_TOLERANCE:
        raise UnsupportedError("curve off sketch plane")
    return (point["x"], point["y"])


# Finds the point halfway along an arc, which CadQuery needs to draw the arc.
def _arc_midpoint(arc: dict) -> tuple[float, float]:
    cx, cy, cz = _xyz(arc["center_point"])
    nx, ny, nz = _xyz(arc["normal"])
    rx, ry, rz = _xyz(arc["reference_vector"])
    # Angles are measured from the reference vector, turning around the normal.
    ux, uy, uz = (ny * rz - nz * ry, nz * rx - nx * rz, nx * ry - ny * rx)
    angle = (arc["start_angle"] + arc["end_angle"]) / 2.0
    r, c, s = arc["radius"], math.cos(angle), math.sin(angle)
    mid = {"x": cx + r * (c * rx + s * ux), "y": cy + r * (c * ry + s * uy)}
    mid["z"] = cz + r * (c * rz + s * uz)
    return _xy(mid)


# Turns one line or arc from the JSON into a segment: (start, end, arc midpoint or None).
def _segment(curve: dict):
    kind = curve["type"]
    if kind == "Line3D":
        return (_xy(curve["start_point"]), _xy(curve["end_point"]), None)
    if kind == "Arc3D":
        return (_xy(curve["start_point"]), _xy(curve["end_point"]), _arc_midpoint(curve))
    # Splines and ellipses are not handled yet.
    raise UnsupportedError(f"curve type {kind}")


# True when two 2D points are close enough to count as the same point.
def _same(p, q) -> bool:
    return math.dist(p, q) <= JOIN_TOLERANCE


# Puts a loop's segments in drawing order, so each one starts where the last one ended.
def _chain(segments: list) -> list:
    # The JSON lists a loop's curves in no particular order or direction.
    ordered = [segments[0]]
    remaining = segments[1:]
    while remaining:
        tip = ordered[-1][1]
        for i, (start, end, mid) in enumerate(remaining):
            if _same(start, tip):
                ordered.append(remaining.pop(i))
                break
            # The segment fits if flipped; an arc's midpoint stays the same when flipped.
            if _same(end, tip):
                remaining.pop(i)
                ordered.append((end, start, mid))
                break
        else:
            raise UnsupportedError("loop does not connect")
    # The last segment must return to where the first one started.
    if not _same(ordered[-1][1], ordered[0][0]):
        raise UnsupportedError("loop is not closed")
    return ordered


# Writes the CadQuery calls that draw one closed loop (an outline or a hole).
def _loop_code(loop: dict) -> str:
    curves = loop["profile_curves"]
    # A full circle is a loop on its own.
    if len(curves) == 1 and curves[0]["type"] == "Circle3D":
        centre = _xy(curves[0]["center_point"])
        return f".moveTo{_tuple(centre)}.circle({_num(curves[0]['radius'])})"
    if any(c["type"] == "Circle3D" for c in curves):
        raise UnsupportedError("circle mixed with other curves in a loop")

    ordered = _chain([_segment(c) for c in curves])
    # A loop of straight lines only is written compactly as a list of corners.
    if all(mid is None for _, _, mid in ordered):
        corners = ", ".join(_tuple(start) for start, _, _ in ordered)
        return f".polyline([{corners}]).close()"

    code = f".moveTo{_tuple(ordered[0][0])}"
    for i, (_, end, mid) in enumerate(ordered):
        last = i == len(ordered) - 1
        if mid is not None:
            code += f".threePointArc({_tuple(mid)}, {_tuple(end)})"
        elif not last:
            code += f".lineTo{_tuple(end)}"
        # A final straight line is drawn by close(), so it is not written out.
    return code + ".close()"


# Writes the CadQuery calls for one profile, one loop per item: outer boundary, then holes.
def _profile_code(profile: dict) -> list[str]:
    loops = sorted(profile["loops"], key=lambda loop: not loop["is_outer"])
    if not loops or not loops[0]["is_outer"]:
        raise UnsupportedError("profile has no outer loop")
    # A closed shape drawn inside another becomes a hole when extruded.
    return [_loop_code(loop) for loop in loops]


# Finds the middle of the box that just encloses the finished part.
def design_centre(design: dict) -> tuple[float, float, float]:
    box = design["properties"]["bounding_box"]
    low, high = _xyz(box["min_point"]), _xyz(box["max_point"])
    # Rounded like every other number, so moving the planes adds no new rounding error.
    return tuple(round((a + b) / 2.0, DECIMALS) for a, b in zip(low, high, strict=True))


# Writes the line that defines where a sketch's flat plane sits in 3D.
def _plane_code(name: str, sketch: dict, centre: tuple[float, float, float]) -> str:
    t = sketch["transform"]
    origin, x_dir, normal = _xyz(t["origin"]), _xyz(t["x_axis"]), _xyz(t["z_axis"])
    # Moving every plane by the same amount moves the whole part without changing its shape.
    origin = tuple(o - c for o, c in zip(origin, centre, strict=True))
    return (
        f"{name} = cq.Plane(origin={_tuple(origin)}, xDir={_tuple(x_dir)}, normal={_tuple(normal)})"
    )


# Reads one side's extrude distance, rejecting taper where it is not handled.
def _side_distance(extent: dict) -> float:
    # A taper makes the sides slope; it is only handled for one-direction extrudes.
    if abs(extent["taper_angle"]["value"]) > 1e-9:
        raise UnsupportedError("tapered extrude")
    return extent["distance"]["value"]


# Works out where an extrude starts and how it is pulled; returns (plane shift, extrude call).
def _extrude_code(extrude: dict) -> tuple[str, str]:
    # An extrude can start on the sketch plane or at a distance from it.
    start = extrude["start_extent"]
    if start["type"] == "ProfilePlaneStartDefinition":
        shift = 0.0
    elif start["type"] == "OffsetStartDefinition":
        shift = start["offset"]["value"]
    else:
        raise UnsupportedError(f"start extent {start['type']}")

    kind = extrude["extent_type"]
    one = extrude["extent_one"]
    if kind == "OneSideFeatureExtentType":
        # Pulled one way only, along the plane's normal (a negative distance goes backwards).
        taper = math.degrees(one["taper_angle"]["value"])
        if abs(taper) > 1e-9:
            # Sloped sides. CadQuery measures the angle in degrees with the opposite sign.
            call = f".extrude({_num(one['distance']['value'])}, taper={_num(-taper)})"
        else:
            call = f".extrude({_num(_side_distance(one))})"
    elif kind == "SymmetricFeatureExtentType":
        # Pulled equally both ways. The recorded distance is either the total or one side.
        distance = _side_distance(one)
        half = distance / 2.0 if one.get("is_full_length") else distance
        call = f".extrude({_num(half)}, both=True)"
    elif kind == "TwoSidesFeatureExtentType":
        # Pulled both ways by different amounts: start behind the plane, pull the total.
        forward, backward = _side_distance(one), _side_distance(extrude["extent_two"])
        shift -= backward
        call = f".extrude({_num(forward + backward)})"
    else:
        raise UnsupportedError(f"extent type {kind}")

    # Shifting the plane along its normal moves where the extrude begins.
    plane_shift = f".workplane(offset={_num(shift)})" if abs(shift) > 1e-9 else ""
    return plane_shift, call


# Converts a whole design into the text of a CadQuery program, moved so `centre` is the origin.
def convert(design: dict, centre: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    entities = design["entities"]
    lines = ["import cadquery as cq", ""]
    # Maps each sketch's ID to the variable name of its plane, once written.
    planes: dict[str, str] = {}
    have_result = False
    count = 0

    # Walk the build steps in the order the designer made them.
    for step in design["timeline"]:
        extrude = entities[step["entity"]]
        # Sketches are only read when an extrude uses them.
        if extrude["type"] != "ExtrudeFeature":
            continue
        operation = extrude["operation"]
        if operation not in OPERATIONS:
            raise UnsupportedError(f"operation {operation}")
        if not extrude["profiles"]:
            raise UnsupportedError("extrude with no profiles")
        plane_shift, extrude_call = _extrude_code(extrude)
        count += 1
        part = f"part_{count}"
        # Steps are numbered; the designer's own step names are not copied into the program.
        lines.append(f"# Step {count}: {OPERATIONS[operation]}")

        # Build one solid per profile and merge them into this extrude's solid.
        for i, ref in enumerate(extrude["profiles"]):
            sketch = entities[ref["sketch"]]
            if ref["sketch"] not in planes:
                planes[ref["sketch"]] = f"plane_{len(planes) + 1}"
                lines.append(_plane_code(planes[ref["sketch"]], sketch, centre))
            loops = _profile_code(sketch["profiles"][ref["profile"]])
            plane = planes[ref["sketch"]]
            # One idea per line: the plane, then each loop, then the extrude.
            chain = [f"cq.Workplane({plane}){plane_shift}", *loops, extrude_call]
            lines.append(f"{part} = (" if i == 0 else f"{part} = {part}.union(")
            lines.extend(f"    {call}" for call in chain)
            lines.append(")")

        # Combine this extrude's solid with everything built so far.
        if not have_result:
            if operation in ("CutFeatureOperation", "IntersectFeatureOperation"):
                raise UnsupportedError("cut or intersect before any body exists")
            lines.append(f"result = {part}")
            have_result = True
        elif operation == "CutFeatureOperation":
            lines.append(f"result = result.cut({part})")
        elif operation == "IntersectFeatureOperation":
            lines.append(f"result = result.intersect({part})")
        else:
            # A new body and a join both add material.
            lines.append(f"result = result.union({part})")
        lines.append("")

    if not have_result:
        raise UnsupportedError("no extrudes")
    return "\n".join(lines)
