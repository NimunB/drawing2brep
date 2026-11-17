"""The converter turns recorded build steps into programs that build the right solid."""

# These tests use small hand-made designs, so they run without the dataset.

import math

import pytest

from d2b.data.to_cadquery import UnsupportedError, convert
from d2b.data.verify import run_program
from d2b.eval.geometry import to_shape


# Helper: a JSON point in the dataset's format.
def point(x, y, z=0.0):
    return {"type": "Point3D", "x": x, "y": y, "z": z}


# Helper: a JSON straight line between two sketch points.
def line(start, end):
    return {"type": "Line3D", "start_point": point(*start), "end_point": point(*end)}


# Helper: a JSON full circle on the sketch plane.
def circle(centre, radius):
    return {"type": "Circle3D", "center_point": point(*centre), "radius": radius}


# Helper: a JSON sketch on the flat XY plane holding the given profiles.
def sketch(profiles):
    transform = {
        "origin": point(0, 0),
        "x_axis": point(1, 0),
        "y_axis": point(0, 1),
        "z_axis": point(0, 0, 1),
    }
    return {"type": "Sketch", "name": "Sketch1", "transform": transform, "profiles": profiles}


# Helper: a JSON extrude of one profile by a distance.
def extrude(sketch_id, profile_id, distance, operation="NewBodyFeatureOperation", **overrides):
    entity = {
        "type": "ExtrudeFeature",
        "name": "Extrude1",
        "profiles": [{"sketch": sketch_id, "profile": profile_id}],
        "operation": operation,
        "start_extent": {"type": "ProfilePlaneStartDefinition"},
        "extent_type": "OneSideFeatureExtentType",
        "extent_one": {
            "type": "DistanceExtentDefinition",
            "distance": {"value": distance},
            "taper_angle": {"value": 0.0},
        },
    }
    return entity | overrides


# Helper: wraps entities into a design with a timeline in the given order.
def design(entities):
    timeline = [{"index": i, "entity": key} for i, key in enumerate(entities)]
    return {"timeline": timeline, "entities": entities}


# Helper: a 4 x 2 rectangle whose lines are deliberately shuffled and one is reversed.
def shuffled_rectangle():
    return [
        line((4, 2), (0, 2)),
        line((0, 0), (4, 0)),
        line((0, 0), (0, 2)),
        line((4, 0), (4, 2)),
    ]


# Helper: converts a design, runs the program, and returns the solid's volume.
def built_volume(entities):
    return to_shape(run_program(convert(design(entities)))).Volume()


# A plate with a round hole converts to a program with the right volume.
def test_plate_with_hole():
    loops = [
        {"is_outer": True, "profile_curves": shuffled_rectangle()},
        {"is_outer": False, "profile_curves": [circle((2, 1), 0.5)]},
    ]
    entities = {"s": sketch({"p": {"loops": loops}}), "e": extrude("s", "p", 3.0)}
    # Area (8 minus the hole) times the extrude distance.
    assert built_volume(entities) == pytest.approx((8 - math.pi * 0.25) * 3.0)


# A negative distance extrudes the other way and still gives the same volume.
def test_negative_distance():
    loops = [{"is_outer": True, "profile_curves": shuffled_rectangle()}]
    entities = {"s": sketch({"p": {"loops": loops}}), "e": extrude("s", "p", -3.0)}
    assert built_volume(entities) == pytest.approx(24.0)


# A second extrude marked as a cut removes material from the first.
def test_cut_removes_material():
    plate = {"loops": [{"is_outer": True, "profile_curves": shuffled_rectangle()}]}
    hole = {"loops": [{"is_outer": True, "profile_curves": [circle((2, 1), 0.5)]}]}
    entities = {
        "s": sketch({"plate": plate, "hole": hole}),
        "e1": extrude("s", "plate", 3.0),
        "e2": extrude("s", "hole", 3.0, operation="CutFeatureOperation"),
    }
    assert built_volume(entities) == pytest.approx((8 - math.pi * 0.25) * 3.0)


# The program is plain text that starts by importing CadQuery and defines `result`.
def test_program_is_readable_text():
    loops = [{"is_outer": True, "profile_curves": shuffled_rectangle()}]
    code = convert(design({"s": sketch({"p": {"loops": loops}}), "e": extrude("s", "p", 3.0)}))
    assert code.startswith("import cadquery as cq")
    assert "result = part_1" in code
    assert ".extrude(3.0)" in code
    # Steps are numbered, and the dataset's own step name ("Extrude1") is left out.
    assert "# Step 1: new body" in code
    assert "Extrude1" not in code


# Helper: builds the solid for a plain rectangle extruded with the given settings.
def rectangle_solid(**overrides):
    loops = [{"is_outer": True, "profile_curves": shuffled_rectangle()}]
    entities = {"s": sketch({"p": {"loops": loops}}), "e": extrude("s", "p", 3.0, **overrides)}
    return to_shape(run_program(convert(design(entities))))


# A symmetric extrude spreads its full length equally on both sides of the sketch.
def test_symmetric_extrude():
    extent = {
        "type": "SymmetricExtentDefinition",
        "distance": {"value": 3.0},
        "taper_angle": {"value": 0.0},
        "is_full_length": True,
    }
    solid = rectangle_solid(extent_type="SymmetricFeatureExtentType", extent_one=extent)
    box = solid.BoundingBox()
    assert (box.zmin, box.zmax) == pytest.approx((-1.5, 1.5))


# A two-sided extrude goes a different distance each way from the sketch.
def test_two_sided_extrude():
    second = {
        "type": "DistanceExtentDefinition",
        "distance": {"value": 1.0},
        "taper_angle": {"value": 0.0},
    }
    solid = rectangle_solid(extent_type="TwoSidesFeatureExtentType", extent_two=second)
    box = solid.BoundingBox()
    # 3 forward (from the helper) and 1 backward.
    assert (box.zmin, box.zmax) == pytest.approx((-1.0, 3.0))


# An offset start moves the whole extrude away from the sketch plane.
def test_offset_start():
    start = {"type": "OffsetStartDefinition", "offset": {"value": 8.0}}
    box = rectangle_solid(start_extent=start).BoundingBox()
    assert (box.zmin, box.zmax) == pytest.approx((8.0, 11.0))


# A tapered extrude has sloped sides, so it holds less volume than a straight one.
def test_tapered_extrude_narrows():
    tapered = {
        "type": "DistanceExtentDefinition",
        "distance": {"value": 3.0},
        # Fusion 360 records a narrowing taper as a negative angle, in radians.
        "taper_angle": {"value": -0.1},
    }
    solid = rectangle_solid(extent_one=tapered)
    assert solid.isValid()
    # A straight extrude of the 4 x 2 rectangle by 3 would be exactly 24.
    assert 0.0 < solid.Volume() < 24.0


# A feature the converter cannot translate is reported, not silently guessed.
def test_unsupported_feature_is_reported():
    # Taper on a symmetric extrude is not handled yet.
    extent = {
        "type": "SymmetricExtentDefinition",
        "distance": {"value": 3.0},
        "taper_angle": {"value": 0.1},
        "is_full_length": True,
    }
    loops = [{"is_outer": True, "profile_curves": shuffled_rectangle()}]
    symmetric = extrude("s", "p", 3.0, extent_type="SymmetricFeatureExtentType", extent_one=extent)
    with pytest.raises(UnsupportedError, match="tapered"):
        convert(design({"s": sketch({"p": {"loops": loops}}), "e": symmetric}))


# A loop with a gap in it is rejected.
def test_open_loop_is_rejected():
    gap = [line((0, 0), (4, 0)), line((4, 0), (4, 2)), line((4, 2), (0, 2))]
    loops = [{"is_outer": True, "profile_curves": gap}]
    with pytest.raises(UnsupportedError, match="not closed"):
        convert(design({"s": sketch({"p": {"loops": loops}}), "e": extrude("s", "p", 3.0)}))
