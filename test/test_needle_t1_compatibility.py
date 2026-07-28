# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Static and OpenUSD qualification gates for the inactive T1 needle candidate."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path

import pytest
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data/Props/SurgicalClosure/NeedleT1Compatibility"
USD_PATH = ASSET_ROOT / "dranmar_needle_t1_compatibility.usda"
MANIFEST_PATH = ASSET_ROOT / "asset_manifest.json"
GENERATOR_PATH = ROOT / "tools/generate_needle_t1_compatibility.py"
ROOT_PRIM_PATH = "/DrAnmarNeedleT1Compatibility"
RENDER_PRIM_PATH = f"{ROOT_PRIM_PATH}/Geometry/Render"
EXPECTED_TIP_M = (0.02004, -0.019154, 0.0)
EXPECTED_TAIL_M = (0.02004, 0.019154, 0.0)
EXPECTED_SOURCE_ASSETS = {
    "data/Props/Surgical_needle/needle.usd",
    "data/Props/Surgical_needle/needle_sdf.usd",
    "data/Props/SurgicalClosure/Needle/dranmar_needle.usda",
}
NVIDIA_VENDOR_SUBPATH = Path("vendor/nvidia_physicalai_simready_materials_v0_2_0")
NVIDIA_VENDOR_HASHES = {
    "LICENSE.md": ("18f74283f08ff1ed39a9c46dbe2622146d45f771023c3dbd9c631bb058e1421b"),
    "open_pbr_uber_base_class.usda": (
        "bb76ff9fa9cd74b86b6be4ed3c6ed79cdca15eff6d603ca571bdf9ce21e10c5f"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((ASSET_ROOT / name).read_text(encoding="utf-8"))


def _stage() -> Usd.Stage:
    stage = Usd.Stage.Open(str(USD_PATH))
    assert stage is not None
    return stage


def _close_vector(
    actual,
    expected,
    *,
    absolute_tolerance: float = 1.0e-10,
) -> None:
    assert len(actual) == len(expected)
    for observed, target in zip(actual, expected, strict=True):
        assert float(observed) == pytest.approx(
            float(target),
            abs=absolute_tolerance,
        )


def test_manifest_is_complete_hash_locked_and_preserves_source_assets():
    manifest = _json("asset_manifest.json")
    expected_members = {
        path.relative_to(ASSET_ROOT).as_posix()
        for path in ASSET_ROOT.rglob("*")
        if path.is_file() and path.name != MANIFEST_PATH.name
    }
    assert set(manifest["members"]) == expected_members
    assert manifest["status"] == "inactive_qualification_candidate"
    assert manifest["active_replacement"] is False
    assert manifest["clinical_validation"] is False
    assert manifest["dependency_complete_directory"] is True
    assert manifest["provenance"]["vendor_assets_modified"] is False
    assert manifest["license"] == "mixed"
    assert set(manifest["licenses"].values()) == {"Apache-2.0", "MIT-0"}
    assert manifest["generator"]["texture_size_px"] == 2048
    assert manifest["material_contexts"]["primary"].startswith("OpenPBR_1.1")
    assert manifest["material_contexts"]["fallback"] == "UsdPreviewSurface"
    assert manifest["material_contexts"]["mdl_only_claim"] is False
    assert manifest["quaternion_contract"]["runtime_and_JSON_canonical"] == "xyzw"
    assert manifest["quaternion_contract"]["openusd_quat_serialization"] == "wxyz"
    assert _sha256(GENERATOR_PATH) == manifest["generator"]["sha256"]
    for relative, values in manifest["members"].items():
        member = ASSET_ROOT / relative
        assert member.stat().st_size == values["bytes"]
        assert _sha256(member) == values["sha256"]
        if relative in {(NVIDIA_VENDOR_SUBPATH / name).as_posix() for name in NVIDIA_VENDOR_HASHES}:
            assert values["license"] == "MIT-0"
            assert values["provenance"].startswith("verbatim_NVIDIA")
        elif relative == "LICENSE.txt":
            assert values["license"] == "Apache-2.0"
            assert values["provenance"] == "canonical_Apache-2.0_text"
        else:
            assert values["license"] == "Apache-2.0"
            assert values["provenance"] == "independently_generated_by_DrAnmar"

    preserved = manifest["preserved_source_assets"]
    assert set(preserved) == EXPECTED_SOURCE_ASSETS
    for relative, values in preserved.items():
        source = ROOT / relative
        assert values["modified_by_generator"] is False
        assert source.stat().st_size == values["bytes"]
        assert _sha256(source) == values["sha256"]

    dependency = manifest["vendor_dependencies"]["nvidia_physicalai_simready_materials"]
    assert dependency["release"] == "v0.2.0"
    assert dependency["license"] == "MIT-0"
    source_root = ROOT / dependency["source_root"]
    destination_root = ASSET_ROOT / dependency["destination_root"]
    for name, expected_hash in NVIDIA_VENDOR_HASHES.items():
        assert dependency["members"][name]["sha256"] == expected_hash
        assert _sha256(source_root / name) == expected_hash
        assert _sha256(destination_root / name) == expected_hash
        assert (source_root / name).read_bytes() == (destination_root / name).read_bytes()


def test_contract_preserves_the_legacy_envelope_and_fails_closed():
    geometry = _json("geometry_contract.json")
    physics = _json("physics_profile.json")
    frames = _json("interaction_frames.json")
    legacy = geometry["legacy_envelope"]
    _close_vector(legacy["circle_center_m"], (0.02004, 0.0, 0.0))
    _close_vector(legacy["tip_position_m"], EXPECTED_TIP_M)
    _close_vector(legacy["tail_position_m"], EXPECTED_TAIL_M)
    assert legacy["centerline_radius_m"] == pytest.approx(0.019154)
    assert legacy["centerline_arc_radians"] == pytest.approx(math.pi)
    spawn = geometry["runtime_spawn_contract"]
    assert spawn["candidate_scale_xyz"] == [1.0, 1.0, 1.0]
    assert spawn["candidate_root_orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert spawn["legacy_active_scale_xyz"] == [0.4, 0.4, 0.4]
    assert spawn["path_only_substitution_allowed"] is False
    assert spawn["promotion_requires_composed_world_space_frame_parity"] is True
    render = geometry["render_geometry"]
    assert render["shape"] == "single_connected_watertight_half_circle_taper_point"
    assert render["body_diameter_m"] == pytest.approx(0.00165)
    assert 0.0 < render["tip_curvature_radius_m"] < render["body_diameter_m"] / 2.0
    assert render["arc_resolution"] >= 384
    assert render["radial_resolution"] >= 32
    assert render["uv_interpolation"] == "faceVarying"
    assert geometry["active_replacement"] is False
    assert physics["status"] == "unqualified_candidate"
    assert physics["active_replacement"] is False
    assert "legacy_analytic_pickup_parity" in physics["qualification_required"]
    assert "two_arm_handover_parity" in physics["qualification_required"]
    assert "mid_air_transport_retention_parity" in physics["qualification_required"]
    _close_vector(frames["needle_tip"]["position_m"], EXPECTED_TIP_M)
    _close_vector(frames["needle_tail"]["position_m"], EXPECTED_TAIL_M)
    convention = frames["coordinate_convention"]
    assert convention["runtime_quaternion_order"] == "xyzw"
    assert convention["runtime_orientation_field"] == "orientation_xyzw"
    assert convention["openusd_quaternion_serialization"] == "wxyz"
    assert convention["openusd_orientation_field"] == "orientation_usd_wxyz"
    for name in (
        "needle_tip",
        "needle_driver_grasp",
        "needle_plane",
        "needle_tail",
    ):
        frame = frames[name]
        assert "orientation_wxyz" not in frame
        xyzw = frame["orientation_xyzw"]
        usd_wxyz = frame["orientation_usd_wxyz"]
        _close_vector(xyzw, (*usd_wxyz[1:], usd_wxyz[0]))
        assert math.sqrt(sum(value * value for value in xyzw)) == pytest.approx(1.0)
    _close_vector(
        frames["needle_tip"]["orientation_xyzw"],
        (0.0, 0.0, 0.0, 1.0),
    )
    _close_vector(
        frames["needle_tip"]["orientation_usd_wxyz"],
        (1.0, 0.0, 0.0, 0.0),
    )

    mass = physics["mass_properties"]
    assert "principal_axes_wxyz" not in mass
    assert mass["principal_axes_convention"]["runtime"] == "xyzw"
    assert mass["principal_axes_convention"]["openusd_serialization"] == "wxyz"
    _close_vector(
        mass["principal_axes_xyzw"],
        (
            *mass["principal_axes_usd_wxyz"][1:],
            mass["principal_axes_usd_wxyz"][0],
        ),
    )


def test_authored_mesh_is_actually_connected_watertight_and_smooth():
    stage = _stage()
    mesh = UsdGeom.Mesh.Get(stage, RENDER_PRIM_PATH)
    assert mesh
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    normals = mesh.GetNormalsAttr().Get()
    uvs = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st").Get()
    assert len(points) >= 12_000
    assert len(counts) >= 25_000
    assert set(counts) == {3}
    assert len(indices) == 3 * len(counts)
    assert len(uvs) == len(indices)
    assert len(normals) == len(points)
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
    assert all(Gf.IsClose(float(Gf.Vec3d(value).GetLength()), 1.0, 2.0e-6) for value in normals)
    _close_vector(points[0], EXPECTED_TIP_M, absolute_tolerance=2.0e-9)
    _close_vector(points[-1], EXPECTED_TAIL_M, absolute_tolerance=2.0e-9)
    assert float(normals[0][0]) > 0.999

    radial_resolution = 32
    assert (len(points) - 2) % radial_resolution == 0
    ring_count = (len(points) - 2) // radial_resolution
    ring_radii = []
    for ring_index in range(ring_count):
        start = 1 + ring_index * radial_resolution
        ring = [Gf.Vec3d(point) for point in points[start : start + radial_resolution]]
        centroid = sum(ring, Gf.Vec3d()) / radial_resolution
        circle_offset = centroid - Gf.Vec3d(0.02004, 0.0, 0.0)
        assert circle_offset.GetLength() == pytest.approx(0.019154, abs=3.0e-8)
        section_radii = [(point - centroid).GetLength() for point in ring]
        assert max(section_radii) - min(section_radii) < 3.0e-8
        ring_radii.append(sum(section_radii) / radial_resolution)
    assert 0.0 < ring_radii[0] < 0.000025
    assert max(ring_radii) == pytest.approx(0.000825, abs=3.0e-8)
    assert max(ring_radii) <= 0.000825 + 3.0e-8
    expected_full_body_volume = math.pi * 0.000825**2 * (math.pi * 0.019154)

    edges: Counter[tuple[int, int]] = Counter()
    adjacency = {index: set() for index in range(len(points))}
    for start in range(0, len(indices), 3):
        triangle = tuple(int(value) for value in indices[start : start + 3])
        assert len(set(triangle)) == 3
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edges[tuple(sorted((left, right)))] += 1
            adjacency[left].add(right)
            adjacency[right].add(left)
    assert edges
    assert set(edges.values()) == {2}
    unseen = set(adjacency)
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            for neighbor in adjacency[queue.popleft()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    assert components == 1

    report = _json("geometry_report.json")
    topology = report["topology"]
    assert topology["watertight"] is True
    assert topology["boundary_edge_count"] == 0
    assert topology["nonmanifold_edge_count"] == 0
    assert topology["connected_component_count"] == 1
    assert topology["signed_volume_m3"] > 0.0
    assert topology["signed_volume_m3"] == pytest.approx(
        expected_full_body_volume,
        rel=0.08,
    )
    assert topology["vertex_count"] == len(points)
    assert topology["triangle_count"] == len(counts)


def test_usd_separates_render_and_collision_and_authors_mass_and_ccd():
    stage = _stage()
    root = stage.GetDefaultPrim()
    assert str(root.GetPath()) == ROOT_PRIM_PATH
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root.HasAPI(UsdPhysics.MassAPI)
    assert root.GetAttribute("physxRigidBody:enableCCD").Get() is True
    assert root.GetAttribute("physxRigidBody:enableSpeculativeCCD").Get() is True
    assert root.GetCustomDataByKey("drAnmarActiveReplacement") is False

    render = stage.GetPrimAtPath(RENDER_PRIM_PATH)
    assert render.IsValid()
    assert not render.HasAPI(UsdPhysics.CollisionAPI)
    assert not render.HasAttribute("physics:collisionEnabled")
    assert render.GetAttribute("purpose").Get() == UsdGeom.Tokens.render

    capsule_prims = sorted(
        (
            prim
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Capsule)
            and str(prim.GetPath()).startswith(f"{ROOT_PRIM_PATH}/Collision/C")
        ),
        key=lambda prim: str(prim.GetPath()),
    )
    assert len(capsule_prims) == 96
    previous_end = None
    for prim in capsule_prims:
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert prim.GetAttribute("physics:collisionEnabled").Get() is True
        assert prim.GetAttribute("visibility").Get() == UsdGeom.Tokens.invisible
        assert prim.GetAttribute("purpose").Get() == UsdGeom.Tokens.guide
        capsule = UsdGeom.Capsule(prim)
        height = float(capsule.GetHeightAttr().Get())
        radius = float(capsule.GetRadiusAttr().Get())
        assert height > 0.0
        assert radius > 0.0
        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        midpoint = transform.ExtractTranslation()
        axis = transform.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized()
        start = midpoint - axis * (height / 2.0)
        end = midpoint + axis * (height / 2.0)
        if previous_end is not None:
            assert Gf.IsClose(start, previous_end, 5.0e-10)
        previous_end = end

    profile = _json("physics_profile.json")
    mass = profile["mass_properties"]
    assert mass["source"].startswith("actual_connected_watertight_render_mesh")
    assert mass["density_kg_m3"] == pytest.approx(8000.0)
    assert mass["mass_kg"] > 0.0
    assert all(value > 0.0 for value in mass["diagonal_inertia_kg_m2"])
    assert float(root.GetAttribute("physics:mass").Get()) == pytest.approx(
        mass["mass_kg"],
        rel=2.0e-6,
    )
    _close_vector(
        root.GetAttribute("physics:centerOfMass").Get(),
        mass["center_of_mass_m"],
        absolute_tolerance=2.0e-9,
    )
    principal_axes = root.GetAttribute("physics:principalAxes").Get()
    _close_vector(
        (
            principal_axes.GetReal(),
            *principal_axes.GetImaginary(),
        ),
        mass["principal_axes_usd_wxyz"],
        absolute_tolerance=2.0e-7,
    )

    frames = _json("interaction_frames.json")
    for json_name, usd_name in (
        ("needle_tip", "Tip"),
        ("needle_driver_grasp", "DriverGrasp"),
        ("needle_plane", "NeedlePlane"),
        ("needle_tail", "Tail"),
    ):
        usd_frame = stage.GetPrimAtPath(f"{ROOT_PRIM_PATH}/Frames/{usd_name}")
        orientation = usd_frame.GetAttribute("xformOp:orient").Get()
        _close_vector(
            (
                orientation.GetReal(),
                *orientation.GetImaginary(),
            ),
            frames[json_name]["orientation_usd_wxyz"],
            absolute_tolerance=2.0e-10,
        )


def test_materials_textures_and_all_local_usd_dependencies_resolve():
    stage = _stage()
    material = stage.GetPrimAtPath(f"{ROOT_PRIM_PATH}/Looks/SatinSteel")
    assert material.IsValid()
    assert [str(path) for path in material.GetInherits().GetAllDirectInherits()] == [
        "/open_pbr_uber_base"
    ]
    openpbr_base = stage.GetPrimAtPath("/open_pbr_uber_base")
    assert openpbr_base.IsValid()
    assert openpbr_base.GetAttribute("semantics:labels:OpenPBR_Ver").Get() == ["1.1"]
    assert material.GetAttribute("outputs:mtlx:surface").GetConnections() == [
        material.GetPath().AppendPath("open_pbr_surface_surfaceshader.outputs:out")
    ]
    assert material.GetAttribute("drAnmarPrimaryMaterialContext").Get() == ("OpenPBR_1_1_MaterialX")
    assert material.GetAttribute("drAnmarFallbackMaterialContext").Get() == ("UsdPreviewSurface")
    assert material.GetAttribute("drAnmarNativeRTXAppearanceQualified").Get() is False
    preview = stage.GetPrimAtPath(f"{material.GetPath()}/PreviewSurface")
    assert preview.GetAttribute("inputs:metallic").Get() >= 0.9
    assert preview.GetAttribute("inputs:clearcoat").Get() == 0.0
    for shader_name, openpbr_input, file_name, color_space in (
        (
            "BaseColor",
            "inputs:base_color_texture_file",
            "needle_satin_basecolor.png",
            "sRGB",
        ),
        (
            "Roughness",
            "inputs:specular_roughness_texture_file",
            "needle_satin_roughness.png",
            "raw",
        ),
        (
            "Normal",
            "inputs:geometry_normal_texture_file",
            "needle_satin_normal.png",
            "raw",
        ),
    ):
        openpbr_texture = material.GetAttribute(openpbr_input)
        assert openpbr_texture.IsValid()
        assert openpbr_texture.GetMetadata("colorSpace") == color_space
        assert Path(openpbr_texture.Get().resolvedPath) == (ASSET_ROOT / "textures" / file_name)
        shader = stage.GetPrimAtPath(f"{material.GetPath()}/{shader_name}")
        texture = shader.GetAttribute("inputs:file").Get()
        assert Path(texture.resolvedPath) == ASSET_ROOT / "textures" / file_name
        assert Path(texture.resolvedPath).is_file()
        assert shader.GetAttribute("inputs:sourceColorSpace").Get() == color_space
        image = Image.open(texture.resolvedPath)
        assert image.size == (2048, 2048)
        assert image.mode == ("L" if shader_name == "Roughness" else "RGB")
        extrema = image.getextrema()
        if image.mode == "L":
            assert extrema[0] < extrema[1]
            assert 115 <= extrema[0]
            assert extrema[1] <= 164
        else:
            assert any(low < high for low, high in extrema)
    omni = stage.GetPrimAtPath(f"{material.GetPath()}/OmniSurface")
    assert not omni.IsValid()
    usd_text = USD_PATH.read_text(encoding="utf-8")
    assert "OmniSurface" not in usd_text
    assert "outputs:mdl:surface" not in usd_text
    assert "inputs:base_color_texture_file" in usd_text
    assert "inputs:specular_roughness_texture_file" in usd_text
    assert "inputs:geometry_normal_texture_file" in usd_text

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(USD_PATH))
    assert len(layers) == 2
    assert not unresolved
    for asset in assets:
        value = str(asset)
        assert Path(value).is_file(), value


def test_candidate_is_not_referenced_by_active_runtime_or_task_python():
    searched_roots = (
        ROOT / "orbit",
        ROOT.parent / "orbit.surgical.tasks",
    )
    offenders = []
    for searched_root in searched_roots:
        if not searched_root.is_dir():
            continue
        for path in searched_root.rglob("*.py"):
            if "NeedleT1Compatibility" in path.read_text(encoding="utf-8"):
                offenders.append(path)
    assert not offenders


def test_generator_reproduces_every_packaged_byte(tmp_path: Path):
    generated = tmp_path / "NeedleT1Compatibility"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR_PATH),
            "--output-root",
            str(generated),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected_paths = {
        path.relative_to(ASSET_ROOT) for path in ASSET_ROOT.rglob("*") if path.is_file()
    }
    actual_paths = {path.relative_to(generated) for path in generated.rglob("*") if path.is_file()}
    assert actual_paths == expected_paths
    for relative in expected_paths:
        assert _sha256(generated / relative) == _sha256(ASSET_ROOT / relative)
