# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Static/OpenUSD gates for the inactive T1 collision-proxy candidates."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = ROOT.parents[2]
GENERATOR = ROOT / "tools/generate_t1_collider_candidates.py"
PSM_ROOT = ROOT / "data/Robots/dVRK/PSM/T1ColliderCandidate"
TABLE_ROOT = ROOT / "data/Props/Table/T1ColliderCandidate"
PSM_USD = PSM_ROOT / "psm_t1_collider_candidate.usda"
TABLE_USD = TABLE_ROOT / "table_t1_collider_candidate.usda"
PSM_BASE = ROOT / "data/Robots/dVRK/PSM/psm_col.usd"
PSM_VISUAL = ROOT / "data/Props/SurgicalScene/T1/psm_visual_v1.usda"
TABLE_BASE = ROOT / "data/Props/Table/table.usd"
TABLE_VISUAL = ROOT / "data/Props/SurgicalScene/T1/table_visual_v1.usda"

STATUS = "inactive_unqualified_candidate"
PSM_ASSET_ID = "dranmar-psm-t1-collider-candidate-v1"
TABLE_ASSET_ID = "dranmar-table-t1-collider-candidate-v1"
PSM_REFERENCE = "../../../../Props/SurgicalScene/T1/psm_visual_v1.usda"
TABLE_REFERENCE = "../../SurgicalScene/T1/table_visual_v1.usda"
PSM_BASE_SHA256 = (
    "87330b0e46e5554d53fe1b840f2db38a9bdbf00a79ca464159009406156ca0d4"
)
PSM_VISUAL_SHA256 = (
    "534b8072d9e05a82df7707124c5a88715176fba1c58231aa09d8d8af062c2e84"
)
TABLE_BASE_SHA256 = (
    "78f61603d0605896c9076d0a2b16bc97fd6f0109783a3b1e2dd1cfd86698b7c4"
)
TABLE_VISUAL_SHA256 = (
    "9d0eea19c3e01941e5119a2f2f6e8f3a32abaff34eec9548c7f2fb34bd50edfe"
)

PSM_DISABLED_COLLIDERS = {
    "/psm/psm_remote_center_link/visuals_xform/visuals",
    "/psm/psm_pitch_end_link/visuals_xform/visuals",
    "/psm/psm_main_insertion_link/visuals_xform/visuals",
    "/psm/psm_main_insertion_link_2/visuals_xform/tool_main_insert",
    "/psm/psm_main_insertion_link_3/visuals_xform/visuals",
    "/psm/psm_tool_roll_link/visuals_xform/visuals",
    "/psm/psm_tool_roll_link/collisions_xform/collisions",
    "/psm/psm_tool_pitch_link/collisions_xform/collisions",
    "/psm/psm_tool_yaw_link/visuals_xform/tool_yaw_link",
    "/psm/psm_tool_gripper1_link/collisions_xform/collisions",
    "/psm/psm_tool_gripper2_link/collisions_xform/collisions",
}
TABLE_DISABLED_COLLIDER = "/Table/Table/Table"
ROOT_CANDIDATE_PROPERTIES = {
    "dranmar:candidate:active",
    "dranmar:candidate:assetId",
    "dranmar:candidate:qualificationContract",
    "dranmar:candidate:runtimeQuaternionOrder",
    "dranmar:candidate:status",
    "dranmar:candidate:usdQuaternionOrder",
    "dranmar:candidate:version",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path.resolve()))
    assert stage is not None
    return stage


def _package_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_manifest(
    package_root: Path,
    *,
    asset_id: str,
    schema: str,
) -> None:
    manifest = _json(package_root, "asset_manifest.json")
    expected_members = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path.name != "asset_manifest.json"
    }
    assert set(manifest["members"]) == expected_members
    assert manifest["asset_id"] == asset_id
    assert manifest["schema"] == schema
    assert manifest["status"] == STATUS
    assert manifest["active_replacement"] is False
    assert manifest["runtime_references"] == []
    assert manifest["package_role"] == (
        "inactive_collision_overlay_with_external_hash_locked_bases"
    )
    assert manifest["dependency_complete_directory"] is False
    assert manifest["dependency_lock_complete"] is True
    assert manifest["source_static_validation"] is True
    assert manifest["physics_calibration"] is False
    assert manifest["instrument_calibration"] is False
    assert manifest["clinical_validation"] is False
    assert manifest["license"] == "BSD-3-Clause"
    assert manifest["generator"]["standard_library_only"] is True
    assert manifest["generator"]["repository_path"] == (
        "tools/generate_t1_collider_candidates.py"
    )
    assert manifest["generator"]["sha256"] == _sha256(GENERATOR)
    for relative, expected in manifest["members"].items():
        path = package_root / relative
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]
        assert expected["license"] == "BSD-3-Clause"


def test_packages_are_compact_hash_locked_inactive_and_not_runtime_referenced():
    _assert_manifest(
        PSM_ROOT,
        asset_id=PSM_ASSET_ID,
        schema="dr.anmar.psm-t1-collider-candidate-manifest.v1",
    )
    _assert_manifest(
        TABLE_ROOT,
        asset_id=TABLE_ASSET_ID,
        schema="dr.anmar.table-t1-collider-candidate-manifest.v1",
    )
    assert _sha256(PSM_BASE) == PSM_BASE_SHA256
    assert _sha256(PSM_VISUAL) == PSM_VISUAL_SHA256
    assert _sha256(TABLE_BASE) == TABLE_BASE_SHA256
    assert _sha256(TABLE_VISUAL) == TABLE_VISUAL_SHA256
    assert PSM_BASE.stat().st_size == 22_251_170
    assert PSM_VISUAL.stat().st_size == 16_261
    assert TABLE_BASE.stat().st_size == 18_839_808
    assert TABLE_VISUAL.stat().st_size == 748_175

    for package_root, expected_direct in (
        (PSM_ROOT, "psm_visual_overlay"),
        (TABLE_ROOT, "table_visual_overlay"),
    ):
        lock = _json(package_root, "dependency_lock.json")
        assert lock["dependency_complete"] is True
        assert lock["direct_composition_dependency"] == expected_direct
        for dependency in lock["dependencies"].values():
            path = ROOT / dependency["path"]
            assert path.stat().st_size == dependency["bytes"]
            assert _sha256(path) == dependency["sha256"]
            assert dependency["modified_by_candidate"] is False
        assert (package_root / "LICENSE.txt").read_bytes() == (
            ROOT / "LICENSE"
        ).read_bytes()

    payload_bytes = sum(
        path.stat().st_size
        for package_root in (PSM_ROOT, TABLE_ROOT)
        for path in package_root.rglob("*")
        if path.is_file()
    )
    assert payload_bytes <= 90_000

    candidate_names = {PSM_USD.name, TABLE_USD.name}
    scan_roots = (
        WORKTREE_ROOT / "config",
        WORKTREE_ROOT / "scripts",
        ROOT / "orbit",
        ROOT.parent / "orbit.surgical.tasks",
    )
    references: list[Path] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".py", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(name in text for name in candidate_names):
                references.append(path)
    assert references == []


def test_clean_temp_regeneration_is_byte_deterministic(tmp_path: Path):
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--output-root",
            str(tmp_path),
        ],
        check=True,
        cwd=ROOT,
    )
    assert _package_hashes(tmp_path / PSM_ROOT.relative_to(ROOT)) == (
        _package_hashes(PSM_ROOT)
    )
    assert _package_hashes(tmp_path / TABLE_ROOT.relative_to(ROOT)) == (
        _package_hashes(TABLE_ROOT)
    )


def _iter_prim_specs(root_specs: list[Sdf.PrimSpec]):
    stack = list(reversed(root_specs))
    while stack:
        prim_spec = stack.pop()
        yield prim_spec
        stack.extend(reversed(list(prim_spec.nameChildren)))


def _assert_candidate_layer_authority(
    path: Path,
    *,
    root_path: str,
    reference_path: str,
    disabled_colliders: set[str],
) -> None:
    layer = Sdf.Layer.FindOrOpen(str(path.resolve()))
    assert layer is not None
    assert layer.defaultPrim == root_path.removeprefix("/")
    root_spec = layer.GetPrimAtPath(root_path)
    assert root_spec is not None
    references = list(root_spec.referenceList.prependedItems)
    assert len(references) == 1
    assert references[0].assetPath == reference_path
    assert references[0].primPath == Sdf.Path(root_path)

    seen_disabled: set[str] = set()
    for prim_spec in _iter_prim_specs(list(layer.rootPrims)):
        prim_path = str(prim_spec.path)
        property_names = {prop.name for prop in prim_spec.properties}
        in_proxy = "/DrAnmarT1ColliderCandidate" in prim_path
        if prim_path == root_path:
            assert property_names == ROOT_CANDIDATE_PROPERTIES
        elif prim_path in disabled_colliders:
            assert property_names == {"physics:collisionEnabled"}
            collision_spec = prim_spec.attributes["physics:collisionEnabled"]
            assert collision_spec.default is False
            seen_disabled.add(prim_path)
        elif not in_proxy:
            assert property_names == set()

        forbidden = {
            name
            for name in property_names
            if any(
                token in name.lower()
                for token in (
                    "centerofmass",
                    "density",
                    "diagonalinertia",
                    "dynamicfriction",
                    "mass",
                    "material:binding",
                    "staticfriction",
                )
            )
        }
        assert forbidden == set()

    assert seen_disabled == disabled_colliders


def test_candidate_layers_have_exact_collision_only_authority():
    _assert_candidate_layer_authority(
        PSM_USD,
        root_path="/psm",
        reference_path=PSM_REFERENCE,
        disabled_colliders=PSM_DISABLED_COLLIDERS,
    )
    _assert_candidate_layer_authority(
        TABLE_USD,
        root_path="/Table",
        reference_path=TABLE_REFERENCE,
        disabled_colliders={TABLE_DISABLED_COLLIDER},
    )


def _assert_base_properties_preserved(
    base_path: Path,
    candidate_path: Path,
    allowed_changes: set[str],
) -> None:
    base = _stage(base_path)
    candidate = _stage(candidate_path)
    for base_prim in base.Traverse():
        prim_path = base_prim.GetPath()
        candidate_prim = candidate.GetPrimAtPath(prim_path)
        assert candidate_prim
        assert candidate_prim.GetTypeName() == base_prim.GetTypeName()
        for base_property in base_prim.GetProperties():
            property_path = str(base_property.GetPath())
            candidate_property = candidate.GetPropertyAtPath(
                base_property.GetPath()
            )
            assert candidate_property
            if property_path in allowed_changes:
                assert isinstance(base_property, Usd.Attribute)
                assert isinstance(candidate_property, Usd.Attribute)
                assert base_property.Get() is True
                assert candidate_property.Get() is False
                continue
            assert type(candidate_property) is type(base_property)
            if isinstance(base_property, Usd.Attribute):
                assert candidate_property.GetTypeName() == (
                    base_property.GetTypeName()
                )
                assert candidate_property.Get() == base_property.Get()
                assert candidate_property.GetTimeSamples() == (
                    base_property.GetTimeSamples()
                )
                assert candidate_property.GetConnections() == (
                    base_property.GetConnections()
                )
            else:
                assert candidate_property.GetTargets() == (
                    base_property.GetTargets()
                )


def test_composition_preserves_every_existing_property_except_enabled_collision():
    _assert_base_properties_preserved(
        PSM_VISUAL,
        PSM_USD,
        {f"{path}.physics:collisionEnabled" for path in PSM_DISABLED_COLLIDERS},
    )
    _assert_base_properties_preserved(
        TABLE_VISUAL,
        TABLE_USD,
        {f"{TABLE_DISABLED_COLLIDER}.physics:collisionEnabled"},
    )


def _candidate_colliders(stage: Usd.Stage) -> list[Usd.Prim]:
    return [
        prim
        for prim in stage.Traverse()
        if "/DrAnmarT1ColliderCandidate/" in str(prim.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def _mesh_triangle_count(prim: Usd.Prim) -> int:
    if not prim.IsA(UsdGeom.Mesh):
        return 0
    return sum(
        max(int(count) - 2, 0)
        for count in UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
    )


def _assert_closed_triangle_mesh(prim: Usd.Prim) -> None:
    mesh = UsdGeom.Mesh(prim)
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    assert set(counts) == {3}
    assert len(indices) == sum(counts)
    assert max(indices) < len(mesh.GetPointsAttr().Get())
    edge_counts: Counter[tuple[int, int]] = Counter()
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        for index, first in enumerate(face):
            second = face[(index + 1) % count]
            edge_counts[tuple(sorted((first, second)))] += 1
    assert edge_counts
    assert set(edge_counts.values()) == {2}


def test_proxy_counts_budgets_local_ownership_and_closed_convex_meshes():
    psm = _stage(PSM_USD)
    table = _stage(TABLE_USD)
    psm_colliders = _candidate_colliders(psm)
    table_colliders = _candidate_colliders(table)
    assert len(psm_colliders) == 31
    assert len(table_colliders) == 7
    assert Counter(prim.GetTypeName() for prim in psm_colliders) == {
        "Capsule": 5,
        "Cube": 20,
        "Cylinder": 1,
        "Mesh": 4,
        "Sphere": 1,
    }
    assert Counter(prim.GetTypeName() for prim in table_colliders) == {
        "Cube": 6,
        "Mesh": 1,
    }
    assert sum(map(_mesh_triangle_count, psm_colliders)) == 48
    assert sum(map(_mesh_triangle_count, table_colliders)) == 28

    psm_owners = set()
    for prim in psm_colliders:
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
        assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == "invisible"
        assert prim.GetParent().GetName() == "DrAnmarT1ColliderCandidate"
        owner = prim.GetParent().GetParent()
        assert re.search(r"_link(?:_\d+)?$", owner.GetName())
        assert owner.HasAPI(UsdPhysics.RigidBodyAPI)
        psm_owners.add(str(owner.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            assert (
                UsdPhysics.MeshCollisionAPI(prim)
                .GetApproximationAttr()
                .Get()
                == "convexHull"
            )
            _assert_closed_triangle_mesh(prim)
    assert len(psm_owners) == 10
    assert "/psm/DrAnmarT1ColliderCandidate" not in {
        str(prim.GetParent().GetPath()) for prim in psm_colliders
    }

    for prim in table_colliders:
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
        assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == "invisible"
        assert str(prim.GetParent().GetPath()) == (
            "/Table/Table/DrAnmarT1ColliderCandidate"
        )
        if prim.IsA(UsdGeom.Mesh):
            _assert_closed_triangle_mesh(prim)

    for path in PSM_DISABLED_COLLIDERS:
        assert (
            UsdPhysics.CollisionAPI(psm.GetPrimAtPath(path))
            .GetCollisionEnabledAttr()
            .Get()
            is False
        )
    assert (
        UsdPhysics.CollisionAPI(table.GetPrimAtPath(TABLE_DISABLED_COLLIDER))
        .GetCollisionEnabledAttr()
        .Get()
        is False
    )


def _proxy_local_bounds(prim: Usd.Prim) -> tuple[list[float], list[float]]:
    kind = prim.GetTypeName()
    if kind == "Mesh":
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        return (
            [min(float(point[index]) for point in points) for index in range(3)],
            [max(float(point[index]) for point in points) for index in range(3)],
        )
    translate_attr = prim.GetAttribute("xformOp:translate")
    center = (
        [float(value) for value in translate_attr.Get()]
        if translate_attr
        else [0.0, 0.0, 0.0]
    )
    if kind == "Cube":
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        scale = [float(value) for value in prim.GetAttribute("xformOp:scale").Get()]
        half = [size * value * 0.5 for value in scale]
    elif kind == "Sphere":
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        half = [radius, radius, radius]
    else:
        radius = float(prim.GetAttribute("radius").Get())
        height = float(prim.GetAttribute("height").Get())
        axis = str(prim.GetAttribute("axis").Get())
        half = [radius, radius, radius]
        half["XYZ".index(axis)] = height * 0.5
        if kind == "Capsule":
            half["XYZ".index(axis)] += radius
    return (
        [center[index] - half[index] for index in range(3)],
        [center[index] + half[index] for index in range(3)],
    )


def _union_proxy_bounds(prims: list[Usd.Prim]) -> tuple[list[float], list[float]]:
    bounds = [_proxy_local_bounds(prim) for prim in prims]
    return (
        [min(bound[0][index] for bound in bounds) for index in range(3)],
        [max(bound[1][index] for bound in bounds) for index in range(3)],
    )


def _is_link(prim: Usd.Prim) -> bool:
    return prim.GetTypeName() == "Xform" and bool(
        re.search(r"_link(?:_\d+)?$", prim.GetName())
    )


def _render_bounds_in_link(stage: Usd.Stage, link: Usd.Prim):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    inverse_link = cache.GetLocalToWorldTransform(link).GetInverse()
    transformed_points = []
    for prim in Usd.PrimRange(link):
        if prim == link:
            continue
        ancestor = prim.GetParent()
        while ancestor and ancestor != link and not _is_link(ancestor):
            ancestor = ancestor.GetParent()
        if ancestor != link or "/collisions_xform/" in str(prim.GetPath()):
            continue
        points = []
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        elif prim.IsA(UsdGeom.Sphere):
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            points = [
                (x, y, z)
                for x in (-radius, radius)
                for y in (-radius, radius)
                for z in (-radius, radius)
            ]
        if not points:
            continue
        transform = cache.GetLocalToWorldTransform(prim) * inverse_link
        transformed_points.extend(transform.Transform(point) for point in points)
    assert transformed_points
    return (
        [
            min(float(point[index]) for point in transformed_points)
            for index in range(3)
        ],
        [
            max(float(point[index]) for point in transformed_points)
            for index in range(3)
        ],
    )


def test_approximation_reports_match_composed_geometry_and_locked_render_bounds():
    psm = _stage(PSM_USD)
    psm_base = _stage(PSM_VISUAL)
    report = _json(PSM_ROOT, "approximation_report.json")
    assert report["broad_single_shell_used"] is False
    assert report["candidate_primitive_count"] == 31
    assert report["candidate_mesh_triangles"] == 48
    assert report["legacy_enabled_collision_mesh_triangles"] == 288_052
    assert report["maximum_absolute_aabb_support_error_m"] <= 0.000200001
    assert report["triangle_reduction_fraction"] > 0.9998
    for link_name, measured in report["links"].items():
        link_path = f"/psm/{link_name}"
        base_bounds = _render_bounds_in_link(
            psm_base,
            psm_base.GetPrimAtPath(link_path),
        )
        assert base_bounds[0] == pytest.approx(
            measured["render_target_aabb_m"]["min"],
            abs=5.0e-8,
        )
        assert base_bounds[1] == pytest.approx(
            measured["render_target_aabb_m"]["max"],
            abs=5.0e-8,
        )
        scope = psm.GetPrimAtPath(
            f"{link_path}/DrAnmarT1ColliderCandidate"
        )
        proxy_bounds = _union_proxy_bounds(list(scope.GetChildren()))
        assert proxy_bounds[0] == pytest.approx(
            measured["proxy_aabb_m"]["min"],
            abs=1.0e-9,
        )
        assert proxy_bounds[1] == pytest.approx(
            measured["proxy_aabb_m"]["max"],
            abs=1.0e-9,
        )

    table = _stage(TABLE_USD)
    table_base = _stage(TABLE_VISUAL)
    table_report = _json(TABLE_ROOT, "approximation_report.json")
    assert table_report["broad_single_shell_used"] is False
    assert table_report["candidate_primitive_count"] == 7
    assert table_report["candidate_mesh_triangles"] == 28
    assert table_report["legacy_collision_mesh_triangles"] == 127_622
    assert table_report["triangle_reduction_fraction"] > 0.9997
    table_scope = table.GetPrimAtPath(
        "/Table/Table/DrAnmarT1ColliderCandidate"
    )
    table_proxies = {prim.GetName(): prim for prim in table_scope.GetChildren()}
    visual_root = "/Table/DrAnmarT1Visual"
    visual_paths = {
        "CenterColumn": f"{visual_root}/Frame/CenterColumn",
        "DeckFrame": f"{visual_root}/Frame/DeckFrame",
        "FloorBase": f"{visual_root}/Frame/FloorBase",
        "LeftFoot": f"{visual_root}/Frame/LeftFoot",
        "PatientPad": f"{visual_root}/RoundedPad",
        "RightFoot": f"{visual_root}/Frame/RightFoot",
    }
    for name, path in visual_paths.items():
        extent = UsdGeom.Boundable(
            table_base.GetPrimAtPath(path)
        ).GetExtentAttr().Get()
        recorded = table_report["components"][name]
        assert [float(value) for value in extent[0]] == pytest.approx(
            recorded["render_target_aabb_m"]["min"],
            abs=5.0e-8,
        )
        assert [float(value) for value in extent[1]] == pytest.approx(
            recorded["render_target_aabb_m"]["max"],
            abs=5.0e-8,
        )
        proxy_bounds = _proxy_local_bounds(table_proxies[name])
        assert proxy_bounds[0] == pytest.approx(
            recorded["proxy_aabb_m"]["min"],
            abs=5.0e-8,
        )
        assert proxy_bounds[1] == pytest.approx(
            recorded["proxy_aabb_m"]["max"],
            abs=5.0e-8,
        )
    sterile = table_report["components"]["SterileFieldTop"]
    assert sterile["unsupported_hanging_drape_is_rigid"] is False
    assert sterile["vertical_top_support_error_m"] == 0.0
    assert _proxy_local_bounds(table_proxies["SterileFieldTop"])[1][2] == (
        pytest.approx(0.459881078)
    )


def test_runtime_quaternion_scale_authority_and_validation_boundaries_fail_closed():
    for package_root, root_prim, primitive_count, triangles in (
        (PSM_ROOT, "/psm", 31, 48),
        (TABLE_ROOT, "/Table", 7, 28),
    ):
        geometry = _json(package_root, "geometry_contract.json")
        activation = geometry["activation"]
        assert activation == {
            "active_replacement": False,
            "default": "blocked",
            "requires_explicit_human_review": True,
        }
        runtime = geometry["runtime_contract"]
        assert runtime["root_prim"] == root_prim
        assert runtime["meters_per_unit"] == 1.0
        assert runtime["stage_up_axis"] == "Z"
        assert runtime["allowed_runtime_scale"] == [1.0, 1.0, 1.0]
        assert runtime["non_unit_scale_invalidates_qualification"] is True
        assert runtime["runtime_quaternion_order"] == "xyzw"
        assert runtime["identity_quaternion_runtime_xyzw"] == [
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        assert runtime["usd_quaternion_order"] == "wxyz"
        assert runtime["identity_quaternion_usd_wxyz"] == [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        assert runtime["root_transform_authored_by_candidate"] is False
        assert runtime["root_transform_inherited_exactly"] is True
        budget = geometry["primitive_budget"]
        assert budget["observed_primitives"] == primitive_count
        assert budget["maximum_primitives"] == primitive_count
        assert budget["observed_mesh_triangles"] == triangles
        assert budget["maximum_mesh_triangles"] == triangles

        qualification = _json(
            package_root,
            "qualification_contract.json",
        )
        assert qualification["activation"]["default"] == "blocked"
        assert qualification["activation"]["active_replacement"] is False
        assert (
            qualification["activation"][
                "requires_native_isaac_physx_evidence"
            ]
            is True
        )
        assert (
            qualification["activation"]["decision_owner"]
            == "explicit_human_review"
        )
        assert all(
            value is False
            for value in qualification["validation_boundaries"].values()
        )
        integrity = qualification["evidence_integrity"]
        assert integrity["paired_baseline_and_candidate_seeds"] is True
        assert integrity["pre_registered_thresholds"] is True
        assert integrity["reward_contract_must_remain_identical"] is True

    psm_required = _json(
        PSM_ROOT,
        "qualification_contract.json",
    )["required_suites"]
    assert set(psm_required) == {
        "articulation_and_joint_parity",
        "contact_stability",
        "task_noninferiority",
        "wrist_and_jaw_contact",
    }
    assert "conditional_mid_air_retention_given_pickup" in (
        psm_required["task_noninferiority"]
    )
    assert "needle_relative_slip_distance_m" in (
        psm_required["wrist_and_jaw_contact"]
    )


def test_articulation_joint_link_and_table_root_transform_signatures_are_exact():
    psm_base = _stage(PSM_VISUAL)
    psm_candidate = _stage(PSM_USD)
    base_joints = {
        str(prim.GetPath()): prim
        for prim in psm_base.Traverse()
        if prim.IsA(UsdPhysics.Joint)
    }
    candidate_joints = {
        str(prim.GetPath()): prim
        for prim in psm_candidate.Traverse()
        if prim.IsA(UsdPhysics.Joint)
    }
    assert set(candidate_joints) == set(base_joints)
    for path, base_joint_prim in base_joints.items():
        candidate_joint_prim = candidate_joints[path]
        assert candidate_joint_prim.GetTypeName() == (
            base_joint_prim.GetTypeName()
        )
        base_joint = UsdPhysics.Joint(base_joint_prim)
        candidate_joint = UsdPhysics.Joint(candidate_joint_prim)
        assert candidate_joint.GetBody0Rel().GetTargets() == (
            base_joint.GetBody0Rel().GetTargets()
        )
        assert candidate_joint.GetBody1Rel().GetTargets() == (
            base_joint.GetBody1Rel().GetTargets()
        )
        assert candidate_joint.GetLocalPos0Attr().Get() == (
            base_joint.GetLocalPos0Attr().Get()
        )
        assert candidate_joint.GetLocalPos1Attr().Get() == (
            base_joint.GetLocalPos1Attr().Get()
        )
        assert candidate_joint.GetLocalRot0Attr().Get() == (
            base_joint.GetLocalRot0Attr().Get()
        )
        assert candidate_joint.GetLocalRot1Attr().Get() == (
            base_joint.GetLocalRot1Attr().Get()
        )

    for base_prim in psm_base.Traverse():
        if not _is_link(base_prim):
            continue
        candidate_prim = psm_candidate.GetPrimAtPath(base_prim.GetPath())
        base_xform = UsdGeom.Xformable(base_prim)
        candidate_xform = UsdGeom.Xformable(candidate_prim)
        assert [op.GetOpName() for op in candidate_xform.GetOrderedXformOps()] == [
            op.GetOpName() for op in base_xform.GetOrderedXformOps()
        ]
        assert [op.Get() for op in candidate_xform.GetOrderedXformOps()] == [
            op.Get() for op in base_xform.GetOrderedXformOps()
        ]

    table_base = _stage(TABLE_VISUAL)
    table_candidate = _stage(TABLE_USD)
    for path in ("/Table", "/Table/Table"):
        base_xform = UsdGeom.Xformable(table_base.GetPrimAtPath(path))
        candidate_xform = UsdGeom.Xformable(
            table_candidate.GetPrimAtPath(path)
        )
        assert [op.GetOpName() for op in candidate_xform.GetOrderedXformOps()] == [
            op.GetOpName() for op in base_xform.GetOrderedXformOps()
        ]
        assert [op.Get() for op in candidate_xform.GetOrderedXformOps()] == [
            op.Get() for op in base_xform.GetOrderedXformOps()
        ]
    base_fixed_joint = UsdPhysics.FixedJoint(
        table_base.GetPrimAtPath("/Table/FixedJoint")
    )
    candidate_fixed_joint = UsdPhysics.FixedJoint(
        table_candidate.GetPrimAtPath("/Table/FixedJoint")
    )
    assert candidate_fixed_joint.GetBody0Rel().GetTargets() == (
        base_fixed_joint.GetBody0Rel().GetTargets()
    )
    assert candidate_fixed_joint.GetBody1Rel().GetTargets() == (
        base_fixed_joint.GetBody1Rel().GetTargets()
    )
