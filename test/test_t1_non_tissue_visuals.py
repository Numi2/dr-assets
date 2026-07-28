# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Static OpenUSD gates for the T1 non-tissue visual overlays."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data/Props/SurgicalScene/T1"
MANIFEST_PATH = ASSET_ROOT / "asset_manifest.json"
MATERIALS_PATH = ASSET_ROOT / "materials.usda"
NVIDIA_VENDOR_ROOT = (
    ASSET_ROOT
    / "vendor/nvidia_physicalai_simready_materials_v0_2_0"
)
NVIDIA_VENDOR_HASHES = {
    "LICENSE.md": (
        "18f74283f08ff1ed39a9c46dbe2622146d45f771023c3dbd9c631bb058e1421b"
    ),
    "open_pbr_uber_base_class.usda": (
        "bb76ff9fa9cd74b86b6be4ed3c6ed79cdca15eff6d603ca571bdf9ce21e10c5f"
    ),
}
ENTRYPOINTS = {
    "psm": ASSET_ROOT / "psm_visual_v1.usda",
    "table": ASSET_ROOT / "table_visual_v1.usda",
    "legacy_needle": ASSET_ROOT / "legacy_needle_visual_v1.usda",
}
REFERENCE_RIG = ASSET_ROOT / "reference_or_rig_v1.usda"
BASE_ASSETS = {
    "psm": ROOT / "data/Robots/dVRK/PSM/psm_col.usd",
    "table": ROOT / "data/Props/Table/table.usd",
    "legacy_needle": ROOT / "data/Props/Surgical_needle/needle_sdf.usd",
}
PSM_STEEL_MESHES = {
    "/psm/psm_main_insertion_link_2/visuals_xform/tool_main_insert": (
        "ShaftSatinSteel"
    ),
    "/psm/psm_tool_roll_link/visuals_xform/tool_roll_link": (
        "WristSatinSteel"
    ),
    "/psm/psm_tool_pitch_link/visuals_xform/tool_pitch_link": (
        "WristSatinSteel"
    ),
    "/psm/psm_tool_yaw_link/visuals_xform/tool_yaw_link": (
        "WristSatinSteel"
    ),
    "/psm/psm_tool_gripper1_link/visuals_xform/gripper_right": (
        "JawSatinSteel"
    ),
    "/psm/psm_tool_gripper2_link/visuals_xform/gripper_left": (
        "JawSatinSteel"
    ),
}
PSM_MATTE_POLYMER_MESHES = (
    "/psm/psm_pitch_end_link/visuals_xform/visuals",
    "/psm/psm_main_insertion_link/visuals_xform/visuals",
    "/psm/psm_main_insertion_link_3/visuals_xform/visuals",
)
AUTHORITY_PROPERTY_PREFIXES = (
    "drive:",
    "joint:",
    "limit:",
    "physics:",
    "physx",
    "state:",
    "xformOp:",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path))
    assert stage is not None, path
    return stage


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()[:24]
    assert payload[:8] == b"\x89PNG\r\n\x1a\n", path
    assert payload[12:16] == b"IHDR", path
    return struct.unpack(">II", payload[16:24])


def _walk_specs(spec: Sdf.PrimSpec):
    yield spec
    for child in spec.nameChildren:
        yield from _walk_specs(child)


def _raw_layer_prim_specs(path: Path):
    layer = Sdf.Layer.FindOrOpen(str(path))
    assert layer is not None, path
    for root_prim in layer.rootPrims:
        yield from _walk_specs(root_prim)


def _property_snapshot(prim: Usd.Prim) -> dict[str, Any]:
    attributes = {}
    for attribute in prim.GetAttributes():
        if attribute.GetName().startswith(AUTHORITY_PROPERTY_PREFIXES):
            attributes[attribute.GetName()] = repr(attribute.Get())
    relationships = {}
    for relationship in prim.GetRelationships():
        if relationship.GetName().startswith(AUTHORITY_PROPERTY_PREFIXES):
            relationships[relationship.GetName()] = tuple(
                str(target) for target in relationship.GetTargets()
            )
    schemas = tuple(
        schema
        for schema in prim.GetAppliedSchemas()
        if schema.startswith(("Physics", "Physx"))
    )
    return {
        "attributes": attributes,
        "relationships": relationships,
        "schemas": schemas,
    }


def _assert_base_authority_is_composition_identical(
    base_path: Path,
    overlay_path: Path,
) -> None:
    base = _stage(base_path)
    overlay = _stage(overlay_path)
    base_root = str(base.GetDefaultPrim().GetPath())
    overlay_root = str(overlay.GetDefaultPrim().GetPath())
    assert base_root == overlay_root
    for base_prim in base.Traverse():
        path = str(base_prim.GetPath())
        overlay_prim = overlay.GetPrimAtPath(path)
        assert overlay_prim.IsValid(), path
        assert overlay_prim.GetTypeName() == base_prim.GetTypeName(), path
        assert _property_snapshot(overlay_prim) == _property_snapshot(
            base_prim
        ), path


def _binding_target(stage: Usd.Stage, path: str) -> tuple[str, str]:
    prim = stage.GetPrimAtPath(path)
    assert prim.IsValid(), path
    relationship = prim.GetRelationship("material:binding")
    assert relationship.IsValid(), path
    targets = relationship.GetTargets()
    assert len(targets) == 1, (path, targets)
    target = str(targets[0])
    assert stage.GetPrimAtPath(target).IsValid(), target
    return target, str(relationship.GetMetadata("bindMaterialAs"))


def test_manifest_is_complete_deterministic_and_locks_base_bytes():
    payload = _manifest()
    expected_members = {
        path.relative_to(ASSET_ROOT).as_posix()
        for path in ASSET_ROOT.rglob("*")
        if path.is_file() and path.name != "asset_manifest.json"
    }
    assert set(payload["members"]) == expected_members
    assert payload["schema"].endswith(".v2")
    assert payload["license"] == "mixed"
    assert payload["licenses"] == [
        "Apache-2.0",
        "BSD-3-Clause",
        "MIT-0",
    ]
    assert payload["package_role"] == (
        "render_overlay_bundle_with_external_pinned_dependencies"
    )
    assert payload["dependency_complete_directory"] is False
    assert payload["license_sources"]["repository_foundations"]["license"] == (
        "BSD-3-Clause"
    )
    assert payload["visual_only"] is True
    assert payload["physics_authority"] is False
    assert payload["collision_authority"] is False
    assert payload["vendor_assets_modified"] is False
    assert payload["clinical_validation"] is False
    generator = ROOT / payload["generator"]["path"]
    validator = ROOT / payload["validator"]["path"]
    assert _sha256(generator) == payload["generator"]["sha256"]
    assert _sha256(validator) == payload["validator"]["sha256"]
    assert payload["generator"]["texture_size_px"] == 2048
    for relative, values in payload["members"].items():
        path = ASSET_ROOT / relative
        assert path.stat().st_size == values["bytes"]
        assert _sha256(path) == values["sha256"]
        if relative in {
            (
                "vendor/nvidia_physicalai_simready_materials_v0_2_0/"
                "LICENSE.md"
            ),
            (
                "vendor/nvidia_physicalai_simready_materials_v0_2_0/"
                "open_pbr_uber_base_class.usda"
            ),
        }:
            assert values["license"] == "MIT-0"
            assert values["provenance"].startswith("verbatim_NVIDIA")
        elif relative == "LICENSE.txt":
            assert values["license"] == "Apache-2.0"
            assert values["provenance"].startswith(
                "canonical_Apache-2.0"
            )
        elif path.suffix == ".png" or path.suffix.startswith(".usd"):
            assert values["license"] == "Apache-2.0"
            assert (
                values["provenance"]
                == "deterministically_generated_by_DrAnmar"
            )
        else:
            assert values["license"] == "Apache-2.0"
            assert values["provenance"] == "authored_by_DrAnmar"
    for name, base_path in BASE_ASSETS.items():
        values = payload["base_assets"][name]
        assert values["path"] == base_path.relative_to(ROOT).as_posix()
        assert values["bytes"] == base_path.stat().st_size
        assert values["sha256"] == _sha256(base_path)
        assert values["modified_by_generator"] is False
    vendor = payload["vendor_dependencies"][
        "nvidia_physicalai_simready_materials"
    ]
    assert vendor["release"] == "v0.2.0"
    assert vendor["license"] == "MIT-0"
    assert vendor["material_interface"] == "OpenPBR_1.1_MaterialX"
    for name, expected_hash in NVIDIA_VENDOR_HASHES.items():
        path = NVIDIA_VENDOR_ROOT / name
        assert _sha256(path) == expected_hash
        assert vendor["members"][name] == {
            "copied_byte_identical": True,
            "modified_by_generator": False,
            "sha256": expected_hash,
        }
    assert payload["material_contexts"] == {
        "fallback": "UsdPreviewSurface",
        "mdl_only_claim": False,
        "native_rtx_appearance_qualified": False,
        "primary": "OpenPBR_1.1_MaterialX_inherited",
    }


def test_clean_temp_regeneration_is_byte_exact_and_removes_legacy_normals(
    tmp_path: Path,
):
    candidate = tmp_path / "T1"
    stale_paths = (
        candidate / "textures/pad_normal.jpg",
        candidate / "textures/drape_normal.jpg",
    )
    for stale in stale_paths:
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"stale-lossless-normal")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_t1_non_tissue_visuals.py"),
            "--asset-root",
            str(candidate),
        ],
        check=True,
        cwd=ROOT,
    )

    assert not any(path.exists() for path in stale_paths)
    canonical_manifest = _manifest()
    regenerated_manifest = json.loads(
        (candidate / "asset_manifest.json").read_text(encoding="utf-8")
    )
    assert regenerated_manifest == canonical_manifest
    for relative in canonical_manifest["members"]:
        assert (candidate / relative).read_bytes() == (
            ASSET_ROOT / relative
        ).read_bytes()


def test_all_local_composition_dependencies_resolve():
    expected_defaults = {
        "psm": "/psm",
        "table": "/Table",
        "legacy_needle": "/Needle",
    }
    for name, entrypoint in ENTRYPOINTS.items():
        stage = _stage(entrypoint)
        assert str(stage.GetDefaultPrim().GetPath()) == expected_defaults[name]
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            str(entrypoint)
        )
        assert layers
        assert not {str(value) for value in unresolved}
        for asset in assets:
            text = str(asset)
            assert Path(text).is_file(), text
    rig = _stage(REFERENCE_RIG)
    assert str(rig.GetDefaultPrim().GetPath()) == "/DrAnmarT1ReferenceRig"
    assert rig.GetPrimAtPath(
        "/DrAnmarT1ReferenceRig/ReferenceCamera"
    ).IsA(UsdGeom.Camera)


def test_overlay_layers_author_no_physics_or_existing_transform_opinions():
    overlay_layers = (*ENTRYPOINTS.values(), ASSET_ROOT / "materials.usda")
    for entrypoint in overlay_layers:
        for prim_spec in _raw_layer_prim_specs(entrypoint):
            schemas = prim_spec.GetInfo("apiSchemas")
            if schemas:
                for schema in schemas.GetAddedOrExplicitItems():
                    assert not str(schema).startswith(("Physics", "Physx"))
            for property_spec in prim_spec.properties:
                assert not property_spec.name.startswith(
                    AUTHORITY_PROPERTY_PREFIXES
                ), (entrypoint.name, prim_spec.path, property_spec.name)


def test_composed_base_physics_joints_and_transforms_are_identical():
    for name in ("psm", "table", "legacy_needle"):
        _assert_base_authority_is_composition_identical(
            BASE_ASSETS[name],
            ENTRYPOINTS[name],
        )


def test_psm_has_no_dangling_material_targets_and_strong_restrained_bindings():
    stage = _stage(ENTRYPOINTS["psm"])
    dangling = []
    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            if relationship.GetName().startswith("material:binding"):
                for target in relationship.GetTargets():
                    if not stage.GetPrimAtPath(target).IsValid():
                        dangling.append((str(prim.GetPath()), str(target)))
    assert dangling == []
    for path, material in PSM_STEEL_MESHES.items():
        assert _binding_target(stage, path) == (
            f"/psm/DrAnmarT1Looks/{material}",
            "strongerThanDescendants",
        )
    for path in PSM_MATTE_POLYMER_MESHES:
        assert _binding_target(stage, path) == (
            "/psm/DrAnmarT1Looks/MattePolymer",
            "strongerThanDescendants",
        )
    raw = ENTRYPOINTS["psm"].read_text(encoding="utf-8")
    assert raw.count("rel material:binding = None") == 100
    root_data = stage.GetDefaultPrim().GetCustomData()
    assert root_data["drAnmarClearedDanglingMaterialBindings"] == 100
    assert root_data["drAnmarPhysicsAuthority"] is False
    for required_link in (
        "/psm/psm_tool_tip_link",
        "/psm/psm_tool_gripper1_link",
        "/psm/psm_tool_gripper2_link",
    ):
        assert stage.GetPrimAtPath(required_link).IsValid()


def test_table_hides_only_legacy_render_and_keeps_collision_composed():
    base = _stage(BASE_ASSETS["table"])
    overlay = _stage(ENTRYPOINTS["table"])
    path = "/Table/Table/Table"
    base_mesh = base.GetPrimAtPath(path)
    overlay_mesh = overlay.GetPrimAtPath(path)
    assert base_mesh.IsValid() and overlay_mesh.IsValid()
    assert (
        UsdGeom.Imageable(base_mesh).ComputeVisibility()
        == UsdGeom.Tokens.inherited
    )
    assert (
        UsdGeom.Imageable(overlay_mesh).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    )
    assert (
        overlay_mesh.GetAttribute("physics:collisionEnabled").Get()
        == base_mesh.GetAttribute("physics:collisionEnabled").Get()
        is True
    )
    visual_root = overlay.GetPrimAtPath("/Table/DrAnmarT1Visual")
    assert visual_root.IsValid()
    meshes = [
        prim
        for prim in Usd.PrimRange(visual_root)
        if prim.IsA(UsdGeom.Mesh)
    ]
    assert len(meshes) == 7
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        assert UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.render
        assert mesh.GetPointsAttr().Get()
        assert mesh.GetFaceVertexCountsAttr().Get()
        assert mesh.GetNormalsAttr().Get()
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        assert st.IsDefined(), prim.GetPath()
        assert st.Get()
        assert st.GetInterpolation() in (
            UsdGeom.Tokens.vertex,
            UsdGeom.Tokens.faceVarying,
        )
        assert not any(
            schema.startswith(("Physics", "Physx"))
            for schema in prim.GetAppliedSchemas()
        )
        material, relationship = (
            UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        )
        assert material.GetPrim().IsValid(), prim.GetPath()
        assert relationship.IsValid(), prim.GetPath()


def test_legacy_needle_is_explicitly_compatibility_only():
    stage = _stage(ENTRYPOINTS["legacy_needle"])
    data = stage.GetDefaultPrim().GetCustomData()
    assert data["drAnmarCompatibilityOnly"] is True
    assert data["drAnmarGeometryPromotion"] is False
    assert data["drAnmarPhysicsAuthority"] is False
    assert data["drAnmarGeometryStatus"].startswith(
        "legacy_disconnected_shells"
    )
    assert _binding_target(stage, "/Needle/Needle/Needle") == (
        "/Needle/DrAnmarT1Looks/NeedleSatinSteel",
        "strongerThanDescendants",
    )


def test_textures_are_exactly_2048_and_material_color_spaces_are_explicit():
    materials = MATERIALS_PATH.read_text(encoding="utf-8")
    for stem in ("steel", "pad", "drape"):
        for suffix in ("basecolor", "roughness"):
            path = ASSET_ROOT / "textures" / f"{stem}_{suffix}.png"
            assert _png_dimensions(path) == (2048, 2048)
            assert f"@textures/{stem}_{suffix}.png@" in materials
        normal = ASSET_ROOT / "textures" / f"{stem}_normal.png"
        assert _png_dimensions(normal) == (2048, 2048)
        with Image.open(normal) as image:
            assert image.mode == "RGB"
        assert f"@textures/{stem}_normal.png@" in materials
    manifest = _manifest()
    assert manifest["texture_encoding"] == {
        "all_basecolor_roughness_and_normal_maps": "lossless_PNG",
        "native_rtx_qualified": False,
    }
    assert materials.count('token inputs:sourceColorSpace = "sRGB"') == 3
    assert materials.count('token inputs:sourceColorSpace = "raw"') == 6
    assert materials.count('uniform token info:id = "UsdPreviewSurface"') == 8
    assert materials.count("inherits = </open_pbr_uber_base>") == 8
    assert "OmniSurface" not in materials
    assert "outputs:mdl:surface" not in materials
    assert materials.count("float inputs:coat_weight = 0") == 8
    for roughness in ("0.54", "0.58", "0.62", "0.78", "0.82", "0.9"):
        assert f"float inputs:specular_roughness = {roughness}" in materials


def test_openpbr_materialx_is_primary_and_preview_is_real_fallback():
    stage = _stage(MATERIALS_PATH)
    uniform_names = (
        "ShaftSatinSteel",
        "WristSatinSteel",
        "JawSatinSteel",
        "NeedleSatinSteel",
        "MattePolymer",
    )
    textured = {
        "TableFrameSteel": "steel",
        "TablePad": "pad",
        "SterileDrape": "drape",
    }
    for name in (*uniform_names, *textured):
        path = f"/DrAnmarT1VisualMaterials/{name}"
        prim = stage.GetPrimAtPath(path)
        assert prim.IsA(UsdShade.Material), path
        assert tuple(
            str(value)
            for value in prim.GetInherits().GetAllDirectInherits()
        ) == ("/open_pbr_uber_base",)
        preview = prim.GetAttribute("outputs:surface")
        materialx = prim.GetAttribute("outputs:mtlx:surface")
        assert preview.IsValid()
        assert preview.GetConnections() == [
            Sdf.Path(f"{path}/PreviewSurface.outputs:surface")
        ]
        assert materialx.IsValid()
        assert materialx.GetConnections() == [
            Sdf.Path(f"{path}/open_pbr_surface_surfaceshader.outputs:out")
        ]
        assert not prim.GetAttribute("outputs:mdl:surface").IsValid()
    for name in uniform_names:
        prim = stage.GetPrimAtPath(f"/DrAnmarT1VisualMaterials/{name}")
        assert prim.GetAttribute("drAnmarOpenPBRInputMode").Get() == (
            "constants_no_uv_contract"
        )
        for attribute_name in (
            "inputs:base_color_texture_file",
            "inputs:specular_roughness_texture_file",
            "inputs:geometry_normal_texture_file",
        ):
            assert prim.GetAttribute(attribute_name).Get().path == ""
    for name, stem in textured.items():
        prim = stage.GetPrimAtPath(f"/DrAnmarT1VisualMaterials/{name}")
        assert prim.GetAttribute("drAnmarOpenPBRInputMode").Get() == (
            "uv_texture_maps"
        )
        expected = {
            "inputs:base_color_texture_file": (
                f"textures/{stem}_basecolor.png"
            ),
            "inputs:specular_roughness_texture_file": (
                f"textures/{stem}_roughness.png"
            ),
            "inputs:geometry_normal_texture_file": (
                f"textures/{stem}_normal.png"
            ),
        }
        for attribute_name, relative in expected.items():
            assert prim.GetAttribute(attribute_name).Get().path == relative
        assert (
            prim.GetAttribute(
                "inputs:base_color_texture_file"
            ).GetColorSpace()
            == "sRGB"
        )
        for attribute_name in (
            "inputs:specular_roughness_texture_file",
            "inputs:geometry_normal_texture_file",
        ):
            assert prim.GetAttribute(attribute_name).GetColorSpace() == "raw"
    assert (
        stage.GetPrimAtPath(
            "/DrAnmarT1VisualMaterials/SterileDrape"
        ).GetAttribute("inputs:geometry_thin_walled").Get()
        is True
    )


def test_vendored_openpbr_dependency_is_byte_exact_and_self_documenting():
    source_root = (
        ROOT
        / "data/Props/SurgicalTissue/NeedleReadyTissueUnit/visual/vendor"
        / "nvidia_physicalai_simready_materials_v0_2_0"
    )
    for name, expected_hash in NVIDIA_VENDOR_HASHES.items():
        source = source_root / name
        destination = NVIDIA_VENDOR_ROOT / name
        assert source.read_bytes() == destination.read_bytes()
        assert _sha256(destination) == expected_hash
    openpbr = _stage(NVIDIA_VENDOR_ROOT / "open_pbr_uber_base_class.usda")
    root = openpbr.GetDefaultPrim()
    assert root.GetName() == "open_pbr_uber_base"
    assert root.IsAbstract()
    assert root.GetAttribute("semantics:labels:OpenPBR_Ver").Get() == [
        "1.1"
    ]
    provenance = (NVIDIA_VENDOR_ROOT / "PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    assert "MIT-0 NVIDIA content" in provenance
    assert "OpenPBR 1.1 MaterialX" in provenance


def test_package_provenance_rejects_unrecorded_third_party_content():
    provenance = (ASSET_ROOT / "PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    notice = (ASSET_ROOT / "NOTICE.txt").read_text(encoding="utf-8")
    readme = (ASSET_ROOT / "README.md").read_text(encoding="utf-8")
    assert "No patient data" in provenance
    assert "third-party texture" in provenance
    assert "byte-identical copies" in provenance
    assert "OpenPBR 1.1 MaterialX" in provenance
    assert "does not rewrite or relicense" in notice
    assert "compatibility-only" in readme
    assert "not clinically validated" in readme
    assert "deliberately use constants" in readme
    assert "pretending texture fidelity" in readme
