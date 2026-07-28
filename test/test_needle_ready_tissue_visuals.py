# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Static integrity gates for the needle-ready tissue visual package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path
import sys
import types

import pytest
from PIL import Image, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
VISUAL_ROOT = ASSET_ROOT / "visual"
MANIFEST_PATH = ASSET_ROOT / "visual_manifest.json"
BASE_PHYSICS_FILES = {
    "needle_ready_tissue_training.usda",
    "needle_ready_tissue_contact.usda",
    "needle_ready_tissue_validation.usda",
    "needle_ready_tissue_unit.usda",
}
LODS = ("training", "contact", "validation")
TISSUE_MATERIALS = ("surface", "bulk", "fascia", "wound")
NVIDIA_VENDOR_MEMBERS = {
    "visual/vendor/nvidia_physicalai_simready_materials_v0_2_0/LICENSE.md",
    "visual/vendor/nvidia_physicalai_simready_materials_v0_2_0/Skin_Medium_normal.jpg",
    "visual/vendor/nvidia_physicalai_simready_materials_v0_2_0/open_pbr_uber_base_class.usda",
}
HELPER_PATH = ROOT / "orbit/surgical/assets/needle_ready_tissue.py"
for package_name in ("orbit", "orbit.surgical"):
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules.setdefault(package_name, package)
assets_package = types.ModuleType("orbit.surgical.assets")
assets_package.__path__ = [str(HELPER_PATH.parent)]
assets_package.ORBITSURGICAL_ASSETS_DATA_DIR = str(ROOT / "data")
sys.modules["orbit.surgical.assets"] = assets_package
HELPER_SPEC = importlib.util.spec_from_file_location(
    "orbit.surgical.assets.needle_ready_tissue",
    HELPER_PATH,
)
assert HELPER_SPEC and HELPER_SPEC.loader
HELPER = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = HELPER
HELPER_SPEC.loader.exec_module(HELPER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()[:24]
    assert payload[:8] == b"\x89PNG\r\n\x1a\n", path
    assert payload[12:16] == b"IHDR", path
    return struct.unpack(">II", payload[16:24])


def test_visual_manifest_is_recursive_and_every_member_is_licensed():
    payload = _manifest()
    expected_paths = {f"needle_ready_tissue_{lod}_visual.usda" for lod in LODS}
    expected_paths.add("needle_ready_tissue_visual_unit.usda")
    expected_paths.update(
        path.relative_to(ASSET_ROOT).as_posix() for path in VISUAL_ROOT.rglob("*") if path.is_file()
    )
    assert set(payload["members"]) == expected_paths
    assert payload["license"] == "mixed"
    assert payload["licenses"] == ["Apache-2.0", "MIT-0"]
    assert payload["visual_only"] is True
    assert payload["physics_authority"] is False
    assert payload["clinical_validation"] is False
    generator_path = ROOT / payload["generator"]["path"]
    assert _sha256(generator_path) == payload["generator"]["sha256"]
    for relative, values in payload["members"].items():
        path = ASSET_ROOT / relative
        assert path.stat().st_size == values["bytes"]
        assert _sha256(path) == values["sha256"]
        if relative in NVIDIA_VENDOR_MEMBERS:
            assert values["license"] == "MIT-0"
            assert values["provenance"].startswith(
                "vendored_unchanged_from_NVIDIA_"
            )
        else:
            assert values["license"] == "Apache-2.0"
            assert values["provenance"].endswith("_by_DrAnmar")


def test_visual_manifest_locks_but_does_not_modify_base_physics_layers():
    payload = _manifest()
    assert set(payload["base_physics_sha256"]) == BASE_PHYSICS_FILES
    for relative, expected_hash in payload["base_physics_sha256"].items():
        assert _sha256(ASSET_ROOT / relative) == expected_hash


def test_clean_temp_regeneration_is_byte_exact_and_removes_legacy_normals(
    tmp_path: Path,
):
    candidate = tmp_path / "NeedleReadyTissueUnit"
    for relative in (*BASE_PHYSICS_FILES, "geometry_contract.json"):
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSET_ROOT / relative, destination)
    for relative in NVIDIA_VENDOR_MEMBERS:
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSET_ROOT / relative, destination)
    stale = candidate / "visual/textures/surface_normal.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale-lossless-normal")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_needle_ready_tissue_visuals.py"),
            "--asset-root",
            str(candidate),
        ],
        check=True,
        cwd=ROOT,
    )

    assert not stale.exists()
    canonical_manifest = _manifest()
    regenerated_manifest = json.loads(
        (candidate / "visual_manifest.json").read_text(encoding="utf-8")
    )
    assert regenerated_manifest == canonical_manifest
    for relative in canonical_manifest["members"]:
        assert (candidate / relative).read_bytes() == (
            ASSET_ROOT / relative
        ).read_bytes()


def test_helper_selects_the_matching_visual_overlay_without_lod_substitution():
    for lod in LODS:
        assert HELPER.needle_ready_tissue_usd(lod) == (
            ASSET_ROOT / f"needle_ready_tissue_{lod}.usda"
        )
        assert HELPER.needle_ready_tissue_visual_usd(lod) == (
            ASSET_ROOT / f"needle_ready_tissue_{lod}_visual.usda"
        )
    with pytest.raises(ValueError):
        HELPER.needle_ready_tissue_visual_usd("prettier-than-physics")


def test_visual_texture_contract_is_2048_and_color_spaces_are_explicit():
    payload = _manifest()
    materials = (VISUAL_ROOT / "materials.usda").read_text(encoding="utf-8")
    assert payload["texture_resolution_px"] == 2048
    for material in (*TISSUE_MATERIALS, "drape", "cassette"):
        for suffix in ("basecolor", "roughness"):
            path = VISUAL_ROOT / "textures" / f"{material}_{suffix}.png"
            width, height = _png_dimensions(path)
            assert (width, height) == (2048, 2048)
        normal = VISUAL_ROOT / "textures" / f"{material}_normal.jpg"
        with Image.open(normal) as image:
            assert image.size == (2048, 2048)
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert JpegImagePlugin.get_sampling(image) == 0
        assert f"@textures/{material}_basecolor.png@" in materials
        assert f"@textures/{material}_roughness.png@" in materials
        assert f"@textures/{material}_normal.jpg@" in materials
    for material in TISSUE_MATERIALS:
        path = VISUAL_ROOT / "textures" / f"{material}_subsurface_weight.png"
        assert _png_dimensions(path) == (2048, 2048)
        assert f"@textures/{material}_subsurface_weight.png@" in materials
    for material in ("surface", "wound"):
        roughness = Image.open(
            VISUAL_ROOT / "textures" / f"{material}_roughness.png"
        )
        minimum, maximum = roughness.getextrema()
        assert minimum >= round(0.58 * 255.0) - 1
        assert maximum > minimum
    assert payload["texture_color_contract"]["basecolor"] == "sRGB"
    assert payload["texture_color_contract"]["roughness"] == "raw"
    assert payload["texture_color_contract"]["normal"].startswith("raw_")
    assert payload["texture_color_contract"]["subsurface_weight"] == "raw"
    assert payload["texture_encoding"]["normal"] == {
        "format": "JPEG",
        "subsampling": "4:4:4",
        "quality": 99,
    }
    assert materials.count('token inputs:sourceColorSpace = "sRGB"') == 6
    assert materials.count('token inputs:sourceColorSpace = "raw"') == 12


def test_materials_supply_openpbr_and_portable_restrained_preview_contexts():
    materials = (VISUAL_ROOT / "materials.usda").read_text(encoding="utf-8")
    assert materials.count('uniform token info:id = "UsdPreviewSurface"') == 6
    assert "OmniSurfaceBase.mdl" not in materials
    assert materials.count("inherits = </open_pbr_uber_base>") == 6
    assert "open_pbr_uber_base_class.usda" in materials
    assert materials.count("bool inputs:geometry_thin_walled = false") == 6
    assert materials.count("float inputs:coat_weight = 0") == 6
    assert materials.count("float inputs:metallic = 0") == 6
    assert materials.count("float inputs:clearcoat = 0") == 6
    for material in ("Surface", "Bulk", "Fascia", "Wound"):
        block = materials.split(f'def Material "{material}"', 1)[1].split(
            "        def Material ",
            1,
        )[0]
        assert "inputs:subsurface_weight_texture_file" in block
        assert "inputs:geometry_normal_texture_file" in block
        subsurface_weight = re.search(
            r"float inputs:subsurface_weight = ([0-9.]+)",
            block,
        )
        specular_weight = re.search(
            r"float inputs:specular_weight = ([0-9.]+)",
            block,
        )
        specular_roughness = re.search(
            r"float inputs:specular_roughness = ([0-9.]+)",
            block,
        )
        assert subsurface_weight and float(subsurface_weight.group(1)) > 0.0
        assert specular_weight and float(specular_weight.group(1)) <= 0.24
        assert specular_roughness and float(specular_roughness.group(1)) >= 0.62


def test_nvidia_openpbr_vendor_inputs_are_exact_and_provenanced():
    payload = _manifest()
    provenance = payload["provenance"]["nvidia"]
    assert provenance["release"] == "v0.2.0"
    assert provenance["license"] == "MIT-0"
    assert set(provenance["member_sha256"]) == {
        "LICENSE.md",
        "Skin_Medium_normal.jpg",
        "open_pbr_uber_base_class.usda",
    }
    vendor = (
        VISUAL_ROOT
        / "vendor/nvidia_physicalai_simready_materials_v0_2_0"
    )
    for name, expected_hash in provenance["member_sha256"].items():
        assert _sha256(vendor / name) == expected_hash
    open_pbr = (vendor / "open_pbr_uber_base_class.usda").read_text(
        encoding="utf-8"
    )
    assert 'uniform token info:id = "ND_open_pbr_surface_surfaceshader"' in open_pbr


def test_each_overlay_reuses_base_visual_topology_and_has_face_varying_uvs():
    report = json.loads((ASSET_ROOT / "geometry_report.json").read_text(encoding="utf-8"))
    for lod in LODS:
        overlay = (ASSET_ROOT / f"needle_ready_tissue_{lod}_visual.usda").read_text(
            encoding="utf-8"
        )
        assert (
            f"references = @./needle_ready_tissue_{lod}.usda@</DrAnmarNeedleReadyTissue>"
        ) in overlay
        assert (
            'string drAnmarVisualMeshBinding = "one_to_one_existing_deforming_visual_points"'
            in overlay
        )
        assert 'interpolation = "faceVarying"' in overlay
        uv_payload = overlay.split("texCoord2f[] primvars:st = [", 1)[1].split(
            "] (",
            1,
        )[0]
        uv_count = len(
            re.findall(
                r"\([-+0-9.eE]+,\s*[-+0-9.eE]+\)",
                uv_payload,
            )
        )
        expected_corners = 3 * report["lods"][lod]["surface_triangle_count"]
        assert uv_count == expected_corners
        for material in ("Surface", "Bulk", "Fascia", "Wound"):
            assert (
                f"rel material:binding = </DrAnmarNeedleReadyTissue/VisualMaterials/{material}>"
            ) in overlay


def test_visual_variant_root_composes_each_overlay_under_one_geometry_child():
    root = (ASSET_ROOT / "needle_ready_tissue_visual_unit.usda").read_text(encoding="utf-8")
    assert 'string geometryLod = "validation"' in root
    assert 'def Xform "Geometry"' in root
    for lod in LODS:
        assert (
            f"prepend references = "
            f"@./needle_ready_tissue_{lod}_visual.usda@"
            f"</DrAnmarNeedleReadyTissue>"
        ) in root


def test_support_is_visible_but_has_no_physics_or_collision_authority():
    support = (VISUAL_ROOT / "support_cassette.usda").read_text(encoding="utf-8")
    assert 'token purpose = "render"' in support
    assert "double drAnmarSupportedGapM = 0.017" in support
    assert "bool drAnmarCollisionAuthority = false" in support
    assert "bool drAnmarPhysicsAuthority = false" in support
    forbidden = (
        "PhysicsCollisionAPI",
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysxCollisionAPI",
        "collisionEnabled",
    )
    assert not any(token in support for token in forbidden)
