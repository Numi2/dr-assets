#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Generate render-only T1 overlays for the legacy PSM, table, and needle.

The referenced vendor/foundation USD files are immutable inputs.  This tool
creates stronger OpenUSD composition layers that:

* repair dangling PSM visual material relationships without changing joints,
  transforms, collision, mass, or other physics opinions;
* replace only the table's rendered appearance with a UV-authored operating
  table, while retaining the referenced legacy collision mesh; and
* give the legacy needle a restrained satin-steel appearance while labeling
  it compatibility-only; and
* provide a neutral surgical-field reference camera and soft-light rig without
  adding any task, physics, or clinical authority.

All geometry and 2048 px texture maps in this package are deterministic,
procedurally authored DrAnmar content.  NVIDIA's unmodified MIT-0 OpenPBR 1.1
MaterialX base class is copied from the repository's pinned v0.2.0 source and
used as the primary material context.  No patient data, scanned anatomy, or
third-party texture asset is used.

Run from the asset-extension root:

    uv run --no-project \
      --with numpy==2.2.6 \
      --with Pillow==11.3.0 \
      --with usd-core==25.11 \
      python tools/generate_t1_non_tissue_visuals.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from pxr import Usd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_SUBPATH = Path("data/Props/SurgicalScene/T1")
DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / ASSET_SUBPATH
MATERIAL_ROOT = "DrAnmarT1VisualMaterials"
TEXTURE_SIZE = 2048
PACKAGE_VERSION = "1.2.0"
GENERATOR_VERSIONS = {
    "numpy": "2.2.6",
    "Pillow": "11.3.0",
    "usd-core": "25.11",
}

BASE_ASSETS = {
    "psm": REPOSITORY_ROOT / "data/Robots/dVRK/PSM/psm_col.usd",
    "table": REPOSITORY_ROOT / "data/Props/Table/table.usd",
    "legacy_needle": (
        REPOSITORY_ROOT / "data/Props/Surgical_needle/needle_sdf.usd"
    ),
}
BASE_LICENSE = REPOSITORY_ROOT / "LICENSE"
APACHE_LICENSE_SOURCE = (
    REPOSITORY_ROOT
    / "data/Props/SurgicalClosure/Needle/LICENSE.txt"
)
NVIDIA_SOURCE_VENDOR_ROOT = (
    REPOSITORY_ROOT
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit/visual/vendor"
    / "nvidia_physicalai_simready_materials_v0_2_0"
)
NVIDIA_VENDOR_SUBPATH = Path(
    "vendor/nvidia_physicalai_simready_materials_v0_2_0"
)
NVIDIA_VENDOR_INPUTS = {
    "LICENSE.md": (
        "18f74283f08ff1ed39a9c46dbe2622146d45f771023c3dbd9c631bb058e1421b"
    ),
    "open_pbr_uber_base_class.usda": (
        "bb76ff9fa9cd74b86b6be4ed3c6ed79cdca15eff6d603ca571bdf9ce21e10c5f"
    ),
}
VALIDATOR_PATH = REPOSITORY_ROOT / "test/test_t1_non_tissue_visuals.py"
EXPECTED_MEMBER_PATHS = {
    "LICENSE.txt",
    "NOTICE.txt",
    "PROVENANCE.md",
    "README.md",
    "legacy_needle_visual_v1.usda",
    "materials.usda",
    "psm_visual_v1.usda",
    "reference_or_rig_v1.usda",
    "table_visual_v1.usda",
    (
        "vendor/nvidia_physicalai_simready_materials_v0_2_0/"
        "LICENSE.md"
    ),
    (
        "vendor/nvidia_physicalai_simready_materials_v0_2_0/"
        "PROVENANCE.md"
    ),
    (
        "vendor/nvidia_physicalai_simready_materials_v0_2_0/"
        "open_pbr_uber_base_class.usda"
    ),
    *{
        f"textures/{stem}_{suffix}.png"
        for stem in ("steel", "pad", "drape")
        for suffix in ("basecolor", "roughness")
    },
    "textures/steel_normal.png",
    "textures/pad_normal.png",
    "textures/drape_normal.png",
}

PSM_SHAFT_STEEL_MESHES = (
    "/psm/psm_main_insertion_link_2/visuals_xform/tool_main_insert",
)
PSM_WRIST_STEEL_MESHES = (
    "/psm/psm_tool_roll_link/visuals_xform/tool_roll_link",
    "/psm/psm_tool_pitch_link/visuals_xform/tool_pitch_link",
    "/psm/psm_tool_yaw_link/visuals_xform/tool_yaw_link",
)
PSM_JAW_STEEL_MESHES = (
    "/psm/psm_tool_gripper1_link/visuals_xform/gripper_right",
    "/psm/psm_tool_gripper2_link/visuals_xform/gripper_left",
)
PSM_MATTE_POLYMER_MESHES = (
    "/psm/psm_pitch_end_link/visuals_xform/visuals",
    "/psm/psm_main_insertion_link/visuals_xform/visuals",
    "/psm/psm_main_insertion_link_3/visuals_xform/visuals",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.rstrip() + "\n"
    path.write_text(normalized, encoding="utf-8", newline="\n")


def _save_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(values, 0.0, 1.0)
    pixels = np.round(clipped * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels)
    image.save(
        path,
        format="PNG",
        compress_level=9,
        optimize=False,
    )


def _resampled_noise(
    rng: np.random.Generator,
    size: int,
    coarse_size: int,
) -> np.ndarray:
    values = rng.random((coarse_size, coarse_size), dtype=np.float32)
    image = Image.fromarray(np.round(values * 65535.0).astype(np.uint16))
    image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 65535.0


def _multiscale_noise(
    rng: np.random.Generator,
    size: int,
    octaves: tuple[tuple[int, float], ...],
) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.float32)
    total = 0.0
    for coarse_size, weight in octaves:
        result += weight * _resampled_noise(rng, size, coarse_size)
        total += weight
    result /= total
    minimum = float(result.min())
    span = max(float(result.max()) - minimum, 1.0e-8)
    return ((result - minimum) / span).astype(np.float32)


def _normal_from_height(height: np.ndarray, strength: float) -> np.ndarray:
    gradient_y, gradient_x = np.gradient(height)
    normal = np.stack(
        (
            -gradient_x * float(strength),
            -gradient_y * float(strength),
            np.ones_like(height),
        ),
        axis=-1,
    )
    normal /= np.maximum(
        np.linalg.norm(normal, axis=-1, keepdims=True),
        1.0e-8,
    )
    return normal * 0.5 + 0.5


def _author_steel_textures(output: Path) -> list[Path]:
    size = TEXTURE_SIZE
    rng = np.random.default_rng(41117)
    coarse = _multiscale_noise(
        rng,
        size,
        ((7, 0.50), (23, 0.30), (83, 0.15), (311, 0.05)),
    )
    directional = rng.random((size, 1), dtype=np.float32)
    directional = np.repeat(directional, size, axis=1)
    directional = (
        np.asarray(
            Image.fromarray(
                np.round(directional * 255.0).astype(np.uint8),
            ).filter(ImageFilter.GaussianBlur(radius=0.8)),
            dtype=np.float32,
        )
        / 255.0
    )
    base_value = 0.49 + (coarse - 0.5) * 0.045
    base_color = np.stack(
        (
            base_value * 0.94,
            base_value * 0.98,
            base_value * 1.04,
        ),
        axis=-1,
    )
    roughness = np.clip(
        0.58 + (coarse - 0.5) * 0.10 + (directional - 0.5) * 0.07,
        0.48,
        0.68,
    )
    height = 0.30 * coarse + 0.70 * directional
    normal = _normal_from_height(height, 2.2)
    paths = [
        output / "steel_basecolor.png",
        output / "steel_roughness.png",
        output / "steel_normal.png",
    ]
    _save_png(paths[0], base_color)
    _save_png(paths[1], roughness)
    _save_png(paths[2], normal)
    return paths


def _author_pad_textures(output: Path) -> list[Path]:
    size = TEXTURE_SIZE
    rng = np.random.default_rng(58733)
    coarse = _multiscale_noise(
        rng,
        size,
        ((9, 0.50), (31, 0.30), (127, 0.15), (401, 0.05)),
    )
    pores = _multiscale_noise(
        rng,
        size,
        ((71, 0.30), (257, 0.70)),
    )
    base = np.stack(
        (
            0.050 + 0.020 * coarse,
            0.060 + 0.022 * coarse,
            0.067 + 0.024 * coarse,
        ),
        axis=-1,
    )
    roughness = np.clip(
        0.78 + (coarse - 0.5) * 0.09 + (pores - 0.5) * 0.06,
        0.68,
        0.88,
    )
    normal = _normal_from_height(
        0.55 * coarse + 0.45 * pores,
        2.0,
    )
    paths = [
        output / "pad_basecolor.png",
        output / "pad_roughness.png",
        output / "pad_normal.png",
    ]
    (output / "pad_normal.jpg").unlink(missing_ok=True)
    _save_png(paths[0], base)
    _save_png(paths[1], roughness)
    _save_png(paths[2], normal)
    return paths


def _author_drape_textures(output: Path) -> list[Path]:
    size = TEXTURE_SIZE
    rng = np.random.default_rng(73199)
    coarse = _multiscale_noise(
        rng,
        size,
        ((7, 0.55), (29, 0.30), (113, 0.15)),
    )
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    warp = (
        0.50
        + 0.25 * np.sin((xx + 0.17 * yy) * (2.0 * math.pi / 8.7))
        + 0.25 * np.sin((xx - 0.11 * yy) * (2.0 * math.pi / 13.1))
    )
    weft = (
        0.50
        + 0.25 * np.sin((yy + 0.09 * xx) * (2.0 * math.pi / 9.3))
        + 0.25 * np.sin((yy - 0.13 * xx) * (2.0 * math.pi / 14.7))
    )
    weave = np.clip(0.5 * warp + 0.5 * weft, 0.0, 1.0)
    color_variation = 0.62 * coarse + 0.38 * weave
    base = np.stack(
        (
            0.040 + 0.022 * color_variation,
            0.245 + 0.050 * color_variation,
            0.275 + 0.054 * color_variation,
        ),
        axis=-1,
    )
    roughness = np.clip(
        0.90 + (coarse - 0.5) * 0.07 + (weave - 0.5) * 0.035,
        0.84,
        0.96,
    )
    height = 0.34 * coarse + 0.66 * weave
    normal = _normal_from_height(height, 3.3)
    paths = [
        output / "drape_basecolor.png",
        output / "drape_roughness.png",
        output / "drape_normal.png",
    ]
    (output / "drape_normal.jpg").unlink(missing_ok=True)
    _save_png(paths[0], base)
    _save_png(paths[1], roughness)
    _save_png(paths[2], normal)
    return paths


def _usd_number(value: float) -> str:
    if abs(float(value)) < 5.0e-13:
        return "0"
    return f"{float(value):.9g}"


def _usd_vec(values: Sequence[float]) -> str:
    return "(" + ", ".join(_usd_number(value) for value in values) + ")"


def _usd_array(
    values: Iterable[str],
    *,
    indent: str = "            ",
) -> str:
    items = list(values)
    if not items:
        return "[]"
    return "[\n" + "".join(
        f"{indent}{value}{',' if index + 1 < len(items) else ''}\n"
        for index, value in enumerate(items)
    ) + indent[:-4] + "]"


def _uniform_material(
    name: str,
    *,
    diffuse: tuple[float, float, float],
    metallic: float,
    roughness: float,
    ior: float,
    specular_weight: float,
    roughness_anisotropy: float = 0.0,
) -> str:
    path = f"/{MATERIAL_ROOT}/{name}"
    return f'''    def Material "{name}" (
        inherits = </open_pbr_uber_base>
    )
    {{
        token outputs:surface.connect = <{path}/PreviewSurface.outputs:surface>
        custom string drAnmarOpenPBRInputMode = "constants_no_uv_contract"
        custom string drAnmarPreviewFallback = "UsdPreviewSurface"

        float inputs:base_weight = 1
        color3f inputs:base_color = {_usd_vec(diffuse)}
        float inputs:base_diffuse_roughness = 0
        float inputs:base_metalness = {_usd_number(metallic)}
        float inputs:specular_weight = {_usd_number(specular_weight)}
        color3f inputs:specular_color = (1, 1, 1)
        float inputs:specular_roughness = {_usd_number(roughness)}
        float inputs:specular_roughness_anisotropy = {_usd_number(roughness_anisotropy)}
        float inputs:specular_ior = {_usd_number(ior)}
        float inputs:coat_weight = 0
        float inputs:subsurface_weight = 0
        float inputs:transmission_weight = 0
        bool inputs:geometry_thin_walled = false

        def Shader "PreviewSurface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = {_usd_vec(diffuse)}
            float inputs:metallic = {_usd_number(metallic)}
            float inputs:roughness = {_usd_number(roughness)}
            float inputs:ior = {_usd_number(ior)}
            float inputs:clearcoat = 0
            float inputs:clearcoatRoughness = 1
            token outputs:surface
        }}
    }}'''


def _textured_material(
    name: str,
    *,
    texture_stem: str,
    diffuse_fallback: tuple[float, float, float],
    metallic: float,
    roughness_fallback: float,
    ior: float,
    specular_weight: float,
    roughness_anisotropy: float = 0.0,
    thin_walled: bool = False,
    fuzz_weight: float = 0.0,
) -> str:
    path = f"/{MATERIAL_ROOT}/{name}"
    normal_extension = "png"
    return f'''    def Material "{name}" (
        inherits = </open_pbr_uber_base>
    )
    {{
        token outputs:surface.connect = <{path}/PreviewSurface.outputs:surface>
        custom string drAnmarOpenPBRInputMode = "uv_texture_maps"
        custom string drAnmarPreviewFallback = "UsdPreviewSurface"

        float inputs:base_weight = 1
        color3f inputs:base_color = {_usd_vec(diffuse_fallback)}
        asset inputs:base_color_texture_file = @textures/{texture_stem}_basecolor.png@ (
            colorSpace = "sRGB"
        )
        float inputs:base_diffuse_roughness = 0
        float inputs:base_metalness = {_usd_number(metallic)}
        float inputs:specular_weight = {_usd_number(specular_weight)}
        color3f inputs:specular_color = (1, 1, 1)
        float inputs:specular_roughness = {_usd_number(roughness_fallback)}
        float inputs:specular_roughness_anisotropy = {_usd_number(roughness_anisotropy)}
        asset inputs:specular_roughness_texture_file = @textures/{texture_stem}_roughness.png@ (
            colorSpace = "raw"
        )
        float inputs:specular_ior = {_usd_number(ior)}
        float inputs:fuzz_weight = {_usd_number(fuzz_weight)}
        color3f inputs:fuzz_color = {_usd_vec(diffuse_fallback)}
        float inputs:fuzz_roughness = 0.92
        float inputs:coat_weight = 0
        float inputs:subsurface_weight = 0
        float inputs:transmission_weight = 0
        bool inputs:geometry_thin_walled = {"true" if thin_walled else "false"}
        float inputs:geometry_normal_scale = 1
        asset inputs:geometry_normal_texture_file = @textures/{texture_stem}_normal.{normal_extension}@ (
            colorSpace = "raw"
        )
        bool inputs:geometry_normal_texture_flip_g = false

        def Shader "PreviewSurface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = <{path}/BaseColorTexture.outputs:rgb>
            float inputs:metallic = {_usd_number(metallic)}
            float inputs:roughness.connect = <{path}/RoughnessTexture.outputs:r>
            float inputs:ior = {_usd_number(ior)}
            float inputs:clearcoat = 0
            float inputs:clearcoatRoughness = 1
            normal3f inputs:normal.connect = <{path}/NormalTexture.outputs:rgb>
            token outputs:surface
        }}

        def Shader "Texcoord"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}

        def Shader "BaseColorTexture"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @textures/{texture_stem}_basecolor.png@
            token inputs:sourceColorSpace = "sRGB"
            float2 inputs:st.connect = <{path}/Texcoord.outputs:result>
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float3 outputs:rgb
        }}

        def Shader "RoughnessTexture"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @textures/{texture_stem}_roughness.png@
            token inputs:sourceColorSpace = "raw"
            float2 inputs:st.connect = <{path}/Texcoord.outputs:result>
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            float outputs:r
        }}

        def Shader "NormalTexture"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @textures/{texture_stem}_normal.{normal_extension}@
            token inputs:sourceColorSpace = "raw"
            float4 inputs:scale = (2, 2, 2, 1)
            float4 inputs:bias = (-1, -1, -1, 0)
            float2 inputs:st.connect = <{path}/Texcoord.outputs:result>
            token inputs:wrapS = "repeat"
            token inputs:wrapT = "repeat"
            normal3f outputs:rgb
        }}
    }}'''


def author_materials() -> str:
    blocks = [
        _uniform_material(
            "ShaftSatinSteel",
            diffuse=(0.40, 0.43, 0.47),
            metallic=0.91,
            roughness=0.58,
            ior=1.50,
            specular_weight=0.88,
            roughness_anisotropy=0.34,
        ),
        _uniform_material(
            "WristSatinSteel",
            diffuse=(0.43, 0.46, 0.50),
            metallic=0.91,
            roughness=0.54,
            ior=1.50,
            specular_weight=0.88,
            roughness_anisotropy=0.20,
        ),
        _uniform_material(
            "JawSatinSteel",
            diffuse=(0.36, 0.39, 0.42),
            metallic=0.90,
            roughness=0.62,
            ior=1.50,
            specular_weight=0.84,
            roughness_anisotropy=0.10,
        ),
        _uniform_material(
            "NeedleSatinSteel",
            diffuse=(0.39, 0.42, 0.46),
            metallic=0.88,
            roughness=0.62,
            ior=1.50,
            specular_weight=0.78,
            roughness_anisotropy=0.16,
        ),
        _uniform_material(
            "MattePolymer",
            diffuse=(0.028, 0.034, 0.040),
            metallic=0.0,
            roughness=0.82,
            ior=1.46,
            specular_weight=0.30,
        ),
        _textured_material(
            "TableFrameSteel",
            texture_stem="steel",
            diffuse_fallback=(0.39, 0.42, 0.46),
            metallic=0.86,
            roughness_fallback=0.58,
            ior=1.50,
            specular_weight=0.86,
            roughness_anisotropy=0.24,
        ),
        _textured_material(
            "TablePad",
            texture_stem="pad",
            diffuse_fallback=(0.055, 0.064, 0.071),
            metallic=0.0,
            roughness_fallback=0.78,
            ior=1.46,
            specular_weight=0.32,
        ),
        _textured_material(
            "SterileDrape",
            texture_stem="drape",
            diffuse_fallback=(0.050, 0.27, 0.30),
            metallic=0.0,
            roughness_fallback=0.90,
            ior=1.43,
            specular_weight=0.12,
            thin_walled=True,
            fuzz_weight=0.12,
        ),
    ]
    return f'''#usda 1.0
(
    defaultPrim = "{MATERIAL_ROOT}"
    doc = "DrAnmar T1 render-only materials: NVIDIA OpenPBR 1.1 MaterialX primary with portable PreviewSurface fallback."
    metersPerUnit = 1
    subLayers = [
        @./vendor/nvidia_physicalai_simready_materials_v0_2_0/open_pbr_uber_base_class.usda@
    ]
    upAxis = "Z"
)

def Scope "{MATERIAL_ROOT}" (
    customData = {{
        string drAnmarAssetRole = "render_materials_only"
        bool drAnmarClinicalValidation = false
        bool drAnmarPhysicsAuthority = false
        string drAnmarFallbackMaterialContext = "UsdPreviewSurface"
        string drAnmarPrimaryMaterialContext = "OpenPBR_1_1_MaterialX"
        string drAnmarTextureColorContract = "basecolor_sRGB_roughness_and_normal_raw"
        string drAnmarVendorDependency = "NVIDIA_PhysicalAI_SimReady_Materials_v0.2.0_MIT-0"
        string drAnmarVersion = "{PACKAGE_VERSION}"
    }}
)
{{
{chr(10).join(blocks)}
}}
'''


def _new_tree_node() -> dict[str, Any]:
    return {
        "children": {},
        "material": None,
        "clear_material": False,
    }


def _tree_insert(
    tree: dict[str, Any],
    path: str,
    *,
    root: str,
    material: str | None = None,
    clear_material: bool = False,
) -> None:
    parts = [part for part in path.split("/") if part]
    if not parts or parts[0] != root:
        raise ValueError(f"{path!r} is not below /{root}")
    node = tree
    for part in parts[1:]:
        node = node["children"].setdefault(part, _new_tree_node())
    if material is not None:
        node["material"] = material
    if clear_material:
        node["clear_material"] = True


def _render_over_children(
    node: dict[str, Any],
    *,
    root: str,
    depth: int,
) -> str:
    blocks: list[str] = []
    indent = "    " * depth
    for name in sorted(node["children"]):
        child = node["children"][name]
        api_block = ""
        if child["material"] is not None:
            api_block = ' (\n' + indent + '    prepend apiSchemas = ["MaterialBindingAPI"]\n' + indent + ")"
        lines = [f'{indent}over "{name}"{api_block}', f"{indent}{{"]
        if child["clear_material"]:
            lines.append(f"{indent}    rel material:binding = None")
        if child["material"] is not None:
            lines.extend(
                (
                    f"{indent}    rel material:binding = "
                    f"</{root}/DrAnmarT1Looks/{child['material']}> (",
                    f'{indent}        bindMaterialAs = "strongerThanDescendants"',
                    f"{indent}    )",
                )
            )
        nested = _render_over_children(
            child,
            root=root,
            depth=depth + 1,
        )
        if nested:
            lines.append(nested)
        lines.append(f"{indent}}}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _dangling_psm_material_prim_paths() -> tuple[str, ...]:
    stage = Usd.Stage.Open(str(BASE_ASSETS["psm"]))
    if stage is None:
        raise RuntimeError(f"cannot open {BASE_ASSETS['psm']}")
    missing: list[str] = []
    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            if not relationship.GetName().startswith("material:binding"):
                continue
            if any(
                not stage.GetPrimAtPath(target).IsValid()
                for target in relationship.GetTargets()
            ):
                missing.append(str(prim.GetPath()))
    if len(missing) != 100:
        raise RuntimeError(
            "expected 100 known dangling PSM bindings, found "
            f"{len(missing)}"
        )
    return tuple(sorted(missing))


def author_psm_overlay() -> str:
    tree = _new_tree_node()
    for path in PSM_SHAFT_STEEL_MESHES:
        _tree_insert(tree, path, root="psm", material="ShaftSatinSteel")
    for path in PSM_WRIST_STEEL_MESHES:
        _tree_insert(tree, path, root="psm", material="WristSatinSteel")
    for path in PSM_JAW_STEEL_MESHES:
        _tree_insert(tree, path, root="psm", material="JawSatinSteel")
    for path in PSM_MATTE_POLYMER_MESHES:
        _tree_insert(tree, path, root="psm", material="MattePolymer")
    for path in _dangling_psm_material_prim_paths():
        _tree_insert(tree, path, root="psm", clear_material=True)
    overrides = _render_over_children(tree, root="psm", depth=1)
    base_hash = sha256(BASE_ASSETS["psm"])
    return f'''#usda 1.0
(
    defaultPrim = "psm"
    doc = "DrAnmar T1 render-only PSM material overlay; referenced articulation and physics remain authoritative."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "psm" (
    prepend references = @../../../Robots/dVRK/PSM/psm_col.usd@</psm>
    assetInfo = {{
        string name = "DrAnmarT1PSMVisualOverlay"
        string version = "{PACKAGE_VERSION}"
    }}
    customData = {{
        string drAnmarAssetRole = "render_overlay_only"
        string drAnmarBaseAssetSha256 = "{base_hash}"
        bool drAnmarClinicalValidation = false
        bool drAnmarPhysicsAuthority = false
        bool drAnmarPreservesJointAndLinkTransforms = true
        int drAnmarClearedDanglingMaterialBindings = 100
    }}
    kind = "component"
)
{{
    def Scope "DrAnmarT1Looks" (
        prepend references = @./materials.usda@</{MATERIAL_ROOT}>
    )
    {{
    }}

{overrides}
}}
'''


def _mesh_block(
    name: str,
    *,
    points: Sequence[Sequence[float]],
    face_counts: Sequence[int],
    face_indices: Sequence[int],
    normals: Sequence[Sequence[float]],
    normal_interpolation: str,
    uvs: Sequence[Sequence[float]],
    uv_interpolation: str,
    material: str,
    double_sided: bool = False,
    indent: str = "        ",
) -> str:
    if sum(face_counts) != len(face_indices):
        raise ValueError(f"{name}: face counts do not match indices")
    if normal_interpolation == "vertex" and len(normals) != len(points):
        raise ValueError(f"{name}: vertex normal count does not match points")
    if normal_interpolation == "faceVarying" and len(normals) != len(face_indices):
        raise ValueError(f"{name}: face-varying normals do not match corners")
    if uv_interpolation == "vertex" and len(uvs) != len(points):
        raise ValueError(f"{name}: vertex UV count does not match points")
    if uv_interpolation == "faceVarying" and len(uvs) != len(face_indices):
        raise ValueError(f"{name}: face-varying UVs do not match corners")
    values = np.asarray(points, dtype=np.float64)
    extent_min = values.min(axis=0)
    extent_max = values.max(axis=0)
    inner = indent + "    "
    return f'''{indent}def Mesh "{name}" (
{inner}prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{
{inner}uniform bool doubleSided = {"true" if double_sided else "false"}
{inner}float3[] extent = [{_usd_vec(extent_min)}, {_usd_vec(extent_max)}]
{inner}int[] faceVertexCounts = {_usd_array((str(value) for value in face_counts), indent=inner + "    ")}
{inner}int[] faceVertexIndices = {_usd_array((str(value) for value in face_indices), indent=inner + "    ")}
{inner}normal3f[] normals = {_usd_array((_usd_vec(value) for value in normals), indent=inner + "    ")} (
{inner}    interpolation = "{normal_interpolation}"
{inner})
{inner}point3f[] points = {_usd_array((_usd_vec(value) for value in points), indent=inner + "    ")}
{inner}texCoord2f[] primvars:st = {_usd_array((_usd_vec(value) for value in uvs), indent=inner + "    ")} (
{inner}    interpolation = "{uv_interpolation}"
{inner})
{inner}token purpose = "render"
{inner}rel material:binding = </Table/DrAnmarT1Looks/{material}>
{inner}uniform token subdivisionScheme = "none"
{indent}}}'''


def _box_geometry(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> tuple[
    list[tuple[float, float, float]],
    list[int],
    list[int],
    list[tuple[float, float, float]],
    list[tuple[float, float]],
]:
    cx, cy, cz = center
    hx, hy, hz = (value * 0.5 for value in size)
    faces = (
        ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), (0, 0, 1)),
        ((-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz), (0, 0, -1)),
        ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz), (0, -1, 0)),
        ((hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz), (1, 0, 0)),
        ((hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (0, 1, 0)),
        ((-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-1, 0, 0)),
    )
    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    for face in faces:
        normal = face[4]
        for x, y, z in face[:4]:
            points.append((cx + x, cy + y, cz + z))
            normals.append(normal)
        uvs.extend(((0, 0), (1, 0), (1, 1), (0, 1)))
    return points, [4] * 6, list(range(24)), normals, uvs


def _rounded_rectangle_perimeter(
    half_x: float,
    half_y: float,
    radius: float,
    segments_per_corner: int,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    centers = (
        (half_x - radius, half_y - radius, 0.0),
        (-half_x + radius, half_y - radius, 90.0),
        (-half_x + radius, -half_y + radius, 180.0),
        (half_x - radius, -half_y + radius, 270.0),
    )
    for center_x, center_y, start_degrees in centers:
        for index in range(segments_per_corner):
            fraction = index / segments_per_corner
            angle = math.radians(start_degrees + 90.0 * fraction)
            points.append(
                (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                )
            )
    return points


def _rounded_prism_geometry(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    corner_radius: float,
    edge_bevel: float,
    segments_per_corner: int = 8,
) -> tuple[
    list[tuple[float, float, float]],
    list[int],
    list[int],
    list[tuple[float, float, float]],
    list[tuple[float, float]],
]:
    cx, cy, cz = center
    half_x, half_y, half_z = (value * 0.5 for value in size)
    if not 0.0 < edge_bevel < min(half_x, half_y, half_z):
        raise ValueError("edge bevel must fit inside the rounded prism")
    if not edge_bevel < corner_radius < min(half_x, half_y):
        raise ValueError("corner radius must exceed bevel and fit footprint")
    z_bottom = cz - half_z
    z_top = cz + half_z
    outer = _rounded_rectangle_perimeter(
        half_x,
        half_y,
        corner_radius,
        segments_per_corner,
    )
    inner = _rounded_rectangle_perimeter(
        half_x - edge_bevel,
        half_y - edge_bevel,
        corner_radius - edge_bevel,
        segments_per_corner,
    )
    rings = (
        (inner, z_top, 0.0, 1.0),
        (outer, z_top - edge_bevel, 1.0, 1.0),
        (outer, z_bottom + edge_bevel, 1.0, -1.0),
        (inner, z_bottom, 0.0, -1.0),
    )
    points: list[tuple[float, float, float]] = [(cx, cy, z_top)]
    normals: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    for ring, z_value, radial_weight, z_weight in rings:
        for x_value, y_value in ring:
            corner_center_x = float(
                np.clip(
                    x_value,
                    -half_x + corner_radius,
                    half_x - corner_radius,
                )
            )
            corner_center_y = float(
                np.clip(
                    y_value,
                    -half_y + corner_radius,
                    half_y - corner_radius,
                )
            )
            radial = np.asarray(
                (
                    x_value - corner_center_x,
                    y_value - corner_center_y,
                    0.0,
                ),
                dtype=np.float64,
            )
            radial /= max(float(np.linalg.norm(radial)), 1.0e-8)
            normal = np.asarray(
                (
                    radial[0] * radial_weight,
                    radial[1] * radial_weight,
                    z_weight,
                ),
                dtype=np.float64,
            )
            normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
            points.append((cx + x_value, cy + y_value, z_value))
            normals.append(tuple(map(float, normal)))
    bottom_center = len(points)
    points.append((cx, cy, z_bottom))
    normals.append((0.0, 0.0, -1.0))
    count = len(outer)
    ring_offsets = tuple(1 + index * count for index in range(4))
    faces: list[list[int]] = []
    for index in range(count):
        next_index = (index + 1) % count
        faces.append(
            [0, ring_offsets[0] + index, ring_offsets[0] + next_index]
        )
    for ring_index in range(3):
        first = ring_offsets[ring_index]
        second = ring_offsets[ring_index + 1]
        for index in range(count):
            next_index = (index + 1) % count
            faces.append(
                [
                    first + index,
                    second + index,
                    second + next_index,
                    first + next_index,
                ]
            )
    for index in range(count):
        next_index = (index + 1) % count
        faces.append(
            [
                bottom_center,
                ring_offsets[3] + next_index,
                ring_offsets[3] + index,
            ]
        )
    face_counts = [len(face) for face in faces]
    face_indices = [value for face in faces for value in face]
    uvs: list[tuple[float, float]] = []
    for face in faces:
        for point_index in face:
            world_x, world_y, z_value = points[point_index]
            x_value = world_x - cx
            y_value = world_y - cy
            if z_value >= z_top - 1.0e-9:
                uvs.append(
                    (
                        x_value / (2.0 * half_x) + 0.5,
                        y_value / (2.0 * half_y) + 0.5,
                    )
                )
            elif z_value <= z_bottom + 1.0e-9:
                uvs.append(
                    (
                        x_value / (2.0 * half_x) + 0.5,
                        y_value / (2.0 * half_y) + 0.5,
                    )
                )
            else:
                angle = math.atan2(
                    y_value / half_y,
                    x_value / half_x,
                )
                uvs.append(
                    (
                        (angle + math.pi) / (2.0 * math.pi),
                        (z_value - z_bottom) / (z_top - z_bottom),
                    )
                )
    return points, face_counts, face_indices, normals, uvs


def _pad_geometry() -> tuple[
    list[tuple[float, float, float]],
    list[int],
    list[int],
    list[tuple[float, float, float]],
    list[tuple[float, float]],
]:
    return _rounded_prism_geometry(
        center=(0.0, 0.0, 0.424),
        size=(1.66, 0.70, 0.065),
        corner_radius=0.065,
        edge_bevel=0.009,
        segments_per_corner=12,
    )


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _drape_geometry(
    nx: int = 81,
    ny: int = 49,
) -> tuple[
    list[tuple[float, float, float]],
    list[int],
    list[int],
    list[tuple[float, float, float]],
    list[tuple[float, float]],
]:
    x_values = np.linspace(-0.89, 0.89, nx, dtype=np.float64)
    y_values = np.linspace(-0.44, 0.44, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x_values, y_values)
    drop_x = _smoothstep((np.abs(xx) - 0.80) / 0.09)
    drop_y = _smoothstep((np.abs(yy) - 0.325) / 0.115)
    edge_drop = 1.0 - (1.0 - drop_x) * (1.0 - drop_y)
    center_mask = 1.0 - np.clip(edge_drop, 0.0, 1.0)
    wrinkles = (
        0.00070 * np.sin(19.0 * xx + 7.0 * yy)
        + 0.00045 * np.sin(31.0 * yy - 4.0 * xx)
        + 0.00025 * np.sin(53.0 * xx + 17.0 * yy)
    )
    z_values = 0.4585 - 0.098 * edge_drop + wrinkles * (
        0.35 + 0.65 * center_mask
    )
    dz_dy, dz_dx = np.gradient(
        z_values,
        y_values,
        x_values,
    )
    normals_array = np.stack(
        (-dz_dx, -dz_dy, np.ones_like(z_values)),
        axis=-1,
    )
    normals_array /= np.maximum(
        np.linalg.norm(normals_array, axis=-1, keepdims=True),
        1.0e-8,
    )
    u_values = (xx - x_values.min()) / (
        x_values.max() - x_values.min()
    )
    v_values = (yy - y_values.min()) / (
        y_values.max() - y_values.min()
    )
    points = [
        (float(xx[row, column]), float(yy[row, column]), float(z_values[row, column]))
        for row in range(ny)
        for column in range(nx)
    ]
    normals = [
        tuple(map(float, normals_array[row, column]))
        for row in range(ny)
        for column in range(nx)
    ]
    uvs = [
        (float(u_values[row, column]), float(v_values[row, column]))
        for row in range(ny)
        for column in range(nx)
    ]
    faces: list[list[int]] = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            lower_left = row * nx + column
            lower_right = lower_left + 1
            upper_left = (row + 1) * nx + column
            upper_right = upper_left + 1
            faces.append(
                [lower_left, lower_right, upper_right, upper_left]
            )
    return (
        points,
        [4] * len(faces),
        [value for face in faces for value in face],
        normals,
        uvs,
    )


def _table_meshes() -> str:
    frame_parts = (
        (
            "DeckFrame",
            (0.0, 0.0, 0.368),
            (1.74, 0.76, 0.050),
            0.025,
            0.008,
        ),
        (
            "CenterColumn",
            (0.0, 0.0, -0.035),
            (0.22, 0.32, 0.756),
            0.035,
            0.012,
        ),
        (
            "FloorBase",
            (0.0, 0.0, -0.425),
            (1.02, 0.60, 0.058),
            0.040,
            0.010,
        ),
        (
            "LeftFoot",
            (-0.61, 0.0, -0.420),
            (0.26, 0.16, 0.068),
            0.028,
            0.010,
        ),
        (
            "RightFoot",
            (0.61, 0.0, -0.420),
            (0.26, 0.16, 0.068),
            0.028,
            0.010,
        ),
    )
    blocks: list[str] = []
    for name, center, size, radius, bevel in frame_parts:
        points, counts, indices, normals, uvs = _rounded_prism_geometry(
            center,
            size,
            corner_radius=radius,
            edge_bevel=bevel,
        )
        blocks.append(
            _mesh_block(
                name,
                points=points,
                face_counts=counts,
                face_indices=indices,
                normals=normals,
                normal_interpolation="vertex",
                uvs=uvs,
                uv_interpolation="faceVarying",
                material="TableFrameSteel",
                indent="            ",
            )
        )
    points, counts, indices, normals, uvs = _pad_geometry()
    pad = _mesh_block(
        "RoundedPad",
        points=points,
        face_counts=counts,
        face_indices=indices,
        normals=normals,
        normal_interpolation="vertex",
        uvs=uvs,
        uv_interpolation="faceVarying",
        material="TablePad",
        indent="        ",
    )
    points, counts, indices, normals, uvs = _drape_geometry()
    drape = _mesh_block(
        "SterileDrape",
        points=points,
        face_counts=counts,
        face_indices=indices,
        normals=normals,
        normal_interpolation="vertex",
        uvs=uvs,
        uv_interpolation="vertex",
        material="SterileDrape",
        double_sided=True,
        indent="        ",
    )
    return f'''    def Xform "DrAnmarT1Visual"
    {{
        token purpose = "render"

        def Xform "Frame"
        {{
{chr(10).join(blocks)}
        }}

{pad}

{drape}
    }}'''


def author_table_overlay() -> str:
    base_hash = sha256(BASE_ASSETS["table"])
    return f'''#usda 1.0
(
    defaultPrim = "Table"
    doc = "DrAnmar T1 render-only operating-table overlay; legacy collision remains composed and authoritative."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Table" (
    prepend references = @../../Table/table.usd@</Table>
    assetInfo = {{
        string name = "DrAnmarT1OperatingTableVisualOverlay"
        string version = "{PACKAGE_VERSION}"
    }}
    customData = {{
        string drAnmarAssetRole = "render_overlay_only"
        string drAnmarBaseAssetSha256 = "{base_hash}"
        bool drAnmarClinicalValidation = false
        bool drAnmarPhysicsAuthority = false
        bool drAnmarPreservesLegacyCollision = true
        string drAnmarRenderGeometry = "independently_generated_uv_authored_table_frame_pad_and_drape"
    }}
    kind = "component"
)
{{
    def Scope "DrAnmarT1Looks" (
        prepend references = @./materials.usda@</{MATERIAL_ROOT}>
    )
    {{
    }}

    over "Table"
    {{
        over "Table"
        {{
            token visibility = "invisible"
        }}
    }}

{_table_meshes()}
}}
'''


def author_legacy_needle_overlay() -> str:
    base_hash = sha256(BASE_ASSETS["legacy_needle"])
    return f'''#usda 1.0
(
    defaultPrim = "Needle"
    doc = "Compatibility-only render overlay for the legacy T1 needle; not a geometry or dynamics promotion."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Needle" (
    prepend references = @../../Surgical_needle/needle_sdf.usd@</Needle>
    assetInfo = {{
        string name = "DrAnmarT1LegacyNeedleVisualOverlay"
        string version = "{PACKAGE_VERSION}"
    }}
    customData = {{
        string drAnmarAssetRole = "render_overlay_only"
        string drAnmarBaseAssetSha256 = "{base_hash}"
        bool drAnmarClinicalValidation = false
        bool drAnmarCompatibilityOnly = true
        bool drAnmarGeometryPromotion = false
        bool drAnmarPhysicsAuthority = false
        string drAnmarGeometryStatus = "legacy_disconnected_shells_no_topology_repair"
        string drAnmarReplacementTarget = "DrAnmar watertight needle after grasp and retention parity"
    }}
    kind = "component"
)
{{
    def Scope "DrAnmarT1Looks" (
        prepend references = @./materials.usda@</{MATERIAL_ROOT}>
    )
    {{
    }}

    over "Needle"
    {{
        over "Needle" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            rel material:binding = </Needle/DrAnmarT1Looks/NeedleSatinSteel> (
                bindMaterialAs = "strongerThanDescendants"
            )
        }}
    }}
}}
'''


def _look_at_matrix(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> str:
    position_vector = np.asarray(position, dtype=np.float64)
    target_vector = np.asarray(target, dtype=np.float64)
    backward = position_vector - target_vector
    backward /= max(float(np.linalg.norm(backward)), 1.0e-12)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(world_up, backward)
    if float(np.linalg.norm(right)) < 1.0e-8:
        world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(world_up, backward)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    camera_up = np.cross(backward, right)
    camera_up /= max(float(np.linalg.norm(camera_up)), 1.0e-12)
    rows = (
        (*map(float, right), 0.0),
        (*map(float, camera_up), 0.0),
        (*map(float, backward), 0.0),
        (*map(float, position_vector), 1.0),
    )
    return "(" + ", ".join(_usd_vec(row) for row in rows) + ")"


def author_reference_or_rig() -> str:
    field_target = (0.0, 0.0, 0.455)
    key_transform = _look_at_matrix((0.52, -0.48, 1.32), field_target)
    fill_transform = _look_at_matrix((-0.58, -0.18, 1.08), field_target)
    rim_transform = _look_at_matrix((0.12, 0.68, 1.15), field_target)
    camera_transform = _look_at_matrix((0.72, -0.78, 0.96), field_target)
    return f'''#usda 1.0
(
    defaultPrim = "DrAnmarT1ReferenceRig"
    doc = "Neutral render-reference camera and soft surgical-field lighting for DrAnmar T1 asset review."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "DrAnmarT1ReferenceRig" (
    assetInfo = {{
        string name = "DrAnmarT1ReferenceORRig"
        string version = "{PACKAGE_VERSION}"
    }}
    customData = {{
        string drAnmarAssetRole = "render_reference_only"
        bool drAnmarClinicalValidation = false
        bool drAnmarPhysicsAuthority = false
        string drAnmarLightingIntent = "neutral_soft_4300K_surgical_field_asset_review"
        string drAnmarQualification = "source_authored_native_RTX_review_pending"
    }}
    kind = "component"
)
{{
    def Scope "Lighting"
    {{
        def DomeLight "NeutralAmbient"
        {{
            color3f inputs:color = (0.91, 0.94, 1)
            float inputs:exposure = 0
            float inputs:intensity = 0.18
            bool inputs:normalize = true
        }}

        def RectLight "SurgicalKey"
        {{
            color3f inputs:color = (1, 1, 1)
            float inputs:colorTemperature = 4300
            bool inputs:enableColorTemperature = true
            float inputs:exposure = 0
            float inputs:height = 0.42
            float inputs:intensity = 1450
            bool inputs:normalize = true
            float inputs:width = 0.90
            matrix4d xformOp:transform = {key_transform}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}

        def RectLight "SoftFill"
        {{
            color3f inputs:color = (1, 1, 1)
            float inputs:colorTemperature = 5000
            bool inputs:enableColorTemperature = true
            float inputs:exposure = 0
            float inputs:height = 0.48
            float inputs:intensity = 650
            bool inputs:normalize = true
            float inputs:width = 0.72
            matrix4d xformOp:transform = {fill_transform}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}

        def RectLight "EdgeSeparation"
        {{
            color3f inputs:color = (1, 1, 1)
            float inputs:colorTemperature = 4600
            bool inputs:enableColorTemperature = true
            float inputs:exposure = 0
            float inputs:height = 0.30
            float inputs:intensity = 520
            bool inputs:normalize = true
            float inputs:width = 0.62
            matrix4d xformOp:transform = {rim_transform}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}
    }}

    def Camera "ReferenceCamera"
    {{
        float2 clippingRange = (0.05, 8)
        double focalLength = 58
        float horizontalAperture = 20.955
        token projection = "perspective"
        float verticalAperture = 15.2908
        matrix4d xformOp:transform = {camera_transform}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
}}
'''


def author_readme() -> str:
    return f"""# DrAnmar T1 Non-Tissue Visual Package {PACKAGE_VERSION}

This package supplies render-only OpenUSD overlays for the active T1 scene:

- `psm_visual_v1.usda` repairs unresolved visual bindings and assigns restrained
  satin steel and matte polymer without changing the referenced articulation;
- `table_visual_v1.usda` hides only the legacy table render mesh and adds an
  independently generated UV-authored rounded-edge table frame, upholstered
  pad, and woven sterile drape while retaining legacy collision;
- `legacy_needle_visual_v1.usda` deliberately subdues the compatibility
  needle instead of making its obsolete geometry visually dominant; and
- `reference_or_rig_v1.usda` provides a neutral 58 mm reference camera and
  three broad surgical-field lights for consistent asset review.

The legacy needle remains compatibility-only. Its disconnected source geometry,
mass properties, and collision contract are not repaired or promoted here.

## Authority boundary

These layers are not physics, collision, joint, transform, task, reward,
success, or clinical authority. The PSM and needle wrappers reference their
existing source assets. The table wrapper changes `visibility` only on
`/Table/Table/Table`; its collision opinion remains composed from the base
layer. New table geometry has `purpose = "render"` and no physics API.

All textures are deterministic 2048 x 2048 procedural DrAnmar work. There is no
patient imagery, scanned anatomy, web download, or third-party texture content.
Each material inherits NVIDIA's OpenPBR 1.1 MaterialX base class, vendored
unchanged from PhysicalAI-SimReady-Materials v0.2.0 under MIT-0.
`UsdPreviewSurface` remains the portable fallback.

The UV-authored table frame, pad, and drape feed their 2K base-color,
specular-roughness, and tangent-normal maps into both the OpenPBR and Preview
paths. Every normal map is lossless PNG. PSM shaft, wrist, jaws, and polymer
parts receive distinct, restrained material responses; none use a chrome-like
clear coat. The PSM and compatibility needle source meshes do not provide a
qualified UV contract, so their materials deliberately use constants rather
than pretending texture fidelity.

## Regeneration

From the asset-extension root:

```bash
uv run --no-project \\
  --with numpy==2.2.6 \\
  --with Pillow==11.3.0 \\
  --with usd-core==25.11 \\
  python tools/generate_t1_non_tissue_visuals.py
```

Native Isaac/RTX appearance, articulation spawning, and frame/contact parity
remain separate qualification steps.
This package is research-only, not clinically validated, and not approved for
patient care.
"""


def author_provenance() -> str:
    base_rows = "\n".join(
        f"- `{path.relative_to(REPOSITORY_ROOT).as_posix()}`: "
        f"`sha256:{sha256(path)}`"
        for path in BASE_ASSETS.values()
    )
    return f"""# Provenance

## DrAnmar-authored content

The overlay composition, derived material parameter overrides, UV-authored
operating-table render geometry, and all nine 2048 px texture maps were
independently and
deterministically authored by the Dr.Anmar project for package version
`{PACKAGE_VERSION}`. The generator seeds and pinned dependency versions are
recorded in `asset_manifest.json`.

No patient data, clinical photography, scanned anatomy, purchased model,
internet image, or third-party texture is included.

## Referenced immutable foundations

The overlays reference, but do not copy or rewrite, these repository assets:

{base_rows}

The PSM foundation is ORBIT-Surgical-derived and retains its BSD-3-Clause
repository attribution. Asset-level origin metadata for the legacy table and
needle remains incomplete; this package does not upgrade or obscure that
boundary. Their exact byte hashes are locked in the manifest.

## Vendored NVIDIA OpenPBR dependency

The package includes byte-identical copies of NVIDIA's
`open_pbr_uber_base_class.usda` and `LICENSE.md` from
PhysicalAI-SimReady-Materials v0.2.0. They are MIT-0 inputs and are not
DrAnmar-authored. Exact source and destination hashes are locked in the
manifest. No NVIDIA texture, geometry, or patient data is included.

The inherited base supplies the OpenPBR 1.1 MaterialX graph. DrAnmar authors
only material parameter overrides and retains a `UsdPreviewSurface` fallback.
This is a source interoperability contract, not evidence of native RTX visual
quality.

## Validation boundary

The checked-in gates establish deterministic source integrity, OpenUSD
composition/reference closure, resolved visual material targets, UV coverage,
and absence of overlay-authored physics opinions. They do not establish native
Isaac spawn behavior, RTX appearance, physical calibration, or clinical
validity.
"""


def author_notice() -> str:
    return f"""DrAnmar T1 Non-Tissue Visual Package {PACKAGE_VERSION}

Copyright 2026 Dr.Anmar Project Developers.

The OpenUSD overlays, generated table render geometry, material graphs, UVs,
and textures in this directory are independently authored DrAnmar content
licensed under Apache-2.0.

Referenced PSM, table, and legacy needle assets remain separate repository
foundations under their existing notices and terms.
This package does not rewrite or relicense those files.

The unmodified NVIDIA OpenPBR 1.1 MaterialX base class and its MIT-0 license
are redistributed from PhysicalAI-SimReady-Materials v0.2.0. See
`vendor/nvidia_physicalai_simready_materials_v0_2_0/PROVENANCE.md`.
No third-party texture or model asset is redistributed.

Research simulation only. Not clinically validated and not for patient care.
"""


def author_vendor_provenance() -> str:
    return f"""# NVIDIA PhysicalAI SimReady Material Input

This directory vendors two immutable inputs from NVIDIA's
`PhysicalAI-SimReady-Materials` release `v0.2.0`:

- `open_pbr_uber_base_class.usda`
- `LICENSE.md`

Upstream:
`https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials`

The source copies used by this generator are pinned inside this repository at:
`data/Props/SurgicalTissue/NeedleReadyTissueUnit/visual/vendor/nvidia_physicalai_simready_materials_v0_2_0`

Vendored member SHA-256 values:

- `open_pbr_uber_base_class.usda`:
  `{NVIDIA_VENDOR_INPUTS["open_pbr_uber_base_class.usda"]}`
- `LICENSE.md`:
  `{NVIDIA_VENDOR_INPUTS["LICENSE.md"]}`

Both files are copied byte-for-byte and remain MIT-0 NVIDIA content. The base
class provides an OpenPBR 1.1 MaterialX graph. This package uses no NVIDIA
texture, geometry, biomechanics, patient data, or clinical calibration.

This provenance document is DrAnmar-authored under Apache-2.0.
"""


def _member_entry(path: Path, asset_root: Path) -> dict[str, Any]:
    relative = path.relative_to(asset_root).as_posix()
    vendor_prefix = NVIDIA_VENDOR_SUBPATH.as_posix() + "/"
    if relative in {
        f"{vendor_prefix}LICENSE.md",
        f"{vendor_prefix}open_pbr_uber_base_class.usda",
    }:
        license_id = "MIT-0"
        provenance = (
            "verbatim_NVIDIA_PhysicalAI_SimReady_Materials_v0.2.0"
        )
    elif relative == "LICENSE.txt":
        license_id = "Apache-2.0"
        provenance = (
            "canonical_Apache-2.0_text_copied_from_repository_license"
        )
    elif path.suffix == ".png" or path.suffix.startswith(".usd"):
        license_id = "Apache-2.0"
        provenance = "deterministically_generated_by_DrAnmar"
    else:
        license_id = "Apache-2.0"
        provenance = "authored_by_DrAnmar"
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "license": license_id,
        "provenance": provenance,
        "relative_path": relative,
    }


def _base_entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "modified_by_generator": False,
        "license_path": BASE_LICENSE.relative_to(REPOSITORY_ROOT).as_posix(),
        "license_sha256": sha256(BASE_LICENSE),
        "provenance_status": (
            "repository foundation locked by exact bytes; asset-level "
            "origin metadata remains as documented upstream"
        ),
    }


def _vendor_nvidia_openpbr(asset_root: Path) -> None:
    destination_root = asset_root / NVIDIA_VENDOR_SUBPATH
    destination_root.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in NVIDIA_VENDOR_INPUTS.items():
        source = NVIDIA_SOURCE_VENDOR_ROOT / name
        if not source.is_file():
            raise RuntimeError(f"missing pinned NVIDIA source input: {source}")
        source_hash = sha256(source)
        if source_hash != expected_hash:
            raise RuntimeError(
                f"NVIDIA source hash mismatch for {name}: "
                f"{source_hash} != {expected_hash}"
            )
        destination = destination_root / name
        shutil.copyfile(source, destination)
        destination_hash = sha256(destination)
        if destination_hash != expected_hash:
            raise RuntimeError(
                f"NVIDIA vendored copy hash mismatch for {name}: "
                f"{destination_hash} != {expected_hash}"
            )
    _write_text(
        destination_root / "PROVENANCE.md",
        author_vendor_provenance(),
    )


def author_manifest(asset_root: Path) -> dict[str, Any]:
    generated_files = sorted(
        (
            path
            for path in asset_root.rglob("*")
            if path.is_file() and path.name != "asset_manifest.json"
        ),
        key=lambda path: path.relative_to(asset_root).as_posix(),
    )
    member_paths = {
        path.relative_to(asset_root).as_posix()
        for path in generated_files
    }
    if member_paths != EXPECTED_MEMBER_PATHS:
        missing = sorted(EXPECTED_MEMBER_PATHS - member_paths)
        unexpected = sorted(member_paths - EXPECTED_MEMBER_PATHS)
        raise RuntimeError(
            "visual package member set is not deterministic: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "schema": "dr.anmar.t1-non-tissue-visual-package.v2",
        "asset_id": "dranmar-t1-non-tissue-visuals",
        "version": PACKAGE_VERSION,
        "status": "source_generated_native_qualification_pending",
        "provider": "DrAnmar Project Developers",
        "package_role": "render_overlay_bundle_with_external_pinned_dependencies",
        "dependency_complete_directory": False,
        "license": "mixed",
        "licenses": ["Apache-2.0", "BSD-3-Clause", "MIT-0"],
        "license_sources": {
            "dranmar": {
                "license": "Apache-2.0",
                "path": APACHE_LICENSE_SOURCE.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "sha256": sha256(APACHE_LICENSE_SOURCE),
            },
            "nvidia_openpbr": {
                "license": "MIT-0",
                "path": (
                    NVIDIA_VENDOR_SUBPATH / "LICENSE.md"
                ).as_posix(),
                "sha256": NVIDIA_VENDOR_INPUTS["LICENSE.md"],
            },
            "repository_foundations": {
                "license": "BSD-3-Clause",
                "path": BASE_LICENSE.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": sha256(BASE_LICENSE),
                "scope": "externally_referenced_psm_table_and_legacy_needle_foundations",
            },
        },
        "visual_only": True,
        "physics_authority": False,
        "collision_authority": False,
        "clinical_validation": False,
        "vendor_assets_modified": False,
        "vendor_dependencies": {
            "nvidia_physicalai_simready_materials": {
                "provider": "NVIDIA",
                "upstream": (
                    "https://github.com/NVIDIA-Omniverse/"
                    "PhysicalAI-SimReady-Materials"
                ),
                "release": "v0.2.0",
                "license": "MIT-0",
                "material_interface": "OpenPBR_1.1_MaterialX",
                "source_root": NVIDIA_SOURCE_VENDOR_ROOT.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "destination_root": NVIDIA_VENDOR_SUBPATH.as_posix(),
                "members": {
                    name: {
                        "sha256": expected_hash,
                        "copied_byte_identical": True,
                        "modified_by_generator": False,
                    }
                    for name, expected_hash in NVIDIA_VENDOR_INPUTS.items()
                },
            }
        },
        "material_contexts": {
            "primary": "OpenPBR_1.1_MaterialX_inherited",
            "fallback": "UsdPreviewSurface",
            "mdl_only_claim": False,
            "native_rtx_appearance_qualified": False,
        },
        "entrypoints": {
            "psm": "psm_visual_v1.usda",
            "table": "table_visual_v1.usda",
            "legacy_needle": "legacy_needle_visual_v1.usda",
            "reference_or_rig": "reference_or_rig_v1.usda",
        },
        "generator": {
            "path": Path(__file__).resolve()
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
            "versions": GENERATOR_VERSIONS,
            "texture_size_px": TEXTURE_SIZE,
            "seeds": {
                "steel": 41117,
                "pad": 58733,
                "drape": 73199,
            },
        },
        "validator": {
            "path": VALIDATOR_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": (
                sha256(VALIDATOR_PATH)
                if VALIDATOR_PATH.is_file()
                else None
            ),
        },
        "base_assets": {
            name: _base_entry(path)
            for name, path in BASE_ASSETS.items()
        },
        "composition_contract": {
            "psm_root": "/psm",
            "table_root": "/Table",
            "legacy_needle_root": "/Needle",
            "cleared_dangling_psm_material_bindings": 100,
            "table_legacy_render_mesh": "/Table/Table/Table",
            "table_legacy_collision_preserved": True,
            "legacy_needle_compatibility_only": True,
            "existing_joint_link_and_asset_transforms_authored": False,
            "physics_or_physx_properties_authored": False,
        },
        "texture_color_contract": {
            "basecolor": "sRGB",
            "roughness": "raw",
            "normal": (
                "raw_tangent_space_renderer_orientation_pending_"
                "native_validation"
            ),
        },
        "texture_encoding": {
            "all_basecolor_roughness_and_normal_maps": "lossless_PNG",
            "native_rtx_qualified": False,
        },
        "material_input_contract": {
            "table_uv_authored": True,
            "table_openpbr_texture_inputs": [
                "base_color_texture_file",
                "specular_roughness_texture_file",
                "geometry_normal_texture_file",
            ],
            "psm_uv_qualified": False,
            "psm_openpbr_input_mode": "constants",
            "legacy_needle_uv_qualified": False,
            "legacy_needle_openpbr_input_mode": "constants",
        },
        "members": {
            path.relative_to(asset_root).as_posix(): _member_entry(
                path,
                asset_root,
            )
            for path in generated_files
        },
        "qualification_boundary": {
            "source_static": "implemented",
            "native_isaac_spawn": "pending",
            "rtx_reference_render": "pending",
            "physics_contact_or_retention": "not_claimed",
            "clinical": "not_validated",
        },
    }


def generate(asset_root: Path) -> None:
    base_hashes_before = {
        name: sha256(path) for name, path in BASE_ASSETS.items()
    }
    asset_root.mkdir(parents=True, exist_ok=True)
    texture_root = asset_root / "textures"
    texture_root.mkdir(parents=True, exist_ok=True)

    _vendor_nvidia_openpbr(asset_root)
    _author_steel_textures(texture_root)
    _author_pad_textures(texture_root)
    _author_drape_textures(texture_root)
    _write_text(asset_root / "materials.usda", author_materials())
    _write_text(asset_root / "psm_visual_v1.usda", author_psm_overlay())
    _write_text(
        asset_root / "table_visual_v1.usda",
        author_table_overlay(),
    )
    _write_text(
        asset_root / "legacy_needle_visual_v1.usda",
        author_legacy_needle_overlay(),
    )
    _write_text(
        asset_root / "reference_or_rig_v1.usda",
        author_reference_or_rig(),
    )
    _write_text(asset_root / "README.md", author_readme())
    _write_text(asset_root / "PROVENANCE.md", author_provenance())
    _write_text(asset_root / "NOTICE.txt", author_notice())
    _write_text(
        asset_root / "LICENSE.txt",
        APACHE_LICENSE_SOURCE.read_text(encoding="utf-8"),
    )

    base_hashes_after = {
        name: sha256(path) for name, path in BASE_ASSETS.items()
    }
    if base_hashes_after != base_hashes_before:
        raise RuntimeError("a referenced base asset changed during generation")

    manifest = author_manifest(asset_root)
    _write_text(
        asset_root / "asset_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="output directory for the generated visual package",
    )
    args = parser.parse_args()
    generate(args.asset_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
