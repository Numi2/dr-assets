#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Generate the visual-only package for the needle-ready tissue asset.

The checked-in training, contact, and validation USD layers are physics
authority and are never rewritten by this tool.  Generated overlay layers
reference those assets, add face-varying UVs to the existing one-to-one
deforming Visual mesh, rebind its material subsets, and add a render-purpose
support cassette.  The textures are deterministic procedural research art;
they contain no patient data and are not clinically calibrated.

Run from the asset-extension root:

    uv run --no-project --with numpy==2.2.6 --with Pillow==11.3.0 \
      python tools/generate_needle_ready_tissue_visuals.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT_PRIM = "DrAnmarNeedleReadyTissue"
MATERIAL_ROOT = "NeedleReadyTissueVisualMaterials"
SUPPORT_ROOT = "DrAnmarTissueRenderSupport"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_SUBPATH = Path("data/Props/SurgicalTissue/NeedleReadyTissueUnit")
DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / ASSET_SUBPATH
BASE_GENERATOR_PATH = REPOSITORY_ROOT / "tools/generate_needle_ready_tissue.py"
TEXTURE_SIZE = 2048
NORMAL_JPEG_QUALITY = 99
GENERATOR_VERSIONS = {
    "numpy": "2.2.6",
    "Pillow": "11.3.0",
}
NVIDIA_VENDOR_SUBPATH = Path(
    "visual/vendor/nvidia_physicalai_simready_materials_v0_2_0"
)
NVIDIA_OPENPBR_BASE = NVIDIA_VENDOR_SUBPATH / "open_pbr_uber_base_class.usda"
NVIDIA_SKIN_NORMAL = NVIDIA_VENDOR_SUBPATH / "Skin_Medium_normal.jpg"
NVIDIA_VENDOR_HASHES = {
    "LICENSE.md": "18f74283f08ff1ed39a9c46dbe2622146d45f771023c3dbd9c631bb058e1421b",
    "Skin_Medium_normal.jpg": (
        "a881565f49e80e5c486f84fbd1e87595515199d67063009e68415e791d190bc5"
    ),
    "open_pbr_uber_base_class.usda": (
        "bb76ff9fa9cd74b86b6be4ed3c6ed79cdca15eff6d603ca571bdf9ce21e10c5f"
    ),
}
BASE_PHYSICS_FILES = (
    "needle_ready_tissue_training.usda",
    "needle_ready_tissue_contact.usda",
    "needle_ready_tissue_validation.usda",
    "needle_ready_tissue_unit.usda",
)
LODS = ("training", "contact", "validation")


MATERIALS: dict[str, dict[str, Any]] = {
    "surface": {
        "seed": 14071,
        "base_srgb": (0.54, 0.18, 0.16),
        "variation": (0.095, 0.060, 0.050),
        "roughness": 0.74,
        "roughness_variation": 0.12,
        "moisture_drop": 0.20,
        "normal_strength": 2.2,
        "vascular_strength": 1.0,
        "specular_weight": 0.22,
        "specular_roughness": 0.66,
        "subsurface_weight": 0.72,
        "subsurface_color": (0.70, 0.27, 0.20),
        "subsurface_radius_m": (0.0010, 0.00045, 0.00022),
    },
    "bulk": {
        "seed": 27103,
        "base_srgb": (0.60, 0.25, 0.22),
        "variation": (0.085, 0.055, 0.045),
        "roughness": 0.79,
        "roughness_variation": 0.11,
        "moisture_drop": 0.15,
        "normal_strength": 1.8,
        "fiber": True,
        "fiber_color_strength": 0.52,
        "fiber_height_strength": 0.20,
        "specular_weight": 0.18,
        "specular_roughness": 0.71,
        "subsurface_weight": 0.80,
        "subsurface_color": (0.72, 0.31, 0.23),
        "subsurface_radius_m": (0.0013, 0.00058, 0.00028),
    },
    "fascia": {
        "seed": 39019,
        "base_srgb": (0.66, 0.49, 0.34),
        "variation": (0.070, 0.060, 0.045),
        "roughness": 0.84,
        "roughness_variation": 0.10,
        "moisture_drop": 0.11,
        "normal_strength": 3.0,
        "fiber": True,
        "fiber_color_strength": 1.15,
        "fiber_height_strength": 0.48,
        "specular_weight": 0.13,
        "specular_roughness": 0.77,
        "subsurface_weight": 0.42,
        "subsurface_color": (0.70, 0.53, 0.36),
        "subsurface_radius_m": (0.00070, 0.00036, 0.00019),
    },
    "wound": {
        "seed": 51283,
        "base_srgb": (0.29, 0.045, 0.035),
        "variation": (0.080, 0.035, 0.030),
        "roughness": 0.69,
        "roughness_variation": 0.13,
        "moisture_drop": 0.21,
        "normal_strength": 1.7,
        "vascular_strength": 0.42,
        "specular_weight": 0.24,
        "specular_roughness": 0.62,
        "subsurface_weight": 0.86,
        "subsurface_color": (0.49, 0.08, 0.055),
        "subsurface_radius_m": (0.0016, 0.00062, 0.00028),
    },
    "drape": {
        "seed": 62861,
        "base_srgb": (0.055, 0.30, 0.34),
        "variation": (0.025, 0.050, 0.055),
        "roughness": 0.89,
        "roughness_variation": 0.08,
        "moisture_drop": 0.0,
        "normal_strength": 3.6,
        "fiber": True,
        "fiber_color_strength": 0.72,
        "fiber_height_strength": 0.56,
        "specular_weight": 0.08,
        "specular_roughness": 0.84,
        "subsurface_weight": 0.0,
        "subsurface_color": (0.055, 0.30, 0.34),
        "subsurface_radius_m": (0.0, 0.0, 0.0),
    },
    "cassette": {
        "seed": 74093,
        "base_srgb": (0.065, 0.080, 0.088),
        "variation": (0.018, 0.020, 0.022),
        "roughness": 0.57,
        "roughness_variation": 0.08,
        "moisture_drop": 0.0,
        "normal_strength": 1.2,
        "specular_weight": 0.18,
        "specular_roughness": 0.63,
        "subsurface_weight": 0.0,
        "subsurface_color": (0.065, 0.080, 0.088),
        "subsurface_radius_m": (0.0, 0.0, 0.0),
    },
}


def _load_base_generator():
    spec = importlib.util.spec_from_file_location(
        "dranmar_needle_ready_tissue_geometry_generator",
        BASE_GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resampled_noise(
    rng: np.random.Generator,
    size: int,
    coarse_size: int,
) -> np.ndarray:
    values = rng.random((coarse_size, coarse_size), dtype=np.float32)
    image = Image.fromarray(np.round(values * 65535.0).astype(np.uint16))
    image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    result = np.asarray(image, dtype=np.float32) / 65535.0
    return result


def _multiscale_noise(
    rng: np.random.Generator,
    size: int,
    octaves: tuple[tuple[int, float], ...],
) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.float32)
    total_weight = 0.0
    for coarse_size, weight in octaves:
        result += weight * _resampled_noise(rng, size, coarse_size)
        total_weight += weight
    result /= total_weight
    minimum = float(result.min())
    span = max(float(result.max()) - minimum, 1.0e-8)
    return (result - minimum) / span


def _sparse_moisture(
    rng: np.random.Generator,
    size: int,
    *,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    field = np.zeros((size, size), dtype=np.float32)
    for _ in range(11):
        center_x = rng.uniform(0.0, size)
        center_y = rng.uniform(0.0, size)
        sigma_x = rng.uniform(size * 0.012, size * 0.045)
        sigma_y = rng.uniform(size * 0.008, size * 0.032)
        angle = rng.uniform(0.0, np.pi)
        dx = xx - center_x
        dy = yy - center_y
        rotated_x = np.cos(angle) * dx + np.sin(angle) * dy
        rotated_y = -np.sin(angle) * dx + np.cos(angle) * dy
        blob = np.exp(-0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2))
        field = np.maximum(field, blob.astype(np.float32))
    return np.clip((field - 0.62) / 0.38, 0.0, 1.0) ** 2


def _fiber_pattern(
    rng: np.random.Generator,
    size: int,
    phase: float,
) -> np.ndarray:
    """Build irregular anisotropic bundles instead of periodic sine stripes."""

    def directional(coarse_size: int, angle_degrees: float) -> np.ndarray:
        values = rng.random((1, coarse_size), dtype=np.float32)
        image = Image.fromarray(np.round(values * 65535.0).astype(np.uint16))
        image = image.resize(
            (size * 2, size * 2),
            resample=Image.Resampling.BICUBIC,
        )
        image = image.rotate(
            angle_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
        )
        image = image.crop((size // 2, size // 2, size + size // 2, size + size // 2))
        return np.asarray(image, dtype=np.float32) / 65535.0

    angle = 15.0 + 12.0 * np.sin(phase)
    primary = 0.62 * directional(91, angle) + 0.38 * directional(277, angle)
    cross = directional(137, angle + 71.0)
    result = 0.84 * primary + 0.16 * cross
    minimum = float(result.min())
    return ((result - minimum) / max(float(result.max()) - minimum, 1.0e-8)).astype(np.float32)


def _vascular_mask(
    rng: np.random.Generator,
    size: int,
    *,
    branch_count: int,
) -> np.ndarray:
    """Create sparse, soft branching lines without source imagery."""

    canvas = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(canvas)

    def draw_branch(
        start: tuple[float, float],
        direction: float,
        segments: int,
        width: int,
        value: int,
        depth: int,
    ) -> None:
        points = [start]
        x, y = start
        for _ in range(segments):
            direction += rng.normal(0.0, 0.16 + 0.05 * depth)
            length = rng.uniform(0.022, 0.052) * size * (0.82**depth)
            x += float(np.cos(direction) * length)
            y += float(np.sin(direction) * length)
            if not (-0.04 * size <= x <= 1.04 * size):
                break
            if not (-0.04 * size <= y <= 1.04 * size):
                break
            points.append((x, y))
        if len(points) < 2:
            return
        draw.line(points, fill=value, width=width, joint="curve")
        if depth >= 2 or len(points) < 5:
            return
        split_candidates = np.linspace(2, len(points) - 2, 3, dtype=int)
        for split_at in split_candidates:
            if rng.random() > (0.72 if depth == 0 else 0.48):
                continue
            source = points[int(split_at)]
            child_direction = direction + rng.choice((-1.0, 1.0)) * rng.uniform(
                0.38,
                0.78,
            )
            draw_branch(
                source,
                child_direction,
                max(3, int(segments * rng.uniform(0.40, 0.62))),
                max(1, width - 1),
                max(38, value - 28),
                depth + 1,
            )

    for _ in range(branch_count):
        edge = int(rng.integers(0, 4))
        if edge == 0:
            start = (-0.02 * size, rng.uniform(0.06, 0.94) * size)
            direction = rng.normal(0.0, 0.30)
        elif edge == 1:
            start = (1.02 * size, rng.uniform(0.06, 0.94) * size)
            direction = np.pi + rng.normal(0.0, 0.30)
        elif edge == 2:
            start = (rng.uniform(0.06, 0.94) * size, -0.02 * size)
            direction = np.pi / 2.0 + rng.normal(0.0, 0.30)
        else:
            start = (rng.uniform(0.06, 0.94) * size, 1.02 * size)
            direction = -np.pi / 2.0 + rng.normal(0.0, 0.30)
        draw_branch(
            start,
            float(direction),
            int(rng.integers(13, 22)),
            int(rng.integers(2, 5)),
            int(rng.integers(80, 150)),
            0,
        )
    blurred = canvas.filter(ImageFilter.GaussianBlur(radius=1.15))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    encoded = np.round(clipped * 255.0).astype(np.uint8)
    Image.fromarray(encoded).save(
        path,
        format="PNG",
        compress_level=9,
        optimize=False,
    )


def _save_normal_jpeg(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    encoded = np.round(clipped * 255.0).astype(np.uint8)
    Image.fromarray(encoded).save(
        path,
        format="JPEG",
        quality=NORMAL_JPEG_QUALITY,
        subsampling=0,
        optimize=False,
        progressive=False,
    )


def _blend_published_skin_microstructure(
    procedural_normal: np.ndarray,
    source_path: Path,
    *,
    size: int,
) -> np.ndarray:
    """Blend NVIDIA's published skin micro-normal into DrAnmar macro relief.

    The source map is retained and hashed separately. The derived output uses
    a bounded micro-detail contribution so the physical surface remains
    governed by the deforming mesh rather than texture displacement.
    """

    with Image.open(source_path) as source_image:
        resized = source_image.convert("RGB").resize(
            (size, size),
            resample=Image.Resampling.LANCZOS,
        )
        published = np.asarray(resized, dtype=np.float32) / 255.0
    published = published * 2.0 - 1.0
    combined = procedural_normal.copy()
    combined[..., 0] += 0.34 * published[..., 0]
    combined[..., 1] += 0.34 * published[..., 1]
    combined[..., 2] = np.maximum(
        0.12,
        combined[..., 2] * (0.90 + 0.10 * np.clip(published[..., 2], 0.0, 1.0)),
    )
    combined /= np.maximum(
        np.linalg.norm(combined, axis=-1, keepdims=True),
        1.0e-8,
    )
    return combined


def generate_material_textures(
    output_dir: Path,
    name: str,
    spec: dict[str, Any],
    *,
    size: int,
    published_skin_normal: Path | None = None,
) -> list[Path]:
    rng = np.random.default_rng(int(spec["seed"]))
    macro = _multiscale_noise(
        rng,
        size,
        ((5, 0.35), (11, 0.28), (27, 0.21), (67, 0.16)),
    )
    micro = _multiscale_noise(
        rng,
        size,
        ((31, 0.26), (73, 0.28), (151, 0.25), (293, 0.21)),
    )
    chroma_a = _multiscale_noise(rng, size, ((7, 0.55), (23, 0.45)))
    chroma_b = _multiscale_noise(rng, size, ((9, 0.52), (37, 0.48)))
    moisture = _sparse_moisture(
        rng,
        size,
        enabled=float(spec["moisture_drop"]) > 0.0,
    )

    structured = np.zeros_like(macro)
    if bool(spec.get("fiber", False)):
        structured = _fiber_pattern(
            rng,
            size,
            rng.uniform(0.0, 2.0 * np.pi),
        )

    base = np.asarray(spec["base_srgb"], dtype=np.float32)
    variation = np.asarray(spec["variation"], dtype=np.float32)
    signed_macro = (macro - 0.5)[..., None]
    signed_micro = (micro - 0.5)[..., None]
    chroma = np.stack(
        (
            chroma_a - 0.5,
            chroma_b - 0.5,
            0.55 * (chroma_a - chroma_b),
        ),
        axis=-1,
    )
    base_color = (
        base[None, None, :]
        + signed_macro * variation[None, None, :]
        + 0.32 * chroma * variation[None, None, :]
        + 0.10 * signed_micro * variation[None, None, :]
    )
    if bool(spec.get("fiber", False)):
        base_color += (
            (structured - 0.5)[..., None]
            * variation[None, None, :]
            * float(spec.get("fiber_color_strength", 0.30))
        )
    vascular_strength = float(spec.get("vascular_strength", 0.0))
    if vascular_strength > 0.0:
        vessels = _vascular_mask(
            rng,
            size,
            branch_count=5 if name == "surface" else 3,
        )
        base_color[..., 0] -= 0.080 * vascular_strength * vessels
        base_color[..., 1] -= 0.050 * vascular_strength * vessels
        base_color[..., 2] += 0.020 * vascular_strength * vessels
    base_color *= 1.0 - 0.025 * moisture[..., None]

    roughness = (
        float(spec["roughness"])
        + float(spec["roughness_variation"]) * (0.58 * macro + 0.42 * micro - 0.5)
        - float(spec["moisture_drop"]) * moisture
    )
    minimum_roughness = 0.58 if name in {"surface", "wound"} else 0.55
    roughness = np.clip(roughness, minimum_roughness, 0.96)

    height = 0.58 * micro + 0.30 * macro
    if bool(spec.get("fiber", False)):
        height += float(spec.get("fiber_height_strength", 0.35)) * structured
    gradient_y, gradient_x = np.gradient(height)
    strength = float(spec["normal_strength"])
    normal = np.stack(
        (-gradient_x * strength, -gradient_y * strength, np.ones_like(height)),
        axis=-1,
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-8)
    if published_skin_normal is not None:
        normal = _blend_published_skin_microstructure(
            normal,
            published_skin_normal,
            size=size,
        )
    normal = normal * 0.5 + 0.5

    paths = [
        output_dir / f"{name}_basecolor.png",
        output_dir / f"{name}_roughness.png",
        output_dir / f"{name}_normal.jpg",
    ]
    _save_png(paths[0], base_color)
    _save_png(paths[1], roughness)
    _save_normal_jpeg(paths[2], normal)
    if float(spec["subsurface_weight"]) > 0.0:
        subsurface_weight = np.clip(
            float(spec["subsurface_weight"])
            * (0.88 + 0.18 * macro + 0.06 * moisture),
            0.0,
            1.0,
        )
        paths.append(output_dir / f"{name}_subsurface_weight.png")
        _save_png(paths[-1], subsurface_weight)
    return paths


def _usd_number(value: float) -> str:
    return f"{float(value):.9g}"


def _usd_vec(values: tuple[float, ...] | list[float]) -> str:
    return "(" + ", ".join(_usd_number(value) for value in values) + ")"


def _preview_material_block(name: str, spec: dict[str, Any]) -> str:
    title = name.title()
    texture_stem = name.lower()
    base_color = tuple(float(value) for value in spec["base_srgb"])
    sss_color = tuple(float(value) for value in spec["subsurface_color"])
    sss_radius = tuple(float(value) for value in spec["subsurface_radius_m"])
    sss_weight = float(spec["subsurface_weight"])
    scalar_radius = max(sss_radius)
    radius_scale = (
        tuple(value / scalar_radius for value in sss_radius)
        if scalar_radius > 0.0
        else (1.0, 1.0, 1.0)
    )
    subsurface_texture = (
        f'''            asset inputs:subsurface_weight_texture_file = @textures/{texture_stem}_subsurface_weight.png@ (
                colorSpace = "raw"
            )
'''
        if sss_weight > 0.0
        else ""
    )
    return f'''        def Material "{title}" (
            inherits = </open_pbr_uber_base>
        )
        {{
            token outputs:surface.connect = <{_material_path(title)}/PreviewSurface.outputs:surface>

            float inputs:base_weight = 1
            color3f inputs:base_color = {_usd_vec(base_color)}
            asset inputs:base_color_texture_file = @textures/{texture_stem}_basecolor.png@ (
                colorSpace = "sRGB"
            )
            float inputs:base_diffuse_roughness = 0.18
            float inputs:base_metalness = 0
            float inputs:specular_weight = {_usd_number(float(spec["specular_weight"]))}
            color3f inputs:specular_color = (1, 1, 1)
            float inputs:specular_roughness = {_usd_number(float(spec["specular_roughness"]))}
            asset inputs:specular_roughness_texture_file = @textures/{texture_stem}_roughness.png@ (
                colorSpace = "raw"
            )
            float inputs:specular_ior = 1.38
            float inputs:subsurface_weight = {_usd_number(sss_weight)}
{subsurface_texture}            color3f inputs:subsurface_color = {_usd_vec(sss_color)}
            float inputs:subsurface_radius = {_usd_number(scalar_radius)}
            color3f inputs:subsurface_radius_scale = {_usd_vec(radius_scale)}
            float inputs:coat_weight = 0
            bool inputs:geometry_thin_walled = false
            float inputs:geometry_normal_scale = 1
                asset inputs:geometry_normal_texture_file = @textures/{texture_stem}_normal.jpg@ (
                colorSpace = "raw"
            )
            bool inputs:geometry_normal_texture_flip_g = false

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{_material_path(title)}/BaseColorTexture.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness.connect = <{_material_path(title)}/RoughnessTexture.outputs:r>
                float inputs:ior = 1.38
                float inputs:clearcoat = 0
                float inputs:clearcoatRoughness = 1
                normal3f inputs:normal.connect = <{_material_path(title)}/NormalTexture.outputs:rgb>
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
                float2 inputs:st.connect = <{_material_path(title)}/Texcoord.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}

            def Shader "RoughnessTexture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/{texture_stem}_roughness.png@
                token inputs:sourceColorSpace = "raw"
                float2 inputs:st.connect = <{_material_path(title)}/Texcoord.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float outputs:r
            }}

            def Shader "NormalTexture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @textures/{texture_stem}_normal.jpg@
                token inputs:sourceColorSpace = "raw"
                float4 inputs:scale = (2, 2, 2, 1)
                float4 inputs:bias = (-1, -1, -1, 0)
                float2 inputs:st.connect = <{_material_path(title)}/Texcoord.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                normal3f outputs:rgb
            }}
        }}'''


def _material_path(title: str) -> str:
    return f"/{MATERIAL_ROOT}/{title}"


def author_materials() -> str:
    material_blocks = "\n\n".join(
        _preview_material_block(name, MATERIALS[name]) for name in MATERIALS
    )
    return f'''#usda 1.0
(
    defaultPrim = "{MATERIAL_ROOT}"
    doc = "DrAnmar visual-only research materials; procedural and not clinically color-calibrated."
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
        string drAnmarPrimaryMaterialContext = "OpenPBR_1_1_MaterialX"
        string drAnmarTextureColorContract = "basecolor_sRGB_roughness_and_normal_raw"
    }}
)
{{
{material_blocks}
}}
'''


def _box(
    name: str,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: str,
) -> str:
    scale = tuple(value / 2.0 for value in size)
    return f'''        def Cube "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double size = 2
            token purpose = "render"
            rel material:binding = <{_support_material_path(material)}>
            double3 xformOp:translate = {_usd_vec(center)}
            double3 xformOp:scale = {_usd_vec(scale)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}'''


def _support_material_path(title: str) -> str:
    return f"/{SUPPORT_ROOT}/Materials/{title}"


def author_support() -> str:
    boxes = [
        _box(
            "RailLeft",
            center=(-0.0425, 0.0, -0.0055),
            size=(0.0050, 0.058, 0.0050),
            material="Cassette",
        ),
        _box(
            "RailRight",
            center=(0.0425, 0.0, -0.0055),
            size=(0.0050, 0.058, 0.0050),
            material="Cassette",
        ),
        _box(
            "RailAnterior",
            center=(0.0, -0.0265, -0.0055),
            size=(0.080, 0.0050, 0.0050),
            material="Cassette",
        ),
        _box(
            "RailPosterior",
            center=(0.0, 0.0265, -0.0055),
            size=(0.080, 0.0050, 0.0050),
            material="Cassette",
        ),
        _box(
            "ColumnLeftAnterior",
            center=(-0.0425, -0.0265, -0.0120),
            size=(0.0050, 0.0050, 0.0130),
            material="Cassette",
        ),
        _box(
            "ColumnLeftPosterior",
            center=(-0.0425, 0.0265, -0.0120),
            size=(0.0050, 0.0050, 0.0130),
            material="Cassette",
        ),
        _box(
            "ColumnRightAnterior",
            center=(0.0425, -0.0265, -0.0120),
            size=(0.0050, 0.0050, 0.0130),
            material="Cassette",
        ),
        _box(
            "ColumnRightPosterior",
            center=(0.0425, 0.0265, -0.0120),
            size=(0.0050, 0.0050, 0.0130),
            material="Cassette",
        ),
        _box(
            "SurgicalDrape",
            center=(0.0, 0.0, -0.0197),
            size=(0.120, 0.090, 0.0006),
            material="Drape",
        ),
    ]
    return f'''#usda 1.0
(
    defaultPrim = "{SUPPORT_ROOT}"
    doc = "Render-only support cassette and drape spanning the 17 mm tissue-to-table offset."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{SUPPORT_ROOT}" (
    customData = {{
        string drAnmarAssetRole = "render_support_only"
        bool drAnmarClinicalValidation = false
        bool drAnmarCollisionAuthority = false
        bool drAnmarPhysicsAuthority = false
        double drAnmarSupportedGapM = 0.017
        string drAnmarSupportExplanation = "tissue_bottom_minus_0p003_to_table_plane_minus_0p020"
    }}
)
{{
    token purpose = "render"

    def Scope "Materials" (
        references = @./materials.usda@</{MATERIAL_ROOT}>
    )
    {{
    }}

    def Xform "CassetteGeometry"
    {{
{chr(10).join(boxes)}
    }}
}}
'''


def _face_varying_uvs(mesh) -> tuple[tuple[float, float], ...]:
    minimum = np.asarray(mesh.extent_min, dtype=np.float64)
    maximum = np.asarray(mesh.extent_max, dtype=np.float64)
    span = np.maximum(maximum - minimum, 1.0e-12)
    result: list[tuple[float, float]] = []
    for triangle in mesh.surface_triangles:
        a, b, c = (np.asarray(mesh.points[index], dtype=np.float64) for index in triangle)
        normal = np.cross(b - a, c - a)
        axis = int(np.argmax(np.abs(normal)))
        for index in triangle:
            point = np.asarray(mesh.points[index], dtype=np.float64)
            if axis == 2:
                u = (point[0] - minimum[0]) / span[0]
                v = (point[1] - minimum[1]) / span[1]
                tile = 3.0
            elif axis == 0:
                u = (point[1] - minimum[1]) / span[1]
                v = (point[2] - minimum[2]) / span[2]
                tile = 2.0
            else:
                u = (point[0] - minimum[0]) / span[0]
                v = (point[2] - minimum[2]) / span[2]
                tile = 2.0
            result.append((float(u * tile), float(v * tile)))
    return tuple(result)


def _format_uvs(values: tuple[tuple[float, float], ...]) -> str:
    lines: list[str] = []
    chunk_size = 6
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size]
        lines.append("            " + ", ".join(_usd_vec(value) for value in chunk))
    return ",\n".join(lines)


def author_overlay(lod: str, mesh) -> str:
    uvs = _face_varying_uvs(mesh)
    subset_bindings = "\n".join(
        f'''            over "{title}Faces"
            {{
                rel material:binding = </{ROOT_PRIM}/VisualMaterials/{title}>
            }}'''
        for title in ("Surface", "Bulk", "Fascia", "Wound")
    )
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Visual-only {lod} overlay for DrAnmar needle-ready tissue; base physics geometry remains authoritative."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{ROOT_PRIM}" (
    references = @./needle_ready_tissue_{lod}.usda@</{ROOT_PRIM}>
    customData = {{
        string drAnmarAssetRole = "visual_overlay"
        bool drAnmarClinicalValidation = false
        string drAnmarGeometryLod = "{lod}"
        bool drAnmarPhysicsAuthority = false
        string drAnmarVisualMeshBinding = "one_to_one_existing_deforming_visual_points"
    }}
)
{{
    def Scope "VisualMaterials" (
        references = @./visual/materials.usda@</{MATERIAL_ROOT}>
    )
    {{
    }}

    def Xform "RenderSupport" (
        references = @./visual/support_cassette.usda@</{SUPPORT_ROOT}>
    )
    {{
    }}

    over "Visual"
    {{
        texCoord2f[] primvars:st = [
{_format_uvs(uvs)}
        ] (
            interpolation = "faceVarying"
        )
        custom string drAnmar:visualOverlay = "uv_and_material_opinions_only"

{subset_bindings}
    }}
}}
'''


def author_visual_variant_root() -> str:
    variants = []
    for lod in LODS:
        variants.append(
            f'''        "{lod}" {{
            over "Geometry" (
                prepend references = @./needle_ready_tissue_{lod}_visual.usda@</{ROOT_PRIM}>
            )
            {{
            }}
        }}'''
        )
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Visual-only LOD selector for the DrAnmar needle-ready tissue research asset."
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{ROOT_PRIM}" (
    variants = {{
        string geometryLod = "validation"
    }}
    prepend variantSets = "geometryLod"
    customData = {{
        string drAnmarAssetId = "dranmar-needle-ready-tissue-visual-v1"
        string drAnmarAssetVersion = "1.0.0"
        bool drAnmarClinicalValidation = false
        bool drAnmarPhysicsAuthority = false
    }}
)
{{
    def Xform "Geometry"
    {{
    }}
    variantSet "geometryLod" = {{
{chr(10).join(variants)}
    }}
}}
'''


def _visual_member_paths(asset_root: Path) -> list[Path]:
    paths = [asset_root / f"needle_ready_tissue_{lod}_visual.usda" for lod in LODS]
    paths.append(asset_root / "needle_ready_tissue_visual_unit.usda")
    paths.extend(path for path in sorted((asset_root / "visual").rglob("*")) if path.is_file())
    return sorted(paths)


def validate_nvidia_vendor(asset_root: Path) -> None:
    vendor_root = asset_root / NVIDIA_VENDOR_SUBPATH
    failures = {}
    for name, expected_hash in NVIDIA_VENDOR_HASHES.items():
        path = vendor_root / name
        actual_hash = sha256(path) if path.is_file() else None
        if actual_hash != expected_hash:
            failures[name] = {
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
            }
    if failures:
        raise RuntimeError(f"NVIDIA vendor input integrity failed: {failures}")


def write_visual_manifest(
    asset_root: Path,
    *,
    base_hashes: dict[str, str],
    texture_size: int,
) -> dict[str, Any]:
    members: dict[str, Any] = {}
    for path in _visual_member_paths(asset_root):
        relative = path.relative_to(asset_root).as_posix()
        is_nvidia_member = (
            path.parent == asset_root / NVIDIA_VENDOR_SUBPATH
            and path.name in NVIDIA_VENDOR_HASHES
        )
        members[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "license": "MIT-0" if is_nvidia_member else "Apache-2.0",
            "provenance": (
                "vendored_unchanged_from_NVIDIA_PhysicalAI_SimReady_Materials_v0.2.0"
                if is_nvidia_member
                else "deterministically_generated_or_authored_by_DrAnmar"
            ),
        }
    geometry_contract = json.loads(
        (asset_root / "geometry_contract.json").read_text(encoding="utf-8")
    )
    return {
        "schema": "dr.anmar.visual-asset-manifest.v1",
        "asset_id": "dranmar-needle-ready-tissue-visual-v1",
        "asset_version": "1.0.0",
        "primary_usd": "needle_ready_tissue_visual_unit.usda",
        "default_lod": "validation",
        "visual_only": True,
        "physics_authority": False,
        "clinical_validation": False,
        "texture_resolution_px": texture_size,
        "texture_color_contract": {
            "basecolor": "sRGB",
            "roughness": "raw",
            "normal": (
                "raw_tangent_space_OpenGL_JPEG_quality_99_4:4:4"
            ),
            "subsurface_weight": "raw",
        },
        "texture_encoding": {
            "basecolor_roughness_subsurface": "lossless_PNG",
            "normal": {
                "format": "JPEG",
                "subsampling": "4:4:4",
                "quality": NORMAL_JPEG_QUALITY,
            },
        },
        "materials": {
            "portable_context": "UsdPreviewSurface",
            "primary_context": "OpenPBR_1.1_MaterialX",
            "nvidia_source": "PhysicalAI_SimReady_Materials_v0.2.0",
            "thin_walled": False,
            "coat_weight": 0.0,
            "specular_policy": "restrained_and_rough_no_uniform_wet_gloss",
            "subsurface_policy": "texture_modulated_volumetric_research_seed_not_clinically_calibrated",
        },
        "support": {
            "asset": "visual/support_cassette.usda",
            "purpose": "render",
            "collision_authority": False,
            "physics_authority": False,
            "explained_tissue_to_table_gap_m": 0.017,
        },
        "base_asset": {
            "asset_id": geometry_contract["id"],
            "asset_version": geometry_contract["version"],
        },
        "base_physics_sha256": base_hashes,
        "generator": {
            "path": "tools/generate_needle_ready_tissue_visuals.py",
            "sha256": sha256(Path(__file__)),
            "seeded": True,
            "dependency_versions": GENERATOR_VERSIONS,
        },
        "provenance": {
            "textures": "DrAnmar_seeded_procedural_maps_plus_bounded_NVIDIA_skin_micro_normal_input_no_patient_imagery",
            "geometry": "visual_overlays_reference_independently_authored_DrAnmar_base_geometry",
            "nvidia": {
                "repository": "https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials",
                "release": "v0.2.0",
                "release_archive": "Skin.zip",
                "release_archive_sha256": "e6831eb8129179d9d05be80a582fe0f5824a7c2d9f5252db0f7f4a3102082380",
                "license": "MIT-0",
                "member_sha256": NVIDIA_VENDOR_HASHES,
                "use": "unchanged_OpenPBR_base_and_bounded_surface_micro_normal_input",
            },
        },
        "license": "mixed",
        "licenses": ["Apache-2.0", "MIT-0"],
        "license_sources": {
            "DrAnmar": "visual/LICENSE.txt",
            "NVIDIA_PhysicalAI_SimReady_Materials": (
                "visual/vendor/"
                "nvidia_physicalai_simready_materials_v0_2_0/LICENSE.md"
            ),
        },
        "members": members,
    }


def write_visual_readme(path: Path) -> None:
    path.write_text(
        """# DrAnmar Needle-Ready Tissue Visual Package

This directory contains a visual-only research presentation layer for the
canonical needle-ready tissue asset. It adds deterministic 2048 px PBR maps,
an NVIDIA-derived OpenPBR 1.1 MaterialX graph, portable `UsdPreviewSurface`
fallbacks, and a render-purpose support cassette/drape. It does not replace
or modify the training, contact, or validation TetMesh assets.

The overlays reuse the existing `Visual` mesh and author face-varying UVs on
that mesh. This preserves the one-to-one particle/visual point order required
by the Isaac Lab Newton/Fabric visual sync. No detached high-resolution mesh
is introduced.

The support cassette spans from the tissue lower face near -3 mm to the draped
table plane near -20 mm, making the 17 mm scene offset visually intelligible.
It is explicitly render-only: it has no collision API, rigid-body API, mass,
contact material, or solver authority.

Regenerate from the asset-extension root:

```bash
uv run --no-project --with numpy==2.2.6 --with Pillow==11.3.0 \\
  python tools/generate_needle_ready_tissue_visuals.py
```

The generator is deterministic. `visual_manifest.json` recursively hashes the
generated package and records the immutable base-physics hashes. Base color
maps use sRGB; roughness and subsurface-weight maps use raw PNG data. Normal
maps use raw tangent-space quality-99 JPEG with 4:4:4 sampling, matching the
portable texture practice in NVIDIA's source library without reducing the 2K
resolution. Moisture is sparse and bounded rather than a uniform glossy
coating.

The unchanged OpenPBR base class and a medium-skin micro-normal input come
from NVIDIA `PhysicalAI-SimReady-Materials` release `v0.2.0` under MIT-0.
Their exact archive/member hashes and license are retained under `vendor/`.
The published normal contributes bounded surface microstructure only; it does
not supply geometry, biomechanics, patient imagery, or clinical calibration.

These colors, scattering distances, geometry, and textures are research
seeds. They are not patient-specific, clinically color-calibrated, physically
calibrated, or approved for patient care. Native RTX material compilation,
lighting/camera calibration, and screenshot qualification remain separate
runtime gates.
""",
        encoding="utf-8",
    )


def write_visual_license(path: Path) -> None:
    path.write_text(
        """Copyright 2026 Dr.Anmar Project Developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
""",
        encoding="utf-8",
    )


def write_visual_notice(path: Path) -> None:
    path.write_text(
        """DrAnmar Needle-Ready Tissue Visual Package 1.0.0

All texture maps, UV opinions, support geometry, material graphs, and
authoring code outside the explicitly vendored NVIDIA inputs were independently
and procedurally authored by the Dr.Anmar project. No patient imagery or
scanned anatomy is included.

The unchanged OpenPBR 1.1 MaterialX base and medium-skin normal input are
vendored from NVIDIA PhysicalAI-SimReady-Materials `v0.2.0` under MIT-0. Their
license, provenance, source-archive hash, and member hashes are retained in
the package. The normal is a bounded microstructure input to a derived
DrAnmar normal map; it has no physics or task authority.

This is a research simulation asset. It is not clinically validated and is
not for patient care or clinical decision-making.
""",
        encoding="utf-8",
    )


def write_nvidia_vendor_provenance(path: Path) -> None:
    path.write_text(
        f"""# NVIDIA PhysicalAI SimReady Material Inputs

This directory vendors three immutable inputs from NVIDIA's
`PhysicalAI-SimReady-Materials` release `v0.2.0`:

- `open_pbr_uber_base_class.usda`
- `Skin_Medium_normal.jpg`
- `LICENSE.md`

Upstream:
`https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials`

Release archive:
`https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials/releases/download/v0.2.0/Skin.zip`

Release archive SHA-256:
`e6831eb8129179d9d05be80a582fe0f5824a7c2d9f5252db0f7f4a3102082380`

Vendored member SHA-256 values:

- `open_pbr_uber_base_class.usda`:
  `{NVIDIA_VENDOR_HASHES["open_pbr_uber_base_class.usda"]}`
- `Skin_Medium_normal.jpg`:
  `{NVIDIA_VENDOR_HASHES["Skin_Medium_normal.jpg"]}`
- `LICENSE.md`:
  `{NVIDIA_VENDOR_HASHES["LICENSE.md"]}`

The upstream package is MIT-0. The OpenPBR 1.1 MaterialX base class is used
unchanged. The published medium-skin tangent-space normal is used only as a
microstructure input to the deterministic DrAnmar surface-normal generator;
it does not supply geometry, biomechanics, patient data, clinical
calibration, or task semantics. DrAnmar's output textures remain research
art and must pass native RTX review before promotion.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--texture-size", type=int, default=TEXTURE_SIZE)
    args = parser.parse_args()
    if args.texture_size < 1024:
        parser.error("--texture-size must be at least 1024")

    asset_root = args.asset_root.resolve()
    visual_root = asset_root / "visual"
    texture_root = visual_root / "textures"
    visual_root.mkdir(parents=True, exist_ok=True)
    texture_root.mkdir(parents=True, exist_ok=True)
    validate_nvidia_vendor(asset_root)
    write_nvidia_vendor_provenance(
        asset_root / NVIDIA_VENDOR_SUBPATH / "PROVENANCE.md"
    )

    base_hashes = {name: sha256(asset_root / name) for name in BASE_PHYSICS_FILES}
    geometry_generator = _load_base_generator()
    contract = geometry_generator.load_json(asset_root / "geometry_contract.json")
    meshes = {lod: geometry_generator.build_mesh(contract, lod) for lod in LODS}

    for name, spec in MATERIALS.items():
        legacy_normal = texture_root / f"{name}_normal.png"
        if legacy_normal.is_file():
            legacy_normal.unlink()
        generate_material_textures(
            texture_root,
            name,
            spec,
            size=args.texture_size,
            published_skin_normal=(
                asset_root / NVIDIA_SKIN_NORMAL if name == "surface" else None
            ),
        )
    (visual_root / "materials.usda").write_text(
        author_materials(),
        encoding="utf-8",
    )
    (visual_root / "support_cassette.usda").write_text(
        author_support(),
        encoding="utf-8",
    )
    write_visual_readme(visual_root / "README.md")
    write_visual_license(visual_root / "LICENSE.txt")
    write_visual_notice(visual_root / "NOTICE.txt")

    for lod, mesh in meshes.items():
        (asset_root / f"needle_ready_tissue_{lod}_visual.usda").write_text(
            author_overlay(lod, mesh),
            encoding="utf-8",
        )
    (asset_root / "needle_ready_tissue_visual_unit.usda").write_text(
        author_visual_variant_root(),
        encoding="utf-8",
    )
    manifest = write_visual_manifest(
        asset_root,
        base_hashes=base_hashes,
        texture_size=args.texture_size,
    )
    (asset_root / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    final_hashes = {name: sha256(asset_root / name) for name in BASE_PHYSICS_FILES}
    if final_hashes != base_hashes:
        raise RuntimeError("base physics USD changed while generating visual assets")
    print(
        json.dumps(
            {
                "asset_root": str(asset_root),
                "generated_members": len(manifest["members"]),
                "base_physics_unchanged": True,
                "texture_size": args.texture_size,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
