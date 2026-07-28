#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Generate the inactive DrAnmar 22 mm half-circle taper-point needle candidate.

The package is category-level research geometry, not a manufacturer digital
twin. It remains unreferenced by task and runtime registries. The entry layer
defaults to ``Physics=none`` and exposes explicit ``physics`` and ``physx``
payload variants for qualification.

Run from the asset-extension root:

    uv run --no-project \
      --with numpy==2.2.6 --with Pillow==11.3.0 --with usd-core==25.11 \
      python tools/generate_needle_22_half_circle_taper_candidate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pxr import Sdf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_SUBPATH = Path("data/Props/SurgicalClosure/Needle22HalfCircleTaperCandidate")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / ASSET_SUBPATH
CANONICAL_APACHE_LICENSE = REPOSITORY_ROOT / "data/Props/SurgicalClosure/Needle/LICENSE.txt"
SELECTED_DIMENSIONAL_INPUT = (
    REPOSITORY_ROOT / "data/Props/SurgicalClosure/Needle/physics_profile.json"
)
SELECTED_DIMENSIONAL_INPUT_SHA256 = (
    "746b3199e63cc42fefe42b94e883fb329f88c5f50c82844aa77c8573c39e9b6e"
)
COMPARISON_ONLY_PHYSICS_NEXT_SHA256 = (
    "309981bb46ab14c86a6964d58dd623d9756c972f7ca83f7f0d1771af66d60d85"
)
NVIDIA_SOURCE_VENDOR_ROOT = (
    REPOSITORY_ROOT
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit/visual/vendor"
    / "nvidia_physicalai_simready_materials_v0_2_0"
)
NVIDIA_VENDOR_SUBPATH = Path("vendor/nvidia_physicalai_simready_materials_v0_2_0")
NVIDIA_VENDOR_INPUTS = {
    "LICENSE.md": "18f74283f08ff1ed39a9c46dbe2622146d45f771023c3dbd9c631bb058e1421b",
    "open_pbr_uber_base_class.usda": (
        "bb76ff9fa9cd74b86b6be4ed3c6ed79cdca15eff6d603ca571bdf9ce21e10c5f"
    ),
}

ROOT_PRIM = "DrAnmarNeedle22HalfCircleTaperCandidate"
ASSET_ID = "dranmar-needle-22mm-half-circle-taper-category-candidate-v1"
ASSET_VERSION = "1.0.0"
PRIMARY_USD = "dranmar_needle_22mm_half_circle_taper_candidate.usda"
BASE_USD = "dranmar_needle_22mm_half_circle_taper_candidate_base.usda"
GEOMETRY_USD = "dranmar_needle_22mm_half_circle_taper_candidate_geometry.usdc"
MATERIALS_USD = "dranmar_needle_22mm_half_circle_taper_candidate_materials.usda"
PHYSICS_USD = "dranmar_needle_22mm_half_circle_taper_candidate_physics.usda"
PHYSX_USD = "dranmar_needle_22mm_half_circle_taper_candidate_physx.usda"

ARC_LENGTH_M = 0.022
CENTERLINE_RADIUS_M = ARC_LENGTH_M / math.pi
BODY_DIAMETER_M = 0.00053
BODY_RADIUS_M = 0.5 * BODY_DIAMETER_M
TIP_CURVATURE_RADIUS_M = 0.000018
TIP_TAPER_END_M = 0.0038
SWAGE_START_M = 0.0192
SWAGE_FULL_M = 0.0206
SWAGE_OUTER_RADIUS_M = 0.00018
SWAGE_RECESS_RADIUS_M = 0.000105
SWAGE_RECESS_DEPTH_M = 0.00082
GRASP_LAND_START_M = 0.0122
GRASP_LAND_END_M = 0.0164
GRASP_BLEND_M = 0.00035
PREFERRED_GRASP_FRACTION = 0.65
GRASP_FLAT_HALF_THICKNESS_RATIO = 0.78
GRASP_HALF_WIDTH_RATIO = 1.025
GRASP_RIB_HEIGHT_M = 0.000011
STEEL_DENSITY_KG_M3 = 8000.0
STATIC_FRICTION = 0.28
DYNAMIC_FRICTION = 0.20
RESTITUTION = 0.01
CONTACT_OFFSET_MIN_M = 0.000005
CONTACT_OFFSET_MAX_M = 0.000015
ARC_RESOLUTION = 512
RADIAL_RESOLUTION = 64
COLLISION_SEGMENT_COUNT = 48
GRASP_COLLISION_SEGMENT_COUNT = 16
TIP_COLLISION_END_M = 0.00105
TIP_CAPSULE_START_M = 0.00075
TEXTURE_SIZE = 2048
TEXTURE_REPEAT_M = 0.0015
PLANE_NORMAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)

MANUFACTURER_CATEGORY_SOURCE = {
    "provider": "Johnson & Johnson MedTech / Ethicon",
    "document": "157677-230117-AH Suture Catalog",
    "url": (
        "https://www.jnjmedtech.com/system/files/pdf/"
        "157677-230117-AH-Suture-Catalog-157677-230117.pdf"
    ),
    "accessed_utc_date": "2026-07-28",
    "category_facts_used": [
        "22_mm_is_published_as_needle_length",
        "one_half_circle_curvature_category",
        "taper_point_category",
        "longitudinal_ribbing_is_described_as_resisting_turning_rocking_and_twisting",
        "swaging_is_described_as_a_smooth_factory_attachment",
    ],
    "example_catalog_designations_not_modeled_as_twins": ["SH-1", "CT-3"],
    "not_sourced_from_catalog": [
        "body_diameter",
        "tip_taper_length",
        "swage_dimensions",
        "rib_dimensions",
        "mass_properties",
        "friction",
        "collision",
        "textures",
    ],
}

GENERATOR_DEPENDENCIES = {
    "numpy": "2.2.6",
    "Pillow": "11.3.0",
    "usd-core": "25.11",
}

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


@dataclass
class MeshData:
    points: list[Vec3]
    triangles: list[Triangle]
    face_varying_uvs: list[Vec2]
    face_varying_uvs_m: list[Vec2]
    normals: list[Vec3]
    ring_distances_m: list[float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smootherstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _ramp(value: float, start: float, end: float) -> float:
    if end <= start:
        raise ValueError("ramp end must exceed start")
    return _smootherstep((value - start) / (end - start))


def centerline(distance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point, tangent toward the swage, and in-plane section normal."""

    distance_m = min(ARC_LENGTH_M, max(0.0, distance_m))
    theta = -math.pi / 2.0 + distance_m / CENTERLINE_RADIUS_M
    point = np.asarray(
        (
            CENTERLINE_RADIUS_M * math.cos(theta),
            CENTERLINE_RADIUS_M * math.sin(theta),
            0.0,
        ),
        dtype=np.float64,
    )
    tangent = np.asarray((-math.sin(theta), math.cos(theta), 0.0), dtype=np.float64)
    section_normal = np.asarray((math.cos(theta), math.sin(theta), 0.0), dtype=np.float64)
    return point, tangent, section_normal


def round_radius(distance_m: float) -> float:
    distance_m = min(ARC_LENGTH_M, max(0.0, distance_m))
    if distance_m <= TIP_CURVATURE_RADIUS_M:
        squared = (
            2.0 * TIP_CURVATURE_RADIUS_M * distance_m - distance_m * distance_m
        )
        return math.sqrt(max(0.0, squared))
    if distance_m < TIP_TAPER_END_M:
        fraction = (distance_m - TIP_CURVATURE_RADIUS_M) / (
            TIP_TAPER_END_M - TIP_CURVATURE_RADIUS_M
        )
        return TIP_CURVATURE_RADIUS_M + (
            BODY_RADIUS_M - TIP_CURVATURE_RADIUS_M
        ) * _smootherstep(fraction)
    if distance_m < SWAGE_START_M:
        return BODY_RADIUS_M
    if distance_m < SWAGE_FULL_M:
        fraction = (distance_m - SWAGE_START_M) / (SWAGE_FULL_M - SWAGE_START_M)
        return BODY_RADIUS_M + (
            SWAGE_OUTER_RADIUS_M - BODY_RADIUS_M
        ) * _smootherstep(fraction)
    return SWAGE_OUTER_RADIUS_M


def grasp_land_weight(distance_m: float) -> float:
    rise = _ramp(
        distance_m,
        GRASP_LAND_START_M,
        GRASP_LAND_START_M + GRASP_BLEND_M,
    )
    fall = 1.0 - _ramp(
        distance_m,
        GRASP_LAND_END_M - GRASP_BLEND_M,
        GRASP_LAND_END_M,
    )
    return rise * fall


def section_offset(
    distance_m: float,
    angle: float,
    *,
    include_ribs: bool = True,
) -> np.ndarray:
    """Return the authored cross-section offset in needle-local coordinates."""

    _, _, normal = centerline(distance_m)
    radius = round_radius(distance_m)
    land = grasp_land_weight(distance_m)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    round_n = radius * cosine
    round_b = radius * sine

    exponent = 4.0
    flat_n = (
        radius
        * GRASP_HALF_WIDTH_RATIO
        * math.copysign(abs(cosine) ** (2.0 / exponent), cosine)
    )
    flat_b = (
        radius
        * GRASP_FLAT_HALF_THICKNESS_RATIO
        * math.copysign(abs(sine) ** (2.0 / exponent), sine)
    )
    section_n = (1.0 - land) * round_n + land * flat_n
    section_b = (1.0 - land) * round_b + land * flat_b

    if include_ribs and land > 0.0 and abs(sine) > 1.0e-12:
        half_width = max(radius * GRASP_HALF_WIDTH_RATIO, 1.0e-12)
        normalized_width = section_n / half_width
        rib_profile = sum(
            math.exp(-((normalized_width - center) / 0.085) ** 2)
            for center in (-0.55, 0.0, 0.55)
        )
        face_gate = abs(sine) ** 6
        section_b += (
            math.copysign(1.0, sine)
            * land
            * GRASP_RIB_HEIGHT_M
            * rib_profile
            * face_gate
        )
    return section_n * normal + section_b * PLANE_NORMAL


def maximum_section_radius(distance_m: float) -> float:
    return max(
        float(np.linalg.norm(section_offset(distance_m, angle)))
        for angle in np.linspace(0.0, 2.0 * math.pi, 257, endpoint=True)
    )


def _ring_distances() -> list[float]:
    special = [
        TIP_CURVATURE_RADIUS_M * fraction
        for fraction in (0.02, 0.08, 0.18, 0.35, 0.55, 0.75, 1.0)
    ]
    special.extend(
        (
            TIP_TAPER_END_M,
            GRASP_LAND_START_M,
            GRASP_LAND_START_M + GRASP_BLEND_M,
            PREFERRED_GRASP_FRACTION * ARC_LENGTH_M,
            GRASP_LAND_END_M - GRASP_BLEND_M,
            GRASP_LAND_END_M,
            SWAGE_START_M,
            SWAGE_FULL_M,
            ARC_LENGTH_M,
        )
    )
    uniform = np.linspace(
        2.0 * TIP_CURVATURE_RADIUS_M,
        ARC_LENGTH_M,
        ARC_RESOLUTION + 1,
        dtype=np.float64,
    )
    return sorted(
        {
            round(float(value), 15)
            for value in (*special, *uniform)
            if 0.0 < value <= ARC_LENGTH_M
        }
    )


def _triangle_uvs(
    vertex_metadata: list[tuple[float, float]],
    triangle: Triangle,
) -> tuple[list[Vec2], list[Vec2]]:
    metadata = [vertex_metadata[index] for index in triangle]
    angles = [item[1] for item in metadata]
    if max(angles) - min(angles) > math.pi:
        angles = [angle + (2.0 * math.pi if angle < math.pi else 0.0) for angle in angles]
    circumference = 2.0 * math.pi * BODY_RADIUS_M
    meter_values = [
        (metadata[index][0], angles[index] * circumference / (2.0 * math.pi))
        for index in range(3)
    ]
    repeated_values = [
        (value[0] / TEXTURE_REPEAT_M, value[1] / TEXTURE_REPEAT_M)
        for value in meter_values
    ]
    return repeated_values, meter_values


def _orient_triangles(
    points: list[Vec3],
    triangles: list[Triangle],
    uvs: list[list[Vec2]],
    uvs_m: list[list[Vec2]],
) -> tuple[list[Triangle], list[list[Vec2]], list[list[Vec2]]]:
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = tuple(sorted((left, right)))
            direction = 1 if (left, right) == key else -1
            edge_uses[key].append((triangle_index, direction))
    invalid = [key for key, uses in edge_uses.items() if len(uses) != 2]
    if invalid:
        raise ValueError(f"mesh is not closed before orientation: {len(invalid)} edges")

    parity: dict[int, int] = {}
    for seed in range(len(triangles)):
        if seed in parity:
            continue
        parity[seed] = 0
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            triangle = triangles[current]
            for left, right in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                key = tuple(sorted((left, right)))
                current_direction = 1 if (left, right) == key else -1
                uses = edge_uses[key]
                neighbor, neighbor_direction = uses[0] if uses[1][0] == current else uses[1]
                required = parity[current] ^ int(current_direction == neighbor_direction)
                if neighbor in parity and parity[neighbor] != required:
                    raise ValueError("mesh winding constraints are inconsistent")
                if neighbor not in parity:
                    parity[neighbor] = required
                    queue.append(neighbor)

    oriented = list(triangles)
    oriented_uvs = list(uvs)
    oriented_uvs_m = list(uvs_m)
    for index, flip in parity.items():
        if flip:
            a, b, c = oriented[index]
            oriented[index] = (a, c, b)
            oriented_uvs[index] = [
                oriented_uvs[index][0],
                oriented_uvs[index][2],
                oriented_uvs[index][1],
            ]
            oriented_uvs_m[index] = [
                oriented_uvs_m[index][0],
                oriented_uvs_m[index][2],
                oriented_uvs_m[index][1],
            ]
    if signed_mesh_volume(points, oriented) < 0.0:
        oriented = [(a, c, b) for a, b, c in oriented]
        oriented_uvs = [[values[0], values[2], values[1]] for values in oriented_uvs]
        oriented_uvs_m = [
            [values[0], values[2], values[1]] for values in oriented_uvs_m
        ]
    return oriented, oriented_uvs, oriented_uvs_m


def _vertex_normals(points: list[Vec3], triangles: list[Triangle]) -> list[Vec3]:
    coordinates = np.asarray(points, dtype=np.float64)
    accumulated = np.zeros_like(coordinates)
    for a, b, c in triangles:
        normal = np.cross(coordinates[b] - coordinates[a], coordinates[c] - coordinates[a])
        accumulated[a] += normal
        accumulated[b] += normal
        accumulated[c] += normal
    lengths = np.linalg.norm(accumulated, axis=1)
    if np.any(lengths <= 1.0e-20):
        raise ValueError("render mesh contains a vertex without a valid normal")
    accumulated /= lengths[:, None]
    return [tuple(map(float, value)) for value in accumulated]


def signed_mesh_volume(points: list[Vec3], triangles: list[Triangle]) -> float:
    coordinates = np.asarray(points, dtype=np.float64)
    volume = 0.0
    for a, b, c in triangles:
        volume += float(
            np.dot(coordinates[a], np.cross(coordinates[b], coordinates[c]))
        ) / 6.0
    return volume


def build_render_mesh() -> MeshData:
    distances = _ring_distances()
    tip, _, _ = centerline(0.0)
    points: list[Vec3] = [tuple(map(float, tip))]
    metadata: list[tuple[float, float]] = [(0.0, 0.0)]
    outer_rings: list[int] = []

    for distance in distances:
        center, _, _ = centerline(distance)
        outer_rings.append(len(points))
        for segment in range(RADIAL_RESOLUTION):
            angle = 2.0 * math.pi * segment / RADIAL_RESOLUTION
            point = center + section_offset(distance, angle)
            points.append(tuple(map(float, point)))
            metadata.append((distance, angle))

    triangles: list[Triangle] = []
    first_ring = outer_rings[0]
    for segment in range(RADIAL_RESOLUTION):
        triangles.append(
            (
                0,
                first_ring + (segment + 1) % RADIAL_RESOLUTION,
                first_ring + segment,
            )
        )
    for current_start, next_start in pairwise(outer_rings):
        for segment in range(RADIAL_RESOLUTION):
            a = current_start + segment
            b = current_start + (segment + 1) % RADIAL_RESOLUTION
            c = next_start + (segment + 1) % RADIAL_RESOLUTION
            d = next_start + segment
            triangles.extend(((a, b, c), (a, c, d)))

    inner_rings: list[int] = []
    cavity_distances = np.linspace(
        ARC_LENGTH_M,
        ARC_LENGTH_M - SWAGE_RECESS_DEPTH_M,
        17,
        dtype=np.float64,
    )
    for cavity_index, distance_value in enumerate(cavity_distances):
        distance = float(distance_value)
        depth_fraction = cavity_index / (len(cavity_distances) - 1)
        radius = SWAGE_RECESS_RADIUS_M * (1.0 - 0.14 * _smootherstep(depth_fraction))
        center, _, normal = centerline(distance)
        inner_rings.append(len(points))
        for segment in range(RADIAL_RESOLUTION):
            angle = 2.0 * math.pi * segment / RADIAL_RESOLUTION
            point = center + radius * (
                math.cos(angle) * normal + math.sin(angle) * PLANE_NORMAL
            )
            points.append(tuple(map(float, point)))
            metadata.append((distance, angle))

    outer_tail = outer_rings[-1]
    inner_mouth = inner_rings[0]
    for segment in range(RADIAL_RESOLUTION):
        a = outer_tail + segment
        b = outer_tail + (segment + 1) % RADIAL_RESOLUTION
        c = inner_mouth + (segment + 1) % RADIAL_RESOLUTION
        d = inner_mouth + segment
        triangles.extend(((a, b, c), (a, c, d)))
    for current_start, next_start in pairwise(inner_rings):
        for segment in range(RADIAL_RESOLUTION):
            a = current_start + segment
            b = current_start + (segment + 1) % RADIAL_RESOLUTION
            c = next_start + (segment + 1) % RADIAL_RESOLUTION
            d = next_start + segment
            triangles.extend(((a, c, b), (a, d, c)))

    recess_bottom_distance = ARC_LENGTH_M - SWAGE_RECESS_DEPTH_M
    recess_bottom, _, _ = centerline(recess_bottom_distance)
    bottom_index = len(points)
    points.append(tuple(map(float, recess_bottom)))
    metadata.append((recess_bottom_distance, 0.0))
    last_inner = inner_rings[-1]
    for segment in range(RADIAL_RESOLUTION):
        triangles.append(
            (
                bottom_index,
                last_inner + segment,
                last_inner + (segment + 1) % RADIAL_RESOLUTION,
            )
        )

    triangle_uvs: list[list[Vec2]] = []
    triangle_uvs_m: list[list[Vec2]] = []
    for triangle in triangles:
        repeated, metric = _triangle_uvs(metadata, triangle)
        triangle_uvs.append(repeated)
        triangle_uvs_m.append(metric)
    triangles, triangle_uvs, triangle_uvs_m = _orient_triangles(
        points,
        triangles,
        triangle_uvs,
        triangle_uvs_m,
    )
    normals = _vertex_normals(points, triangles)
    return MeshData(
        points=points,
        triangles=triangles,
        face_varying_uvs=[value for values in triangle_uvs for value in values],
        face_varying_uvs_m=[
            value for values in triangle_uvs_m for value in values
        ],
        normals=normals,
        ring_distances_m=distances,
    )


def topology_report(mesh: MeshData) -> dict[str, Any]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(mesh.points))}
    minimum_area = float("inf")
    points = np.asarray(mesh.points, dtype=np.float64)
    for triangle in mesh.triangles:
        a, b, c = triangle
        area = 0.5 * float(
            np.linalg.norm(np.cross(points[b] - points[a], points[c] - points[a]))
        )
        minimum_area = min(minimum_area, area)
        for left, right in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((left, right)))
            edge_counts[edge] += 1
            adjacency[left].add(right)
            adjacency[right].add(left)
    unseen = set(adjacency)
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return {
        "vertex_count": len(mesh.points),
        "triangle_count": len(mesh.triangles),
        "edge_count": len(edge_counts),
        "boundary_edge_count": sum(count == 1 for count in edge_counts.values()),
        "nonmanifold_edge_count": sum(count != 2 for count in edge_counts.values()),
        "connected_component_count": components,
        "minimum_triangle_area_m2": minimum_area,
        "signed_volume_m3": signed_mesh_volume(mesh.points, mesh.triangles),
        "watertight": all(count == 2 for count in edge_counts.values()),
    }


def rotation_matrix_to_usd_quaternion_wxyz(
    matrix: np.ndarray,
) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            quaternion = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            quaternion = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            quaternion = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    result = tuple(float(value / norm) for value in quaternion)
    return tuple(-value for value in result) if result[0] < 0.0 else result


def usd_wxyz_to_runtime_xyzw(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w, x, y, z = quaternion
    return (x, y, z, w)


def frame_quaternion_usd_wxyz(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
) -> tuple[float, float, float, float]:
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return rotation_matrix_to_usd_quaternion_wxyz(
        np.column_stack((x_axis, y_axis, z_axis))
    )


def quaternion_from_x_axis_usd_wxyz(
    direction: np.ndarray,
) -> tuple[float, float, float, float]:
    x_axis = direction / np.linalg.norm(direction)
    z_axis = PLANE_NORMAL
    if abs(float(np.dot(x_axis, z_axis))) > 0.99:
        z_axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    return frame_quaternion_usd_wxyz(x_axis, z_axis)


def mesh_mass_properties(mesh: MeshData, *, density_kg_m3: float) -> dict[str, Any]:
    coordinates = np.asarray(mesh.points, dtype=np.float64)
    volume = 0.0
    first_moment = np.zeros(3, dtype=np.float64)
    second_moment = np.zeros((3, 3), dtype=np.float64)
    for ia, ib, ic in mesh.triangles:
        vertices = (coordinates[ia], coordinates[ib], coordinates[ic])
        signed_volume = float(np.dot(vertices[0], np.cross(vertices[1], vertices[2]))) / 6.0
        volume += signed_volume
        first_moment += signed_volume * sum(vertices) / 4.0
        for row in range(3):
            for column in range(3):
                same = sum(vertex[row] * vertex[column] for vertex in vertices)
                cross_terms = sum(
                    vertices[left][row] * vertices[right][column]
                    for left in range(3)
                    for right in range(3)
                    if left != right
                )
                second_moment[row, column] += signed_volume * (
                    same / 10.0 + cross_terms / 20.0
                )
    if volume <= 0.0:
        raise ValueError("mass integration requires positive oriented volume")
    center = first_moment / volume
    inertia_origin = np.trace(second_moment) * np.identity(3) - second_moment
    parallel_axis = volume * (
        float(np.dot(center, center)) * np.identity(3) - np.outer(center, center)
    )
    inertia_com = density_kg_m3 * (inertia_origin - parallel_axis)
    inertia_com = 0.5 * (inertia_com + inertia_com.T)
    diagonal, axes = np.linalg.eigh(inertia_com)
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    if np.any(diagonal <= 0.0):
        raise ValueError("mass integration produced non-positive inertia")
    if any(
        diagonal[index] > sum(diagonal) - diagonal[index] + 1.0e-18
        for index in range(3)
    ):
        raise ValueError("principal inertia violates rigid-body triangle inequality")
    principal_usd = rotation_matrix_to_usd_quaternion_wxyz(axes)
    reconstructed = axes @ np.diag(diagonal) @ axes.T
    if not np.allclose(reconstructed, inertia_com, rtol=1.0e-10, atol=1.0e-18):
        raise ValueError("principal inertia reconstruction failed")
    return {
        "density_kg_m3": density_kg_m3,
        "volume_m3": volume,
        "mass_kg": density_kg_m3 * volume,
        "center_of_mass_m": center.tolist(),
        "inertia_tensor_kg_m2": inertia_com.tolist(),
        "diagonal_inertia_kg_m2": diagonal.tolist(),
        "principal_axes_xyzw": list(usd_wxyz_to_runtime_xyzw(principal_usd)),
        "principal_axes_usd_wxyz": list(principal_usd),
        "principal_axes_convention": {
            "runtime": "xyzw",
            "openusd_serialization": "wxyz",
        },
        "source": "actual_connected_watertight_render_mesh_tetrahedral_integration",
        "includes_true_swage_recess": True,
    }


def interaction_frames() -> dict[str, Any]:
    def frame(
        *,
        distance_m: float,
        x_axis: np.ndarray,
        role: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        position, _, _ = centerline(distance_m)
        usd = frame_quaternion_usd_wxyz(x_axis, PLANE_NORMAL)
        result = {
            "position_m": position.tolist(),
            "orientation_xyzw": list(usd_wxyz_to_runtime_xyzw(usd)),
            "orientation_usd_wxyz": list(usd),
            "role": role,
        }
        if extra:
            result.update(extra)
        return result

    _, tip_bodyward, _ = centerline(0.0)
    grasp_distance = PREFERRED_GRASP_FRACTION * ARC_LENGTH_M
    _, grasp_tangent, _ = centerline(grasp_distance)
    _, tail_tangent, _ = centerline(ARC_LENGTH_M)
    identity_usd = (1.0, 0.0, 0.0, 0.0)
    return {
        "schema": "dr.anmar.needle-interaction-frames.v2",
        "asset_id": ASSET_ID,
        "coordinate_convention": {
            "units": "meters",
            "up_axis": "Z",
            "needle_plane": "XY",
            "frame_axes": "local_X_forward_local_Z_needle_plane_normal",
            "runtime_quaternion_order": "xyzw",
            "runtime_orientation_field": "orientation_xyzw",
            "openusd_quaternion_serialization": "wxyz",
            "openusd_orientation_field": "orientation_usd_wxyz",
            "bare_orientation_wxyz_field_forbidden": True,
        },
        "runtime_spawn": {
            "scale_xyz": [1.0, 1.0, 1.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "orientation_usd_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "needle_tip": frame(
            distance_m=0.0,
            x_axis=-tip_bodyward,
            role="finite_curvature_taper_point_puncture_frame",
            extra={
                "penetration_forward_axis_world": (-tip_bodyward).tolist(),
                "bodyward_tangent_world": tip_bodyward.tolist(),
            },
        ),
        "needle_driver_grasp": frame(
            distance_m=grasp_distance,
            x_axis=grasp_tangent,
            role="preferred_flattened_ribbed_driver_grasp_frame",
            extra={
                "arc_fraction_from_tip": PREFERRED_GRASP_FRACTION,
                "tangent_toward_swage_world": grasp_tangent.tolist(),
            },
        ),
        "needle_plane": {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": list(usd_wxyz_to_runtime_xyzw(identity_usd)),
            "orientation_usd_wxyz": list(identity_usd),
            "normal_world": PLANE_NORMAL.tolist(),
            "role": "half_circle_plane_reference",
        },
        "swage_mouth": frame(
            distance_m=ARC_LENGTH_M,
            x_axis=-tail_tangent,
            role="true_blind_swage_recess_mouth",
            extra={"recess_axis_toward_body_world": (-tail_tangent).tolist()},
        ),
        "swage_recess_bottom": frame(
            distance_m=ARC_LENGTH_M - SWAGE_RECESS_DEPTH_M,
            x_axis=-tail_tangent,
            role="blind_swage_recess_bottom",
        ),
    }


def collision_boundaries() -> list[float]:
    intervals = (
        (TIP_CAPSULE_START_M, TIP_TAPER_END_M, 10),
        (TIP_TAPER_END_M, 0.0118, 10),
        (0.0118, 0.0168, GRASP_COLLISION_SEGMENT_COUNT),
        (0.0168, SWAGE_START_M, 5),
        (SWAGE_START_M, ARC_LENGTH_M, 7),
    )
    boundaries = [TIP_CAPSULE_START_M]
    for start, end, count in intervals:
        if abs(boundaries[-1] - start) > 1.0e-12:
            raise ValueError("adaptive collision intervals are discontinuous")
        boundaries.extend(
            float(value)
            for value in np.linspace(start, end, count + 1, dtype=np.float64)[1:]
        )
    if len(boundaries) != COLLISION_SEGMENT_COUNT + 1:
        raise ValueError("adaptive collision did not generate 48 segments")
    return boundaries


def _convex_segment_mesh(
    start_distance_m: float,
    end_distance_m: float,
    *,
    radial_count: int = 20,
) -> tuple[list[Vec3], list[Triangle]]:
    points: list[Vec3] = []
    for distance in (start_distance_m, end_distance_m):
        center, _, _ = centerline(distance)
        for segment in range(radial_count):
            angle = 2.0 * math.pi * segment / radial_count
            point = center + section_offset(distance, angle)
            points.append(tuple(map(float, point)))
    triangles: list[Triangle] = []
    for segment in range(radial_count):
        following = (segment + 1) % radial_count
        triangles.extend(
            (
                (segment, following, radial_count + following),
                (segment, radial_count + following, radial_count + segment),
                (0, following, segment),
                (
                    radial_count,
                    radial_count + segment,
                    radial_count + following,
                ),
            )
        )
    return points, triangles


def build_tip_collision_mesh() -> tuple[list[Vec3], list[Triangle]]:
    distances = (
        0.0,
        TIP_CURVATURE_RADIUS_M * 0.25,
        TIP_CURVATURE_RADIUS_M,
        0.00018,
        0.00042,
        0.00072,
        TIP_COLLISION_END_M,
    )
    radial_count = 20
    tip, _, _ = centerline(0.0)
    points: list[Vec3] = [tuple(map(float, tip))]
    ring_starts: list[int] = []
    for distance in distances[1:]:
        center, _, _ = centerline(distance)
        ring_starts.append(len(points))
        for segment in range(radial_count):
            angle = 2.0 * math.pi * segment / radial_count
            point = center + section_offset(distance, angle, include_ribs=False)
            points.append(tuple(map(float, point)))
    triangles: list[Triangle] = []
    for segment in range(radial_count):
        triangles.append(
            (0, ring_starts[0] + segment, ring_starts[0] + (segment + 1) % radial_count)
        )
    for current, following_ring in pairwise(ring_starts):
        for segment in range(radial_count):
            following = (segment + 1) % radial_count
            triangles.extend(
                (
                    (current + segment, following_ring + segment, following_ring + following),
                    (current + segment, following_ring + following, current + following),
                )
            )
    last = ring_starts[-1]
    for segment in range(1, radial_count - 1):
        triangles.append((last, last + segment + 1, last + segment))
    return points, triangles


def collision_segments() -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    boundaries = collision_boundaries()
    grasp_start = 0.0118
    grasp_end = 0.0168
    for index, (start_distance, end_distance) in enumerate(pairwise(boundaries)):
        start, _, _ = centerline(start_distance)
        end, _, _ = centerline(end_distance)
        vector = end - start
        chord = float(np.linalg.norm(vector))
        midpoint = 0.5 * (start + end)
        samples = np.linspace(start_distance, end_distance, 17, dtype=np.float64)
        maximum_surface_radius = max(maximum_section_radius(float(value)) for value in samples)
        arc_span = end_distance - start_distance
        sagitta = CENTERLINE_RADIUS_M * (
            1.0 - math.cos(0.5 * arc_span / CENTERLINE_RADIUS_M)
        )
        contact_offset = min(
            CONTACT_OFFSET_MAX_M,
            max(CONTACT_OFFSET_MIN_M, 0.08 * maximum_surface_radius),
        )
        if start_distance >= grasp_start - 1.0e-12 and end_distance <= grasp_end + 1.0e-12:
            points, triangles = _convex_segment_mesh(start_distance, end_distance)
            segments.append(
                {
                    "index": index,
                    "type": "convex_grasp_land_segment",
                    "start_distance_m": start_distance,
                    "end_distance_m": end_distance,
                    "points": points,
                    "triangles": triangles,
                    "contact_offset_m": contact_offset,
                    "curvature_sagitta_m": sagitta,
                    "maximum_surface_radius_m": maximum_surface_radius,
                }
            )
            continue
        collision_radius = maximum_surface_radius + sagitta
        quaternion_usd = quaternion_from_x_axis_usd_wxyz(vector)
        segments.append(
            {
                "index": index,
                "type": "capsule",
                "start_distance_m": start_distance,
                "end_distance_m": end_distance,
                "start_m": start.tolist(),
                "end_m": end.tolist(),
                "midpoint_m": midpoint.tolist(),
                "height_m": chord,
                "radius_m": collision_radius,
                "maximum_surface_radius_m": maximum_surface_radius,
                "curvature_sagitta_m": sagitta,
                "orientation_usd_wxyz": list(quaternion_usd),
                "orientation_xyzw": list(usd_wxyz_to_runtime_xyzw(quaternion_usd)),
                "contact_offset_m": contact_offset,
            }
        )
    return segments


def _resized_noise(
    rng: np.random.Generator,
    size: int,
    coarse_width: int,
    coarse_height: int,
) -> np.ndarray:
    values = rng.random((coarse_height, coarse_width), dtype=np.float32)
    image = Image.fromarray(np.round(values * 65535.0).astype(np.uint16))
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 65535.0


def generate_textures(output_root: Path, *, size: int) -> list[Path]:
    rng = np.random.default_rng(220105)
    broad = 0.65 * _resized_noise(rng, size, 17, 17) + 0.35 * _resized_noise(
        rng, size, 47, 47
    )
    longitudinal = (
        0.65 * _resized_noise(rng, size, 4, 513)
        + 0.25 * _resized_noise(rng, size, 9, 257)
        + 0.10 * _resized_noise(rng, size, 67, 257)
    )
    micro = _resized_noise(rng, size, 257, 257)
    base = np.asarray((0.57, 0.59, 0.62), dtype=np.float32)
    neutral_variation = (0.022 * (broad - 0.5) + 0.008 * (longitudinal - 0.5))[
        ..., None
    ]
    base_color = np.clip(base + neutral_variation, 0.0, 1.0)
    roughness = np.clip(
        0.43 + 0.10 * (longitudinal - 0.5) + 0.045 * (broad - 0.5),
        0.36,
        0.52,
    )
    height = 0.68 * longitudinal + 0.20 * micro + 0.12 * broad
    gradient_y, gradient_x = np.gradient(height)
    normal = np.stack(
        (-0.9 * gradient_x, -1.7 * gradient_y, np.ones_like(height)),
        axis=-1,
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-8)
    normal = normal * 0.5 + 0.5

    texture_root = output_root / "textures"
    texture_root.mkdir(parents=True, exist_ok=True)
    paths = [
        texture_root / "needle_satin_basecolor.png",
        texture_root / "needle_satin_roughness.png",
        texture_root / "needle_satin_normal.png",
    ]
    for path, array in zip(
        paths,
        (base_color, roughness, normal),
        strict=True,
    ):
        encoded = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(encoded).save(
            path,
            format="PNG",
            compress_level=9,
            optimize=False,
        )
    return paths


def _usd_number(value: float) -> str:
    return f"{float(value):.12g}"


def _usd_vec(values: Any) -> str:
    return "(" + ", ".join(_usd_number(value) for value in values) + ")"


def _format_array(
    values: Any,
    formatter,
    *,
    indent: int = 12,
    chunk: int = 6,
) -> str:
    prefix = " " * indent
    rendered = [formatter(value) for value in values]
    return ",\n".join(
        prefix + ", ".join(rendered[start : start + chunk])
        for start in range(0, len(rendered), chunk)
    )


def _path(suffix: str) -> str:
    return f"/{ROOT_PRIM}/{suffix}"


def author_geometry_layer(mesh: MeshData) -> str:
    points = np.asarray(mesh.points, dtype=np.float64)
    extent = (points.min(axis=0), points.max(axis=0))
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Geometry-only layer for the inactive DrAnmar 22 mm category needle candidate."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{ROOT_PRIM}"
{{
    def Xform "Geometry"
    {{
        def Mesh "Render"
        {{
            float3[] extent = [{_usd_vec(extent[0])}, {_usd_vec(extent[1])}]
            int[] faceVertexCounts = [{", ".join("3" for _ in mesh.triangles)}]
            int[] faceVertexIndices = [
{_format_array(mesh.triangles, lambda value: ", ".join(map(str, value)), chunk=4)}
            ]
            point3f[] points = [
{_format_array(mesh.points, _usd_vec, chunk=3)}
            ]
            normal3f[] normals = [
{_format_array(mesh.normals, _usd_vec, chunk=3)}
            ] (
                interpolation = "vertex"
            )
            texCoord2f[] primvars:st = [
{_format_array(mesh.face_varying_uvs, _usd_vec, chunk=6)}
            ] (
                interpolation = "faceVarying"
            )
            texCoord2f[] primvars:stMeters = [
{_format_array(mesh.face_varying_uvs_m, _usd_vec, chunk=6)}
            ] (
                interpolation = "faceVarying"
            )
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            token purpose = "render"
            custom bool drAnmar:collisionAuthority = false
            custom bool drAnmar:connected = true
            custom bool drAnmar:trueSwageRecess = true
            custom bool drAnmar:watertight = true
            custom string drAnmar:massPropertyAuthority = "this_render_mesh"
            custom string drAnmar:uvContract = "st_repeats_at_1p5mm_stMeters_is_metric"
        }}
    }}
}}
'''


def _frame_definitions(frames: dict[str, Any]) -> str:
    definitions = []
    for json_name, usd_name in (
        ("needle_tip", "Tip"),
        ("needle_driver_grasp", "DriverGrasp"),
        ("needle_plane", "NeedlePlane"),
        ("swage_mouth", "SwageMouth"),
        ("swage_recess_bottom", "SwageRecessBottom"),
    ):
        values = frames[json_name]
        definitions.append(
            f'''        def Xform "{usd_name}"
        {{
            custom string drAnmar:role = "{values["role"]}"
            double3 xformOp:translate = {_usd_vec(values["position_m"])}
            quatd xformOp:orient = {_usd_vec(values["orientation_usd_wxyz"])}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}'''
        )
    return "\n".join(definitions)


def author_base_layer(frames: dict[str, Any]) -> str:
    return f'''#usda 1.0
(
    subLayers = [
        @./{MATERIALS_USD}@,
        @./{GEOMETRY_USD}@
    ]
    defaultPrim = "{ROOT_PRIM}"
    doc = "Identity, hierarchy, frames, geometry and appearance for an inactive category-level research needle."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{ROOT_PRIM}" (
    assetInfo = {{
        string identifier = "{ASSET_ID}"
        string name = "DrAnmar 22 mm Half-Circle Taper-Point Closure Needle Category Candidate"
        string version = "{ASSET_VERSION}"
    }}
    customData = {{
        bool drAnmarActiveReplacement = false
        bool drAnmarClinicalValidation = false
        bool drAnmarManufacturerDigitalTwin = false
        bool drAnmarNativeQualification = false
        bool drAnmarPhysicsCalibration = false
        string drAnmarGeometrySource = "independent_parametric_DrAnmar_authoring"
        string drAnmarNominal22mmSemantics = "centerline_arc_length"
        string drAnmarPrimaryMaterialContext = "OpenPBR_1_1_MaterialX"
        string drAnmarFallbackMaterialContext = "UsdPreviewSurface"
        string drAnmarRepresentation = "watertight_swept_body_true_swage_recess_ribbed_grasp_land"
        string drAnmarStatus = "inactive_unqualified_category_candidate"
    }}
    displayName = "DrAnmar 22 mm Half-Circle Taper-Point Needle Candidate"
    kind = "component"
)
{{
    def Scope "Frames"
    {{
{_frame_definitions(frames)}
    }}
}}
'''


def author_entry_layer() -> str:
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Inactive 22 mm category-level closure-needle candidate; Physics defaults to none and no clinical or native qualification is claimed."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{ROOT_PRIM}" (
    prepend references = @./{BASE_USD}@
    variants = {{
        string Physics = "none"
    }}
    append variantSets = "Physics"
)
{{
    variantSet "Physics" = {{
        "none" {{
        }}
        "physics" (
            prepend payload = @./{PHYSICS_USD}@
        ) {{
        }}
        "physx" (
            prepend payload = @./{PHYSX_USD}@
        ) {{
        }}
    }}
}}
'''


def author_materials_layer() -> str:
    return f'''#usda 1.0
(
    subLayers = [
        @./{NVIDIA_VENDOR_SUBPATH.as_posix()}/open_pbr_uber_base_class.usda@
    ]
    defaultPrim = "{ROOT_PRIM}"
    doc = "OpenPBR satin-steel appearance with portable PreviewSurface fallback."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{ROOT_PRIM}"
{{
    def Scope "Looks"
    {{
        def Material "SatinSteel" (
            inherits = </open_pbr_uber_base>
        )
        {{
            token outputs:surface.connect = <{_path("Looks/SatinSteel/PreviewSurface")}.outputs:surface>
            custom string drAnmarPrimaryMaterialContext = "OpenPBR_1_1_MaterialX"
            custom string drAnmarFallbackMaterialContext = "UsdPreviewSurface"
            custom bool drAnmarNativeRTXAppearanceQualified = false

            float inputs:base_weight = 1
            color3f inputs:base_color = (0.57, 0.59, 0.62)
            asset inputs:base_color_texture_file = @./textures/needle_satin_basecolor.png@ (
                colorSpace = "sRGB"
            )
            float inputs:base_metalness = 0.96
            float inputs:specular_weight = 1
            color3f inputs:specular_color = (1, 1, 1)
            float inputs:specular_roughness = 0.43
            asset inputs:specular_roughness_texture_file = @./textures/needle_satin_roughness.png@ (
                colorSpace = "raw"
            )
            float inputs:specular_ior = 1.52
            float inputs:specular_roughness_anisotropy = 0.12
            float inputs:coat_weight = 0
            float inputs:subsurface_weight = 0
            float inputs:transmission_weight = 0
            bool inputs:geometry_thin_walled = false
            float inputs:geometry_normal_scale = 0.55
            asset inputs:geometry_normal_texture_file = @./textures/needle_satin_normal.png@ (
                colorSpace = "raw"
            )
            bool inputs:geometry_normal_texture_flip_g = false

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{_path("Looks/SatinSteel/BaseColor")}.outputs:rgb>
                float inputs:metallic = 0.96
                float inputs:roughness.connect = <{_path("Looks/SatinSteel/Roughness")}.outputs:r>
                float inputs:clearcoat = 0
                normal3f inputs:normal.connect = <{_path("Looks/SatinSteel/Normal")}.outputs:rgb>
                token outputs:surface
            }}

            def Shader "Texcoord"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }}

            def Shader "BaseColor"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./textures/needle_satin_basecolor.png@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = <{_path("Looks/SatinSteel/Texcoord")}.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float3 outputs:rgb
            }}

            def Shader "Roughness"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./textures/needle_satin_roughness.png@
                token inputs:sourceColorSpace = "raw"
                float2 inputs:st.connect = <{_path("Looks/SatinSteel/Texcoord")}.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float outputs:r
            }}

            def Shader "Normal"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./textures/needle_satin_normal.png@
                token inputs:sourceColorSpace = "raw"
                float4 inputs:scale = (2, 2, 2, 1)
                float4 inputs:bias = (-1, -1, -1, 0)
                float2 inputs:st.connect = <{_path("Looks/SatinSteel/Texcoord")}.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                normal3f outputs:rgb
            }}
        }}
    }}

    over "Geometry"
    {{
        over "Render" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            rel material:binding = <{_path("Looks/SatinSteel")}>
        }}
    }}
}}
'''


def _collision_shape_blocks(
    segments: list[dict[str, Any]],
    tip_mesh: tuple[list[Vec3], list[Triangle]],
) -> str:
    blocks = []
    for segment in segments:
        name = f"C{segment['index']:03d}"
        if segment["type"] == "capsule":
            radius = segment["radius_m"]
            half_extent = 0.5 * segment["height_m"] + radius
            blocks.append(
                f'''        def Capsule "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            uniform token axis = "X"
            float height = {_usd_number(segment["height_m"])}
            float radius = {_usd_number(radius)}
            float3[] extent = [({_usd_number(-half_extent)}, {_usd_number(-radius)}, {_usd_number(-radius)}), ({_usd_number(half_extent)}, {_usd_number(radius)}, {_usd_number(radius)})]
            bool physics:collisionEnabled = true
            rel material:binding:physics = <{_path("PhysicsMaterials/NeedleSteelPhysics")}>
            token purpose = "guide"
            token visibility = "invisible"
            double3 xformOp:translate = {_usd_vec(segment["midpoint_m"])}
            quatd xformOp:orient = {_usd_vec(segment["orientation_usd_wxyz"])}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}'''
            )
        else:
            points = segment["points"]
            triangles = segment["triangles"]
            coordinates = np.asarray(points, dtype=np.float64)
            extent = (coordinates.min(axis=0), coordinates.max(axis=0))
            blocks.append(
                f'''        def Mesh "{name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float3[] extent = [{_usd_vec(extent[0])}, {_usd_vec(extent[1])}]
            int[] faceVertexCounts = [{", ".join("3" for _ in triangles)}]
            int[] faceVertexIndices = [
{_format_array(triangles, lambda value: ", ".join(map(str, value)), indent=12, chunk=4)}
            ]
            point3f[] points = [
{_format_array(points, _usd_vec, indent=12, chunk=3)}
            ]
            bool physics:collisionEnabled = true
            uniform token physics:approximation = "convexHull"
            rel material:binding:physics = <{_path("PhysicsMaterials/NeedleSteelPhysics")}>
            token purpose = "guide"
            token visibility = "invisible"
            uniform token subdivisionScheme = "none"
        }}'''
            )
    tip_points, tip_triangles = tip_mesh
    coordinates = np.asarray(tip_points, dtype=np.float64)
    extent = (coordinates.min(axis=0), coordinates.max(axis=0))
    blocks.append(
        f'''        def Mesh "PreciseTaperTip" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            float3[] extent = [{_usd_vec(extent[0])}, {_usd_vec(extent[1])}]
            int[] faceVertexCounts = [{", ".join("3" for _ in tip_triangles)}]
            int[] faceVertexIndices = [
{_format_array(tip_triangles, lambda value: ", ".join(map(str, value)), indent=12, chunk=4)}
            ]
            point3f[] points = [
{_format_array(tip_points, _usd_vec, indent=12, chunk=3)}
            ]
            bool physics:collisionEnabled = true
            uniform token physics:approximation = "convexHull"
            rel material:binding:physics = <{_path("PhysicsMaterials/NeedleSteelPhysics")}>
            token purpose = "guide"
            token visibility = "invisible"
            uniform token subdivisionScheme = "none"
        }}'''
    )
    return "\n".join(blocks)


def author_physics_layer(
    mass: dict[str, Any],
    segments: list[dict[str, Any]],
    tip_mesh: tuple[list[Vec3], list[Triangle]],
) -> str:
    capsule_count = sum(item["type"] == "capsule" for item in segments)
    convex_count = len(segments) - capsule_count
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Engine-neutral rigid-body, mass, material and adaptive collision layer for the inactive 22 mm needle candidate."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{ROOT_PRIM}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = {_usd_number(mass["mass_kg"])}
    point3f physics:centerOfMass = {_usd_vec(mass["center_of_mass_m"])}
    float3 physics:diagonalInertia = {_usd_vec(mass["diagonal_inertia_kg_m2"])}
    quatf physics:principalAxes = {_usd_vec(mass["principal_axes_usd_wxyz"])}

    def Scope "PhysicsMaterials"
    {{
        def Material "NeedleSteelPhysics" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = {STATIC_FRICTION}
            float physics:dynamicFriction = {DYNAMIC_FRICTION}
            float physics:restitution = {RESTITUTION}
        }}
    }}

    def Xform "Collision"
    {{
        custom bool drAnmar:adaptiveCenterlinePartition = true
        custom int drAnmar:capsuleSegmentCount = {capsule_count}
        custom int drAnmar:graspConvexSegmentCount = {convex_count}
        custom int drAnmar:segmentCount = {len(segments)}
        custom bool drAnmar:separatePreciseTipShape = true
        custom string drAnmar:authority = "adaptive_48_segment_compound_plus_precise_tip"
{_collision_shape_blocks(segments, tip_mesh)}
    }}
}}
'''


def author_physx_layer(segments: list[dict[str, Any]]) -> str:
    overrides = []
    for segment in segments:
        overrides.append(
            f'''        over "C{segment["index"]:03d}" (
            prepend apiSchemas = ["PhysxCollisionAPI"]
        )
        {{
            float physxCollision:contactOffset = {_usd_number(segment["contact_offset_m"])}
            float physxCollision:restOffset = 0
        }}'''
        )
    overrides.append(
        f'''        over "PreciseTaperTip" (
            prepend apiSchemas = ["PhysxCollisionAPI"]
        )
        {{
            float physxCollision:contactOffset = {_usd_number(CONTACT_OFFSET_MIN_M)}
            float physxCollision:restOffset = 0
        }}'''
    )
    return f'''#usda 1.0
(
    subLayers = [
        @./{PHYSICS_USD}@
    ]
    defaultPrim = "{ROOT_PRIM}"
    doc = "PhysX-specific CCD, solver, contact-offset and conservative material-combine layer."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

over "{ROOT_PRIM}" (
    prepend apiSchemas = ["PhysxRigidBodyAPI"]
)
{{
    bool physxRigidBody:enableCCD = true
    bool physxRigidBody:enableSpeculativeCCD = true
    float physxRigidBody:linearDamping = 0.003
    float physxRigidBody:angularDamping = 0.006
    int physxRigidBody:solverPositionIterationCount = 24
    int physxRigidBody:solverVelocityIterationCount = 8
    float physxRigidBody:maxDepenetrationVelocity = 0.5

    over "PhysicsMaterials"
    {{
        over "NeedleSteelPhysics" (
            prepend apiSchemas = ["PhysxMaterialAPI"]
        )
        {{
            uniform token physxMaterial:frictionCombineMode = "min"
            uniform token physxMaterial:restitutionCombineMode = "min"
        }}
    }}

    over "Collision"
    {{
{chr(10).join(overrides)}
    }}
}}
'''


def _write_usdc_from_usda(source: str, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".usda",
        dir=destination.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        stream.write(source)
        temporary_path = Path(stream.name)
    try:
        layer = Sdf.Layer.FindOrOpen(str(temporary_path))
        if layer is None:
            raise RuntimeError("OpenUSD could not parse generated geometry layer")
        if not layer.Export(str(destination), args={"format": "usdc"}):
            raise RuntimeError("OpenUSD could not export binary geometry layer")
    finally:
        temporary_path.unlink(missing_ok=True)


def copy_nvidia_vendor_inputs(output_root: Path) -> None:
    destination_root = output_root / NVIDIA_VENDOR_SUBPATH
    destination_root.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in NVIDIA_VENDOR_INPUTS.items():
        source = NVIDIA_SOURCE_VENDOR_ROOT / name
        if sha256(source) != expected_hash:
            raise RuntimeError(f"NVIDIA vendor source hash drifted: {source}")
        destination = destination_root / name
        shutil.copyfile(source, destination)
        if sha256(destination) != expected_hash:
            raise RuntimeError(f"NVIDIA vendor copy hash mismatch: {destination}")
    (destination_root / "PROVENANCE.md").write_text(
        f"""# NVIDIA PhysicalAI SimReady Material Input

This directory contains byte-identical copies of the OpenPBR 1.1 MaterialX
base class and MIT-0 license from NVIDIA PhysicalAI-SimReady-Materials v0.2.0.

Upstream: https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials

- `open_pbr_uber_base_class.usda`: `{NVIDIA_VENDOR_INPUTS["open_pbr_uber_base_class.usda"]}`
- `LICENSE.md`: `{NVIDIA_VENDOR_INPUTS["LICENSE.md"]}`

No NVIDIA geometry, texture, patient data, biomechanics, or clinical
calibration is included. The package does not claim native RTX appearance
qualification.
""",
        encoding="utf-8",
    )


def geometry_contract() -> dict[str, Any]:
    return {
        "schema": "dr.anmar.needle-22mm-half-circle-taper-geometry.v1",
        "asset_id": ASSET_ID,
        "version": ASSET_VERSION,
        "status": "inactive_unqualified_category_candidate",
        "active_replacement": False,
        "clinical_validation": False,
        "physics_calibration": False,
        "manufacturer_digital_twin": False,
        "native_qualified": False,
        "nominal_22_mm_semantics": "centerline_arc_length",
        "centerline": {
            "arc_length_m": ARC_LENGTH_M,
            "arc_radians": math.pi,
            "radius_m": CENTERLINE_RADIUS_M,
            "shape": "one_half_circle",
            "plane": "XY",
        },
        "design_authority": {
            "selected_dimensional_input": {
                "path": SELECTED_DIMENSIONAL_INPUT.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": SELECTED_DIMENSIONAL_INPUT_SHA256,
                "fields_adopted": [
                    "centerline_arc_length_m_0p022",
                    "body_diameter_m_0p00053",
                    "collision_segment_count_48",
                ],
            },
            "comparison_only_not_averaged": {
                "path": "physics_next/needles/dr-anmar-needle-v1.json",
                "sha256": COMPARISON_ONLY_PHYSICS_NEXT_SHA256,
                "different_body_diameter_m": 0.00052,
                "different_collision_capsule_count": 40,
            },
        },
        "manufacturer_category_provenance": MANUFACTURER_CATEGORY_SOURCE,
        "render_geometry": {
            "shape": "single_connected_watertight_swept_half_circle",
            "body_diameter_m": BODY_DIAMETER_M,
            "tip": {
                "type": "finite_curvature_taper_point",
                "curvature_radius_m": TIP_CURVATURE_RADIUS_M,
                "taper_end_distance_m": TIP_TAPER_END_M,
            },
            "swage": {
                "type": "reduced_outer_body_with_true_blind_recess",
                "transition_start_distance_m": SWAGE_START_M,
                "full_reduction_distance_m": SWAGE_FULL_M,
                "outer_diameter_m": 2.0 * SWAGE_OUTER_RADIUS_M,
                "recess_diameter_m": 2.0 * SWAGE_RECESS_RADIUS_M,
                "recess_depth_m": SWAGE_RECESS_DEPTH_M,
                "cosmetic_overlapping_cylinder": False,
            },
            "grasp_land": {
                "start_distance_m": GRASP_LAND_START_M,
                "end_distance_m": GRASP_LAND_END_M,
                "preferred_frame_arc_fraction": PREFERRED_GRASP_FRACTION,
                "cross_section": "smoothly_blended_flattened_superellipse",
                "longitudinal_rib_count_per_face": 3,
                "rib_height_m": GRASP_RIB_HEIGHT_M,
            },
            "arc_resolution": ARC_RESOLUTION,
            "radial_resolution": RADIAL_RESOLUTION,
            "normal_interpolation": "vertex",
            "uv_interpolation": "faceVarying",
            "metric_uv_primvar": "primvars:stMeters",
            "repeated_uv_primvar": "primvars:st",
            "texture_repeat_m": TEXTURE_REPEAT_M,
        },
        "collision": {
            "representation": "adaptive_48_segments_plus_precise_taper_tip",
            "segment_count": COLLISION_SEGMENT_COUNT,
            "grasp_convex_segment_count": GRASP_COLLISION_SEGMENT_COUNT,
            "render_mesh_collision_authority": False,
            "tip_shape": "separate_convex_hull_from_authored_taper",
        },
        "runtime_spawn_contract": {
            "scale_xyz": [1.0, 1.0, 1.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "orientation_usd_wxyz": [1.0, 0.0, 0.0, 0.0],
            "root_scale_op_authored": False,
        },
    }


def physics_profile(
    mass: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    capsule_count = sum(item["type"] == "capsule" for item in segments)
    return {
        "schema": "dr.anmar.needle-22mm-half-circle-taper-physics.v1",
        "asset_id": ASSET_ID,
        "status": "inactive_unqualified_category_candidate",
        "active_replacement": False,
        "clinical_validation": False,
        "physics_calibration": False,
        "native_qualified": False,
        "material": {
            "class": "surgical_stainless_steel_engineering_proxy",
            "density_kg_m3": STEEL_DENSITY_KG_M3,
            "static_friction": STATIC_FRICTION,
            "dynamic_friction": DYNAMIC_FRICTION,
            "ordered_friction": STATIC_FRICTION >= DYNAMIC_FRICTION,
            "restitution": RESTITUTION,
            "physx_friction_combine_mode": "min",
            "physx_restitution_combine_mode": "min",
            "max_combine_forbidden": True,
            "adhesion": False,
            "magnetism": False,
            "suction": False,
            "implicit_attachment": False,
        },
        "mass_properties": mass,
        "collision": {
            "segment_count": len(segments),
            "capsule_count": capsule_count,
            "grasp_convex_segment_count": len(segments) - capsule_count,
            "separate_precise_tip_shape_count": 1,
            "capsule_height_semantics": "cylinder_spine_excluding_hemispherical_caps",
            "adaptive_partition_m": collision_boundaries(),
            "maximum_segment_span_m": max(
                item["end_distance_m"] - item["start_distance_m"] for item in segments
            ),
            "maximum_curvature_sagitta_m": max(
                item["curvature_sagitta_m"] for item in segments
            ),
            "contact_offset_range_m": [
                min(item["contact_offset_m"] for item in segments),
                max(item["contact_offset_m"] for item in segments),
            ],
            "rest_offset_m": 0.0,
        },
        "rigid_body": {
            "ccd": True,
            "speculative_ccd": True,
            "solver_position_iterations": 24,
            "solver_velocity_iterations": 8,
            "max_depenetration_velocity_m_s": 0.5,
        },
        "layer_contract": {
            "entry": PRIMARY_USD,
            "base": BASE_USD,
            "geometry": GEOMETRY_USD,
            "materials": MATERIALS_USD,
            "physics": PHYSICS_USD,
            "physx": PHYSX_USD,
            "variant_set": "Physics",
            "variant_choices": ["none", "physics", "physx"],
            "default_variant": "none",
        },
        "qualification_required": [
            "native_Isaac_PhysX_load_and_contact_evidence",
            "analytic_pickup_first_attempt_parity",
            "single_arm_mid_air_retention",
            "two_arm_handover_retention",
            "safe_bite_approach_without_premature_contact",
            "calibrated_puncture_force_displacement_evidence",
        ],
    }


def collision_report(segments: list[dict[str, Any]]) -> dict[str, Any]:
    capsule_segments = [item for item in segments if item["type"] == "capsule"]
    grasp_segments = [item for item in segments if item["type"] != "capsule"]
    return {
        "schema": "dr.anmar.needle-collision-report.v1",
        "asset_id": ASSET_ID,
        "segment_count": len(segments),
        "capsule_segment_count": len(capsule_segments),
        "grasp_convex_segment_count": len(grasp_segments),
        "precise_tip_shape_count": 1,
        "adaptive_regions": [
            {"name": "taper", "start_m": TIP_CAPSULE_START_M, "end_m": TIP_TAPER_END_M, "count": 10},
            {"name": "body", "start_m": TIP_TAPER_END_M, "end_m": 0.0118, "count": 10},
            {"name": "grasp_land", "start_m": 0.0118, "end_m": 0.0168, "count": 16},
            {"name": "body_to_swage", "start_m": 0.0168, "end_m": SWAGE_START_M, "count": 5},
            {"name": "swage", "start_m": SWAGE_START_M, "end_m": ARC_LENGTH_M, "count": 7},
        ],
        "coverage": {
            "capsule_radius_formula": "sampled_maximum_surface_radius_plus_exact_centerline_arc_sagitta",
            "sampled_maximum_undercoverage_m": 0.0,
            "maximum_capsule_geometric_overreach_m": max(
                item["curvature_sagitta_m"] for item in capsule_segments
            ),
            "tip_mesh_end_m": TIP_COLLISION_END_M,
            "first_segment_start_m": TIP_CAPSULE_START_M,
            "tip_to_capsule_overlap_m": TIP_COLLISION_END_M - TIP_CAPSULE_START_M,
        },
        "shape_count_at_2400_environments": 2400 * (len(segments) + 1),
        "render_mesh_collision_authority": False,
        "segments": [
            {
                key: value
                for key, value in item.items()
                if key not in {"points", "triangles"}
            }
            for item in segments
        ],
    }


def write_readme(path: Path) -> None:
    path.write_text(
        f"""# DrAnmar 22 mm Half-Circle Taper-Point Needle Candidate

This is a new, inactive, category-level closure-needle research asset. It is
not registered in `needle_thread.py`, task configuration, the learning path,
or a runtime catalog. The entry layer defaults to `Physics=none`; `physics`
and `physx` must be selected deliberately during later qualification.

The nominal 22 mm dimension means **centerline arc length**. A half-circle
therefore has radius `0.022 / pi = {CENTERLINE_RADIUS_M:.12f} m`. The
0.530 mm body is adopted from the existing DrAnmar 22 mm asset profile, whose
SHA-256 is pinned in `geometry_contract.json`. The separate newer
`physics_next` 0.520 mm profile is recorded as comparison-only and is not
silently averaged.

## Geometry and contact

- one connected, watertight swept body at final metric scale;
- finite-curvature taper-point apex;
- reduced swage body with a real 0.82 mm blind recess in the mass mesh;
- smoothly flattened driver land centered on the preferred 65% arc frame;
- three shallow longitudinal ribs on each broad grasp face;
- authored smooth normals, repeating UVs, and metric `primvars:stMeters`;
- 48 adaptive collision segments: 32 capsules and 16 convex grasp-land
  segments, plus one separate precise taper-tip convex shape;
- mass, center of mass, full inertia tensor, principal inertia and principal
  axes integrated from the actual watertight recessed render solid;
- ordered friction (`static >= dynamic`) with PhysX `min` combine, so this
  candidate cannot unexpectedly select the jaw's larger coefficient through
  `max` combine.

## Appearance

The satin-steel look uses deterministic 2048 px base-color, roughness and
normal maps. NVIDIA PhysicalAI SimReady Materials v0.2.0 provides the pinned
MIT-0 OpenPBR 1.1 MaterialX base class. `UsdPreviewSurface` is a portable
fallback. Roughness is restrained to 0.36-0.52, anisotropy is 0.12, and
clearcoat is disabled. Native RTX appearance is not yet qualified.

## Manufacturer-category boundary

The J&J MedTech / Ethicon catalog is used only to ground the published
category facts “22 mm”, “1/2 circle”, “taper point”, longitudinal ribbing and
factory swaging. DrAnmar independently authored all dimensions not listed as
catalog facts, all geometry, collision, mass properties and textures. This is
not an SH-1 or CT-3 replica, manufacturer digital twin, clinically validated
device, physics-calibrated model, or patient-care asset.

Regenerate from the asset-extension root:

```bash
uv run --no-project \\
  --with numpy==2.2.6 --with Pillow==11.3.0 --with usd-core==25.11 \\
  python tools/generate_needle_22_half_circle_taper_candidate.py
```
""",
        encoding="utf-8",
    )


def write_provenance(path: Path) -> None:
    path.write_text(
        f"""# Provenance

DrAnmar independently generated the swept geometry, true swage recess,
flattened and ribbed grasp land, collision decomposition, frames, textures and
mass properties under Apache-2.0.

Selected internal dimensional input:

- `data/Props/SurgicalClosure/Needle/physics_profile.json`
- SHA-256 `{SELECTED_DIMENSIONAL_INPUT_SHA256}`
- adopted: 22 mm centerline arc, 0.530 mm body, 48 collision segments

Comparison-only input, not averaged:

- `physics_next/needles/dr-anmar-needle-v1.json`
- SHA-256 `{COMPARISON_ONLY_PHYSICS_NEXT_SHA256}`
- differs: 0.520 mm body and 40 capsules

Manufacturer-category source:

- J&J MedTech / Ethicon, `157677-230117-AH Suture Catalog`
- {MANUFACTURER_CATEGORY_SOURCE["url"]}
- used only for the category facts 22 mm needle length, one-half circle,
  taper point, longitudinal ribbing and factory swaging
- no vendor mesh, drawing, texture, body diameter, taper dimension, swage
  dimension, patient data or proprietary digital geometry is included

NVIDIA's byte-identical MIT-0 OpenPBR base-class inputs are documented under
`vendor/nvidia_physicalai_simready_materials_v0_2_0/PROVENANCE.md`.

No manufacturer digital-twin, native RTX appearance, physics calibration,
clinical validation, regulatory approval, or patient-care claim is made.
""",
        encoding="utf-8",
    )


def write_notice(path: Path) -> None:
    path.write_text(
        f"""DrAnmar 22 mm Half-Circle Taper-Point Needle Category Candidate {ASSET_VERSION}

DrAnmar independently authored the geometry, topology, collision, textures,
frames and mass-property integration. The referenced J&J MedTech / Ethicon
catalog contributes category terminology only; no vendor digital asset or
proprietary device geometry is redistributed.

The unmodified NVIDIA OpenPBR 1.1 MaterialX base class and MIT-0 license are
redistributed from PhysicalAI-SimReady-Materials v0.2.0. No NVIDIA geometry,
texture, patient data, biomechanics or clinical calibration is included.

This package is inactive, unqualified, research-only and not clinically
validated or approved for patient care.
""",
        encoding="utf-8",
    )


def write_manifest(output_root: Path, *, texture_size_px: int) -> dict[str, Any]:
    members: dict[str, Any] = {}
    for path in sorted(
        candidate
        for candidate in output_root.rglob("*")
        if candidate.is_file() and candidate.name != "asset_manifest.json"
    ):
        relative = path.relative_to(output_root).as_posix()
        vendor_prefix = NVIDIA_VENDOR_SUBPATH.as_posix() + "/"
        if relative in {
            f"{vendor_prefix}LICENSE.md",
            f"{vendor_prefix}open_pbr_uber_base_class.usda",
        }:
            license_id = "MIT-0"
            provenance = "verbatim_NVIDIA_PhysicalAI_SimReady_Materials_v0.2.0"
        elif relative == "LICENSE.txt":
            license_id = "Apache-2.0"
            provenance = "canonical_Apache-2.0_text"
        else:
            license_id = "Apache-2.0"
            provenance = "independently_generated_by_DrAnmar"
        members[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "license": license_id,
            "provenance": provenance,
        }
    return {
        "schema": "dr.anmar.sim-ready-asset-manifest.v3",
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "primary_usd": PRIMARY_USD,
        "status": "inactive_unqualified_category_candidate",
        "active_replacement": False,
        "clinical_validation": False,
        "physics_calibration": False,
        "manufacturer_digital_twin": False,
        "native_qualified": False,
        "dependency_complete_directory": True,
        "generator": {
            "path": "tools/generate_needle_22_half_circle_taper_candidate.py",
            "sha256": sha256(Path(__file__)),
            "dependencies": GENERATOR_DEPENDENCIES,
            "deterministic": True,
            "texture_size_px": texture_size_px,
        },
        "design_inputs": {
            "selected_dimensional_input": {
                "path": SELECTED_DIMENSIONAL_INPUT.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": SELECTED_DIMENSIONAL_INPUT_SHA256,
            },
            "comparison_only_physics_next": {
                "path": "physics_next/needles/dr-anmar-needle-v1.json",
                "sha256": COMPARISON_ONLY_PHYSICS_NEXT_SHA256,
                "used_as_dimensional_authority": False,
            },
            "manufacturer_category_source": MANUFACTURER_CATEGORY_SOURCE,
        },
        "license": "mixed",
        "licenses": {
            "DrAnmar_authored_members": "Apache-2.0",
            "NVIDIA_OpenPBR_members": "MIT-0",
        },
        "material_contexts": {
            "primary": "OpenPBR_1.1_MaterialX_inherited",
            "fallback": "UsdPreviewSurface",
            "native_rtx_appearance_qualified": False,
        },
        "quaternion_contract": {
            "runtime_and_JSON_canonical": "xyzw",
            "openusd_quat_serialization": "wxyz",
            "explicit_conversion_fields": True,
            "bare_orientation_wxyz_forbidden": True,
        },
        "members": members,
    }


def validate_inputs() -> None:
    if sha256(SELECTED_DIMENSIONAL_INPUT) != SELECTED_DIMENSIONAL_INPUT_SHA256:
        raise RuntimeError(
            "selected 22 mm dimensional input changed; review rather than silently regenerating"
        )
    if STATIC_FRICTION < DYNAMIC_FRICTION:
        raise RuntimeError("static friction must be greater than or equal to dynamic friction")
    if abs(CENTERLINE_RADIUS_M * math.pi - ARC_LENGTH_M) > 1.0e-15:
        raise RuntimeError("22 mm centerline arc contract drifted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--texture-size", type=int, default=TEXTURE_SIZE)
    args = parser.parse_args()
    if args.texture_size < 2048:
        parser.error("--texture-size must be at least 2048")

    validate_inputs()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mesh = build_render_mesh()
    topology = topology_report(mesh)
    if not topology["watertight"]:
        raise RuntimeError("render mesh is not watertight")
    if topology["connected_component_count"] != 1:
        raise RuntimeError("render mesh is not a single component")
    if topology["minimum_triangle_area_m2"] <= 0.0:
        raise RuntimeError("render mesh contains a degenerate triangle")
    if len(mesh.face_varying_uvs) != 3 * len(mesh.triangles):
        raise RuntimeError("face-varying UV count does not match triangle corners")
    if len(mesh.face_varying_uvs_m) != len(mesh.face_varying_uvs):
        raise RuntimeError("metric and repeated UV counts differ")

    mass = mesh_mass_properties(mesh, density_kg_m3=STEEL_DENSITY_KG_M3)
    segments = collision_segments()
    if len(segments) != COLLISION_SEGMENT_COUNT:
        raise RuntimeError("collision segment count drifted")
    if sum(item["type"] != "capsule" for item in segments) != GRASP_COLLISION_SEGMENT_COUNT:
        raise RuntimeError("grasp collision segment count drifted")
    tip_mesh = build_tip_collision_mesh()
    frames = interaction_frames()

    copy_nvidia_vendor_inputs(output_root)
    generate_textures(output_root, size=args.texture_size)
    _write_usdc_from_usda(
        author_geometry_layer(mesh),
        output_root / GEOMETRY_USD,
    )
    (output_root / PRIMARY_USD).write_text(author_entry_layer(), encoding="utf-8")
    (output_root / BASE_USD).write_text(author_base_layer(frames), encoding="utf-8")
    (output_root / MATERIALS_USD).write_text(author_materials_layer(), encoding="utf-8")
    (output_root / PHYSICS_USD).write_text(
        author_physics_layer(mass, segments, tip_mesh),
        encoding="utf-8",
    )
    (output_root / PHYSX_USD).write_text(
        author_physx_layer(segments),
        encoding="utf-8",
    )

    contract = geometry_contract()
    (output_root / "geometry_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coordinates = np.asarray(mesh.points, dtype=np.float64)
    report = {
        "schema": "dr.anmar.needle-22mm-half-circle-taper-report.v1",
        "asset_id": ASSET_ID,
        "topology": topology,
        "bounds_m": {
            "minimum": coordinates.min(axis=0).tolist(),
            "maximum": coordinates.max(axis=0).tolist(),
        },
        "mesh_mass_properties": mass,
        "finite_tip_position_m": centerline(0.0)[0].tolist(),
        "swage_mouth_center_m": centerline(ARC_LENGTH_M)[0].tolist(),
        "swage_recess_bottom_center_m": centerline(
            ARC_LENGTH_M - SWAGE_RECESS_DEPTH_M
        )[0].tolist(),
        "true_swage_recess": True,
        "face_varying_uv_count": len(mesh.face_varying_uvs),
        "face_varying_metric_uv_count": len(mesh.face_varying_uvs_m),
        "vertex_normal_count": len(mesh.normals),
        "generated_geometry": GEOMETRY_USD,
    }
    (output_root / "geometry_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "physics_profile.json").write_text(
        json.dumps(physics_profile(mass, segments), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "collision_report.json").write_text(
        json.dumps(collision_report(segments), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "interaction_frames.json").write_text(
        json.dumps(frames, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "provenance.json").write_text(
        json.dumps(
            {
                "schema": "dr.anmar.asset-provenance.v1",
                "asset_id": ASSET_ID,
                "manufacturer_category_source": MANUFACTURER_CATEGORY_SOURCE,
                "independent_dranmar_authoring": True,
                "manufacturer_digital_twin": False,
                "clinical_validation": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_readme(output_root / "README.md")
    write_provenance(output_root / "PROVENANCE.md")
    write_notice(output_root / "NOTICE.txt")
    (output_root / "LICENSE.txt").write_bytes(CANONICAL_APACHE_LICENSE.read_bytes())
    manifest = write_manifest(output_root, texture_size_px=args.texture_size)
    (output_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "vertices": len(mesh.points),
                "triangles": len(mesh.triangles),
                "watertight": topology["watertight"],
                "connected_components": topology["connected_component_count"],
                "mass_kg": mass["mass_kg"],
                "collision_segments": len(segments),
                "collision_shapes_including_tip": len(segments) + 1,
                "default_physics_variant": "none",
                "active_replacement": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
