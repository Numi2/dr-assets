#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Generate the isolated DrAnmar T1 legacy-envelope needle candidate.

This tool does not modify or select any active needle asset. It authors a
single connected watertight render mesh, a separate continuous compound
capsule collision representation, mesh-derived mass properties, explicit
interaction frames, and deterministic satin-steel textures.

Run from the asset-extension root:

    uv run --no-project --with numpy==2.2.6 --with Pillow==11.3.0 \
      python tools/generate_needle_t1_compatibility.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, deque
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_SUBPATH = Path("data/Props/SurgicalClosure/NeedleT1Compatibility")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / ASSET_SUBPATH
CANONICAL_APACHE_LICENSE = REPOSITORY_ROOT / "data/Props/SurgicalClosure/Needle/LICENSE.txt"
NVIDIA_SOURCE_VENDOR_ROOT = (
    REPOSITORY_ROOT
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit/visual/vendor"
    / "nvidia_physicalai_simready_materials_v0_2_0"
)
NVIDIA_VENDOR_SUBPATH = Path("vendor/nvidia_physicalai_simready_materials_v0_2_0")
NVIDIA_VENDOR_INPUTS = {
    "LICENSE.md": ("18f74283f08ff1ed39a9c46dbe2622146d45f771023c3dbd9c631bb058e1421b"),
    "open_pbr_uber_base_class.usda": (
        "bb76ff9fa9cd74b86b6be4ed3c6ed79cdca15eff6d603ca571bdf9ce21e10c5f"
    ),
}
ROOT_PRIM = "DrAnmarNeedleT1Compatibility"
ASSET_ID = "dranmar-needle-t1-legacy-envelope-candidate-v1"
ASSET_VERSION = "1.1.0"

CENTER_X_M = 0.02004
CENTERLINE_RADIUS_M = 0.019154
BODY_RADIUS_M = 0.000825
BODY_DIAMETER_M = 2.0 * BODY_RADIUS_M
TIP_CURVATURE_RADIUS_M = 0.000025
TAPER_LENGTH_M = 0.005
STEEL_DENSITY_KG_M3 = 8000.0
ARC_RESOLUTION = 384
RADIAL_RESOLUTION = 32
COLLISION_CAPSULE_COUNT = 96
TEXTURE_SIZE = 2048
TIP_POSITION = (CENTER_X_M, -CENTERLINE_RADIUS_M, 0.0)
TAIL_POSITION = (CENTER_X_M, CENTERLINE_RADIUS_M, 0.0)
PLANE_NORMAL = (0.0, 0.0, 1.0)
GENERATOR_DEPENDENCIES = {
    "numpy": "2.2.6",
    "Pillow": "11.3.0",
}
PRESERVED_SOURCE_ASSETS = (
    Path("data/Props/Surgical_needle/needle.usd"),
    Path("data/Props/Surgical_needle/needle_sdf.usd"),
    Path("data/Props/SurgicalClosure/Needle/dranmar_needle.usda"),
)

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


class MeshData:
    """Small immutable-enough container for deterministic mesh output."""

    def __init__(
        self,
        *,
        points: list[Vec3],
        triangles: list[Triangle],
        face_varying_uvs: list[Vec2],
        normals: list[Vec3],
        ring_distances_m: list[float],
    ) -> None:
        self.points = tuple(points)
        self.triangles = tuple(triangles)
        self.face_varying_uvs = tuple(face_varying_uvs)
        self.normals = tuple(normals)
        self.ring_distances_m = tuple(ring_distances_m)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smootherstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def cross_section_radius(distance_m: float) -> float:
    """Round a finite-curvature apex, then smoothly grow to body radius."""

    distance_m = min(math.pi * CENTERLINE_RADIUS_M, max(0.0, distance_m))
    if distance_m <= TIP_CURVATURE_RADIUS_M:
        squared = 2.0 * TIP_CURVATURE_RADIUS_M * distance_m - distance_m * distance_m
        return math.sqrt(max(0.0, squared))
    if distance_m < TAPER_LENGTH_M:
        fraction = (distance_m - TIP_CURVATURE_RADIUS_M) / (TAPER_LENGTH_M - TIP_CURVATURE_RADIUS_M)
        return TIP_CURVATURE_RADIUS_M + (BODY_RADIUS_M - TIP_CURVATURE_RADIUS_M) * _smootherstep(
            fraction
        )
    return BODY_RADIUS_M


def centerline(distance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point, unit tangent toward tail, and in-plane section normal."""

    theta = -math.pi / 2.0 - distance_m / CENTERLINE_RADIUS_M
    point = np.asarray(
        (
            CENTER_X_M + CENTERLINE_RADIUS_M * math.cos(theta),
            CENTERLINE_RADIUS_M * math.sin(theta),
            0.0,
        ),
        dtype=np.float64,
    )
    tangent = np.asarray(
        (math.sin(theta), -math.cos(theta), 0.0),
        dtype=np.float64,
    )
    section_normal = np.asarray(
        (math.cos(theta), math.sin(theta), 0.0),
        dtype=np.float64,
    )
    return point, tangent, section_normal


def _ring_distances() -> list[float]:
    tip = TIP_CURVATURE_RADIUS_M
    special = [tip * fraction for fraction in (0.02, 0.08, 0.18, 0.35, 0.55, 0.75, 1.0)]
    total = math.pi * CENTERLINE_RADIUS_M
    uniform = np.linspace(
        2.0 * tip,
        total,
        ARC_RESOLUTION + 1,
        dtype=np.float64,
    )
    return sorted({round(float(value), 15) for value in (*special, *uniform) if value > 0.0})


def _triangle_uvs(
    distance_fraction: float,
    next_distance_fraction: float,
    segment: int,
) -> tuple[Vec2, Vec2, Vec2, Vec2, Vec2, Vec2]:
    low = segment / RADIAL_RESOLUTION
    high = (segment + 1) / RADIAL_RESOLUTION
    return (
        (distance_fraction, low),
        (distance_fraction, high),
        (next_distance_fraction, high),
        (distance_fraction, low),
        (next_distance_fraction, high),
        (next_distance_fraction, low),
    )


def _vertex_normals(
    points: list[Vec3],
    triangles: list[Triangle],
) -> list[Vec3]:
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


def signed_mesh_volume(
    points: tuple[Vec3, ...] | list[Vec3],
    triangles: tuple[Triangle, ...] | list[Triangle],
) -> float:
    coordinates = np.asarray(points, dtype=np.float64)
    volume = 0.0
    for a, b, c in triangles:
        volume += (
            float(
                np.dot(
                    coordinates[a],
                    np.cross(coordinates[b], coordinates[c]),
                )
            )
            / 6.0
        )
    return volume


def build_render_mesh() -> MeshData:
    total = math.pi * CENTERLINE_RADIUS_M
    distances = _ring_distances()
    points: list[Vec3] = [TIP_POSITION]
    ring_starts: list[int] = []
    for distance in distances:
        center, _, normal = centerline(distance)
        binormal = np.asarray(PLANE_NORMAL, dtype=np.float64)
        radius = cross_section_radius(distance)
        ring_starts.append(len(points))
        for segment in range(RADIAL_RESOLUTION):
            angle = 2.0 * math.pi * segment / RADIAL_RESOLUTION
            point = center + radius * (math.cos(angle) * normal + math.sin(angle) * binormal)
            points.append(tuple(map(float, point)))

    tail_index = len(points)
    points.append(TAIL_POSITION)
    triangles: list[Triangle] = []
    uvs: list[Vec2] = []

    first_ring = ring_starts[0]
    first_fraction = distances[0] / total
    for segment in range(RADIAL_RESOLUTION):
        current = first_ring + segment
        following = first_ring + (segment + 1) % RADIAL_RESOLUTION
        triangles.append((0, following, current))
        uvs.extend(
            (
                (0.0, (segment + 0.5) / RADIAL_RESOLUTION),
                (first_fraction, (segment + 1) / RADIAL_RESOLUTION),
                (first_fraction, segment / RADIAL_RESOLUTION),
            )
        )

    for ring_index in range(len(ring_starts) - 1):
        current_start = ring_starts[ring_index]
        next_start = ring_starts[ring_index + 1]
        current_fraction = distances[ring_index] / total
        next_fraction = distances[ring_index + 1] / total
        for segment in range(RADIAL_RESOLUTION):
            a = current_start + segment
            b = current_start + (segment + 1) % RADIAL_RESOLUTION
            c = next_start + (segment + 1) % RADIAL_RESOLUTION
            d = next_start + segment
            triangles.extend(((a, b, c), (a, c, d)))
            uvs.extend(
                _triangle_uvs(
                    current_fraction,
                    next_fraction,
                    segment,
                )
            )

    last_ring = ring_starts[-1]
    for segment in range(RADIAL_RESOLUTION):
        current = last_ring + segment
        following = last_ring + (segment + 1) % RADIAL_RESOLUTION
        triangles.append((tail_index, current, following))
        uvs.extend(
            (
                (1.0, (segment + 0.5) / RADIAL_RESOLUTION),
                (1.0, segment / RADIAL_RESOLUTION),
                (1.0, (segment + 1) / RADIAL_RESOLUTION),
            )
        )

    volume = signed_mesh_volume(points, triangles)
    if volume < 0.0:
        triangles = [(a, c, b) for a, b, c in triangles]
        uvs = [
            value
            for index in range(0, len(uvs), 3)
            for value in (uvs[index], uvs[index + 2], uvs[index + 1])
        ]
    normals = _vertex_normals(points, triangles)
    return MeshData(
        points=points,
        triangles=triangles,
        face_varying_uvs=uvs,
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
        area = 0.5 * float(np.linalg.norm(np.cross(points[b] - points[a], points[c] - points[a])))
        minimum_area = min(minimum_area, area)
        for left, right in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((left, right)))
            edge_counts[edge] += 1
            adjacency[left].add(right)
            adjacency[right].add(left)

    unseen = set(adjacency)
    component_count = 0
    while unseen:
        component_count += 1
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
        "connected_component_count": component_count,
        "minimum_triangle_area_m2": minimum_area,
        "signed_volume_m3": signed_mesh_volume(mesh.points, mesh.triangles),
        "watertight": all(count == 2 for count in edge_counts.values()),
    }


def mesh_mass_properties(
    mesh: MeshData,
    *,
    density_kg_m3: float,
) -> dict[str, Any]:
    """Integrate signed tetrahedra formed by each oriented face and origin."""

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
                second_moment[row, column] += signed_volume * (same / 10.0 + cross_terms / 20.0)
    if volume <= 0.0:
        raise ValueError("mesh mass integration requires positive oriented volume")
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
        raise ValueError("mesh integration produced non-positive principal inertia")
    principal_axes_usd_wxyz = rotation_matrix_to_usd_quaternion_wxyz(axes)
    return {
        "density_kg_m3": density_kg_m3,
        "volume_m3": volume,
        "mass_kg": density_kg_m3 * volume,
        "center_of_mass_m": center.tolist(),
        "inertia_tensor_kg_m2": inertia_com.tolist(),
        "diagonal_inertia_kg_m2": diagonal.tolist(),
        "principal_axes_xyzw": list(usd_wxyz_to_runtime_xyzw(principal_axes_usd_wxyz)),
        "principal_axes_usd_wxyz": list(principal_axes_usd_wxyz),
        "principal_axes_convention": {
            "runtime": "xyzw",
            "openusd_serialization": "wxyz",
        },
        "source": "actual_connected_watertight_render_mesh_tetrahedral_integration",
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
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    result = tuple(float(value / norm) for value in quaternion)
    if result[0] < 0.0:
        result = tuple(-value for value in result)
    return result


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
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation_matrix_to_usd_quaternion_wxyz(rotation)


def interaction_frames() -> dict[str, Any]:
    total = math.pi * CENTERLINE_RADIUS_M
    grasp_distance = 0.65 * total
    grasp_point, grasp_tangent, _ = centerline(grasp_distance)
    tip_quaternion_usd_wxyz = frame_quaternion_usd_wxyz(
        np.asarray((1.0, 0.0, 0.0)),
        np.asarray(PLANE_NORMAL),
    )
    grasp_quaternion_usd_wxyz = frame_quaternion_usd_wxyz(
        grasp_tangent,
        np.asarray(PLANE_NORMAL),
    )
    identity_usd_wxyz = (1.0, 0.0, 0.0, 0.0)
    return {
        "coordinate_convention": {
            "units": "meters",
            "up_axis": "Z",
            "frame_axes": "local_X_forward_local_Z_needle_plane_normal",
            "runtime_quaternion_order": "xyzw",
            "runtime_orientation_field": "orientation_xyzw",
            "openusd_quaternion_serialization": "wxyz",
            "openusd_orientation_field": "orientation_usd_wxyz",
        },
        "needle_tip": {
            "position_m": list(TIP_POSITION),
            "orientation_xyzw": list(usd_wxyz_to_runtime_xyzw(tip_quaternion_usd_wxyz)),
            "orientation_usd_wxyz": list(tip_quaternion_usd_wxyz),
            "forward_axis_world": [1.0, 0.0, 0.0],
            "bodyward_tangent_world": [-1.0, 0.0, 0.0],
            "role": "rounded_taper_tip",
        },
        "needle_driver_grasp": {
            "arc_fraction_from_tip": 0.65,
            "position_m": grasp_point.tolist(),
            "orientation_xyzw": list(usd_wxyz_to_runtime_xyzw(grasp_quaternion_usd_wxyz)),
            "orientation_usd_wxyz": list(grasp_quaternion_usd_wxyz),
            "tangent_toward_tail_world": grasp_tangent.tolist(),
            "role": "candidate_driver_grasp",
        },
        "needle_plane": {
            "position_m": [CENTER_X_M, 0.0, 0.0],
            "orientation_xyzw": list(usd_wxyz_to_runtime_xyzw(identity_usd_wxyz)),
            "orientation_usd_wxyz": list(identity_usd_wxyz),
            "normal_world": list(PLANE_NORMAL),
            "role": "needle_plane_reference",
        },
        "needle_tail": {
            "position_m": list(TAIL_POSITION),
            "orientation_xyzw": list(usd_wxyz_to_runtime_xyzw(identity_usd_wxyz)),
            "orientation_usd_wxyz": list(identity_usd_wxyz),
            "role": "blunt_attachment_end_reference",
        },
    }


def collision_capsules() -> list[dict[str, Any]]:
    total = math.pi * CENTERLINE_RADIUS_M
    distances = np.linspace(
        0.0,
        total,
        COLLISION_CAPSULE_COUNT + 1,
        dtype=np.float64,
    )
    capsules = []
    for index, (start_distance, end_distance) in enumerate(pairwise(distances)):
        start, _, _ = centerline(float(start_distance))
        end, _, _ = centerline(float(end_distance))
        vector = end - start
        length = float(np.linalg.norm(vector))
        direction = vector / length
        midpoint = 0.5 * (start + end)
        radius = max(
            TIP_CURVATURE_RADIUS_M,
            cross_section_radius(float(start_distance)),
            cross_section_radius(float(end_distance)),
        )
        quaternion_usd_wxyz = quaternion_from_z_axis_usd_wxyz(direction)
        capsules.append(
            {
                "index": index,
                "start_m": start.tolist(),
                "end_m": end.tolist(),
                "midpoint_m": midpoint.tolist(),
                "height_m": length,
                "radius_m": radius,
                "orientation_usd_wxyz": list(quaternion_usd_wxyz),
            }
        )
    return capsules


def quaternion_from_z_axis_usd_wxyz(
    direction: np.ndarray,
) -> tuple[float, float, float, float]:
    direction = direction / np.linalg.norm(direction)
    dot = float(direction[2])
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    real = math.sqrt(max(0.0, (1.0 + dot) / 2.0))
    scale = 1.0 / max(2.0 * real, 1.0e-12)
    cross = np.cross(np.asarray((0.0, 0.0, 1.0)), direction)
    quaternion = (
        real,
        float(cross[0] * scale),
        float(cross[1] * scale),
        float(cross[2] * scale),
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)


def _resized_noise(
    rng: np.random.Generator,
    size: int,
    coarse_width: int,
    coarse_height: int,
) -> np.ndarray:
    values = rng.random(
        (coarse_height, coarse_width),
        dtype=np.float32,
    )
    image = Image.fromarray(np.round(values * 65535.0).astype(np.uint16))
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 65535.0


def generate_textures(output_root: Path, *, size: int) -> list[Path]:
    rng = np.random.default_rng(190154)
    broad = 0.60 * _resized_noise(rng, size, 17, 17) + 0.40 * _resized_noise(rng, size, 53, 53)
    brushed = (
        0.55 * _resized_noise(rng, size, 3, 257)
        + 0.30 * _resized_noise(rng, size, 5, 509)
        + 0.15 * _resized_noise(rng, size, 31, 257)
    )
    micro = _resized_noise(rng, size, 257, 257)
    base = np.asarray((0.61, 0.64, 0.69), dtype=np.float32)
    tint = np.stack(
        (
            broad - 0.5,
            0.75 * (broad - 0.5),
            0.55 * (broad - 0.5),
        ),
        axis=-1,
    )
    base_color = base + 0.035 * tint + 0.010 * (brushed - 0.5)[..., None]
    roughness = np.clip(
        0.54 + 0.11 * (brushed - 0.5) + 0.035 * (broad - 0.5),
        0.45,
        0.64,
    )
    height = 0.72 * brushed + 0.18 * micro + 0.10 * broad
    gradient_y, gradient_x = np.gradient(height)
    normal = np.stack(
        (-1.1 * gradient_x, -2.4 * gradient_y, np.ones_like(height)),
        axis=-1,
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-8)
    normal = normal * 0.5 + 0.5

    texture_root = output_root / "textures"
    texture_root.mkdir(parents=True, exist_ok=True)
    legacy_normal_path = texture_root / "needle_satin_normal.jpg"
    legacy_normal_path.unlink(missing_ok=True)
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


def _format_array(values: Any, formatter, *, indent: int = 12, chunk: int = 6) -> str:
    prefix = " " * indent
    rendered = [formatter(value) for value in values]
    return ",\n".join(
        prefix + ", ".join(rendered[start : start + chunk])
        for start in range(0, len(rendered), chunk)
    )


def _material_block() -> str:
    return f"""    def Scope "Looks"
    {{
        def Material "SatinSteel" (
            inherits = </open_pbr_uber_base>
        )
        {{
            token outputs:surface.connect = <{_path("Looks/SatinSteel/PreviewSurface")}.outputs:surface>
            custom string drAnmarFallbackMaterialContext = "UsdPreviewSurface"
            custom string drAnmarOpenPBRInputMode = "uv_texture_maps"
            custom string drAnmarPrimaryMaterialContext = "OpenPBR_1_1_MaterialX"
            custom bool drAnmarNativeRTXAppearanceQualified = false

            float inputs:base_weight = 1
            color3f inputs:base_color = (0.61, 0.64, 0.69)
            asset inputs:base_color_texture_file = @./textures/needle_satin_basecolor.png@ (
                colorSpace = "sRGB"
            )
            float inputs:base_diffuse_roughness = 0
            float inputs:base_metalness = 0.96
            float inputs:specular_weight = 1
            color3f inputs:specular_color = (1, 1, 1)
            float inputs:specular_roughness = 0.54
            asset inputs:specular_roughness_texture_file = @./textures/needle_satin_roughness.png@ (
                colorSpace = "raw"
            )
            float inputs:specular_ior = 1.52
            float inputs:specular_roughness_anisotropy = 0.18
            float inputs:coat_weight = 0
            float inputs:subsurface_weight = 0
            float inputs:transmission_weight = 0
            bool inputs:geometry_thin_walled = false
            float inputs:geometry_normal_scale = 0.72
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

    def Scope "PhysicsMaterials"
    {{
        def Material "NeedleSteelPhysics" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            float physics:staticFriction = 0.32
            float physics:dynamicFriction = 0.24
            float physics:restitution = 0.01
            uniform token physxMaterial:frictionCombineMode = "max"
            uniform token physxMaterial:restitutionCombineMode = "min"
        }}
    }}"""


def _path(suffix: str) -> str:
    return f"/{ROOT_PRIM}/{suffix}"


def _capsule_block(capsules: list[dict[str, Any]]) -> str:
    blocks = []
    for capsule in capsules:
        blocks.append(
            f"""        def Capsule "C{capsule["index"]:03d}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
        )
        {{
            uniform token axis = "Z"
            double height = {_usd_number(capsule["height_m"])}
            double radius = {_usd_number(capsule["radius_m"])}
            bool physics:collisionEnabled = true
            float physxCollision:contactOffset = 1.5e-05
            float physxCollision:restOffset = 0
            rel material:binding:physics = <{_path("PhysicsMaterials/NeedleSteelPhysics")}>
            token purpose = "guide"
            token visibility = "invisible"
            double3 xformOp:translate = {_usd_vec(capsule["midpoint_m"])}
            quatd xformOp:orient = {_usd_vec(capsule["orientation_usd_wxyz"])}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}"""
        )
    return "\n".join(blocks)


def _frame_block(frames: dict[str, Any]) -> str:
    definitions = []
    for json_name, usd_name in (
        ("needle_tip", "Tip"),
        ("needle_driver_grasp", "DriverGrasp"),
        ("needle_plane", "NeedlePlane"),
        ("needle_tail", "Tail"),
    ):
        values = frames[json_name]
        orientation = values["orientation_usd_wxyz"]
        definitions.append(
            f'''        def Xform "{usd_name}"
        {{
            custom string drAnmar:role = "{values["role"]}"
            double3 xformOp:translate = {_usd_vec(values["position_m"])}
            quatd xformOp:orient = {_usd_vec(orientation)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}'''
        )
    return "\n".join(definitions)


def author_usda(
    mesh: MeshData,
    mass: dict[str, Any],
    capsules: list[dict[str, Any]],
    frames: dict[str, Any],
) -> str:
    points = np.asarray(mesh.points, dtype=np.float64)
    extent = (points.min(axis=0), points.max(axis=0))
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Inactive DrAnmar T1 legacy-envelope needle candidate with NVIDIA OpenPBR 1.1 MaterialX primary and PreviewSurface fallback; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    subLayers = [
        @./vendor/nvidia_physicalai_simready_materials_v0_2_0/open_pbr_uber_base_class.usda@
    ]
    upAxis = "Z"
)

def Xform "{ROOT_PRIM}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    assetInfo = {{
        string identifier = "{ASSET_ID}"
        string name = "DrAnmar T1 Legacy-Envelope Needle Candidate"
        string version = "{ASSET_VERSION}"
    }}
    displayName = "DrAnmar Needle T1 Compatibility Candidate"
    kind = "component"
    customData = {{
        bool drAnmarActiveReplacement = false
        bool drAnmarClinicalValidation = false
        string drAnmarFallbackMaterialContext = "UsdPreviewSurface"
        string drAnmarPrimaryMaterialContext = "OpenPBR_1_1_MaterialX"
        string drAnmarPromotionGate = "pickup_handover_retention_parity_required"
        string drAnmarRepresentation = "connected_watertight_render_mesh_plus_compound_capsules"
        string drAnmarVendorDependency = "NVIDIA_PhysicalAI_SimReady_Materials_v0.2.0_MIT-0"
    }}
)
{{
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = {_usd_number(mass["mass_kg"])}
    point3f physics:centerOfMass = {_usd_vec(mass["center_of_mass_m"])}
    float3 physics:diagonalInertia = {_usd_vec(mass["diagonal_inertia_kg_m2"])}
    quatf physics:principalAxes = {_usd_vec(mass["principal_axes_usd_wxyz"])}
    bool physxRigidBody:enableCCD = true
    bool physxRigidBody:enableSpeculativeCCD = true
    float physxRigidBody:linearDamping = 0.003
    float physxRigidBody:angularDamping = 0.006
    int physxRigidBody:solverPositionIterationCount = 24
    int physxRigidBody:solverVelocityIterationCount = 8
    float physxRigidBody:maxDepenetrationVelocity = 0.5

{_material_block()}

    def Xform "Geometry"
    {{
        def Mesh "Render" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
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
            rel material:binding = <{_path("Looks/SatinSteel")}>
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            token purpose = "render"
            custom bool drAnmar:connected = true
            custom bool drAnmar:watertight = true
            custom string drAnmar:massPropertyAuthority = "this_render_mesh"
        }}
    }}

    def Xform "Collision"
    {{
        custom bool drAnmar:continuousCenterlineCoverage = true
        custom int drAnmar:capsuleCount = {len(capsules)}
        custom string drAnmar:authority = "compound_capsules_only"
{_capsule_block(capsules)}
    }}

    def Scope "Frames"
    {{
{_frame_block(frames)}
    }}
}}
'''


def geometry_contract() -> dict[str, Any]:
    return {
        "schema": "dr.anmar.needle-t1-compatibility-geometry.v1",
        "asset_id": ASSET_ID,
        "version": ASSET_VERSION,
        "status": "inactive_qualification_candidate",
        "active_replacement": False,
        "promotion_gate": "pickup_handover_retention_parity_required",
        "clinical_validation": False,
        "coordinate_system": {
            "units": "meters",
            "up_axis": "Z",
            "needle_plane": "XY",
        },
        "runtime_spawn_contract": {
            "candidate_scale_xyz": [1.0, 1.0, 1.0],
            "candidate_root_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "legacy_active_scale_xyz": [0.4, 0.4, 0.4],
            "legacy_active_asset": (
                "data/Props/Surgical_needle/needle_sdf.usd"
            ),
            "path_only_substitution_allowed": False,
            "promotion_requires_composed_world_space_frame_parity": True,
        },
        "legacy_envelope": {
            "circle_center_m": [CENTER_X_M, 0.0, 0.0],
            "centerline_radius_m": CENTERLINE_RADIUS_M,
            "centerline_arc_radians": math.pi,
            "centerline_arc_length_m": math.pi * CENTERLINE_RADIUS_M,
            "tip_position_m": list(TIP_POSITION),
            "midpoint_position_m": [
                CENTER_X_M - CENTERLINE_RADIUS_M,
                0.0,
                0.0,
            ],
            "tail_position_m": list(TAIL_POSITION),
        },
        "render_geometry": {
            "shape": "single_connected_watertight_half_circle_taper_point",
            "body_diameter_m": BODY_DIAMETER_M,
            "tip_curvature_radius_m": TIP_CURVATURE_RADIUS_M,
            "taper_length_m": TAPER_LENGTH_M,
            "arc_resolution": ARC_RESOLUTION,
            "radial_resolution": RADIAL_RESOLUTION,
            "uv_interpolation": "faceVarying",
            "normal_interpolation": "vertex",
        },
        "collision": {
            "representation": "continuous_compound_capsules",
            "capsule_count": COLLISION_CAPSULE_COUNT,
            "render_mesh_collision_authority": False,
        },
    }


def physics_profile(
    mass: dict[str, Any],
    capsules: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "dr.anmar.needle-t1-compatibility-physics.v1",
        "asset_id": ASSET_ID,
        "status": "unqualified_candidate",
        "active_replacement": False,
        "clinical_validation": False,
        "material": {
            "class": "surgical_stainless_steel_engineering_proxy",
            "density_kg_m3": STEEL_DENSITY_KG_M3,
            "static_friction": 0.32,
            "dynamic_friction": 0.24,
            "restitution": 0.01,
        },
        "mass_properties": mass,
        "collision": {
            "capsule_count": len(capsules),
            "maximum_centerline_segment_m": max(capsule["height_m"] for capsule in capsules),
            "minimum_radius_m": min(capsule["radius_m"] for capsule in capsules),
            "maximum_radius_m": max(capsule["radius_m"] for capsule in capsules),
            "continuous_by_shared_centerline_endpoints": True,
            "contact_offset_m": 1.5e-05,
            "rest_offset_m": 0.0,
        },
        "rigid_body": {
            "ccd": True,
            "speculative_ccd": True,
            "solver_position_iterations": 24,
            "solver_velocity_iterations": 8,
        },
        "qualification_required": [
            "legacy_analytic_pickup_parity",
            "single_arm_retention_parity",
            "two_arm_handover_parity",
            "mid_air_transport_retention_parity",
            "native_Isaac_contact_and_CCD_evidence",
        ],
    }


def write_readme(path: Path) -> None:
    path.write_text(
        """# DrAnmar Needle T1 Compatibility Candidate

This package is an isolated, inactive qualification candidate for the legacy
T1 pickup/handover envelope. It is not wired into `needle_thread.py`, any task
configuration, or any runtime asset selector.

The candidate preserves the legacy analytic frame:

- circle center X: 20.040 mm;
- centerline radius: 19.154 mm;
- tip frame: `(20.040, -19.154, 0.000) mm`;
- round body diameter: approximately 1.650 mm;
- XY needle plane with +Z plane normal.

The candidate is authored at final metric size and must spawn at scale
`(1, 1, 1)` with runtime identity quaternion `(0, 0, 0, 1)`. The active legacy
needle is scaled by `(0.4, 0.4, 0.4)`. A path-only swap is therefore forbidden:
promotion must change the scale deliberately and prove composed world-space
tip, center, radius, grasp-frame, and root-orientation parity.

Unlike the inherited legacy USD, the render body is one connected watertight
taper-point solid. The apex has a finite 25 micrometre curvature seed rather
than an infinite mathematical sharpness. High-resolution render geometry has
authored UVs, smooth normals, and deterministic 2048-pixel satin-steel PBR
maps. NVIDIA's vendored OpenPBR 1.1 MaterialX graph is the primary material
context; all three maps drive that graph directly. `UsdPreviewSurface` is an
explicit portable fallback. This package makes no native RTX qualification
claim.

Runtime metadata uses the project-canonical quaternion order `(x, y, z, w)`
under fields named `orientation_xyzw`. OpenUSD-authored quaternions are
separately exposed as `orientation_usd_wxyz` because `quatd` and `quatf`
serialize `(w, x, y, z)`. The two conventions are never silently conflated.

Collision is separate from rendering: 96 overlapping capsules follow the
entire centerline and share segment endpoints. The render mesh has no collision
authority. Mass, center of mass, full inertia tensor, principal inertia, and
principal axes are integrated from the actual watertight render mesh at an
8000 kg/m3 stainless-steel engineering density seed.

Regenerate from the asset-extension root:

```bash
uv run --no-project --with numpy==2.2.6 --with Pillow==11.3.0 \\
  python tools/generate_needle_t1_compatibility.py
```

## Promotion boundary

Do not replace the active needle merely because this asset is cleaner or looks
better. Promotion requires held-out parity for the qualified analytic pickup,
single-arm retention, two-arm handover, mid-air transport retention, and
native Isaac contact/CCD evidence. No such parity is claimed by this package.

This is category-level research geometry. It is not a manufacturer digital
twin, clinically validated, physics-calibrated, or approved for patient care.
""",
        encoding="utf-8",
    )


def write_license(path: Path) -> None:
    path.write_bytes(CANONICAL_APACHE_LICENSE.read_bytes())


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

This directory vendors two immutable inputs from NVIDIA's
`PhysicalAI-SimReady-Materials` release `v0.2.0`:

- `open_pbr_uber_base_class.usda`
- `LICENSE.md`

Upstream:
`https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials`

The source copies are pinned in this repository under:
`data/Props/SurgicalTissue/NeedleReadyTissueUnit/visual/vendor/nvidia_physicalai_simready_materials_v0_2_0`

Vendored member SHA-256 values:

- `open_pbr_uber_base_class.usda`:
  `{NVIDIA_VENDOR_INPUTS["open_pbr_uber_base_class.usda"]}`
- `LICENSE.md`:
  `{NVIDIA_VENDOR_INPUTS["LICENSE.md"]}`

Both files are copied byte-for-byte and remain NVIDIA MIT-0 content. The base
class supplies the OpenPBR 1.1 MaterialX graph. This needle package uses no
NVIDIA texture, geometry, biomechanics, patient data, or clinical calibration.

This provenance document is DrAnmar-authored under Apache-2.0.
""",
        encoding="utf-8",
    )


def write_notice(path: Path) -> None:
    path.write_text(
        f"""DrAnmar Needle T1 Compatibility Candidate {ASSET_VERSION}

The geometry, topology, compound collision layout, textures, mass-property
integration, frames, and authoring code were independently created by the
Dr.Anmar project. No vendor needle mesh, texture, patient data, or proprietary
medical-device geometry is included.

The unmodified NVIDIA OpenPBR 1.1 MaterialX base class and its MIT-0 license
are redistributed from PhysicalAI-SimReady-Materials v0.2.0. See
`vendor/nvidia_physicalai_simready_materials_v0_2_0/PROVENANCE.md`. No NVIDIA
texture, geometry, patient data, or clinical calibration is redistributed.

This package is an inactive research candidate and is not clinically validated
or approved for patient care.
""",
        encoding="utf-8",
    )


def write_manifest(
    output_root: Path,
    *,
    texture_size_px: int,
) -> dict[str, Any]:
    members = {}
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
        "primary_usd": "dranmar_needle_t1_compatibility.usda",
        "status": "inactive_qualification_candidate",
        "active_replacement": False,
        "clinical_validation": False,
        "dependency_complete_directory": True,
        "generator": {
            "path": "tools/generate_needle_t1_compatibility.py",
            "sha256": sha256(Path(__file__)),
            "dependencies": GENERATOR_DEPENDENCIES,
            "deterministic": True,
            "texture_size_px": texture_size_px,
        },
        "provenance": {
            "geometry": "independent_parametric_DrAnmar_authoring",
            "textures": "seeded_procedural_no_vendor_or_patient_imagery",
            "vendor_assets_modified": False,
        },
        "preserved_source_assets": {
            relative.as_posix(): {
                "bytes": (REPOSITORY_ROOT / relative).stat().st_size,
                "sha256": sha256(REPOSITORY_ROOT / relative),
                "modified_by_generator": False,
            }
            for relative in PRESERVED_SOURCE_ASSETS
        },
        "license": "mixed",
        "licenses": {
            "DrAnmar_authored_members": "Apache-2.0",
            "NVIDIA_OpenPBR_members": "MIT-0",
        },
        "vendor_dependencies": {
            "nvidia_physicalai_simready_materials": {
                "provider": "NVIDIA",
                "upstream": ("https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials"),
                "release": "v0.2.0",
                "license": "MIT-0",
                "material_interface": "OpenPBR_1.1_MaterialX",
                "source_root": NVIDIA_SOURCE_VENDOR_ROOT.relative_to(REPOSITORY_ROOT).as_posix(),
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
        "material_input_contract": {
            "uv_authored": True,
            "openpbr_input_mode": "uv_texture_maps",
            "openpbr_texture_inputs": [
                "base_color_texture_file",
                "specular_roughness_texture_file",
                "geometry_normal_texture_file",
            ],
            "texture_color_contract": {
                "basecolor": "sRGB",
                "roughness": "raw",
                "normal": ("raw_tangent_space_renderer_orientation_pending_native_validation"),
            },
        },
        "quaternion_contract": {
            "runtime_and_JSON_canonical": "xyzw",
            "openusd_quat_serialization": "wxyz",
            "explicit_conversion_fields": True,
        },
        "members": members,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=TEXTURE_SIZE,
    )
    args = parser.parse_args()
    if args.texture_size < 2048:
        parser.error("--texture-size must be at least 2048")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mesh = build_render_mesh()
    topology = topology_report(mesh)
    if not topology["watertight"]:
        raise RuntimeError("candidate render mesh is not watertight")
    if topology["connected_component_count"] != 1:
        raise RuntimeError("candidate render mesh is not a single component")
    if topology["minimum_triangle_area_m2"] <= 0.0:
        raise RuntimeError("candidate render mesh contains a degenerate triangle")
    if len(mesh.face_varying_uvs) != 3 * len(mesh.triangles):
        raise RuntimeError("face-varying UV count does not match triangle corners")

    mass = mesh_mass_properties(mesh, density_kg_m3=STEEL_DENSITY_KG_M3)
    capsules = collision_capsules()
    frames = interaction_frames()
    copy_nvidia_vendor_inputs(output_root)
    generate_textures(output_root, size=args.texture_size)
    (output_root / "dranmar_needle_t1_compatibility.usda").write_text(
        author_usda(mesh, mass, capsules, frames),
        encoding="utf-8",
    )
    (output_root / "geometry_contract.json").write_text(
        json.dumps(geometry_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "dr.anmar.needle-t1-compatibility-report.v1",
        "asset_id": ASSET_ID,
        "topology": topology,
        "bounds_m": {
            "minimum": np.min(np.asarray(mesh.points), axis=0).tolist(),
            "maximum": np.max(np.asarray(mesh.points), axis=0).tolist(),
        },
        "mesh_mass_properties": mass,
        "tip_position_m": list(mesh.points[0]),
        "tail_cap_center_m": list(mesh.points[-1]),
        "face_varying_uv_count": len(mesh.face_varying_uvs),
        "vertex_normal_count": len(mesh.normals),
        "collision_capsule_count": len(capsules),
        "generated_usd": "dranmar_needle_t1_compatibility.usda",
    }
    (output_root / "geometry_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "physics_profile.json").write_text(
        json.dumps(
            physics_profile(mass, capsules),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "interaction_frames.json").write_text(
        json.dumps(frames, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_root / "README.md")
    write_license(output_root / "LICENSE.txt")
    write_notice(output_root / "NOTICE.txt")
    manifest = write_manifest(
        output_root,
        texture_size_px=args.texture_size,
    )
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
                "collision_capsules": len(capsules),
                "active_replacement": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
