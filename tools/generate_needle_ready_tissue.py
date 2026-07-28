#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Author the canonical Dr.Anmar needle-ready layered tissue asset.

The generator is deliberately dependency-free.  It creates nested,
co-registered tetrahedral LODs with stable semantic coordinates.  Runtime
solvers may consume the authored OpenUSD TetMesh directly; no solver-specific
cooking step is part of the geometry identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ASSET_ID = "dranmar-needle-ready-tissue-v2"
ASSET_NAME = "DrAnmar Needle-Ready Tissue Unit"
ASSET_VERSION = "2.2.0"
ROOT_PRIM = "DrAnmarNeedleReadyTissue"
CATALOG_SUBPATH = Path("data/Props/SurgicalTissue/NeedleReadyTissueUnit")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = REPOSITORY_ROOT / CATALOG_SUBPATH
DEFAULT_CONTRACT = DEFAULT_ASSET_ROOT / "geometry_contract.json"

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
Tet = tuple[int, int, int, int]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class TissueMesh:
    lod: str
    points: tuple[Vec3, ...]
    parametric_coordinates: tuple[Vec4, ...]
    point_components: tuple[int, ...]
    tetrahedra: tuple[Tet, ...]
    tetrahedron_layers: tuple[int, ...]
    tetrahedron_fibers: tuple[Vec3, ...]
    surface_triangles: tuple[Triangle, ...]
    material_face_sets: dict[str, tuple[int, ...]]
    semantic_face_sets: dict[str, tuple[int, ...]]
    node_sets: dict[str, tuple[int, ...]]
    element_sets: dict[str, tuple[int, ...]]
    extent_min: Vec3
    extent_max: Vec3
    volume_m3: float
    minimum_tetra_volume_m3: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_tetra_volume(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float:
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    ad = tuple(d[index] - a[index] for index in range(3))
    cross = (
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    )
    return sum(ab[index] * cross[index] for index in range(3)) / 6.0


def _squared_distance(a: Vec3, b: Vec3) -> float:
    return sum((a[index] - b[index]) ** 2 for index in range(3))


def _determinant(vectors: tuple[Vec3, Vec3, Vec3]) -> float:
    a, b, c = vectors
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def tetrahedron_quality(mesh: TissueMesh) -> dict[str, float]:
    """Report scale-independent sliver and aspect-ratio diagnostics.

    Mean ratio and vertex-scaled Jacobian are normalized to one for a regular
    tetrahedron. Edge ratio is one for equal edge lengths and increases with
    anisotropy.
    """

    mean_ratios: list[float] = []
    scaled_jacobians: list[float] = []
    edge_ratios: list[float] = []
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for tetrahedron in mesh.tetrahedra:
        points = tuple(mesh.points[index] for index in tetrahedron)
        volume = abs(signed_tetra_volume(*points))
        squared_edges = [
            _squared_distance(points[left], points[right])
            for left, right in edge_pairs
        ]
        mean_ratios.append(
            12.0 * (3.0 * volume) ** (2.0 / 3.0) / sum(squared_edges)
        )
        edge_ratios.append(
            math.sqrt(max(squared_edges) / min(squared_edges))
        )
        vertex_jacobians = []
        for vertex_index, vertex in enumerate(points):
            edges = tuple(
                tuple(other[axis] - vertex[axis] for axis in range(3))
                for other_index, other in enumerate(points)
                if other_index != vertex_index
            )
            denominator = math.prod(
                math.sqrt(sum(component * component for component in edge))
                for edge in edges
            )
            vertex_jacobians.append(
                min(
                    1.0,
                    math.sqrt(2.0)
                    * abs(_determinant(edges))
                    / max(denominator, 1.0e-30),
                )
            )
        scaled_jacobians.append(min(vertex_jacobians))
    return {
        "minimum_mean_ratio": min(mean_ratios),
        "p01_mean_ratio": _percentile(mean_ratios, 0.01),
        "median_mean_ratio": _percentile(mean_ratios, 0.50),
        "minimum_scaled_jacobian": min(scaled_jacobians),
        "p01_scaled_jacobian": _percentile(scaled_jacobians, 0.01),
        "median_scaled_jacobian": _percentile(scaled_jacobians, 0.50),
        "maximum_edge_ratio": max(edge_ratios),
        "p99_edge_ratio": _percentile(edge_ratios, 0.99),
        "median_edge_ratio": _percentile(edge_ratios, 0.50),
    }


def validate_tetrahedron_quality(
    lod: str,
    quality: dict[str, float],
    gates: dict[str, float],
) -> None:
    comparisons = {
        "minimum_mean_ratio": (
            quality["minimum_mean_ratio"],
            float(gates["minimum_mean_ratio"]),
            lambda actual, threshold: actual >= threshold,
        ),
        "minimum_scaled_jacobian": (
            quality["minimum_scaled_jacobian"],
            float(gates["minimum_scaled_jacobian"]),
            lambda actual, threshold: actual >= threshold,
        ),
        "maximum_edge_ratio": (
            quality["maximum_edge_ratio"],
            float(gates["maximum_edge_ratio"]),
            lambda actual, threshold: actual <= threshold,
        ),
    }
    failed = {
        name: {"actual": actual, "threshold": threshold}
        for name, (actual, threshold, passed) in comparisons.items()
        if not passed(actual, threshold)
    }
    if failed:
        raise ValueError(f"{lod} tetrahedron quality gates failed: {failed}")


def _triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    return (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )


def _surface_faces(
    points: list[Vec3],
    tetrahedra: list[Tet],
    point_components: list[int],
) -> list[Triangle]:
    stored: dict[tuple[int, int, int], Triangle] = {}
    counts: Counter[tuple[int, int, int]] = Counter()
    for a, b, c, d in tetrahedra:
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(face))
            counts[key] += 1
            stored.setdefault(key, face)
    if any(count not in (1, 2) for count in counts.values()):
        raise ValueError("tetrahedral mesh contains a non-manifold face")

    component_points: dict[int, list[Vec3]] = {0: [], 1: []}
    for point, component in zip(points, point_components, strict=True):
        component_points[component].append(point)
    component_centers = {
        component: tuple(
            sum(point[axis] for point in values) / len(values)
            for axis in range(3)
        )
        for component, values in component_points.items()
    }

    surface: list[Triangle] = []
    for key in sorted(key for key, count in counts.items() if count == 1):
        face = stored[key]
        component = point_components[face[0]]
        if any(point_components[index] != component for index in face):
            raise ValueError("surface face crosses tissue components")
        center = component_centers[component]
        a, b, c = (points[index] for index in face)
        normal = _triangle_normal(a, b, c)
        centroid = tuple(
            (a[index] + b[index] + c[index]) / 3.0
            for index in range(3)
        )
        outward = tuple(centroid[index] - center[index] for index in range(3))
        if sum(normal[index] * outward[index] for index in range(3)) < 0.0:
            face = (face[0], face[2], face[1])
        surface.append(face)
    return surface


def _layer_for_fraction(
    layers: list[dict[str, Any]],
    depth_fraction: float,
) -> tuple[int, str]:
    matches = []
    for index, layer in enumerate(layers):
        low, high = map(float, layer["depth_fraction"])
        if low - 1.0e-12 <= depth_fraction <= high + 1.0e-12:
            matches.append((index, str(layer["id"])))
    if len(matches) != 1:
        raise ValueError(
            f"depth fraction {depth_fraction} belongs to {len(matches)} layers"
        )
    return matches[0]


def _fiber_for_layer(layer_id: str, component: int) -> Vec3:
    sign = -1.0 if component == 0 else 1.0
    if layer_id == "surface":
        return (0.0, 1.0, 0.0)
    if layer_id == "fascia":
        inverse_root_two = 1.0 / math.sqrt(2.0)
        return (sign * inverse_root_two, inverse_root_two, 0.0)
    return (sign, 0.0, 0.0)


def _weighted_sines(
    coordinate_m: float,
    terms: tuple[tuple[float, float, float], ...],
) -> float:
    """Evaluate a bounded deterministic multi-scale rest-shape field."""

    return sum(
        weight * math.sin(2.0 * math.pi * coordinate_m / wavelength + phase)
        for wavelength, weight, phase in terms
    )


def _wound_lip_offsets(
    y: float,
    geometry: dict[str, Any],
) -> tuple[float, float]:
    """Return correlated but non-identical left/right wound-lip offsets."""

    primary_wavelength = float(geometry["wound_irregularity_wavelength_m"])
    shared_amplitude = float(geometry["wound_irregularity_amplitude_m"])
    independent_amplitude = float(
        geometry["wound_lip_independent_amplitude_m"]
    )
    shared = shared_amplitude * _weighted_sines(
        y,
        (
            (primary_wavelength, 0.62, 0.18),
            (primary_wavelength * 0.53, 0.25, 1.17),
            (primary_wavelength * 1.71, 0.13, -0.64),
        ),
    )
    left_detail = independent_amplitude * _weighted_sines(
        y,
        (
            (primary_wavelength * 0.79, 0.68, 0.83),
            (primary_wavelength * 0.41, 0.32, -1.28),
        ),
    )
    right_detail = independent_amplitude * _weighted_sines(
        y,
        (
            (primary_wavelength * 0.91, 0.64, -0.47),
            (primary_wavelength * 0.46, 0.36, 1.61),
        ),
    )
    return shared + left_detail, shared + right_detail


def _rest_vertical_profile(
    component: int,
    u_fraction: float,
    v_fraction: float,
    geometry: dict[str, Any],
) -> tuple[float, float]:
    """Return local mid-surface elevation and positive local thickness."""

    depth = float(geometry["depth_m"])
    nominal_thickness = float(geometry["thickness_m"])
    topography = float(geometry["surface_topography_amplitude_m"])
    topography_wavelength = float(geometry["surface_topography_wavelength_m"])
    secondary_topography = float(
        geometry["surface_secondary_topography_amplitude_m"]
    )
    thickness_variation = float(
        geometry["thickness_variation_amplitude_m"]
    )
    y = -depth / 2.0 + depth * v_fraction
    woundward = u_fraction if component == 0 else 1.0 - u_fraction
    free_weight = math.sin(0.5 * math.pi * max(0.0, min(1.0, woundward)))
    component_phase = 0.37 if component == 0 else -0.29

    center_elevation = free_weight * (
        topography
        * _weighted_sines(
            y,
            (
                (topography_wavelength, 0.67, component_phase),
                (topography_wavelength * 0.48, 0.33, 1.12 - component_phase),
            ),
        )
        + secondary_topography
        * math.sin(
            2.0 * math.pi * (0.73 * woundward + 1.31 * v_fraction)
            + (0.54 if component == 0 else -0.71)
        )
    )
    local_thickness = nominal_thickness + free_weight * thickness_variation * (
        0.61
        * math.sin(
            2.0 * math.pi * y / (depth * 0.82)
            + (0.31 if component == 0 else -0.23)
        )
        + 0.39
        * math.sin(
            2.0 * math.pi * (0.57 * woundward - 0.88 * v_fraction)
            + (1.09 if component == 0 else -0.94)
        )
    )
    if local_thickness <= nominal_thickness * 0.75:
        raise ValueError("local tissue thickness fell below the safety bound")
    return center_elevation, local_thickness


def _point_distance_from_wound(
    point: Vec3,
    parametric: Vec4,
    contract: dict[str, Any],
) -> float:
    component, _, v_fraction, w_fraction = parametric
    geometry = contract["geometry"]
    depth = float(geometry["depth_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    y = -depth / 2.0 + depth * v_fraction
    left_offset, right_offset = _wound_lip_offsets(y, geometry)
    if int(component) == 0:
        inner_x = -gap / 2.0 + left_offset - bevel * w_fraction
    else:
        inner_x = gap / 2.0 + right_offset + bevel * w_fraction
    return abs(point[0] - inner_x)


def build_mesh(contract: dict[str, Any], lod: str) -> TissueMesh:
    geometry = contract["geometry"]
    lod_contract = contract["lods"][lod]
    width = float(geometry["overall_width_m"])
    depth = float(geometry["depth_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    cells_x = int(lod_contract["cells_per_flap_x"])
    cells_y = int(lod_contract["cells_y"])
    z_fractions = tuple(map(float, lod_contract["z_fractions"]))
    edge_power = float(lod_contract["wound_edge_refinement_power"])
    if len(z_fractions) < 2 or z_fractions[0] != 0.0 or z_fractions[-1] != 1.0:
        raise ValueError(f"{lod} z fractions must span exactly [0, 1]")
    if any(right <= left for left, right in zip(z_fractions, z_fractions[1:])):
        raise ValueError(f"{lod} z fractions must be strictly increasing")

    layer_boundaries = {
        float(value)
        for layer in contract["layers"]
        for value in layer["depth_fraction"]
    }
    if not layer_boundaries.issubset(set(z_fractions)):
        raise ValueError(f"{lod} does not conform to every material interface")

    points: list[Vec3] = []
    parametric_coordinates: list[Vec4] = []
    point_components: list[int] = []
    point_index: dict[tuple[int, int, int, int], int] = {}

    for component in range(2):
        for z_index, w_fraction in enumerate(z_fractions):
            for y_index in range(cells_y + 1):
                v_fraction = y_index / cells_y
                y = -depth / 2.0 + depth * v_fraction
                left_offset, right_offset = _wound_lip_offsets(y, geometry)
                if component == 0:
                    outer_x = -width / 2.0
                    inner_x = (
                        -gap / 2.0 + left_offset - bevel * w_fraction
                    )
                else:
                    inner_x = (
                        gap / 2.0 + right_offset + bevel * w_fraction
                    )
                    outer_x = width / 2.0
                for x_index in range(cells_x + 1):
                    u_fraction = x_index / cells_x
                    if component == 0:
                        shaped_u = 1.0 - (1.0 - u_fraction) ** edge_power
                        x = outer_x + (inner_x - outer_x) * shaped_u
                    else:
                        shaped_u = u_fraction**edge_power
                        x = inner_x + (outer_x - inner_x) * shaped_u
                    center_elevation, local_thickness = (
                        _rest_vertical_profile(
                            component,
                            u_fraction,
                            v_fraction,
                            geometry,
                        )
                    )
                    point_index[(component, x_index, y_index, z_index)] = len(points)
                    points.append(
                        (
                            x,
                            y,
                            center_elevation
                            + local_thickness * (w_fraction - 0.5),
                        )
                    )
                    parametric_coordinates.append(
                        (float(component), u_fraction, v_fraction, w_fraction)
                    )
                    point_components.append(component)

    def vertex(component: int, x: int, y: int, z: int) -> int:
        return point_index[(component, x, y, z)]

    cell_pattern = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    tetrahedra: list[Tet] = []
    tetrahedron_layers: list[int] = []
    tetrahedron_fibers: list[Vec3] = []
    minimum_volume = math.inf
    total_volume = 0.0
    for component in range(2):
        for z_index in range(len(z_fractions) - 1):
            mid_fraction = 0.5 * (
                z_fractions[z_index] + z_fractions[z_index + 1]
            )
            layer_index, layer_id = _layer_for_fraction(
                contract["layers"], mid_fraction
            )
            for y_index in range(cells_y):
                for x_index in range(cells_x):
                    corners = (
                        vertex(component, x_index, y_index, z_index),
                        vertex(component, x_index + 1, y_index, z_index),
                        vertex(component, x_index, y_index + 1, z_index),
                        vertex(component, x_index + 1, y_index + 1, z_index),
                        vertex(component, x_index, y_index, z_index + 1),
                        vertex(component, x_index + 1, y_index, z_index + 1),
                        vertex(component, x_index, y_index + 1, z_index + 1),
                        vertex(component, x_index + 1, y_index + 1, z_index + 1),
                    )
                    for local in cell_pattern:
                        tet = tuple(corners[index] for index in local)
                        volume = signed_tetra_volume(
                            *(points[index] for index in tet)
                        )
                        if volume < 0.0:
                            tet = (tet[1], tet[0], tet[2], tet[3])
                            volume = -volume
                        if volume <= 1.0e-16:
                            raise ValueError(
                                f"{lod} contains a degenerate tetrahedron"
                            )
                        tetrahedra.append(tet)
                        tetrahedron_layers.append(layer_index)
                        tetrahedron_fibers.append(
                            _fiber_for_layer(layer_id, component)
                        )
                        minimum_volume = min(minimum_volume, volume)
                        total_volume += volume

    surface = _surface_faces(points, tetrahedra, point_components)
    material_face_sets: dict[str, list[int]] = {
        "surface": [],
        "bulk": [],
        "fascia": [],
        "wound_surface": [],
        "wound_bulk": [],
        "wound_fascia": [],
    }
    semantic_face_sets: dict[str, list[int]] = {
        "safe_bite_surface": [],
        "unsafe_edge_surface": [],
        "contact_roi_surface": [],
        "wound_surface": [],
    }
    safe_min, safe_max = map(
        float, contract["semantics"]["safe_bite_distance_from_wound_m"]
    )
    unsafe_max = float(contract["semantics"]["unsafe_edge_distance_m"])
    contact_max = float(contract["semantics"]["contact_roi_distance_m"])
    y_margin = float(contract["semantics"]["longitudinal_end_margin_m"])

    for face_index, face in enumerate(surface):
        params = [parametric_coordinates[index] for index in face]
        all_top = all(abs(value[3] - 1.0) <= 1.0e-12 for value in params)
        all_bottom = all(abs(value[3]) <= 1.0e-12 for value in params)
        all_wound = all(
            (
                int(value[0]) == 0 and abs(value[1] - 1.0) <= 1.0e-12
            )
            or (
                int(value[0]) == 1 and abs(value[1]) <= 1.0e-12
            )
            for value in params
        )
        if all_top:
            material_face_sets["surface"].append(face_index)
        elif all_bottom:
            material_face_sets["fascia"].append(face_index)
        elif all_wound:
            average_depth = sum(value[3] for value in params) / 3.0
            _, wound_layer = _layer_for_fraction(
                contract["layers"],
                average_depth,
            )
            material_face_sets[f"wound_{wound_layer}"].append(face_index)
            semantic_face_sets["wound_surface"].append(face_index)
        else:
            material_face_sets["bulk"].append(face_index)

        if all_top:
            distances = [
                _point_distance_from_wound(
                    points[index], parametric_coordinates[index], contract
                )
                for index in face
            ]
            distance = sum(distances) / 3.0
            y = sum(points[index][1] for index in face) / 3.0
            if abs(y) <= depth / 2.0 - y_margin:
                if safe_min <= distance <= safe_max:
                    semantic_face_sets["safe_bite_surface"].append(face_index)
                if distance <= unsafe_max:
                    semantic_face_sets["unsafe_edge_surface"].append(face_index)
                if distance <= contact_max:
                    semantic_face_sets["contact_roi_surface"].append(face_index)

    node_sets: dict[str, list[int]] = {
        "anchor_outer": [],
        "left_flap": [],
        "right_flap": [],
        "top_surface": [],
        "bottom_surface": [],
        "wound_edge": [],
        "wound_edge_top": [],
        "safe_bite_top": [],
        "contact_roi_top": [],
    }
    attachment_width = float(contract["semantics"]["outer_attachment_width_m"])
    for index, (point, parametric) in enumerate(
        zip(points, parametric_coordinates, strict=True)
    ):
        component, u_fraction, _, w_fraction = parametric
        component_id = int(component)
        node_sets["left_flap" if component_id == 0 else "right_flap"].append(
            index
        )
        if abs(w_fraction - 1.0) <= 1.0e-12:
            node_sets["top_surface"].append(index)
        if abs(w_fraction) <= 1.0e-12:
            node_sets["bottom_surface"].append(index)
        wound_edge = (
            component_id == 0 and abs(u_fraction - 1.0) <= 1.0e-12
        ) or (component_id == 1 and abs(u_fraction) <= 1.0e-12)
        if wound_edge:
            node_sets["wound_edge"].append(index)
            if abs(w_fraction - 1.0) <= 1.0e-12:
                node_sets["wound_edge_top"].append(index)
        outer_distance = (
            point[0] + width / 2.0
            if component_id == 0
            else width / 2.0 - point[0]
        )
        if outer_distance <= attachment_width + 1.0e-12:
            node_sets["anchor_outer"].append(index)
        if abs(w_fraction - 1.0) <= 1.0e-12:
            distance = _point_distance_from_wound(point, parametric, contract)
            if abs(point[1]) <= depth / 2.0 - y_margin:
                if safe_min <= distance <= safe_max:
                    node_sets["safe_bite_top"].append(index)
                if distance <= contact_max:
                    node_sets["contact_roi_top"].append(index)

    layer_names = [str(layer["id"]) for layer in contract["layers"]]
    element_sets: dict[str, list[int]] = {
        f"layer_{name}": [] for name in layer_names
    }
    element_sets["safe_bite_volume"] = []
    element_sets["contact_roi_volume"] = []
    for index, tet in enumerate(tetrahedra):
        element_sets[f"layer_{layer_names[tetrahedron_layers[index]]}"].append(
            index
        )
        centroid = tuple(
            sum(points[vertex_index][axis] for vertex_index in tet) / 4.0
            for axis in range(3)
        )
        parametric = tuple(
            sum(parametric_coordinates[vertex_index][axis] for vertex_index in tet)
            / 4.0
            for axis in range(4)
        )
        distance = _point_distance_from_wound(
            centroid, parametric, contract
        )
        if abs(centroid[1]) <= depth / 2.0 - y_margin:
            if safe_min <= distance <= safe_max:
                element_sets["safe_bite_volume"].append(index)
            if distance <= contact_max:
                element_sets["contact_roi_volume"].append(index)

    for name, values in {
        **material_face_sets,
        **semantic_face_sets,
        **node_sets,
        **element_sets,
    }.items():
        if name in {"unsafe_edge_surface"}:
            continue
        if not values:
            raise ValueError(f"{lod} semantic set {name!r} is empty")

    return TissueMesh(
        lod=lod,
        points=tuple(points),
        parametric_coordinates=tuple(parametric_coordinates),
        point_components=tuple(point_components),
        tetrahedra=tuple(tetrahedra),
        tetrahedron_layers=tuple(tetrahedron_layers),
        tetrahedron_fibers=tuple(tetrahedron_fibers),
        surface_triangles=tuple(surface),
        material_face_sets={
            name: tuple(values) for name, values in material_face_sets.items()
        },
        semantic_face_sets={
            name: tuple(values) for name, values in semantic_face_sets.items()
        },
        node_sets={name: tuple(values) for name, values in node_sets.items()},
        element_sets={
            name: tuple(values) for name, values in element_sets.items()
        },
        extent_min=tuple(min(point[axis] for point in points) for axis in range(3)),
        extent_max=tuple(max(point[axis] for point in points) for axis in range(3)),
        volume_m3=total_volume,
        minimum_tetra_volume_m3=minimum_volume,
    )


def usd_float(value: float) -> str:
    return f"{value:.12g}"


def usd_vec(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(usd_float(value) for value in values) + ")"


def _encoded_scalars(values: tuple[int, ...] | list[int]) -> str:
    return ", ".join(str(value) for value in values)


def _encoded_vectors(values: tuple[tuple[float, ...], ...]) -> str:
    return ",\n            ".join(usd_vec(value) for value in values)


def _usd_identifier(identifier: str) -> str:
    return "".join(part.title() for part in identifier.split("_"))


def _normalized(vector: Vec3) -> Vec3:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1.0e-30:
        return (0.0, 0.0, 1.0)
    return tuple(value / length for value in vector)


def smooth_face_varying_normals(mesh: TissueMesh) -> tuple[Vec3, ...]:
    """Build area-weighted smooth normals without smoothing material seams."""

    face_material: dict[int, str] = {}
    for material, face_indices in mesh.material_face_sets.items():
        for face_index in face_indices:
            if face_index in face_material:
                raise ValueError("surface face belongs to multiple materials")
            face_material[face_index] = material
    if len(face_material) != len(mesh.surface_triangles):
        raise ValueError("surface material partition is incomplete")

    accumulated: dict[tuple[int, str], list[float]] = {}
    for face_index, triangle in enumerate(mesh.surface_triangles):
        material = face_material[face_index]
        a, b, c = (mesh.points[index] for index in triangle)
        face_normal = _triangle_normal(a, b, c)
        for point_index in triangle:
            values = accumulated.setdefault(
                (point_index, material),
                [0.0, 0.0, 0.0],
            )
            for axis in range(3):
                values[axis] += face_normal[axis]

    return tuple(
        _normalized(tuple(accumulated[(point_index, face_material[face_index])]))
        for face_index, triangle in enumerate(mesh.surface_triangles)
        for point_index in triangle
    )


def _materials_block(contract: dict[str, Any]) -> str:
    appearance = contract["appearance"]
    blocks: list[str] = []
    for identifier, values in appearance["materials"].items():
        color = tuple(map(float, values["diffuse_color"]))
        roughness = float(values["roughness"])
        name = _usd_identifier(identifier)
        blocks.append(
            f'''        def Material "{name}"
        {{
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {usd_vec(color)}
                float inputs:metallic = 0
                float inputs:roughness = {usd_float(roughness)}
                token outputs:surface
            }}
            token outputs:surface.connect = </{ROOT_PRIM}/Materials/{name}/PreviewSurface.outputs:surface>
        }}'''
        )
    return '    def Scope "Materials"\n    {\n' + "\n\n".join(blocks) + "\n    }"


def _material_subsets(mesh: TissueMesh) -> str:
    blocks = []
    for group, indices in mesh.material_face_sets.items():
        name = _usd_identifier(group)
        blocks.append(
            f'''        def GeomSubset "{name}Faces" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [{_encoded_scalars(indices)}]
            rel material:binding = </{ROOT_PRIM}/Materials/{name}>
        }}'''
        )
    return "\n\n".join(blocks)


def _custom_integer_arrays(
    namespace: str, groups: dict[str, tuple[int, ...]], indentation: int
) -> str:
    prefix = " " * indentation
    return "\n".join(
        f"{prefix}custom int[] drAnmar:{namespace}:{name} = "
        f"[{_encoded_scalars(values)}]"
        for name, values in groups.items()
    )


def _interaction_frame_positions(
    contract: dict[str, Any],
) -> tuple[Vec3, Vec3, Vec3]:
    """Place curriculum seed frames on the authored non-planar rest surface."""

    geometry = contract["geometry"]
    width = float(geometry["overall_width_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    left_offset, right_offset = _wound_lip_offsets(0.0, geometry)
    left_inner = -gap / 2.0 + left_offset - bevel
    right_inner = gap / 2.0 + right_offset + bevel
    safe_distance = sum(
        map(float, contract["semantics"]["safe_bite_distance_from_wound_m"])
    ) / 2.0
    edge_power = float(contract["lods"]["contact"]["wound_edge_refinement_power"])

    left_target_x = left_inner - safe_distance
    left_shaped = (left_target_x + width / 2.0) / (
        left_inner + width / 2.0
    )
    left_u = 1.0 - max(0.0, 1.0 - left_shaped) ** (1.0 / edge_power)
    right_target_x = right_inner + safe_distance
    right_shaped = (right_target_x - right_inner) / (
        width / 2.0 - right_inner
    )
    right_u = max(0.0, right_shaped) ** (1.0 / edge_power)

    left_center, left_thickness = _rest_vertical_profile(
        0,
        left_u,
        0.5,
        geometry,
    )
    right_center, right_thickness = _rest_vertical_profile(
        1,
        right_u,
        0.5,
        geometry,
    )
    left_lip_center, left_lip_thickness = _rest_vertical_profile(
        0,
        1.0,
        0.5,
        geometry,
    )
    right_lip_center, right_lip_thickness = _rest_vertical_profile(
        1,
        0.0,
        0.5,
        geometry,
    )
    wound_center = (
        0.5 * (left_inner + right_inner),
        0.0,
        0.5
        * (
            left_lip_center
            + 0.5 * left_lip_thickness
            + right_lip_center
            + 0.5 * right_lip_thickness
        ),
    )
    return (
        wound_center,
        (left_target_x, 0.0, left_center + 0.5 * left_thickness),
        (right_target_x, 0.0, right_center + 0.5 * right_thickness),
    )


def author_lod(contract: dict[str, Any], mesh: TissueMesh) -> str:
    points = _encoded_vectors(mesh.points)
    parametric = _encoded_vectors(mesh.parametric_coordinates)
    tets = ",\n            ".join(
        "(" + ", ".join(str(index) for index in tet) + ")"
        for tet in mesh.tetrahedra
    )
    surface = ",\n            ".join(
        "(" + ", ".join(str(index) for index in face) + ")"
        for face in mesh.surface_triangles
    )
    surface_counts = ", ".join("3" for _ in mesh.surface_triangles)
    surface_indices = ", ".join(
        str(index) for face in mesh.surface_triangles for index in face
    )
    normals = _encoded_vectors(smooth_face_varying_normals(mesh))
    fibers = _encoded_vectors(mesh.tetrahedron_fibers)
    wound_center, left_safe, right_safe = _interaction_frame_positions(
        contract
    )
    custom_data = f'''customData = {{
        string drAnmarAssetId = "{ASSET_ID}"
        string drAnmarAssetName = "{ASSET_NAME}"
        string drAnmarAssetVersion = "{ASSET_VERSION}"
        bool drAnmarClinicalValidation = false
        int drAnmarConnectedComponents = 2
        string drAnmarGeometryLod = "{mesh.lod}"
        string drAnmarRepresentation = "nested_layer_conforming_openusd_tetmesh"
        string drAnmarStatus = "{contract["status"]}"
        int drAnmarSurfaceTriangleCount = {len(mesh.surface_triangles)}
        int drAnmarTetrahedronCount = {len(mesh.tetrahedra)}
        int drAnmarVertexCount = {len(mesh.points)}
    }}'''
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "{ASSET_NAME} {mesh.lod} LOD; research geometry, not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{ROOT_PRIM}" (
    {custom_data}
)
{{
{_materials_block(contract)}

    def Scope "Frames"
    {{
        def Xform "TissueCenter"
        {{
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "WoundCenter"
        {{
            double3 xformOp:translate = {usd_vec(wound_center)}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "LeftSafeBiteSeed"
        {{
            double3 xformOp:translate = {usd_vec(left_safe)}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "RightSafeBiteSeed"
        {{
            double3 xformOp:translate = {usd_vec(right_safe)}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

    def TetMesh "Simulation"
    {{
        float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
        point3f[] points = [
            {points}
        ]
        int3[] surfaceFaceVertexIndices = [
            {surface}
        ]
        int4[] tetVertexIndices = [
            {tets}
        ]
        custom int[] drAnmar:pointComponentIds = [{_encoded_scalars(mesh.point_components)}]
        custom float4[] drAnmar:parametricCoordinates = [
            {parametric}
        ]
        custom int[] drAnmar:tetLayerIds = [{_encoded_scalars(mesh.tetrahedron_layers)}]
        custom vector3f[] drAnmar:tetFiberDirections = [
            {fibers}
        ]
{_custom_integer_arrays("nodeSet", mesh.node_sets, 8)}
{_custom_integer_arrays("elementSet", mesh.element_sets, 8)}
    }}

    def Mesh "Visual" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        float3[] extent = [{usd_vec(mesh.extent_min)}, {usd_vec(mesh.extent_max)}]
        int[] faceVertexCounts = [{surface_counts}]
        int[] faceVertexIndices = [{surface_indices}]
        point3f[] points = [
            {points}
        ]
        normal3f[] normals = [
            {normals}
        ] (
            interpolation = "faceVarying"
        )
        custom string drAnmar:normalContract = "area_weighted_smooth_with_material_seams"
        uniform token subdivisionScheme = "none"
        uniform token orientation = "rightHanded"
        uniform token subsetFamily:materialBind:familyType = "partition"
{_custom_integer_arrays("faceSet", mesh.semantic_face_sets, 8)}

{_material_subsets(mesh)}
    }}
}}
'''


def author_variant_root(contract: dict[str, Any]) -> str:
    variants = []
    for lod in contract["lods"]:
        variants.append(
            f'''        "{lod}" {{
            over "Geometry" (
                prepend references = @./needle_ready_tissue_{lod}.usda@</{ROOT_PRIM}>
            )
            {{
            }}
        }}'''
        )
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "{ASSET_NAME}: geometry-LOD composition root; not clinically validated."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{ROOT_PRIM}" (
    customData = {{
        string drAnmarAssetId = "{ASSET_ID}"
        string drAnmarAssetName = "{ASSET_NAME}"
        string drAnmarAssetVersion = "{ASSET_VERSION}"
        bool drAnmarClinicalValidation = false
        string drAnmarStatus = "{contract["status"]}"
    }}
    variants = {{
        string geometryLod = "contact"
    }}
    prepend variantSets = "geometryLod"
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


def _semantic_key(values: Vec4) -> tuple[int, int, int, int]:
    return tuple(round(float(value) * 1_000_000_000) for value in values)


def build_lod_mapping(meshes: dict[str, TissueMesh]) -> dict[str, Any]:
    mappings: dict[str, Any] = {}
    pairs = (
        ("training", "contact"),
        ("training", "validation"),
        ("contact", "validation"),
    )
    for source_name, target_name in pairs:
        source = meshes[source_name]
        target = meshes[target_name]
        target_index = {
            _semantic_key(values): index
            for index, values in enumerate(target.parametric_coordinates)
        }
        indices = []
        for values in source.parametric_coordinates:
            key = _semantic_key(values)
            if key not in target_index:
                raise ValueError(
                    f"{source_name} is not point-nested in {target_name}"
                )
            indices.append(target_index[key])
        mappings[f"{source_name}_to_{target_name}"] = {
            "source_point_count": len(source.points),
            "target_point_count": len(target.points),
            "target_indices": indices,
            "maximum_parametric_error": 0.0,
        }
    return {
        "schema": "dr.anmar.tissue-lod-mapping.v1",
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "coordinate_system": "(component,u,v,w)",
        "nested_exactly": True,
        "mappings": mappings,
    }


def build_report(
    contract: dict[str, Any],
    meshes: dict[str, TissueMesh],
    output_root: Path,
) -> dict[str, Any]:
    density = float(contract["physics"]["density_kg_m3_seed"])
    lod_reports = {}
    for lod, mesh in meshes.items():
        path = output_root / f"needle_ready_tissue_{lod}.usda"
        layer_names = [str(layer["id"]) for layer in contract["layers"]]
        quality = tetrahedron_quality(mesh)
        quality_gates = contract["lods"][lod]["quality_gates"]
        validate_tetrahedron_quality(lod, quality, quality_gates)
        points_by_semantic = {
            _semantic_key(values): point
            for point, values in zip(
                mesh.points,
                mesh.parametric_coordinates,
                strict=True,
            )
        }
        thicknesses = []
        wound_gaps = []
        for values in mesh.parametric_coordinates:
            component, u_fraction, v_fraction, w_fraction = values
            if abs(w_fraction) <= 1.0e-12:
                top_key = _semantic_key(
                    (component, u_fraction, v_fraction, 1.0)
                )
                thicknesses.append(
                    abs(
                        points_by_semantic[top_key][2]
                        - points_by_semantic[_semantic_key(values)][2]
                    )
                )
            is_left_wound = (
                int(component) == 0
                and abs(u_fraction - 1.0) <= 1.0e-12
            )
            if is_left_wound:
                right_key = _semantic_key(
                    (1.0, 0.0, v_fraction, w_fraction)
                )
                left_point = points_by_semantic[_semantic_key(values)]
                right_point = points_by_semantic[right_key]
                wound_gaps.append(right_point[0] - left_point[0])
        lod_reports[lod] = {
            "usd": path.name,
            "usd_sha256": sha256(path),
            "point_count": len(mesh.points),
            "tetrahedron_count": len(mesh.tetrahedra),
            "surface_triangle_count": len(mesh.surface_triangles),
            "connected_components": 2,
            "volume_m3": mesh.volume_m3,
            "mass_kg_seed": mesh.volume_m3 * density,
            "minimum_tetra_volume_m3": mesh.minimum_tetra_volume_m3,
            "local_thickness_range_m": [
                min(thicknesses),
                max(thicknesses),
            ],
            "rest_wound_gap_range_m": [
                min(wound_gaps),
                max(wound_gaps),
            ],
            "visual_schema_normal_count": len(
                smooth_face_varying_normals(mesh)
            ),
            "tetrahedron_quality": quality,
            "tetrahedron_quality_gates": quality_gates,
            "node_set_counts": {
                name: len(values) for name, values in mesh.node_sets.items()
            },
            "element_set_counts": {
                name: len(values) for name, values in mesh.element_sets.items()
            },
            "face_set_counts": {
                name: len(values)
                for name, values in mesh.semantic_face_sets.items()
            },
            "material_face_counts": {
                name: len(values)
                for name, values in mesh.material_face_sets.items()
            },
            "layer_tetrahedron_counts": {
                layer_name: mesh.tetrahedron_layers.count(index)
                for index, layer_name in enumerate(layer_names)
            },
        }
    return {
        "schema": "dr.anmar.needle-ready-tissue-geometry-report.v2",
        "asset_id": ASSET_ID,
        "asset_name": ASSET_NAME,
        "asset_version": ASSET_VERSION,
        "generator": {
            "path": "tools/generate_needle_ready_tissue.py",
            "sha256": sha256(Path(__file__)),
            "dependency_policy": "Python_standard_library_only",
            "deterministic": True,
        },
        "contract": "geometry_contract.json",
        "contract_sha256": sha256(output_root / "geometry_contract.json"),
        "root_usd": "needle_ready_tissue_unit.usda",
        "root_usd_sha256": sha256(output_root / "needle_ready_tissue_unit.usda"),
        "lods": lod_reports,
        "semantic_coordinate_system": "(component,u,v,w)",
        "lods_point_nested": True,
        "material_interfaces_conforming": True,
        "qualification_scope": "source_static_only",
        "stable_capabilities": contract["capabilities"]["stable"],
        "native_requalification_pending": contract["capabilities"][
            "native_requalification_pending"
        ],
        "gated_capabilities": contract["capabilities"]["gated"],
        "clinical_validation": False,
    }


def write_manifest(output_root: Path) -> dict[str, Any]:
    members = {}
    for path in sorted(output_root.iterdir()):
        if (
            not path.is_file()
            or path.name in {
                "asset_manifest.json",
                "visual_manifest.json",
                "needle_ready_tissue_visual_unit.usda",
            }
            or path.name.endswith("_visual.usda")
        ):
            continue
        members[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return {
        "schema": "dr.anmar.sim-ready-asset-manifest.v2",
        "asset_id": ASSET_ID,
        "asset_name": ASSET_NAME,
        "asset_version": ASSET_VERSION,
        "primary_usd": "needle_ready_tissue_unit.usda",
        "default_lod": "contact",
        "dependency_complete_directory": False,
        "manifest_scope": "base_physics_geometry_members_only",
        "nested_package_manifests": {
            "render_only_visual": "visual_manifest.json",
        },
        "generated_geometry": True,
        "generator": {
            "path": "tools/generate_needle_ready_tissue.py",
            "sha256": sha256(Path(__file__)),
            "dependency_policy": "Python_standard_library_only",
            "deterministic": True,
        },
        "members": members,
        "license": "Apache-2.0",
        "clinical_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()
    contract_path = args.contract.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    contract = load_json(contract_path)
    if contract["id"] != ASSET_ID or contract["version"] != ASSET_VERSION:
        raise ValueError("generator and geometry contract identity disagree")
    output_root.mkdir(parents=True, exist_ok=True)

    meshes = {
        lod: build_mesh(contract, lod)
        for lod in contract["lods"]
    }
    for lod, mesh in meshes.items():
        (output_root / f"needle_ready_tissue_{lod}.usda").write_text(
            author_lod(contract, mesh),
            encoding="utf-8",
        )
    (output_root / "needle_ready_tissue_unit.usda").write_text(
        author_variant_root(contract),
        encoding="utf-8",
    )
    (output_root / "lod_mapping.json").write_text(
        json.dumps(build_lod_mapping(meshes), indent=2) + "\n",
        encoding="utf-8",
    )
    report = build_report(contract, meshes, output_root)
    (output_root / "geometry_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = write_manifest(output_root)
    (output_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
