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
ASSET_VERSION = "2.0.0"
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
    amplitude = float(geometry["wound_irregularity_amplitude_m"])
    wavelength = float(geometry["wound_irregularity_wavelength_m"])
    y = -depth / 2.0 + depth * v_fraction
    center_offset = amplitude * math.sin(
        2.0 * math.pi * (y + depth / 2.0) / wavelength
    )
    if int(component) == 0:
        inner_x = -gap / 2.0 + center_offset - bevel * w_fraction
    else:
        inner_x = gap / 2.0 + center_offset + bevel * w_fraction
    return abs(point[0] - inner_x)


def build_mesh(contract: dict[str, Any], lod: str) -> TissueMesh:
    geometry = contract["geometry"]
    lod_contract = contract["lods"][lod]
    width = float(geometry["overall_width_m"])
    depth = float(geometry["depth_m"])
    thickness = float(geometry["thickness_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    irregularity = float(geometry["wound_irregularity_amplitude_m"])
    wavelength = float(geometry["wound_irregularity_wavelength_m"])
    topography = float(geometry["surface_topography_amplitude_m"])
    topography_wavelength = float(geometry["surface_topography_wavelength_m"])
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
            base_z = -thickness / 2.0 + thickness * w_fraction
            for y_index in range(cells_y + 1):
                v_fraction = y_index / cells_y
                y = -depth / 2.0 + depth * v_fraction
                wound_offset = irregularity * math.sin(
                    2.0 * math.pi * (y + depth / 2.0) / wavelength
                )
                if component == 0:
                    outer_x = -width / 2.0
                    inner_x = -gap / 2.0 + wound_offset - bevel * w_fraction
                else:
                    inner_x = gap / 2.0 + wound_offset + bevel * w_fraction
                    outer_x = width / 2.0
                for x_index in range(cells_x + 1):
                    u_fraction = x_index / cells_x
                    if component == 0:
                        shaped_u = 1.0 - (1.0 - u_fraction) ** edge_power
                        x = outer_x + (inner_x - outer_x) * shaped_u
                    else:
                        shaped_u = u_fraction**edge_power
                        x = inner_x + (outer_x - inner_x) * shaped_u
                    centrality = max(0.0, 1.0 - abs(x) / max(width / 2.0, 1.0e-9))
                    z_offset = (
                        topography
                        * centrality
                        * math.sin(
                            2.0
                            * math.pi
                            * (y + depth / 2.0)
                            / topography_wavelength
                            + (0.35 if component == 0 else -0.35)
                        )
                    )
                    point_index[(component, x_index, y_index, z_index)] = len(points)
                    points.append((x, y, base_z + z_offset))
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
        "wound": [],
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
            material_face_sets["wound"].append(face_index)
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


def _materials_block(contract: dict[str, Any]) -> str:
    appearance = contract["appearance"]
    blocks: list[str] = []
    for identifier, values in appearance["materials"].items():
        color = tuple(map(float, values["diffuse_color"]))
        roughness = float(values["roughness"])
        name = identifier.title()
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
        name = group.title()
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
    fibers = _encoded_vectors(mesh.tetrahedron_fibers)
    geometry = contract["geometry"]
    top_z = float(geometry["thickness_m"]) / 2.0
    safe_distance = sum(
        map(float, contract["semantics"]["safe_bite_distance_from_wound_m"])
    ) / 2.0
    half_gap = float(geometry["rest_wound_gap_m"]) / 2.0
    bevel = float(geometry["wound_bevel_m"])
    left_safe = (-half_gap - bevel - safe_distance, 0.0, top_z)
    right_safe = (half_gap + bevel + safe_distance, 0.0, top_z)
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
            double3 xformOp:translate = (0, 0, {usd_float(top_z)})
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
        "contract": "geometry_contract.json",
        "contract_sha256": sha256(output_root / "geometry_contract.json"),
        "root_usd": "needle_ready_tissue_unit.usda",
        "root_usd_sha256": sha256(output_root / "needle_ready_tissue_unit.usda"),
        "lods": lod_reports,
        "semantic_coordinate_system": "(component,u,v,w)",
        "lods_point_nested": True,
        "material_interfaces_conforming": True,
        "stable_capabilities": contract["capabilities"]["stable"],
        "gated_capabilities": contract["capabilities"]["gated"],
        "clinical_validation": False,
    }


def write_manifest(output_root: Path) -> dict[str, Any]:
    members = {}
    for path in sorted(output_root.iterdir()):
        if not path.is_file() or path.name == "asset_manifest.json":
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
        "dependency_complete_directory": True,
        "generated_geometry": True,
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
