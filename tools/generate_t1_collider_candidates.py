#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate inactive, source-qualified T1 collision-proxy candidates.

The packages produced here are deliberately not wired into any runtime
configuration.  They reference the existing T1 visual overlays and replace
only the collision geometry in stronger composition layers:

* PSM: 31 link-local analytic/convex components replace 11 enabled legacy
  colliders without changing articulation, joint, link, tool, or render data.
* Table: seven analytic/convex components replace the single legacy
  high-triangle collider while matching the supported table/pad envelope.

Run from the asset-extension root:

    python3 tools/generate_t1_collider_candidates.py
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PSM_SUBPATH = Path("data/Robots/dVRK/PSM/T1ColliderCandidate")
TABLE_SUBPATH = Path("data/Props/Table/T1ColliderCandidate")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT
GENERATOR_REPOSITORY_PATH = "tools/generate_t1_collider_candidates.py"
REPOSITORY_LICENSE = REPOSITORY_ROOT / "LICENSE"
REPOSITORY_LICENSE_SHA256 = (
    "9c69bf1f27dd03eeed8dd8aa2fa8a86702f6608b7f6397e8ea9d4c6a1d895c76"
)
REPOSITORY_LICENSE_BYTES = 1_613

STATUS = "inactive_unqualified_candidate"
VERSION = "1.0.0"
PSM_ASSET_ID = "dranmar-psm-t1-collider-candidate-v1"
TABLE_ASSET_ID = "dranmar-table-t1-collider-candidate-v1"
PSM_ENTRYPOINT = "psm_t1_collider_candidate.usda"
TABLE_ENTRYPOINT = "table_t1_collider_candidate.usda"

PSM_VISUAL_REFERENCE = (
    "../../../../Props/SurgicalScene/T1/psm_visual_v1.usda"
)
TABLE_VISUAL_REFERENCE = (
    "../../SurgicalScene/T1/table_visual_v1.usda"
)

LOCKED_DEPENDENCIES = {
    "psm_collision_foundation": {
        "bytes": 22_251_170,
        "default_prim": "/psm",
        "license": "BSD-3-Clause",
        "license_path": "LICENSE",
        "path": "data/Robots/dVRK/PSM/psm_col.usd",
        "sha256": (
            "87330b0e46e5554d53fe1b840f2db38a9bdbf00a79ca464159009406156ca0d4"
        ),
    },
    "psm_visual_overlay": {
        "bytes": 16_261,
        "default_prim": "/psm",
        "license": "Apache-2.0",
        "license_path": "data/Props/SurgicalScene/T1/LICENSE.txt",
        "path": "data/Props/SurgicalScene/T1/psm_visual_v1.usda",
        "sha256": (
            "534b8072d9e05a82df7707124c5a88715176fba1c58231aa09d8d8af062c2e84"
        ),
    },
    "table_collision_foundation": {
        "bytes": 18_839_808,
        "default_prim": "/Table",
        "license": "BSD-3-Clause",
        "license_path": "LICENSE",
        "path": "data/Props/Table/table.usd",
        "sha256": (
            "78f61603d0605896c9076d0a2b16bc97fd6f0109783a3b1e2dd1cfd86698b7c4"
        ),
    },
    "table_visual_overlay": {
        "bytes": 748_175,
        "default_prim": "/Table",
        "license": "Apache-2.0",
        "license_path": "data/Props/SurgicalScene/T1/LICENSE.txt",
        "path": "data/Props/SurgicalScene/T1/table_visual_v1.usda",
        "sha256": (
            "9d0eea19c3e01941e5119a2f2f6e8f3a32abaff34eec9548c7f2fb34bd50edfe"
        ),
    },
}

PSM_DISABLED_COLLIDERS = {
    "psm_remote_center_link": (
        ("visuals_xform", "visuals"),
    ),
    "psm_pitch_end_link": (
        ("visuals_xform", "visuals"),
    ),
    "psm_main_insertion_link": (
        ("visuals_xform", "visuals"),
    ),
    "psm_main_insertion_link_2": (
        ("visuals_xform", "tool_main_insert"),
    ),
    "psm_main_insertion_link_3": (
        ("visuals_xform", "visuals"),
    ),
    "psm_tool_roll_link": (
        ("visuals_xform", "visuals"),
        ("collisions_xform", "collisions"),
    ),
    "psm_tool_pitch_link": (
        ("collisions_xform", "collisions"),
    ),
    "psm_tool_yaw_link": (
        ("visuals_xform", "tool_yaw_link"),
    ),
    "psm_tool_gripper1_link": (
        ("collisions_xform", "collisions"),
    ),
    "psm_tool_gripper2_link": (
        ("collisions_xform", "collisions"),
    ),
}

PSM_TARGET_BOUNDS = {
    "psm_remote_center_link": (
        (-0.015, -0.015, -0.015),
        (0.015, 0.015, 0.015),
    ),
    "psm_pitch_end_link": (
        (-0.012199989, -0.036579925, -0.038754925),
        (0.067300002, 0.538675785, 0.038757566),
    ),
    "psm_main_insertion_link": (
        (0.01325, -0.041510999, -0.052016002),
        (0.0286, 0.041503001, 0.069995),
    ),
    "psm_main_insertion_link_2": (
        (-0.016508, -0.036690999, -0.057404),
        (0.018161001, 0.036690999, 0.051426001),
    ),
    "psm_main_insertion_link_3": (
        (-0.0057, -0.005693, 0.038699),
        (0.0057, 0.005707, 0.407001),
    ),
    "psm_tool_roll_link": (
        (-0.007938, -0.007934, -0.477021),
        (0.007938, 0.007934, 0.001900732),
    ),
    "psm_tool_pitch_link": (
        (-0.00302, -0.003200469, -0.00414),
        (0.011331, 0.0032, 0.004120113),
    ),
    "psm_tool_yaw_link": (
        (-0.000644955, -0.000649582, -0.002609),
        (0.000649956, 0.00064844, 0.002852),
    ),
    "psm_tool_gripper1_link": (
        (-0.002499, -0.002499, -0.001994),
        (0.002493, 0.0102, 0.001005),
    ),
    "psm_tool_gripper2_link": (
        (-0.002493, -0.002499, -0.001031),
        (0.002499, 0.0102, 0.001969),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def verify_locked_inputs() -> None:
    if REPOSITORY_LICENSE.stat().st_size != REPOSITORY_LICENSE_BYTES:
        raise RuntimeError("repository LICENSE byte count changed")
    if sha256(REPOSITORY_LICENSE) != REPOSITORY_LICENSE_SHA256:
        raise RuntimeError("repository LICENSE hash changed")
    for name, dependency in LOCKED_DEPENDENCIES.items():
        path = REPOSITORY_ROOT / dependency["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing locked dependency {name}: {path}")
        if path.stat().st_size != dependency["bytes"]:
            raise RuntimeError(f"{name} byte count changed; review before regeneration")
        if sha256(path) != dependency["sha256"]:
            raise RuntimeError(f"{name} hash changed; review before regeneration")


def cube(
    name: str,
    center: Sequence[float],
    size: Sequence[float],
    *,
    rationale: str,
) -> dict[str, Any]:
    return {
        "center": tuple(float(value) for value in center),
        "kind": "Cube",
        "name": name,
        "rationale": rationale,
        "size": tuple(float(value) for value in size),
        "triangles": 0,
    }


def cube_from_bounds(
    name: str,
    minimum: Sequence[float],
    maximum: Sequence[float],
    *,
    margin: float,
    rationale: str,
) -> dict[str, Any]:
    lower = tuple(float(value) - margin for value in minimum)
    upper = tuple(float(value) + margin for value in maximum)
    center = tuple((low + high) * 0.5 for low, high in zip(lower, upper))
    size = tuple(high - low for low, high in zip(lower, upper))
    return cube(name, center, size, rationale=rationale)


def sphere(
    name: str,
    center: Sequence[float],
    radius: float,
    *,
    rationale: str,
) -> dict[str, Any]:
    return {
        "center": tuple(float(value) for value in center),
        "kind": "Sphere",
        "name": name,
        "radius": float(radius),
        "rationale": rationale,
        "triangles": 0,
    }


def capsule(
    name: str,
    center: Sequence[float],
    *,
    axis: str,
    radius: float,
    total_length: float,
    rationale: str,
) -> dict[str, Any]:
    height = float(total_length) - 2.0 * float(radius)
    if height < 0.0:
        raise ValueError(f"capsule {name} total length is less than its diameter")
    return {
        "axis": axis,
        "center": tuple(float(value) for value in center),
        "height": height,
        "kind": "Capsule",
        "name": name,
        "radius": float(radius),
        "rationale": rationale,
        "total_length": float(total_length),
        "triangles": 0,
    }


def cylinder(
    name: str,
    center: Sequence[float],
    *,
    axis: str,
    radius: float,
    height: float,
    rationale: str,
) -> dict[str, Any]:
    return {
        "axis": axis,
        "center": tuple(float(value) for value in center),
        "height": float(height),
        "kind": "Cylinder",
        "name": name,
        "radius": float(radius),
        "rationale": rationale,
        "triangles": 0,
    }


FRUSTUM_FACES = (
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
)


def convex_frustum(
    name: str,
    *,
    y0: float,
    y1: float,
    xz0: tuple[float, float, float, float],
    xz1: tuple[float, float, float, float],
    rationale: str,
) -> dict[str, Any]:
    x0_min, x0_max, z0_min, z0_max = xz0
    x1_min, x1_max, z1_min, z1_max = xz1
    points = (
        (x0_min, y0, z0_min),
        (x0_max, y0, z0_min),
        (x0_max, y0, z0_max),
        (x0_min, y0, z0_max),
        (x1_min, y1, z1_min),
        (x1_max, y1, z1_min),
        (x1_max, y1, z1_max),
        (x1_min, y1, z1_max),
    )
    return {
        "faces": FRUSTUM_FACES,
        "kind": "Mesh",
        "name": name,
        "points": points,
        "rationale": rationale,
        "triangles": len(FRUSTUM_FACES),
    }


def chamfered_prism(
    name: str,
    *,
    half_x: float,
    half_y: float,
    chamfer: float,
    z_min: float,
    z_max: float,
    rationale: str,
) -> dict[str, Any]:
    profile = (
        (-half_x + chamfer, -half_y),
        (half_x - chamfer, -half_y),
        (half_x, -half_y + chamfer),
        (half_x, half_y - chamfer),
        (half_x - chamfer, half_y),
        (-half_x + chamfer, half_y),
        (-half_x, half_y - chamfer),
        (-half_x, -half_y + chamfer),
    )
    points = tuple((x, y, z_min) for x, y in profile) + tuple(
        (x, y, z_max) for x, y in profile
    )
    faces: list[tuple[int, int, int]] = []
    for index in range(1, 7):
        faces.append((0, index + 1, index))
        faces.append((8, 8 + index, 8 + index + 1))
    for index in range(8):
        next_index = (index + 1) % 8
        faces.append((index, next_index, 8 + next_index))
        faces.append((index, 8 + next_index, 8 + index))
    return {
        "faces": tuple(faces),
        "kind": "Mesh",
        "name": name,
        "points": points,
        "rationale": rationale,
        "triangles": len(faces),
    }


PSM_PROXIES = {
    "psm_remote_center_link": (
        sphere(
            "RemoteCenter",
            (0.0, 0.0, 0.0),
            0.015,
            rationale="exact analytic replacement for the locked spherical envelope",
        ),
    ),
    "psm_pitch_end_link": (
        cube_from_bounds(
            "RemoteRod",
            (-0.005949974, -0.036579925, -0.005281446),
            (0.004550022, -0.033079941, 0.005218552),
            margin=0.0002,
            rationale="isolated remote-center-side component; open span remains empty",
        ),
        cube_from_bounds(
            "PitchHousingA",
            (-0.011851501, 0.068717105, -0.030007038),
            (0.032200597, 0.103725288, 0.029992949),
            margin=0.0002,
            rationale="first occupied pitch-housing segment",
        ),
        cube_from_bounds(
            "PitchHousingB",
            (-0.012199989, 0.112713983, -0.038754925),
            (0.067300002, 0.154238142, 0.038751844),
            margin=0.0002,
            rationale="central pitch-housing segment",
        ),
        cube_from_bounds(
            "PitchHousingC",
            (0.033531338, 0.155377574, -0.038747936),
            (0.067299996, 0.186726807, 0.038757566),
            margin=0.0002,
            rationale="third occupied pitch-housing segment",
        ),
        cube_from_bounds(
            "PitchEndBlock",
            (0.028299971, 0.48481649, -0.03568235),
            (0.064299971, 0.538675785, 0.035817638),
            margin=0.0002,
            rationale="separate distal block; 298 mm empty mechanical span is preserved",
        ),
    ),
    "psm_main_insertion_link": tuple(
        cube_from_bounds(
            name,
            minimum,
            maximum,
            margin=0.00015,
            rationale="piecewise occupied insertion-housing slice",
        )
        for name, minimum, maximum in (
            (
                "InsertionA",
                (0.01325, -0.041510999, -0.052016002),
                (0.0286, 0.041489001, -0.032465001),
            ),
            (
                "InsertionB",
                (0.01325, -0.034108002, -0.030405999),
                (0.019750001, 0.034092002, -0.013146999),
            ),
            (
                "InsertionC",
                (0.01325, -0.034104999, -0.010654),
                (0.019750001, 0.034095, 0.007493),
            ),
            (
                "InsertionD",
                (0.01425, -0.041501001, 0.009897),
                (0.019750001, 0.041499, 0.025495),
            ),
            (
                "InsertionE",
                (0.01725, -0.041497, 0.029502),
                (0.0286, 0.041503001, 0.049035001),
            ),
            (
                "InsertionF",
                (0.01725, -0.032099999, 0.049717001),
                (0.019750001, 0.032117002, 0.069995),
            ),
        )
    ),
    "psm_main_insertion_link_2": tuple(
        cube_from_bounds(
            name,
            minimum,
            maximum,
            margin=0.00015,
            rationale="piecewise occupied main-insertion housing slice",
        )
        for name, minimum, maximum in (
            (
                "MainHousingA",
                (-0.015482, -0.032109, -0.057404),
                (0.018161001, 0.032090999, -0.039000001),
            ),
            (
                "MainHousingB",
                (-0.01575, -0.035305999, -0.038998999),
                (0.018161001, 0.035305999, -0.021002),
            ),
            (
                "MainHousingC",
                (-0.016461, -0.036690999, -0.021),
                (0.01725, 0.036690999, -0.003005),
            ),
            (
                "MainHousingD",
                (-0.016508, -0.036690999, -0.002998),
                (0.01725, 0.036690999, 0.014996),
            ),
            (
                "MainHousingE",
                (-0.015998, -0.032512002, 0.015001),
                (0.01725, 0.032512002, 0.033),
            ),
            (
                "MainHousingF",
                (-0.014725, -0.024907, 0.033000998),
                (0.01725, 0.024904, 0.051426001),
            ),
        )
    ),
    "psm_main_insertion_link_3": (
        capsule(
            "InsertionShaft",
            (0.0, 0.000007, 0.22285),
            axis="Z",
            radius=0.00571,
            total_length=0.368302,
            rationale="analytic shaft with locked endpoint and radial support bounds",
        ),
    ),
    "psm_tool_roll_link": (
        capsule(
            "RollShaft",
            (0.0, 0.0, -0.240928),
            axis="Z",
            radius=0.004178,
            total_length=0.472186,
            rationale="continuous slender roll shaft",
        ),
        capsule(
            "DistalCollar",
            (0.0, 0.0, -0.445017),
            axis="Z",
            radius=0.005565,
            total_length=0.021336,
            rationale="separate distal collar around the roll shaft",
        ),
        cylinder(
            "DistalFlange",
            (0.0, 0.0, -0.439886),
            axis="Z",
            radius=0.00794,
            height=0.003962,
            rationale="thin distal flange retained without widening the whole shaft",
        ),
        capsule(
            "RollHousing",
            (0.0, 0.0, -0.003499134),
            axis="Z",
            radius=0.00418,
            total_length=0.010799732,
            rationale="proximal roll housing",
        ),
    ),
    "psm_tool_pitch_link": tuple(
        cube_from_bounds(
            name,
            minimum,
            maximum,
            margin=0.0001,
            rationale="piecewise wrist housing segment",
        )
        for name, minimum, maximum in (
            (
                "PitchWristA",
                (-0.00302, -0.003113067, -0.00414),
                (0.0018, 0.003017, 0.004120113),
            ),
            (
                "PitchWristB",
                (0.001801, -0.003200469, -0.0032),
                (0.006599, 0.0032, 0.0032),
            ),
            (
                "PitchWristC",
                (0.006601, -0.003198379, -0.00296),
                (0.011331, 0.003098319, 0.00296),
            ),
        )
    ),
    "psm_tool_yaw_link": (
        capsule(
            "YawPin",
            (0.0000025, -0.000000571, 0.0001215),
            axis="Z",
            radius=0.00065,
            total_length=0.005461,
            rationale="analytic yaw pin at the locked render envelope",
        ),
    ),
    "psm_tool_gripper1_link": (
        convex_frustum(
            "JawProximal",
            y0=-0.002599,
            y1=0.0042,
            xz0=(-0.002599, 0.002593, -0.002094, 0.000087),
            xz1=(-0.0019, 0.00185, -0.002094, 0.001105),
            rationale="convex proximal jaw body; follows link-local taper",
        ),
        convex_frustum(
            "JawDistal",
            y0=0.0038,
            y1=0.0103,
            xz0=(-0.0019, 0.00185, -0.002094, 0.001105),
            xz1=(-0.000163, 0.0016, -0.001044, 0.001018),
            rationale="convex distal contact jaw; no attachment or adhesion",
        ),
    ),
    "psm_tool_gripper2_link": (
        convex_frustum(
            "JawProximal",
            y0=-0.002599,
            y1=0.0042,
            xz0=(-0.002593, 0.002599, -0.000113, 0.002068),
            xz1=(-0.00185, 0.0019, -0.001131, 0.002069),
            rationale="convex proximal jaw body; mirrored link-local taper",
        ),
        convex_frustum(
            "JawDistal",
            y0=0.0038,
            y1=0.0103,
            xz0=(-0.00185, 0.0019, -0.001131, 0.002069),
            xz1=(-0.0016, 0.000163, -0.001043, 0.001018),
            rationale="convex distal contact jaw; no attachment or adhesion",
        ),
    ),
}

TABLE_PROXIES = (
    cube_from_bounds(
        "DeckFrame",
        (-0.87, -0.38, 0.343),
        (0.87, 0.38, 0.393),
        margin=0.0,
        rationale="exact supported deck-frame AABB",
    ),
    chamfered_prism(
        "PatientPad",
        half_x=0.83,
        half_y=0.35,
        chamfer=0.04,
        z_min=0.3915,
        z_max=0.4565,
        rationale="eight-sided convex pad with exact render support bounds",
    ),
    cube_from_bounds(
        "SterileFieldTop",
        (-0.83, -0.35, 0.4565),
        (0.83, 0.35, 0.459881078),
        margin=0.0,
        rationale=(
            "thin supported drape contact surface; unsupported hanging fabric is "
            "intentionally not made rigid"
        ),
    ),
    cube_from_bounds(
        "CenterColumn",
        (-0.11, -0.16, -0.413),
        (0.11, 0.16, 0.343),
        margin=0.0,
        rationale="exact center-column render bounds",
    ),
    cube_from_bounds(
        "FloorBase",
        (-0.51, -0.3, -0.454),
        (0.51, 0.3, -0.396),
        margin=0.0,
        rationale="exact floor-base render bounds",
    ),
    cube_from_bounds(
        "LeftFoot",
        (-0.74, -0.08, -0.454),
        (-0.48, 0.08, -0.386),
        margin=0.0,
        rationale="separate left support foot",
    ),
    cube_from_bounds(
        "RightFoot",
        (0.48, -0.08, -0.454),
        (0.74, 0.08, -0.386),
        margin=0.0,
        rationale="separate right support foot",
    ),
)

TABLE_RENDER_BOUNDS = {
    "CenterColumn": ((-0.11, -0.16, -0.413), (0.11, 0.16, 0.343)),
    "DeckFrame": ((-0.87, -0.38, 0.343), (0.87, 0.38, 0.393)),
    "FloorBase": ((-0.51, -0.3, -0.454), (0.51, 0.3, -0.396)),
    "LeftFoot": ((-0.74, -0.08, -0.454), (-0.48, 0.08, -0.386)),
    "PatientPad": ((-0.83, -0.35, 0.3915), (0.83, 0.35, 0.4565)),
    "RightFoot": ((0.48, -0.08, -0.454), (0.74, 0.08, -0.386)),
    "SterileDrape": (
        (-0.89, -0.44, 0.360018317),
        (0.89, 0.44, 0.459881078),
    ),
}

PSM_GENERATED_MEMBERS = (
    PSM_ENTRYPOINT,
    "LICENSE.txt",
    "NOTICE.txt",
    "README.md",
    "approximation_report.json",
    "dependency_lock.json",
    "geometry_contract.json",
    "provenance.json",
    "qualification_contract.json",
)
TABLE_GENERATED_MEMBERS = (
    TABLE_ENTRYPOINT,
    "LICENSE.txt",
    "NOTICE.txt",
    "README.md",
    "approximation_report.json",
    "dependency_lock.json",
    "geometry_contract.json",
    "provenance.json",
    "qualification_contract.json",
)


def _format_float(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=5.0e-13):
        value = 0.0
    return f"{value:.9g}"


def _format_vec(values: Sequence[float]) -> str:
    return "(" + ", ".join(_format_float(float(value)) for value in values) + ")"


def _proxy_bounds(proxy: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    kind = proxy["kind"]
    if kind == "Cube":
        return tuple(
            center - size * 0.5
            for center, size in zip(proxy["center"], proxy["size"])
        ), tuple(
            center + size * 0.5
            for center, size in zip(proxy["center"], proxy["size"])
        )
    if kind == "Sphere":
        return tuple(
            center - proxy["radius"] for center in proxy["center"]
        ), tuple(center + proxy["radius"] for center in proxy["center"])
    if kind in {"Capsule", "Cylinder"}:
        half_axis = proxy["height"] * 0.5
        if kind == "Capsule":
            half_axis += proxy["radius"]
        half = [proxy["radius"], proxy["radius"], proxy["radius"]]
        half["XYZ".index(proxy["axis"])] = half_axis
        return tuple(
            center - extent for center, extent in zip(proxy["center"], half)
        ), tuple(
            center + extent for center, extent in zip(proxy["center"], half)
        )
    points = proxy["points"]
    return tuple(min(point[index] for point in points) for index in range(3)), tuple(
        max(point[index] for point in points) for index in range(3)
    )


def _union_bounds(
    proxies: Iterable[dict[str, Any]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    bounds = [_proxy_bounds(proxy) for proxy in proxies]
    return tuple(min(bound[0][index] for bound in bounds) for index in range(3)), tuple(
        max(bound[1][index] for bound in bounds) for index in range(3)
    )


def _author_common_properties(indent: str, proxy: dict[str, Any]) -> list[str]:
    return [
        f'{indent}custom bool dranmar:candidate:active = false',
        f'{indent}custom string dranmar:candidate:approximation = "{proxy["rationale"]}"',
        f'{indent}bool physics:collisionEnabled = true',
        f'{indent}token visibility = "invisible"',
    ]


def _author_proxy(proxy: dict[str, Any], level: int) -> str:
    indent = "    " * level
    inner = "    " * (level + 1)
    kind = proxy["kind"]
    api_schemas = '["PhysicsCollisionAPI"]'
    if kind == "Mesh":
        api_schemas = '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]'
    lines = [
        f'{indent}def {kind} "{proxy["name"]}" (',
        f"{inner}prepend apiSchemas = {api_schemas}",
        f"{indent})",
        f"{indent}{{",
    ]
    lines.extend(_author_common_properties(inner, proxy))
    if kind == "Cube":
        lower, upper = (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    elif kind == "Sphere":
        radius = proxy["radius"]
        lower, upper = (-radius, -radius, -radius), (
            radius,
            radius,
            radius,
        )
    elif kind in {"Capsule", "Cylinder"}:
        half_axis = proxy["height"] * 0.5
        if kind == "Capsule":
            half_axis += proxy["radius"]
        half = [proxy["radius"], proxy["radius"], proxy["radius"]]
        half["XYZ".index(proxy["axis"])] = half_axis
        lower, upper = tuple(-value for value in half), tuple(half)
    else:
        lower, upper = _proxy_bounds(proxy)
    lines.append(
        f"{inner}float3[] extent = [{_format_vec(lower)}, {_format_vec(upper)}]"
    )
    if kind == "Cube":
        lines.extend(
            (
                f"{inner}double size = 1",
                f"{inner}double3 xformOp:translate = {_format_vec(proxy['center'])}",
                f"{inner}double3 xformOp:scale = {_format_vec(proxy['size'])}",
                (
                    f'{inner}uniform token[] xformOpOrder = '
                    '["xformOp:translate", "xformOp:scale"]'
                ),
            )
        )
    elif kind == "Sphere":
        lines.extend(
            (
                f"{inner}double radius = {_format_float(proxy['radius'])}",
                f"{inner}double3 xformOp:translate = {_format_vec(proxy['center'])}",
                f'{inner}uniform token[] xformOpOrder = ["xformOp:translate"]',
            )
        )
    elif kind in {"Capsule", "Cylinder"}:
        lines.extend(
            (
                f'{inner}uniform token axis = "{proxy["axis"]}"',
                f"{inner}double height = {_format_float(proxy['height'])}",
                f"{inner}double radius = {_format_float(proxy['radius'])}",
                f"{inner}double3 xformOp:translate = {_format_vec(proxy['center'])}",
                f'{inner}uniform token[] xformOpOrder = ["xformOp:translate"]',
            )
        )
    else:
        points = ",\n".join(
            f"{inner}    {_format_vec(point)}" for point in proxy["points"]
        )
        counts = ", ".join("3" for _ in proxy["faces"])
        indices = ", ".join(
            str(index) for face in proxy["faces"] for index in face
        )
        lines.extend(
            (
                f'{inner}uniform token physics:approximation = "convexHull"',
                f"{inner}int[] faceVertexCounts = [{counts}]",
                f"{inner}int[] faceVertexIndices = [{indices}]",
                f"{inner}point3f[] points = [\n{points}\n{inner}]",
                f'{inner}uniform token subdivisionScheme = "none"',
            )
        )
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def _author_disable_path(parts: Sequence[str], level: int) -> str:
    indent = "    " * level
    name = parts[0]
    lines = [f'{indent}over "{name}"', f"{indent}{{"]
    if len(parts) == 1:
        lines.append(f"{indent}    bool physics:collisionEnabled = false")
    else:
        lines.append(_author_disable_path(parts[1:], level + 1))
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def psm_usda() -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "psm"',
        (
            '    doc = "Inactive DrAnmar T1 link-local PSM collider candidate; '
            'render and articulation remain externally composed."'
        ),
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "psm" (',
        f"    prepend references = @{PSM_VISUAL_REFERENCE}@</psm>",
        ")",
        "{",
        "    custom bool dranmar:candidate:active = false",
        f'    custom string dranmar:candidate:assetId = "{PSM_ASSET_ID}"',
        (
            '    custom string dranmar:candidate:qualificationContract = '
            '"./qualification_contract.json"'
        ),
        f'    custom string dranmar:candidate:status = "{STATUS}"',
        (
            '    custom string dranmar:candidate:runtimeQuaternionOrder = '
            '"xyzw"'
        ),
        '    custom string dranmar:candidate:usdQuaternionOrder = "wxyz"',
        f'    custom string dranmar:candidate:version = "{VERSION}"',
    ]
    for link, disabled_paths in PSM_DISABLED_COLLIDERS.items():
        lines.extend((f'    over "{link}"', "    {"))
        for relative_path in disabled_paths:
            lines.append(_author_disable_path(relative_path, 2))
        lines.extend(
            (
                '        def Scope "DrAnmarT1ColliderCandidate"',
                "        {",
                "            custom bool dranmar:candidate:active = false",
                (
                    '            custom string dranmar:candidate:geometryContract = '
                    '"../../geometry_contract.json"'
                ),
            )
        )
        for proxy in PSM_PROXIES[link]:
            lines.append(_author_proxy(proxy, 3))
        lines.extend(("        }", "    }"))
    lines.extend(("}", ""))
    return "\n".join(lines)


def table_usda() -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Table"',
        (
            '    doc = "Inactive DrAnmar T1 decomposed table collider candidate; '
            'render remains externally composed."'
        ),
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Table" (',
        f"    prepend references = @{TABLE_VISUAL_REFERENCE}@</Table>",
        ")",
        "{",
        "    custom bool dranmar:candidate:active = false",
        f'    custom string dranmar:candidate:assetId = "{TABLE_ASSET_ID}"',
        (
            '    custom string dranmar:candidate:qualificationContract = '
            '"./qualification_contract.json"'
        ),
        f'    custom string dranmar:candidate:status = "{STATUS}"',
        (
            '    custom string dranmar:candidate:runtimeQuaternionOrder = '
            '"xyzw"'
        ),
        '    custom string dranmar:candidate:usdQuaternionOrder = "wxyz"',
        f'    custom string dranmar:candidate:version = "{VERSION}"',
        '    over "Table"',
        "    {",
        '        over "Table"',
        "        {",
        "            bool physics:collisionEnabled = false",
        "        }",
        '        def Scope "DrAnmarT1ColliderCandidate"',
        "        {",
        "            custom bool dranmar:candidate:active = false",
        (
            '            custom string dranmar:candidate:geometryContract = '
            '"../../geometry_contract.json"'
        ),
    ]
    for proxy in TABLE_PROXIES:
        lines.append(_author_proxy(proxy, 3))
    lines.extend(("        }", "    }", "}", ""))
    return "\n".join(lines)


def runtime_contract(root_prim: str) -> dict[str, Any]:
    return {
        "allowed_runtime_scale": [1.0, 1.0, 1.0],
        "identity_quaternion_runtime_xyzw": [0.0, 0.0, 0.0, 1.0],
        "identity_quaternion_usd_wxyz": [1.0, 0.0, 0.0, 0.0],
        "meters_per_unit": 1.0,
        "non_unit_scale_invalidates_qualification": True,
        "root_prim": root_prim,
        "root_transform_authored_by_candidate": False,
        "root_transform_inherited_exactly": True,
        "runtime_quaternion_order": "xyzw",
        "stage_up_axis": "Z",
        "usd_quaternion_order": "wxyz",
    }


def dependency_lock(kind: str) -> dict[str, Any]:
    names = (
        ("psm_collision_foundation", "psm_visual_overlay")
        if kind == "psm"
        else ("table_collision_foundation", "table_visual_overlay")
    )
    return {
        "asset_id": PSM_ASSET_ID if kind == "psm" else TABLE_ASSET_ID,
        "asset_version": VERSION,
        "dependency_complete": True,
        "dependencies": {
            name: {
                **LOCKED_DEPENDENCIES[name],
                "modified_by_candidate": False,
            }
            for name in names
        },
        "direct_composition_dependency": names[1],
        "resolution_scope": (
            "All direct and transitive repository foundations are locked by "
            "exact byte count and SHA-256. No network or runtime asset lookup "
            "is permitted by this candidate."
        ),
        "status": STATUS,
    }


def psm_approximation_report() -> dict[str, Any]:
    links: dict[str, Any] = {}
    maximum_error = 0.0
    for link, proxies in PSM_PROXIES.items():
        target_min, target_max = PSM_TARGET_BOUNDS[link]
        proxy_min, proxy_max = _union_bounds(proxies)
        lower_delta = [
            proxy_min[index] - target_min[index] for index in range(3)
        ]
        upper_delta = [
            proxy_max[index] - target_max[index] for index in range(3)
        ]
        support_error = max(
            abs(value) for value in (*lower_delta, *upper_delta)
        )
        maximum_error = max(maximum_error, support_error)
        links[link] = {
            "component_count": len(proxies),
            "maximum_absolute_aabb_support_error_m": support_error,
            "proxy_aabb_m": {"max": proxy_max, "min": proxy_min},
            "proxy_components": [
                {
                    "kind": proxy["kind"],
                    "name": proxy["name"],
                    "rationale": proxy["rationale"],
                    "triangles": proxy["triangles"],
                }
                for proxy in proxies
            ],
            "proxy_minus_target_max_m": upper_delta,
            "proxy_minus_target_min_m": lower_delta,
            "render_target_aabb_m": {"max": target_max, "min": target_min},
            "target_measurement": (
                "locked render geometry transformed into owning link coordinates"
            ),
        }
    return {
        "asset_id": PSM_ASSET_ID,
        "asset_version": VERSION,
        "broad_single_shell_used": False,
        "candidate_mesh_triangles": sum(
            proxy["triangles"]
            for proxies in PSM_PROXIES.values()
            for proxy in proxies
        ),
        "candidate_primitive_count": sum(
            len(proxies) for proxies in PSM_PROXIES.values()
        ),
        "legacy_enabled_collision_mesh_triangles": 288_052,
        "legacy_enabled_collision_primitive_count": 11,
        "links": links,
        "maximum_absolute_aabb_support_error_m": maximum_error,
        "measurement_method": (
            "OpenUSD 25.11 default-time render vertices transformed into each "
            "owning rigid-link frame; analytic support bounds evaluated exactly"
        ),
        "limitations": [
            (
                "AABB support agreement is a source-level geometry check, not "
                "contact, penetration, inertia, torque, or stability qualification."
            ),
            (
                "Piecewise boxes conservatively cover occupied render slices; "
                "they do not claim surface-scanned instrument geometry."
            ),
            (
                "Jaw frusta are low-poly contact-envelope hypotheses and add no "
                "serration, adhesion, suction, attachment, or force injection."
            ),
        ],
        "status": STATUS,
        "triangle_reduction_fraction": 1.0 - (48.0 / 288_052.0),
    }


def table_approximation_report() -> dict[str, Any]:
    by_name = {proxy["name"]: proxy for proxy in TABLE_PROXIES}
    components: dict[str, Any] = {}
    for name in (
        "DeckFrame",
        "PatientPad",
        "CenterColumn",
        "FloorBase",
        "LeftFoot",
        "RightFoot",
    ):
        proxy_min, proxy_max = _proxy_bounds(by_name[name])
        target_min, target_max = TABLE_RENDER_BOUNDS[name]
        components[name] = {
            "maximum_absolute_aabb_support_error_m": max(
                abs(proxy_min[index] - target_min[index])
                for index in range(3)
            )
            if proxy_max == target_max
            else max(
                *(
                    abs(proxy_min[index] - target_min[index])
                    for index in range(3)
                ),
                *(
                    abs(proxy_max[index] - target_max[index])
                    for index in range(3)
                ),
            ),
            "proxy_aabb_m": {"max": proxy_max, "min": proxy_min},
            "render_target_aabb_m": {"max": target_max, "min": target_min},
        }
    sterile_min, sterile_max = _proxy_bounds(by_name["SterileFieldTop"])
    components["SterileFieldTop"] = {
        "proxy_aabb_m": {"max": sterile_max, "min": sterile_min},
        "render_target_aabb_m": {
            "max": TABLE_RENDER_BOUNDS["SterileDrape"][1],
            "min": TABLE_RENDER_BOUNDS["SterileDrape"][0],
        },
        "supported_interaction_footprint_m": {
            "x": [-0.83, 0.83],
            "y": [-0.35, 0.35],
            "z": [0.4565, 0.459881078],
        },
        "unsupported_hanging_drape_is_rigid": False,
        "vertical_top_support_error_m": 0.0,
    }
    return {
        "asset_id": TABLE_ASSET_ID,
        "asset_version": VERSION,
        "broad_single_shell_used": False,
        "candidate_mesh_triangles": 28,
        "candidate_primitive_count": 7,
        "components": components,
        "legacy_collision_mesh_triangles": 127_622,
        "limitations": [
            (
                "The thin drape proxy covers only the pad-supported sterile field; "
                "hanging fabric is intentionally excluded from rigid collision."
            ),
            (
                "Static envelope agreement is not PhysX contact, load, friction, "
                "or table-calibration evidence."
            ),
        ],
        "measurement_method": (
            "OpenUSD 25.11 authored render extents from the exact locked T1 "
            "table visual overlay"
        ),
        "status": STATUS,
        "triangle_reduction_fraction": 1.0 - (28.0 / 127_622.0),
    }


def geometry_contract(kind: str) -> dict[str, Any]:
    is_psm = kind == "psm"
    return {
        "asset_id": PSM_ASSET_ID if is_psm else TABLE_ASSET_ID,
        "asset_version": VERSION,
        "activation": {
            "active_replacement": False,
            "default": "blocked",
            "requires_explicit_human_review": True,
        },
        "allowed_layer_authority": (
            [
                "candidate namespaced metadata on /psm",
                (
                    "physics:collisionEnabled=false on the 11 exact locked "
                    "legacy enabled collider prims"
                ),
                (
                    "new collision-only prims below each owning rigid link's "
                    "DrAnmarT1ColliderCandidate scope"
                ),
            ]
            if is_psm
            else [
                "candidate namespaced metadata on /Table",
                (
                    "physics:collisionEnabled=false on the exact locked legacy "
                    "mesh /Table/Table/Table"
                ),
                (
                    "new collision-only prims below "
                    "/Table/Table/DrAnmarT1ColliderCandidate"
                ),
            ]
        ),
        "forbidden_authority": [
            "articulation_or_joint_properties",
            "existing_link_or_tool_transforms",
            "root_transform",
            "render_geometry_or_visibility",
            "material_or_friction_values",
            "mass_center_of_mass_or_inertia",
            "adhesion_suction_attachment_or_force_injection",
        ],
        "primitive_budget": {
            "maximum_mesh_triangles": 48 if is_psm else 28,
            "maximum_primitives": 31 if is_psm else 7,
            "observed_mesh_triangles": 48 if is_psm else 28,
            "observed_primitives": 31 if is_psm else 7,
        },
        "proxy_ownership": (
            "link_local_one_scope_per_owning_rigid_link"
            if is_psm
            else "decomposed_components_below_existing_table_rigid_body"
        ),
        "runtime_contract": runtime_contract("/psm" if is_psm else "/Table"),
        "status": STATUS,
    }


def qualification_contract(kind: str) -> dict[str, Any]:
    is_psm = kind == "psm"
    required_suites = (
        {
            "articulation_and_joint_parity": [
                "body_count_and_names",
                "joint_count_names_types_limits_and_axes",
                "default_link_and_tool_transforms",
                "mass_center_of_mass_and_inertia",
            ],
            "contact_stability": [
                "penetration_depth_distribution_m",
                "contact_impulse_distribution_ns",
                "solver_divergence_and_nan_count",
                "high_velocity_tunneling_count",
            ],
            "task_noninferiority": [
                "analytic_pickup_first_attempt_success",
                "conditional_mid_air_retention_given_pickup",
                "transport_completion",
                "receiver_retention_after_handover",
                "drop_and_regrasp_rates",
            ],
            "wrist_and_jaw_contact": [
                "bilateral_jaw_contact_duty_cycle",
                "per_jaw_normal_force_n",
                "needle_relative_slip_distance_m",
                "false_contact_and_self_contact_count",
            ],
        }
        if is_psm
        else {
            "composition_and_spawn": [
                "root_transform_and_fixed_joint_parity",
                "native_isaac_stage_spawn",
                "collision_component_discovery",
            ],
            "contact_stability": [
                "resting_penetration_depth_m",
                "contact_impulse_distribution_ns",
                "solver_divergence_and_nan_count",
                "high_velocity_tunneling_count",
            ],
            "supported_surface": [
                "object_rest_height_error_m",
                "tissue_support_height_error_m",
                "edge_false_contact_count",
                "unsupported_drape_phantom_contact_count",
            ],
        }
    )
    return {
        "activation": {
            "active_replacement": False,
            "allowed_after_all_required_suites_pass": True,
            "decision_owner": "explicit_human_review",
            "default": "blocked",
            "requires_native_isaac_physx_evidence": True,
        },
        "asset_id": PSM_ASSET_ID if is_psm else TABLE_ASSET_ID,
        "asset_version": VERSION,
        "evidence_integrity": {
            "archive": [
                "resolved_base_and_candidate_hashes",
                "runtime_configuration_hash",
                "seed_lists",
                "per_episode_metrics",
                "contact_telemetry",
                "video_linked_failure_taxonomy",
            ],
            "paired_baseline_and_candidate_seeds": True,
            "pre_registered_thresholds": True,
            "reward_contract_must_remain_identical": True,
        },
        "required_suites": required_suites,
        "status": STATUS,
        "validation_boundaries": {
            "clinical_validation": False,
            "instrument_calibration": False,
            "native_isaac_spawn_qualified": False,
            "physics_calibration": False,
            "source_static_generation_is_qualification": False,
        },
    }


def provenance(kind: str) -> dict[str, Any]:
    names = (
        ("psm_collision_foundation", "psm_visual_overlay")
        if kind == "psm"
        else ("table_collision_foundation", "table_visual_overlay")
    )
    return {
        "asset_id": PSM_ASSET_ID if kind == "psm" else TABLE_ASSET_ID,
        "asset_version": VERSION,
        "authorship": (
            "Collision proxy geometry, composition, contracts, generator, and "
            "tests were independently authored by Dr.Anmar Project Developers."
        ),
        "external_downloads": [],
        "generator": GENERATOR_REPOSITORY_PATH,
        "locked_measurement_sources": {
            name: {
                "path": LOCKED_DEPENDENCIES[name]["path"],
                "sha256": LOCKED_DEPENDENCIES[name]["sha256"],
            }
            for name in names
        },
        "patient_or_clinical_data": False,
        "provider": "Dr.Anmar Project Developers",
        "source_method": (
            "Deterministic analytic primitives and low-poly convex envelopes "
            "derived from exact, hash-locked OpenUSD render bounds."
        ),
        "third_party_geometry_or_code_added": False,
        "validation_boundary": (
            "Source-static package only; not native-physics qualified, "
            "physics-calibrated, instrument-calibrated, or clinically validated."
        ),
    }


def readme(kind: str) -> str:
    if kind == "psm":
        return """# T1 PSM collider candidate

This is a deterministic, **inactive and unqualified** collision overlay for the
T1 PSM. It composes the exact hash-locked high-realism PSM visual overlay and
leaves its render meshes, articulation, joints, tools, link transforms, mass,
center of mass, inertia, and material values unchanged.

The stronger layer disables exactly 11 legacy enabled colliders and adds 31
link-local components: 27 analytic primitives and four closed convex jaw
frusta totaling 48 triangles. No broad robot-wide shell is used. The
298 mm unoccupied pitch-link span remains empty, the slender roll shaft is not
inflated to its flange radius, and each jaw remains owned by its moving link.

`approximation_report.json` records measured link-local bounds and limitations.
`geometry_contract.json` fixes meters, root, runtime `xyzw`, USD `wxyz`, unit
scale, authority, and budgets. `qualification_contract.json` blocks activation
until paired native Isaac/PhysX evidence passes.

No runtime configuration references this package. Static generation does not
establish contact stability, task noninferiority, instrument calibration,
physics calibration, or clinical validity.
"""
    return """# T1 table collider candidate

This is a deterministic, **inactive and unqualified** collision overlay for the
T1 operating table. It composes the exact hash-locked high-realism table visual
overlay without authoring render geometry, render visibility, materials, the
root transform, fixed joint, rigid-body properties, mass, or inertia.

The stronger layer disables the single 127,622-triangle legacy collider and
adds seven decomposed components: deck, chamfered patient pad, a thin supported
sterile-field top, center column, floor base, and separate feet. Only the drape
area supported by the pad is rigid; hanging fabric deliberately remains
non-rigid to avoid phantom contact.

`approximation_report.json` records the exact support envelopes and limits.
`geometry_contract.json` fixes meters, root, runtime `xyzw`, USD `wxyz`, unit
scale, authority, and budgets. `qualification_contract.json` blocks activation
until paired native Isaac/PhysX support and contact evidence passes.

No runtime configuration references this package. Static generation does not
establish contact stability, load calibration, physics calibration, instrument
calibration, or clinical validity.
"""


def notice(kind: str) -> str:
    label = "PSM" if kind == "psm" else "operating-table"
    return f"""Dr.Anmar T1 {label} collider candidate {VERSION}

Status: inactive, unqualified research candidate.

The local proxy geometry, generator, contracts, and documentation were authored
by Dr.Anmar Project Developers under BSD-3-Clause. The package composes exact,
unchanged, hash-locked repository foundations; their licenses and provenance
are recorded in dependency_lock.json.

No native Isaac/PhysX qualification, physics calibration, instrument
calibration, clinical validation, or activation claim is made.
"""


def build_manifest(
    kind: str,
    package_root: Path,
    member_names: Sequence[str],
) -> dict[str, Any]:
    is_psm = kind == "psm"
    members = {
        name: {
            "bytes": (package_root / name).stat().st_size,
            "license": "BSD-3-Clause",
            "provenance": "independently_authored_DrAnmar_candidate",
            "sha256": sha256(package_root / name),
        }
        for name in member_names
    }
    dependency_names = (
        ("psm_collision_foundation", "psm_visual_overlay")
        if is_psm
        else ("table_collision_foundation", "table_visual_overlay")
    )
    return {
        "active_replacement": False,
        "asset_id": PSM_ASSET_ID if is_psm else TABLE_ASSET_ID,
        "asset_version": VERSION,
        "clinical_validation": False,
        "dependency_complete_directory": False,
        "dependency_lock_complete": True,
        "external_dependencies": {
            name: {
                "bytes": LOCKED_DEPENDENCIES[name]["bytes"],
                "modified": False,
                "repository_path": LOCKED_DEPENDENCIES[name]["path"],
                "sha256": LOCKED_DEPENDENCIES[name]["sha256"],
            }
            for name in dependency_names
        },
        "generator": {
            "repository_path": GENERATOR_REPOSITORY_PATH,
            "sha256": sha256(Path(__file__).resolve()),
            "standard_library_only": True,
        },
        "instrument_calibration": False,
        "license": "BSD-3-Clause",
        "members": members,
        "package_role": (
            "inactive_collision_overlay_with_external_hash_locked_bases"
        ),
        "physics_calibration": False,
        "provider": "Dr.Anmar Project Developers",
        "runtime_references": [],
        "schema": (
            "dr.anmar.psm-t1-collider-candidate-manifest.v1"
            if is_psm
            else "dr.anmar.table-t1-collider-candidate-manifest.v1"
        ),
        "source_static_validation": True,
        "status": STATUS,
    }


def _prepare_package(path: Path, allowed_members: set[str]) -> None:
    if path.exists():
        unknown = {
            member.relative_to(path).as_posix()
            for member in path.rglob("*")
            if member.is_file()
            and member.relative_to(path).as_posix() not in allowed_members
        }
        if unknown:
            raise RuntimeError(
                f"refusing to replace package with unknown members: {sorted(unknown)}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def generate(output_root: Path) -> None:
    verify_locked_inputs()
    generator = {
        "psm": {
            "approximation": psm_approximation_report(),
            "entrypoint": (PSM_ENTRYPOINT, psm_usda()),
            "members": PSM_GENERATED_MEMBERS,
            "path": output_root / PSM_SUBPATH,
        },
        "table": {
            "approximation": table_approximation_report(),
            "entrypoint": (TABLE_ENTRYPOINT, table_usda()),
            "members": TABLE_GENERATED_MEMBERS,
            "path": output_root / TABLE_SUBPATH,
        },
    }
    for kind, package in generator.items():
        package_root = package["path"]
        members = package["members"]
        _prepare_package(package_root, set(members) | {"asset_manifest.json"})
        entrypoint_name, entrypoint_text = package["entrypoint"]
        write_text(package_root / entrypoint_name, entrypoint_text)
        shutil.copyfile(REPOSITORY_LICENSE, package_root / "LICENSE.txt")
        write_text(package_root / "NOTICE.txt", notice(kind))
        write_text(package_root / "README.md", readme(kind))
        write_json(
            package_root / "approximation_report.json",
            package["approximation"],
        )
        write_json(
            package_root / "dependency_lock.json",
            dependency_lock(kind),
        )
        write_json(
            package_root / "geometry_contract.json",
            geometry_contract(kind),
        )
        write_json(package_root / "provenance.json", provenance(kind))
        write_json(
            package_root / "qualification_contract.json",
            qualification_contract(kind),
        )
        write_json(
            package_root / "asset_manifest.json",
            build_manifest(kind, package_root, members),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root below which canonical data/... package paths are generated "
            "(default: asset-extension root)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.output_root.resolve())


if __name__ == "__main__":
    main()
