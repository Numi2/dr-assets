# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python regression gates for the needle-ready tissue generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools/generate_needle_ready_tissue.py"
ASSET_ROOT = (
    ROOT / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_needle_ready_tissue", GENERATOR_PATH
)
assert SPEC and SPEC.loader
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


def contract():
    return json.loads(
        (ASSET_ROOT / "geometry_contract.json").read_text(encoding="utf-8")
    )


def test_mesh_counts_layer_conformity_and_semantics():
    payload = contract()
    for lod, lod_contract in payload["lods"].items():
        mesh = GENERATOR.build_mesh(payload, lod)
        assert len(mesh.points) == lod_contract["expected_points"]
        assert len(mesh.tetrahedra) == lod_contract["expected_tetrahedra"]
        assert mesh.minimum_tetra_volume_m3 > 0.0
        assert sum(map(len, mesh.material_face_sets.values())) == len(
            mesh.surface_triangles
        )
        assert all(mesh.node_sets.values())
        assert all(mesh.element_sets.values())
        assert all(mesh.semantic_face_sets.values())
        assert set(mesh.tetrahedron_layers) == set(
            range(len(payload["layers"]))
        )


def test_lods_are_exactly_point_nested():
    payload = contract()
    meshes = {
        lod: GENERATOR.build_mesh(payload, lod)
        for lod in payload["lods"]
    }
    mapping = GENERATOR.build_lod_mapping(meshes)
    assert mapping["nested_exactly"] is True
    for values in mapping["mappings"].values():
        assert values["maximum_parametric_error"] == 0.0
        assert len(values["target_indices"]) == values["source_point_count"]
        assert len(set(values["target_indices"])) == values["source_point_count"]


def test_generated_report_matches_checked_in_assets():
    report = json.loads(
        (ASSET_ROOT / "geometry_report.json").read_text(encoding="utf-8")
    )
    assert report["asset_id"] == "dranmar-needle-ready-tissue-v2"
    assert report["material_interfaces_conforming"] is True
    assert report["lods_point_nested"] is True
    assert report["clinical_validation"] is False
    for lod, values in report["lods"].items():
        path = ASSET_ROOT / values["usd"]
        assert path.is_file(), lod
        assert GENERATOR.sha256(path) == values["usd_sha256"]
