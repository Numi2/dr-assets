# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Static/OpenUSD gates for the inactive PSM T1 jaw-contact candidate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = ROOT.parents[2]
ASSET_ROOT = ROOT / "data/Robots/dVRK/PSM/T1JawContactCandidate"
BASE_USD = ROOT / "data/Robots/dVRK/PSM/psm_col.usd"
CANDIDATE_USD = ASSET_ROOT / "psm_t1_jaw_contact_candidate.usda"
GENERATOR = ROOT / "tools/generate_psm_t1_jaw_contact_candidate.py"
MANIFEST = ASSET_ROOT / "asset_manifest.json"

ASSET_ID = "dranmar-psm-t1-jaw-contact-candidate-v1"
BASE_SHA256 = "87330b0e46e5554d53fe1b840f2db38a9bdbf00a79ca464159009406156ca0d4"
BASE_BYTES = 22_251_170
ROOT_PRIM = "/psm"
MATERIAL_PRIM = "/psm/Looks/PhysicsMaterial"
JAW_COLLISION_PRIMS = {
    "/psm/psm_tool_gripper1_link/collisions_xform/collisions",
    "/psm/psm_tool_gripper2_link/collisions_xform/collisions",
}
ROOT_CANDIDATE_PROPERTIES = {
    "dranmar:candidate:active",
    "dranmar:candidate:assetId",
    "dranmar:candidate:qualificationContract",
    "dranmar:candidate:status",
    "dranmar:candidate:version",
}
MATERIAL_CANDIDATE_PROPERTIES = {
    "dranmar:candidate:calibrationStatus",
    "dranmar:candidate:contactModel",
    "dranmar:candidate:frictionHypothesis",
}
MATERIAL_OVERRIDES = {
    "physics:dynamicFriction",
    "physics:staticFriction",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((ASSET_ROOT / name).read_text(encoding="utf-8"))


def _stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path.resolve()))
    assert stage is not None
    return stage


def test_package_is_dependency_locked_hash_locked_and_inactive():
    manifest = _json("asset_manifest.json")
    expected_members = {
        path.relative_to(ASSET_ROOT).as_posix()
        for path in ASSET_ROOT.rglob("*")
        if path.is_file() and path.name != MANIFEST.name
    }
    assert set(manifest["members"]) == expected_members
    assert manifest["asset_id"] == ASSET_ID
    assert manifest["schema"].endswith(".v1")
    assert manifest["status"] == "inactive_unqualified_candidate"
    assert manifest["active_replacement"] is False
    assert manifest["dependency_complete_directory"] is False
    assert manifest["dependency_lock_complete"] is True
    assert manifest["package_role"] == (
        "inactive_overlay_with_external_hash_locked_base"
    )
    assert manifest["runtime_references"] == []
    assert manifest["physics_calibration"] is False
    assert manifest["instrument_calibration"] is False
    assert manifest["clinical_validation"] is False
    assert manifest["license"] == "BSD-3-Clause"
    assert manifest["generator"]["standard_library_only"] is True
    assert _sha256(GENERATOR) == manifest["generator"]["sha256"]
    for relative, expected in manifest["members"].items():
        member = ASSET_ROOT / relative
        assert member.stat().st_size == expected["bytes"]
        assert _sha256(member) == expected["sha256"]
        assert expected["license"] == "BSD-3-Clause"

    dependency = manifest["base_dependency"]
    assert dependency["repository_path"] == "data/Robots/dVRK/PSM/psm_col.usd"
    assert dependency["modified"] is False
    assert dependency["bytes"] == BASE_BYTES
    assert dependency["sha256"] == BASE_SHA256
    assert BASE_USD.stat().st_size == BASE_BYTES
    assert _sha256(BASE_USD) == BASE_SHA256

    lock = _json("dependency_lock.json")
    assert lock["dependency_complete"] is True
    locked_base = lock["dependencies"]["base_psm_collision_asset"]
    assert locked_base["reference_from_candidate"] == "../psm_col.usd"
    assert locked_base["default_prim"] == ROOT_PRIM
    assert locked_base["modified_by_candidate"] is False
    assert locked_base["sha256"] == BASE_SHA256
    assert (ASSET_ROOT / "LICENSE.txt").read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_friction_seed_and_randomization_are_bounded_hypotheses_without_adhesion():
    hypothesis = _json("friction_hypothesis.json")
    assert hypothesis["calibration_status"] == "uncalibrated_engineering_hypothesis"
    seed = hypothesis["candidate_seed"]
    assert seed["static_friction"] == pytest.approx(0.60)
    assert seed["dynamic_friction"] == pytest.approx(0.45)
    assert 0.0 <= seed["dynamic_friction"] <= seed["static_friction"] <= 1.0
    assert seed["dynamic_to_static_ratio"] == pytest.approx(0.75)

    model = hypothesis["contact_model"]
    assert model["model"] == "ordinary_coulomb_friction"
    for shortcut in ("adhesion", "cohesion", "magnetism", "suction"):
        assert model[shortcut] == 0.0
    assert model["restitution"] == "inherited_from_locked_base_value_0.0"
    assert all(hypothesis["forbidden_shortcuts"].values())

    envelope = hypothesis["randomization_envelope"]
    static = envelope["static_friction"]
    ratio = envelope["dynamic_to_static_ratio"]
    dynamic = envelope["dynamic_friction"]
    assert static == {
        "distribution": "uniform",
        "maximum": 0.90,
        "minimum": 0.35,
    }
    assert ratio == {
        "distribution": "uniform",
        "maximum": 0.90,
        "minimum": 0.65,
    }
    assert dynamic["derived_absolute_minimum"] == pytest.approx(
        static["minimum"] * ratio["minimum"]
    )
    assert dynamic["derived_absolute_maximum"] == pytest.approx(
        static["maximum"] * ratio["maximum"]
    )
    assert dynamic["derived_absolute_maximum"] <= static["maximum"]
    restitution = envelope["restitution"]
    assert restitution["minimum"] == 0.0
    assert restitution["maximum"] <= 0.05


def test_qualification_contract_fails_closed_and_requires_all_retention_evidence():
    contract = _json("qualification_contract.json")
    activation = contract["activation"]
    assert activation["default"] == "blocked"
    assert activation["active_replacement"] is False
    assert activation["requires_native_isaac_physx_evidence"] is True
    assert activation["decision_owner"] == "explicit_human_review"

    required = contract["required_suites"]
    assert set(required) == {
        "analytic_pickup",
        "first_attempt_retention",
        "mid_air_transport",
        "two_arm_handover",
        "normal_force_and_contact",
        "friction_sensitivity",
        "held_out_seeds",
    }
    assert "first_attempt_pickup_success" in required["analytic_pickup"]["metrics"]
    assert "recovered_pickup_success" in required["analytic_pickup"]["metrics"]
    assert "conditional_retention_given_pickup" in required["mid_air_transport"]["metrics"]
    assert "receiver_retention_after_release" in required["two_arm_handover"]["metrics"]
    assert "per_jaw_normal_force_n" in required["normal_force_and_contact"]["metrics"]
    assert "needle_relative_slip_speed_m_s" in required["normal_force_and_contact"]["metrics"]
    assert required["held_out_seeds"]["minimum_independent_seed_sets"] >= 3
    assert contract["evidence_integrity"]["forbid_post_hoc_threshold_changes"] is True
    assert (
        contract["evidence_integrity"][
            "reward_contract_must_be_identical_between_baseline_and_candidate"
        ]
        is True
    )
    assert all(value is False for value in contract["validation_boundaries"].values())


def test_candidate_layer_authors_only_reference_metadata_and_two_friction_values():
    layer = Sdf.Layer.FindOrOpen(str(CANDIDATE_USD.resolve()))
    assert layer is not None
    assert layer.defaultPrim == "psm"
    root_spec = layer.GetPrimAtPath(ROOT_PRIM)
    assert root_spec is not None
    references = list(root_spec.referenceList.prependedItems)
    assert len(references) == 1
    assert references[0].assetPath == "../psm_col.usd"
    assert references[0].primPath == Sdf.Path(ROOT_PRIM)

    observed_specs: set[str] = set()

    def collect(path: Sdf.Path) -> bool:
        if str(path) != "/":
            observed_specs.add(str(path))
        return True

    layer.Traverse("/", collect)
    expected_specs = {
        ROOT_PRIM,
        "/psm/Looks",
        MATERIAL_PRIM,
        *(f"{ROOT_PRIM}.{name}" for name in ROOT_CANDIDATE_PROPERTIES),
        *(f"{MATERIAL_PRIM}.{name}" for name in MATERIAL_CANDIDATE_PROPERTIES),
        *(f"{MATERIAL_PRIM}.{name}" for name in MATERIAL_OVERRIDES),
    }
    assert observed_specs == expected_specs
    assert layer.GetPrimAtPath("/psm/Looks").specifier == Sdf.SpecifierOver
    assert layer.GetPrimAtPath(MATERIAL_PRIM).specifier == Sdf.SpecifierOver
    forbidden_tokens = (
        "adhesion",
        "cohesion",
        "magnet",
        "suction",
        "attach",
        "joint",
        "collision",
        "transform",
        "xform",
        "force",
    )
    authored_property_names = {
        path.rsplit(".", 1)[-1].lower()
        for path in observed_specs
        if "." in path
    }
    assert not any(
        token in property_name
        for property_name in authored_property_names
        for token in forbidden_tokens
    )


def test_composed_candidate_changes_nothing_else_and_retains_both_jaw_bindings():
    base = _stage(BASE_USD)
    candidate = _stage(CANDIDATE_USD)
    assert str(candidate.GetDefaultPrim().GetPath()) == ROOT_PRIM
    assert {str(prim.GetPath()) for prim in candidate.Traverse()} == {
        str(prim.GetPath()) for prim in base.Traverse()
    }

    for base_prim in base.Traverse():
        path = str(base_prim.GetPath())
        candidate_prim = candidate.GetPrimAtPath(base_prim.GetPath())
        assert candidate_prim
        assert candidate_prim.GetTypeName() == base_prim.GetTypeName()
        assert list(candidate_prim.GetAppliedSchemas()) == list(base_prim.GetAppliedSchemas())

        base_properties = set(base_prim.GetPropertyNames())
        candidate_properties = set(candidate_prim.GetPropertyNames())
        allowed_additions: set[str] = set()
        if path == ROOT_PRIM:
            allowed_additions = ROOT_CANDIDATE_PROPERTIES
        elif path == MATERIAL_PRIM:
            allowed_additions = MATERIAL_CANDIDATE_PROPERTIES
        assert candidate_properties == base_properties | allowed_additions

        for name in base_properties:
            base_property = base_prim.GetProperty(name)
            candidate_property = candidate_prim.GetProperty(name)
            if path == MATERIAL_PRIM and name in MATERIAL_OVERRIDES:
                continue
            assert type(candidate_property) is type(base_property)
            if isinstance(base_property, Usd.Attribute):
                assert candidate_property.GetTypeName() == base_property.GetTypeName()
                assert candidate_property.GetVariability() == base_property.GetVariability()
                assert candidate_property.IsCustom() == base_property.IsCustom()
                assert candidate_property.GetConnections() == base_property.GetConnections()
                assert candidate_property.GetTimeSamples() == base_property.GetTimeSamples()
                assert candidate_property.Get() == base_property.Get()
                for time in base_property.GetTimeSamples():
                    assert candidate_property.Get(time) == base_property.Get(time)
            else:
                assert isinstance(base_property, Usd.Relationship)
                assert candidate_property.GetTargets() == base_property.GetTargets()

    material = candidate.GetPrimAtPath(MATERIAL_PRIM)
    assert material.HasAPI(UsdPhysics.MaterialAPI)
    static_friction = material.GetAttribute("physics:staticFriction")
    dynamic_friction = material.GetAttribute("physics:dynamicFriction")
    assert static_friction.Get() == pytest.approx(0.60)
    assert dynamic_friction.Get() == pytest.approx(0.45)
    assert static_friction.Get() >= dynamic_friction.Get()
    assert material.GetAttribute("physics:restitution").Get() == pytest.approx(0.0)
    assert material.GetAttribute("physxMaterial:improvePatchFriction").Get() is True

    bound_prims = set()
    for prim in candidate.Traverse():
        binding = prim.GetRelationship("material:binding:physics")
        if binding and binding.HasAuthoredTargets():
            assert binding.GetTargets() == [Sdf.Path(MATERIAL_PRIM)]
            bound_prims.add(str(prim.GetPath()))
    assert bound_prims == JAW_COLLISION_PRIMS
    for jaw_path in JAW_COLLISION_PRIMS:
        jaw = candidate.GetPrimAtPath(jaw_path)
        assert jaw.HasAPI(UsdPhysics.CollisionAPI)
    for prim_path in {MATERIAL_PRIM, *JAW_COLLISION_PRIMS}:
        property_names = {
            name.lower() for name in candidate.GetPrimAtPath(prim_path).GetPropertyNames()
        }
        assert not any(
            token in property_name
            for property_name in property_names
            for token in ("adhesion", "cohesion", "magnet", "suction", "attachment")
        )


def test_candidate_is_not_selected_by_runtime_sources():
    needles = {
        ASSET_ID,
        "T1JawContactCandidate",
        "psm_t1_jaw_contact_candidate.usda",
    }
    scan_roots = (
        ROOT / "orbit",
        ROOT / "config",
        WORKTREE_ROOT / "source/extensions/orbit.surgical.tasks/orbit",
        WORKTREE_ROOT / "scripts",
    )
    selected = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".json", ".py", ".sh", ".toml", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(needle in text for needle in needles):
                selected.append(path.relative_to(WORKTREE_ROOT).as_posix())
    assert selected == []


def test_generation_is_byte_deterministic(tmp_path: Path):
    generated = tmp_path / "T1JawContactCandidate"
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output-root", str(generated)],
        cwd=ROOT,
        check=True,
    )
    canonical_members = {
        path.relative_to(ASSET_ROOT).as_posix()
        for path in ASSET_ROOT.rglob("*")
        if path.is_file()
    }
    generated_members = {
        path.relative_to(generated).as_posix()
        for path in generated.rglob("*")
        if path.is_file()
    }
    assert generated_members == canonical_members
    for relative in canonical_members:
        assert (generated / relative).read_bytes() == (ASSET_ROOT / relative).read_bytes()
