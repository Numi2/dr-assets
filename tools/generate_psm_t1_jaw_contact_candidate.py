#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the inactive DrAnmar T1 PSM jaw-contact material candidate.

The generated layer references the existing collision-qualified PSM and
overrides only the two friction coefficients on the physics material already
bound exclusively to the two jaw collision meshes.  It is deliberately not
wired into any runtime configuration.

Run from the asset-extension root:

    python3 tools/generate_psm_t1_jaw_contact_candidate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET_SUBPATH = Path("data/Robots/dVRK/PSM/T1JawContactCandidate")
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / ASSET_SUBPATH
BASE_ASSET_SUBPATH = Path("data/Robots/dVRK/PSM/psm_col.usd")
BASE_ASSET = REPOSITORY_ROOT / BASE_ASSET_SUBPATH
BASE_ASSET_SHA256 = "87330b0e46e5554d53fe1b840f2db38a9bdbf00a79ca464159009406156ca0d4"
BASE_ASSET_BYTES = 22_251_170
REPOSITORY_LICENSE = REPOSITORY_ROOT / "LICENSE"
REPOSITORY_LICENSE_SHA256 = (
    "9c69bf1f27dd03eeed8dd8aa2fa8a86702f6608b7f6397e8ea9d4c6a1d895c76"
)

ASSET_ID = "dranmar-psm-t1-jaw-contact-candidate-v1"
ASSET_VERSION = "1.0.0"
STATUS = "inactive_unqualified_candidate"
CANDIDATE_USD = "psm_t1_jaw_contact_candidate.usda"
BASE_REFERENCE = "../psm_col.usd"
BASE_ROOT_PRIM = "/psm"
PHYSICS_MATERIAL_PATH = "/psm/Looks/PhysicsMaterial"
JAW_COLLISION_PRIMS = (
    "/psm/psm_tool_gripper1_link/collisions_xform/collisions",
    "/psm/psm_tool_gripper2_link/collisions_xform/collisions",
)

# These values are engineering hypotheses, not identified or calibrated
# coefficients.  Their only purpose is to replace the impossible inherited
# ordering dynamicFriction > staticFriction with a bounded qualification seed.
STATIC_FRICTION_SEED = 0.60
DYNAMIC_FRICTION_SEED = 0.45
STATIC_FRICTION_RANGE = (0.35, 0.90)
DYNAMIC_TO_STATIC_RATIO_RANGE = (0.65, 0.90)
RESTITUTION_RANGE = (0.00, 0.05)

GENERATED_MEMBERS = (
    CANDIDATE_USD,
    "LICENSE.txt",
    "NOTICE.txt",
    "README.md",
    "dependency_lock.json",
    "friction_hypothesis.json",
    "provenance.json",
    "qualification_contract.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_locked_inputs() -> None:
    if not BASE_ASSET.is_file():
        raise FileNotFoundError(f"missing locked PSM dependency: {BASE_ASSET}")
    if BASE_ASSET.stat().st_size != BASE_ASSET_BYTES:
        raise RuntimeError("psm_col.usd byte count changed; review before regenerating")
    if sha256(BASE_ASSET) != BASE_ASSET_SHA256:
        raise RuntimeError("psm_col.usd hash changed; review before regenerating")
    if not REPOSITORY_LICENSE.is_file():
        raise FileNotFoundError(f"missing repository license: {REPOSITORY_LICENSE}")
    if sha256(REPOSITORY_LICENSE) != REPOSITORY_LICENSE_SHA256:
        raise RuntimeError("repository LICENSE hash changed; review before regenerating")


def candidate_usda() -> str:
    return f"""#usda 1.0
(
    defaultPrim = "psm"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "psm" (
    prepend references = @{BASE_REFERENCE}@<{BASE_ROOT_PRIM}>
)
{{
    custom bool dranmar:candidate:active = false
    custom string dranmar:candidate:assetId = "{ASSET_ID}"
    custom string dranmar:candidate:qualificationContract = "./qualification_contract.json"
    custom string dranmar:candidate:status = "{STATUS}"
    custom string dranmar:candidate:version = "{ASSET_VERSION}"

    over "Looks"
    {{
        over "PhysicsMaterial"
        {{
            custom string dranmar:candidate:calibrationStatus = "uncalibrated_engineering_hypothesis"
            custom string dranmar:candidate:contactModel = "coulomb_only_no_adhesion"
            custom string dranmar:candidate:frictionHypothesis = "./friction_hypothesis.json"
            float physics:dynamicFriction = {DYNAMIC_FRICTION_SEED:.2f}
            float physics:staticFriction = {STATIC_FRICTION_SEED:.2f}
        }}
    }}
}}
"""


def dependency_lock() -> dict[str, Any]:
    return {
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "dependency_complete": True,
        "dependencies": {
            "base_psm_collision_asset": {
                "bytes": BASE_ASSET_BYTES,
                "default_prim": BASE_ROOT_PRIM,
                "license": "BSD-3-Clause",
                "modified_by_candidate": False,
                "reference_from_candidate": BASE_REFERENCE,
                "repository_path": BASE_ASSET_SUBPATH.as_posix(),
                "sha256": BASE_ASSET_SHA256,
            }
        },
        "resolution_scope": (
            "The sole runtime dependency is the adjacent, repository-tracked "
            "psm_col.usd and is locked by byte count and SHA-256."
        ),
        "status": STATUS,
    }


def friction_hypothesis() -> dict[str, Any]:
    static_low, static_high = STATIC_FRICTION_RANGE
    ratio_low, ratio_high = DYNAMIC_TO_STATIC_RATIO_RANGE
    return {
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "calibration_status": "uncalibrated_engineering_hypothesis",
        "candidate_seed": {
            "dynamic_friction": DYNAMIC_FRICTION_SEED,
            "dynamic_to_static_ratio": DYNAMIC_FRICTION_SEED / STATIC_FRICTION_SEED,
            "static_friction": STATIC_FRICTION_SEED,
        },
        "contact_model": {
            "adhesion": 0.0,
            "cohesion": 0.0,
            "magnetism": 0.0,
            "model": "ordinary_coulomb_friction",
            "restitution": "inherited_from_locked_base_value_0.0",
            "suction": 0.0,
        },
        "engineering_rationale": [
            (
                "The seed restores the physical ordering static friction greater "
                "than or equal to dynamic friction."
            ),
            (
                "The seed is intentionally below unity so friction cannot silently "
                "replace bilateral grasp geometry or normal force."
            ),
            (
                "The collision meshes do not resolve serration teeth. The envelope "
                "is only an effective steel/serrated-jaw research hypothesis and "
                "does not claim measured material identification."
            ),
            (
                "Activation requires native contact and task qualification; no "
                "clinical, instrument, or physics-calibration claim is made."
            ),
        ],
        "forbidden_shortcuts": {
            "adhesion": True,
            "contact_state_teleport": True,
            "force_injection": True,
            "magnetism": True,
            "object_attachment": True,
            "suction": True,
        },
        "randomization_envelope": {
            "constraints": [
                "0 <= dynamic_friction <= static_friction <= 0.90",
                "sample dynamic friction as static friction times the sampled ratio",
                "do not tune on evaluation seeds",
                "do not change adhesion, cohesion, magnetism, suction, or attachment",
            ],
            "dynamic_friction": {
                "derived_absolute_maximum": static_high * ratio_high,
                "derived_absolute_minimum": static_low * ratio_low,
                "formula": "static_friction * dynamic_to_static_ratio",
            },
            "dynamic_to_static_ratio": {
                "distribution": "uniform",
                "maximum": ratio_high,
                "minimum": ratio_low,
            },
            "restitution": {
                "distribution": "uniform",
                "maximum": RESTITUTION_RANGE[1],
                "minimum": RESTITUTION_RANGE[0],
                "note": "sensitivity-only; candidate USDA inherits zero",
            },
            "static_friction": {
                "distribution": "uniform",
                "maximum": static_high,
                "minimum": static_low,
            },
        },
        "status": STATUS,
    }


def qualification_contract() -> dict[str, Any]:
    required_suites = {
        "analytic_pickup": {
            "comparison": "paired_against_locked_base_on_identical_held_out_seeds",
            "metrics": [
                "pickup_success",
                "first_attempt_pickup_success",
                "recovered_pickup_success",
                "attempt_count",
            ],
            "requirement": "pre_registered_noninferiority_and_no_reward_definition_change",
        },
        "first_attempt_retention": {
            "metrics": [
                "bilateral_jaw_contact_at_lift",
                "needle_retained_without_regrasp",
                "time_to_secure_grasp_s",
            ],
            "requirement": "reported_separately_from_recovery_success",
        },
        "mid_air_transport": {
            "metrics": [
                "conditional_retention_given_pickup",
                "slip_distance_m",
                "needle_drop",
                "transport_completion",
            ],
            "requirement": "no_attachment_and_no_contact_state_override",
        },
        "two_arm_handover": {
            "metrics": [
                "giver_retention_before_receiver_contact",
                "simultaneous_bilateral_contact",
                "receiver_retention_after_release",
                "handover_drop",
            ],
            "requirement": "physics_contact_owned_transition_only",
        },
        "normal_force_and_contact": {
            "metrics": [
                "per_jaw_normal_force_n",
                "normal_impulse_ns",
                "bilateral_contact_duty_cycle",
                "peak_contact_force_n",
                "contact_point_count",
                "needle_relative_slip_speed_m_s",
            ],
            "requirement": (
                "archive distributions and video-linked outliers; success cannot "
                "be inferred from commanded jaw closure"
            ),
        },
        "friction_sensitivity": {
            "metrics": [
                "success_by_static_friction_bin",
                "success_by_dynamic_ratio_bin",
                "force_and_slip_by_bin",
                "failure_mode_by_bin",
            ],
            "requirement": (
                "evaluate the complete predeclared envelope and reject brittle "
                "single-coefficient optima"
            ),
        },
        "held_out_seeds": {
            "minimum_independent_seed_sets": 3,
            "requirement": (
                "evaluation seeds remain disjoint from controller or residual "
                "training and all exclusions are reported"
            ),
        },
    }
    return {
        "activation": {
            "active_replacement": False,
            "allowed_after_all_required_suites_pass": True,
            "decision_owner": "explicit_human_review",
            "default": "blocked",
            "requires_native_isaac_physx_evidence": True,
        },
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "evidence_integrity": {
            "archive": [
                "resolved_runtime_asset_hashes",
                "candidate_and_base_layer_hashes",
                "configuration_hash",
                "checkpoint_hash_or_null",
                "seed_lists",
                "per_episode_metrics",
                "contact_telemetry",
                "failure_taxonomy",
            ],
            "forbid_post_hoc_threshold_changes": True,
            "reward_contract_must_be_identical_between_baseline_and_candidate": True,
        },
        "required_suites": required_suites,
        "status": STATUS,
        "validation_boundaries": {
            "clinical_validation": False,
            "instrument_calibration": False,
            "physics_calibration": False,
            "source_only_generation_is_qualification": False,
        },
    }


def provenance() -> dict[str, Any]:
    return {
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "authorship": {
            "candidate_layer_and_contracts": "Dr.Anmar Project Developers",
            "base_asset": "ORBIT-Surgical Project Developers",
        },
        "base_asset": {
            "bytes": BASE_ASSET_BYTES,
            "license": "BSD-3-Clause",
            "modified": False,
            "repository_path": BASE_ASSET_SUBPATH.as_posix(),
            "sha256": BASE_ASSET_SHA256,
        },
        "generation": {
            "deterministic": True,
            "external_downloads": [],
            "generator": "tools/generate_psm_t1_jaw_contact_candidate.py",
            "third_party_content_added": False,
        },
        "status": STATUS,
    }


def readme() -> str:
    return f"""# T1 PSM jaw-contact candidate

This directory contains an **inactive, unqualified** OpenUSD composition that
references `../psm_col.usd` at `{BASE_ROOT_PRIM}` and changes only
`physics:staticFriction` and `physics:dynamicFriction` on
`{PHYSICS_MATERIAL_PATH}`. The locked base binds that material only to:

- `{JAW_COLLISION_PRIMS[0]}`
- `{JAW_COLLISION_PRIMS[1]}`

The selected seed (`static={STATIC_FRICTION_SEED:.2f}`,
`dynamic={DYNAMIC_FRICTION_SEED:.2f}`) is an engineering hypothesis for a
steel/serrated-jaw contact model. It is not measured, physics-calibrated,
instrument-calibrated, or clinically validated. The layer adds no adhesion,
magnetism, suction, attachment, force injection, geometry, collision, joint, or
transform opinions.

`friction_hypothesis.json` defines the bounded, correlated sensitivity
envelope. `qualification_contract.json` blocks activation until native PhysX
evidence covers analytic pickup, first-attempt retention, mid-air transport,
two-arm handover, contact/normal-force telemetry, friction sensitivity, and
held-out seeds.

No task or robot configuration selects this asset. Use of the wrapper as a
runtime replacement requires an explicit review and a separately committed
activation change after the qualification contract passes.
"""


def notice() -> str:
    return f"""Dr.Anmar T1 PSM jaw-contact candidate {ASSET_VERSION}

Status: inactive, unqualified research candidate.

The candidate layer and local contracts were authored by the Dr.Anmar Project
Developers under BSD-3-Clause. The wrapper references the unchanged
ORBIT-Surgical `data/Robots/dVRK/PSM/psm_col.usd`, also under BSD-3-Clause.
See LICENSE.txt, provenance.json, and dependency_lock.json.

No clinical validation, instrument calibration, physics calibration, or
activation claim is made.
"""


def build_manifest(output_root: Path) -> dict[str, Any]:
    members = {}
    for name in GENERATED_MEMBERS:
        path = output_root / name
        members[name] = {
            "bytes": path.stat().st_size,
            "license": "BSD-3-Clause",
            "provenance": "independently_authored_DrAnmar_candidate",
            "sha256": sha256(path),
        }
    return {
        "schema": "dr.anmar.psm-t1-jaw-contact-candidate-manifest.v1",
        "active_replacement": False,
        "asset_id": ASSET_ID,
        "asset_version": ASSET_VERSION,
        "base_dependency": {
            "bytes": BASE_ASSET_BYTES,
            "modified": False,
            "repository_path": BASE_ASSET_SUBPATH.as_posix(),
            "sha256": BASE_ASSET_SHA256,
        },
        "clinical_validation": False,
        "dependency_complete_directory": False,
        "dependency_lock_complete": True,
        "package_role": "inactive_overlay_with_external_hash_locked_base",
        "generator": {
            "repository_path": "tools/generate_psm_t1_jaw_contact_candidate.py",
            "sha256": sha256(Path(__file__).resolve()),
            "standard_library_only": True,
        },
        "instrument_calibration": False,
        "license": "BSD-3-Clause",
        "members": members,
        "physics_calibration": False,
        "runtime_references": [],
        "status": STATUS,
    }


def generate(output_root: Path) -> None:
    verify_locked_inputs()
    output_root.mkdir(parents=True, exist_ok=True)
    allowed = set(GENERATED_MEMBERS) | {"asset_manifest.json"}
    unknown = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.relative_to(output_root).as_posix() not in allowed
    }
    if unknown:
        raise RuntimeError(f"refusing to overwrite package with unknown members: {sorted(unknown)}")

    (output_root / CANDIDATE_USD).write_text(candidate_usda(), encoding="utf-8")
    shutil.copyfile(REPOSITORY_LICENSE, output_root / "LICENSE.txt")
    (output_root / "NOTICE.txt").write_text(notice(), encoding="utf-8")
    (output_root / "README.md").write_text(readme(), encoding="utf-8")
    write_json(output_root / "dependency_lock.json", dependency_lock())
    write_json(output_root / "friction_hypothesis.json", friction_hypothesis())
    write_json(output_root / "provenance.json", provenance())
    write_json(output_root / "qualification_contract.json", qualification_contract())
    write_json(output_root / "asset_manifest.json", build_manifest(output_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Generated package root (default: canonical asset directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.output_root.resolve())


if __name__ == "__main__":
    main()
