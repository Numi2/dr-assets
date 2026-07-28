# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Executable orchestration for the DrAnmar Autonomous Rescue OR.

The module separates three surfaces:

* policy intent: reversible requests such as compress, clip, patch, or verify;
* scene evidence: monotonic post-physics contact and attachment observations;
* patient outcome: bleeding, perfusion, repair integrity, and rescue success.

Policy actions never write patient outcomes.  The environment-owned scene
adapter is the only ingress to :mod:`deformable_rescue`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final

from .deformable_rescue import (
    ContactDrivenRescueEffects,
    RescueEffectsSnapshot,
    _PhysicsEvidenceFrame,
    _SceneEvidenceAdapter,
)
from .autonomous_rescue_scene_evidence import RescueVesselSceneEvidence
from .patient_authority import PatientAuthority
from .resuscitation_effects import (
    ContactDrivenResuscitationEffects,
    ResuscitationSnapshot,
)
from .resuscitation_scene_evidence import (
    PumpSceneEvidence,
    VentilationSceneEvidence,
)

ASSET_DIRECTORY: Final = (
    Path(__file__).resolve().parents[3] / "data/Environments/SurgicalAutonomy/AutonomousRescueOR"
)
AUTONOMOUS_RESCUE_OR_USD: Final = ASSET_DIRECTORY / "dranmar_autonomous_rescue_or.usda"
DEFORMABLE_RESCUE_SUITE_USD: Final = ASSET_DIRECTORY / "dranmar_deformable_rescue_suite.usda"
RESCUE_VESSEL_USD: Final = ASSET_DIRECTORY / "dranmar_rescue_vessel.usda"
RESUSCITATION_MODULE_USD: Final = ASSET_DIRECTORY / "dranmar_resuscitation_module.usda"
TOOL_CHANGER_PAYLOAD_USD: Final = ASSET_DIRECTORY / "dranmar_universal_tool_changer_payload.usda"

CONTRACT_FILES: Final = MappingProxyType(
    {
        "benchmark_scenarios": "benchmark_scenarios.json",
        "capability_matrix": "capability_matrix.json",
        "complications": "complication_library.json",
        "dependencies": "dependency_manifest.json",
        "episode_schema": "episode_schema.json",
        "imitation_dataset": "imitation_dataset_contract.json",
        "interaction_frames": "interaction_frames.json",
        "procedure_graphs": "procedure_graphs.json",
        "rescue_protocols": "rescue_protocols.json",
        "resources": "resource_inventory.json",
        "robot_stations": "robot_station_contract.json",
        "tools": "tool_registry.json",
        "workspace": "workspace_topology.json",
    }
)


def _load_contract(name: str) -> dict[str, object]:
    try:
        filename = CONTRACT_FILES[name]
    except KeyError as error:
        raise KeyError(
            f"unknown rescue contract {name!r}; expected {sorted(CONTRACT_FILES)}"
        ) from error
    path = ASSET_DIRECTORY / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("schema"):
        raise ValueError(f"invalid DrAnmar rescue contract: {path}")
    return payload


class ActionStatus(str, Enum):
    REQUESTED = "requested"
    EXECUTING = "executing"
    EFFECT_OBSERVED = "effect_observed"
    REJECTED = "rejected"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PolicyAction:
    """A policy request containing no patient outcome fields."""

    action_id: str
    station_id: str
    tool_id: str
    target_id: str
    requested_action: str

    def __post_init__(self) -> None:
        for name in (
            "action_id",
            "station_id",
            "tool_id",
            "target_id",
            "requested_action",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class ActionRecord:
    sequence: int
    physics_step: int
    action: PolicyAction
    status: ActionStatus
    reason: str


@dataclass(frozen=True)
class ComplicationObservation:
    complication_id: str
    priority: int
    rescue_protocol: str
    target_id: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RescuePlan:
    complication_id: str
    protocol_id: str
    actions: tuple[Mapping[str, object], ...]


@dataclass
class ResourceLedger:
    resources: dict[str, float]
    consumed: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resources = self._validated_entries(
            self.resources,
            "resource inventory",
        )
        self.consumed = self._validated_entries(
            self.consumed,
            "resource consumption",
        )

    @staticmethod
    def _validated_entries(
        entries: Mapping[str, float],
        label: str,
    ) -> dict[str, float]:
        validated: dict[str, float] = {}
        for raw_resource_id, raw_amount in entries.items():
            resource_id = str(raw_resource_id)
            if not resource_id:
                raise ValueError(f"{label} contains an empty resource id")
            amount = float(raw_amount)
            if not math.isfinite(amount) or amount < 0.0:
                raise ValueError(f"{label} for {resource_id} must be finite and nonnegative")
            validated[resource_id] = amount
        return validated

    @classmethod
    def from_contract(cls) -> ResourceLedger:
        payload = _load_contract("resources")
        inventory = payload.get("resources", payload.get("inventory", {}))
        resources: dict[str, float] = {}
        if isinstance(inventory, dict):
            resources = {
                str(key): float(value)
                for key, value in inventory.items()
                if isinstance(value, (int, float))
            }
        elif isinstance(inventory, list):
            for item in inventory:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                quantity = item.get("quantity", item.get("count", 0))
                if isinstance(quantity, (int, float)):
                    resources[str(item["id"])] = float(quantity)
        return cls(resources)

    def available(self, resource_id: str) -> float:
        return self.resources.get(resource_id, 0.0)

    def require_available(
        self,
        resource_id: str,
        amount: float = 1.0,
    ) -> float:
        amount = float(amount)
        if not math.isfinite(amount) or amount <= 0.0:
            raise ValueError("resource consumption must be finite and positive")
        remaining = self.available(resource_id)
        if not math.isfinite(remaining) or remaining < 0.0:
            raise RuntimeError(f"invalid inventory state for {resource_id}: {remaining}")
        if remaining < amount:
            raise RuntimeError(
                f"insufficient {resource_id}: requested {amount}, available {remaining}"
            )
        return remaining

    def consume(self, resource_id: str, amount: float = 1.0) -> None:
        amount = float(amount)
        remaining = self.require_available(resource_id, amount)
        self.resources[resource_id] = remaining - amount
        self.consumed[resource_id] = self.consumed.get(resource_id, 0.0) + amount

    def snapshot(self) -> Mapping[str, float]:
        return MappingProxyType(dict(sorted(self.resources.items())))


class ActionAdmissionKernel:
    """Fail closed when policy intent does not match installed OR capability."""

    _ACTION_CAPABILITIES: Final = MappingProxyType(
        {
            "temporary_compression": frozenset({"temporary_compression"}),
            "clip": frozenset({"vascular_clipping"}),
            "patch": frozenset({"hemostatic_patch"}),
            "release_compression": frozenset({"temporary_compression"}),
            "flow_verify": frozenset({"flow_verification", "perfusion_verification"}),
            "pressure_challenge": frozenset({"flow_verification", "perfusion_verification"}),
            "capture": frozenset({"hollow_tissue_capture"}),
            "align": frozenset({"lumen_alignment"}),
            "staple_ring": frozenset({"staple_ring"}),
            "reinforce": frozenset({"reinforcement"}),
            "pressure_test": frozenset({"leak_test"}),
            "approximate": frozenset({"wound_approximation"}),
            "adhesive": frozenset({"skin_adhesive"}),
            "load_test": frozenset({"closure_verification"}),
            "leak_test": frozenset({"leak_test", "perfusion_verification"}),
        }
    )
    _SYSTEM_ACTIONS: Final = frozenset(
        {
            "bond_perimeter",
            "transfuse",
            "crystalloid",
            "vasopressor",
            "set_fio2",
            "restore_ventilation",
        }
    )
    _SYSTEM_TOOL_IDS: Final = frozenset(
        {
            "host_film_applicator",
            "orchestrator",
            "resuscitation_module",
            "system",
        }
    )

    def __init__(self) -> None:
        station_contract = _load_contract("robot_stations")
        self._station_ids = frozenset(
            str(item["id"])
            for item in station_contract.get("stations", ())
            if isinstance(item, dict) and item.get("id")
        )
        tool_contract = _load_contract("tools")
        self._tool_capabilities = {
            str(item["id"]): frozenset(
                str(capability)
                for capability in item.get("capabilities", ())
                if isinstance(capability, str) and capability
            )
            for item in tool_contract.get("tools", ())
            if isinstance(item, dict) and item.get("id")
        }
        if not self._station_ids or not self._tool_capabilities:
            raise ValueError("action admission requires nonempty station and tool contracts")

    @property
    def allowed_actions(self) -> frozenset[str]:
        return frozenset(self._ACTION_CAPABILITIES) | self._SYSTEM_ACTIONS

    def rejection_reason(
        self,
        action: PolicyAction,
        *,
        existing_action_ids: frozenset[str],
    ) -> str | None:
        if action.action_id in existing_action_ids:
            return "duplicate_action_id"
        if action.requested_action not in self.allowed_actions:
            return "unsupported_policy_intent"

        if action.station_id == "system":
            if action.requested_action not in self._SYSTEM_ACTIONS:
                return "physical_intent_requires_robot_station"
            if action.tool_id not in self._SYSTEM_TOOL_IDS:
                return "unknown_system_tool"
            return None

        if action.station_id not in self._station_ids:
            return "unknown_robot_station"
        capabilities = self._tool_capabilities.get(action.tool_id)
        if capabilities is None:
            return "unknown_registered_tool"
        if action.requested_action in self._SYSTEM_ACTIONS:
            return "system_intent_requires_system_station"
        required = self._ACTION_CAPABILITIES[action.requested_action]
        if required.isdisjoint(capabilities):
            return "tool_lacks_requested_capability"
        return None


class ComplicationDetector:
    """Derive rescue triggers from patient effects, not authored labels."""

    def __init__(self) -> None:
        contract = _load_contract("complications")
        self._definitions = {
            str(item["id"]): item
            for item in contract.get("complications", [])
            if isinstance(item, dict) and item.get("id")
        }

    def _observation(
        self,
        complication_id: str,
        target_id: str,
        evidence: Sequence[str],
    ) -> ComplicationObservation:
        definition = self._definitions[complication_id]
        return ComplicationObservation(
            complication_id=complication_id,
            priority=int(definition.get("priority", 0)),
            rescue_protocol=str(definition["rescue_protocol"]),
            target_id=target_id,
            evidence=tuple(evidence),
        )

    def detect(
        self,
        snapshot: RescueEffectsSnapshot,
        *,
        blood_volume_fraction: float,
        spo2_fraction: float | None = None,
    ) -> tuple[ComplicationObservation, ...]:
        vessel = snapshot.vessel
        observations: list[ComplicationObservation] = []
        residual_flow = float(vessel["residual_flow_ml_s"])
        flow_evidence_available = bool(
            vessel["flow_evidence_available"]
        )
        blood_fraction = max(
            0.0,
            min(2.0, float(blood_volume_fraction)),
        )
        distal_perfusion = float(vessel["distal_perfusion_fraction"])
        if (
            flow_evidence_available
            and residual_flow >= 2.0
            and "catastrophic_hemorrhage" in self._definitions
        ):
            observations.append(
                self._observation(
                    "catastrophic_hemorrhage",
                    "rescue_vessel",
                    (
                        f"scene_residual_flow_ml_s={residual_flow:.6f}",
                        f"physics_step={snapshot.physics_step}",
                    ),
                )
            )
        if blood_fraction < 0.75 and "hypovolemic_shock" in self._definitions:
            observations.append(
                self._observation(
                    "hypovolemic_shock",
                    "patient",
                    (f"conserved_blood_volume_fraction={blood_fraction:.6f}",),
                )
            )
        if (
            spo2_fraction is not None
            and float(spo2_fraction) < 0.90
            and "hypoxemia" in self._definitions
        ):
            observations.append(
                self._observation(
                    "hypoxemia",
                    "patient",
                    (f"patient_spo2_fraction={float(spo2_fraction):.6f}",),
                )
            )
        if distal_perfusion < 0.45 and "regional_ischemia" in self._definitions:
            observations.append(
                self._observation(
                    "regional_ischemia",
                    "distal_vessel_territory",
                    (f"scene_distal_perfusion_fraction={distal_perfusion:.6f}",),
                )
            )
        for target_id, repair in snapshot.repairs.items():
            leak = float(repair["leak_rate_ml_s"])
            if (
                target_id == "bowel_anastomosis"
                and leak > 0.25
                and "anastomotic_leak" in self._definitions
            ):
                observations.append(
                    self._observation(
                        "anastomotic_leak",
                        target_id,
                        (
                            f"scene_leak_rate_ml_s={leak:.6f}",
                            f"physics_step={snapshot.physics_step}",
                        ),
                    )
                )
            if (
                target_id == "abdominal_wall"
                and float(repair["retention_fraction"]) < 0.9
                and float(repair["approximation_fraction"]) > 0.2
                and "abdominal_wall_dehiscence" in self._definitions
            ):
                observations.append(
                    self._observation(
                        "abdominal_wall_dehiscence",
                        target_id,
                        (f"scene_retention_fraction={float(repair['retention_fraction']):.6f}",),
                    )
                )
            if target_id == "occlusive_film" and int(repair["last_physics_step"]) >= 0:
                retention = float(repair["retention_fraction"])
                coverage = float(repair["contact_coverage_fraction"])
                pressure_kpa = float(repair["measured_pressure_kpa"])
                seal_quality = float(repair["seal_quality"])
                if (
                    retention > 0.05
                    and (coverage < 0.75 or seal_quality < 0.60)
                    and "dressing_delamination" in self._definitions
                ):
                    observations.append(
                        self._observation(
                            "dressing_delamination",
                            target_id,
                            (
                                f"scene_retention_fraction={retention:.6f}",
                                f"scene_contact_coverage_fraction={coverage:.6f}",
                                f"scene_seal_quality={seal_quality:.6f}",
                            ),
                        )
                    )
                if pressure_kpa < -9.0 and "dressing_compression_ischemia" in self._definitions:
                    observations.append(
                        self._observation(
                            "dressing_compression_ischemia",
                            target_id,
                            (f"scene_cavity_pressure_kpa={pressure_kpa:.6f}",),
                        )
                    )
        return tuple(
            sorted(
                observations,
                key=lambda item: (-item.priority, item.complication_id),
            )
        )


class RescuePlanner:
    """Resolve the highest-priority detected complication into a protocol."""

    def __init__(self) -> None:
        contract = _load_contract("rescue_protocols")
        raw = contract.get("protocols", {})
        if not isinstance(raw, dict):
            raise TypeError("rescue_protocols.json must define a protocol map")
        self._protocols = raw

    def plan(
        self,
        complications: Sequence[ComplicationObservation],
    ) -> RescuePlan | None:
        if not complications:
            return None
        selected = min(
            complications,
            key=lambda item: (-item.priority, item.complication_id),
        )
        raw_actions = self._protocols.get(selected.rescue_protocol, ())
        if not isinstance(raw_actions, list):
            raise TypeError(f"protocol {selected.rescue_protocol!r} must contain an action list")
        return RescuePlan(
            complication_id=selected.complication_id,
            protocol_id=selected.rescue_protocol,
            actions=tuple(
                MappingProxyType(dict(action)) for action in raw_actions if isinstance(action, dict)
            ),
        )


class _DynamicPatientRescueBridge:
    """Project evidence-derived effects into one shared patient authority."""

    def __init__(
        self,
        patient: object,
        *,
        perfusion_region: str = "small_bowel",
        ventilation_target_chest_excursion_m: float = 0.02,
    ) -> None:
        self.patient = patient
        if not isinstance(
            getattr(patient, "patient_authority", None),
            PatientAuthority,
        ):
            raise TypeError(
                "Autonomous Rescue OR requires a dynamic patient exposing "
                "PatientAuthority as patient_authority"
            )
        self.perfusion_region = perfusion_region
        self.ventilation_target_chest_excursion_m = float(ventilation_target_chest_excursion_m)
        if self.ventilation_target_chest_excursion_m <= 0.0:
            raise ValueError("ventilation_target_chest_excursion_m must be positive")
        self._applied_modeled_leak_volume_ml = 0.0
        self._applied_crystalloid_delivery_ml = 0.0
        self._applied_blood_product_delivery_ml = 0.0
        respiration = getattr(patient, "respiration", None)
        self._baseline_tidal_volume_ml = float(getattr(respiration, "tidal_volume_ml", 500.0))
        self._baseline_fio2_fraction = float(getattr(respiration, "inspired_oxygen_fraction", 0.21))
        self._applied_airway_pressure_damage = 0.0

    @property
    def authority(self) -> PatientAuthority:
        authority = getattr(self.patient, "patient_authority", None)
        if not isinstance(authority, PatientAuthority):
            raise TypeError(
                "dynamic patient no longer exposes its PatientAuthority"
            )
        return authority

    def reset(self) -> None:
        reset_patient = getattr(self.patient, "reset", None)
        if not callable(reset_patient):
            raise TypeError("dynamic patient must expose reset()")
        reset_patient()
        self._applied_modeled_leak_volume_ml = 0.0
        self._applied_crystalloid_delivery_ml = 0.0
        self._applied_blood_product_delivery_ml = 0.0
        self._applied_airway_pressure_damage = 0.0
        respiration = getattr(self.patient, "respiration", None)
        if respiration is not None:
            self._baseline_tidal_volume_ml = float(
                getattr(respiration, "tidal_volume_ml", 500.0)
            )
            self._baseline_fio2_fraction = float(
                getattr(respiration, "inspired_oxygen_fraction", 0.21)
            )
            respiration.tidal_volume_ml = self._baseline_tidal_volume_ml
            respiration.inspired_oxygen_fraction = self._baseline_fio2_fraction

    def apply(self, snapshot: RescueEffectsSnapshot) -> None:
        vessel = snapshot.vessel
        modeled_cumulative_leak_ml = float(
            vessel["modeled_cumulative_leak_volume_ml"]
        )
        requested_loss_ml = max(
            0.0,
            modeled_cumulative_leak_ml
            - self._applied_modeled_leak_volume_ml,
        )
        if requested_loss_ml:
            self.authority.withdraw_blood(requested_loss_ml)
        self._applied_modeled_leak_volume_ml = (
            modeled_cumulative_leak_ml
        )

        perfusion = getattr(self.patient, "perfusion", None)
        if perfusion is None or not hasattr(perfusion, "set_occlusion"):
            raise TypeError("dynamic patient must expose perfusion.set_occlusion")
        distal = float(vessel["distal_perfusion_fraction"])
        perfusion.set_occlusion(self.perfusion_region, 1.0 - distal)

        anastomosis = snapshot.repairs["bowel_anastomosis"]
        if hasattr(self.patient, "anastomoses"):
            self.patient.anastomoses["autonomous_rescue_bowel"] = {
                # These are reduced-model observations, not lumen geometry.
                # Keep their native meanings instead of fabricating patency
                # or converting volumetric leak rate into an area.
                "repair_retention_fraction_proxy": float(
                    anastomosis["retention_fraction"]
                ),
                "modeled_leak_rate_ml_s": float(
                    anastomosis["leak_rate_ml_s"]
                ),
                "distal_perfusion_fraction": distal,
            }
        tissue_state = getattr(self.patient, "tissue_state", None)
        if tissue_state is not None and hasattr(tissue_state, "get"):
            wall = snapshot.repairs["abdominal_wall"]
            wall_state = tissue_state.get("abdominal_wall")
            retained_closure = min(
                float(wall["approximation_fraction"]),
                float(wall["retention_fraction"]),
            )
            wall_state.closure_fraction = retained_closure
            wall_state.staples = round(14 * retained_closure)
        film = snapshot.repairs["occlusive_film"]
        if int(film["last_physics_step"]) >= 0 and hasattr(self.patient, "dressing_state"):
            self.patient.dressing_state = {
                "target": "skin",
                "pressure_kpa": float(film["measured_pressure_kpa"]),
                "seal_fraction": float(film["seal_quality"]),
                "seal_verified": bool(film["seal_verified"]),
                "applied": float(film["retention_fraction"]) > 0.0,
                "source": "autonomous_rescue_contact_effects",
            }
            if perfusion is not None and hasattr(
                perfusion,
                "set_compression",
            ):
                compression = max(
                    0.0,
                    min(
                        1.0,
                        abs(
                            min(
                                0.0,
                                float(film["measured_pressure_kpa"]),
                            )
                        )
                        / 40.0,
                    ),
                )
                perfusion.set_compression("skin", compression)

    def apply_resuscitation(
        self,
        snapshot: ResuscitationSnapshot,
        *,
        physics_step: int,
    ) -> None:
        """Project only scene-supported circulation and ventilation effects."""

        crystalloid_ml = float(snapshot.channels["crystalloid"]["delivered_to_patient_ml"])
        blood_product_ml = float(snapshot.channels["blood_product"]["delivered_to_patient_ml"])
        crystalloid_delta = max(
            0.0,
            crystalloid_ml - self._applied_crystalloid_delivery_ml,
        )
        blood_delta = max(
            0.0,
            blood_product_ml - self._applied_blood_product_delivery_ml,
        )
        if crystalloid_delta:
            self.authority.infuse_crystalloid(crystalloid_delta)
            self._applied_crystalloid_delivery_ml = crystalloid_ml
        if blood_delta:
            self.authority.transfuse_blood(blood_delta)
            self._applied_blood_product_delivery_ml = blood_product_ml

        ventilation = snapshot.ventilation
        if int(ventilation["last_physics_step"]) >= 0:
            respiration = getattr(self.patient, "respiration", None)
            if respiration is None:
                raise TypeError("dynamic patient must expose respiration")
            if int(ventilation["last_physics_step"]) != physics_step:
                respiration.tidal_volume_ml = self._baseline_tidal_volume_ml
                respiration.inspired_oxygen_fraction = self._baseline_fio2_fraction
                return
            connected = bool(ventilation["airway_connected"])
            effective_l_min = float(ventilation["effective_minute_ventilation_l_min"])
            respiratory_rate = max(
                1.0,
                float(getattr(respiration, "respiratory_rate_bpm", 14.0)),
            )
            flow_supported_tidal_ml = effective_l_min * 1000.0 / respiratory_rate
            chest_supported_tidal_ml = self._baseline_tidal_volume_ml * min(
                2.0,
                float(ventilation["chest_excursion_m"]) / self.ventilation_target_chest_excursion_m,
            )
            delivered_tidal_ml = min(
                flow_supported_tidal_ml,
                chest_supported_tidal_ml,
            )
            respiration.tidal_volume_ml = (
                delivered_tidal_ml if connected else self._baseline_tidal_volume_ml
            )
            respiration.inspired_oxygen_fraction = (
                float(ventilation["delivered_fio2_fraction"])
                if connected
                else self._baseline_fio2_fraction
            )
            pressure_damage = float(ventilation["pressure_damage_fraction"])
            pressure_damage_delta = max(
                0.0,
                pressure_damage - self._applied_airway_pressure_damage,
            )
            if pressure_damage_delta:
                respiration.airway_obstruction_fraction = min(
                    0.95,
                    float(respiration.airway_obstruction_fraction) + pressure_damage_delta,
                )
                self._applied_airway_pressure_damage = pressure_damage

    def advance_physiology(self, dt_s: float) -> None:
        """Advance the shared patient exactly once for one physics interval."""

        if not hasattr(self.patient, "step"):
            raise TypeError("dynamic patient must expose step(dt_s)")
        self.patient.step(dt_s)


class AutonomousRescueORRuntime:
    """Policy/runtime boundary for the autonomous rescue environment."""

    _OUTCOME_FIELD_NAMES: Final = frozenset(
        {
            "bleeding_controlled",
            "closure_fraction",
            "compression_fraction",
            "division_complete",
            "distal_perfusion_fraction",
            "effectiveness",
            "force_n",
            "hemostasis_verified",
            "leak_area_mm2",
            "leak_rate_ml_s",
            "map_mmhg",
            "occlusion_fraction",
            "patch_seal_fraction",
            "patency_fraction",
            "perfusion_restoration",
            "residual_flow_ml_s",
            "seal_fraction",
            "seal_quality",
            "success",
        }
    )

    def __init__(
        self,
        *,
        dynamic_patient: object,
        seed: int = 0,
    ) -> None:
        if not isinstance(
            getattr(dynamic_patient, "patient_authority", None),
            PatientAuthority,
        ):
            raise TypeError(
                "dynamic_patient with a PatientAuthority is mandatory"
            )
        self.effects = ContactDrivenRescueEffects(seed=seed)
        self._scene_adapter: _SceneEvidenceAdapter = (
            self.effects._create_scene_adapter()
        )
        self.resuscitation = ContactDrivenResuscitationEffects()
        self._pump_adapter = self.resuscitation._create_scene_adapter()
        self._patient_bridge = _DynamicPatientRescueBridge(
            dynamic_patient,
            ventilation_target_chest_excursion_m=(
                self.resuscitation.calibration.ventilation_target_chest_excursion_m
            ),
        )
        self.detector = ComplicationDetector()
        self.planner = RescuePlanner()
        self.action_admission = ActionAdmissionKernel()
        self.resources = ResourceLedger.from_contract()
        self._actions: list[ActionRecord] = []
        self._sequence = 0
        self._previous_flow_ml_s = 0.0
        self._previous_perfusion_fraction = 1.0
        self._previous_overload_damage_fraction = 0.0
        self._pending_support_reward = 0.0
        self._hemostasis_rewarded = False
        self._film_seal_rewarded = False
        self._latest_complications: tuple[ComplicationObservation, ...] = ()
        self._latest_plan: RescuePlan | None = None
        self._last_reward = 0.0
        self._last_resuscitation_reward = 0.0
        self._last_unqualified_mechanics_score = 0.0
        self._rescue_episode_id: str | None = None
        self._rescue_environment_id: str | None = None
        self._rescue_topology_revision: str | None = None
        self._last_rescue_evidence_digest_sha256: str | None = None

    @property
    def patient_authority(self) -> PatientAuthority:
        return self._patient_bridge.authority

    def _patient_spo2_fraction(self) -> float | None:
        vital_signs = getattr(
            self._patient_bridge.patient,
            "vital_signs",
            None,
        )
        value = getattr(vital_signs, "spo2_fraction", None)
        return None if value is None else float(value)

    def _patient_map_mmhg(self) -> float | None:
        vital_signs = getattr(
            self._patient_bridge.patient,
            "vital_signs",
            None,
        )
        value = getattr(vital_signs, "mean_arterial_pressure_mmhg", None)
        return None if value is None else float(value)

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]:
        self.effects.reset(seed=seed)
        self.resuscitation.reset()
        self._patient_bridge.reset()
        self.resources = ResourceLedger.from_contract()
        self._actions.clear()
        self._sequence = 0
        self._previous_flow_ml_s = 0.0
        self._previous_perfusion_fraction = 1.0
        self._previous_overload_damage_fraction = 0.0
        self._pending_support_reward = 0.0
        self._hemostasis_rewarded = False
        self._film_seal_rewarded = False
        self._latest_complications = ()
        self._latest_plan = None
        self._last_reward = 0.0
        self._last_resuscitation_reward = 0.0
        self._last_unqualified_mechanics_score = 0.0
        self._rescue_episode_id = None
        self._rescue_environment_id = None
        self._rescue_topology_revision = None
        self._last_rescue_evidence_digest_sha256 = None
        return self.policy_observation()

    def _validate_patient_wide_evidence_identity(
        self,
        *,
        episode_id: str,
        environment_id: str,
        topology_revision: str,
    ) -> None:
        """Reject cross-device evidence assembled from different runtimes."""

        if self._rescue_episode_id is not None and (
            episode_id != self._rescue_episode_id
            or environment_id != self._rescue_environment_id
        ):
            raise ValueError(
                "rescue, pump, and ventilation evidence must share one "
                "patient episode and environment until runtime reset"
            )
        if (
            self._rescue_topology_revision is not None
            and topology_revision != self._rescue_topology_revision
        ):
            raise ValueError(
                "patient topology changed without an explicit rescue-wide "
                "topology-transition barrier"
            )

    def _commit_patient_wide_evidence_identity(
        self,
        *,
        episode_id: str,
        environment_id: str,
        topology_revision: str,
    ) -> None:
        self._rescue_episode_id = episode_id
        self._rescue_environment_id = environment_id
        self._rescue_topology_revision = topology_revision

    def request_action(
        self,
        action: PolicyAction,
        **unexpected_outcomes: object,
    ) -> ActionRecord:
        """Record intent while rejecting every caller-authored patient result."""

        forbidden = self._OUTCOME_FIELD_NAMES.intersection(unexpected_outcomes)
        if forbidden:
            raise ValueError(
                "policy actions cannot author patient outcomes: " + ", ".join(sorted(forbidden))
            )
        if unexpected_outcomes:
            raise ValueError(
                "unsupported policy action fields: " + ", ".join(sorted(unexpected_outcomes))
            )
        rejection_reason = self.action_admission.rejection_reason(
            action,
            existing_action_ids=frozenset(record.action.action_id for record in self._actions),
        )
        if rejection_reason is not None:
            record = self._record(
                action,
                ActionStatus.REJECTED,
                rejection_reason,
            )
            return record
        record = self._record(
            action,
            ActionStatus.REQUESTED,
            "intent_recorded_waiting_for_scene_effect",
        )
        return record

    def _record(
        self,
        action: PolicyAction,
        status: ActionStatus,
        reason: str,
    ) -> ActionRecord:
        self._sequence += 1
        record = ActionRecord(
            sequence=self._sequence,
            physics_step=self.effects.snapshot().physics_step,
            action=action,
            status=status,
            reason=reason,
        )
        self._actions.append(record)
        return record

    def advance_vessel_scene(
        self,
        evidence: RescueVesselSceneEvidence,
    ) -> Mapping[str, object]:
        """Advance from one prim-bound, post-physics vessel interval."""

        if not isinstance(evidence, RescueVesselSceneEvidence):
            raise TypeError(
                "Autonomous Rescue OR requires RescueVesselSceneEvidence"
            )
        provenance = evidence.envelope.provenance
        self._validate_patient_wide_evidence_identity(
            episode_id=provenance.episode_id,
            environment_id=provenance.environment_id,
            topology_revision=provenance.topology_revision,
        )
        if (
            evidence.evidence_digest_sha256
            == self._last_rescue_evidence_digest_sha256
        ):
            raise ValueError("duplicate rescue scene evidence envelope")
        previous = self.effects.snapshot()
        previous_spo2 = self._patient_spo2_fraction()
        previous_map = self._patient_map_mmhg()
        previous_loss_ml = (
            self.patient_authority.cumulative_blood_loss_ml
        )
        frame = _PhysicsEvidenceFrame(
            physics_step=evidence.physics_step,
            simulation_time_s=evidence.simulation_time_s,
            dt_s=evidence.dt_s,
            station_id=evidence.station_id,
            tool_id=evidence.tool_id,
            target_id=evidence.target_id,
            left_normal_force_n=evidence.left_normal_force_n,
            right_normal_force_n=evidence.right_normal_force_n,
            separation_m=evidence.separation_m,
            tool_speed_m_s=evidence.tool_speed_m_s,
            target_distance_m=evidence.target_distance_m,
            evidence_digest_sha256=(
                evidence.evidence_digest_sha256
            ),
            retained_attachment_prim_ids=(
                evidence.attachment_prim_ids
            ),
            measured_upstream_pressure_mmhg=(
                evidence.measured_upstream_pressure_mmhg
            ),
        )
        current = self._scene_adapter.publish(frame)
        current = self._scene_adapter.finalize_interval(
            frozenset({"rescue_vessel"})
        )
        self._patient_bridge.apply(current)
        self._patient_bridge.apply_resuscitation(
            self.resuscitation.snapshot(),
            physics_step=frame.physics_step,
        )
        self._patient_bridge.advance_physiology(frame.dt_s)
        current_spo2 = self._patient_spo2_fraction()
        current_map = self._patient_map_mmhg()
        self._latest_complications = self.detector.detect(
            current,
            blood_volume_fraction=(
                self.patient_authority.blood_volume_fraction
            ),
            spo2_fraction=current_spo2,
        )
        self._latest_plan = self.planner.plan(self._latest_complications)
        previous_flow = float(previous.vessel["residual_flow_ml_s"])
        current_flow = float(current.vessel["residual_flow_ml_s"])
        previous_flow_available = bool(
            previous.vessel["flow_evidence_available"]
        )
        current_flow_available = bool(
            current.vessel["flow_evidence_available"]
        )
        current_loss_ml = (
            self.patient_authority.cumulative_blood_loss_ml
        )
        perfusion = float(current.vessel["distal_perfusion_fraction"])
        verified = bool(current.vessel["hemostasis_verified"])
        overload = float(current.vessel["overload_damage_fraction"])
        newly_verified = verified and not self._hemostasis_rewarded
        previous_film = previous.repairs["occlusive_film"]
        current_film = current.repairs["occlusive_film"]
        film_leak = float(current_film["leak_rate_ml_s"])
        film_quality = float(current_film["seal_quality"])
        film_verified = bool(current_film["seal_verified"])
        newly_film_verified = film_verified and not self._film_seal_rewarded
        film_was_observed = int(previous_film["last_physics_step"]) >= 0
        film_leak_improvement = (
            float(previous_film["leak_rate_ml_s"]) - film_leak if film_was_observed else 0.0
        )
        film_quality_improvement = (
            film_quality - float(previous_film["seal_quality"]) if film_was_observed else 0.0
        )
        flow_improvement = (
            previous_flow - current_flow
            if (
                previous.physics_step >= 0
                and previous_flow_available
                and current_flow_available
            )
            else 0.0
        )
        oxygenation_improvement = (
            current_spo2 - previous_spo2
            if current_spo2 is not None and previous_spo2 is not None
            else 0.0
        )
        map_improvement = (
            current_map - previous_map
            if current_map is not None and previous_map is not None
            else 0.0
        )
        self._last_unqualified_mechanics_score = (
            2.0 * flow_improvement
            - 0.5 * (current_loss_ml - previous_loss_ml)
            - 1.5 * max(0.0, self._previous_perfusion_fraction - perfusion)
            - 4.0
            * max(
                0.0,
                overload - self._previous_overload_damage_fraction,
            )
            + (8.0 if newly_verified else 0.0)
            + 1.5 * film_leak_improvement
            + 2.0 * film_quality_improvement
            + (6.0 if newly_film_verified else 0.0)
            + 20.0 * oxygenation_improvement
            + map_improvement
            + self._pending_support_reward
        )
        # No evidence object is yet bound to one admitted PolicyAction, and the
        # public runtime has no same-step batch barrier for every active rescue
        # target.  Mechanics deltas are therefore diagnostic only and cannot
        # become a training reward or success signal.
        self._last_reward = 0.0
        self._pending_support_reward = 0.0
        self._hemostasis_rewarded = self._hemostasis_rewarded or verified
        self._film_seal_rewarded = self._film_seal_rewarded or film_verified
        self._previous_flow_ml_s = current_flow
        self._previous_perfusion_fraction = perfusion
        self._previous_overload_damage_fraction = overload
        self._commit_patient_wide_evidence_identity(
            episode_id=provenance.episode_id,
            environment_id=provenance.environment_id,
            topology_revision=provenance.topology_revision,
        )
        self._last_rescue_evidence_digest_sha256 = (
            evidence.evidence_digest_sha256
        )
        return self.policy_observation()

    def advance_resuscitation(
        self,
        evidence: PumpSceneEvidence,
    ) -> Mapping[str, object]:
        """Advance volume support from conserved post-physics pump evidence."""

        if not isinstance(evidence, PumpSceneEvidence):
            raise TypeError(
                "Autonomous Rescue OR requires PumpSceneEvidence"
            )
        self._validate_patient_wide_evidence_identity(
            episode_id=evidence.episode_id,
            environment_id=evidence.environment_id,
            topology_revision=evidence.topology_revision,
        )
        previous = self.resuscitation.snapshot()
        authority_before = self.patient_authority.fluid_snapshot()
        resource_id = {
            "crystalloid": "crystalloid_ml",
            "blood_product": "blood_products_ml",
            "vasopressor": "vasopressor_syringes",
        }[evidence.channel_id]
        projected_withdrawal = (
            self._pump_adapter.projected_reservoir_withdrawal_ml(evidence)
        )
        projected_resource_amount = projected_withdrawal
        if evidence.channel_id == "vasopressor":
            projected_resource_amount /= (
                self.resuscitation.calibration.vasopressor_volume_per_stroke_ml
            )
        if projected_resource_amount:
            self.resources.require_available(
                resource_id,
                projected_resource_amount,
            )

        current = self._pump_adapter.publish(evidence)
        previous_withdrawn = float(
            previous.channels[evidence.channel_id]["withdrawn_from_reservoir_ml"]
        )
        current_withdrawn = float(
            current.channels[evidence.channel_id][
                "withdrawn_from_reservoir_ml"
            ]
        )
        withdrawn_delta = max(0.0, current_withdrawn - previous_withdrawn)
        resource_amount = withdrawn_delta
        if evidence.channel_id == "vasopressor":
            resource_amount /= self.resuscitation.calibration.vasopressor_volume_per_stroke_ml
        if not math.isclose(
            resource_amount,
            projected_resource_amount,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("resuscitation inventory projection diverged from scene evidence")
        if resource_amount:
            self.resources.consume(resource_id, resource_amount)

        self._patient_bridge.apply_resuscitation(
            current,
            physics_step=evidence.physics_step,
        )
        authority_after = self.patient_authority.fluid_snapshot()
        rescue_snapshot = self.effects.snapshot()
        self._latest_complications = self.detector.detect(
            rescue_snapshot,
            blood_volume_fraction=authority_after.blood_volume_fraction,
            spo2_fraction=self._patient_spo2_fraction(),
        )
        self._latest_plan = self.planner.plan(self._latest_complications)

        circulating_gain_ml = max(
            0.0,
            authority_after.intravascular_volume_ml
            - authority_before.intravascular_volume_ml,
        )
        deficit_before = max(
            0.0,
            authority_before.baseline_blood_volume_ml
            - authority_before.intravascular_volume_ml,
        )
        useful_gain = min(circulating_gain_ml, deficit_before)
        previous_channel = previous.channels[evidence.channel_id]
        current_channel = current.channels[evidence.channel_id]
        waste_delta = max(
            0.0,
            float(current_channel["wasted_or_extravasated_ml"])
            - float(previous_channel["wasted_or_extravasated_ml"]),
        )
        pressure_damage_delta = max(
            0.0,
            float(current_channel["pressure_damage_fraction"])
            - float(previous_channel["pressure_damage_fraction"]),
        )
        self._last_unqualified_mechanics_score = (
            0.01 * useful_gain - 0.02 * waste_delta - 10.0 * pressure_damage_delta
        )
        self._last_resuscitation_reward = 0.0
        self._last_reward = 0.0
        self._commit_patient_wide_evidence_identity(
            episode_id=evidence.episode_id,
            environment_id=evidence.environment_id,
            topology_revision=evidence.topology_revision,
        )
        return self.policy_observation()

    def advance_ventilation(
        self,
        evidence: VentilationSceneEvidence,
    ) -> Mapping[str, object]:
        """Advance airway support from connected circuit and chest evidence."""

        if not isinstance(evidence, VentilationSceneEvidence):
            raise TypeError(
                "Autonomous Rescue OR requires VentilationSceneEvidence"
            )
        self._validate_patient_wide_evidence_identity(
            episode_id=evidence.episode_id,
            environment_id=evidence.environment_id,
            topology_revision=evidence.topology_revision,
        )
        previous = self.resuscitation.snapshot()
        current = self._pump_adapter.publish_ventilation(evidence)
        self._patient_bridge.apply_resuscitation(
            current,
            physics_step=evidence.physics_step,
        )
        damage_delta = max(
            0.0,
            float(current.ventilation["pressure_damage_fraction"])
            - float(previous.ventilation["pressure_damage_fraction"]),
        )
        self._last_unqualified_mechanics_score = -10.0 * damage_delta
        self._last_resuscitation_reward = 0.0
        self._last_reward = 0.0
        self._commit_patient_wide_evidence_identity(
            episode_id=evidence.episode_id,
            environment_id=evidence.environment_id,
            topology_revision=evidence.topology_revision,
        )
        return self.policy_observation()

    def policy_observation(self) -> Mapping[str, object]:
        patient = MappingProxyType(
            {
                **dict(self.effects.policy_observation()),
                "authority": self.patient_authority.snapshot(),
            }
        )
        active = tuple(
            MappingProxyType(
                {
                    "id": item.complication_id,
                    "priority": item.priority,
                    "target_id": item.target_id,
                    "evidence": item.evidence,
                }
            )
            for item in self._latest_complications
        )
        plan = None
        if self._latest_plan is not None:
            plan = MappingProxyType(
                {
                    "complication_id": self._latest_plan.complication_id,
                    "protocol_id": self._latest_plan.protocol_id,
                    "actions": self._latest_plan.actions,
                }
            )
        return MappingProxyType(
            {
                "patient": patient,
                "resuscitation": self.resuscitation.snapshot().channels,
                "ventilation": self.resuscitation.snapshot().ventilation,
                "active_complications": active,
                "rescue_plan": plan,
                "resources": self.resources.snapshot(),
                "last_reward": self._last_reward,
                "last_resuscitation_reward": (self._last_resuscitation_reward),
                "reward_authority": (
                    "disabled_pending_action_bound_same_step_target_batch"
                ),
                "action_count": len(self._actions),
            }
        )

    def trace(self) -> tuple[ActionRecord, ...]:
        return tuple(self._actions)


def autonomous_rescue_or_cfg(
    prim_path: str = "{ENV_REGEX_NS}/AutonomousRescueOR",
):
    """Return a deferred Isaac Lab asset configuration for the OR stage."""

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Isaac Lab is required to create the Autonomous Rescue OR stage"
        ) from error
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(AUTONOMOUS_RESCUE_OR_USD),
        ),
    )


def _create_vertex_xform_attachment(
    stage,
    *,
    attachment_path: str,
    source_path: str,
    target_path: str,
    vertex_indices: Sequence[int],
) -> str:
    """Bind selected TetMesh vertices to an authored fixture without snapping."""

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    source_prim = stage.GetPrimAtPath(source_path)
    target_prim = stage.GetPrimAtPath(target_path)
    if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.PointBased):
        raise RuntimeError(f"rescue attachment source is not point based: {source_path}")
    if not target_prim.IsValid() or not UsdGeom.Xformable(target_prim):
        raise RuntimeError(f"rescue attachment target is not xformable: {target_path}")
    points = list(UsdGeom.PointBased(source_prim).GetPointsAttr().Get() or ())
    selected = tuple(dict.fromkeys(int(index) for index in vertex_indices))
    if not selected or any(index < 0 or index >= len(points) for index in selected):
        raise RuntimeError(f"rescue attachment has no valid source vertices: {source_path}")

    source_to_world = UsdGeom.Xformable(source_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_to_target = (
        UsdGeom.Xformable(target_prim)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .GetInverse()
    )
    local_positions = [
        Gf.Vec3f(world_to_target.Transform(source_to_world.Transform(Gf.Vec3d(points[index]))))
        for index in selected
    ]
    parent_path = str(Sdf.Path(attachment_path).GetParentPath())
    if not stage.GetPrimAtPath(parent_path).IsValid():
        stage.DefinePrim(parent_path, "Scope")
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)
    attachment = stage.DefinePrim(
        attachment_path,
        "OmniPhysicsVtxXformAttachment",
    )
    attachment.CreateRelationship("omniphysics:src0").SetTargets([Sdf.Path(source_path)])
    attachment.CreateRelationship("omniphysics:src1").SetTargets([Sdf.Path(target_path)])
    attachment.CreateAttribute(
        "omniphysics:vtxIndicesSrc0",
        Sdf.ValueTypeNames.IntArray,
    ).Set(Vt.IntArray(selected))
    attachment.CreateAttribute(
        "omniphysics:localPositionsSrc1",
        Sdf.ValueTypeNames.Point3fArray,
    ).Set(Vt.Vec3fArray(local_positions))
    attachment.CreateAttribute(
        "omniphysics:attachmentEnabled",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    return attachment_path


def anchor_rescue_vessel(
    vessel_path: str,
    *,
    stage=None,
    endpoint_tolerance_m: float = 1.0e-6,
) -> tuple[str, str]:
    """Fix the two vessel ends while leaving the central rescue zone deformable."""

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    root = vessel_path.rstrip("/")
    simulation_path = f"{root}/VesselWall/SimulationMesh"
    simulation_prim = stage.GetPrimAtPath(simulation_path)
    if not simulation_prim.IsValid() or not simulation_prim.IsA(UsdGeom.PointBased):
        raise RuntimeError(f"rescue vessel has no simulation TetMesh at {simulation_path}")
    points = list(UsdGeom.PointBased(simulation_prim).GetPointsAttr().Get() or ())
    if not points:
        raise RuntimeError(f"rescue vessel TetMesh has no points: {simulation_path}")
    x_values = [float(point[0]) for point in points]
    minimum_x = min(x_values)
    maximum_x = max(x_values)
    endpoint_indices = {
        "Left": [
            index
            for index, value in enumerate(x_values)
            if abs(value - minimum_x) <= endpoint_tolerance_m
        ],
        "Right": [
            index
            for index, value in enumerate(x_values)
            if abs(value - maximum_x) <= endpoint_tolerance_m
        ],
    }
    created = []
    for side in ("Left", "Right"):
        indices = endpoint_indices[side]
        if len(indices) < 4:
            raise RuntimeError(
                f"{side.lower()} rescue vessel endpoint is too sparse: " f"{len(indices)} vertices"
            )
        created.append(
            _create_vertex_xform_attachment(
                stage,
                attachment_path=f"{root}/RuntimeAttachments/{side}Fixture",
                source_path=simulation_path,
                target_path=f"{root}/AttachmentMasks/{side}Fixture",
                vertex_indices=indices,
            )
        )
    return tuple(created)


def rescue_vessel_cfg(
    prim_path: str = "{ENV_REGEX_NS}/AutonomousRescueVessel",
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.06),
):
    """Return the live, endpoint-anchored rescue substrate for a native bench."""

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Isaac Lab is required to create the Autonomous Rescue vessel"
        ) from error

    spawn = sim_utils.UsdFileCfg(
        usd_path=str(RESCUE_VESSEL_USD),
        variants={"hemostasis_state": "bleeding"},
    )
    source_spawn = spawn.func

    def spawn_anchored_rescue_vessel(
        spawned_prim_path: str,
        cfg: sim_utils.UsdFileCfg,
        translation=None,
        orientation=None,
        **kwargs,
    ):
        root_prim = source_spawn(
            spawned_prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        anchor_rescue_vessel(str(root_prim.GetPath()))
        return root_prim

    spawn.func = spawn_anchored_rescue_vessel
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=spawn,
    )


def resuscitation_module_cfg(
    prim_path: str = "{ENV_REGEX_NS}/ResuscitationModule",
    *,
    position: tuple[float, float, float] = (-0.75, 0.0, 0.0),
):
    """Return the articulated plunger/ventilation module for a live scene."""

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("Isaac Lab is required to create the resuscitation module") from error
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(RESUSCITATION_MODULE_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*plunger_joint": 0.0,
                "ventilation_valve_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "fluid_pumps": ImplicitActuatorCfg(
                joint_names_expr=[".*plunger_joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=0.06,
                stiffness=2200.0,
                damping=80.0,
            ),
            "ventilation": ImplicitActuatorCfg(
                joint_names_expr=["ventilation_valve_joint"],
                effort_limit_sim=18.0,
                velocity_limit_sim=1.5,
                stiffness=120.0,
                damping=8.0,
            ),
        },
    )


__all__ = [
    "ASSET_DIRECTORY",
    "AUTONOMOUS_RESCUE_OR_USD",
    "DEFORMABLE_RESCUE_SUITE_USD",
    "RESCUE_VESSEL_USD",
    "RESUSCITATION_MODULE_USD",
    "TOOL_CHANGER_PAYLOAD_USD",
    "ActionAdmissionKernel",
    "ActionRecord",
    "ActionStatus",
    "AutonomousRescueORRuntime",
    "ComplicationDetector",
    "ComplicationObservation",
    "PolicyAction",
    "RescuePlan",
    "RescuePlanner",
    "ResourceLedger",
    "anchor_rescue_vessel",
    "autonomous_rescue_or_cfg",
    "rescue_vessel_cfg",
    "resuscitation_module_cfg",
]
