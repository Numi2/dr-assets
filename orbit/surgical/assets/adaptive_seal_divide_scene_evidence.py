# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound mechanics evidence for the Adaptive Seal-and-Divide workcell.

The public ingress carries raw registered scene measurements together with
observations from the shared cohesive and vessel mechanics.  It does not
accept caller-authored seal maturity, residual gap, stump-flow, division,
damage, or success scalars.

All thresholds and sensor interpretations remain provisional engineering
parameters.  This module is not clinically validated or approved for
patient-care use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Mapping, Protocol, runtime_checkable

from .scene_evidence import (
    EvidenceProvenance,
    SceneEvidenceEnvelope,
    SceneEvidenceRegistry,
    SceneEvidenceSource,
    SceneMeasurement,
)
from orbit.surgical.physics.cohesive import CohesiveInterface, CohesiveResponse
from orbit.surgical.physics.contact import ContactSample
from orbit.surgical.physics.vessel import VesselObservation


_SIDES = ("left", "right")
_CONTACT_VALUE_COUNT = 15
_ELECTRICAL_VALUE_COUNT = 3
_COHESIVE_VALUE_COUNT = 11
_VESSEL_VALUE_COUNT = 13


def _required_text(value: str, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _scene_path(value: str, label: str) -> str:
    result = _required_text(value, label)
    if not result.startswith("/"):
        raise ValueError(f"{label} must be an absolute scene path")
    return result


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive(value: float, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _fraction(value: float, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _vector3(
    value: tuple[float, float, float],
    label: str,
) -> tuple[float, float, float]:
    result = tuple(_finite(component, f"{label} component") for component in value)
    if len(result) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return result


def _norm(value: tuple[float, ...]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _attachment_ids(
    values: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    result = tuple(
        sorted(_scene_path(value, f"{label} item") for value in values)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique prim paths")
    return result


def _contact_values(
    sample: ContactSample,
    contact_area_m2: float,
    *,
    source_prim_path: str,
    target_prim_path: str,
) -> tuple[float, ...]:
    observed_pair = (sample.key.body_a, sample.key.body_b)
    expected_pair = (source_prim_path, target_prim_path)
    if observed_pair == expected_pair:
        orientation = 1.0
    elif observed_pair == tuple(reversed(expected_pair)):
        orientation = -1.0
    else:
        raise ValueError(
            "contact sample bodies do not match the registered source-to-"
            "target pair"
        )
    return (
        *sample.point_m,
        *(orientation * value for value in sample.normal_a_to_b),
        sample.separation_m,
        sample.normal_force_n,
        *(orientation * value for value in sample.tangential_force_n),
        *(
            orientation * value
            for value in sample.relative_tangential_velocity_m_s
        ),
        contact_area_m2,
    )


def _cohesive_values(response: CohesiveResponse) -> tuple[float, ...]:
    return (
        response.contact_area_m2,
        response.normal_pressure_pa,
        response.normal_traction_pa,
        *response.shear_traction_pa,
        response.resultant_force_n,
        response.damage,
        1.0 if response.failed else 0.0,
        response.stored_energy_j_m2,
        response.dissipated_energy_j_m2,
    )


def _vessel_values(observation: VesselObservation) -> tuple[float, ...]:
    return (
        observation.residual_defect_area_m2,
        observation.upstream_pressure_pa,
        observation.downstream_pressure_pa,
        observation.inlet_flow_m3_s,
        observation.measured_flow_m3_s,
        observation.leak_flow_m3_s,
        observation.storage_flow_m3_s,
        observation.mass_balance_residual_m3_s,
        observation.lumen_area_m2,
        observation.occlusion_fraction,
        observation.wall_damage_fraction,
        observation.contact_area_m2,
        observation.contact_pressure_pa,
    )


def _measurement(
    envelope: SceneEvidenceEnvelope,
    source_id: str,
) -> SceneMeasurement:
    try:
        return next(
            item
            for item in envelope.measurements
            if item.source_id == source_id
        )
    except StopIteration as error:
        raise ValueError(
            f"seal/divide evidence is missing source {source_id!r}"
        ) from error


def _raw_ids(
    values_by_source: Mapping[str, tuple[str, ...]],
    source_id: str,
    *,
    identity_marker: str | None = None,
) -> tuple[str, ...]:
    values = tuple(
        sorted(
            _required_text(value, f"{source_id} raw sample ID")
            for value in values_by_source[source_id]
        )
    )
    if not values:
        raise ValueError(
            f"seal/divide source {source_id!r} requires raw sample identity"
        )
    if len(values) != len(set(values)):
        raise ValueError(
            f"seal/divide source {source_id!r} has duplicate raw sample IDs"
        )
    if identity_marker is None:
        return values
    if identity_marker in values:
        raise ValueError(
            f"{source_id!r} raw sample IDs must not use the reserved "
            "mechanics identity marker"
        )
    return tuple(sorted((*values, identity_marker)))


@dataclass(frozen=True)
class SealDivideZoneSources:
    """Exact source and attachment prims for one stump seal zone."""

    side: str
    vessel_prim_path: str
    upper_contact_prim_path: str
    lower_contact_prim_path: str
    temperature_sensor_prim_path: str
    impedance_sensor_prim_path: str
    electrical_sensor_prim_path: str
    seal_band_prim_path: str
    cohesive_interface_id: str
    vessel_segment_id: str
    upper_compression_attachment_prim_ids: tuple[str, ...]
    lower_compression_attachment_prim_ids: tuple[str, ...]
    seal_attachment_prim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES!r}")
        for name in (
            "vessel_prim_path",
            "upper_contact_prim_path",
            "lower_contact_prim_path",
            "temperature_sensor_prim_path",
            "impedance_sensor_prim_path",
            "electrical_sensor_prim_path",
            "seal_band_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )
        for name in ("cohesive_interface_id", "vessel_segment_id"):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name),
            )
        for name in (
            "upper_compression_attachment_prim_ids",
            "lower_compression_attachment_prim_ids",
            "seal_attachment_prim_ids",
        ):
            object.__setattr__(
                self,
                name,
                _attachment_ids(getattr(self, name), name),
            )
            if not getattr(self, name):
                raise ValueError(
                    f"{name} must register at least one attachment prim"
                )
        all_attachments = (
            *self.upper_compression_attachment_prim_ids,
            *self.lower_compression_attachment_prim_ids,
            *self.seal_attachment_prim_ids,
        )
        if len(all_attachments) != len(set(all_attachments)):
            raise ValueError(
                f"{self.side} zone attachment prim identities must be unique"
            )


@dataclass(frozen=True)
class AdaptiveSealDivideSceneSources:
    """Complete scene-time registration for one seal/divide workcell."""

    workcell_id: str
    tool_root_prim_path: str
    left: SealDivideZoneSources
    right: SealDivideZoneSources
    blade_prim_path: str
    blade_joint_prim_path: str
    blade_tip_prim_path: str
    blade_contact_prim_path: str
    blade_guard_prim_path: str
    blade_guard_joint_prim_path: str
    bridge_topology_prim_path: str
    cut_plane_prim_path: str
    tissue_center_reference_prim_path: str
    vessel_center_prim_path: str
    bridge_attachment_prim_ids: tuple[str, ...]
    blade_start_position_m: float = 0.0
    blade_end_position_m: float = 0.041
    blade_tip_start_position_tool_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.176,
    )
    blade_travel_axis_tool: tuple[float, float, float] = (0.0, 0.0, 1.0)
    blade_tip_consistency_tolerance_m: float = 0.001
    guard_retraction_sign: int = -1
    guard_retracted_coordinate_m: float = 0.010
    centering_tolerance_m: float = 0.002
    minimum_centering_force_n: float = 1.0
    blade_position_tolerance_m: float = 0.001
    maximum_blade_contact_separation_m: float = 0.0005
    minimum_blade_contact_force_n: float = 0.05
    calibration_profile_id: str = "dranmar-seal-divide-engineering-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workcell_id",
            _required_text(self.workcell_id, "workcell_id"),
        )
        object.__setattr__(
            self,
            "tool_root_prim_path",
            _scene_path(self.tool_root_prim_path, "tool_root_prim_path"),
        )
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError(
                "left and right zone registrations must match their sides"
            )
        if self.left.vessel_prim_path == self.right.vessel_prim_path:
            raise ValueError(
                "left and right zones must register distinct vessel prims"
            )
        if self.left.seal_band_prim_path == self.right.seal_band_prim_path:
            raise ValueError(
                "left and right zones must register distinct seal-band prims"
            )
        contact_paths = (
            self.left.upper_contact_prim_path,
            self.left.lower_contact_prim_path,
            self.right.upper_contact_prim_path,
            self.right.lower_contact_prim_path,
        )
        if len(contact_paths) != len(set(contact_paths)):
            raise ValueError(
                "each seal-zone jaw contact must register a distinct prim"
            )
        if (
            self.left.cohesive_interface_id
            == self.right.cohesive_interface_id
            or self.left.vessel_segment_id
            == self.right.vessel_segment_id
        ):
            raise ValueError(
                "left and right zones must register distinct shared-mechanics "
                "identities"
            )
        tool_prefix = f"{self.tool_root_prim_path.rstrip('/')}/"
        for label, path in (
            (
                "left upper contact",
                self.left.upper_contact_prim_path,
            ),
            (
                "left lower contact",
                self.left.lower_contact_prim_path,
            ),
            (
                "left temperature sensor",
                self.left.temperature_sensor_prim_path,
            ),
            (
                "left impedance sensor",
                self.left.impedance_sensor_prim_path,
            ),
            (
                "left electrical sensor",
                self.left.electrical_sensor_prim_path,
            ),
            (
                "right upper contact",
                self.right.upper_contact_prim_path,
            ),
            (
                "right lower contact",
                self.right.lower_contact_prim_path,
            ),
            (
                "right temperature sensor",
                self.right.temperature_sensor_prim_path,
            ),
            (
                "right impedance sensor",
                self.right.impedance_sensor_prim_path,
            ),
            (
                "right electrical sensor",
                self.right.electrical_sensor_prim_path,
            ),
        ):
            if not path.startswith(tool_prefix):
                raise ValueError(
                    f"{label} must be beneath tool_root_prim_path"
                )
        for name in (
            "blade_prim_path",
            "blade_joint_prim_path",
            "blade_tip_prim_path",
            "blade_contact_prim_path",
            "blade_guard_prim_path",
            "blade_guard_joint_prim_path",
            "bridge_topology_prim_path",
            "cut_plane_prim_path",
            "tissue_center_reference_prim_path",
            "vessel_center_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )
        for name in (
            "blade_prim_path",
            "blade_joint_prim_path",
            "blade_tip_prim_path",
            "blade_contact_prim_path",
            "blade_guard_prim_path",
            "blade_guard_joint_prim_path",
            "cut_plane_prim_path",
            "tissue_center_reference_prim_path",
        ):
            if not getattr(self, name).startswith(tool_prefix):
                raise ValueError(
                    f"{name} must be beneath tool_root_prim_path"
                )
        if not self.blade_contact_prim_path.startswith(
            f"{self.blade_prim_path.rstrip('/')}/"
        ):
            raise ValueError(
                "blade_contact_prim_path must be beneath blade_prim_path"
            )
        if not self.blade_tip_prim_path.startswith(
            f"{self.blade_prim_path.rstrip('/')}/"
        ):
            raise ValueError(
                "blade_tip_prim_path must be beneath blade_prim_path"
            )
        bridge_prefix = f"{self.bridge_topology_prim_path.rstrip('/')}/"
        object.__setattr__(
            self,
            "bridge_attachment_prim_ids",
            _attachment_ids(
                self.bridge_attachment_prim_ids,
                "bridge_attachment_prim_ids",
            ),
        )
        if not self.bridge_attachment_prim_ids:
            raise ValueError(
                "bridge_attachment_prim_ids must register the severable bridge"
            )
        if any(
            not attachment.startswith(bridge_prefix)
            for attachment in self.bridge_attachment_prim_ids
        ):
            raise ValueError(
                "every bridge attachment must be beneath "
                "bridge_topology_prim_path"
            )
        all_attachment_ids = (
            *self.left.upper_compression_attachment_prim_ids,
            *self.left.lower_compression_attachment_prim_ids,
            *self.left.seal_attachment_prim_ids,
            *self.right.upper_compression_attachment_prim_ids,
            *self.right.lower_compression_attachment_prim_ids,
            *self.right.seal_attachment_prim_ids,
            *self.bridge_attachment_prim_ids,
        )
        if len(all_attachment_ids) != len(set(all_attachment_ids)):
            raise ValueError(
                "compression, seal, and bridge attachment registrations "
                "must be globally unique"
            )
        object.__setattr__(
            self,
            "calibration_profile_id",
            _required_text(
                self.calibration_profile_id,
                "calibration_profile_id",
            ),
        )
        start = _finite(self.blade_start_position_m, "blade_start_position_m")
        end = _finite(self.blade_end_position_m, "blade_end_position_m")
        if math.isclose(start, end, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("blade start and end positions must differ")
        object.__setattr__(
            self,
            "blade_tip_start_position_tool_m",
            _vector3(
                self.blade_tip_start_position_tool_m,
                "blade_tip_start_position_tool_m",
            ),
        )
        axis = _vector3(
            self.blade_travel_axis_tool,
            "blade_travel_axis_tool",
        )
        if not math.isclose(
            _norm(axis),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("blade_travel_axis_tool must be unit length")
        object.__setattr__(self, "blade_travel_axis_tool", axis)
        if self.guard_retraction_sign not in (-1, 1):
            raise ValueError("guard_retraction_sign must be either -1 or 1")
        for name in (
            "guard_retracted_coordinate_m",
            "centering_tolerance_m",
            "minimum_centering_force_n",
            "blade_tip_consistency_tolerance_m",
            "blade_position_tolerance_m",
            "maximum_blade_contact_separation_m",
            "minimum_blade_contact_force_n",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        source_ids = tuple(
            source.source_id for source in self.registered_sources
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("seal/divide source IDs must be unique")

    def zone(self, side: str) -> SealDivideZoneSources:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError(f"side must be one of {_SIDES!r}")

    def zone_source_id(self, side: str, quantity: str) -> str:
        self.zone(side)
        return f"{self.workcell_id}.{side}.{quantity}"

    @property
    def blade_joint_source_id(self) -> str:
        return f"{self.workcell_id}.blade.joint_position"

    @property
    def blade_tip_source_id(self) -> str:
        return f"{self.workcell_id}.blade.tip_position"

    def blade_contact_source_id(self, side: str) -> str:
        self.zone(side)
        return f"{self.workcell_id}.blade.{side}_vessel_contact"

    @property
    def bridge_topology_source_id(self) -> str:
        return f"{self.workcell_id}.bridge.topology"

    @property
    def guard_source_id(self) -> str:
        return f"{self.workcell_id}.blade.guard"

    @property
    def alignment_source_id(self) -> str:
        return f"{self.workcell_id}.tissue.alignment"

    @property
    def registered_sources(self) -> tuple[SceneEvidenceSource, ...]:
        profile = self.calibration_profile_id
        result: list[SceneEvidenceSource] = []
        for zone in (self.left, self.right):
            prefix = zone.side
            result.extend(
                (
                    SceneEvidenceSource(
                        source_id=self.zone_source_id(
                            prefix,
                            "upper_contact",
                        ),
                        prim_path=zone.upper_contact_prim_path,
                        quantity="solver_contact_sample_and_area",
                        unit="SI_vector",
                        coordinate_frame="contact",
                        calibration_profile_id=profile,
                    ),
                    SceneEvidenceSource(
                        source_id=self.zone_source_id(
                            prefix,
                            "lower_contact",
                        ),
                        prim_path=zone.lower_contact_prim_path,
                        quantity="solver_contact_sample_and_area",
                        unit="SI_vector",
                        coordinate_frame="contact",
                        calibration_profile_id=profile,
                    ),
                    SceneEvidenceSource(
                        source_id=self.zone_source_id(
                            prefix,
                            "temperature",
                        ),
                        prim_path=zone.temperature_sensor_prim_path,
                        quantity="interface_temperature",
                        unit="degC",
                        coordinate_frame="seal_zone",
                        calibration_profile_id=profile,
                    ),
                    SceneEvidenceSource(
                        source_id=self.zone_source_id(
                            prefix,
                            "impedance",
                        ),
                        prim_path=zone.impedance_sensor_prim_path,
                        quantity="interface_impedance",
                        unit="ohm",
                        coordinate_frame="seal_zone",
                        calibration_profile_id=profile,
                    ),
                    SceneEvidenceSource(
                        source_id=self.zone_source_id(
                            prefix,
                            "electrical",
                        ),
                        prim_path=zone.electrical_sensor_prim_path,
                        quantity=(
                            "rms_voltage_current_and_active_power_factor"
                        ),
                        unit="V_A_1",
                        coordinate_frame="generator_channel",
                        calibration_profile_id=profile,
                    ),
                    SceneEvidenceSource(
                        source_id=self.zone_source_id(
                            prefix,
                            "cohesive",
                        ),
                        prim_path=zone.seal_band_prim_path,
                        quantity="shared_cohesive_response",
                        unit="SI_vector",
                        coordinate_frame="seal_interface",
                        calibration_profile_id=profile,
                    ),
                    SceneEvidenceSource(
                        source_id=self.zone_source_id(
                            prefix,
                            "vessel",
                        ),
                        prim_path=zone.vessel_prim_path,
                        quantity="shared_vessel_observation",
                        unit="SI_vector",
                        coordinate_frame="vessel_lumen",
                        calibration_profile_id=profile,
                    ),
                )
            )
        result.extend(
            (
                SceneEvidenceSource(
                    source_id=self.blade_joint_source_id,
                    prim_path=self.blade_joint_prim_path,
                    quantity="blade_joint_position",
                    unit="m",
                    coordinate_frame="joint_axis",
                    calibration_profile_id=profile,
                ),
                SceneEvidenceSource(
                    source_id=self.blade_tip_source_id,
                    prim_path=self.blade_tip_prim_path,
                    quantity="blade_tip_position",
                    unit="m",
                    coordinate_frame="tool_root",
                    calibration_profile_id=profile,
                ),
                *(
                    SceneEvidenceSource(
                        source_id=self.blade_contact_source_id(side),
                        prim_path=self.blade_contact_prim_path,
                        quantity="solver_contact_sample_and_area",
                        unit="SI_vector",
                        coordinate_frame="contact",
                        calibration_profile_id=profile,
                    )
                    for side in _SIDES
                ),
                SceneEvidenceSource(
                    source_id=self.bridge_topology_source_id,
                    prim_path=self.bridge_topology_prim_path,
                    quantity="active_attachment_count_and_identities",
                    unit="count",
                    coordinate_frame="scene_topology",
                    calibration_profile_id=profile,
                ),
                SceneEvidenceSource(
                    source_id=self.guard_source_id,
                    prim_path=self.blade_guard_joint_prim_path,
                    quantity="guard_joint_position",
                    unit="m",
                    coordinate_frame="guard_axis",
                    calibration_profile_id=profile,
                ),
                SceneEvidenceSource(
                    source_id=self.alignment_source_id,
                    prim_path=self.tissue_center_reference_prim_path,
                    quantity="tool_reference_and_vessel_center_position",
                    unit="m",
                    coordinate_frame="world",
                    calibration_profile_id=profile,
                ),
            )
        )
        return tuple(result)

    @property
    def expected_source_ids(self) -> frozenset[str]:
        return frozenset(
            source.source_id for source in self.registered_sources
        )

    @property
    def registration_digest_sha256(self) -> str:
        canonical = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def bind_envelope_digest(self, envelope_digest_sha256: str) -> str:
        payload = (
            f"{_required_text(envelope_digest_sha256, 'envelope digest')}:"
            f"{self.registration_digest_sha256}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_envelope(self, envelope: SceneEvidenceEnvelope) -> None:
        provenance = envelope.provenance
        if provenance.adapter_id != AdaptiveSealDivideSceneEvidence.ADAPTER_ID:
            raise ValueError("seal/divide evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != AdaptiveSealDivideSceneEvidence.ADAPTER_VERSION
        ):
            raise ValueError("seal/divide evidence has the wrong adapter_version")
        registry = SceneEvidenceRegistry(self.registered_sources)
        registry.validate(envelope)
        measurements = {
            item.source_id: item for item in envelope.measurements
        }
        if frozenset(measurements) != self.expected_source_ids:
            raise ValueError(
                "seal/divide evidence source set does not match registered "
                "workcell sources"
            )
        expected_lengths = {
            self.blade_joint_source_id: 1,
            self.blade_tip_source_id: 3,
            self.blade_contact_source_id(
                "left"
            ): _CONTACT_VALUE_COUNT,
            self.blade_contact_source_id(
                "right"
            ): _CONTACT_VALUE_COUNT,
            self.bridge_topology_source_id: 1,
            self.guard_source_id: 1,
            self.alignment_source_id: 6,
        }
        for side in _SIDES:
            expected_lengths.update(
                {
                    self.zone_source_id(
                        side,
                        "upper_contact",
                    ): _CONTACT_VALUE_COUNT,
                    self.zone_source_id(
                        side,
                        "lower_contact",
                    ): _CONTACT_VALUE_COUNT,
                    self.zone_source_id(side, "temperature"): 1,
                    self.zone_source_id(side, "impedance"): 1,
                    self.zone_source_id(
                        side,
                        "electrical",
                    ): _ELECTRICAL_VALUE_COUNT,
                    self.zone_source_id(
                        side,
                        "cohesive",
                    ): _COHESIVE_VALUE_COUNT,
                    self.zone_source_id(
                        side,
                        "vessel",
                    ): _VESSEL_VALUE_COUNT,
                }
            )
        for source_id, expected_length in expected_lengths.items():
            if len(measurements[source_id].value) != expected_length:
                raise ValueError(
                    f"seal/divide source {source_id!r} must contain "
                    f"{expected_length} values"
                )
            if not measurements[source_id].raw_sample_ids:
                raise ValueError(
                    f"seal/divide source {source_id!r} requires raw sample "
                    "identity"
                )
        for zone in (self.left, self.right):
            side = zone.side
            contact_specs = (
                (
                    "upper_contact",
                    zone.upper_contact_prim_path,
                    zone.upper_compression_attachment_prim_ids,
                ),
                (
                    "lower_contact",
                    zone.lower_contact_prim_path,
                    zone.lower_compression_attachment_prim_ids,
                ),
            )
            for quantity, contact_path, expected_attachments in contact_specs:
                item = measurements[
                    self.zone_source_id(side, quantity)
                ]
                expected_pair = f"{contact_path}|{zone.vessel_prim_path}"
                if (
                    item.target_prim_path != zone.vessel_prim_path
                    or item.contact_pair_id != expected_pair
                ):
                    raise ValueError(
                        f"{side} {quantity} is not bound to its registered "
                        "contact pair"
                    )
                live = _attachment_ids(
                    item.attachment_prim_ids,
                    f"{side} {quantity} attachments",
                )
                if live != item.attachment_prim_ids:
                    raise ValueError(
                        "attachment prim IDs must use canonical sorted order"
                    )
                if not set(live).issubset(expected_attachments):
                    raise ValueError(
                        f"{side} {quantity} contains an unregistered "
                        "compression attachment"
                    )
            for quantity in ("temperature", "impedance", "electrical"):
                item = measurements[
                    self.zone_source_id(side, quantity)
                ]
                if item.target_prim_path != zone.seal_band_prim_path:
                    raise ValueError(
                        f"{side} {quantity} is not bound to its seal band"
                    )
            cohesive = measurements[
                self.zone_source_id(side, "cohesive")
            ]
            if (
                cohesive.target_prim_path != zone.vessel_prim_path
                or cohesive.contact_pair_id
                != f"{zone.seal_band_prim_path}|{zone.vessel_prim_path}"
            ):
                raise ValueError(
                    f"{side} cohesive response is not bound to its seal pair"
                )
            if (
                f"cohesive_interface:{zone.cohesive_interface_id}"
                not in cohesive.raw_sample_ids
                or len(cohesive.raw_sample_ids) < 2
            ):
                raise ValueError(
                    f"{side} cohesive response requires its registered "
                    "interface identity and a raw mechanics sample ID"
                )
            live_seal = _attachment_ids(
                cohesive.attachment_prim_ids,
                f"{side} seal attachments",
            )
            if live_seal != cohesive.attachment_prim_ids:
                raise ValueError(
                    "seal attachment prim IDs must use canonical sorted order"
                )
            if not set(live_seal).issubset(zone.seal_attachment_prim_ids):
                raise ValueError(
                    f"{side} cohesive evidence contains an unregistered "
                    "seal attachment"
                )
            vessel = measurements[self.zone_source_id(side, "vessel")]
            if vessel.target_prim_path != zone.seal_band_prim_path:
                raise ValueError(
                    f"{side} vessel evidence is not bound to its seal band"
                )
            if (
                f"vessel_segment:{zone.vessel_segment_id}"
                not in vessel.raw_sample_ids
                or len(vessel.raw_sample_ids) < 2
            ):
                raise ValueError(
                    f"{side} vessel response requires its registered segment "
                    "identity and a raw mechanics sample ID"
                )
        for side in _SIDES:
            target_path = self.zone(side).vessel_prim_path
            blade_contact = measurements[
                self.blade_contact_source_id(side)
            ]
            if (
                blade_contact.target_prim_path != target_path
                or blade_contact.contact_pair_id
                != f"{self.blade_contact_prim_path}|{target_path}"
            ):
                raise ValueError(
                    f"blade contact evidence is not bound to the {side} "
                    "vessel wall"
                )
            if blade_contact.attachment_prim_ids:
                raise ValueError(
                    "blade contact samples must not carry bridge topology"
                )
        bridge_topology = measurements[self.bridge_topology_source_id]
        if bridge_topology.target_prim_path is not None:
            raise ValueError(
                "bridge topology must be read directly from its registered "
                "attachment scope"
            )
        live_bridge = _attachment_ids(
            bridge_topology.attachment_prim_ids,
            "active bridge attachment prim IDs",
        )
        if live_bridge != bridge_topology.attachment_prim_ids:
            raise ValueError(
                "bridge attachment prim IDs must use canonical sorted order"
            )
        if not set(live_bridge).issubset(
            self.bridge_attachment_prim_ids
        ):
            raise ValueError(
                "bridge evidence contains an unregistered attachment"
            )
        if bridge_topology.value[0] != float(len(live_bridge)):
            raise ValueError(
                "bridge topology count does not match active attachment IDs"
            )
        blade_joint = measurements[self.blade_joint_source_id]
        if blade_joint.target_prim_path != self.blade_prim_path:
            raise ValueError(
                "blade joint evidence is not tied to the registered blade body"
            )
        blade_tip = measurements[self.blade_tip_source_id]
        if blade_tip.target_prim_path != self.cut_plane_prim_path:
            raise ValueError(
                "blade-tip evidence is not tied to the registered cut plane"
            )
        guard = measurements[self.guard_source_id]
        if guard.target_prim_path != self.blade_guard_prim_path:
            raise ValueError(
                "guard joint evidence is not tied to the registered guard body"
            )
        alignment = measurements[self.alignment_source_id]
        if alignment.target_prim_path != self.vessel_center_prim_path:
            raise ValueError(
                "alignment evidence is not tied to the registered vessel "
                "center"
            )


@dataclass(frozen=True)
class SealDivideZoneSceneObservation:
    """Post-physics measurements for one seal zone."""

    physics_step: int
    simulation_time_s: float
    upper_contact: ContactSample
    lower_contact: ContactSample
    upper_contact_area_m2: float
    lower_contact_area_m2: float
    active_upper_compression_attachment_prim_ids: tuple[str, ...]
    active_lower_compression_attachment_prim_ids: tuple[str, ...]
    interface_temperature_c: float
    interface_impedance_ohm: float
    rms_voltage_v: float
    rms_current_a: float
    electrical_power_factor: float
    cohesive_interface: CohesiveInterface
    cohesive_response: CohesiveResponse
    vessel_observation: VesselObservation
    active_seal_attachment_prim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step <= 0:
            raise ValueError("physics_step must be a positive integer")
        object.__setattr__(self, "physics_step", physics_step)
        object.__setattr__(
            self,
            "simulation_time_s",
            _nonnegative(self.simulation_time_s, "simulation_time_s"),
        )
        if not isinstance(self.upper_contact, ContactSample):
            raise TypeError("upper_contact must be a shared ContactSample")
        if not isinstance(self.lower_contact, ContactSample):
            raise TypeError("lower_contact must be a shared ContactSample")
        if not isinstance(self.cohesive_interface, CohesiveInterface):
            raise TypeError(
                "cohesive_interface must be a shared CohesiveInterface"
            )
        if not isinstance(self.cohesive_response, CohesiveResponse):
            raise TypeError(
                "cohesive_response must be a shared CohesiveResponse"
            )
        if self.cohesive_interface.traction_only() is not self.cohesive_response:
            raise ValueError(
                "cohesive_response must be the interface's exact latest "
                "mechanics response"
            )
        if (
            self.cohesive_response.interface_id
            != self.cohesive_interface.interface_id
            or self.cohesive_response.body_a_component_id
            != self.cohesive_interface.body_a_component_id
            or self.cohesive_response.body_b_component_id
            != self.cohesive_interface.body_b_component_id
        ):
            raise ValueError(
                "cohesive response identity does not match its interface"
            )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(
                self.cohesive_response.normal_a_to_b,
                self.cohesive_interface.normal_a_to_b,
                strict=True,
            )
        ):
            raise ValueError(
                "cohesive response normal does not match its interface"
            )
        if not isinstance(self.vessel_observation, VesselObservation):
            raise TypeError(
                "vessel_observation must be a shared VesselObservation"
            )
        for name in (
            "upper_contact_area_m2",
            "lower_contact_area_m2",
            "interface_impedance_ohm",
            "rms_voltage_v",
            "rms_current_a",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "electrical_power_factor",
            _fraction(
                self.electrical_power_factor,
                "electrical_power_factor",
            ),
        )
        object.__setattr__(
            self,
            "interface_temperature_c",
            _finite(
                self.interface_temperature_c,
                "interface_temperature_c",
            ),
        )
        if self.interface_temperature_c < -273.15:
            raise ValueError(
                "interface_temperature_c cannot be below absolute zero"
            )
        for name in (
            "active_upper_compression_attachment_prim_ids",
            "active_lower_compression_attachment_prim_ids",
            "active_seal_attachment_prim_ids",
        ):
            object.__setattr__(
                self,
                name,
                _attachment_ids(getattr(self, name), name),
            )
        for status in (
            self.cohesive_response.parameter_status,
            self.vessel_observation.parameter_status,
        ):
            if not str(status).startswith("provisional_engineering"):
                raise ValueError(
                    "seal/divide shared mechanics must retain provisional "
                    "engineering parameter status"
                )


@dataclass(frozen=True)
class SealDivideKinematicsObservation:
    """Raw blade, guard, alignment, and bridge-topology measurements."""

    physics_step: int
    simulation_time_s: float
    blade_joint_position_m: float
    blade_tip_position_tool_m: tuple[float, float, float]
    left_blade_contact: ContactSample
    right_blade_contact: ContactSample
    left_blade_contact_area_m2: float
    right_blade_contact_area_m2: float
    guard_joint_position_m: float
    tissue_center_reference_position_w_m: tuple[float, float, float]
    vessel_center_position_w_m: tuple[float, float, float]
    active_bridge_attachment_prim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step <= 0:
            raise ValueError("physics_step must be a positive integer")
        object.__setattr__(self, "physics_step", physics_step)
        object.__setattr__(
            self,
            "simulation_time_s",
            _nonnegative(self.simulation_time_s, "simulation_time_s"),
        )
        object.__setattr__(
            self,
            "blade_joint_position_m",
            _finite(
                self.blade_joint_position_m,
                "blade_joint_position_m",
            ),
        )
        object.__setattr__(
            self,
            "guard_joint_position_m",
            _finite(
                self.guard_joint_position_m,
                "guard_joint_position_m",
            ),
        )
        for name in (
            "blade_tip_position_tool_m",
            "tissue_center_reference_position_w_m",
            "vessel_center_position_w_m",
        ):
            object.__setattr__(
                self,
                name,
                _vector3(getattr(self, name), name),
            )
        for name in ("left_blade_contact", "right_blade_contact"):
            if not isinstance(getattr(self, name), ContactSample):
                raise TypeError(
                    f"{name} must be a shared ContactSample"
                )
        for name in (
            "left_blade_contact_area_m2",
            "right_blade_contact_area_m2",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "active_bridge_attachment_prim_ids",
            _attachment_ids(
                self.active_bridge_attachment_prim_ids,
                "active_bridge_attachment_prim_ids",
            ),
        )


@dataclass(frozen=True)
class SealDivideContactSample:
    """One contact sample decoded from a validated evidence envelope."""

    measurement: SceneMeasurement

    def __post_init__(self) -> None:
        if len(self.measurement.value) != _CONTACT_VALUE_COUNT:
            raise ValueError(
                f"contact evidence must contain {_CONTACT_VALUE_COUNT} values"
            )
        _nonnegative(self.normal_force_n, "normal_force_n")
        _nonnegative(self.contact_area_m2, "contact_area_m2")
        _vector3(self.point_w_m, "contact point")
        normal = _vector3(
            self.normal_a_to_b_w,
            "contact normal",
        )
        if not math.isclose(
            _norm(normal),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("contact normal evidence must be unit length")
        _finite(self.separation_m, "separation_m")
        _vector3(
            self.tangential_force_w_n,
            "tangential_force_w_n",
        )
        _vector3(
            self.relative_tangential_velocity_w_m_s,
            "relative_tangential_velocity_w_m_s",
        )
        if self.normal_force_n > 0.0 and self.contact_area_m2 <= 0.0:
            raise ValueError(
                "positive contact force requires positive contact area"
            )
        for vector, label in (
            (self.tangential_force_w_n, "tangential force"),
            (
                self.relative_tangential_velocity_w_m_s,
                "tangential velocity",
            ),
        ):
            normal_component = sum(
                left * right
                for left, right in zip(
                    vector,
                    normal,
                    strict=True,
                )
            )
            if not math.isclose(
                normal_component,
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    f"{label} evidence must lie in the contact tangent plane"
                )

    @property
    def point_w_m(self) -> tuple[float, float, float]:
        return tuple(self.measurement.value[:3])

    @property
    def normal_a_to_b_w(self) -> tuple[float, float, float]:
        return tuple(self.measurement.value[3:6])

    @property
    def separation_m(self) -> float:
        return self.measurement.value[6]

    @property
    def normal_force_n(self) -> float:
        return self.measurement.value[7]

    @property
    def tangential_force_w_n(self) -> tuple[float, float, float]:
        return tuple(self.measurement.value[8:11])

    @property
    def relative_tangential_velocity_w_m_s(
        self,
    ) -> tuple[float, float, float]:
        return tuple(self.measurement.value[11:14])

    @property
    def slip_speed_m_s(self) -> float:
        return _norm(self.relative_tangential_velocity_w_m_s)

    @property
    def contact_area_m2(self) -> float:
        return self.measurement.value[14]

    @property
    def mean_pressure_pa(self) -> float:
        if self.contact_area_m2 <= 0.0:
            return 0.0
        return _finite(
            self.normal_force_n / self.contact_area_m2,
            "mean_pressure_pa",
        )

    @property
    def attachment_prim_ids(self) -> tuple[str, ...]:
        return self.measurement.attachment_prim_ids


@dataclass(frozen=True)
class SealDivideZoneMechanicsSample:
    """One stump zone derived only from a validated scene envelope."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveSealDivideSceneSources
    side: str

    def __post_init__(self) -> None:
        self.sources.zone(self.side)
        self.sources.validate_envelope(self.envelope)
        if (
            _finite(
                self.interface_temperature_c,
                "interface_temperature_c",
            )
            < -273.15
        ):
            raise ValueError(
                "interface_temperature_c cannot be below absolute zero"
            )
        _nonnegative(self.interface_impedance_ohm, "interface_impedance_ohm")
        _nonnegative(self.rms_voltage_v, "rms_voltage_v")
        _nonnegative(self.rms_current_a, "rms_current_a")
        _fraction(
            self.electrical_power_factor,
            "electrical_power_factor",
        )
        _nonnegative(self.compression_force_n, "compression_force_n")
        _nonnegative(
            self.maximum_contact_force_n,
            "maximum_contact_force_n",
        )
        _fraction(
            self.compression_force_imbalance_fraction,
            "compression_force_imbalance_fraction",
        )
        _nonnegative(
            self.compression_pressure_pa,
            "compression_pressure_pa",
        )
        _fraction(self.cohesive_damage_fraction, "cohesive_damage_fraction")
        _fraction(self.wall_damage_fraction, "wall_damage_fraction")
        cohesive = self._cohesive
        _positive(cohesive[0], "cohesive_contact_area_m2")
        for index, label in (
            (1, "cohesive_pressure_pa"),
            (2, "cohesive_normal_traction_pa"),
            (6, "cohesive_resultant_force_n"),
            (9, "cohesive_stored_energy_j_m2"),
            (10, "cohesive_dissipated_energy_j_m2"),
        ):
            _nonnegative(cohesive[index], label)
        _vector3(tuple(cohesive[3:6]), "cohesive_shear_traction_pa")
        if cohesive[8] not in (0.0, 1.0):
            raise ValueError("cohesive failed evidence must be 0 or 1")
        vessel = self._vessel
        for index, label in (
            (0, "residual_defect_area_m2"),
            (3, "inlet_flow_m3_s"),
            (4, "measured_flow_m3_s"),
            (5, "leak_flow_m3_s"),
            (11, "vessel_contact_area_m2"),
            (12, "vessel_contact_pressure_pa"),
        ):
            _nonnegative(vessel[index], label)
        _finite(vessel[1], "upstream_pressure_pa")
        _finite(vessel[2], "downstream_pressure_pa")
        _finite(vessel[6], "storage_flow_m3_s")
        _finite(vessel[7], "mass_balance_residual_m3_s")
        _positive(vessel[8], "lumen_area_m2")
        _fraction(vessel[9], "occlusion_fraction")
        _fraction(vessel[10], "wall_damage_fraction")

    def _zone_measurement(self, quantity: str) -> SceneMeasurement:
        return _measurement(
            self.envelope,
            self.sources.zone_source_id(self.side, quantity),
        )

    @property
    def physics_step(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def simulation_time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def upper_contact(self) -> SealDivideContactSample:
        return SealDivideContactSample(
            self._zone_measurement("upper_contact")
        )

    @property
    def lower_contact(self) -> SealDivideContactSample:
        return SealDivideContactSample(
            self._zone_measurement("lower_contact")
        )

    @property
    def compression_force_n(self) -> float:
        return min(
            self.upper_contact.normal_force_n,
            self.lower_contact.normal_force_n,
        )

    @property
    def maximum_contact_force_n(self) -> float:
        return max(
            self.upper_contact.normal_force_n,
            self.lower_contact.normal_force_n,
        )

    @property
    def compression_force_imbalance_fraction(self) -> float:
        maximum = self.maximum_contact_force_n
        if maximum <= 0.0:
            return 0.0
        return (
            maximum - self.compression_force_n
        ) / maximum

    @property
    def compression_attached(self) -> bool:
        zone = self.sources.zone(self.side)
        return (
            self.upper_contact.attachment_prim_ids
            == zone.upper_compression_attachment_prim_ids
            and self.lower_contact.attachment_prim_ids
            == zone.lower_compression_attachment_prim_ids
        )

    @property
    def compression_contact_area_m2(self) -> float:
        return (
            self.upper_contact.contact_area_m2
            + self.lower_contact.contact_area_m2
        )

    @property
    def compression_pressure_pa(self) -> float:
        return min(
            self.upper_contact.mean_pressure_pa,
            self.lower_contact.mean_pressure_pa,
        )

    @property
    def maximum_slip_speed_m_s(self) -> float:
        return max(
            self.upper_contact.slip_speed_m_s,
            self.lower_contact.slip_speed_m_s,
        )

    @property
    def interface_temperature_c(self) -> float:
        return self._zone_measurement("temperature").value[0]

    @property
    def interface_impedance_ohm(self) -> float:
        return self._zone_measurement("impedance").value[0]

    @property
    def rms_voltage_v(self) -> float:
        return self._zone_measurement("electrical").value[0]

    @property
    def rms_current_a(self) -> float:
        return self._zone_measurement("electrical").value[1]

    @property
    def electrical_power_factor(self) -> float:
        return self._zone_measurement("electrical").value[2]

    @property
    def measured_power_w(self) -> float:
        return _finite(
            (
                self.rms_voltage_v
                * self.rms_current_a
                * self.electrical_power_factor
            ),
            "measured_power_w",
        )

    @property
    def _cohesive(self) -> tuple[float, ...]:
        return self._zone_measurement("cohesive").value

    @property
    def cohesive_contact_area_m2(self) -> float:
        return self._cohesive[0]

    @property
    def cohesive_pressure_pa(self) -> float:
        return self._cohesive[1]

    @property
    def cohesive_interface_traction_pa(self) -> float:
        return math.sqrt(
            self._cohesive[2] ** 2
            + sum(value * value for value in self._cohesive[3:6])
        )

    @property
    def cohesive_resultant_force_n(self) -> float:
        return self._cohesive[6]

    @property
    def cohesive_damage_fraction(self) -> float:
        return self._cohesive[7]

    @property
    def cohesive_failed(self) -> bool:
        return bool(round(self._cohesive[8]))

    @property
    def cohesive_dissipated_energy_j_m2(self) -> float:
        return self._cohesive[10]

    @property
    def seal_attachment_prim_ids(self) -> tuple[str, ...]:
        return self._zone_measurement(
            "cohesive"
        ).attachment_prim_ids

    @property
    def seal_attached(self) -> bool:
        return (
            self.seal_attachment_prim_ids
            == self.sources.zone(self.side).seal_attachment_prim_ids
        )

    @property
    def seal_integrity_fraction(self) -> float:
        if self.cohesive_failed or not self.seal_attached:
            return 0.0
        return max(0.0, 1.0 - self.cohesive_damage_fraction)

    @property
    def _vessel(self) -> tuple[float, ...]:
        return self._zone_measurement("vessel").value

    @property
    def residual_defect_area_m2(self) -> float:
        return self._vessel[0]

    @property
    def upstream_pressure_pa(self) -> float:
        return self._vessel[1]

    @property
    def downstream_pressure_pa(self) -> float:
        return self._vessel[2]

    @property
    def measured_flow_ml_min(self) -> float:
        return self._vessel[4] * 60.0 * 1.0e6

    @property
    def observed_leak_ml_min(self) -> float:
        return self._vessel[5] * 60.0 * 1.0e6

    @property
    def lumen_area_m2(self) -> float:
        return self._vessel[8]

    @property
    def occlusion_fraction(self) -> float:
        return self._vessel[9]

    @property
    def wall_damage_fraction(self) -> float:
        return self._vessel[10]

    @property
    def vessel_contact_area_m2(self) -> float:
        return self._vessel[11]

    @property
    def vessel_contact_pressure_pa(self) -> float:
        return self._vessel[12]

    @property
    def inlet_flow_ml_min(self) -> float:
        return self._vessel[3] * 60.0 * 1.0e6

    @property
    def storage_flow_ml_min(self) -> float:
        return self._vessel[6] * 60.0 * 1.0e6

    @property
    def mass_balance_residual_ml_min(self) -> float:
        return self._vessel[7] * 60.0 * 1.0e6

    @property
    def evidence_digest_sha256(self) -> str:
        return self.sources.bind_envelope_digest(
            self.envelope.digest_sha256
        )


@dataclass(frozen=True)
class SealDivideKinematicsSample:
    """Blade, guard, centering, and bridge topology from one envelope."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveSealDivideSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        _finite(self.blade_joint_position_m, "blade_joint_position_m")
        _finite(self.guard_joint_position_m, "guard_joint_position_m")
        _vector3(
            self.blade_tip_position_tool_m,
            "blade_tip_position_tool_m",
        )
        self.left_blade_contact
        self.right_blade_contact
        _fraction(self.bridge_release_fraction, "bridge_release_fraction")
        _vector3(
            self.tissue_center_reference_position_w_m,
            "tissue_center_reference_position_w_m",
        )
        _vector3(
            self.vessel_center_position_w_m,
            "vessel_center_position_w_m",
        )

    @property
    def blade_joint_position_m(self) -> float:
        return _measurement(
            self.envelope,
            self.sources.blade_joint_source_id,
        ).value[0]

    @property
    def blade_tip_position_tool_m(self) -> tuple[float, float, float]:
        return tuple(
            _measurement(
                self.envelope,
                self.sources.blade_tip_source_id,
            ).value
        )

    @property
    def blade_tip_expected_position_tool_m(
        self,
    ) -> tuple[float, float, float]:
        displacement = (
            self.blade_joint_position_m
            - self.sources.blade_start_position_m
        )
        return tuple(
            origin + axis * displacement
            for origin, axis in zip(
                self.sources.blade_tip_start_position_tool_m,
                self.sources.blade_travel_axis_tool,
                strict=True,
            )
        )

    @property
    def blade_tip_position_error_m(self) -> float:
        return _norm(
            tuple(
                observed - expected
                for observed, expected in zip(
                    self.blade_tip_position_tool_m,
                    self.blade_tip_expected_position_tool_m,
                    strict=True,
                )
            )
        )

    @property
    def blade_tip_position_consistent(self) -> bool:
        return (
            self.blade_tip_position_error_m
            <= self.sources.blade_tip_consistency_tolerance_m
        )

    def blade_contact(self, side: str) -> SealDivideContactSample:
        self.sources.zone(side)
        return SealDivideContactSample(
            _measurement(
                self.envelope,
                self.sources.blade_contact_source_id(side),
            )
        )

    @property
    def left_blade_contact(self) -> SealDivideContactSample:
        return self.blade_contact("left")

    @property
    def right_blade_contact(self) -> SealDivideContactSample:
        return self.blade_contact("right")

    @property
    def blade_contacts(
        self,
    ) -> tuple[SealDivideContactSample, SealDivideContactSample]:
        return (self.left_blade_contact, self.right_blade_contact)

    @property
    def blade_progress(self) -> float:
        start = self.sources.blade_start_position_m
        end = self.sources.blade_end_position_m
        return max(
            0.0,
            min(
                1.0,
                (self.blade_joint_position_m - start) / (end - start),
            ),
        )

    @property
    def blade_position_within_limits(self) -> bool:
        lower = min(
            self.sources.blade_start_position_m,
            self.sources.blade_end_position_m,
        )
        upper = max(
            self.sources.blade_start_position_m,
            self.sources.blade_end_position_m,
        )
        tolerance = self.sources.blade_position_tolerance_m
        return (
            lower - tolerance
            <= self.blade_joint_position_m
            <= upper + tolerance
        )

    @property
    def blade_in_contact(self) -> bool:
        return any(
            contact.normal_force_n
            >= self.sources.minimum_blade_contact_force_n
            and contact.separation_m
            <= self.sources.maximum_blade_contact_separation_m
            for contact in self.blade_contacts
        )

    @property
    def blade_contact_force_n(self) -> float:
        return sum(
            contact.normal_force_n for contact in self.blade_contacts
        )

    @property
    def active_bridge_attachment_prim_ids(self) -> tuple[str, ...]:
        return _measurement(
            self.envelope,
            self.sources.bridge_topology_source_id,
        ).attachment_prim_ids

    @property
    def bridge_release_fraction(self) -> float:
        total = len(self.sources.bridge_attachment_prim_ids)
        active = len(self.active_bridge_attachment_prim_ids)
        return (total - active) / total

    @property
    def division_complete(self) -> bool:
        return not self.active_bridge_attachment_prim_ids

    @property
    def guard_joint_position_m(self) -> float:
        return _measurement(
            self.envelope,
            self.sources.guard_source_id,
        ).value[0]

    @property
    def guard_retracted(self) -> bool:
        return (
            self.sources.guard_retraction_sign
            * self.guard_joint_position_m
            >= self.sources.guard_retracted_coordinate_m
        )

    @property
    def _alignment(self) -> tuple[float, ...]:
        return _measurement(
            self.envelope,
            self.sources.alignment_source_id,
        ).value

    @property
    def tissue_center_reference_position_w_m(
        self,
    ) -> tuple[float, float, float]:
        return tuple(self._alignment[:3])

    @property
    def vessel_center_position_w_m(self) -> tuple[float, float, float]:
        return tuple(self._alignment[3:6])

    @property
    def tissue_center_distance_m(self) -> float:
        return _norm(
            tuple(
                left - right
                for left, right in zip(
                    self.tissue_center_reference_position_w_m,
                    self.vessel_center_position_w_m,
                    strict=True,
                )
            )
        )


@dataclass(frozen=True)
class SealDivideSceneEvidence:
    """One coherent post-physics seal/divide workcell interval."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveSealDivideSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        self.left
        self.right
        self.kinematics

    @property
    def left(self) -> SealDivideZoneMechanicsSample:
        return SealDivideZoneMechanicsSample(
            self.envelope,
            self.sources,
            "left",
        )

    @property
    def right(self) -> SealDivideZoneMechanicsSample:
        return SealDivideZoneMechanicsSample(
            self.envelope,
            self.sources,
            "right",
        )

    @property
    def kinematics(self) -> SealDivideKinematicsSample:
        return SealDivideKinematicsSample(self.envelope, self.sources)

    @property
    def provenance(self) -> EvidenceProvenance:
        return self.envelope.provenance

    @property
    def physics_step(self) -> int:
        return self.provenance.physics_step

    @property
    def simulation_time_s(self) -> float:
        return self.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.provenance.dt_s

    @property
    def evidence_digest_sha256(self) -> str:
        return self.sources.bind_envelope_digest(
            self.envelope.digest_sha256
        )

    @property
    def envelope_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def registration_digest_sha256(self) -> str:
        return self.sources.registration_digest_sha256

    @property
    def episode_id(self) -> str:
        return self.provenance.episode_id

    @property
    def environment_id(self) -> str:
        return self.provenance.environment_id

    @property
    def topology_revision(self) -> str:
        return self.provenance.topology_revision

    @property
    def source(self) -> str:
        return (
            f"{self.provenance.adapter_id}:"
            f"{self.provenance.adapter_version}"
        )

    @property
    def tissue_centered(self) -> bool:
        return (
            self.kinematics.tissue_center_distance_m
            <= self.sources.centering_tolerance_m
            and self.left.compression_force_n
            >= self.sources.minimum_centering_force_n
            and self.right.compression_force_n
            >= self.sources.minimum_centering_force_n
            and self.left.compression_attached
            and self.right.compression_attached
        )

    def zone(self, side: str) -> SealDivideZoneMechanicsSample:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError(f"side must be one of {_SIDES!r}")


@runtime_checkable
class SealDivideSceneEvidenceSource(Protocol):
    """Environment-owned provider for one post-physics interval."""

    def sample_seal_divide_scene(self) -> SealDivideSceneEvidence:
        ...


class AdaptiveSealDivideSceneEvidence:
    """Collect immutable evidence from a registered workcell scene."""

    ADAPTER_ID = "dranmar.adaptive-seal-divide.scene-evidence"
    ADAPTER_VERSION = "3.0.0"

    def __init__(self, sources: AdaptiveSealDivideSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(sources.registered_sources)

    def _validate_contact_pair(
        self,
        *,
        sample: ContactSample,
        source_prim_path: str,
        target_prim_path: str,
        label: str,
    ) -> None:
        bodies = {sample.key.body_a, sample.key.body_b}
        expected = {source_prim_path, target_prim_path}
        if bodies != expected:
            raise ValueError(
                f"{label} contact bodies {sorted(bodies)!r} do not match "
                f"registered prims {sorted(expected)!r}"
            )

    def collect_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        left: SealDivideZoneSceneObservation,
        right: SealDivideZoneSceneObservation,
        kinematics: SealDivideKinematicsObservation,
        raw_sample_ids_by_source: Mapping[str, tuple[str, ...]],
    ) -> SealDivideSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        if not isinstance(left, SealDivideZoneSceneObservation):
            raise TypeError("left must be SealDivideZoneSceneObservation")
        if not isinstance(right, SealDivideZoneSceneObservation):
            raise TypeError("right must be SealDivideZoneSceneObservation")
        if not isinstance(kinematics, SealDivideKinematicsObservation):
            raise TypeError(
                "kinematics must be SealDivideKinematicsObservation"
            )
        for label, observation in (
            ("left zone", left),
            ("right zone", right),
            ("kinematics", kinematics),
        ):
            if (
                observation.physics_step != provenance.physics_step
                or not math.isclose(
                    observation.simulation_time_s,
                    provenance.simulation_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"{label} observation is not from the envelope's "
                    "physics interval"
                )
        raw_ids = dict(raw_sample_ids_by_source)
        if frozenset(raw_ids) != self.sources.expected_source_ids:
            raise ValueError(
                "raw sample identity must exactly match all registered "
                "seal/divide sources"
            )
        measurements: list[SceneMeasurement] = []
        for zone_sources, observation in (
            (self.sources.left, left),
            (self.sources.right, right),
        ):
            side = zone_sources.side
            if (
                observation.vessel_observation.scene_component_id
                != zone_sources.vessel_prim_path
            ):
                raise ValueError(
                    f"{side} vessel observation is not from the registered "
                    "scene component"
                )
            if (
                observation.vessel_observation.segment_id
                != zone_sources.vessel_segment_id
            ):
                raise ValueError(
                    f"{side} vessel observation is not from the registered "
                    "segment"
                )
            if (
                observation.vessel_observation.physics_step
                != provenance.physics_step
                or not math.isclose(
                    observation.vessel_observation.time_s,
                    provenance.simulation_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    observation.vessel_observation.dt_s,
                    provenance.dt_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    observation.vessel_observation.previous_time_s,
                    provenance.simulation_time_s - provenance.dt_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"{side} vessel observation is not from the envelope's "
                    "exact physics interval"
                )
            if (
                observation.cohesive_response.interface_id
                != zone_sources.cohesive_interface_id
            ):
                raise ValueError(
                    f"{side} cohesive response is not from the registered "
                    "interface"
                )
            if (
                observation.cohesive_interface.traction_only()
                is not observation.cohesive_response
            ):
                raise ValueError(
                    f"{side} cohesive response is no longer the exact latest "
                    "response of its registered shared mechanics interface"
                )
            if {
                observation.cohesive_response.body_a_component_id,
                observation.cohesive_response.body_b_component_id,
            } != {
                zone_sources.seal_band_prim_path,
                zone_sources.vessel_prim_path,
            }:
                raise ValueError(
                    f"{side} cohesive response bodies do not match the "
                    "registered seal-band/vessel pair"
                )
            if (
                observation.cohesive_response.step_index
                != provenance.physics_step
                or not math.isclose(
                    observation.cohesive_response.time_s,
                    provenance.simulation_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise ValueError(
                    f"{side} cohesive response is not from the envelope's "
                    "physics interval"
                )
            contact_specs = (
                (
                    "upper_contact",
                    zone_sources.upper_contact_prim_path,
                    observation.upper_contact,
                    observation.upper_contact_area_m2,
                    observation.active_upper_compression_attachment_prim_ids,
                    zone_sources.upper_compression_attachment_prim_ids,
                ),
                (
                    "lower_contact",
                    zone_sources.lower_contact_prim_path,
                    observation.lower_contact,
                    observation.lower_contact_area_m2,
                    observation.active_lower_compression_attachment_prim_ids,
                    zone_sources.lower_compression_attachment_prim_ids,
                ),
            )
            for (
                quantity,
                source_path,
                contact,
                area,
                active_attachments,
                expected_attachments,
            ) in contact_specs:
                self._validate_contact_pair(
                    sample=contact,
                    source_prim_path=source_path,
                    target_prim_path=zone_sources.vessel_prim_path,
                    label=f"{side} {quantity}",
                )
                if not set(active_attachments).issubset(
                    expected_attachments
                ):
                    raise ValueError(
                        f"{side} {quantity} includes an unregistered "
                        "attachment prim"
                    )
                measurements.append(
                    SceneMeasurement(
                        source_id=self.sources.zone_source_id(
                            side,
                            quantity,
                        ),
                        value=_contact_values(
                            contact,
                            area,
                            source_prim_path=source_path,
                            target_prim_path=zone_sources.vessel_prim_path,
                        ),
                        source_prim_path=source_path,
                        target_prim_path=zone_sources.vessel_prim_path,
                        contact_pair_id=(
                            f"{source_path}|"
                            f"{zone_sources.vessel_prim_path}"
                        ),
                        attachment_prim_ids=active_attachments,
                        raw_sample_ids=_raw_ids(
                            raw_ids,
                            self.sources.zone_source_id(
                                side,
                                quantity,
                            ),
                        ),
                    )
                )
            for quantity, value, source_path in (
                (
                    "temperature",
                    (observation.interface_temperature_c,),
                    zone_sources.temperature_sensor_prim_path,
                ),
                (
                    "impedance",
                    (observation.interface_impedance_ohm,),
                    zone_sources.impedance_sensor_prim_path,
                ),
                (
                    "electrical",
                    (
                        observation.rms_voltage_v,
                        observation.rms_current_a,
                        observation.electrical_power_factor,
                    ),
                    zone_sources.electrical_sensor_prim_path,
                ),
            ):
                measurements.append(
                    SceneMeasurement(
                        source_id=self.sources.zone_source_id(
                            side,
                            quantity,
                        ),
                        value=value,
                        source_prim_path=source_path,
                        target_prim_path=zone_sources.seal_band_prim_path,
                        raw_sample_ids=_raw_ids(
                            raw_ids,
                            self.sources.zone_source_id(
                                side,
                                quantity,
                            ),
                        ),
                    )
                )
            if not set(
                observation.active_seal_attachment_prim_ids
            ).issubset(zone_sources.seal_attachment_prim_ids):
                raise ValueError(
                    f"{side} cohesive evidence includes an unregistered "
                    "attachment prim"
                )
            measurements.extend(
                (
                    SceneMeasurement(
                        source_id=self.sources.zone_source_id(
                            side,
                            "cohesive",
                        ),
                        value=_cohesive_values(
                            observation.cohesive_response
                        ),
                        source_prim_path=zone_sources.seal_band_prim_path,
                        target_prim_path=zone_sources.vessel_prim_path,
                        contact_pair_id=(
                            f"{zone_sources.seal_band_prim_path}|"
                            f"{zone_sources.vessel_prim_path}"
                        ),
                        attachment_prim_ids=(
                            observation.active_seal_attachment_prim_ids
                        ),
                        raw_sample_ids=_raw_ids(
                            raw_ids,
                            self.sources.zone_source_id(
                                side,
                                "cohesive",
                            ),
                            identity_marker=(
                                "cohesive_interface:"
                                f"{zone_sources.cohesive_interface_id}"
                            ),
                        ),
                    ),
                    SceneMeasurement(
                        source_id=self.sources.zone_source_id(
                            side,
                            "vessel",
                        ),
                        value=_vessel_values(
                            observation.vessel_observation
                        ),
                        source_prim_path=zone_sources.vessel_prim_path,
                        target_prim_path=zone_sources.seal_band_prim_path,
                        raw_sample_ids=_raw_ids(
                            raw_ids,
                            self.sources.zone_source_id(
                                side,
                                "vessel",
                            ),
                            identity_marker=(
                                "vessel_segment:"
                                f"{zone_sources.vessel_segment_id}"
                            ),
                        ),
                    ),
                )
            )
        for side, contact, area in (
            (
                "left",
                kinematics.left_blade_contact,
                kinematics.left_blade_contact_area_m2,
            ),
            (
                "right",
                kinematics.right_blade_contact,
                kinematics.right_blade_contact_area_m2,
            ),
        ):
            target_path = self.sources.zone(side).vessel_prim_path
            self._validate_contact_pair(
                sample=contact,
                source_prim_path=self.sources.blade_contact_prim_path,
                target_prim_path=target_path,
                label=f"blade/{side} vessel",
            )
            measurements.append(
                SceneMeasurement(
                    source_id=self.sources.blade_contact_source_id(side),
                    value=_contact_values(
                        contact,
                        area,
                        source_prim_path=(
                            self.sources.blade_contact_prim_path
                        ),
                        target_prim_path=target_path,
                    ),
                    source_prim_path=self.sources.blade_contact_prim_path,
                    target_prim_path=target_path,
                    contact_pair_id=(
                        f"{self.sources.blade_contact_prim_path}|"
                        f"{target_path}"
                    ),
                    raw_sample_ids=_raw_ids(
                        raw_ids,
                        self.sources.blade_contact_source_id(side),
                    ),
                )
            )
        if not set(
            kinematics.active_bridge_attachment_prim_ids
        ).issubset(self.sources.bridge_attachment_prim_ids):
            raise ValueError(
                "blade evidence includes an unregistered bridge attachment"
            )
        measurements.extend(
            (
                SceneMeasurement(
                    source_id=self.sources.blade_joint_source_id,
                    value=(kinematics.blade_joint_position_m,),
                    source_prim_path=self.sources.blade_joint_prim_path,
                    target_prim_path=self.sources.blade_prim_path,
                    raw_sample_ids=_raw_ids(
                        raw_ids,
                        self.sources.blade_joint_source_id,
                    ),
                ),
                SceneMeasurement(
                    source_id=self.sources.blade_tip_source_id,
                    value=kinematics.blade_tip_position_tool_m,
                    source_prim_path=self.sources.blade_tip_prim_path,
                    target_prim_path=self.sources.cut_plane_prim_path,
                    raw_sample_ids=_raw_ids(
                        raw_ids,
                        self.sources.blade_tip_source_id,
                    ),
                ),
                SceneMeasurement(
                    source_id=self.sources.bridge_topology_source_id,
                    value=(
                        float(
                            len(
                                kinematics.active_bridge_attachment_prim_ids
                            )
                        ),
                    ),
                    source_prim_path=self.sources.bridge_topology_prim_path,
                    attachment_prim_ids=(
                        kinematics.active_bridge_attachment_prim_ids
                    ),
                    raw_sample_ids=_raw_ids(
                        raw_ids,
                        self.sources.bridge_topology_source_id,
                    ),
                ),
                SceneMeasurement(
                    source_id=self.sources.guard_source_id,
                    value=(kinematics.guard_joint_position_m,),
                    source_prim_path=(
                        self.sources.blade_guard_joint_prim_path
                    ),
                    target_prim_path=self.sources.blade_guard_prim_path,
                    raw_sample_ids=_raw_ids(
                        raw_ids,
                        self.sources.guard_source_id,
                    ),
                ),
                SceneMeasurement(
                    source_id=self.sources.alignment_source_id,
                    value=(
                        *kinematics.tissue_center_reference_position_w_m,
                        *kinematics.vessel_center_position_w_m,
                    ),
                    source_prim_path=(
                        self.sources.tissue_center_reference_prim_path
                    ),
                    target_prim_path=self.sources.vessel_center_prim_path,
                    raw_sample_ids=_raw_ids(
                        raw_ids,
                        self.sources.alignment_source_id,
                    ),
                ),
            )
        )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=tuple(measurements),
        )
        self.registry.validate(envelope)
        return SealDivideSceneEvidence(envelope, self.sources)


@dataclass
class SealDivideEvidenceCursor:
    """Reject replay, cross-episode evidence, and clock regressions."""

    episode_id: str | None = field(default=None, init=False)
    environment_id: str | None = field(default=None, init=False)
    topology_revision: str | None = field(default=None, init=False)
    registration_digest_sha256: str | None = field(
        default=None,
        init=False,
    )
    last_physics_step: int = field(default=-1, init=False)
    last_simulation_time_s: float = field(default=-1.0, init=False)
    last_evidence_digest_sha256: str | None = field(
        default=None,
        init=False,
    )

    def validate(
        self,
        evidence: SealDivideSceneEvidence,
    ) -> SealDivideSceneEvidence:
        """Validate the next interval without mutating replay state."""

        if not isinstance(evidence, SealDivideSceneEvidence):
            raise TypeError(
                "seal/divide cursor requires SealDivideSceneEvidence"
            )
        provenance = evidence.provenance
        if self.episode_id is not None and (
            provenance.episode_id != self.episode_id
            or provenance.environment_id != self.environment_id
            or provenance.topology_revision != self.topology_revision
            or evidence.registration_digest_sha256
            != self.registration_digest_sha256
        ):
            raise ValueError(
                "seal/divide evidence changed episode, environment, "
                "topology, or source registration without a cursor reset"
            )
        if provenance.physics_step <= self.last_physics_step:
            raise ValueError(
                "seal/divide evidence must use a strictly increasing "
                "physics step"
            )
        if (
            self.last_physics_step >= 0
            and provenance.physics_step != self.last_physics_step + 1
        ):
            raise ValueError(
                "seal/divide evidence must consume every consecutive "
                "physics interval"
            )
        if (
            provenance.simulation_time_s <= self.last_simulation_time_s
            and self.last_physics_step >= 0
        ):
            raise ValueError(
                "seal/divide evidence must use increasing simulation time"
            )
        if self.last_physics_step >= 0:
            interval_start_s = (
                provenance.simulation_time_s - provenance.dt_s
            )
            time_scale = max(
                abs(interval_start_s),
                abs(self.last_simulation_time_s),
                provenance.dt_s,
            )
            time_tolerance = max(
                64.0 * math.ulp(time_scale),
                1.0e-12 * time_scale,
            )
            if not math.isclose(
                interval_start_s,
                self.last_simulation_time_s,
                rel_tol=0.0,
                abs_tol=time_tolerance,
            ):
                raise ValueError(
                    "seal/divide evidence interval must begin at the "
                    "previous accepted interval endpoint"
                )
        digest = evidence.evidence_digest_sha256
        if digest == self.last_evidence_digest_sha256:
            raise ValueError("seal/divide evidence digest was replayed")
        return evidence

    def consume(
        self,
        evidence: SealDivideSceneEvidence,
    ) -> SealDivideSceneEvidence:
        """Commit one already valid interval to replay state."""

        evidence = self.validate(evidence)
        provenance = evidence.provenance
        if self.episode_id is None:
            self.episode_id = provenance.episode_id
            self.environment_id = provenance.environment_id
            self.topology_revision = provenance.topology_revision
            self.registration_digest_sha256 = (
                evidence.registration_digest_sha256
            )
        self.last_physics_step = provenance.physics_step
        self.last_simulation_time_s = provenance.simulation_time_s
        self.last_evidence_digest_sha256 = (
            evidence.evidence_digest_sha256
        )
        return evidence

    def reset(self) -> None:
        self.episode_id = None
        self.environment_id = None
        self.topology_revision = None
        self.registration_digest_sha256 = None
        self.last_physics_step = -1
        self.last_simulation_time_s = -1.0
        self.last_evidence_digest_sha256 = None


__all__ = [
    "AdaptiveSealDivideSceneEvidence",
    "AdaptiveSealDivideSceneSources",
    "SealDivideContactSample",
    "SealDivideEvidenceCursor",
    "SealDivideKinematicsObservation",
    "SealDivideKinematicsSample",
    "SealDivideSceneEvidence",
    "SealDivideSceneEvidenceSource",
    "SealDivideZoneMechanicsSample",
    "SealDivideZoneSceneObservation",
    "SealDivideZoneSources",
]
