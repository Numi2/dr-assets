# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound mechanics evidence for the Adaptive Hemostasis workcell.

The hemostasis policy never accepts caller-authored closure, retention, seal,
or success scalars.  A scene adapter snapshots shared vessel, clip, contact,
and cohesive mechanics into one immutable :class:`SceneEvidenceEnvelope`.
Controller-facing samples derive every value from that envelope after its
source IDs and exact prim paths have been checked against the workcell's
scene-time registration.

This is an engineering evidence boundary.  The provisional material and
hemodynamic parameters are not clinically validated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Protocol, runtime_checkable

from .scene_evidence import (
    EvidenceProvenance,
    SceneEvidenceEnvelope,
    SceneEvidenceRegistry,
    SceneEvidenceSource,
    SceneMeasurement,
)
from orbit.surgical.physics.clip import ClipObservation
from orbit.surgical.physics.cohesive import CohesiveInterface, CohesiveResponse
from orbit.surgical.physics.vessel import VesselObservation


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
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive(value: float, label: str) -> float:
    result = _nonnegative(value, label)
    if result == 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _fraction(value: float, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _vector3(
    value: tuple[float, float, float],
    label: str,
) -> tuple[float, float, float]:
    result = tuple(_finite(component, label) for component in value)
    if len(result) != 3:
        raise ValueError(f"{label} must contain three finite values")
    return result


def _unit_vector3(
    value: tuple[float, float, float],
    label: str,
) -> tuple[float, float, float]:
    vector = _vector3(value, label)
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude <= 1.0e-12:
        raise ValueError(f"{label} must be non-zero")
    return tuple(component / magnitude for component in vector)


def _raw_ids(
    raw_sample_ids_by_source: Mapping[str, tuple[str, ...]],
    source_id: str,
) -> tuple[str, ...]:
    values = tuple(
        sorted(
            _required_text(value, f"{source_id} raw sample ID")
            for value in raw_sample_ids_by_source.get(source_id, ())
        )
    )
    if not values:
        raise ValueError(
            f"hemostasis source {source_id!r} requires raw sample identity"
        )
    if len(values) != len(set(values)):
        raise ValueError(
            f"hemostasis source {source_id!r} has duplicate raw sample IDs"
        )
    return values


@dataclass(frozen=True)
class HemostasisClipSceneSource:
    """One deployed clip registered to one mechanics source."""

    clip_path: str
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clip_path",
            _scene_path(self.clip_path, "clip_path"),
        )
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )


@dataclass(frozen=True)
class HemostasisPatchSceneSource:
    """One deployed patch registered to one cohesive-response source."""

    patch_path: str
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "patch_path",
            _scene_path(self.patch_path, "patch_path"),
        )
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )


@dataclass(frozen=True)
class AdaptiveHemostasisSceneSources:
    """Exact prim registration for one vessel, tool, and deployed repairs."""

    workcell_id: str
    vessel_prim_path: str
    vessel_segment_id: str
    defect_frame_prim_path: str
    left_compression_prim_path: str
    right_compression_prim_path: str
    clip_sources: tuple[HemostasisClipSceneSource, ...] = ()
    patch_sources: tuple[HemostasisPatchSceneSource, ...] = ()
    calibration_profile_id: str = "dranmar-hemostasis-engineering-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workcell_id",
            _required_text(self.workcell_id, "workcell_id"),
        )
        object.__setattr__(
            self,
            "calibration_profile_id",
            _required_text(
                self.calibration_profile_id,
                "calibration_profile_id",
            ),
        )
        object.__setattr__(
            self,
            "vessel_segment_id",
            _required_text(self.vessel_segment_id, "vessel_segment_id"),
        )
        for name in (
            "vessel_prim_path",
            "defect_frame_prim_path",
            "left_compression_prim_path",
            "right_compression_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )
        object.__setattr__(self, "clip_sources", tuple(self.clip_sources))
        object.__setattr__(self, "patch_sources", tuple(self.patch_sources))
        clip_paths = [source.clip_path for source in self.clip_sources]
        patch_paths = [source.patch_path for source in self.patch_sources]
        source_ids = [
            self.vessel_observation_source_id,
            self.defect_geometry_source_id,
            self.left_compression_source_id,
            self.right_compression_source_id,
            *(source.source_id for source in self.clip_sources),
            *(source.source_id for source in self.patch_sources),
        ]
        if len(clip_paths) != len(set(clip_paths)):
            raise ValueError("clip prim paths must be unique")
        if len(patch_paths) != len(set(patch_paths)):
            raise ValueError("patch prim paths must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("hemostasis scene source IDs must be unique")

    @property
    def vessel_observation_source_id(self) -> str:
        return (
            f"{self.workcell_id}.vessel."
            f"{self.vessel_segment_id}.observation"
        )

    @property
    def defect_geometry_source_id(self) -> str:
        return f"{self.workcell_id}.defect_geometry"

    @property
    def left_compression_source_id(self) -> str:
        return f"{self.workcell_id}.left_compression"

    @property
    def right_compression_source_id(self) -> str:
        return f"{self.workcell_id}.right_compression"

    @property
    def registered_sources(self) -> tuple[SceneEvidenceSource, ...]:
        profile = self.calibration_profile_id
        sources = [
            SceneEvidenceSource(
                source_id=self.vessel_observation_source_id,
                prim_path=self.vessel_prim_path,
                quantity=(
                    "shared_vessel_residual_defect_pressure_flow_geometry"
                ),
                unit="SI_vector",
                coordinate_frame="vessel_lumen",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.defect_geometry_source_id,
                prim_path=self.defect_frame_prim_path,
                quantity="defect_centroid_and_normal",
                unit="m_and_unit_vector",
                coordinate_frame="world",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.left_compression_source_id,
                prim_path=self.left_compression_prim_path,
                quantity="normal_force_contact_area_separation",
                unit="N_m2_m",
                coordinate_frame="contact",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.right_compression_source_id,
                prim_path=self.right_compression_prim_path,
                quantity="normal_force_contact_area_separation",
                unit="N_m2_m",
                coordinate_frame="contact",
                calibration_profile_id=profile,
            ),
        ]
        sources.extend(
            SceneEvidenceSource(
                source_id=source.source_id,
                prim_path=source.clip_path,
                quantity="shared_clip_mechanics_observation",
                unit="SI_vector",
                coordinate_frame="clip_vessel_interface",
                calibration_profile_id=profile,
            )
            for source in self.clip_sources
        )
        sources.extend(
            SceneEvidenceSource(
                source_id=source.source_id,
                prim_path=source.patch_path,
                quantity="shared_cohesive_mechanics_observation",
                unit="SI_vector",
                coordinate_frame="patch_vessel_interface",
                calibration_profile_id=profile,
            )
            for source in self.patch_sources
        )
        return tuple(sources)

    @property
    def expected_source_ids(self) -> frozenset[str]:
        return frozenset(
            source.source_id for source in self.registered_sources
        )

    def clip_source_for(self, clip_path: str) -> HemostasisClipSceneSource:
        try:
            return next(
                source
                for source in self.clip_sources
                if source.clip_path == clip_path
            )
        except StopIteration as error:
            raise ValueError(
                f"clip path {clip_path!r} was not registered"
            ) from error

    def patch_source_for(self, patch_path: str) -> HemostasisPatchSceneSource:
        try:
            return next(
                source
                for source in self.patch_sources
                if source.patch_path == patch_path
            )
        except StopIteration as error:
            raise ValueError(
                f"patch path {patch_path!r} was not registered"
            ) from error

    def validate_envelope(self, envelope: SceneEvidenceEnvelope) -> None:
        provenance = envelope.provenance
        if provenance.adapter_id != AdaptiveHemostasisSceneEvidence.ADAPTER_ID:
            raise ValueError("hemostasis evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != AdaptiveHemostasisSceneEvidence.ADAPTER_VERSION
        ):
            raise ValueError("hemostasis evidence has the wrong adapter_version")
        registry = SceneEvidenceRegistry(self.registered_sources)
        registry.validate(envelope)
        measurements = {
            measurement.source_id: measurement
            for measurement in envelope.measurements
        }
        if frozenset(measurements) != self.expected_source_ids:
            raise ValueError(
                "hemostasis evidence source set does not match the registered "
                "workcell sources"
            )
        expected_lengths = {
            self.vessel_observation_source_id: 13,
            self.defect_geometry_source_id: 6,
            self.left_compression_source_id: 3,
            self.right_compression_source_id: 3,
            **{source.source_id: 13 for source in self.clip_sources},
            **{source.source_id: 9 for source in self.patch_sources},
        }
        for source_id, expected_length in expected_lengths.items():
            if len(measurements[source_id].value) != expected_length:
                raise ValueError(
                    f"hemostasis measurement {source_id!r} must contain "
                    f"{expected_length} values"
                )
            if not measurements[source_id].raw_sample_ids:
                raise ValueError(
                    f"hemostasis measurement {source_id!r} requires raw "
                    "sample identity"
                )
        defect_measurement = measurements[self.defect_geometry_source_id]
        if defect_measurement.target_prim_path != self.vessel_prim_path:
            raise ValueError(
                "defect geometry measurement is not tied to the registered "
                "vessel prim"
            )
        contact_source_ids = (
            self.left_compression_source_id,
            self.right_compression_source_id,
            *(source.source_id for source in self.clip_sources),
            *(source.source_id for source in self.patch_sources),
        )
        for source_id in contact_source_ids:
            measurement = measurements[source_id]
            if measurement.target_prim_path != self.vessel_prim_path:
                raise ValueError(
                    f"hemostasis measurement {source_id!r} is not tied to "
                    "the registered vessel prim"
                )
            if not measurement.contact_pair_id:
                raise ValueError(
                    f"hemostasis measurement {source_id!r} must identify "
                    "its contact pair"
                )
            expected_pair_id = (
                f"{measurement.source_prim_path}|"
                f"{self.vessel_prim_path}"
            )
            if measurement.contact_pair_id != expected_pair_id:
                raise ValueError(
                    f"hemostasis measurement {source_id!r} has the wrong "
                    "contact pair identity"
                )
        # Contact identity is carried by target_prim_path/contact_pair_id.
        # attachment_prim_ids are reserved for exact live attachment prims;
        # a contacted vessel path is not itself an attachment.


def _measurement(
    envelope: SceneEvidenceEnvelope,
    source_id: str,
) -> SceneMeasurement:
    try:
        return next(
            measurement
            for measurement in envelope.measurements
            if measurement.source_id == source_id
        )
    except StopIteration as error:
        raise ValueError(
            f"hemostasis evidence is missing source {source_id!r}"
        ) from error


@dataclass(frozen=True)
class CompressionSceneObservation:
    """Post-physics bilateral contact values, before envelope construction."""

    left_scene_component_id: str
    right_scene_component_id: str
    contact_component_ids: tuple[str, ...]
    physics_step: int
    simulation_time_s: float
    left_normal_force_n: float
    right_normal_force_n: float
    left_contact_area_m2: float
    right_contact_area_m2: float
    left_interface_separation_m: float = 0.0
    right_interface_separation_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_scene_component_id",
            _scene_path(
                self.left_scene_component_id,
                "left_scene_component_id",
            ),
        )
        object.__setattr__(
            self,
            "right_scene_component_id",
            _scene_path(
                self.right_scene_component_id,
                "right_scene_component_id",
            ),
        )
        object.__setattr__(
            self,
            "contact_component_ids",
            tuple(
                _scene_path(component_id, "contact_component_id")
                for component_id in self.contact_component_ids
            ),
        )
        if not self.contact_component_ids:
            raise ValueError(
                "compression observation must identify contacted components"
            )
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step <= 0:
            raise ValueError("physics_step must be a positive integer")
        object.__setattr__(self, "physics_step", physics_step)
        object.__setattr__(
            self,
            "simulation_time_s",
            _nonnegative(self.simulation_time_s, "simulation_time_s"),
        )
        for name in (
            "left_normal_force_n",
            "right_normal_force_n",
            "left_contact_area_m2",
            "right_contact_area_m2",
            "left_interface_separation_m",
            "right_interface_separation_m",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )


@dataclass(frozen=True)
class PatchSceneObservation:
    """Shared cohesive response plus scene-measured interface conditions."""

    scene_component_id: str
    contact_component_ids: tuple[str, ...]
    interface: CohesiveInterface
    response: CohesiveResponse
    nominal_area_m2: float
    interface_separation_m: float
    surface_wetness_fraction: float
    interface_temperature_c: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scene_component_id",
            _scene_path(self.scene_component_id, "scene_component_id"),
        )
        object.__setattr__(
            self,
            "contact_component_ids",
            tuple(
                _scene_path(component_id, "contact_component_id")
                for component_id in self.contact_component_ids
            ),
        )
        if not self.contact_component_ids:
            raise ValueError(
                "patch observation must identify contacted components"
            )
        if not isinstance(self.interface, CohesiveInterface):
            raise TypeError(
                "patch observation requires a shared CohesiveInterface"
            )
        if not isinstance(self.response, CohesiveResponse):
            raise TypeError(
                "patch observation requires a shared CohesiveResponse"
            )
        if self.interface.traction_only() is not self.response:
            raise ValueError(
                "patch response must be the interface's exact latest "
                "mechanics response"
            )
        if (
            self.response.interface_id != self.interface.interface_id
            or self.response.body_a_component_id
            != self.interface.body_a_component_id
            or self.response.body_b_component_id
            != self.interface.body_b_component_id
        ):
            raise ValueError(
                "patch response identity does not match its interface"
            )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(
                self.response.normal_a_to_b,
                self.interface.normal_a_to_b,
                strict=True,
            )
        ):
            raise ValueError(
                "patch response normal does not match its interface"
            )
        if self.response.interface_id != self.scene_component_id:
            raise ValueError(
                "cohesive response does not identify the patch interface"
            )
        response_pair = {
            self.response.body_a_component_id,
            self.response.body_b_component_id,
        }
        if self.scene_component_id not in response_pair or not (
            response_pair - {self.scene_component_id}
        ).intersection(self.contact_component_ids):
            raise ValueError(
                "cohesive response body identity does not match the patch "
                "contact components"
            )
        object.__setattr__(
            self,
            "nominal_area_m2",
            _positive(self.nominal_area_m2, "nominal_area_m2"),
        )
        object.__setattr__(
            self,
            "interface_separation_m",
            _nonnegative(
                self.interface_separation_m,
                "interface_separation_m",
            ),
        )
        object.__setattr__(
            self,
            "surface_wetness_fraction",
            _fraction(
                self.surface_wetness_fraction,
                "surface_wetness_fraction",
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


@dataclass(frozen=True)
class CompressionMechanicsSample:
    """Bilateral compression derived only from a validated envelope."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveHemostasisSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        for name in (
            "left_normal_force_n",
            "right_normal_force_n",
            "left_contact_area_m2",
            "right_contact_area_m2",
            "left_interface_separation_m",
            "right_interface_separation_m",
        ):
            _nonnegative(getattr(self, name), name)

    @property
    def _left(self) -> tuple[float, ...]:
        return _measurement(
            self.envelope,
            self.sources.left_compression_source_id,
        ).value

    @property
    def _right(self) -> tuple[float, ...]:
        return _measurement(
            self.envelope,
            self.sources.right_compression_source_id,
        ).value

    @property
    def left_normal_force_n(self) -> float:
        return self._left[0]

    @property
    def right_normal_force_n(self) -> float:
        return self._right[0]

    @property
    def left_contact_area_m2(self) -> float:
        return self._left[1]

    @property
    def right_contact_area_m2(self) -> float:
        return self._right[1]

    @property
    def left_interface_separation_m(self) -> float:
        return self._left[2]

    @property
    def right_interface_separation_m(self) -> float:
        return self._right[2]

    @property
    def step_index(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def left_mean_pressure_pa(self) -> float:
        if self.left_contact_area_m2 == 0.0:
            return 0.0
        return self.left_normal_force_n / self.left_contact_area_m2

    @property
    def right_mean_pressure_pa(self) -> float:
        if self.right_contact_area_m2 == 0.0:
            return 0.0
        return self.right_normal_force_n / self.right_contact_area_m2


@dataclass(frozen=True)
class VesselMechanicsSample:
    """Shared vessel observation derived only from a validated envelope."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveHemostasisSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        for name in (
            "upstream_pressure_pa",
            "downstream_pressure_pa",
            "residual_defect_area_m2",
            "lumen_area_m2",
            "measured_leak_flow_ml_min",
            "inlet_flow_m3_s",
            "outlet_flow_m3_s",
            "vessel_contact_area_m2",
            "vessel_contact_pressure_pa",
        ):
            _nonnegative(getattr(self, name), name)
        _fraction(self.wall_damage_fraction, "wall_damage_fraction")
        _fraction(self.occlusion_fraction, "occlusion_fraction")
        _finite(self.storage_flow_m3_s, "storage_flow_m3_s")
        _finite(
            self.mass_balance_residual_m3_s,
            "mass_balance_residual_m3_s",
        )
        _vector3(self.defect_centroid_w, "defect_centroid_w")
        normal = _unit_vector3(self.defect_normal_w, "defect_normal_w")
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-6)
            for left, right in zip(
                normal,
                self.defect_normal_w,
                strict=True,
            )
        ):
            raise ValueError("defect normal evidence must be unit length")

    @property
    def _vessel(self) -> tuple[float, ...]:
        return _measurement(
            self.envelope,
            self.sources.vessel_observation_source_id,
        ).value

    @property
    def _defect(self) -> tuple[float, ...]:
        return _measurement(
            self.envelope,
            self.sources.defect_geometry_source_id,
        ).value

    @property
    def step_index(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

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
    def measured_leak_flow_ml_min(self) -> float:
        return self._vessel[5] * 60.0 * 1.0e6

    @property
    def lumen_area_m2(self) -> float:
        return self._vessel[8]

    @property
    def wall_damage_fraction(self) -> float:
        return self._vessel[10]

    @property
    def inlet_flow_m3_s(self) -> float:
        return self._vessel[3]

    @property
    def outlet_flow_m3_s(self) -> float:
        return self._vessel[4]

    @property
    def storage_flow_m3_s(self) -> float:
        return self._vessel[6]

    @property
    def mass_balance_residual_m3_s(self) -> float:
        return self._vessel[7]

    @property
    def occlusion_fraction(self) -> float:
        return self._vessel[9]

    @property
    def vessel_contact_area_m2(self) -> float:
        return self._vessel[11]

    @property
    def vessel_contact_pressure_pa(self) -> float:
        return self._vessel[12]

    @property
    def defect_centroid_w(self) -> tuple[float, float, float]:
        return tuple(self._defect[:3])

    @property
    def defect_normal_w(self) -> tuple[float, float, float]:
        return tuple(self._defect[3:])

    @property
    def compression(self) -> CompressionMechanicsSample:
        return CompressionMechanicsSample(self.envelope, self.sources)

    @property
    def pressure_drop_pa(self) -> float:
        return max(0.0, self.upstream_pressure_pa - self.downstream_pressure_pa)


@dataclass(frozen=True)
class ClipMechanicsSample:
    """Shared clip mechanics derived only from its registered prim source."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveHemostasisSceneSources
    clip_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clip_path",
            _scene_path(self.clip_path, "clip_path"),
        )
        self.sources.validate_envelope(self.envelope)
        self.sources.clip_source_for(self.clip_path)
        for name in (
            "residual_gap_m",
            "formed_span_m",
            "retention_load_n",
            "retention_capacity_n",
            "contact_area_m2",
            "contact_pressure_pa",
            "interface_traction_pa",
            "max_relative_slip_speed_m_s",
        ):
            _nonnegative(getattr(self, name), name)
        _finite(self.plastic_curvature_1_m, "plastic_curvature_1_m")
        _fraction(self.damage_fraction, "damage_fraction")
        measured_magnitude = math.sqrt(
            sum(value * value for value in self.measured_tangential_force_n)
        )
        if not math.isclose(
            self.retention_load_n,
            measured_magnitude,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "retention load must equal the measured tangential-force "
                "vector magnitude"
            )

    @property
    def _values(self) -> tuple[float, ...]:
        source = self.sources.clip_source_for(self.clip_path)
        return _measurement(self.envelope, source.source_id).value

    @property
    def step_index(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def live_attachment_prim_ids(self) -> tuple[str, ...]:
        source = self.sources.clip_source_for(self.clip_path)
        return _measurement(
            self.envelope,
            source.source_id,
        ).attachment_prim_ids

    @property
    def residual_gap_m(self) -> float:
        return self._values[0]

    @property
    def formed_span_m(self) -> float:
        return self._values[1]

    @property
    def retention_load_n(self) -> float:
        return self._values[2]

    @property
    def contact_area_m2(self) -> float:
        return self._values[3]

    @property
    def contact_pressure_pa(self) -> float:
        return self._values[4]

    @property
    def interface_traction_pa(self) -> float:
        return self._values[5]

    @property
    def plastic_curvature_1_m(self) -> float:
        return self._values[6]

    @property
    def damage_fraction(self) -> float:
        return self._values[7]

    @property
    def retention_capacity_n(self) -> float:
        return self._values[8]

    @property
    def measured_tangential_force_n(
        self,
    ) -> tuple[float, float, float]:
        return tuple(self._values[9:12])

    @property
    def max_relative_slip_speed_m_s(self) -> float:
        return self._values[12]

    @property
    def retention_utilization(self) -> float:
        if self.retention_capacity_n == 0.0:
            return math.inf if self.retention_load_n > 0.0 else 0.0
        return self.retention_load_n / self.retention_capacity_n


@dataclass(frozen=True)
class PatchMechanicsSample:
    """Shared cohesive response derived only from its registered patch prim."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveHemostasisSceneSources
    patch_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "patch_path",
            _scene_path(self.patch_path, "patch_path"),
        )
        self.sources.validate_envelope(self.envelope)
        self.sources.patch_source_for(self.patch_path)
        for name in (
            "contact_area_m2",
            "mean_contact_pressure_pa",
            "interface_traction_n",
            "interface_separation_m",
        ):
            _nonnegative(getattr(self, name), name)
        _positive(self.nominal_area_m2, "nominal_area_m2")
        if self.contact_area_m2 > self.nominal_area_m2 * 1.001:
            raise ValueError("contact area cannot exceed nominal patch area")
        _fraction(
            self.surface_wetness_fraction,
            "surface_wetness_fraction",
        )
        _finite(self.interface_temperature_c, "interface_temperature_c")
        _fraction(
            self.cohesive_damage_fraction,
            "cohesive_damage_fraction",
        )
        if self._values[8] not in (0.0, 1.0):
            raise ValueError("cohesive failure evidence must be binary")

    @property
    def _values(self) -> tuple[float, ...]:
        source = self.sources.patch_source_for(self.patch_path)
        return _measurement(self.envelope, source.source_id).value

    @property
    def step_index(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def live_attachment_prim_ids(self) -> tuple[str, ...]:
        source = self.sources.patch_source_for(self.patch_path)
        return _measurement(
            self.envelope,
            source.source_id,
        ).attachment_prim_ids

    @property
    def contact_area_m2(self) -> float:
        return self._values[0]

    @property
    def nominal_area_m2(self) -> float:
        return self._values[1]

    @property
    def mean_contact_pressure_pa(self) -> float:
        return self._values[2]

    @property
    def interface_traction_n(self) -> float:
        return self._values[3]

    @property
    def interface_separation_m(self) -> float:
        return self._values[4]

    @property
    def surface_wetness_fraction(self) -> float:
        return self._values[5]

    @property
    def interface_temperature_c(self) -> float:
        return self._values[6]

    @property
    def cohesive_damage_fraction(self) -> float:
        return self._values[7]

    @property
    def cohesive_failed(self) -> bool:
        return bool(self._values[8])

    @property
    def contact_fraction(self) -> float:
        return min(1.0, self.contact_area_m2 / self.nominal_area_m2)


@dataclass(frozen=True)
class HemostasisSceneEvidence:
    """One validated post-physics interval for all hemostasis mechanics."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveHemostasisSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        VesselMechanicsSample(self.envelope, self.sources)
        for source in self.sources.clip_sources:
            ClipMechanicsSample(
                self.envelope,
                self.sources,
                source.clip_path,
            )
        for source in self.sources.patch_sources:
            PatchMechanicsSample(
                self.envelope,
                self.sources,
                source.patch_path,
            )

    @property
    def vessel(self) -> VesselMechanicsSample:
        return VesselMechanicsSample(self.envelope, self.sources)

    @property
    def clips(self) -> tuple[ClipMechanicsSample, ...]:
        return tuple(
            ClipMechanicsSample(
                self.envelope,
                self.sources,
                source.clip_path,
            )
            for source in self.sources.clip_sources
        )

    @property
    def patches(self) -> tuple[PatchMechanicsSample, ...]:
        return tuple(
            PatchMechanicsSample(
                self.envelope,
                self.sources,
                source.patch_path,
            )
            for source in self.sources.patch_sources
        )

    @property
    def provenance(self) -> EvidenceProvenance:
        return self.envelope.provenance

    @property
    def step_index(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def physics_step(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def simulation_time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def source(self) -> str:
        return self.envelope.provenance.adapter_id

    def clip_for(self, clip_path: str) -> ClipMechanicsSample | None:
        return next(
            (sample for sample in self.clips if sample.clip_path == clip_path),
            None,
        )

    def patch_for(self, patch_path: str) -> PatchMechanicsSample | None:
        return next(
            (
                sample
                for sample in self.patches
                if sample.patch_path == patch_path
            ),
            None,
        )


@runtime_checkable
class HemostasisSceneEvidenceSource(Protocol):
    """Single provenance-bearing entry point consumed by the sequence."""

    def sample_hemostasis_scene(self) -> HemostasisSceneEvidence:
        ...


class AdaptiveHemostasisSceneEvidence:
    """Snapshot shared mechanics into registered, immutable scene evidence."""

    ADAPTER_ID = "dranmar.adaptive-hemostasis.scene-evidence"
    ADAPTER_VERSION = "3.0.0"

    def __init__(self, sources: AdaptiveHemostasisSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(sources.registered_sources)

    def collect_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        vessel_observation: VesselObservation,
        compression_observation: CompressionSceneObservation,
        defect_centroid_w: tuple[float, float, float],
        defect_normal_w: tuple[float, float, float],
        raw_sample_ids_by_source: Mapping[
            str,
            tuple[str, ...],
        ],
        live_attachment_prim_ids_by_repair_path: Mapping[
            str,
            tuple[str, ...],
        ],
        clip_observations: Mapping[str, ClipObservation] | None = None,
        patch_observations: Mapping[str, PatchSceneObservation] | None = None,
    ) -> HemostasisSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        if not isinstance(vessel_observation, VesselObservation):
            raise TypeError(
                "vessel observation must come from shared VesselMechanics"
            )
        if (
            vessel_observation.scene_component_id
            != self.sources.vessel_prim_path
        ):
            raise ValueError(
                "shared vessel observation does not identify the registered "
                "vessel prim"
            )
        if (
            vessel_observation.segment_id
            != self.sources.vessel_segment_id
        ):
            raise ValueError(
                "shared vessel observation does not identify the registered "
                "vessel segment"
            )
        if (
            vessel_observation.physics_step != provenance.physics_step
            or not math.isclose(
                vessel_observation.time_s,
                provenance.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                vessel_observation.dt_s,
                provenance.dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                vessel_observation.previous_time_s,
                provenance.simulation_time_s - provenance.dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "shared vessel observation is not from the envelope's exact "
                "physics interval"
            )
        if not isinstance(
            compression_observation,
            CompressionSceneObservation,
        ):
            raise TypeError(
                "compression observation must be a "
                "CompressionSceneObservation"
            )
        if (
            compression_observation.left_scene_component_id
            != self.sources.left_compression_prim_path
            or compression_observation.right_scene_component_id
            != self.sources.right_compression_prim_path
        ):
            raise ValueError(
                "compression observation does not identify the registered "
                "compression prims"
            )
        if (
            self.sources.vessel_prim_path
            not in compression_observation.contact_component_ids
        ):
            raise ValueError(
                "compression observation is not tied to the registered "
                "vessel prim"
            )
        if (
            compression_observation.physics_step
            != provenance.physics_step
            or not math.isclose(
                compression_observation.simulation_time_s,
                provenance.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "compression observation is not from the envelope's physics "
                "interval"
            )
        clips = {} if clip_observations is None else dict(clip_observations)
        patches = (
            {} if patch_observations is None else dict(patch_observations)
        )
        raw_ids = dict(raw_sample_ids_by_source)
        live_attachments = {
            _scene_path(path, "repair attachment owner path"): tuple(
                sorted(
                    _scene_path(
                        attachment,
                        "live repair attachment prim ID",
                    )
                    for attachment in attachments
                )
            )
            for path, attachments in (
                live_attachment_prim_ids_by_repair_path.items()
            )
        }
        registered_clip_paths = {
            source.clip_path for source in self.sources.clip_sources
        }
        registered_patch_paths = {
            source.patch_path for source in self.sources.patch_sources
        }
        if set(clips) != registered_clip_paths:
            raise ValueError(
                "clip observations must exactly match registered clip prims"
            )
        if set(patches) != registered_patch_paths:
            raise ValueError(
                "patch observations must exactly match registered patch prims"
            )
        registered_repair_paths = (
            registered_clip_paths | registered_patch_paths
        )
        if set(live_attachments) != registered_repair_paths:
            raise ValueError(
                "live attachment evidence must exactly match registered "
                "clip and patch prims"
            )
        for repair_path, attachment_ids in live_attachments.items():
            if len(attachment_ids) != len(set(attachment_ids)):
                raise ValueError(
                    f"repair {repair_path!r} has duplicate live attachment "
                    "prim IDs"
                )
        for source in self.sources.clip_sources:
            observation = clips[source.clip_path]
            if not isinstance(observation, ClipObservation):
                raise TypeError(
                    "clip observations must come from shared ClipMechanics"
                )
            if observation.scene_component_id != source.clip_path:
                raise ValueError(
                    f"clip observation for {source.clip_path!r} does not "
                    "identify its registered clip prim"
                )
            if (
                observation.physics_step != provenance.physics_step
                or not math.isclose(
                    observation.time_s,
                    provenance.simulation_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    observation.dt_s,
                    provenance.dt_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    observation.previous_time_s,
                    provenance.simulation_time_s - provenance.dt_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"clip observation for {source.clip_path!r} is not from "
                    "the envelope's exact physics interval"
                )
            if (
                observation.contact_area_m2 > 0.0
                and self.sources.vessel_prim_path
                not in observation.contact_component_ids
            ):
                raise ValueError(
                    f"clip observation for {source.clip_path!r} is not tied "
                    "to the registered vessel prim"
                )
            if not set(observation.raw_contact_ids).issubset(
                set(_raw_ids(raw_ids, source.source_id))
            ):
                raise ValueError(
                    f"clip observation for {source.clip_path!r} has contact "
                    "records missing from the envelope raw-sample identity"
                )
        for source in self.sources.patch_sources:
            observation = patches[source.patch_path]
            if not isinstance(observation, PatchSceneObservation):
                raise TypeError(
                    "patch observations must be PatchSceneObservation values"
                )
            if observation.scene_component_id != source.patch_path:
                raise ValueError(
                    f"patch observation for {source.patch_path!r} does not "
                    "identify its registered patch prim"
                )
            if (
                self.sources.vessel_prim_path
                not in observation.contact_component_ids
            ):
                raise ValueError(
                    f"patch observation for {source.patch_path!r} is not tied "
                    "to the registered vessel prim"
                )
            if {
                observation.response.body_a_component_id,
                observation.response.body_b_component_id,
            } != {
                source.patch_path,
                self.sources.vessel_prim_path,
            }:
                raise ValueError(
                    f"patch response for {source.patch_path!r} is not bound "
                    "to the registered patch-vessel pair"
                )
            if (
                observation.response.step_index
                != provenance.physics_step
                or not math.isclose(
                    observation.response.time_s,
                    provenance.simulation_time_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise ValueError(
                    f"patch response for {source.patch_path!r} is not from "
                    "the envelope's physics interval"
                )
        unknown_raw_sources = set(raw_ids) - self.sources.expected_source_ids
        if unknown_raw_sources:
            raise ValueError(
                "raw sample IDs reference unregistered hemostasis sources: "
                f"{sorted(unknown_raw_sources)!r}"
            )

        defect_centroid = _vector3(
            defect_centroid_w,
            "defect_centroid_w",
        )
        defect_normal = _unit_vector3(
            defect_normal_w,
            "defect_normal_w",
        )
        vessel_values = (
            vessel_observation.residual_defect_area_m2,
            vessel_observation.upstream_pressure_pa,
            vessel_observation.downstream_pressure_pa,
            vessel_observation.inlet_flow_m3_s,
            vessel_observation.measured_flow_m3_s,
            vessel_observation.leak_flow_m3_s,
            vessel_observation.storage_flow_m3_s,
            vessel_observation.mass_balance_residual_m3_s,
            vessel_observation.lumen_area_m2,
            vessel_observation.occlusion_fraction,
            vessel_observation.wall_damage_fraction,
            vessel_observation.contact_area_m2,
            vessel_observation.contact_pressure_pa,
        )
        left_pair_id = (
            f"{self.sources.left_compression_prim_path}|"
            f"{self.sources.vessel_prim_path}"
        )
        right_pair_id = (
            f"{self.sources.right_compression_prim_path}|"
            f"{self.sources.vessel_prim_path}"
        )
        measurements = [
            SceneMeasurement(
                source_id=self.sources.vessel_observation_source_id,
                value=vessel_values,
                source_prim_path=self.sources.vessel_prim_path,
                raw_sample_ids=_raw_ids(
                    raw_ids,
                    self.sources.vessel_observation_source_id,
                ),
            ),
            SceneMeasurement(
                source_id=self.sources.defect_geometry_source_id,
                value=(*defect_centroid, *defect_normal),
                source_prim_path=self.sources.defect_frame_prim_path,
                target_prim_path=self.sources.vessel_prim_path,
                raw_sample_ids=_raw_ids(
                    raw_ids,
                    self.sources.defect_geometry_source_id,
                ),
            ),
            SceneMeasurement(
                source_id=self.sources.left_compression_source_id,
                value=(
                    compression_observation.left_normal_force_n,
                    compression_observation.left_contact_area_m2,
                    compression_observation.left_interface_separation_m,
                ),
                source_prim_path=self.sources.left_compression_prim_path,
                target_prim_path=self.sources.vessel_prim_path,
                contact_pair_id=left_pair_id,
                raw_sample_ids=_raw_ids(
                    raw_ids,
                    self.sources.left_compression_source_id,
                ),
            ),
            SceneMeasurement(
                source_id=self.sources.right_compression_source_id,
                value=(
                    compression_observation.right_normal_force_n,
                    compression_observation.right_contact_area_m2,
                    compression_observation.right_interface_separation_m,
                ),
                source_prim_path=self.sources.right_compression_prim_path,
                target_prim_path=self.sources.vessel_prim_path,
                contact_pair_id=right_pair_id,
                raw_sample_ids=_raw_ids(
                    raw_ids,
                    self.sources.right_compression_source_id,
                ),
            ),
        ]
        for source in self.sources.clip_sources:
            observation = clips[source.clip_path]
            measurements.append(
                SceneMeasurement(
                    source_id=source.source_id,
                    value=(
                        observation.residual_gap_m,
                        observation.formed_span_m,
                        observation.retention_load_n,
                        observation.contact_area_m2,
                        observation.contact_pressure_pa,
                        observation.interface_traction_pa,
                        observation.plastic_curvature_1_m,
                        observation.damage_fraction,
                        observation.retention_capacity_n,
                        *observation.measured_tangential_force_n,
                        observation.max_relative_slip_speed_m_s,
                    ),
                    source_prim_path=source.clip_path,
                    target_prim_path=self.sources.vessel_prim_path,
                    contact_pair_id=(
                        f"{source.clip_path}|"
                        f"{self.sources.vessel_prim_path}"
                    ),
                    attachment_prim_ids=live_attachments[
                        source.clip_path
                    ],
                    raw_sample_ids=_raw_ids(raw_ids, source.source_id),
                )
            )
        for source in self.sources.patch_sources:
            observation = patches[source.patch_path]
            response = observation.response
            measurements.append(
                SceneMeasurement(
                    source_id=source.source_id,
                    value=(
                        response.contact_area_m2,
                        observation.nominal_area_m2,
                        response.normal_pressure_pa,
                        response.resultant_force_n,
                        observation.interface_separation_m,
                        observation.surface_wetness_fraction,
                        observation.interface_temperature_c,
                        response.damage,
                        1.0 if response.failed else 0.0,
                    ),
                    source_prim_path=source.patch_path,
                    target_prim_path=self.sources.vessel_prim_path,
                    contact_pair_id=(
                        f"{source.patch_path}|"
                        f"{self.sources.vessel_prim_path}"
                    ),
                    attachment_prim_ids=live_attachments[
                        source.patch_path
                    ],
                    raw_sample_ids=_raw_ids(raw_ids, source.source_id),
                )
            )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=tuple(measurements),
        )
        self.registry.validate(envelope)
        return HemostasisSceneEvidence(
            envelope=envelope,
            sources=self.sources,
        )


@dataclass
class HemostasisEvidenceCursor:
    """Reject replayed, duplicated, or out-of-order scene evidence."""

    last_step_index: int = field(default=-1, init=False)
    last_time_s: float = field(default=-math.inf, init=False)
    last_digest_sha256: str | None = field(default=None, init=False)
    episode_id: str | None = field(default=None, init=False)
    environment_id: str | None = field(default=None, init=False)

    def consume(
        self,
        evidence: HemostasisSceneEvidence,
    ) -> HemostasisSceneEvidence:
        if not isinstance(evidence, HemostasisSceneEvidence):
            raise TypeError(
                "hemostasis controllers require HemostasisSceneEvidence"
            )
        provenance = evidence.envelope.provenance
        if self.episode_id is None:
            self.episode_id = provenance.episode_id
            self.environment_id = provenance.environment_id
        elif (
            provenance.episode_id != self.episode_id
            or provenance.environment_id != self.environment_id
        ):
            raise ValueError(
                "hemostasis evidence cannot switch episode or environment "
                "without a new cursor"
            )
        if evidence.digest_sha256 == self.last_digest_sha256:
            raise ValueError("hemostasis scene evidence digest was replayed")
        if evidence.step_index <= self.last_step_index:
            raise ValueError(
                "hemostasis scene evidence must advance monotonically: "
                f"last={self.last_step_index}, "
                f"received={evidence.step_index}"
            )
        if evidence.time_s <= self.last_time_s:
            raise ValueError(
                "hemostasis scene time must advance monotonically: "
                f"last={self.last_time_s}, received={evidence.time_s}"
            )
        self.last_step_index = evidence.step_index
        self.last_time_s = evidence.time_s
        self.last_digest_sha256 = evidence.digest_sha256
        return evidence


__all__ = [
    "AdaptiveHemostasisSceneEvidence",
    "AdaptiveHemostasisSceneSources",
    "ClipMechanicsSample",
    "CompressionMechanicsSample",
    "CompressionSceneObservation",
    "HemostasisClipSceneSource",
    "HemostasisEvidenceCursor",
    "HemostasisPatchSceneSource",
    "HemostasisSceneEvidence",
    "HemostasisSceneEvidenceSource",
    "PatchMechanicsSample",
    "PatchSceneObservation",
    "VesselMechanicsSample",
]
