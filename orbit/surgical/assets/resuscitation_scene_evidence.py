# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound post-physics evidence for rescue resuscitation and ventilation.

The public contracts in this module describe raw scene measurements.  They do
not accept delivered volume, effective ventilation, patient response, or
success values.  Those effects are derived by the owning mechanics subsystem
from an immutable :class:`SceneEvidenceEnvelope`, exact live attachment prim
identities, and registered source prims.

This is an engineering evidence boundary.  The sensor interpretations and
calibration values remain provisional and are not clinically validated or
approved for patient-care use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .scene_evidence import (
    EvidenceProvenance,
    SceneEvidenceEnvelope,
    SceneEvidenceRegistry,
    SceneEvidenceSource,
    SceneMeasurement,
)


_CHANNELS = frozenset({"crystalloid", "blood_product", "vasopressor"})


def _required_text(value: str, name: str) -> str:
    rendered = str(value).strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered


def _scene_path(value: str, name: str) -> str:
    rendered = _required_text(value, name)
    if not rendered.startswith("/"):
        raise ValueError(f"{name} must be an absolute scene path")
    return rendered


def _finite(value: float, name: str) -> float:
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be finite")
    return rendered


def _nonnegative(value: float, name: str) -> float:
    rendered = _finite(value, name)
    if rendered < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return rendered


def _vector3(
    value: tuple[float, float, float],
    name: str,
) -> tuple[float, float, float]:
    rendered = tuple(_finite(component, f"{name} component") for component in value)
    if len(rendered) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return rendered


def _unit_vector3(
    value: tuple[float, float, float],
    name: str,
) -> tuple[float, float, float]:
    rendered = _vector3(value, name)
    magnitude = math.sqrt(sum(component * component for component in rendered))
    if magnitude <= 1.0e-12:
        raise ValueError(f"{name} must be nonzero")
    return tuple(component / magnitude for component in rendered)


def _attachment_ids(
    values: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    rendered = tuple(
        sorted(
            _scene_path(value, f"{name} item")
            for value in values
        )
    )
    if len(rendered) != len(set(rendered)):
        raise ValueError(f"{name} must not contain duplicate prim paths")
    return rendered


def _raw_sample_ids(
    values: tuple[str, ...],
    name: str,
) -> tuple[str, ...]:
    rendered = tuple(
        sorted(
            _required_text(value, f"{name} item")
            for value in values
        )
    )
    if not rendered:
        raise ValueError(
            f"{name} must identify at least one underlying sensor sample"
        )
    if len(rendered) != len(set(rendered)):
        raise ValueError(f"{name} must not contain duplicates")
    return rendered


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        a - b for a, b in zip(left, right, strict=True)
    )


@dataclass(frozen=True)
class PumpSceneSources:
    """Exact prim registration for one pump channel and vascular access."""

    station_id: str
    channel_id: str
    plunger_prim_path: str
    downstream_flow_sensor_prim_path: str
    vascular_flow_sensor_prim_path: str
    reservoir_prim_path: str
    line_pressure_sensor_prim_path: str
    access_target_prim_path: str
    calibration_profile_id: str = "dranmar-resuscitation-engineering-v1"
    delivery_axis_sign: int = 1
    downstream_delivery_flow_sign: int = 1
    vascular_inflow_sign: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "station_id",
            _required_text(self.station_id, "station_id"),
        )
        if self.channel_id not in _CHANNELS:
            raise ValueError(
                f"unsupported resuscitation channel {self.channel_id!r}"
            )
        for name in (
            "plunger_prim_path",
            "downstream_flow_sensor_prim_path",
            "vascular_flow_sensor_prim_path",
            "reservoir_prim_path",
            "line_pressure_sensor_prim_path",
            "access_target_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "calibration_profile_id",
            _required_text(
                self.calibration_profile_id,
                "calibration_profile_id",
            ),
        )
        if self.delivery_axis_sign not in (-1, 1):
            raise ValueError("delivery_axis_sign must be either -1 or 1")
        if self.downstream_delivery_flow_sign not in (-1, 1):
            raise ValueError(
                "downstream_delivery_flow_sign must be either -1 or 1"
            )
        if self.vascular_inflow_sign not in (-1, 1):
            raise ValueError("vascular_inflow_sign must be either -1 or 1")
        source_ids = tuple(
            source.source_id for source in self.registered_sources
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("pump scene source IDs must be unique")

    @property
    def source_prefix(self) -> str:
        return f"{self.station_id}.pump.{self.channel_id}"

    @property
    def plunger_source_id(self) -> str:
        return f"{self.source_prefix}.plunger_position"

    @property
    def downstream_flow_source_id(self) -> str:
        return f"{self.source_prefix}.downstream_flow"

    @property
    def vascular_flow_source_id(self) -> str:
        return f"{self.source_prefix}.vascular_inflow"

    @property
    def reservoir_mass_source_id(self) -> str:
        return f"{self.source_prefix}.reservoir_mass"

    @property
    def line_pressure_source_id(self) -> str:
        return f"{self.source_prefix}.line_pressure"

    @property
    def registered_sources(self) -> tuple[SceneEvidenceSource, ...]:
        profile = self.calibration_profile_id
        return (
            SceneEvidenceSource(
                source_id=self.plunger_source_id,
                prim_path=self.plunger_prim_path,
                quantity="linear_joint_position",
                unit="m",
                coordinate_frame="pump_axis",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.downstream_flow_source_id,
                prim_path=self.downstream_flow_sensor_prim_path,
                quantity="downstream_volumetric_flow_rate",
                unit="mL/s",
                coordinate_frame="infusion_line",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.vascular_flow_source_id,
                prim_path=self.vascular_flow_sensor_prim_path,
                quantity="vascular_interface_volumetric_flow_rate",
                unit="mL/s",
                coordinate_frame="vascular_access",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.reservoir_mass_source_id,
                prim_path=self.reservoir_prim_path,
                quantity="reservoir_mass",
                unit="g",
                coordinate_frame="reservoir",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.line_pressure_source_id,
                prim_path=self.line_pressure_sensor_prim_path,
                quantity="line_pressure",
                unit="kPa",
                coordinate_frame="infusion_line",
                calibration_profile_id=profile,
            ),
        )

    @property
    def expected_source_ids(self) -> frozenset[str]:
        return frozenset(
            source.source_id for source in self.registered_sources
        )


@dataclass(frozen=True)
class PumpSensorSample:
    """Raw pump, reservoir, line, and vascular-interface readings."""

    plunger_position_m: float
    downstream_flow_ml_s: float
    vascular_inflow_ml_s: float
    reservoir_mass_g: float
    line_pressure_kpa: float
    access_attachment_prim_ids: tuple[str, ...] = ()
    plunger_raw_sample_ids: tuple[str, ...] = ()
    downstream_flow_raw_sample_ids: tuple[str, ...] = ()
    vascular_flow_raw_sample_ids: tuple[str, ...] = ()
    reservoir_mass_raw_sample_ids: tuple[str, ...] = ()
    line_pressure_raw_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "plunger_position_m",
            "downstream_flow_ml_s",
            "vascular_inflow_ml_s",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        for name in ("reservoir_mass_g",):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "line_pressure_kpa",
            _finite(self.line_pressure_kpa, "line_pressure_kpa"),
        )
        object.__setattr__(
            self,
            "access_attachment_prim_ids",
            _attachment_ids(
                self.access_attachment_prim_ids,
                "access_attachment_prim_ids",
            ),
        )
        for name in (
            "plunger_raw_sample_ids",
            "downstream_flow_raw_sample_ids",
            "vascular_flow_raw_sample_ids",
            "reservoir_mass_raw_sample_ids",
            "line_pressure_raw_sample_ids",
        ):
            object.__setattr__(
                self,
                name,
                _raw_sample_ids(getattr(self, name), name),
            )


@dataclass(frozen=True)
class PumpSceneEvidence:
    """Immutable pump evidence with effects derived from its envelope."""

    envelope: SceneEvidenceEnvelope
    sources: PumpSceneSources

    def __post_init__(self) -> None:
        provenance = self.envelope.provenance
        if provenance.adapter_id != PumpSceneEvidenceCollector.ADAPTER_ID:
            raise ValueError("pump evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != PumpSceneEvidenceCollector.ADAPTER_VERSION
        ):
            raise ValueError("pump evidence has the wrong adapter_version")
        PumpSceneEvidenceCollector.validate_envelope(
            self.sources,
            self.envelope,
        )
        for measurement in self.envelope.measurements:
            if len(measurement.value) != 1:
                raise ValueError(
                    f"pump source {measurement.source_id!r} must be scalar"
                )
            if measurement.attachment_prim_ids != _attachment_ids(
                measurement.attachment_prim_ids,
                "pump attachment_prim_ids",
            ):
                raise ValueError(
                    "pump attachment prim IDs must use canonical sorted order"
                )
            if measurement.target_prim_path != self.sources.access_target_prim_path:
                raise ValueError(
                    f"pump source {measurement.source_id!r} has the wrong "
                    "vascular target prim"
                )
            expected_pair_id = (
                f"{self.sources.vascular_flow_sensor_prim_path}|"
                f"{self.sources.access_target_prim_path}"
            )
            if measurement.contact_pair_id != expected_pair_id:
                raise ValueError(
                    f"pump source {measurement.source_id!r} has the wrong "
                    "line-to-access pair identity"
                )
        for source_id in (
            self.sources.plunger_source_id,
            self.sources.downstream_flow_source_id,
            self.sources.vascular_flow_source_id,
        ):
            _finite(
                self._measurement(source_id).value[0],
                source_id,
            )
        for source_id in (self.sources.reservoir_mass_source_id,):
            _nonnegative(
                self._measurement(source_id).value[0],
                source_id,
            )
        _finite(
            self._measurement(
                self.sources.line_pressure_source_id
            ).value[0],
            self.sources.line_pressure_source_id,
        )
        attachment_sets = {
            measurement.attachment_prim_ids
            for measurement in self.envelope.measurements
        }
        if len(attachment_sets) != 1:
            raise ValueError(
                "pump measurements disagree on live access attachment identity"
            )
        if self.vascular_inflow_ml_s > 1.0e-9 and not self.attachment_prim_ids:
            raise ValueError(
                "vascular inflow evidence has no live access attachment prim"
            )

    def _measurement(self, source_id: str) -> SceneMeasurement:
        try:
            return next(
                measurement
                for measurement in self.envelope.measurements
                if measurement.source_id == source_id
            )
        except StopIteration as error:
            raise ValueError(
                f"pump evidence is missing source {source_id!r}"
            ) from error

    @property
    def channel_id(self) -> str:
        return self.sources.channel_id

    @property
    def station_id(self) -> str:
        return self.sources.station_id

    @property
    def physics_step(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def episode_id(self) -> str:
        return self.envelope.provenance.episode_id

    @property
    def environment_id(self) -> str:
        return self.envelope.provenance.environment_id

    @property
    def topology_revision(self) -> str:
        return self.envelope.provenance.topology_revision

    @property
    def simulation_time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def attachment_prim_ids(self) -> tuple[str, ...]:
        return self._measurement(
            self.sources.vascular_flow_source_id
        ).attachment_prim_ids

    @property
    def access_connected(self) -> bool:
        return bool(self.attachment_prim_ids)

    def _scalar(self, source_id: str) -> float:
        return self._measurement(source_id).value[0]

    @property
    def plunger_position_m(self) -> float:
        return self._scalar(self.sources.plunger_source_id)

    @property
    def delivery_coordinate_m(self) -> float:
        """Plunger position expressed positive in the delivery direction."""

        return self.sources.delivery_axis_sign * self.plunger_position_m

    @property
    def downstream_flow_ml_s(self) -> float:
        return max(
            0.0,
            self.sources.downstream_delivery_flow_sign
            * self.raw_downstream_flow_ml_s,
        )

    @property
    def raw_downstream_flow_ml_s(self) -> float:
        return self._scalar(self.sources.downstream_flow_source_id)

    @property
    def vascular_inflow_ml_s(self) -> float:
        return max(
            0.0,
            self.sources.vascular_inflow_sign
            * self.raw_vascular_flow_ml_s,
        )

    @property
    def raw_vascular_flow_ml_s(self) -> float:
        return self._scalar(self.sources.vascular_flow_source_id)

    @property
    def reservoir_mass_g(self) -> float:
        return self._scalar(self.sources.reservoir_mass_source_id)

    @property
    def line_pressure_kpa(self) -> float:
        return self._scalar(self.sources.line_pressure_source_id)

    @property
    def interval_downstream_volume_ml(self) -> float:
        return self.downstream_flow_ml_s * self.dt_s

    @property
    def interval_intravascular_volume_ml(self) -> float:
        return self.vascular_inflow_ml_s * self.dt_s

    @property
    def interval_extravasated_ml(self) -> float:
        return max(
            0.0,
            self.interval_downstream_volume_ml
            - self.interval_intravascular_volume_ml,
        )


class PumpSceneEvidenceCollector:
    """Collect one coherent pump interval from registered scene prims."""

    ADAPTER_ID = "dranmar.resuscitation.scene-evidence"
    ADAPTER_VERSION = "2.0.0"

    def __init__(self, sources: PumpSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(sources.registered_sources)

    @staticmethod
    def validate_envelope(
        sources: PumpSceneSources,
        envelope: SceneEvidenceEnvelope,
    ) -> None:
        registry = SceneEvidenceRegistry(sources.registered_sources)
        registry.validate(envelope)
        actual = frozenset(
            measurement.source_id for measurement in envelope.measurements
        )
        if actual != sources.expected_source_ids:
            raise ValueError(
                "pump evidence source set does not match registered sources"
            )

    def collect_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        sample: PumpSensorSample,
    ) -> PumpSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        attachments = sample.access_attachment_prim_ids
        target = self.sources.access_target_prim_path
        pair_id = f"{self.sources.vascular_flow_sensor_prim_path}|{target}"
        measurements = (
            SceneMeasurement(
                source_id=self.sources.plunger_source_id,
                value=(sample.plunger_position_m,),
                source_prim_path=self.sources.plunger_prim_path,
                target_prim_path=target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.plunger_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.downstream_flow_source_id,
                value=(sample.downstream_flow_ml_s,),
                source_prim_path=(
                    self.sources.downstream_flow_sensor_prim_path
                ),
                target_prim_path=target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.downstream_flow_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.vascular_flow_source_id,
                value=(sample.vascular_inflow_ml_s,),
                source_prim_path=(
                    self.sources.vascular_flow_sensor_prim_path
                ),
                target_prim_path=target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.vascular_flow_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.reservoir_mass_source_id,
                value=(sample.reservoir_mass_g,),
                source_prim_path=self.sources.reservoir_prim_path,
                target_prim_path=target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.reservoir_mass_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.line_pressure_source_id,
                value=(sample.line_pressure_kpa,),
                source_prim_path=(
                    self.sources.line_pressure_sensor_prim_path
                ),
                target_prim_path=target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.line_pressure_raw_sample_ids,
            ),
        )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=measurements,
        )
        self.registry.validate(envelope)
        return PumpSceneEvidence(envelope=envelope, sources=self.sources)


@dataclass(frozen=True)
class VentilationSceneSources:
    """Exact prim registration for one airway circuit and thorax."""

    station_id: str
    valve_prim_path: str
    inspiratory_flow_sensor_prim_path: str
    airway_interface_flow_sensor_prim_path: str
    airway_pressure_sensor_prim_path: str
    fio2_sensor_prim_path: str
    chest_landmark_prim_path: str
    chest_reference_prim_path: str
    airway_target_prim_path: str
    calibration_profile_id: str = "dranmar-ventilation-engineering-v1"
    valve_closed_angle_deg: float = 0.0
    valve_opening_axis_sign: int = 1
    inspiratory_flow_sign: int = 1
    airway_inflow_sign: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "station_id",
            _required_text(self.station_id, "station_id"),
        )
        for name in (
            "valve_prim_path",
            "inspiratory_flow_sensor_prim_path",
            "airway_interface_flow_sensor_prim_path",
            "airway_pressure_sensor_prim_path",
            "fio2_sensor_prim_path",
            "chest_landmark_prim_path",
            "chest_reference_prim_path",
            "airway_target_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
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
            "valve_closed_angle_deg",
            _finite(self.valve_closed_angle_deg, "valve_closed_angle_deg"),
        )
        if self.valve_opening_axis_sign not in (-1, 1):
            raise ValueError("valve_opening_axis_sign must be either -1 or 1")
        if self.inspiratory_flow_sign not in (-1, 1):
            raise ValueError("inspiratory_flow_sign must be either -1 or 1")
        if self.airway_inflow_sign not in (-1, 1):
            raise ValueError("airway_inflow_sign must be either -1 or 1")
        source_ids = tuple(
            source.source_id for source in self.registered_sources
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("ventilation scene source IDs must be unique")

    @property
    def source_prefix(self) -> str:
        return f"{self.station_id}.ventilation"

    @property
    def valve_source_id(self) -> str:
        return f"{self.source_prefix}.valve_angle"

    @property
    def inspiratory_flow_source_id(self) -> str:
        return f"{self.source_prefix}.inspiratory_flow"

    @property
    def airway_interface_flow_source_id(self) -> str:
        return f"{self.source_prefix}.airway_interface_flow"

    @property
    def airway_pressure_source_id(self) -> str:
        return f"{self.source_prefix}.airway_pressure"

    @property
    def fio2_source_id(self) -> str:
        return f"{self.source_prefix}.fio2"

    @property
    def chest_landmark_source_id(self) -> str:
        return f"{self.source_prefix}.chest_landmark"

    @property
    def chest_reference_source_id(self) -> str:
        return f"{self.source_prefix}.chest_reference"

    @property
    def registered_sources(self) -> tuple[SceneEvidenceSource, ...]:
        profile = self.calibration_profile_id
        return (
            SceneEvidenceSource(
                source_id=self.valve_source_id,
                prim_path=self.valve_prim_path,
                quantity="valve_joint_angle",
                unit="deg",
                coordinate_frame="valve_joint",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.inspiratory_flow_source_id,
                prim_path=self.inspiratory_flow_sensor_prim_path,
                quantity="circuit_inspiratory_flow_rate",
                unit="L/min",
                coordinate_frame="ventilation_circuit",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.airway_interface_flow_source_id,
                prim_path=self.airway_interface_flow_sensor_prim_path,
                quantity="airway_interface_flow_rate",
                unit="L/min",
                coordinate_frame="airway_interface",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.airway_pressure_source_id,
                prim_path=self.airway_pressure_sensor_prim_path,
                quantity="airway_pressure",
                unit="cmH2O",
                coordinate_frame="airway_interface",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.fio2_source_id,
                prim_path=self.fio2_sensor_prim_path,
                quantity="inspired_oxygen_fraction",
                unit="fraction",
                coordinate_frame="airway_interface",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.chest_landmark_source_id,
                prim_path=self.chest_landmark_prim_path,
                quantity="chest_landmark_position",
                unit="m",
                coordinate_frame="world",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.chest_reference_source_id,
                prim_path=self.chest_reference_prim_path,
                quantity="end_expiratory_reference_position_and_expansion_axis",
                unit="m_and_unit_vector",
                coordinate_frame="world",
                calibration_profile_id=profile,
            ),
        )

    @property
    def expected_source_ids(self) -> frozenset[str]:
        return frozenset(
            source.source_id for source in self.registered_sources
        )


@dataclass(frozen=True)
class VentilationSensorSample:
    """Raw circuit, airway-interface, and thorax readings."""

    valve_angle_deg: float
    inspiratory_flow_l_min: float
    airway_interface_flow_l_min: float
    airway_pressure_cmh2o: float
    measured_fio2_fraction: float
    chest_landmark_position_w_m: tuple[float, float, float]
    chest_reference_position_w_m: tuple[float, float, float]
    chest_expansion_axis_w: tuple[float, float, float]
    airway_attachment_prim_ids: tuple[str, ...] = ()
    valve_raw_sample_ids: tuple[str, ...] = ()
    inspiratory_flow_raw_sample_ids: tuple[str, ...] = ()
    airway_interface_flow_raw_sample_ids: tuple[str, ...] = ()
    airway_pressure_raw_sample_ids: tuple[str, ...] = ()
    fio2_raw_sample_ids: tuple[str, ...] = ()
    chest_landmark_raw_sample_ids: tuple[str, ...] = ()
    chest_reference_raw_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "valve_angle_deg",
            _finite(self.valve_angle_deg, "valve_angle_deg"),
        )
        for name in (
            "inspiratory_flow_l_min",
            "airway_interface_flow_l_min",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "airway_pressure_cmh2o",
            _finite(self.airway_pressure_cmh2o, "airway_pressure_cmh2o"),
        )
        fio2 = _finite(
            self.measured_fio2_fraction,
            "measured_fio2_fraction",
        )
        if not 0.21 <= fio2 <= 1.0:
            raise ValueError(
                "measured_fio2_fraction must be within [0.21, 1]"
            )
        object.__setattr__(self, "measured_fio2_fraction", fio2)
        for name in (
            "chest_landmark_position_w_m",
            "chest_reference_position_w_m",
        ):
            object.__setattr__(
                self,
                name,
                _vector3(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "chest_expansion_axis_w",
            _unit_vector3(
                self.chest_expansion_axis_w,
                "chest_expansion_axis_w",
            ),
        )
        object.__setattr__(
            self,
            "airway_attachment_prim_ids",
            _attachment_ids(
                self.airway_attachment_prim_ids,
                "airway_attachment_prim_ids",
            ),
        )
        for name in (
            "valve_raw_sample_ids",
            "inspiratory_flow_raw_sample_ids",
            "airway_interface_flow_raw_sample_ids",
            "airway_pressure_raw_sample_ids",
            "fio2_raw_sample_ids",
            "chest_landmark_raw_sample_ids",
            "chest_reference_raw_sample_ids",
        ):
            object.__setattr__(
                self,
                name,
                _raw_sample_ids(getattr(self, name), name),
            )


@dataclass(frozen=True)
class VentilationSceneEvidence:
    """Immutable airway evidence with leak and excursion derived from raw data."""

    envelope: SceneEvidenceEnvelope
    sources: VentilationSceneSources

    def __post_init__(self) -> None:
        provenance = self.envelope.provenance
        if provenance.adapter_id != VentilationSceneEvidenceCollector.ADAPTER_ID:
            raise ValueError("ventilation evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != VentilationSceneEvidenceCollector.ADAPTER_VERSION
        ):
            raise ValueError("ventilation evidence has the wrong adapter_version")
        VentilationSceneEvidenceCollector.validate_envelope(
            self.sources,
            self.envelope,
        )
        scalar_ids = {
            self.sources.valve_source_id,
            self.sources.inspiratory_flow_source_id,
            self.sources.airway_interface_flow_source_id,
            self.sources.airway_pressure_source_id,
            self.sources.fio2_source_id,
        }
        for source_id in scalar_ids:
            if len(self._measurement(source_id).value) != 1:
                raise ValueError(
                    f"ventilation source {source_id!r} must be scalar"
                )
        _finite(
            self._measurement(self.sources.valve_source_id).value[0],
            self.sources.valve_source_id,
        )
        for source_id in (
            self.sources.inspiratory_flow_source_id,
            self.sources.airway_interface_flow_source_id,
            self.sources.airway_pressure_source_id,
        ):
            _finite(
                self._measurement(source_id).value[0],
                source_id,
            )
        fio2 = _finite(
            self._measurement(self.sources.fio2_source_id).value[0],
            self.sources.fio2_source_id,
        )
        if not 0.21 <= fio2 <= 1.0:
            raise ValueError(
                "measured FiO2 evidence must be within [0.21, 1]"
            )
        if len(self._measurement(self.sources.chest_landmark_source_id).value) != 3:
            raise ValueError("chest landmark measurement must be a 3-vector")
        if len(self._measurement(self.sources.chest_reference_source_id).value) != 6:
            raise ValueError(
                "chest reference measurement must contain position and axis"
            )
        _unit_vector3(
            tuple(
                self._measurement(
                    self.sources.chest_reference_source_id
                ).value[3:]
            ),
            "chest expansion axis",
        )
        airway_ids = (
            self.sources.valve_source_id,
            self.sources.inspiratory_flow_source_id,
            self.sources.airway_interface_flow_source_id,
            self.sources.airway_pressure_source_id,
            self.sources.fio2_source_id,
        )
        attachment_sets = {
            self._measurement(source_id).attachment_prim_ids
            for source_id in airway_ids
        }
        if len(attachment_sets) != 1:
            raise ValueError(
                "airway measurements disagree on live attachment identity"
            )
        for source_id in airway_ids:
            measurement = self._measurement(source_id)
            if measurement.attachment_prim_ids != _attachment_ids(
                measurement.attachment_prim_ids,
                "airway attachment_prim_ids",
            ):
                raise ValueError(
                    "airway attachment prim IDs must use canonical sorted order"
                )
            if measurement.target_prim_path != self.sources.airway_target_prim_path:
                raise ValueError(
                    f"ventilation source {source_id!r} has the wrong airway "
                    "target prim"
                )
            expected_pair_id = (
                f"{self.sources.airway_interface_flow_sensor_prim_path}|"
                f"{self.sources.airway_target_prim_path}"
            )
            if measurement.contact_pair_id != expected_pair_id:
                raise ValueError(
                    f"ventilation source {source_id!r} has the wrong "
                    "circuit-to-airway pair identity"
                )
        chest_landmark = self._measurement(
            self.sources.chest_landmark_source_id
        )
        if (
            chest_landmark.target_prim_path
            != self.sources.chest_reference_prim_path
        ):
            raise ValueError(
                "chest landmark evidence has the wrong reference prim"
            )
        chest_reference = self._measurement(
            self.sources.chest_reference_source_id
        )
        if (
            chest_reference.target_prim_path
            != self.sources.chest_landmark_prim_path
        ):
            raise ValueError(
                "chest reference evidence has the wrong landmark prim"
            )
        if (
            self.airway_interface_flow_l_min > 1.0e-9
            and not self.attachment_prim_ids
        ):
            raise ValueError(
                "airway-interface flow evidence has no live attachment prim"
            )

    def _measurement(self, source_id: str) -> SceneMeasurement:
        try:
            return next(
                measurement
                for measurement in self.envelope.measurements
                if measurement.source_id == source_id
            )
        except StopIteration as error:
            raise ValueError(
                f"ventilation evidence is missing source {source_id!r}"
            ) from error

    def _scalar(self, source_id: str) -> float:
        return self._measurement(source_id).value[0]

    @property
    def station_id(self) -> str:
        return self.sources.station_id

    @property
    def physics_step(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def episode_id(self) -> str:
        return self.envelope.provenance.episode_id

    @property
    def environment_id(self) -> str:
        return self.envelope.provenance.environment_id

    @property
    def topology_revision(self) -> str:
        return self.envelope.provenance.topology_revision

    @property
    def simulation_time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def dt_s(self) -> float:
        return self.envelope.provenance.dt_s

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def attachment_prim_ids(self) -> tuple[str, ...]:
        return self._measurement(
            self.sources.airway_interface_flow_source_id
        ).attachment_prim_ids

    @property
    def airway_connected(self) -> bool:
        return bool(self.attachment_prim_ids)

    @property
    def valve_angle_deg(self) -> float:
        return self._scalar(self.sources.valve_source_id)

    @property
    def valve_opening_angle_deg(self) -> float:
        """Valve travel expressed positive away from the registered stop."""

        return max(
            0.0,
            self.sources.valve_opening_axis_sign
            * (
                self.valve_angle_deg
                - self.sources.valve_closed_angle_deg
            ),
        )

    @property
    def inspiratory_flow_l_min(self) -> float:
        return max(
            0.0,
            self.sources.inspiratory_flow_sign
            * self.raw_inspiratory_flow_l_min,
        )

    @property
    def raw_inspiratory_flow_l_min(self) -> float:
        return self._scalar(self.sources.inspiratory_flow_source_id)

    @property
    def airway_interface_flow_l_min(self) -> float:
        return max(
            0.0,
            self.sources.airway_inflow_sign
            * self.raw_airway_interface_flow_l_min,
        )

    @property
    def raw_airway_interface_flow_l_min(self) -> float:
        return self._scalar(
            self.sources.airway_interface_flow_source_id
        )

    @property
    def leaked_flow_l_min(self) -> float:
        return max(
            0.0,
            self.inspiratory_flow_l_min
            - self.airway_interface_flow_l_min,
        )

    @property
    def airway_pressure_cmh2o(self) -> float:
        return self._scalar(self.sources.airway_pressure_source_id)

    @property
    def measured_fio2_fraction(self) -> float:
        return self._scalar(self.sources.fio2_source_id)

    @property
    def chest_landmark_position_w_m(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.chest_landmark_source_id
            ).value
        )

    @property
    def chest_reference_position_w_m(self) -> tuple[float, float, float]:
        values = self._measurement(
            self.sources.chest_reference_source_id
        ).value
        return tuple(values[:3])

    @property
    def chest_expansion_axis_w(self) -> tuple[float, float, float]:
        values = self._measurement(
            self.sources.chest_reference_source_id
        ).value
        return _unit_vector3(tuple(values[3:]), "chest expansion axis")

    @property
    def chest_excursion_m(self) -> float:
        return max(
            0.0,
            _dot(
                _subtract(
                    self.chest_landmark_position_w_m,
                    self.chest_reference_position_w_m,
                ),
                self.chest_expansion_axis_w,
            ),
        )


class VentilationSceneEvidenceCollector:
    """Collect one coherent ventilation interval from registered scene prims."""

    ADAPTER_ID = PumpSceneEvidenceCollector.ADAPTER_ID
    ADAPTER_VERSION = PumpSceneEvidenceCollector.ADAPTER_VERSION

    def __init__(self, sources: VentilationSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(sources.registered_sources)

    @staticmethod
    def validate_envelope(
        sources: VentilationSceneSources,
        envelope: SceneEvidenceEnvelope,
    ) -> None:
        registry = SceneEvidenceRegistry(sources.registered_sources)
        registry.validate(envelope)
        actual = frozenset(
            measurement.source_id for measurement in envelope.measurements
        )
        if actual != sources.expected_source_ids:
            raise ValueError(
                "ventilation evidence source set does not match registered "
                "sources"
            )

    def collect_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        sample: VentilationSensorSample,
    ) -> VentilationSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        attachments = sample.airway_attachment_prim_ids
        airway_target = self.sources.airway_target_prim_path
        pair_id = (
            f"{self.sources.airway_interface_flow_sensor_prim_path}|"
            f"{airway_target}"
        )
        measurements = (
            SceneMeasurement(
                source_id=self.sources.valve_source_id,
                value=(sample.valve_angle_deg,),
                source_prim_path=self.sources.valve_prim_path,
                target_prim_path=airway_target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.valve_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.inspiratory_flow_source_id,
                value=(sample.inspiratory_flow_l_min,),
                source_prim_path=(
                    self.sources.inspiratory_flow_sensor_prim_path
                ),
                target_prim_path=airway_target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.inspiratory_flow_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.airway_interface_flow_source_id,
                value=(sample.airway_interface_flow_l_min,),
                source_prim_path=(
                    self.sources.airway_interface_flow_sensor_prim_path
                ),
                target_prim_path=airway_target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=(
                    sample.airway_interface_flow_raw_sample_ids
                ),
            ),
            SceneMeasurement(
                source_id=self.sources.airway_pressure_source_id,
                value=(sample.airway_pressure_cmh2o,),
                source_prim_path=(
                    self.sources.airway_pressure_sensor_prim_path
                ),
                target_prim_path=airway_target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.airway_pressure_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.fio2_source_id,
                value=(sample.measured_fio2_fraction,),
                source_prim_path=self.sources.fio2_sensor_prim_path,
                target_prim_path=airway_target,
                contact_pair_id=pair_id,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.fio2_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.chest_landmark_source_id,
                value=sample.chest_landmark_position_w_m,
                source_prim_path=self.sources.chest_landmark_prim_path,
                target_prim_path=self.sources.chest_reference_prim_path,
                raw_sample_ids=sample.chest_landmark_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.chest_reference_source_id,
                value=(
                    *sample.chest_reference_position_w_m,
                    *sample.chest_expansion_axis_w,
                ),
                source_prim_path=self.sources.chest_reference_prim_path,
                target_prim_path=self.sources.chest_landmark_prim_path,
                raw_sample_ids=sample.chest_reference_raw_sample_ids,
            ),
        )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=measurements,
        )
        self.registry.validate(envelope)
        return VentilationSceneEvidence(
            envelope=envelope,
            sources=self.sources,
        )


__all__ = [
    "PumpSceneEvidence",
    "PumpSceneEvidenceCollector",
    "PumpSceneSources",
    "PumpSensorSample",
    "VentilationSceneEvidence",
    "VentilationSceneEvidenceCollector",
    "VentilationSceneSources",
    "VentilationSensorSample",
]
