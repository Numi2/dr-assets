# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound mechanics evidence for the Adaptive Anastomosis workcell.

Staple integrity is derived from shared rod results plus live attachment
identity, collar sectors consume shared cohesive responses, lumen geometry is
read from shared soft-tissue state, and leak verification consumes measured
pressure and flow.  No generic foundation result is relabeled as a clinical
success outcome.

All material and acceptance parameters remain provisional engineering values;
this evidence boundary is not clinically validated.

The shared mechanics remain reduced-order and use provisional engineering
parameters.  This module therefore exposes observed topology, attachments,
cohesive state, lumen geometry, pressure, and leak flow, while keeping retained
repair, bonded-collar, and patency claims fail-closed unless every required
mechanical and sensor source is present in the same post-physics interval.
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
from orbit.surgical.physics.cohesive import CohesiveInterface, CohesiveResponse
from orbit.surgical.physics.rod import CosseratRod, RodStepResult
from orbit.surgical.physics.soft_tissue import (
    LayeredSoftTissue,
    SoftTissueStepResult,
)


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


def _provisional_status(value: str, label: str) -> str:
    result = _required_text(value, label)
    lowered = result.lower()
    if any(
        marker in lowered
        for marker in (
            "clinical_validated",
            "clinically_validated",
            "patient_care_approved",
        )
    ):
        raise ValueError(f"{label} cannot claim clinical validation")
    if not any(
        marker in lowered
        for marker in (
            "provisional",
            "research",
            "engineering",
            "nonclinical",
            "unvalidated",
        )
    ):
        raise ValueError(
            f"{label} must explicitly remain provisional/non-clinical"
        )
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


def _node_indices(
    values: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{label} cannot contain negative indices")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique indices")
    if len(result) < 16:
        raise ValueError(f"{label} must contain at least 16 lumen nodes")
    return result


def _force_magnitude(force: tuple[float, float, float]) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in force))


def _measurement(
    envelope: SceneEvidenceEnvelope,
    source_id: str,
) -> SceneMeasurement | None:
    return next(
        (
            measurement
            for measurement in envelope.measurements
            if measurement.source_id == source_id
        ),
        None,
    )


def _raw_ids(
    values: tuple[str, ...],
    source_id: str,
) -> tuple[str, ...]:
    result = tuple(
        sorted(
            _required_text(value, f"{source_id} raw_sample_id")
            for value in values
        )
    )
    if not result:
        raise ValueError(
            f"anastomosis source {source_id!r} requires raw solver sample IDs"
        )
    if len(result) != len(set(result)):
        raise ValueError(
            f"anastomosis source {source_id!r} has duplicate raw sample IDs"
        )
    return result


@dataclass(frozen=True)
class AnastomosisStapleSceneSource:
    """One deployed staple and its two tissue attachment identities."""

    staple_path: str
    source_id: str
    left_attachment_prim_path: str
    right_attachment_prim_path: str
    left_anchor_node_index: int
    right_anchor_node_index: int

    def __post_init__(self) -> None:
        for name in (
            "staple_path",
            "left_attachment_prim_path",
            "right_attachment_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )
        for name in ("left_anchor_node_index", "right_anchor_node_index"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    @property
    def expected_attachment_prim_ids(self) -> frozenset[str]:
        return frozenset(
            (
                self.left_attachment_prim_path,
                self.right_attachment_prim_path,
            )
        )


@dataclass(frozen=True)
class AnastomosisCollarSceneSource:
    """One collar whose two cohesive interfaces are registered per sector."""

    collar_path: str
    source_id_prefix: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "collar_path",
            _scene_path(self.collar_path, "collar_path"),
        )
        object.__setattr__(
            self,
            "source_id_prefix",
            _required_text(self.source_id_prefix, "source_id_prefix"),
        )

    def source_id(self, sector: int, side: str) -> str:
        return f"{self.source_id_prefix}.{side.lower()}.{sector:02d}"

    def interface_prim_path(self, sector: int, side: str) -> str:
        rendered_side = side.capitalize()
        return (
            f"{self.collar_path}/Collisions/"
            f"{rendered_side}BondCell_{sector:02d}"
        )

    def attachment_prim_path(self, sector: int, side: str) -> str:
        return (
            f"{self.collar_path}/Attachments/"
            f"{side.lower()}_{sector:02d}"
        )


@dataclass(frozen=True)
class AdaptiveAnastomosisSceneSources:
    """Exact source registration for one tissue pair and deployed repairs."""

    workcell_id: str
    left_tissue_prim_path: str
    right_tissue_prim_path: str
    left_lumen_node_indices: tuple[int, ...]
    right_lumen_node_indices: tuple[int, ...]
    pressure_sensor_prim_path: str
    leak_flow_sensor_prim_path: str
    staple_sources: tuple[AnastomosisStapleSceneSource, ...] = ()
    collar_sources: tuple[AnastomosisCollarSceneSource, ...] = ()
    collar_sector_count: int = 16
    calibration_profile_id: str = "dranmar-anastomosis-engineering-v1"

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
        for name in (
            "left_tissue_prim_path",
            "right_tissue_prim_path",
            "pressure_sensor_prim_path",
            "leak_flow_sensor_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "left_lumen_node_indices",
            _node_indices(
                self.left_lumen_node_indices,
                "left_lumen_node_indices",
            ),
        )
        object.__setattr__(
            self,
            "right_lumen_node_indices",
            _node_indices(
                self.right_lumen_node_indices,
                "right_lumen_node_indices",
            ),
        )
        object.__setattr__(self, "staple_sources", tuple(self.staple_sources))
        object.__setattr__(self, "collar_sources", tuple(self.collar_sources))
        sector_count = int(self.collar_sector_count)
        if sector_count <= 0:
            raise ValueError("collar_sector_count must be positive")
        object.__setattr__(self, "collar_sector_count", sector_count)
        staple_paths = [source.staple_path for source in self.staple_sources]
        staple_attachment_paths = [
            attachment_path
            for source in self.staple_sources
            for attachment_path in source.expected_attachment_prim_ids
        ]
        collar_paths = [source.collar_path for source in self.collar_sources]
        if len(staple_paths) != len(set(staple_paths)):
            raise ValueError("staple paths must be unique")
        if len(staple_attachment_paths) != len(
            set(staple_attachment_paths)
        ):
            raise ValueError("staple attachment paths must be unique")
        if len(collar_paths) != len(set(collar_paths)):
            raise ValueError("collar paths must be unique")
        source_ids = [
            source.source_id for source in self.registered_sources
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("anastomosis source IDs must be unique")

    @property
    def left_tissue_source_id(self) -> str:
        return f"{self.workcell_id}.left_tissue_lumen_state"

    @property
    def right_tissue_source_id(self) -> str:
        return f"{self.workcell_id}.right_tissue_lumen_state"

    @property
    def pressure_source_id(self) -> str:
        return f"{self.workcell_id}.chamber_pressure"

    @property
    def leak_flow_source_id(self) -> str:
        return f"{self.workcell_id}.measured_leak_flow"

    @property
    def registered_sources(self) -> tuple[SceneEvidenceSource, ...]:
        profile = self.calibration_profile_id
        sources = [
            SceneEvidenceSource(
                source_id=self.left_tissue_source_id,
                prim_path=self.left_tissue_prim_path,
                quantity="shared_soft_tissue_lumen_nodes_and_state",
                unit="SI_vector",
                coordinate_frame="world",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.right_tissue_source_id,
                prim_path=self.right_tissue_prim_path,
                quantity="shared_soft_tissue_lumen_nodes_and_state",
                unit="SI_vector",
                coordinate_frame="world",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.pressure_source_id,
                prim_path=self.pressure_sensor_prim_path,
                quantity="measured_chamber_pressure",
                unit="Pa",
                coordinate_frame="anastomosis_test_chamber",
                calibration_profile_id=profile,
            ),
            SceneEvidenceSource(
                source_id=self.leak_flow_source_id,
                prim_path=self.leak_flow_sensor_prim_path,
                quantity="measured_external_leak_flow",
                unit="m3_s",
                coordinate_frame="anastomosis_test_chamber",
                calibration_profile_id=profile,
            ),
        ]
        sources.extend(
            SceneEvidenceSource(
                source_id=source.source_id,
                prim_path=source.staple_path,
                quantity="shared_rod_staple_observation",
                unit="SI_vector",
                coordinate_frame="staple_tissue_interface",
                calibration_profile_id=profile,
            )
            for source in self.staple_sources
        )
        for collar in self.collar_sources:
            for sector in range(self.collar_sector_count):
                for side in ("left", "right"):
                    sources.append(
                        SceneEvidenceSource(
                            source_id=collar.source_id(sector, side),
                            prim_path=collar.interface_prim_path(
                                sector,
                                side,
                            ),
                            quantity="shared_cohesive_sector_response",
                            unit="SI_vector",
                            coordinate_frame=(
                                "world_with_registered_interface_normal"
                            ),
                            calibration_profile_id=profile,
                        )
                    )
        return tuple(sources)

    @property
    def registered_source_ids(self) -> frozenset[str]:
        return frozenset(
            source.source_id for source in self.registered_sources
        )

    def staple_source_for(
        self,
        staple_path: str,
    ) -> AnastomosisStapleSceneSource:
        try:
            return next(
                source
                for source in self.staple_sources
                if source.staple_path == staple_path
            )
        except StopIteration as error:
            raise ValueError(
                f"staple path {staple_path!r} was not registered"
            ) from error

    def collar_source_for(
        self,
        collar_path: str,
    ) -> AnastomosisCollarSceneSource:
        try:
            return next(
                source
                for source in self.collar_sources
                if source.collar_path == collar_path
            )
        except StopIteration as error:
            raise ValueError(
                f"collar path {collar_path!r} was not registered"
            ) from error

    def validate_envelope(self, envelope: SceneEvidenceEnvelope) -> None:
        provenance = envelope.provenance
        if provenance.adapter_id != AdaptiveAnastomosisSceneEvidence.ADAPTER_ID:
            raise ValueError("anastomosis evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != AdaptiveAnastomosisSceneEvidence.ADAPTER_VERSION
        ):
            raise ValueError(
                "anastomosis evidence has the wrong adapter_version"
            )
        SceneEvidenceRegistry(self.registered_sources).validate(envelope)
        measurements = {
            measurement.source_id: measurement
            for measurement in envelope.measurements
        }
        if not set(measurements).issubset(self.registered_source_ids):
            raise ValueError("anastomosis evidence contains unknown sources")
        for source_id, measurement in measurements.items():
            _raw_ids(measurement.raw_sample_ids, source_id)
        required = {
            self.left_tissue_source_id,
            self.right_tissue_source_id,
            self.pressure_source_id,
            self.leak_flow_source_id,
        }
        if not required.issubset(measurements):
            raise ValueError(
                "anastomosis evidence is missing tissue or pressure/flow "
                "measurements"
            )
        expected_tissue_lengths = {
            self.left_tissue_source_id: (
                4 + 3 * len(self.left_lumen_node_indices)
            ),
            self.right_tissue_source_id: (
                4 + 3 * len(self.right_lumen_node_indices)
            ),
        }
        for source_id, expected_length in expected_tissue_lengths.items():
            if len(measurements[source_id].value) != expected_length:
                raise ValueError(
                    f"tissue measurement {source_id!r} must contain "
                    f"{expected_length} values"
                )
        for source_id in (self.pressure_source_id, self.leak_flow_source_id):
            if len(measurements[source_id].value) != 1:
                raise ValueError(
                    f"sensor measurement {source_id!r} must be scalar"
                )
        tissue_pair_id = (
            f"{self.left_tissue_prim_path}|"
            f"{self.right_tissue_prim_path}"
        )
        for source_id in (
            self.pressure_source_id,
            self.leak_flow_source_id,
        ):
            measurement = measurements[source_id]
            if (
                measurement.target_prim_path
                != self.left_tissue_prim_path
                or measurement.contact_pair_id != tissue_pair_id
            ):
                raise ValueError(
                    f"sensor measurement {source_id!r} is not tied to the "
                    "registered tissue pair"
                )
        for source in self.staple_sources:
            measurement = measurements.get(source.source_id)
            if measurement is None:
                continue
            if len(measurement.value) != 9:
                raise ValueError(
                    f"staple measurement {source.source_id!r} must contain "
                    "nine values"
                )
            if measurement.contact_pair_id != tissue_pair_id:
                raise ValueError(
                    f"staple measurement {source.source_id!r} has the wrong "
                    "tissue-pair identity"
                )
            if (
                measurement.target_prim_path
                != self.left_tissue_prim_path
            ):
                raise ValueError(
                    f"staple measurement {source.source_id!r} has the wrong "
                    "registered tissue target"
                )
            if not set(measurement.attachment_prim_ids).issubset(
                source.expected_attachment_prim_ids
            ):
                raise ValueError(
                    f"staple measurement {source.source_id!r} names an "
                    "unregistered attachment"
                )
        for collar in self.collar_sources:
            for sector in range(self.collar_sector_count):
                for side, target in (
                    ("left", self.left_tissue_prim_path),
                    ("right", self.right_tissue_prim_path),
                ):
                    source_id = collar.source_id(sector, side)
                    measurement = measurements.get(source_id)
                    if measurement is None:
                        continue
                    if len(measurement.value) != 14:
                        raise ValueError(
                            f"collar measurement {source_id!r} must contain "
                            "fourteen values"
                        )
                    if measurement.target_prim_path != target:
                        raise ValueError(
                            f"collar measurement {source_id!r} has the wrong "
                            "tissue target"
                        )
                    expected_pair = (
                        f"{collar.interface_prim_path(sector, side)}|{target}"
                    )
                    if measurement.contact_pair_id != expected_pair:
                        raise ValueError(
                            f"collar measurement {source_id!r} has the wrong "
                            "interface identity"
                        )
                    expected_attachment = collar.attachment_prim_path(
                        sector,
                        side,
                    )
                    if (
                        measurement.attachment_prim_ids
                        and measurement.attachment_prim_ids
                        != (expected_attachment,)
                    ):
                        raise ValueError(
                            f"collar measurement {source_id!r} names an "
                            "unregistered attachment"
                        )


@dataclass(frozen=True)
class StapleSceneObservation:
    """Shared rod result plus attachment prims active after the same step."""

    rod: CosseratRod
    rod_result: RodStepResult
    active_attachment_prim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rod, CosseratRod):
            raise TypeError("staple observation requires CosseratRod")
        if not isinstance(self.rod_result, RodStepResult):
            raise TypeError("staple observation requires RodStepResult")
        if self.rod.component_id != self.rod_result.component_id:
            raise ValueError(
                "staple rod and result must identify the same component"
            )
        if (
            self.rod.state.physics_step != self.rod_result.physics_step
            or not math.isclose(
                self.rod.state.time_s,
                self.rod_result.time_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "staple observation must use the rod's exact current "
                "post-step result"
            )
        object.__setattr__(
            self,
            "active_attachment_prim_ids",
            tuple(
                _scene_path(path, "active_attachment_prim_id")
                for path in self.active_attachment_prim_ids
            ),
        )


@dataclass(frozen=True)
class CollarSectorSceneObservation:
    """Shared cohesive response for one registered collar-sector interface."""

    interface: CohesiveInterface
    response: CohesiveResponse
    active_attachment_prim_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.interface, CohesiveInterface):
            raise TypeError(
                "collar sector observation requires CohesiveInterface"
            )
        if not isinstance(self.response, CohesiveResponse):
            raise TypeError(
                "collar sector observation requires CohesiveResponse"
            )
        if self.interface.traction_only() is not self.response:
            raise ValueError(
                "collar response must be the interface's exact last "
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
                "cohesive response identity does not match its interface"
            )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-9)
            for left, right in zip(
                self.response.normal_a_to_b,
                self.interface.normal_a_to_b,
                strict=True,
            )
        ):
            raise ValueError(
                "cohesive response normal does not match its interface"
            )
        if self.active_attachment_prim_id is not None:
            object.__setattr__(
                self,
                "active_attachment_prim_id",
                _scene_path(
                    self.active_attachment_prim_id,
                    "active_attachment_prim_id",
                ),
            )


@dataclass(frozen=True)
class PressureFlowSceneObservation:
    """Raw calibrated chamber pressure and external leak-flow readings."""

    pressure_sensor_prim_path: str
    leak_flow_sensor_prim_path: str
    physics_step: int
    simulation_time_s: float
    chamber_pressure_pa: float
    leak_flow_m3_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pressure_sensor_prim_path",
            _scene_path(
                self.pressure_sensor_prim_path,
                "pressure_sensor_prim_path",
            ),
        )
        object.__setattr__(
            self,
            "leak_flow_sensor_prim_path",
            _scene_path(
                self.leak_flow_sensor_prim_path,
                "leak_flow_sensor_prim_path",
            ),
        )
        physics_step = int(self.physics_step)
        if physics_step < 0:
            raise ValueError("physics_step must be non-negative")
        object.__setattr__(self, "physics_step", physics_step)
        object.__setattr__(
            self,
            "simulation_time_s",
            _nonnegative(self.simulation_time_s, "simulation_time_s"),
        )
        object.__setattr__(
            self,
            "chamber_pressure_pa",
            _nonnegative(self.chamber_pressure_pa, "chamber_pressure_pa"),
        )
        object.__setattr__(
            self,
            "leak_flow_m3_s",
            _nonnegative(self.leak_flow_m3_s, "leak_flow_m3_s"),
        )


@dataclass(frozen=True)
class StapleMechanicsSample:
    """One staple observation derived only from a validated envelope."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveAnastomosisSceneSources
    staple_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "staple_path",
            _scene_path(self.staple_path, "staple_path"),
        )
        self.sources.validate_envelope(self.envelope)
        self.sources.staple_source_for(self.staple_path)
        if self._measurement is None:
            raise ValueError("staple measurement is absent")
        for name in (
            "max_abs_strain",
            "max_curvature_1_m",
            "elastic_energy_j",
            "dissipated_energy_j",
            "rod_time_s",
            "max_edge_damage_fraction",
            "left_reaction_force_n",
            "right_reaction_force_n",
        ):
            _nonnegative(getattr(self, name), name)
        failed_count = self._values[5]
        if int(failed_count) != failed_count or failed_count < 0.0:
            raise ValueError("failed edge count must be a non-negative integer")
        _fraction(
            self.max_edge_damage_fraction,
            "max_edge_damage_fraction",
        )
        if not math.isclose(
            self.rod_time_s,
            self.envelope.provenance.simulation_time_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "staple rod time does not match scene provenance"
            )

    @property
    def source(self) -> AnastomosisStapleSceneSource:
        return self.sources.staple_source_for(self.staple_path)

    @property
    def _measurement(self) -> SceneMeasurement | None:
        return _measurement(self.envelope, self.source.source_id)

    @property
    def _values(self) -> tuple[float, ...]:
        measurement = self._measurement
        if measurement is None:
            raise ValueError("staple measurement is absent")
        return measurement.value

    @property
    def max_abs_strain(self) -> float:
        return self._values[0]

    @property
    def max_curvature_1_m(self) -> float:
        return self._values[1]

    @property
    def elastic_energy_j(self) -> float:
        return self._values[2]

    @property
    def dissipated_energy_j(self) -> float:
        return self._values[3]

    @property
    def rod_time_s(self) -> float:
        return self._values[4]

    @property
    def failed_edge_count(self) -> int:
        return int(self._values[5])

    @property
    def max_edge_damage_fraction(self) -> float:
        return self._values[6]

    @property
    def left_reaction_force_n(self) -> float:
        return self._values[7]

    @property
    def right_reaction_force_n(self) -> float:
        return self._values[8]

    @property
    def active_attachment_prim_ids(self) -> frozenset[str]:
        measurement = self._measurement
        if measurement is None:
            return frozenset()
        return frozenset(measurement.attachment_prim_ids)

    @property
    def attachments_present(self) -> bool:
        return (
            self.active_attachment_prim_ids
            == self.source.expected_attachment_prim_ids
        )

    @property
    def topology_edges_intact(self) -> bool:
        return self.failed_edge_count == 0

    @property
    def force_disconnect_validated(self) -> bool:
        """The shared rod excludes failed edges from every force branch."""

        return True

    @property
    def step_index(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def digest_sha256(self) -> str:
        return self.envelope.digest_sha256

@dataclass(frozen=True)
class CollarSectorMechanicsSample:
    """One cohesive sector-side response from a validated envelope."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveAnastomosisSceneSources
    collar_path: str
    sector: int
    side: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "collar_path",
            _scene_path(self.collar_path, "collar_path"),
        )
        sector = int(self.sector)
        if not 0 <= sector < self.sources.collar_sector_count:
            raise ValueError("collar sector is outside the registered range")
        object.__setattr__(self, "sector", sector)
        side = str(self.side).lower()
        if side not in {"left", "right"}:
            raise ValueError("collar sector side must be left or right")
        object.__setattr__(self, "side", side)
        self.sources.validate_envelope(self.envelope)
        self.sources.collar_source_for(self.collar_path)
        if self._measurement is None:
            raise ValueError("collar sector measurement is absent")
        for name in (
            "contact_area_m2",
            "normal_pressure_pa",
            "normal_traction_pa",
            "resultant_force_n",
            "damage_fraction",
            "stored_energy_j_m2",
            "dissipated_energy_j_m2",
        ):
            _nonnegative(getattr(self, name), name)
        _fraction(self.damage_fraction, "damage_fraction")
        if self._values[8] not in (0.0, 1.0):
            raise ValueError("cohesive failure evidence must be binary")
        normal = _unit_vector3(
            tuple(self._values[11:14]),
            "interface_normal_w",
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-6)
            for left, right in zip(
                normal,
                self._values[11:14],
                strict=True,
            )
        ):
            raise ValueError("interface normal evidence must be unit length")

    @property
    def source(self) -> AnastomosisCollarSceneSource:
        return self.sources.collar_source_for(self.collar_path)

    @property
    def _measurement(self) -> SceneMeasurement | None:
        return _measurement(
            self.envelope,
            self.source.source_id(self.sector, self.side),
        )

    @property
    def _values(self) -> tuple[float, ...]:
        measurement = self._measurement
        if measurement is None:
            raise ValueError("collar sector measurement is absent")
        return measurement.value

    @property
    def contact_area_m2(self) -> float:
        return self._values[0]

    @property
    def normal_pressure_pa(self) -> float:
        return self._values[1]

    @property
    def normal_traction_pa(self) -> float:
        return self._values[2]

    @property
    def shear_traction_pa(self) -> tuple[float, float, float]:
        return tuple(self._values[3:6])

    @property
    def resultant_force_n(self) -> float:
        return self._values[6]

    @property
    def net_normal_traction_pa(self) -> float:
        return self.normal_traction_pa - self.normal_pressure_pa

    @property
    def damage_fraction(self) -> float:
        return self._values[7]

    @property
    def cohesive_failed(self) -> bool:
        return bool(self._values[8])

    @property
    def stored_energy_j_m2(self) -> float:
        return self._values[9]

    @property
    def dissipated_energy_j_m2(self) -> float:
        return self._values[10]

    @property
    def interface_normal_w(self) -> tuple[float, float, float]:
        return tuple(self._values[11:14])

    @property
    def attachment_present(self) -> bool:
        measurement = self._measurement
        if measurement is None:
            return False
        expected = self.source.attachment_prim_path(
            self.sector,
            self.side,
        )
        return measurement.attachment_prim_ids == (expected,)

    @property
    def step_index(self) -> int:
        return self.envelope.provenance.physics_step

    @property
    def time_s(self) -> float:
        return self.envelope.provenance.simulation_time_s

    @property
    def digest_sha256(self) -> str:
        return self.envelope.digest_sha256


@dataclass(frozen=True)
class AnastomosisSceneEvidence:
    """One coherent post-physics interval for the anastomosis workcell."""

    envelope: SceneEvidenceEnvelope
    sources: AdaptiveAnastomosisSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        for source_id in (
            self.sources.left_tissue_source_id,
            self.sources.right_tissue_source_id,
        ):
            measurement = _measurement(self.envelope, source_id)
            if measurement is None:
                raise ValueError(f"missing tissue source {source_id!r}")
            _nonnegative(
                measurement.value[0],
                "max_energy_density_j_m3",
            )
            _fraction(
                measurement.value[1],
                "max_damage_fraction",
            )
            _nonnegative(
                measurement.value[2],
                "dissipated_energy_j",
            )
            tissue_time = _nonnegative(
                measurement.value[3],
                "tissue_time_s",
            )
            if not math.isclose(
                tissue_time,
                self.envelope.provenance.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    "tissue time does not match scene provenance"
                )
        for name in (
            "chamber_pressure_pa",
            "measured_leak_flow_m3_s",
        ):
            _nonnegative(getattr(self, name), name)
        for staple in self.staples:
            StapleMechanicsSample(
                self.envelope,
                self.sources,
                staple.staple_path,
            )
        for collar in self.sources.collar_sources:
            for sector in range(self.sources.collar_sector_count):
                for side in ("left", "right"):
                    measurement = _measurement(
                        self.envelope,
                        collar.source_id(sector, side),
                    )
                    if measurement is not None:
                        CollarSectorMechanicsSample(
                            self.envelope,
                            self.sources,
                            collar.collar_path,
                            sector,
                            side,
                        )

    def _tissue_nodes(
        self,
        source_id: str,
    ) -> tuple[tuple[float, float, float], ...]:
        measurement = _measurement(self.envelope, source_id)
        if measurement is None:
            raise ValueError(f"missing tissue source {source_id!r}")
        coordinates = measurement.value[4:]
        return tuple(
            tuple(coordinates[index : index + 3])
            for index in range(0, len(coordinates), 3)
        )

    @property
    def left_lumen_nodes_w(self) -> tuple[tuple[float, float, float], ...]:
        return self._tissue_nodes(self.sources.left_tissue_source_id)

    @property
    def right_lumen_nodes_w(self) -> tuple[tuple[float, float, float], ...]:
        return self._tissue_nodes(self.sources.right_tissue_source_id)

    @property
    def chamber_pressure_pa(self) -> float:
        measurement = _measurement(
            self.envelope,
            self.sources.pressure_source_id,
        )
        if measurement is None:
            raise ValueError("pressure measurement is missing")
        return measurement.value[0]

    @property
    def measured_leak_flow_m3_s(self) -> float:
        measurement = _measurement(
            self.envelope,
            self.sources.leak_flow_source_id,
        )
        if measurement is None:
            raise ValueError("leak-flow measurement is missing")
        return measurement.value[0]

    @property
    def measured_leak_flow_ml_min(self) -> float:
        return self.measured_leak_flow_m3_s * 60.0 * 1.0e6

    @property
    def staples(self) -> tuple[StapleMechanicsSample, ...]:
        samples = []
        for source in self.sources.staple_sources:
            if _measurement(self.envelope, source.source_id) is not None:
                samples.append(
                    StapleMechanicsSample(
                        self.envelope,
                        self.sources,
                        source.staple_path,
                    )
                )
        return tuple(samples)

    def staple_for(self, staple_path: str) -> StapleMechanicsSample | None:
        return next(
            (
                sample
                for sample in self.staples
                if sample.staple_path == staple_path
            ),
            None,
        )

    def collar_sector_for(
        self,
        collar_path: str,
        sector: int,
        side: str,
    ) -> CollarSectorMechanicsSample | None:
        collar = self.sources.collar_source_for(collar_path)
        source_id = collar.source_id(int(sector), str(side).lower())
        if _measurement(self.envelope, source_id) is None:
            return None
        return CollarSectorMechanicsSample(
            self.envelope,
            self.sources,
            collar_path,
            sector,
            side,
        )

    @property
    def provenance(self) -> EvidenceProvenance:
        return self.envelope.provenance

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
    def source(self) -> str:
        return self.envelope.provenance.adapter_id


@runtime_checkable
class AnastomosisSceneEvidenceSource(Protocol):
    def sample_anastomosis_scene(self) -> AnastomosisSceneEvidence:
        ...


class AdaptiveAnastomosisSceneEvidence:
    """Collect shared mechanics and calibrated sensors into one envelope."""

    ADAPTER_ID = "dranmar.adaptive-anastomosis.scene-evidence"
    ADAPTER_VERSION = "3.0.0"

    def __init__(self, sources: AdaptiveAnastomosisSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(sources.registered_sources)

    def _tissue_measurement(
        self,
        *,
        tissue: LayeredSoftTissue,
        step: SoftTissueStepResult,
        source_id: str,
        source_prim_path: str,
        node_indices: tuple[int, ...],
        provenance: EvidenceProvenance,
        raw_sample_ids: tuple[str, ...],
    ) -> SceneMeasurement:
        if not isinstance(tissue, LayeredSoftTissue):
            raise TypeError("tissue source must be LayeredSoftTissue")
        if not isinstance(step, SoftTissueStepResult):
            raise TypeError("tissue step must be SoftTissueStepResult")
        if not step.parameter_status_by_layer:
            raise ValueError(
                "soft-tissue result must expose layer parameter status"
            )
        for layer_id, status in step.parameter_status_by_layer:
            _required_text(layer_id, "soft-tissue layer_id")
            _provisional_status(
                status,
                "soft-tissue parameter_status",
            )
        if (
            tissue.scene_component_id != source_prim_path
            or step.scene_component_id != source_prim_path
        ):
            raise ValueError(
                "soft-tissue state does not identify its registered prim"
            )
        if tissue.state is None:
            raise RuntimeError("soft-tissue state is not initialized")
        if (
            step.physics_step != provenance.physics_step
            or tissue.state.physics_step != step.physics_step
        ):
            raise ValueError(
                "soft-tissue state/result step does not match scene "
                "provenance"
            )
        if not math.isclose(
            step.dt_s,
            provenance.dt_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            step.previous_time_s,
            provenance.simulation_time_s - provenance.dt_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "soft-tissue result interval does not match scene provenance"
            )
        if not math.isclose(
            tissue.state.time_s,
            step.time_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "soft-tissue state and result timestamps disagree"
            )
        if not math.isclose(
            step.time_s,
            provenance.simulation_time_s,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "soft-tissue result time does not match scene provenance"
            )
        if any(index >= len(tissue.state.positions_m) for index in node_indices):
            raise ValueError("registered lumen node index is outside tissue state")
        coordinates = tuple(
            component
            for index in node_indices
            for component in tissue.state.positions_m[index]
        )
        return SceneMeasurement(
            source_id=source_id,
            value=(
                step.max_energy_density_j_m3,
                step.max_damage_fraction,
                step.dissipated_energy_j,
                step.time_s,
                *coordinates,
            ),
            source_prim_path=source_prim_path,
            raw_sample_ids=_raw_ids(raw_sample_ids, source_id),
        )

    def collect_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        left_tissue: LayeredSoftTissue,
        right_tissue: LayeredSoftTissue,
        left_tissue_step: SoftTissueStepResult,
        right_tissue_step: SoftTissueStepResult,
        pressure_flow: PressureFlowSceneObservation,
        raw_sample_ids_by_source: Mapping[
            str,
            tuple[str, ...],
        ],
        staple_observations: Mapping[
            str,
            StapleSceneObservation,
        ] | None = None,
        collar_observations: Mapping[
            tuple[str, int, str],
            CollarSectorSceneObservation,
        ] | None = None,
    ) -> AnastomosisSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        if not isinstance(pressure_flow, PressureFlowSceneObservation):
            raise TypeError(
                "pressure_flow must be PressureFlowSceneObservation"
            )
        if (
            pressure_flow.pressure_sensor_prim_path
            != self.sources.pressure_sensor_prim_path
            or pressure_flow.leak_flow_sensor_prim_path
            != self.sources.leak_flow_sensor_prim_path
        ):
            raise ValueError(
                "pressure/flow observation does not identify registered "
                "sensor prims"
            )
        if pressure_flow.physics_step != provenance.physics_step:
            raise ValueError(
                "pressure/flow observation step does not match provenance"
            )
        if not math.isclose(
            pressure_flow.simulation_time_s,
            provenance.simulation_time_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "pressure/flow observation time does not match provenance"
            )
        staples = (
            {} if staple_observations is None else dict(staple_observations)
        )
        collars = (
            {} if collar_observations is None else dict(collar_observations)
        )
        raw_ids = dict(raw_sample_ids_by_source)
        registered_staple_paths = {
            source.staple_path for source in self.sources.staple_sources
        }
        if not set(staples).issubset(registered_staple_paths):
            raise ValueError("staple observations contain an unregistered prim")
        registered_collar_keys = {
            (collar.collar_path, sector, side)
            for collar in self.sources.collar_sources
            for sector in range(self.sources.collar_sector_count)
            for side in ("left", "right")
        }
        normalized_collars = {
            (path, int(sector), str(side).lower()): observation
            for (path, sector, side), observation in collars.items()
        }
        if len(normalized_collars) != len(collars):
            raise ValueError(
                "collar observation keys collide after normalization"
            )
        if not set(normalized_collars).issubset(registered_collar_keys):
            raise ValueError(
                "collar observations contain an unregistered interface"
            )
        if not set(raw_ids).issubset(self.sources.registered_source_ids):
            raise ValueError("raw sample IDs reference an unregistered source")

        measurements = [
            self._tissue_measurement(
                tissue=left_tissue,
                step=left_tissue_step,
                source_id=self.sources.left_tissue_source_id,
                source_prim_path=self.sources.left_tissue_prim_path,
                node_indices=self.sources.left_lumen_node_indices,
                provenance=provenance,
                raw_sample_ids=raw_ids.get(
                    self.sources.left_tissue_source_id,
                    (),
                ),
            ),
            self._tissue_measurement(
                tissue=right_tissue,
                step=right_tissue_step,
                source_id=self.sources.right_tissue_source_id,
                source_prim_path=self.sources.right_tissue_prim_path,
                node_indices=self.sources.right_lumen_node_indices,
                provenance=provenance,
                raw_sample_ids=raw_ids.get(
                    self.sources.right_tissue_source_id,
                    (),
                ),
            ),
            SceneMeasurement(
                source_id=self.sources.pressure_source_id,
                value=(pressure_flow.chamber_pressure_pa,),
                source_prim_path=self.sources.pressure_sensor_prim_path,
                target_prim_path=self.sources.left_tissue_prim_path,
                contact_pair_id=(
                    f"{self.sources.left_tissue_prim_path}|"
                    f"{self.sources.right_tissue_prim_path}"
                ),
                raw_sample_ids=_raw_ids(
                    raw_ids.get(self.sources.pressure_source_id, ()),
                    self.sources.pressure_source_id,
                ),
            ),
            SceneMeasurement(
                source_id=self.sources.leak_flow_source_id,
                value=(pressure_flow.leak_flow_m3_s,),
                source_prim_path=self.sources.leak_flow_sensor_prim_path,
                target_prim_path=self.sources.left_tissue_prim_path,
                contact_pair_id=(
                    f"{self.sources.left_tissue_prim_path}|"
                    f"{self.sources.right_tissue_prim_path}"
                ),
                raw_sample_ids=_raw_ids(
                    raw_ids.get(self.sources.leak_flow_source_id, ()),
                    self.sources.leak_flow_source_id,
                ),
            ),
        ]
        tissue_pair_id = (
            f"{self.sources.left_tissue_prim_path}|"
            f"{self.sources.right_tissue_prim_path}"
        )
        for source in self.sources.staple_sources:
            observation = staples.get(source.staple_path)
            if observation is None:
                continue
            if not isinstance(observation, StapleSceneObservation):
                raise TypeError(
                    "staple observation must be StapleSceneObservation"
                )
            result = observation.rod_result
            rod = observation.rod
            if (
                result.component_id != source.staple_path
                or rod.component_id != source.staple_path
            ):
                raise ValueError(
                    "rod state/result do not identify the registered staple "
                    "prim"
                )
            _provisional_status(
                result.parameter_status,
                "rod result parameter_status",
            )
            _provisional_status(
                rod.material.parameter_status,
                "rod material parameter_status",
            )
            if len(rod.state.edge_intact) != len(rod.state.edge_damage):
                raise ValueError("staple rod edge state is inconsistent")
            if (
                result.physics_step != provenance.physics_step
                or rod.state.physics_step != result.physics_step
            ):
                raise ValueError(
                    "staple rod state/result step does not match scene "
                    "provenance"
                )
            if not math.isclose(
                result.dt_s,
                provenance.dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                result.previous_time_s,
                provenance.simulation_time_s - provenance.dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "staple rod result interval does not match scene "
                    "provenance"
                )
            if not math.isclose(
                result.time_s,
                provenance.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    "staple rod time does not match scene provenance"
                )
            if not math.isclose(
                rod.state.time_s,
                result.time_s,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError(
                    "staple rod state and result timestamps disagree"
                )
            if (
                source.left_anchor_node_index >= len(result.reaction_force_n)
                or source.right_anchor_node_index
                >= len(result.reaction_force_n)
            ):
                raise ValueError(
                    "staple anchor node is outside the rod observation"
                )
            active_attachments = frozenset(
                observation.active_attachment_prim_ids
            )
            if not active_attachments.issubset(
                source.expected_attachment_prim_ids
            ):
                raise ValueError(
                    "staple observation names an unregistered attachment"
                )
            measurements.append(
                SceneMeasurement(
                    source_id=source.source_id,
                    value=(
                        result.max_abs_strain,
                        result.max_curvature_1_m,
                        result.elastic_energy_j,
                        result.dissipated_energy_j,
                        result.time_s,
                        float(
                            sum(
                                not intact
                                for intact in rod.state.edge_intact
                            )
                        ),
                        max(rod.state.edge_damage, default=0.0),
                        _force_magnitude(
                            result.reaction_force_n[
                                source.left_anchor_node_index
                            ]
                        ),
                        _force_magnitude(
                            result.reaction_force_n[
                                source.right_anchor_node_index
                            ]
                        ),
                    ),
                    source_prim_path=source.staple_path,
                    target_prim_path=self.sources.left_tissue_prim_path,
                    contact_pair_id=tissue_pair_id,
                    attachment_prim_ids=tuple(
                        sorted(active_attachments)
                    ),
                    raw_sample_ids=_raw_ids(
                        raw_ids.get(source.source_id, ()),
                        source.source_id,
                    ),
                )
            )
        for key in sorted(normalized_collars):
            observation = normalized_collars[key]
            collar_path, sector, side = key
            if not isinstance(observation, CollarSectorSceneObservation):
                raise TypeError(
                    "collar observation must be "
                    "CollarSectorSceneObservation"
                )
            collar = self.sources.collar_source_for(collar_path)
            interface_path = collar.interface_prim_path(sector, side)
            if observation.interface.interface_id != interface_path:
                raise ValueError(
                    "cohesive interface does not identify the registered "
                    "collar-sector prim"
                )
            response = observation.response
            if response.step_index != provenance.physics_step:
                raise ValueError(
                    "cohesive response step does not match scene provenance"
                )
            if not math.isclose(
                response.time_s,
                provenance.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "cohesive response time does not match scene provenance"
                )
            expected_attachment = collar.attachment_prim_path(sector, side)
            if (
                observation.active_attachment_prim_id is not None
                and observation.active_attachment_prim_id
                != expected_attachment
            ):
                raise ValueError(
                    "collar observation names an unregistered attachment"
                )
            _provisional_status(
                response.parameter_status,
                "cohesive response parameter_status",
            )
            _provisional_status(
                observation.interface.law.parameter_status,
                "cohesive law parameter_status",
            )
            target = (
                self.sources.left_tissue_prim_path
                if side == "left"
                else self.sources.right_tissue_prim_path
            )
            if (
                response.body_a_component_id != interface_path
                or response.body_b_component_id != target
            ):
                raise ValueError(
                    "cohesive response bodies do not match the registered "
                    "interface-to-tissue direction"
                )
            measurements.append(
                SceneMeasurement(
                    source_id=collar.source_id(sector, side),
                    value=(
                        response.contact_area_m2,
                        response.normal_pressure_pa,
                        response.normal_traction_pa,
                        *response.shear_traction_pa,
                        response.resultant_force_n,
                        response.damage,
                        1.0 if response.failed else 0.0,
                        response.stored_energy_j_m2,
                        response.dissipated_energy_j_m2,
                        *response.normal_a_to_b,
                    ),
                    source_prim_path=interface_path,
                    target_prim_path=target,
                    contact_pair_id=f"{interface_path}|{target}",
                    attachment_prim_ids=(
                        ()
                        if observation.active_attachment_prim_id is None
                        else (observation.active_attachment_prim_id,)
                    ),
                    raw_sample_ids=_raw_ids(
                        raw_ids.get(
                            collar.source_id(sector, side),
                            (),
                        ),
                        collar.source_id(sector, side),
                    ),
                )
            )
        measured_source_ids = {
            measurement.source_id for measurement in measurements
        }
        if set(raw_ids) != measured_source_ids:
            raise ValueError(
                "raw sample IDs must exactly match the sources included in "
                "this anastomosis interval"
            )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=tuple(measurements),
        )
        self.registry.validate(envelope)
        return AnastomosisSceneEvidence(envelope, self.sources)


@dataclass
class AnastomosisEvidenceCursor:
    """Reject replayed or out-of-order anastomosis evidence."""

    last_step_index: int = field(default=-1, init=False)
    last_time_s: float = field(default=-math.inf, init=False)
    last_digest_sha256: str | None = field(default=None, init=False)
    episode_id: str | None = field(default=None, init=False)
    environment_id: str | None = field(default=None, init=False)

    def consume(
        self,
        evidence: AnastomosisSceneEvidence,
    ) -> AnastomosisSceneEvidence:
        if not isinstance(evidence, AnastomosisSceneEvidence):
            raise TypeError(
                "anastomosis controllers require AnastomosisSceneEvidence"
            )
        provenance = evidence.provenance
        if self.episode_id is None:
            self.episode_id = provenance.episode_id
            self.environment_id = provenance.environment_id
        elif (
            provenance.episode_id != self.episode_id
            or provenance.environment_id != self.environment_id
        ):
            raise ValueError(
                "anastomosis evidence cannot switch episode or environment"
            )
        if evidence.digest_sha256 == self.last_digest_sha256:
            raise ValueError("anastomosis evidence digest was replayed")
        if evidence.step_index <= self.last_step_index:
            raise ValueError(
                "anastomosis evidence must advance monotonically"
            )
        if evidence.time_s <= self.last_time_s:
            raise ValueError(
                "anastomosis evidence time must advance monotonically"
            )
        self.last_step_index = evidence.step_index
        self.last_time_s = evidence.time_s
        self.last_digest_sha256 = evidence.digest_sha256
        return evidence


__all__ = [
    "AdaptiveAnastomosisSceneEvidence",
    "AdaptiveAnastomosisSceneSources",
    "AnastomosisCollarSceneSource",
    "AnastomosisEvidenceCursor",
    "AnastomosisSceneEvidence",
    "AnastomosisSceneEvidenceSource",
    "AnastomosisStapleSceneSource",
    "CollarSectorMechanicsSample",
    "CollarSectorSceneObservation",
    "PressureFlowSceneObservation",
    "StapleMechanicsSample",
    "StapleSceneObservation",
]
