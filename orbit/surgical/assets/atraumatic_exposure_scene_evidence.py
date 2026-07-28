# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound contact and visibility evidence for atraumatic exposure.

This module intentionally contains no public setters for pad force, visibility,
retention, injury, or success.  It binds each capture cell to one contact pair,
one live attachment, one raw record set, and one post-physics interval.
Parameters and thresholds remain provisional and are not clinically validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol, runtime_checkable

from .scene_evidence import (
    EvidenceProvenance,
    SceneEvidenceEnvelope,
    SceneEvidenceRegistry,
    SceneEvidenceSource,
    SceneMeasurement,
)
from orbit.surgical.physics.contact import ContactSample


_SIDES = ("left", "right")
_CELL_COUNT_PER_SIDE = 6
_CONTACT_VALUE_COUNT = 18
_ARTICULATION_VALUE_COUNT = 8
_ARTICULATION_JOINT_ORDER = (
    "left_carriage_joint",
    "right_carriage_joint",
    "left_lift_joint",
    "right_lift_joint",
    "left_pitch_joint",
    "right_pitch_joint",
    "left_compliance_joint",
    "right_compliance_joint",
)
_CONTACT_QUANTITY = (
    "contact_point__cell_to_tissue_normal__separation__normal_force_"
    "magnitude__tangential_force_on_cell_by_tissue__cell_relative_"
    "tangential_velocity__contact_area__attachment_reaction_on_cell"
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


def _cell_index(value: object, label: str = "index") -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{label} must be an integral cell index in "
            f"[0, {_CELL_COUNT_PER_SIDE - 1}]"
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{label} must be an integral cell index in "
            f"[0, {_CELL_COUNT_PER_SIDE - 1}]"
        ) from error
    if result != value or not 0 <= result < _CELL_COUNT_PER_SIDE:
        raise ValueError(
            f"{label} must be an integral cell index in "
            f"[0, {_CELL_COUNT_PER_SIDE - 1}]"
        )
    return result


def _contact_values(
    sample: ContactSample,
    contact_area_m2: float,
    attachment_reaction_on_cell_n: tuple[float, float, float],
    *,
    contact_cell_prim_path: str,
    tissue_prim_path: str,
) -> tuple[float, ...]:
    body_pair = (sample.key.body_a, sample.key.body_b)
    expected_pair = (contact_cell_prim_path, tissue_prim_path)
    if body_pair == expected_pair:
        orientation = 1.0
    elif body_pair == tuple(reversed(expected_pair)):
        orientation = -1.0
    else:
        raise ValueError(
            "contact sample bodies do not match the registered cell-to-tissue "
            "pair"
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
        *attachment_reaction_on_cell_n,
    )


def _raw_ids(
    values_by_source: Mapping[str, tuple[str, ...]],
    source_id: str,
) -> tuple[str, ...]:
    try:
        values = tuple(
            sorted(
                _required_text(value, "raw_sample_id")
                for value in values_by_source[source_id]
            )
        )
    except KeyError as error:
        raise ValueError(
            f"exposure source {source_id!r} requires raw sample identity"
        ) from error
    if not values:
        raise ValueError(
            f"exposure source {source_id!r} requires raw sample identity"
        )
    if len(values) != len(set(values)):
        raise ValueError(
            f"exposure source {source_id!r} has duplicate raw sample IDs"
        )
    return values


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
        raise ValueError(f"exposure evidence is missing {source_id!r}") from error


@dataclass(frozen=True)
class ExposureCellSceneSource:
    side: str
    index: int
    source_id: str
    contact_cell_prim_path: str
    tissue_prim_path: str
    attachment_prim_path: str

    def __post_init__(self) -> None:
        side = str(self.side).lower()
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES!r}")
        object.__setattr__(self, "side", side)
        index = _cell_index(self.index)
        object.__setattr__(self, "index", index)
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )
        for name in (
            "contact_cell_prim_path",
            "tissue_prim_path",
            "attachment_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )


@dataclass(frozen=True)
class AtraumaticExposureSceneSources:
    workcell_id: str
    cell_sources: tuple[ExposureCellSceneSource, ...]
    visibility_source_id: str
    visibility_sensor_prim_path: str
    roi_prim_path: str
    articulation_source_id: str
    tool_prim_path: str
    calibration_profile_id: str = "dranmar-exposure-engineering-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workcell_id",
            _required_text(self.workcell_id, "workcell_id"),
        )
        cells = tuple(self.cell_sources)
        if any(not isinstance(value, ExposureCellSceneSource) for value in cells):
            raise TypeError(
                "cell_sources must contain ExposureCellSceneSource values"
            )
        expected_keys = {
            (side, index)
            for side in _SIDES
            for index in range(_CELL_COUNT_PER_SIDE)
        }
        actual_keys = {(source.side, source.index) for source in cells}
        if actual_keys != expected_keys or len(cells) != len(expected_keys):
            raise ValueError(
                "cell_sources must register exactly six unique cells per side"
            )
        object.__setattr__(
            self,
            "cell_sources",
            tuple(sorted(cells, key=lambda value: (value.side, value.index))),
        )
        object.__setattr__(
            self,
            "visibility_source_id",
            _required_text(self.visibility_source_id, "visibility_source_id"),
        )
        object.__setattr__(
            self,
            "articulation_source_id",
            _required_text(
                self.articulation_source_id,
                "articulation_source_id",
            ),
        )
        for name in (
            "visibility_sensor_prim_path",
            "roi_prim_path",
            "tool_prim_path",
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
        source_ids = (
            *(source.source_id for source in self.cell_sources),
            self.visibility_source_id,
            self.articulation_source_id,
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("exposure source IDs must be unique")
        for label, values in (
            (
                "contact cell paths",
                tuple(source.contact_cell_prim_path for source in self.cell_sources),
            ),
            (
                "attachment paths",
                tuple(source.attachment_prim_path for source in self.cell_sources),
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"exposure {label} must be unique")

    @property
    def registered_sources(self) -> tuple[SceneEvidenceSource, ...]:
        cells = tuple(
            SceneEvidenceSource(
                source_id=source.source_id,
                prim_path=source.contact_cell_prim_path,
                quantity=_CONTACT_QUANTITY,
                unit="SI_vector",
                coordinate_frame="world",
                calibration_profile_id=self.calibration_profile_id,
            )
            for source in self.cell_sources
        )
        return (
            *cells,
            SceneEvidenceSource(
                source_id=self.visibility_source_id,
                prim_path=self.visibility_sensor_prim_path,
                quantity="visible_and_total_roi_sample_count",
                unit="count",
                coordinate_frame="registered_camera",
                calibration_profile_id=self.calibration_profile_id,
            ),
            SceneEvidenceSource(
                source_id=self.articulation_source_id,
                prim_path=self.tool_prim_path,
                quantity=(
                    "post_physics_joint_positions__left_carriage__"
                    "right_carriage__left_lift__right_lift__left_pitch__"
                    "right_pitch__left_compliance__right_compliance"
                ),
                unit="ordered_m_and_rad_vector",
                coordinate_frame="registered_articulation_joint_coordinates",
                calibration_profile_id=self.calibration_profile_id,
            ),
        )

    @property
    def expected_source_ids(self) -> frozenset[str]:
        return frozenset(source.source_id for source in self.registered_sources)

    def cell_source(self, side: str, index: int) -> ExposureCellSceneSource:
        key = (str(side).lower(), _cell_index(index))
        try:
            return next(
                source
                for source in self.cell_sources
                if (source.side, source.index) == key
            )
        except StopIteration as error:
            raise ValueError(f"exposure cell {key!r} is not registered") from error

    def validate_envelope(self, envelope: SceneEvidenceEnvelope) -> None:
        provenance = envelope.provenance
        if provenance.adapter_id != AtraumaticExposureSceneEvidenceAdapter.ADAPTER_ID:
            raise ValueError("exposure evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != AtraumaticExposureSceneEvidenceAdapter.ADAPTER_VERSION
        ):
            raise ValueError("exposure evidence has the wrong adapter_version")
        SceneEvidenceRegistry(self.registered_sources).validate(envelope)
        measurements = {
            measurement.source_id: measurement
            for measurement in envelope.measurements
        }
        if frozenset(measurements) != self.expected_source_ids:
            raise ValueError(
                "exposure evidence must contain the exact registered source set"
            )
        for source in self.cell_sources:
            measurement = measurements[source.source_id]
            if len(measurement.value) != _CONTACT_VALUE_COUNT:
                raise ValueError(
                    f"exposure cell {source.source_id!r} must contain "
                    f"{_CONTACT_VALUE_COUNT} values"
                )
            if measurement.target_prim_path != source.tissue_prim_path:
                raise ValueError(
                    f"exposure cell {source.source_id!r} has the wrong tissue target"
                )
            if measurement.contact_pair_id != (
                f"{source.contact_cell_prim_path}|{source.tissue_prim_path}"
            ):
                raise ValueError(
                    f"exposure cell {source.source_id!r} has the wrong contact pair"
                )
            if measurement.attachment_prim_ids not in (
                (),
                (source.attachment_prim_path,),
            ):
                raise ValueError(
                    f"exposure cell {source.source_id!r} has an unregistered "
                    "attachment"
                )
            if not measurement.raw_sample_ids:
                raise ValueError(
                    f"exposure cell {source.source_id!r} requires raw identity"
                )
        visibility = measurements[self.visibility_source_id]
        if (
            len(visibility.value) != 2
            or visibility.target_prim_path != self.roi_prim_path
            or not visibility.raw_sample_ids
        ):
            raise ValueError(
                "visibility evidence requires two counts, the registered ROI, "
                "and raw sample identity"
            )
        articulation = measurements[self.articulation_source_id]
        if (
            len(articulation.value) != _ARTICULATION_VALUE_COUNT
            or articulation.source_prim_path != self.tool_prim_path
            or not articulation.raw_sample_ids
        ):
            raise ValueError(
                "articulation evidence requires eight registered joint "
                "positions, the exact tool prim, and raw sample identity"
            )


@dataclass(frozen=True)
class ExposureCellSceneObservation:
    physics_step: int
    simulation_time_s: float
    side: str
    index: int
    contact: ContactSample
    contact_area_m2: float
    active_attachment_prim_ids: tuple[str, ...]
    attachment_reaction_on_cell_n: tuple[float, float, float]

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
        side = str(self.side).lower()
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES!r}")
        object.__setattr__(self, "side", side)
        index = _cell_index(self.index)
        object.__setattr__(self, "index", index)
        if not isinstance(self.contact, ContactSample):
            raise TypeError("contact must be a shared ContactSample")
        object.__setattr__(
            self,
            "contact_area_m2",
            _nonnegative(self.contact_area_m2, "contact_area_m2"),
        )
        if self.contact.normal_force_n > 0.0 and self.contact_area_m2 <= 0.0:
            raise ValueError("positive cell contact force requires positive area")
        attachments = tuple(
            sorted(
                _scene_path(value, "active_attachment_prim_id")
                for value in self.active_attachment_prim_ids
            )
        )
        if len(attachments) != len(set(attachments)):
            raise ValueError(
                "active_attachment_prim_ids must not contain duplicates"
            )
        object.__setattr__(
            self,
            "active_attachment_prim_ids",
            attachments,
        )
        reaction = tuple(
            _finite(value, "attachment_reaction_on_cell_n component")
            for value in self.attachment_reaction_on_cell_n
        )
        if len(reaction) != 3:
            raise ValueError(
                "attachment_reaction_on_cell_n must contain three values"
            )
        if not attachments and any(abs(value) > 1.0e-12 for value in reaction):
            raise ValueError(
                "attachment reaction must be zero when no attachment is active"
            )
        object.__setattr__(
            self,
            "attachment_reaction_on_cell_n",
            reaction,
        )


@dataclass(frozen=True)
class ExposureVisibilitySceneObservation:
    physics_step: int
    simulation_time_s: float
    visible_roi_sample_count: int
    total_roi_sample_count: int

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
        visible = int(self.visible_roi_sample_count)
        total = int(self.total_roi_sample_count)
        if (
            visible != self.visible_roi_sample_count
            or total != self.total_roi_sample_count
            or visible < 0
            or total <= 0
            or visible > total
        ):
            raise ValueError(
                "visibility counts must be integers with 0 <= visible <= total"
            )
        object.__setattr__(self, "visible_roi_sample_count", visible)
        object.__setattr__(self, "total_roi_sample_count", total)


@dataclass(frozen=True)
class ExposureArticulationSceneObservation:
    physics_step: int
    simulation_time_s: float
    left_carriage_position_m: float
    right_carriage_position_m: float
    left_lift_position_m: float
    right_lift_position_m: float
    left_pitch_position_rad: float
    right_pitch_position_rad: float
    left_compliance_position_m: float
    right_compliance_position_m: float

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
        bounds = {
            "left_carriage_position_m": (0.0, 0.040),
            "right_carriage_position_m": (-0.040, 0.0),
            "left_lift_position_m": (-0.025, 0.030),
            "right_lift_position_m": (-0.025, 0.030),
            "left_pitch_position_rad": (
                math.radians(-42.0),
                math.radians(72.0),
            ),
            "right_pitch_position_rad": (
                math.radians(-72.0),
                math.radians(42.0),
            ),
            "left_compliance_position_m": (-0.006, 0.0),
            "right_compliance_position_m": (-0.006, 0.0),
        }
        for name, (low, high) in bounds.items():
            value = _finite(getattr(self, name), name)
            if not low <= value <= high:
                raise ValueError(
                    f"{name}={value} is outside [{low}, {high}]"
                )
            object.__setattr__(self, name, value)

    @property
    def ordered_joint_positions_m(self) -> tuple[float, ...]:
        return (
            self.left_carriage_position_m,
            self.right_carriage_position_m,
            self.left_lift_position_m,
            self.right_lift_position_m,
            self.left_pitch_position_rad,
            self.right_pitch_position_rad,
            self.left_compliance_position_m,
            self.right_compliance_position_m,
        )


@dataclass(frozen=True)
class ExposureCellMechanicsSample:
    envelope: SceneEvidenceEnvelope
    sources: AtraumaticExposureSceneSources
    side: str
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", str(self.side).lower())
        object.__setattr__(self, "index", _cell_index(self.index))
        self.sources.validate_envelope(self.envelope)
        self.sources.cell_source(self.side, self.index)
        _nonnegative(self.normal_force_n, "normal_force_n")
        _nonnegative(self.contact_area_m2, "contact_area_m2")

    @property
    def source(self) -> ExposureCellSceneSource:
        return self.sources.cell_source(self.side, self.index)

    @property
    def measurement(self) -> SceneMeasurement:
        return _measurement(self.envelope, self.source.source_id)

    @property
    def normal_force_n(self) -> float:
        return self.measurement.value[7]

    @property
    def slip_speed_m_s(self) -> float:
        return math.sqrt(
            sum(value * value for value in self.measurement.value[11:14])
        )

    @property
    def contact_area_m2(self) -> float:
        return self.measurement.value[14]

    @property
    def tangential_force_n(self) -> float:
        return math.sqrt(
            sum(value * value for value in self.measurement.value[8:11])
        )

    @property
    def resultant_contact_force_n(self) -> float:
        return math.hypot(self.normal_force_n, self.tangential_force_n)

    @property
    def attachment_reaction_on_cell_n(self) -> tuple[float, float, float]:
        return tuple(self.measurement.value[15:18])

    @property
    def attachment_reaction_force_n(self) -> float:
        return math.sqrt(
            sum(value * value for value in self.attachment_reaction_on_cell_n)
        )

    @property
    def transmitted_load_n(self) -> float:
        """Conservative upper-bound proxy for load transmitted through a cell.

        The contact resultant and attachment reaction can represent overlapping
        portions of the same load path.  Adding their magnitudes deliberately
        permits double counting so co-directed load cannot be hidden by taking
        only the larger signal.
        """

        return (
            self.resultant_contact_force_n
            + self.attachment_reaction_force_n
        )

    @property
    def attachment_present(self) -> bool:
        return self.measurement.attachment_prim_ids == (
            self.source.attachment_prim_path,
        )


@dataclass(frozen=True)
class AtraumaticExposureSceneEvidence:
    envelope: SceneEvidenceEnvelope
    sources: AtraumaticExposureSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        visible, total = self.visibility_counts
        if (
            int(visible) != visible
            or int(total) != total
            or visible < 0.0
            or total <= 0.0
            or visible > total
        ):
            raise ValueError("visibility evidence contains invalid counts")
        for source in self.sources.cell_sources:
            ExposureCellMechanicsSample(
                self.envelope,
                self.sources,
                source.side,
                source.index,
            )

    def cell(self, side: str, index: int) -> ExposureCellMechanicsSample:
        return ExposureCellMechanicsSample(
            self.envelope,
            self.sources,
            side,
            index,
        )

    def cells(self, side: str) -> tuple[ExposureCellMechanicsSample, ...]:
        normalized = str(side).lower()
        if normalized not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES!r}")
        return tuple(
            self.cell(normalized, index)
            for index in range(_CELL_COUNT_PER_SIDE)
        )

    def total_force_n(self, side: str) -> float:
        """Return summed normal contact force for normal-force control."""

        return sum(sample.normal_force_n for sample in self.cells(side))

    def total_transmitted_load_n(self, side: str) -> float:
        """Return conservative summed per-cell contact/attachment load."""

        return sum(sample.transmitted_load_n for sample in self.cells(side))

    @property
    def visibility_counts(self) -> tuple[float, float]:
        return _measurement(
            self.envelope,
            self.sources.visibility_source_id,
        ).value

    @property
    def visible_fraction(self) -> float:
        visible, total = self.visibility_counts
        return visible / total

    @property
    def measured_joint_positions_m(self) -> dict[str, float]:
        values = _measurement(
            self.envelope,
            self.sources.articulation_source_id,
        ).value
        return dict(zip(_ARTICULATION_JOINT_ORDER, values, strict=True))

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
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256


@runtime_checkable
class AtraumaticExposureSceneEvidenceSource(Protocol):
    def sample_atraumatic_exposure_scene(
        self,
    ) -> AtraumaticExposureSceneEvidence:
        ...


class AtraumaticExposureSceneEvidenceAdapter:
    ADAPTER_ID = "dranmar.atraumatic-exposure.scene-evidence"
    ADAPTER_VERSION = "3.0.0"

    def __init__(self, sources: AtraumaticExposureSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(sources.registered_sources)

    def collect_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        cell_observations: Mapping[
            tuple[str, int],
            ExposureCellSceneObservation,
        ],
        visibility: ExposureVisibilitySceneObservation,
        articulation: ExposureArticulationSceneObservation,
        raw_sample_ids_by_source: Mapping[str, tuple[str, ...]],
    ) -> AtraumaticExposureSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        if not isinstance(visibility, ExposureVisibilitySceneObservation):
            raise TypeError(
                "visibility must be ExposureVisibilitySceneObservation"
            )
        if not isinstance(
            articulation,
            ExposureArticulationSceneObservation,
        ):
            raise TypeError(
                "articulation must be ExposureArticulationSceneObservation"
            )
        expected_keys = {
            (source.side, source.index) for source in self.sources.cell_sources
        }
        normalized_observations = {
            (
                str(side).lower(),
                _cell_index(index, "cell observation index"),
            ): observation
            for (side, index), observation in cell_observations.items()
        }
        if (
            set(normalized_observations) != expected_keys
            or len(normalized_observations) != len(cell_observations)
        ):
            raise ValueError(
                "cell observations must exactly match registered exposure cells"
            )
        if frozenset(raw_sample_ids_by_source) != self.sources.expected_source_ids:
            raise ValueError(
                "raw sample identity must exactly match registered exposure "
                "sources"
            )
        if (
            visibility.physics_step != provenance.physics_step
            or not math.isclose(
                visibility.simulation_time_s,
                provenance.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "visibility observation is not from the envelope interval"
            )
        if (
            articulation.physics_step != provenance.physics_step
            or not math.isclose(
                articulation.simulation_time_s,
                provenance.simulation_time_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "articulation observation is not from the envelope interval"
            )
        measurements: list[SceneMeasurement] = []
        for source in self.sources.cell_sources:
            observation = normalized_observations[(source.side, source.index)]
            if not isinstance(observation, ExposureCellSceneObservation):
                raise TypeError(
                    "cell_observations must contain "
                    "ExposureCellSceneObservation values"
                )
            if (observation.side, observation.index) != (
                source.side,
                source.index,
            ):
                raise ValueError("exposure cell key and observation disagree")
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
                    f"exposure cell {source.source_id!r} is not from the "
                    "envelope interval"
                )
            if {
                observation.contact.key.body_a,
                observation.contact.key.body_b,
            } != {
                source.contact_cell_prim_path,
                source.tissue_prim_path,
            }:
                raise ValueError(
                    f"exposure cell {source.source_id!r} has the wrong contact "
                    "bodies"
                )
            if observation.active_attachment_prim_ids not in (
                (),
                (source.attachment_prim_path,),
            ):
                raise ValueError(
                    f"exposure cell {source.source_id!r} names an "
                    "unregistered attachment"
                )
            measurements.append(
                SceneMeasurement(
                    source_id=source.source_id,
                    value=_contact_values(
                        observation.contact,
                        observation.contact_area_m2,
                        observation.attachment_reaction_on_cell_n,
                        contact_cell_prim_path=source.contact_cell_prim_path,
                        tissue_prim_path=source.tissue_prim_path,
                    ),
                    source_prim_path=source.contact_cell_prim_path,
                    target_prim_path=source.tissue_prim_path,
                    contact_pair_id=(
                        f"{source.contact_cell_prim_path}|"
                        f"{source.tissue_prim_path}"
                    ),
                    attachment_prim_ids=(
                        observation.active_attachment_prim_ids
                    ),
                    raw_sample_ids=_raw_ids(
                        raw_sample_ids_by_source,
                        source.source_id,
                    ),
                )
            )
        measurements.append(
            SceneMeasurement(
                source_id=self.sources.visibility_source_id,
                value=(
                    float(visibility.visible_roi_sample_count),
                    float(visibility.total_roi_sample_count),
                ),
                source_prim_path=self.sources.visibility_sensor_prim_path,
                target_prim_path=self.sources.roi_prim_path,
                raw_sample_ids=_raw_ids(
                    raw_sample_ids_by_source,
                    self.sources.visibility_source_id,
                ),
            )
        )
        measurements.append(
            SceneMeasurement(
                source_id=self.sources.articulation_source_id,
                value=articulation.ordered_joint_positions_m,
                source_prim_path=self.sources.tool_prim_path,
                raw_sample_ids=_raw_ids(
                    raw_sample_ids_by_source,
                    self.sources.articulation_source_id,
                ),
            )
        )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=tuple(measurements),
        )
        self.registry.validate(envelope)
        return AtraumaticExposureSceneEvidence(envelope, self.sources)


@dataclass
class AtraumaticExposureEvidenceCursor:
    episode_id: str | None = field(default=None, init=False)
    environment_id: str | None = field(default=None, init=False)
    topology_revision: str | None = field(default=None, init=False)
    source_registration: AtraumaticExposureSceneSources | None = field(
        default=None,
        init=False,
    )
    last_physics_step: int = field(default=-1, init=False)
    last_simulation_time_s: float = field(default=-1.0, init=False)
    consumed_digests: set[str] = field(default_factory=set, init=False)
    consumed_raw_sample_ids_by_source: dict[str, set[str]] = field(
        default_factory=dict,
        init=False,
    )

    def validate(
        self,
        evidence: AtraumaticExposureSceneEvidence,
        *,
        allow_topology_change: bool = False,
    ) -> AtraumaticExposureSceneEvidence:
        """Check identity, registration, clock, and replay without mutation."""

        if not isinstance(evidence, AtraumaticExposureSceneEvidence):
            raise TypeError(
                "exposure control requires AtraumaticExposureSceneEvidence"
            )
        provenance = evidence.envelope.provenance
        if self.episode_id is not None and (
            provenance.episode_id,
            provenance.environment_id,
        ) != (
            self.episode_id,
            self.environment_id,
        ):
            raise ValueError(
                "exposure episode/environment identity changed without an "
                "explicit episode reset"
            )
        if (
            self.episode_id is not None
            and not allow_topology_change
            and provenance.topology_revision != self.topology_revision
        ):
            raise ValueError(
                "exposure topology changed outside an authorized transition"
            )
        if (
            self.episode_id is not None
            and evidence.sources != self.source_registration
        ):
            raise ValueError(
                "exposure scene source, workcell, calibration, visibility, "
                "or ROI registration changed without reset"
            )
        if provenance.physics_step <= self.last_physics_step:
            raise ValueError("exposure physics step must increase")
        if self.last_physics_step >= 0:
            if provenance.simulation_time_s <= self.last_simulation_time_s:
                raise ValueError("exposure simulation time must increase")
            if not math.isclose(
                provenance.simulation_time_s - self.last_simulation_time_s,
                provenance.dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "exposure dt does not match the consumed clock interval"
                )
        digest = evidence.envelope.digest_sha256
        if digest in self.consumed_digests:
            raise ValueError("exposure evidence replay is forbidden")
        for measurement in evidence.envelope.measurements:
            consumed = self.consumed_raw_sample_ids_by_source.get(
                measurement.source_id,
                set(),
            )
            reused = consumed.intersection(measurement.raw_sample_ids)
            if reused:
                raise ValueError(
                    "exposure raw sample replay is forbidden for "
                    f"{measurement.source_id!r}: {tuple(sorted(reused))!r}"
                )
        return evidence

    def commit(
        self,
        evidence: AtraumaticExposureSceneEvidence,
        *,
        allow_topology_change: bool = False,
    ) -> AtraumaticExposureSceneEvidence:
        """Atomically advance the cursor after all owning-system preflights."""

        evidence = self.validate(
            evidence,
            allow_topology_change=allow_topology_change,
        )
        provenance = evidence.envelope.provenance
        if self.episode_id is None:
            self.episode_id = provenance.episode_id
            self.environment_id = provenance.environment_id
            self.topology_revision = provenance.topology_revision
            self.source_registration = evidence.sources
        elif allow_topology_change:
            self.topology_revision = provenance.topology_revision
        digest = evidence.envelope.digest_sha256
        self.consumed_digests.add(digest)
        for measurement in evidence.envelope.measurements:
            self.consumed_raw_sample_ids_by_source.setdefault(
                measurement.source_id,
                set(),
            ).update(measurement.raw_sample_ids)
        self.last_physics_step = provenance.physics_step
        self.last_simulation_time_s = provenance.simulation_time_s
        return evidence

    def commit_topology_transition(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> AtraumaticExposureSceneEvidence:
        """Commit one fresh revision without releasing episode/source identity."""

        if self.episode_id is None or self.topology_revision is None:
            raise RuntimeError(
                "topology transition requires an initialized exposure cursor"
            )
        next_revision = evidence.envelope.provenance.topology_revision
        if next_revision == self.topology_revision:
            raise ValueError(
                "authorized topology transition requires a new revision"
            )
        return self.commit(evidence, allow_topology_change=True)

    def consume(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> AtraumaticExposureSceneEvidence:
        """Validate and commit one same-topology interval."""

        return self.commit(evidence)

    def reset(self, *, preserve_consumed_digests: bool = True) -> None:
        """Start a new episode, preserving digest and raw-ID replay history."""
        if not isinstance(preserve_consumed_digests, bool):
            raise TypeError("preserve_consumed_digests must be bool")
        self.episode_id = None
        self.environment_id = None
        self.topology_revision = None
        self.source_registration = None
        self.last_physics_step = -1
        self.last_simulation_time_s = -1.0
        if not preserve_consumed_digests:
            self.consumed_digests.clear()
            self.consumed_raw_sample_ids_by_source.clear()


__all__ = [
    "AtraumaticExposureEvidenceCursor",
    "AtraumaticExposureSceneEvidence",
    "AtraumaticExposureSceneEvidenceAdapter",
    "AtraumaticExposureSceneEvidenceSource",
    "AtraumaticExposureSceneSources",
    "ExposureArticulationSceneObservation",
    "ExposureCellMechanicsSample",
    "ExposureCellSceneObservation",
    "ExposureCellSceneSource",
    "ExposureVisibilitySceneObservation",
]
