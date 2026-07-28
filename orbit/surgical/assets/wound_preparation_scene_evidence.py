# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound debridement evidence for the wound-preparation workcell.

The adapter accepts shared contact observations, not debridement progress or
release decisions.  Release work is derived downstream from tangential force
and relative tangential velocity for the exact registered tool/debris pair.
All parameters remain provisional engineering seeds; no physical or clinical
calibration is implied.
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


_CONTACT_VALUE_COUNT = 14


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


def _contact_values(sample: ContactSample) -> tuple[float, ...]:
    return (
        *sample.point_m,
        *sample.normal_a_to_b,
        sample.separation_m,
        sample.normal_force_n,
        *sample.tangential_force_n,
        *sample.relative_tangential_velocity_m_s,
    )


def _raw_ids(
    values_by_source: Mapping[str, tuple[str, ...]],
    source_id: str,
) -> tuple[str, ...]:
    try:
        values = tuple(
            sorted(_required_text(value, "raw_sample_id") for value in values_by_source[source_id])
        )
    except KeyError as error:
        raise ValueError(
            f"wound-preparation source {source_id!r} requires raw sample identity"
        ) from error
    if not values:
        raise ValueError(
            f"wound-preparation source {source_id!r} requires raw sample identity"
        )
    if len(values) != len(set(values)):
        raise ValueError(
            f"wound-preparation source {source_id!r} has duplicate raw sample IDs"
        )
    return values


def _measurement(
    envelope: SceneEvidenceEnvelope,
    source_id: str,
) -> SceneMeasurement:
    try:
        return next(
            value
            for value in envelope.measurements
            if value.source_id == source_id
        )
    except StopIteration as error:
        raise ValueError(
            f"wound-preparation evidence is missing source {source_id!r}"
        ) from error


@dataclass(frozen=True)
class DebridementDebrisSource:
    """One debris body and the only attachment allowed to retain it."""

    source_id: str
    debris_prim_path: str
    attachment_prim_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "debris_prim_path",
            _scene_path(self.debris_prim_path, "debris_prim_path"),
        )
        object.__setattr__(
            self,
            "attachment_prim_path",
            _scene_path(self.attachment_prim_path, "attachment_prim_path"),
        )


@dataclass(frozen=True)
class WoundPreparationSceneSources:
    """Exact contact, debris, and attachment identities for one workcell."""

    workcell_id: str
    debridement_contact_prim_path: str
    debris_sources: tuple[DebridementDebrisSource, ...]
    calibration_profile_id: str = "dranmar-wound-preparation-engineering-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workcell_id",
            _required_text(self.workcell_id, "workcell_id"),
        )
        object.__setattr__(
            self,
            "debridement_contact_prim_path",
            _scene_path(
                self.debridement_contact_prim_path,
                "debridement_contact_prim_path",
            ),
        )
        debris = tuple(self.debris_sources)
        if not debris:
            raise ValueError("at least one debris source must be registered")
        if any(not isinstance(value, DebridementDebrisSource) for value in debris):
            raise TypeError(
                "debris_sources must contain DebridementDebrisSource values"
            )
        object.__setattr__(self, "debris_sources", debris)
        object.__setattr__(
            self,
            "calibration_profile_id",
            _required_text(
                self.calibration_profile_id,
                "calibration_profile_id",
            ),
        )
        for label, values in (
            ("source IDs", tuple(value.source_id for value in debris)),
            ("debris prim paths", tuple(value.debris_prim_path for value in debris)),
            (
                "attachment prim paths",
                tuple(value.attachment_prim_path for value in debris),
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"wound-preparation {label} must be unique")

    @property
    def registered_sources(self) -> tuple[SceneEvidenceSource, ...]:
        return tuple(
            SceneEvidenceSource(
                source_id=source.source_id,
                prim_path=self.debridement_contact_prim_path,
                quantity="debris_contact_force_and_relative_motion",
                unit="m_unitless_m_N_N_m_per_s",
                coordinate_frame="world",
                calibration_profile_id=self.calibration_profile_id,
            )
            for source in self.debris_sources
        )

    @property
    def expected_source_ids(self) -> frozenset[str]:
        return frozenset(source.source_id for source in self.debris_sources)

    def source_for_debris(self, debris_prim_path: str) -> DebridementDebrisSource:
        path = _scene_path(debris_prim_path, "debris_prim_path")
        try:
            return next(
                source
                for source in self.debris_sources
                if source.debris_prim_path == path
            )
        except StopIteration as error:
            raise ValueError(f"debris prim {path!r} is not registered") from error

    def validate_envelope(self, envelope: SceneEvidenceEnvelope) -> None:
        provenance = envelope.provenance
        if provenance.adapter_id != WoundPreparationSceneEvidenceAdapter.ADAPTER_ID:
            raise ValueError("wound-preparation evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != WoundPreparationSceneEvidenceAdapter.ADAPTER_VERSION
        ):
            raise ValueError(
                "wound-preparation evidence has the wrong adapter_version"
            )
        SceneEvidenceRegistry(self.registered_sources).validate(envelope)
        measurements = {
            measurement.source_id: measurement
            for measurement in envelope.measurements
        }
        if frozenset(measurements) != self.expected_source_ids:
            raise ValueError(
                "wound-preparation evidence must contain the exact registered "
                "debris source set"
            )
        for source in self.debris_sources:
            measurement = measurements[source.source_id]
            if len(measurement.value) != _CONTACT_VALUE_COUNT:
                raise ValueError(
                    f"debris source {source.source_id!r} must contain "
                    f"{_CONTACT_VALUE_COUNT} contact values"
                )
            if measurement.target_prim_path != source.debris_prim_path:
                raise ValueError(
                    f"debris source {source.source_id!r} has the wrong target"
                )
            if measurement.contact_pair_id != (
                f"{self.debridement_contact_prim_path}|{source.debris_prim_path}"
            ):
                raise ValueError(
                    f"debris source {source.source_id!r} has the wrong contact pair"
                )
            if not measurement.raw_sample_ids:
                raise ValueError(
                    f"debris source {source.source_id!r} requires raw sample "
                    "identity"
                )
            if measurement.attachment_prim_ids not in (
                (),
                (source.attachment_prim_path,),
            ):
                raise ValueError(
                    f"debris source {source.source_id!r} names an unregistered "
                    "attachment"
                )


@dataclass(frozen=True)
class DebridementContactObservation:
    """One exact post-physics contact and its live retaining attachment."""

    debris_prim_path: str
    contact: ContactSample
    active_attachment_prim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "debris_prim_path",
            _scene_path(self.debris_prim_path, "debris_prim_path"),
        )
        if not isinstance(self.contact, ContactSample):
            raise TypeError("contact must be a shared ContactSample")
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


@dataclass(frozen=True)
class DebridementMechanicsSample:
    """Contact and topology values decoded from a validated envelope."""

    envelope: SceneEvidenceEnvelope
    sources: WoundPreparationSceneSources
    debris_prim_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "debris_prim_path",
            _scene_path(self.debris_prim_path, "debris_prim_path"),
        )
        self.sources.validate_envelope(self.envelope)
        self.sources.source_for_debris(self.debris_prim_path)
        if self.normal_force_n < 0.0:
            raise ValueError("normal_force_n must be nonnegative")
        for value in (
            *self.tangential_force_n,
            *self.relative_tangential_velocity_m_s,
        ):
            _finite(value, "contact vector component")
        signed_power = sum(
            force * velocity
            for force, velocity in zip(
                self.tangential_force_n,
                self.relative_tangential_velocity_m_s,
                strict=True,
            )
        )
        tolerance = max(
            1.0e-12,
            1.0e-9
            * math.sqrt(sum(value * value for value in self.tangential_force_n))
            * max(self.slip_speed_m_s, 1.0e-12),
        )
        if signed_power > tolerance:
            raise ValueError(
                "debridement contact force adds energy under the shared "
                "contact sign convention"
            )

    @property
    def source(self) -> DebridementDebrisSource:
        return self.sources.source_for_debris(self.debris_prim_path)

    @property
    def measurement(self) -> SceneMeasurement:
        return _measurement(self.envelope, self.source.source_id)

    @property
    def values(self) -> tuple[float, ...]:
        return self.measurement.value

    @property
    def normal_force_n(self) -> float:
        return self.values[7]

    @property
    def tangential_force_n(self) -> tuple[float, float, float]:
        return tuple(self.values[8:11])

    @property
    def relative_tangential_velocity_m_s(
        self,
    ) -> tuple[float, float, float]:
        return tuple(self.values[11:14])

    @property
    def slip_speed_m_s(self) -> float:
        return math.sqrt(
            sum(
                value * value
                for value in self.relative_tangential_velocity_m_s
            )
        )

    @property
    def dissipation_power_w(self) -> float:
        return max(
            0.0,
            -sum(
                force * velocity
                for force, velocity in zip(
                    self.tangential_force_n,
                    self.relative_tangential_velocity_m_s,
                    strict=True,
                )
            ),
        )

    @property
    def attachment_present(self) -> bool:
        return self.measurement.attachment_prim_ids == (
            self.source.attachment_prim_path,
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
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256


@dataclass(frozen=True)
class WoundPreparationSceneEvidence:
    envelope: SceneEvidenceEnvelope
    sources: WoundPreparationSceneSources

    def __post_init__(self) -> None:
        self.sources.validate_envelope(self.envelope)
        for source in self.sources.debris_sources:
            DebridementMechanicsSample(
                self.envelope,
                self.sources,
                source.debris_prim_path,
            )

    @property
    def samples(self) -> tuple[DebridementMechanicsSample, ...]:
        return tuple(
            DebridementMechanicsSample(
                self.envelope,
                self.sources,
                source.debris_prim_path,
            )
            for source in self.sources.debris_sources
        )

    def sample_for(
        self,
        debris_prim_path: str,
    ) -> DebridementMechanicsSample:
        return DebridementMechanicsSample(
            self.envelope,
            self.sources,
            debris_prim_path,
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
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256


@runtime_checkable
class WoundPreparationSceneEvidenceSource(Protocol):
    def sample_wound_preparation_scene(self) -> WoundPreparationSceneEvidence:
        ...


class WoundPreparationSceneEvidenceAdapter:
    """Build one immutable interval from registered shared contacts."""

    ADAPTER_ID = "dranmar.wound-preparation.scene-evidence"
    ADAPTER_VERSION = "1.0.0"

    def __init__(self, sources: WoundPreparationSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(sources.registered_sources)

    def collect_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        observations_by_debris: Mapping[
            str,
            DebridementContactObservation,
        ],
        raw_sample_ids_by_source: Mapping[str, tuple[str, ...]],
    ) -> WoundPreparationSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        observations = {
            _scene_path(path, "debris observation key"): value
            for path, value in observations_by_debris.items()
        }
        expected_debris = {
            source.debris_prim_path for source in self.sources.debris_sources
        }
        if set(observations) != expected_debris:
            raise ValueError(
                "debridement observations must exactly match registered debris"
            )
        if frozenset(raw_sample_ids_by_source) != self.sources.expected_source_ids:
            raise ValueError(
                "raw sample identity must exactly match registered debris sources"
            )
        measurements: list[SceneMeasurement] = []
        for source in self.sources.debris_sources:
            observation = observations[source.debris_prim_path]
            if not isinstance(observation, DebridementContactObservation):
                raise TypeError(
                    "observations_by_debris must contain "
                    "DebridementContactObservation values"
                )
            if observation.debris_prim_path != source.debris_prim_path:
                raise ValueError("debris observation key and body identity disagree")
            if {
                observation.contact.key.body_a,
                observation.contact.key.body_b,
            } != {
                self.sources.debridement_contact_prim_path,
                source.debris_prim_path,
            }:
                raise ValueError(
                    f"contact for {source.debris_prim_path!r} does not bind "
                    "the registered debridement tool and debris body"
                )
            if observation.active_attachment_prim_ids not in (
                (),
                (source.attachment_prim_path,),
            ):
                raise ValueError(
                    f"observation for {source.debris_prim_path!r} names an "
                    "unregistered attachment"
                )
            measurements.append(
                SceneMeasurement(
                    source_id=source.source_id,
                    value=_contact_values(observation.contact),
                    source_prim_path=self.sources.debridement_contact_prim_path,
                    target_prim_path=source.debris_prim_path,
                    contact_pair_id=(
                        f"{self.sources.debridement_contact_prim_path}|"
                        f"{source.debris_prim_path}"
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
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=tuple(measurements),
        )
        self.registry.validate(envelope)
        return WoundPreparationSceneEvidence(envelope, self.sources)


@dataclass
class WoundPreparationEvidenceCursor:
    """Reject replay, cross-episode reuse, and inconsistent interval clocks."""

    episode_id: str | None = field(default=None, init=False)
    environment_id: str | None = field(default=None, init=False)
    topology_revision: str | None = field(default=None, init=False)
    last_physics_step: int = field(default=-1, init=False)
    last_simulation_time_s: float = field(default=-1.0, init=False)
    consumed_digests: set[str] = field(default_factory=set, init=False)

    def consume(
        self,
        evidence: WoundPreparationSceneEvidence,
    ) -> WoundPreparationSceneEvidence:
        if not isinstance(evidence, WoundPreparationSceneEvidence):
            raise TypeError(
                "wound preparation requires WoundPreparationSceneEvidence"
            )
        provenance = evidence.envelope.provenance
        identity = (
            provenance.episode_id,
            provenance.environment_id,
            provenance.topology_revision,
        )
        locked = (
            self.episode_id,
            self.environment_id,
            self.topology_revision,
        )
        if self.episode_id is None:
            (
                self.episode_id,
                self.environment_id,
                self.topology_revision,
            ) = identity
        elif identity != locked:
            raise ValueError(
                "wound-preparation evidence identity changed without reset"
            )
        if provenance.physics_step <= self.last_physics_step:
            raise ValueError(
                "wound-preparation physics step must increase monotonically"
            )
        if self.last_physics_step >= 0:
            if provenance.simulation_time_s <= self.last_simulation_time_s:
                raise ValueError(
                    "wound-preparation simulation time must increase"
                )
            if not math.isclose(
                provenance.simulation_time_s - self.last_simulation_time_s,
                provenance.dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "wound-preparation dt does not match the consumed clock "
                    "interval"
                )
        digest = evidence.envelope.digest_sha256
        if digest in self.consumed_digests:
            raise ValueError("wound-preparation evidence replay is forbidden")
        self.consumed_digests.add(digest)
        self.last_physics_step = provenance.physics_step
        self.last_simulation_time_s = provenance.simulation_time_s
        return evidence

    def reset(self) -> None:
        self.episode_id = None
        self.environment_id = None
        self.topology_revision = None
        self.last_physics_step = -1
        self.last_simulation_time_s = -1.0
        self.consumed_digests.clear()


__all__ = [
    "DebridementContactObservation",
    "DebridementDebrisSource",
    "DebridementMechanicsSample",
    "WoundPreparationEvidenceCursor",
    "WoundPreparationSceneEvidence",
    "WoundPreparationSceneEvidenceAdapter",
    "WoundPreparationSceneEvidenceSource",
    "WoundPreparationSceneSources",
]
