# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound bilateral contact evidence for the dynamic patient."""

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


def _text(value: str, name: str) -> str:
    rendered = str(value).strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered


def _path(value: str, name: str) -> str:
    rendered = _text(value, name)
    if not rendered.startswith("/"):
        raise ValueError(f"{name} must be an absolute scene path")
    return rendered


def _vector3(value, name: str) -> tuple[float, float, float]:
    rendered = tuple(float(component) for component in value)
    if len(rendered) != 3 or not all(
        math.isfinite(component) for component in rendered
    ):
        raise ValueError(f"{name} must be a finite 3-vector")
    return rendered


def _subtract(left, right) -> tuple[float, float, float]:
    return tuple(
        a - b for a, b in zip(left, right, strict=True)
    )


def _norm(value) -> float:
    return math.sqrt(sum(component * component for component in value))


def _dot(left, right) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class PatientContactSceneSources:
    """Scene registration fixes target and interaction outside policy input."""

    target_id: str
    interaction: str
    source_robot: str
    left_contact_source_id: str
    right_contact_source_id: str
    left_jaw_prim_path: str
    right_jaw_prim_path: str
    tool_tip_prim_path: str
    target_prim_path: str
    calibration_profile_id: str = "dranmar-patient-contact-engineering-v1"

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "interaction",
            "source_robot",
            "left_contact_source_id",
            "right_contact_source_id",
            "calibration_profile_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name),
            )
        for name in (
            "left_jaw_prim_path",
            "right_jaw_prim_path",
            "tool_tip_prim_path",
            "target_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _path(getattr(self, name), name),
            )

    @property
    def left_position_source_id(self) -> str:
        return f"{self.source_robot}.left_jaw_position"

    @property
    def right_position_source_id(self) -> str:
        return f"{self.source_robot}.right_jaw_position"

    @property
    def tool_position_source_id(self) -> str:
        return f"{self.source_robot}.tool_tip_position"


@dataclass(frozen=True)
class PatientContactSensorSample:
    """Raw post-physics values read from the configured scene sources."""

    physics_step: int
    simulation_time_s: float
    left_force_w_n: tuple[float, float, float]
    right_force_w_n: tuple[float, float, float]
    left_jaw_position_w_m: tuple[float, float, float]
    right_jaw_position_w_m: tuple[float, float, float]
    tool_tip_position_w_m: tuple[float, float, float]
    attachment_prim_ids: tuple[str, ...] = ()
    left_raw_sample_ids: tuple[str, ...] = ()
    right_raw_sample_ids: tuple[str, ...] = ()
    left_position_raw_sample_ids: tuple[str, ...] = ()
    right_position_raw_sample_ids: tuple[str, ...] = ()
    tool_position_raw_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        physics_step = int(self.physics_step)
        if physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        object.__setattr__(self, "physics_step", physics_step)
        simulation_time = float(self.simulation_time_s)
        if not math.isfinite(simulation_time) or simulation_time < 0.0:
            raise ValueError(
                "simulation_time_s must be finite and nonnegative"
            )
        object.__setattr__(
            self,
            "simulation_time_s",
            simulation_time,
        )
        for name in (
            "left_force_w_n",
            "right_force_w_n",
            "left_jaw_position_w_m",
            "right_jaw_position_w_m",
            "tool_tip_position_w_m",
        ):
            object.__setattr__(
                self,
                name,
                _vector3(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "attachment_prim_ids",
            tuple(
                _path(path, "attachment_prim_id")
                for path in self.attachment_prim_ids
            ),
        )
        for name in (
            "left_raw_sample_ids",
            "right_raw_sample_ids",
            "left_position_raw_sample_ids",
            "right_position_raw_sample_ids",
            "tool_position_raw_sample_ids",
        ):
            values = tuple(
                sorted(
                    _text(sample_id, f"{name} item")
                    for sample_id in getattr(self, name)
                )
            )
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(
                self,
                name,
                values,
            )


@dataclass(frozen=True)
class PatientContactSceneEvidence:
    """Contact interval derived exclusively from a scene envelope."""

    envelope: SceneEvidenceEnvelope
    sources: PatientContactSceneSources

    def __post_init__(self) -> None:
        provenance = self.envelope.provenance
        if provenance.adapter_id != DynamicPatientSceneEvidence.ADAPTER_ID:
            raise ValueError("patient contact evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != DynamicPatientSceneEvidence.ADAPTER_VERSION
        ):
            raise ValueError(
                "patient contact evidence has the wrong adapter_version"
            )
        expected = {
            self.sources.left_contact_source_id,
            self.sources.right_contact_source_id,
            self.sources.left_position_source_id,
            self.sources.right_position_source_id,
            self.sources.tool_position_source_id,
        }
        actual = {
            measurement.source_id
            for measurement in self.envelope.measurements
        }
        if actual != expected:
            raise ValueError(
                "patient contact evidence source set does not match its "
                "registered tool"
            )
        registry = SceneEvidenceRegistry(
            (
                SceneEvidenceSource(
                    source_id=self.sources.left_contact_source_id,
                    prim_path=self.sources.left_jaw_prim_path,
                    quantity="force_on_jaw_from_registered_target",
                    unit="N",
                    coordinate_frame="world",
                    calibration_profile_id=(
                        self.sources.calibration_profile_id
                    ),
                ),
                SceneEvidenceSource(
                    source_id=self.sources.right_contact_source_id,
                    prim_path=self.sources.right_jaw_prim_path,
                    quantity="force_on_jaw_from_registered_target",
                    unit="N",
                    coordinate_frame="world",
                    calibration_profile_id=(
                        self.sources.calibration_profile_id
                    ),
                ),
                SceneEvidenceSource(
                    source_id=self.sources.left_position_source_id,
                    prim_path=self.sources.left_jaw_prim_path,
                    quantity="position",
                    unit="m",
                    coordinate_frame="world",
                    calibration_profile_id=(
                        self.sources.calibration_profile_id
                    ),
                ),
                SceneEvidenceSource(
                    source_id=self.sources.right_position_source_id,
                    prim_path=self.sources.right_jaw_prim_path,
                    quantity="position",
                    unit="m",
                    coordinate_frame="world",
                    calibration_profile_id=(
                        self.sources.calibration_profile_id
                    ),
                ),
                SceneEvidenceSource(
                    source_id=self.sources.tool_position_source_id,
                    prim_path=self.sources.tool_tip_prim_path,
                    quantity="position",
                    unit="m",
                    coordinate_frame="world",
                    calibration_profile_id=(
                        self.sources.calibration_profile_id
                    ),
                ),
            )
        )
        # Revalidate here as well as in the collector.  A caller must not be
        # able to bypass prim registration by directly instantiating the typed
        # evidence object around a self-consistent but unregistered envelope.
        registry.validate(self.envelope)
        if any(
            measurement.target_prim_path
            != self.sources.target_prim_path
            for measurement in self.envelope.measurements
        ):
            raise ValueError(
                "patient contact evidence does not target the registered "
                "patient prim"
            )
        expected_contact_pairs = {
            self.sources.left_contact_source_id: (
                f"{self.sources.left_jaw_prim_path}|"
                f"{self.sources.target_prim_path}"
            ),
            self.sources.right_contact_source_id: (
                f"{self.sources.right_jaw_prim_path}|"
                f"{self.sources.target_prim_path}"
            ),
        }
        for source_id, contact_pair_id in expected_contact_pairs.items():
            if (
                self._measurement(source_id).contact_pair_id
                != contact_pair_id
            ):
                raise ValueError(
                    f"patient contact source {source_id!r} does not name "
                    "the exact registered contact pair"
                )
        for source_id in (
            self.sources.left_position_source_id,
            self.sources.right_position_source_id,
            self.sources.tool_position_source_id,
        ):
            if self._measurement(source_id).contact_pair_id is not None:
                raise ValueError(
                    f"position source {source_id!r} must not claim a "
                    "contact-pair identity"
                )
        if any(
            len(self._measurement(source_id).value) != 3
            for source_id in expected
        ):
            raise ValueError(
                "patient contact evidence measurements must be 3-vectors"
            )
        for source_id in expected:
            if not self._measurement(source_id).raw_sample_ids:
                raise ValueError(
                    f"patient evidence source {source_id!r} requires exact "
                    "raw record identity"
                )
        attachment_sets = {
            self._measurement(source_id).attachment_prim_ids
            for source_id in (
                self.sources.left_contact_source_id,
                self.sources.right_contact_source_id,
                self.sources.left_position_source_id,
                self.sources.right_position_source_id,
            )
        }
        if len(attachment_sets) != 1:
            raise ValueError(
                "patient contact sources disagree on attachment identity"
            )
        if (
            self._measurement(
                self.sources.tool_position_source_id
            ).attachment_prim_ids
        ):
            raise ValueError(
                "tool-position evidence must not invent attachment identity"
            )

    def _measurement(self, source_id: str) -> SceneMeasurement:
        try:
            return next(
                value
                for value in self.envelope.measurements
                if value.source_id == source_id
            )
        except StopIteration as error:
            raise ValueError(
                f"missing patient contact source {source_id!r}"
            ) from error

    @property
    def target(self) -> str:
        return self.sources.target_id

    @property
    def source_robot(self) -> str:
        return self.sources.source_robot

    @property
    def interaction(self) -> str:
        return self.sources.interaction

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

    @property
    def left_jaw_position_w_m(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.left_position_source_id
            ).value
        )

    @property
    def right_jaw_position_w_m(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.right_position_source_id
            ).value
        )

    @property
    def _closing_axis_w(self) -> tuple[float, float, float]:
        delta = _subtract(
            self.right_jaw_position_w_m,
            self.left_jaw_position_w_m,
        )
        magnitude = _norm(delta)
        if magnitude <= 1.0e-9:
            raise ValueError("patient contact jaw separation must be positive")
        return tuple(component / magnitude for component in delta)

    @property
    def normal_forces_n(self) -> tuple[float, float]:
        left = self._measurement(
            self.sources.left_contact_source_id
        ).value
        right = self._measurement(
            self.sources.right_contact_source_id
        ).value
        # Sensor vectors are forces on each jaw from the registered target.
        # A compressive reaction points outward: opposite the left-to-right
        # axis on the left jaw and along it on the right jaw.  Absolute value
        # would misclassify separating or mis-oriented loads as compression.
        return (
            max(0.0, -_dot(left, self._closing_axis_w)),
            max(0.0, _dot(right, self._closing_axis_w)),
        )

    @property
    def tool_position_m(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.tool_position_source_id
            ).value
        )

    @property
    def attachment_prim_ids(self) -> tuple[str, ...]:
        return self._measurement(
            self.sources.left_contact_source_id
        ).attachment_prim_ids


class DynamicPatientSceneEvidence:
    """Collect validated bilateral patient contact from exact scene prims."""

    ADAPTER_ID = "dranmar.dynamic-patient.scene-evidence"
    ADAPTER_VERSION = "3.0.0"

    def __init__(self, sources: PatientContactSceneSources) -> None:
        self.sources = sources
        self.registry = SceneEvidenceRegistry(
            (
                SceneEvidenceSource(
                    source_id=sources.left_contact_source_id,
                    prim_path=sources.left_jaw_prim_path,
                    quantity="force_on_jaw_from_registered_target",
                    unit="N",
                    coordinate_frame="world",
                    calibration_profile_id=sources.calibration_profile_id,
                ),
                SceneEvidenceSource(
                    source_id=sources.right_contact_source_id,
                    prim_path=sources.right_jaw_prim_path,
                    quantity="force_on_jaw_from_registered_target",
                    unit="N",
                    coordinate_frame="world",
                    calibration_profile_id=sources.calibration_profile_id,
                ),
                SceneEvidenceSource(
                    source_id=sources.left_position_source_id,
                    prim_path=sources.left_jaw_prim_path,
                    quantity="position",
                    unit="m",
                    coordinate_frame="world",
                    calibration_profile_id=sources.calibration_profile_id,
                ),
                SceneEvidenceSource(
                    source_id=sources.right_position_source_id,
                    prim_path=sources.right_jaw_prim_path,
                    quantity="position",
                    unit="m",
                    coordinate_frame="world",
                    calibration_profile_id=sources.calibration_profile_id,
                ),
                SceneEvidenceSource(
                    source_id=sources.tool_position_source_id,
                    prim_path=sources.tool_tip_prim_path,
                    quantity="position",
                    unit="m",
                    coordinate_frame="world",
                    calibration_profile_id=sources.calibration_profile_id,
                ),
            )
        )

    def collect_contact_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        sample: PatientContactSensorSample,
    ) -> PatientContactSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        if sample.physics_step != provenance.physics_step:
            raise ValueError(
                "patient contact sample step does not match provenance"
            )
        if not math.isclose(
            sample.simulation_time_s,
            provenance.simulation_time_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "patient contact sample time does not match provenance"
            )
        attachments = sample.attachment_prim_ids
        measurements = (
            SceneMeasurement(
                source_id=self.sources.left_contact_source_id,
                value=sample.left_force_w_n,
                source_prim_path=self.sources.left_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                contact_pair_id=(
                    f"{self.sources.left_jaw_prim_path}|"
                    f"{self.sources.target_prim_path}"
                ),
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.left_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.right_contact_source_id,
                value=sample.right_force_w_n,
                source_prim_path=self.sources.right_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                contact_pair_id=(
                    f"{self.sources.right_jaw_prim_path}|"
                    f"{self.sources.target_prim_path}"
                ),
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.right_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.left_position_source_id,
                value=sample.left_jaw_position_w_m,
                source_prim_path=self.sources.left_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.left_position_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.right_position_source_id,
                value=sample.right_jaw_position_w_m,
                source_prim_path=self.sources.right_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                attachment_prim_ids=attachments,
                raw_sample_ids=sample.right_position_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.tool_position_source_id,
                value=sample.tool_tip_position_w_m,
                source_prim_path=self.sources.tool_tip_prim_path,
                target_prim_path=self.sources.target_prim_path,
                raw_sample_ids=sample.tool_position_raw_sample_ids,
            ),
        )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=measurements,
        )
        self.registry.validate(envelope)
        return PatientContactSceneEvidence(
            envelope=envelope,
            sources=self.sources,
        )


__all__ = [
    "DynamicPatientSceneEvidence",
    "PatientContactSceneEvidence",
    "PatientContactSceneSources",
    "PatientContactSensorSample",
]
