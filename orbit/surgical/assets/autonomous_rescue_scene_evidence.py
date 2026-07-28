# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound post-physics evidence for the Autonomous Rescue OR.

The public rescue runtime consumes :class:`RescueVesselSceneEvidence`, not
caller-authored force, closure, perfusion, or success scalars.  Every derived
quantity is recomputed from an immutable :class:`SceneEvidenceEnvelope` whose
measurements name the exact scene prims registered by the environment.

This is an engineering evidence boundary.  It does not make the provisional
contact or vessel parameters clinically validated.
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


def _vector3(
    value: tuple[float, float, float],
    name: str,
) -> tuple[float, float, float]:
    rendered = tuple(float(component) for component in value)
    if len(rendered) != 3 or not all(
        math.isfinite(component) for component in rendered
    ):
        raise ValueError(f"{name} must be a finite 3-vector")
    return rendered


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class RescueToolSceneSources:
    """Exact sources registered for one rescue tool and vessel target."""

    station_id: str
    tool_id: str
    left_contact_source_id: str
    right_contact_source_id: str
    left_jaw_prim_path: str
    right_jaw_prim_path: str
    tool_tip_prim_path: str
    target_prim_path: str
    target_frame_prim_path: str
    calibration_profile_id: str = "dranmar-rescue-contact-engineering-v1"
    pressure_source_id: str | None = None
    pressure_prim_path: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "station_id",
            "tool_id",
            "left_contact_source_id",
            "right_contact_source_id",
            "calibration_profile_id",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name),
            )
        for name in (
            "left_jaw_prim_path",
            "right_jaw_prim_path",
            "tool_tip_prim_path",
            "target_prim_path",
            "target_frame_prim_path",
        ):
            object.__setattr__(
                self,
                name,
                _scene_path(getattr(self, name), name),
            )
        if (self.pressure_source_id is None) != (
            self.pressure_prim_path is None
        ):
            raise ValueError(
                "pressure_source_id and pressure_prim_path must be "
                "configured together"
            )
        if self.pressure_source_id is not None:
            object.__setattr__(
                self,
                "pressure_source_id",
                _required_text(
                    self.pressure_source_id,
                    "pressure_source_id",
                ),
            )
            object.__setattr__(
                self,
                "pressure_prim_path",
                _scene_path(
                    self.pressure_prim_path,
                    "pressure_prim_path",
                ),
            )

    @property
    def left_jaw_position_source_id(self) -> str:
        return f"{self.tool_id}.left_jaw_position"

    @property
    def right_jaw_position_source_id(self) -> str:
        return f"{self.tool_id}.right_jaw_position"

    @property
    def tool_motion_source_id(self) -> str:
        return f"{self.tool_id}.tool_tip_motion"

    @property
    def target_frame_source_id(self) -> str:
        return f"{self.tool_id}.target_frame_position"


@dataclass(frozen=True)
class RescueContactSensorSample:
    """Raw values read by the environment after one physics interval."""

    physics_step: int
    simulation_time_s: float
    left_force_w_n: tuple[float, float, float]
    right_force_w_n: tuple[float, float, float]
    left_jaw_position_w_m: tuple[float, float, float]
    right_jaw_position_w_m: tuple[float, float, float]
    tool_tip_position_w_m: tuple[float, float, float]
    previous_tool_tip_position_w_m: tuple[float, float, float] | None
    target_frame_position_w_m: tuple[float, float, float]
    attachment_prim_ids: tuple[str, ...] = ()
    measured_upstream_pressure_mmhg: float | None = None
    left_raw_sample_ids: tuple[str, ...] = ()
    right_raw_sample_ids: tuple[str, ...] = ()
    left_position_raw_sample_ids: tuple[str, ...] = ()
    right_position_raw_sample_ids: tuple[str, ...] = ()
    tool_motion_raw_sample_ids: tuple[str, ...] = ()
    target_frame_raw_sample_ids: tuple[str, ...] = ()
    pressure_raw_sample_ids: tuple[str, ...] = ()

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
            "target_frame_position_w_m",
        ):
            object.__setattr__(
                self,
                name,
                _vector3(getattr(self, name), name),
            )
        if self.previous_tool_tip_position_w_m is not None:
            object.__setattr__(
                self,
                "previous_tool_tip_position_w_m",
                _vector3(
                    self.previous_tool_tip_position_w_m,
                    "previous_tool_tip_position_w_m",
                ),
            )
        object.__setattr__(
            self,
            "attachment_prim_ids",
            tuple(
                _scene_path(path, "attachment_prim_id")
                for path in self.attachment_prim_ids
            ),
        )
        for name in (
            "left_raw_sample_ids",
            "right_raw_sample_ids",
            "left_position_raw_sample_ids",
            "right_position_raw_sample_ids",
            "tool_motion_raw_sample_ids",
            "target_frame_raw_sample_ids",
            "pressure_raw_sample_ids",
        ):
            values = tuple(
                sorted(
                    _required_text(sample_id, f"{name} item")
                    for sample_id in getattr(self, name)
                )
            )
            if (
                name != "pressure_raw_sample_ids"
                and not values
            ):
                raise ValueError(f"{name} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(
                self,
                name,
                values,
            )
        if self.measured_upstream_pressure_mmhg is not None:
            pressure = float(self.measured_upstream_pressure_mmhg)
            if not math.isfinite(pressure) or pressure < 0.0:
                raise ValueError(
                    "measured_upstream_pressure_mmhg must be finite and "
                    "nonnegative"
                )
            object.__setattr__(
                self,
                "measured_upstream_pressure_mmhg",
                pressure,
            )


@dataclass(frozen=True)
class RescueVesselSceneEvidence:
    """Vessel evidence whose properties derive only from one envelope."""

    envelope: SceneEvidenceEnvelope
    sources: RescueToolSceneSources

    def __post_init__(self) -> None:
        provenance = self.envelope.provenance
        if provenance.adapter_id != AutonomousRescueSceneEvidence.ADAPTER_ID:
            raise ValueError("rescue evidence has the wrong adapter_id")
        if (
            provenance.adapter_version
            != AutonomousRescueSceneEvidence.ADAPTER_VERSION
        ):
            raise ValueError("rescue evidence has the wrong adapter_version")
        measurements = {
            measurement.source_id: measurement
            for measurement in self.envelope.measurements
        }
        expected = {
            self.sources.left_contact_source_id,
            self.sources.right_contact_source_id,
            self.sources.left_jaw_position_source_id,
            self.sources.right_jaw_position_source_id,
            self.sources.tool_motion_source_id,
            self.sources.target_frame_source_id,
        }
        if self.sources.pressure_source_id is not None:
            expected.add(self.sources.pressure_source_id)
        if set(measurements) != expected:
            raise ValueError(
                "rescue evidence source set does not match the registered "
                "tool sources"
            )
        # Repeat prim registration validation at the typed-object boundary.
        # This prevents a caller from bypassing the collector by wrapping an
        # arbitrary envelope directly in RescueVesselSceneEvidence.
        AutonomousRescueSceneEvidence.registry_for(
            self.sources
        ).validate(self.envelope)
        if any(
            measurement.target_prim_path
            != self.sources.target_prim_path
            for measurement in self.envelope.measurements
        ):
            raise ValueError(
                "rescue evidence does not target the registered vessel prim"
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
                    f"rescue source {source_id!r} does not name the exact "
                    "registered contact pair"
                )
        for source_id in expected.difference(expected_contact_pairs):
            if self._measurement(source_id).contact_pair_id is not None:
                raise ValueError(
                    f"non-contact rescue source {source_id!r} must not "
                    "claim contact-pair identity"
                )
        if len(self._measurement(self.sources.left_contact_source_id).value) != 3:
            raise ValueError("left contact measurement must be a 3-vector")
        if len(self._measurement(self.sources.right_contact_source_id).value) != 3:
            raise ValueError("right contact measurement must be a 3-vector")
        for source_id in expected:
            if not self._measurement(source_id).raw_sample_ids:
                raise ValueError(
                    f"rescue source {source_id!r} requires exact raw record "
                    "identity"
                )
        if (
            len(
                self._measurement(
                    self.sources.left_jaw_position_source_id
                ).value
            )
            != 3
            or len(
                self._measurement(
                    self.sources.right_jaw_position_source_id
                ).value
            )
            != 3
        ):
            raise ValueError("jaw position measurements must be 3-vectors")
        if len(self._measurement(self.sources.tool_motion_source_id).value) != 6:
            raise ValueError("tool motion measurement must contain two 3-vectors")
        if len(self._measurement(self.sources.target_frame_source_id).value) != 3:
            raise ValueError("target frame measurement must be a 3-vector")
        if (
            self.sources.pressure_source_id is not None
            and len(
                self._measurement(
                    self.sources.pressure_source_id
                ).value
            )
            != 1
        ):
            raise ValueError("pressure measurement must be scalar")
        attachment_sets = {
            self._measurement(source_id).attachment_prim_ids
            for source_id in (
                self.sources.left_contact_source_id,
                self.sources.right_contact_source_id,
                self.sources.left_jaw_position_source_id,
                self.sources.right_jaw_position_source_id,
            )
        }
        if len(attachment_sets) != 1:
            raise ValueError(
                "contact and jaw measurements disagree on live attachment "
                "prim identity"
            )
        for source_id in (
            self.sources.tool_motion_source_id,
            self.sources.target_frame_source_id,
            self.sources.pressure_source_id,
        ):
            if (
                source_id is not None
                and self._measurement(source_id).attachment_prim_ids
            ):
                raise ValueError(
                    f"non-attachment source {source_id!r} must not invent "
                    "attachment identity"
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
                f"rescue evidence is missing source {source_id!r}"
            ) from error

    @property
    def station_id(self) -> str:
        return self.sources.station_id

    @property
    def tool_id(self) -> str:
        return self.sources.tool_id

    @property
    def target_id(self) -> str:
        return "rescue_vessel"

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
    def left_force_w_n(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.left_contact_source_id
            ).value
        )

    @property
    def right_force_w_n(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.right_contact_source_id
            ).value
        )

    @property
    def left_jaw_position_w_m(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.left_jaw_position_source_id
            ).value
        )

    @property
    def right_jaw_position_w_m(self) -> tuple[float, float, float]:
        return tuple(
            self._measurement(
                self.sources.right_jaw_position_source_id
            ).value
        )

    @property
    def separation_m(self) -> float:
        return _norm(
            _subtract(
                self.right_jaw_position_w_m,
                self.left_jaw_position_w_m,
            )
        )

    @property
    def _closing_axis_w(self) -> tuple[float, float, float]:
        delta = _subtract(
            self.right_jaw_position_w_m,
            self.left_jaw_position_w_m,
        )
        separation = _norm(delta)
        if separation <= 1.0e-9:
            raise ValueError("jaw separation must be positive")
        return tuple(component / separation for component in delta)

    @property
    def left_normal_force_n(self) -> float:
        return max(
            0.0,
            -_dot(self.left_force_w_n, self._closing_axis_w),
        )

    @property
    def right_normal_force_n(self) -> float:
        return max(
            0.0,
            _dot(self.right_force_w_n, self._closing_axis_w),
        )

    @property
    def tool_speed_m_s(self) -> float:
        values = self._measurement(
            self.sources.tool_motion_source_id
        ).value
        current = tuple(values[:3])
        previous = tuple(values[3:])
        return _norm(_subtract(current, previous)) / self.dt_s

    @property
    def target_distance_m(self) -> float:
        center = tuple(
            0.5 * (left + right)
            for left, right in zip(
                self.left_jaw_position_w_m,
                self.right_jaw_position_w_m,
                strict=True,
            )
        )
        target = tuple(
            self._measurement(
                self.sources.target_frame_source_id
            ).value
        )
        return _norm(_subtract(center, target))

    @property
    def attachment_prim_ids(self) -> tuple[str, ...]:
        return self._measurement(
            self.sources.left_contact_source_id
        ).attachment_prim_ids

    @property
    def measured_upstream_pressure_mmhg(self) -> float | None:
        if self.sources.pressure_source_id is None:
            return None
        return self._measurement(
            self.sources.pressure_source_id
        ).value[0]


class AutonomousRescueSceneEvidence:
    """Collect one coherent vessel interval from registered scene sources."""

    ADAPTER_ID = "dranmar.autonomous-rescue.scene-evidence"
    ADAPTER_VERSION = "4.0.0"

    @staticmethod
    def registry_for(
        sources: RescueToolSceneSources,
    ) -> SceneEvidenceRegistry:
        registered = [
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
                source_id=sources.left_jaw_position_source_id,
                prim_path=sources.left_jaw_prim_path,
                quantity="position",
                unit="m",
                coordinate_frame="world",
                calibration_profile_id=sources.calibration_profile_id,
            ),
            SceneEvidenceSource(
                source_id=sources.right_jaw_position_source_id,
                prim_path=sources.right_jaw_prim_path,
                quantity="position",
                unit="m",
                coordinate_frame="world",
                calibration_profile_id=sources.calibration_profile_id,
            ),
            SceneEvidenceSource(
                source_id=sources.tool_motion_source_id,
                prim_path=sources.tool_tip_prim_path,
                quantity="current_and_previous_tool_tip_position",
                unit="m",
                coordinate_frame="world",
                calibration_profile_id=sources.calibration_profile_id,
            ),
            SceneEvidenceSource(
                source_id=sources.target_frame_source_id,
                prim_path=sources.target_frame_prim_path,
                quantity="target_frame_position",
                unit="m",
                coordinate_frame="world",
                calibration_profile_id=sources.calibration_profile_id,
            ),
        ]
        if sources.pressure_source_id is not None:
            registered.append(
                SceneEvidenceSource(
                    source_id=sources.pressure_source_id,
                    prim_path=sources.pressure_prim_path,
                    quantity="upstream_pressure",
                    unit="mmHg",
                    coordinate_frame="vessel_lumen",
                    calibration_profile_id=sources.calibration_profile_id,
                )
            )
        return SceneEvidenceRegistry(registered)

    def __init__(self, sources: RescueToolSceneSources) -> None:
        self.sources = sources
        self.registry = self.registry_for(sources)

    def collect_vessel_interval(
        self,
        *,
        provenance: EvidenceProvenance,
        sample: RescueContactSensorSample,
    ) -> RescueVesselSceneEvidence:
        if provenance.adapter_id != self.ADAPTER_ID:
            raise ValueError(f"adapter_id must be {self.ADAPTER_ID!r}")
        if provenance.adapter_version != self.ADAPTER_VERSION:
            raise ValueError(
                f"adapter_version must be {self.ADAPTER_VERSION!r}"
            )
        if sample.physics_step != provenance.physics_step:
            raise ValueError(
                "rescue sample step does not match provenance"
            )
        if not math.isclose(
            sample.simulation_time_s,
            provenance.simulation_time_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "rescue sample time does not match provenance"
            )
        if (
            sample.measured_upstream_pressure_mmhg is None
        ) != (self.sources.pressure_source_id is None):
            raise ValueError(
                "pressure evidence must be present exactly when a pressure "
                "source is registered"
            )
        previous_tip = (
            sample.previous_tool_tip_position_w_m
            if sample.previous_tool_tip_position_w_m is not None
            else sample.tool_tip_position_w_m
        )
        contact_pair_left = (
            f"{self.sources.left_jaw_prim_path}|"
            f"{self.sources.target_prim_path}"
        )
        contact_pair_right = (
            f"{self.sources.right_jaw_prim_path}|"
            f"{self.sources.target_prim_path}"
        )
        measurements = [
            SceneMeasurement(
                source_id=self.sources.left_contact_source_id,
                value=sample.left_force_w_n,
                source_prim_path=self.sources.left_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                contact_pair_id=contact_pair_left,
                attachment_prim_ids=sample.attachment_prim_ids,
                raw_sample_ids=sample.left_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.right_contact_source_id,
                value=sample.right_force_w_n,
                source_prim_path=self.sources.right_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                contact_pair_id=contact_pair_right,
                attachment_prim_ids=sample.attachment_prim_ids,
                raw_sample_ids=sample.right_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.left_jaw_position_source_id,
                value=sample.left_jaw_position_w_m,
                source_prim_path=self.sources.left_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                attachment_prim_ids=sample.attachment_prim_ids,
                raw_sample_ids=sample.left_position_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.right_jaw_position_source_id,
                value=sample.right_jaw_position_w_m,
                source_prim_path=self.sources.right_jaw_prim_path,
                target_prim_path=self.sources.target_prim_path,
                attachment_prim_ids=sample.attachment_prim_ids,
                raw_sample_ids=sample.right_position_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.tool_motion_source_id,
                value=(
                    *sample.tool_tip_position_w_m,
                    *previous_tip,
                ),
                source_prim_path=self.sources.tool_tip_prim_path,
                target_prim_path=self.sources.target_prim_path,
                raw_sample_ids=sample.tool_motion_raw_sample_ids,
            ),
            SceneMeasurement(
                source_id=self.sources.target_frame_source_id,
                value=sample.target_frame_position_w_m,
                source_prim_path=self.sources.target_frame_prim_path,
                target_prim_path=self.sources.target_prim_path,
                raw_sample_ids=sample.target_frame_raw_sample_ids,
            ),
        ]
        if self.sources.pressure_source_id is not None:
            measurements.append(
                SceneMeasurement(
                    source_id=self.sources.pressure_source_id,
                    value=(sample.measured_upstream_pressure_mmhg,),
                    source_prim_path=self.sources.pressure_prim_path,
                    target_prim_path=self.sources.target_prim_path,
                    raw_sample_ids=sample.pressure_raw_sample_ids,
                )
            )
        envelope = SceneEvidenceEnvelope(
            provenance=provenance,
            measurements=tuple(measurements),
        )
        self.registry.validate(envelope)
        return RescueVesselSceneEvidence(
            envelope=envelope,
            sources=self.sources,
        )


__all__ = [
    "AutonomousRescueSceneEvidence",
    "RescueContactSensorSample",
    "RescueToolSceneSources",
    "RescueVesselSceneEvidence",
]
