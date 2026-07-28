# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Elastoplastic clip forming and scene-derived retention observations.

Jaw gap and force are physical boundary inputs.  Residual gap, curvature,
contact pressure, measured tangential load, retention capacity, and damage are
derived from mutable clip state and scene contact samples.  Measured load is
never synthesized from the material's Coulomb capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from .contact import ContactKey, Vec3


_EPSILON = 1.0e-12
_ZERO: Vec3 = (0.0, 0.0, 0.0)


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: float, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _non_negative(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _fraction(value: float, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _vec3(value: Iterable[float], label: str) -> Vec3:
    result = tuple(float(component) for component in value)
    if len(result) != 3 or not all(
        math.isfinite(component) for component in result
    ):
        raise ValueError(f"{label} must contain three finite values")
    return result  # type: ignore[return-value]


def _add(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _norm(value: Vec3) -> float:
    return math.sqrt(sum(component * component for component in value))


@dataclass(frozen=True)
class ClipMaterial:
    """Reduced-order forming and retention parameters in SI units."""

    open_gap_m: float
    leg_length_m: float
    wire_diameter_m: float
    elastic_closure_stiffness_n_m: float
    yield_force_n: float
    failure_force_n: float
    plastic_compliance_m_n_s: float
    springback_time_s: float
    contact_friction_coefficient: float
    geometric_interlock_fraction: float
    damage_rate_per_s: float = 2.0
    parameter_status: str = "provisional_engineering_seed"
    dynamic_friction_fraction: float = 0.65
    slip_transition_speed_m_s: float = 1.0e-3

    def __post_init__(self) -> None:
        _positive(self.open_gap_m, "open_gap_m")
        _positive(self.leg_length_m, "leg_length_m")
        _positive(self.wire_diameter_m, "wire_diameter_m")
        _positive(
            self.elastic_closure_stiffness_n_m,
            "elastic_closure_stiffness_n_m",
        )
        yield_force = _positive(self.yield_force_n, "yield_force_n")
        failure_force = _positive(self.failure_force_n, "failure_force_n")
        if failure_force <= yield_force:
            raise ValueError("failure_force_n must exceed yield_force_n")
        _non_negative(
            self.plastic_compliance_m_n_s, "plastic_compliance_m_n_s"
        )
        _positive(self.springback_time_s, "springback_time_s")
        _non_negative(
            self.contact_friction_coefficient, "contact_friction_coefficient"
        )
        _fraction(
            self.geometric_interlock_fraction, "geometric_interlock_fraction"
        )
        _non_negative(self.damage_rate_per_s, "damage_rate_per_s")
        _fraction(
            self.dynamic_friction_fraction,
            "dynamic_friction_fraction",
        )
        _positive(
            self.slip_transition_speed_m_s,
            "slip_transition_speed_m_s",
        )
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")


@dataclass
class ClipState:
    """Mutable reduced-order state for one clip."""

    current_gap_m: float
    residual_gap_m: float
    gap_velocity_m_s: float = 0.0
    plastic_curvature_1_m: float = 0.0
    damage_fraction: float = 0.0
    time_s: float = 0.0
    physics_step: int = 0

    def __post_init__(self) -> None:
        self.current_gap_m = _non_negative(self.current_gap_m, "current_gap_m")
        self.residual_gap_m = _non_negative(
            self.residual_gap_m, "residual_gap_m"
        )
        self.gap_velocity_m_s = _finite(
            self.gap_velocity_m_s, "gap_velocity_m_s"
        )
        self.plastic_curvature_1_m = _non_negative(
            self.plastic_curvature_1_m, "plastic_curvature_1_m"
        )
        self.damage_fraction = _fraction(
            self.damage_fraction, "damage_fraction"
        )
        self.time_s = _non_negative(self.time_s, "time_s")
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step < 0:
            raise ValueError("physics_step must be non-negative")
        self.physics_step = physics_step

    @classmethod
    def open(cls, material: ClipMaterial) -> "ClipState":
        return cls(
            current_gap_m=material.open_gap_m,
            residual_gap_m=material.open_gap_m,
        )


@dataclass(frozen=True)
class ClipContactSample:
    """One scene-derived clip contact patch.

    ``tangential_force_n`` is the measured world-frame force acting on the
    registered clip, and ``relative_slip_speed_m_s`` is the magnitude of the
    clip/target tangential relative velocity for this same patch and interval.
    """

    scene_component_id: str
    contact_id: str
    contact_area_m2: float
    normal_force_n: float
    relative_slip_speed_m_s: float = 0.0
    tangential_force_n: Vec3 = _ZERO
    contact_key: ContactKey | None = None

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        if not str(self.contact_id).strip():
            raise ValueError("contact_id must be non-empty")
        object.__setattr__(
            self, "contact_area_m2", _positive(self.contact_area_m2, "contact_area_m2")
        )
        object.__setattr__(
            self, "normal_force_n", _non_negative(self.normal_force_n, "normal_force_n")
        )
        object.__setattr__(
            self,
            "relative_slip_speed_m_s",
            _non_negative(
                self.relative_slip_speed_m_s, "relative_slip_speed_m_s"
            ),
        )
        object.__setattr__(
            self,
            "tangential_force_n",
            _vec3(self.tangential_force_n, "tangential_force_n"),
        )
        if self.contact_key is not None and not isinstance(
            self.contact_key,
            ContactKey,
        ):
            raise TypeError("contact_key must be a ContactKey")


@dataclass(frozen=True)
class ClipObservation:
    """Scene-addressable clip observation derived after one mechanics step."""

    scene_component_id: str
    contact_component_ids: tuple[str, ...]
    residual_gap_m: float
    formed_span_m: float
    retention_load_n: float
    contact_area_m2: float
    contact_pressure_pa: float
    interface_traction_pa: float
    plastic_curvature_1_m: float
    damage_fraction: float
    parameter_status: str
    retention_capacity_n: float
    measured_tangential_force_n: Vec3
    max_relative_slip_speed_m_s: float
    contact_keys: tuple[ContactKey, ...]
    raw_contact_ids: tuple[str, ...]
    physics_step: int
    previous_time_s: float
    time_s: float
    dt_s: float

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        component_ids = tuple(
            sorted(str(value).strip() for value in self.contact_component_ids)
        )
        if any(not value for value in component_ids):
            raise ValueError("contact_component_ids cannot contain empty values")
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("contact_component_ids must not contain duplicates")
        object.__setattr__(
            self,
            "contact_component_ids",
            component_ids,
        )
        object.__setattr__(
            self, "residual_gap_m", _non_negative(self.residual_gap_m, "residual_gap_m")
        )
        object.__setattr__(
            self, "formed_span_m", _non_negative(self.formed_span_m, "formed_span_m")
        )
        object.__setattr__(
            self,
            "retention_load_n",
            _non_negative(self.retention_load_n, "retention_load_n"),
        )
        object.__setattr__(
            self,
            "retention_capacity_n",
            _non_negative(
                self.retention_capacity_n,
                "retention_capacity_n",
            ),
        )
        object.__setattr__(
            self,
            "measured_tangential_force_n",
            _vec3(
                self.measured_tangential_force_n,
                "measured_tangential_force_n",
            ),
        )
        if not math.isclose(
            self.retention_load_n,
            _norm(self.measured_tangential_force_n),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "retention_load_n must equal the measured tangential "
                "force-vector magnitude"
            )
        object.__setattr__(
            self,
            "contact_area_m2",
            _non_negative(self.contact_area_m2, "contact_area_m2"),
        )
        object.__setattr__(
            self,
            "contact_pressure_pa",
            _non_negative(self.contact_pressure_pa, "contact_pressure_pa"),
        )
        object.__setattr__(
            self,
            "interface_traction_pa",
            _non_negative(self.interface_traction_pa, "interface_traction_pa"),
        )
        object.__setattr__(
            self,
            "plastic_curvature_1_m",
            _non_negative(
                self.plastic_curvature_1_m, "plastic_curvature_1_m"
            ),
        )
        object.__setattr__(
            self,
            "damage_fraction",
            _fraction(self.damage_fraction, "damage_fraction"),
        )
        object.__setattr__(
            self,
            "max_relative_slip_speed_m_s",
            _non_negative(
                self.max_relative_slip_speed_m_s,
                "max_relative_slip_speed_m_s",
            ),
        )
        keys = tuple(sorted(self.contact_keys))
        if len(keys) != len(set(keys)):
            raise ValueError("contact_keys must not contain duplicates")
        object.__setattr__(self, "contact_keys", keys)
        raw_ids = tuple(sorted(str(value).strip() for value in self.raw_contact_ids))
        if any(not value for value in raw_ids):
            raise ValueError("raw_contact_ids cannot contain empty values")
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError("raw_contact_ids must not contain duplicates")
        object.__setattr__(self, "raw_contact_ids", raw_ids)
        if len(keys) != len(raw_ids):
            raise ValueError(
                "each canonical contact key requires one raw contact identity"
            )
        for key in keys:
            if self.scene_component_id not in (key.body_a, key.body_b):
                raise ValueError(
                    "contact key does not include the registered clip component"
                )
            target = (
                key.body_b
                if key.body_a == self.scene_component_id
                else key.body_a
            )
            if target not in component_ids:
                raise ValueError(
                    "contact key target is not a registered contact component"
                )
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step <= 0:
            raise ValueError("physics_step must be a positive integer")
        object.__setattr__(self, "physics_step", physics_step)
        previous_time = _non_negative(
            self.previous_time_s,
            "previous_time_s",
        )
        time = _non_negative(self.time_s, "time_s")
        dt = _positive(self.dt_s, "dt_s")
        time_scale = max(previous_time, time, dt)
        time_tolerance = max(
            64.0 * math.ulp(time_scale),
            1.0e-12 * time_scale,
        )
        if time <= previous_time or not math.isclose(
            time,
            previous_time + dt,
            rel_tol=0.0,
            abs_tol=time_tolerance,
        ):
            raise ValueError(
                "clip observation time interval must advance exactly by dt_s"
            )
        object.__setattr__(self, "previous_time_s", previous_time)
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "dt_s", dt)
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")

    @property
    def measured_retention_load_n(self) -> float:
        """Actual sampled tangential load, not a constitutive capacity."""

        return self.retention_load_n

    @property
    def retention_utilization(self) -> float:
        if self.retention_capacity_n == 0.0:
            return math.inf if self.retention_load_n > 0.0 else 0.0
        return self.retention_load_n / self.retention_capacity_n


@dataclass
class ClipMechanics:
    """Causal reduced-order clip forming and retention mechanics."""

    scene_component_id: str
    material: ClipMaterial
    state: ClipState | None = None
    _last_observation: ClipObservation | None = field(
        default=None, init=False, repr=False
    )
    _time_s: float = field(init=False, repr=False)
    _physics_step: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        if self.state is None:
            self.state = ClipState.open(self.material)
        if self.state.current_gap_m > self.material.open_gap_m:
            raise ValueError("current_gap_m cannot exceed material open_gap_m")
        if self.state.residual_gap_m > self.material.open_gap_m:
            raise ValueError("residual_gap_m cannot exceed material open_gap_m")
        self._time_s = self.state.time_s
        self._physics_step = self.state.physics_step

    def advance(
        self,
        dt_s: float,
        *,
        jaw_gap_m: float,
        jaw_force_n: float,
        vessel_contacts: Iterable[ClipContactSample] = (),
    ) -> ClipObservation:
        """Advance forming and separate sampled load from retention capacity."""

        dt = _positive(dt_s, "dt_s")
        jaw_gap = _non_negative(jaw_gap_m, "jaw_gap_m")
        jaw_force = _non_negative(jaw_force_n, "jaw_force_n")
        state = self.state
        if state is None:
            raise RuntimeError("clip mechanics state was not initialized")
        if (
            state.time_s != self._time_s
            or state.physics_step != self._physics_step
        ):
            raise RuntimeError(
                "clip time/step provenance was mutated outside ClipMechanics"
            )
        previous_time = self._time_s
        next_time = previous_time + dt
        if not math.isfinite(next_time) or next_time <= previous_time:
            raise ValueError("dt_s does not advance clip time monotonically")
        next_physics_step = self._physics_step + 1
        previous_gap = state.current_gap_m
        residual_gap = state.residual_gap_m
        damage_fraction = state.damage_fraction

        closure_demand = max(0.0, residual_gap - jaw_gap)
        required_elastic_force = (
            self.material.elastic_closure_stiffness_n_m * closure_demand
        )
        transmitted_force = min(jaw_force, required_elastic_force)
        if transmitted_force > self.material.yield_force_n:
            plastic_force = transmitted_force - self.material.yield_force_n
            plastic_closure = (
                self.material.plastic_compliance_m_n_s * plastic_force * dt
            )
            residual_gap = max(
                jaw_gap,
                residual_gap - plastic_closure,
            )

        elastic_closure = (
            transmitted_force / self.material.elastic_closure_stiffness_n_m
        )
        loaded_gap = (
            residual_gap
            if closure_demand <= _EPSILON
            else max(jaw_gap, residual_gap - elastic_closure)
        )
        if jaw_force <= _EPSILON:
            springback_fraction = 1.0 - math.exp(
                -dt / self.material.springback_time_s
            )
            loaded_gap = state.current_gap_m + springback_fraction * (
                residual_gap - state.current_gap_m
            )
        current_gap = min(
            self.material.open_gap_m, max(0.0, loaded_gap)
        )
        gap_velocity = (current_gap - previous_gap) / dt
        formed_span = max(0.0, self.material.open_gap_m - residual_gap)
        plastic_curvature = (
            formed_span / self.material.leg_length_m**2
        )

        if transmitted_force > self.material.yield_force_n:
            force_utilization = (
                transmitted_force - self.material.yield_force_n
            ) / (
                self.material.failure_force_n - self.material.yield_force_n
            )
            damage_fraction = min(
                1.0,
                damage_fraction
                + self.material.damage_rate_per_s
                * max(0.0, force_utilization)
                * dt,
            )
        if transmitted_force >= self.material.failure_force_n:
            damage_fraction = 1.0

        contacts = tuple(vessel_contacts)
        keyed_contacts: list[
            tuple[ClipContactSample, ContactKey]
        ] = []
        for contact in contacts:
            if contact.scene_component_id == self.scene_component_id:
                raise ValueError("clip vessel contact cannot be self-contact")
            key = contact.contact_key
            if key is None:
                key = ContactKey.canonical(
                    self.scene_component_id,
                    f"clip:{contact.contact_id}",
                    contact.scene_component_id,
                    f"target:{contact.contact_id}",
                )
            if {
                key.body_a,
                key.body_b,
            } != {
                self.scene_component_id,
                contact.scene_component_id,
            }:
                raise ValueError(
                    f"contact key {key!r} does not bind the registered clip "
                    f"{self.scene_component_id!r} to target "
                    f"{contact.scene_component_id!r}"
                )
            keyed_contacts.append((contact, key))
        contact_keys = [key for _, key in keyed_contacts]
        if len(contact_keys) != len(set(contact_keys)):
            raise ValueError(
                "publish one aggregate ClipContactSample per canonical "
                "contact key and physics interval"
            )
        contact_area = sum(contact.contact_area_m2 for contact in contacts)
        contact_force = sum(contact.normal_force_n for contact in contacts)
        measured_tangential_force = _ZERO
        for contact in contacts:
            measured_tangential_force = _add(
                measured_tangential_force,
                contact.tangential_force_n,
            )
        measured_retention_load = _norm(
            measured_tangential_force
        )
        contact_pressure = (
            contact_force / contact_area if contact_area > 0.0 else 0.0
        )
        formed_fraction = min(
            1.0,
            formed_span / self.material.open_gap_m,
        )
        intact_fraction = max(0.0, 1.0 - damage_fraction)
        friction_capacity = sum(
            self.material.contact_friction_coefficient
            * (
                self.material.dynamic_friction_fraction
                + (1.0 - self.material.dynamic_friction_fraction)
                * math.exp(
                    -(
                        contact.relative_slip_speed_m_s
                        / self.material.slip_transition_speed_m_s
                    )
                    ** 2
                )
            )
            * contact.normal_force_n
            for contact in contacts
        )
        interlock_capacity = (
            self.material.geometric_interlock_fraction
            * formed_fraction
            * contact_force
        )
        retention_capacity = (
            friction_capacity + interlock_capacity
        ) * intact_fraction
        tangential_traction = (
            measured_retention_load / contact_area
            if contact_area > 0.0
            else 0.0
        )
        interface_traction = math.hypot(
            contact_pressure, tangential_traction
        )
        contact_component_ids = tuple(
            sorted({contact.scene_component_id for contact in contacts})
        )
        raw_contact_ids = tuple(
            f"{contact.scene_component_id}|{contact.contact_id}"
            for contact in contacts
        )
        observation = ClipObservation(
            scene_component_id=self.scene_component_id,
            contact_component_ids=contact_component_ids,
            residual_gap_m=residual_gap,
            formed_span_m=formed_span,
            retention_load_n=measured_retention_load,
            retention_capacity_n=retention_capacity,
            contact_area_m2=contact_area,
            contact_pressure_pa=contact_pressure,
            interface_traction_pa=interface_traction,
            plastic_curvature_1_m=plastic_curvature,
            damage_fraction=damage_fraction,
            measured_tangential_force_n=measured_tangential_force,
            max_relative_slip_speed_m_s=max(
                (
                    contact.relative_slip_speed_m_s
                    for contact in contacts
                ),
                default=0.0,
            ),
            contact_keys=tuple(contact_keys),
            raw_contact_ids=raw_contact_ids,
            physics_step=next_physics_step,
            previous_time_s=previous_time,
            time_s=next_time,
            dt_s=dt,
            parameter_status=self.material.parameter_status,
        )
        # Commit only after the immutable evidence object validates.
        state.current_gap_m = current_gap
        state.residual_gap_m = residual_gap
        state.gap_velocity_m_s = gap_velocity
        state.plastic_curvature_1_m = plastic_curvature
        state.damage_fraction = damage_fraction
        state.time_s = next_time
        state.physics_step = next_physics_step
        self._time_s = next_time
        self._physics_step = next_physics_step
        self._last_observation = observation
        return observation

    def observe(self) -> ClipObservation:
        if self._last_observation is None:
            raise RuntimeError("advance the clip mechanics before observing it")
        return self._last_observation


__all__ = [
    "ClipContactSample",
    "ClipMaterial",
    "ClipMechanics",
    "ClipObservation",
    "ClipState",
]
