# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral cohesive interfaces with irreversible mixed-mode damage.

The implementation is a compact bilinear traction-separation foundation.  It
is suitable for tissue tears, adhesive patches, staple tracts, and breakable
instrument interfaces.  Compression remains a contact response after tensile
or shear cohesion has failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from .contact import Vec3


_ZERO: Vec3 = (0.0, 0.0, 0.0)
_EPSILON = 1.0e-12


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
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise ValueError(f"{label} must contain three finite values")
    return result  # type: ignore[return-value]


def _norm(value: Vec3) -> float:
    return math.sqrt(sum(component * component for component in value))


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _subtract(left: Vec3, right: Vec3) -> Vec3:
    return tuple(
        a - b for a, b in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _scale(value: Vec3, amount: float) -> Vec3:
    return tuple(component * amount for component in value)  # type: ignore[return-value]


def _unit(value: Vec3, label: str) -> Vec3:
    magnitude = _norm(value)
    if magnitude <= _EPSILON:
        raise ValueError(f"{label} must be non-zero")
    return _scale(value, 1.0 / magnitude)


@dataclass(frozen=True)
class CohesiveLaw:
    """Bilinear mixed-mode traction-separation parameters in SI units."""

    normal_stiffness_pa_per_m: float
    shear_stiffness_pa_per_m: float
    peak_normal_traction_pa: float
    peak_shear_traction_pa: float
    normal_failure_opening_m: float
    shear_failure_slip_m: float
    compression_stiffness_pa_per_m: float | None = None
    normal_viscosity_pa_s_per_m: float = 0.0
    shear_viscosity_pa_s_per_m: float = 0.0
    mixed_mode_power: float = 2.0
    parameter_status: str = "provisional_engineering_seed"

    def __post_init__(self) -> None:
        normal_stiffness = _positive(
            self.normal_stiffness_pa_per_m, "normal_stiffness_pa_per_m"
        )
        shear_stiffness = _positive(
            self.shear_stiffness_pa_per_m, "shear_stiffness_pa_per_m"
        )
        peak_normal = _positive(
            self.peak_normal_traction_pa, "peak_normal_traction_pa"
        )
        peak_shear = _positive(
            self.peak_shear_traction_pa, "peak_shear_traction_pa"
        )
        normal_failure = _positive(
            self.normal_failure_opening_m, "normal_failure_opening_m"
        )
        shear_failure = _positive(
            self.shear_failure_slip_m, "shear_failure_slip_m"
        )
        normal_onset = peak_normal / normal_stiffness
        shear_onset = peak_shear / shear_stiffness
        if normal_failure <= normal_onset:
            raise ValueError(
                "normal_failure_opening_m must exceed elastic damage onset"
            )
        if shear_failure <= shear_onset:
            raise ValueError(
                "shear_failure_slip_m must exceed elastic damage onset"
            )
        if self.compression_stiffness_pa_per_m is not None:
            _positive(
                self.compression_stiffness_pa_per_m,
                "compression_stiffness_pa_per_m",
            )
        _non_negative(
            self.normal_viscosity_pa_s_per_m, "normal_viscosity_pa_s_per_m"
        )
        _non_negative(
            self.shear_viscosity_pa_s_per_m, "shear_viscosity_pa_s_per_m"
        )
        mixed_mode_power = _finite(self.mixed_mode_power, "mixed_mode_power")
        if mixed_mode_power < 1.0:
            raise ValueError("mixed_mode_power must be at least one")
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")

    @property
    def normal_damage_onset_m(self) -> float:
        return self.peak_normal_traction_pa / self.normal_stiffness_pa_per_m

    @property
    def shear_damage_onset_m(self) -> float:
        return self.peak_shear_traction_pa / self.shear_stiffness_pa_per_m

    @property
    def compression_stiffness(self) -> float:
        return (
            self.normal_stiffness_pa_per_m
            if self.compression_stiffness_pa_per_m is None
            else self.compression_stiffness_pa_per_m
        )

    def directional_limits(
        self,
        normal_opening_m: float,
        shear_slip_m: float,
    ) -> tuple[float, float, float]:
        """Return effective separation, onset, and failure along this mode mix."""

        normal = max(0.0, _finite(normal_opening_m, "normal_opening_m"))
        shear = _non_negative(shear_slip_m, "shear_slip_m")
        effective = math.hypot(normal, shear)
        if effective <= _EPSILON:
            return 0.0, self.normal_damage_onset_m, self.normal_failure_opening_m
        normal_direction = normal / effective
        shear_direction = shear / effective
        power = self.mixed_mode_power

        def directional_radius(normal_limit: float, shear_limit: float) -> float:
            inverse = (
                (normal_direction / normal_limit) ** power
                + (shear_direction / shear_limit) ** power
            )
            return inverse ** (-1.0 / power)

        onset = directional_radius(
            self.normal_damage_onset_m,
            self.shear_damage_onset_m,
        )
        failure = directional_radius(
            self.normal_failure_opening_m,
            self.shear_failure_slip_m,
        )
        return effective, onset, failure


@dataclass
class CohesiveState:
    """Irreversible state for one cohesive patch."""

    damage: float = 0.0
    max_effective_separation_m: float = 0.0
    dissipated_energy_j_m2: float = 0.0
    age_s: float = 0.0
    step_index: int = -1
    last_simulation_time_s: float = -1.0
    failed: bool = False
    last_normal_opening_m: float = 0.0
    last_shear_slip_m: Vec3 = _ZERO

    def __post_init__(self) -> None:
        self.damage = _fraction(self.damage, "damage")
        self.max_effective_separation_m = _non_negative(
            self.max_effective_separation_m, "max_effective_separation_m"
        )
        self.dissipated_energy_j_m2 = _non_negative(
            self.dissipated_energy_j_m2, "dissipated_energy_j_m2"
        )
        self.age_s = _non_negative(self.age_s, "age_s")
        if int(self.step_index) < -1:
            raise ValueError("step_index must be at least -1")
        self.step_index = int(self.step_index)
        self.last_simulation_time_s = _finite(
            self.last_simulation_time_s,
            "last_simulation_time_s",
        )
        if self.step_index < 0 and self.last_simulation_time_s >= 0.0:
            raise ValueError(
                "unadvanced cohesive state cannot have simulation time"
            )
        if self.step_index >= 0 and self.last_simulation_time_s < 0.0:
            raise ValueError(
                "advanced cohesive state requires simulation time"
            )
        self.last_normal_opening_m = _finite(
            self.last_normal_opening_m, "last_normal_opening_m"
        )
        self.last_shear_slip_m = _vec3(
            self.last_shear_slip_m, "last_shear_slip_m"
        )
        self.failed = bool(self.failed or self.damage >= 1.0 - 1.0e-9)
        if self.failed:
            self.damage = 1.0


@dataclass(frozen=True)
class CohesiveResponse:
    """Observable interface response after one causal state update."""

    interface_id: str
    body_a_component_id: str
    body_b_component_id: str
    normal_a_to_b: Vec3
    step_index: int
    time_s: float
    contact_area_m2: float
    normal_pressure_pa: float
    normal_traction_pa: float
    shear_traction_pa: Vec3
    resultant_force_n: float
    damage: float
    failed: bool
    stored_energy_j_m2: float
    dissipated_energy_j_m2: float
    parameter_status: str = "provisional_engineering_seed"

    def __post_init__(self) -> None:
        if not str(self.interface_id).strip():
            raise ValueError("interface_id must be non-empty")
        body_a = str(self.body_a_component_id).strip()
        body_b = str(self.body_b_component_id).strip()
        if not body_a:
            raise ValueError("body_a_component_id must be non-empty")
        if not body_b:
            raise ValueError("body_b_component_id must be non-empty")
        if body_a == body_b:
            raise ValueError("a cohesive interface requires two distinct bodies")
        object.__setattr__(self, "body_a_component_id", body_a)
        object.__setattr__(self, "body_b_component_id", body_b)
        normal = _unit(
            _vec3(self.normal_a_to_b, "normal_a_to_b"),
            "normal_a_to_b",
        )
        object.__setattr__(self, "normal_a_to_b", normal)
        if int(self.step_index) < 0:
            raise ValueError("step_index must be non-negative")
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(
            self,
            "time_s",
            _non_negative(self.time_s, "time_s"),
        )
        object.__setattr__(
            self,
            "contact_area_m2",
            _positive(self.contact_area_m2, "contact_area_m2"),
        )
        object.__setattr__(
            self,
            "normal_pressure_pa",
            _non_negative(self.normal_pressure_pa, "normal_pressure_pa"),
        )
        object.__setattr__(
            self,
            "normal_traction_pa",
            _non_negative(self.normal_traction_pa, "normal_traction_pa"),
        )
        object.__setattr__(
            self,
            "shear_traction_pa",
            _vec3(self.shear_traction_pa, "shear_traction_pa"),
        )
        if abs(_dot(self.shear_traction_pa, normal)) > max(
            1.0e-9,
            1.0e-9 * _norm(self.shear_traction_pa),
        ):
            raise ValueError(
                "shear_traction_pa must lie in the registered interface plane"
            )
        object.__setattr__(
            self,
            "resultant_force_n",
            _non_negative(self.resultant_force_n, "resultant_force_n"),
        )
        object.__setattr__(self, "damage", _fraction(self.damage, "damage"))
        if self.failed and self.damage < 1.0 - 1.0e-9:
            raise ValueError("failed cohesive response must have unit damage")
        object.__setattr__(
            self,
            "stored_energy_j_m2",
            _non_negative(self.stored_energy_j_m2, "stored_energy_j_m2"),
        )
        object.__setattr__(
            self,
            "dissipated_energy_j_m2",
            _non_negative(
                self.dissipated_energy_j_m2, "dissipated_energy_j_m2"
            ),
        )
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")

    @property
    def interface_traction_pa(self) -> float:
        return math.sqrt(
            self.net_normal_traction_pa * self.net_normal_traction_pa
            + sum(value * value for value in self.shear_traction_pa)
        )

    @property
    def cohesive_traction_magnitude_pa(self) -> float:
        """Magnitude of tensile/shear cohesion, excluding compression."""

        return math.sqrt(
            self.normal_traction_pa * self.normal_traction_pa
            + sum(value * value for value in self.shear_traction_pa)
        )

    @property
    def net_normal_traction_pa(self) -> float:
        """Signed normal traction: tension positive, compression negative."""

        return self.normal_traction_pa - self.normal_pressure_pa


@dataclass
class CohesiveInterface:
    """Stateful cohesive patch with a fixed reference contact area."""

    interface_id: str
    body_a_component_id: str
    body_b_component_id: str
    normal_a_to_b: Vec3
    contact_area_m2: float
    law: CohesiveLaw
    state: CohesiveState = field(default_factory=CohesiveState)
    _last_response: CohesiveResponse | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not str(self.interface_id).strip():
            raise ValueError("interface_id must be non-empty")
        if not str(self.body_a_component_id).strip():
            raise ValueError("body_a_component_id must be non-empty")
        if not str(self.body_b_component_id).strip():
            raise ValueError("body_b_component_id must be non-empty")
        if self.body_a_component_id == self.body_b_component_id:
            raise ValueError("a cohesive interface requires two distinct bodies")
        self.normal_a_to_b = _unit(
            _vec3(self.normal_a_to_b, "normal_a_to_b"),
            "normal_a_to_b",
        )
        self.contact_area_m2 = _positive(self.contact_area_m2, "contact_area_m2")
        normal_slip = _dot(
            self.state.last_shear_slip_m,
            self.normal_a_to_b,
        )
        if abs(normal_slip) > max(
            1.0e-12,
            1.0e-9 * _norm(self.state.last_shear_slip_m),
        ):
            raise ValueError(
                "state shear slip must lie in the registered interface plane"
            )

    def advance(
        self,
        *,
        normal_opening_m: float,
        shear_slip_m: Iterable[float],
        dt_s: float,
        physics_step: int,
        simulation_time_s: float,
        compressive_pressure_pa: float = 0.0,
    ) -> CohesiveResponse:
        """Advance irreversible damage and return the current interface traction."""

        dt = _positive(dt_s, "dt_s")
        step = int(physics_step)
        if step < 0:
            raise ValueError("physics_step must be non-negative")
        simulation_time = _non_negative(
            simulation_time_s,
            "simulation_time_s",
        )
        if step <= self.state.step_index:
            raise ValueError(
                "cohesive evidence must use a strictly increasing physics step"
            )
        if (
            self.state.step_index >= 0
            and simulation_time <= self.state.last_simulation_time_s
        ):
            raise ValueError(
                "cohesive evidence must use increasing simulation time"
            )
        if (
            self.state.step_index >= 0
            and not math.isclose(
                simulation_time - self.state.last_simulation_time_s,
                dt,
                rel_tol=1.0e-6,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError(
                "dt_s must match the cohesive evidence clock interval"
            )
        opening = _finite(normal_opening_m, "normal_opening_m")
        shear = _vec3(shear_slip_m, "shear_slip_m")
        normal_shear = _dot(shear, self.normal_a_to_b)
        if abs(normal_shear) > max(
            1.0e-12,
            1.0e-9 * _norm(shear),
        ):
            raise ValueError(
                "shear_slip_m must lie in the registered interface plane"
            )
        shear = _subtract(
            shear,
            _scale(self.normal_a_to_b, normal_shear),
        )
        shear_magnitude = _norm(shear)
        imposed_pressure = _non_negative(
            compressive_pressure_pa, "compressive_pressure_pa"
        )
        effective, onset, failure = self.law.directional_limits(
            opening, shear_magnitude
        )
        previous_damage = self.state.damage
        candidate_damage = previous_damage
        if effective > onset:
            # Secant damage for a genuinely bilinear law.  A simple
            # ``(delta-onset)/(failure-onset)`` damage ramp multiplied by
            # ``K * delta`` produces a spurious post-onset force increase.
            # This form makes the effective traction fall linearly from the
            # peak at onset to zero at failure.
            softening_damage = 1.0 - (
                onset
                * max(0.0, failure - effective)
                / max(effective * (failure - onset), _EPSILON)
            )
            candidate_damage = min(
                1.0,
                max(
                    previous_damage,
                    softening_damage,
                ),
            )

        positive_opening = max(0.0, opening)
        previous_opening = max(0.0, self.state.last_normal_opening_m)
        normal_rate = (positive_opening - previous_opening) / dt
        shear_rate = tuple(
            (shear[index] - self.state.last_shear_slip_m[index]) / dt
            for index in range(3)
        )
        undamaged_energy = 0.5 * (
            self.law.normal_stiffness_pa_per_m * positive_opening**2
            + self.law.shear_stiffness_pa_per_m * shear_magnitude**2
        )
        damage_increment = candidate_damage - previous_damage
        self.state.dissipated_energy_j_m2 += max(
            0.0, damage_increment * undamaged_energy
        )
        self.state.damage = candidate_damage
        self.state.failed = bool(
            self.state.failed
            or candidate_damage >= 1.0 - 1.0e-9
        )
        if self.state.failed:
            self.state.damage = 1.0
        self.state.max_effective_separation_m = max(
            self.state.max_effective_separation_m, effective
        )
        self.state.age_s += dt
        self.state.step_index = step
        self.state.last_simulation_time_s = simulation_time
        self.state.last_normal_opening_m = opening
        self.state.last_shear_slip_m = shear

        intact_fraction = max(0.0, 1.0 - self.state.damage)
        viscous_dissipation = intact_fraction * dt * (
            self.law.normal_viscosity_pa_s_per_m
            * max(0.0, normal_rate) ** 2
            + self.law.shear_viscosity_pa_s_per_m
            * sum(value * value for value in shear_rate)
        )
        self.state.dissipated_energy_j_m2 += max(
            0.0,
            viscous_dissipation,
        )
        elastic_normal = (
            self.law.normal_stiffness_pa_per_m * positive_opening * intact_fraction
        )
        viscous_normal = (
            self.law.normal_viscosity_pa_s_per_m
            * max(0.0, normal_rate)
            * intact_fraction
        )
        normal_traction = elastic_normal + viscous_normal
        shear_traction = tuple(
            intact_fraction
            * (
                self.law.shear_stiffness_pa_per_m * shear[index]
                + self.law.shear_viscosity_pa_s_per_m * shear_rate[index]
            )
            for index in range(3)
        )
        compression = max(
            imposed_pressure,
            self.law.compression_stiffness * max(0.0, -opening),
        )
        compression_penetration = max(0.0, -opening)
        compression_energy = (
            0.5
            * self.law.compression_stiffness
            * compression_penetration**2
        )
        stored_energy = (
            intact_fraction * undamaged_energy + compression_energy
        )
        # Compression and opening traction act in opposite normal directions;
        # adding their magnitudes would invent load when an imposed pressure
        # coexists with positive opening.
        net_normal_traction = normal_traction - compression
        resultant_force = self.contact_area_m2 * math.sqrt(
            net_normal_traction**2
            + sum(value * value for value in shear_traction)
        )
        response = CohesiveResponse(
            interface_id=self.interface_id,
            body_a_component_id=self.body_a_component_id,
            body_b_component_id=self.body_b_component_id,
            normal_a_to_b=self.normal_a_to_b,
            step_index=self.state.step_index,
            time_s=simulation_time,
            contact_area_m2=self.contact_area_m2,
            normal_pressure_pa=compression,
            normal_traction_pa=normal_traction,
            shear_traction_pa=shear_traction,
            resultant_force_n=resultant_force,
            damage=self.state.damage,
            failed=self.state.failed,
            stored_energy_j_m2=stored_energy,
            dissipated_energy_j_m2=self.state.dissipated_energy_j_m2,
            parameter_status=self.law.parameter_status,
        )
        self._last_response = response
        return response

    def traction_only(self) -> CohesiveResponse:
        """Return the exact last physical response without evolving state."""

        if self._last_response is None:
            raise RuntimeError(
                "advance the cohesive interface before observing traction"
            )
        return self._last_response


__all__ = [
    "CohesiveInterface",
    "CohesiveLaw",
    "CohesiveResponse",
    "CohesiveState",
]
