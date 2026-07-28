# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Persistent surgical-scale contact state and regularized Coulomb response.

The classes in this module are backend-neutral.  PhysX, Warp, or another
solver may supply contact samples, while :class:`PersistentContactGraph`
retains the history needed for friction, abrasion, grasp slip, and damage.
No class treats contact presence as proof of attachment or retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping


Vec3 = tuple[float, float, float]
_ZERO: Vec3 = (0.0, 0.0, 0.0)
_EPSILON = 1.0e-12


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
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


def _add(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _subtract(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _scale(value: Vec3, amount: float) -> Vec3:
    return (value[0] * amount, value[1] * amount, value[2] * amount)


def _dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _norm(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _normalized(value: Vec3, label: str) -> Vec3:
    magnitude = _norm(value)
    if magnitude <= _EPSILON:
        raise ValueError(f"{label} must be non-zero")
    return _scale(value, 1.0 / magnitude)


@dataclass(frozen=True, order=True)
class ContactKey:
    """Stable identity for one pair of contacting body features."""

    body_a: str
    feature_a: str
    body_b: str
    feature_b: str

    def __post_init__(self) -> None:
        values = (self.body_a, self.feature_a, self.body_b, self.feature_b)
        if any(not str(value).strip() for value in values):
            raise ValueError("contact body and feature identifiers must be non-empty")

    @classmethod
    def canonical(
        cls,
        body_a: str,
        feature_a: str,
        body_b: str,
        feature_b: str,
    ) -> "ContactKey":
        """Return a deterministic identity independent of input pair ordering."""

        left = (str(body_a), str(feature_a))
        right = (str(body_b), str(feature_b))
        if right < left:
            left, right = right, left
        return cls(left[0], left[1], right[0], right[1])


@dataclass(frozen=True)
class ContactSample:
    """One solver contact sample expressed in SI units.

    ``normal_a_to_b`` follows the order stored in ``key``.  A negative
    separation denotes geometric penetration.  ``tangential_force_n`` is the
    solver's trial force on body A by body B, while
    ``relative_tangential_velocity_m_s`` is A relative to B.  The force is
    clamped by :func:`regularized_coulomb_response`.
    """

    key: ContactKey
    point_m: Vec3
    normal_a_to_b: Vec3
    separation_m: float
    normal_force_n: float
    tangential_force_n: Vec3 = _ZERO
    relative_tangential_velocity_m_s: Vec3 = _ZERO

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_m", _vec3(self.point_m, "point_m"))
        normal = _normalized(
            _vec3(self.normal_a_to_b, "normal_a_to_b"),
            "normal_a_to_b",
        )
        object.__setattr__(
            self,
            "normal_a_to_b",
            normal,
        )
        object.__setattr__(
            self, "separation_m", _finite(self.separation_m, "separation_m")
        )
        object.__setattr__(
            self,
            "normal_force_n",
            _non_negative(self.normal_force_n, "normal_force_n"),
        )
        trial_force = _vec3(self.tangential_force_n, "tangential_force_n")
        trial_velocity = _vec3(
            self.relative_tangential_velocity_m_s,
            "relative_tangential_velocity_m_s",
        )
        object.__setattr__(
            self,
            "tangential_force_n",
            _subtract(trial_force, _scale(normal, _dot(trial_force, normal))),
        )
        object.__setattr__(
            self,
            "relative_tangential_velocity_m_s",
            _subtract(
                trial_velocity,
                _scale(normal, _dot(trial_velocity, normal)),
            ),
        )

    @property
    def slip_speed_m_s(self) -> float:
        return _norm(self.relative_tangential_velocity_m_s)

    @classmethod
    def canonical(
        cls,
        *,
        body_a: str,
        feature_a: str,
        body_b: str,
        feature_b: str,
        point_m: Iterable[float],
        normal_a_to_b: Iterable[float],
        separation_m: float,
        normal_force_n: float,
        tangential_force_on_a_n: Iterable[float] = _ZERO,
        relative_velocity_a_to_b_m_s: Iterable[float] = _ZERO,
    ) -> "ContactSample":
        """Canonicalize identity and every oriented vector atomically."""

        key = ContactKey.canonical(
            body_a,
            feature_a,
            body_b,
            feature_b,
        )
        swapped = (key.body_a, key.feature_a) != (
            str(body_a),
            str(feature_a),
        )
        orientation = -1.0 if swapped else 1.0
        return cls(
            key=key,
            point_m=_vec3(point_m, "point_m"),
            normal_a_to_b=_scale(
                _vec3(normal_a_to_b, "normal_a_to_b"),
                orientation,
            ),
            separation_m=separation_m,
            normal_force_n=normal_force_n,
            tangential_force_n=_scale(
                _vec3(
                    tangential_force_on_a_n,
                    "tangential_force_on_a_n",
                ),
                orientation,
            ),
            relative_tangential_velocity_m_s=_scale(
                _vec3(
                    relative_velocity_a_to_b_m_s,
                    "relative_velocity_a_to_b_m_s",
                ),
                orientation,
            ),
        )


@dataclass(frozen=True)
class CoulombFrictionLaw:
    """Regularized load-, wetness-, and damage-dependent friction law."""

    static_coefficient: float
    dynamic_coefficient: float
    transition_speed_m_s: float = 1.0e-3
    load_reference_n: float = 1.0
    load_exponent: float = 0.0
    wet_scale_at_saturation: float = 1.0
    damage_scale_at_failure: float = 1.0
    minimum_coefficient: float = 0.0
    maximum_coefficient: float = 4.0
    parameter_status: str = "provisional_engineering_seed"

    def __post_init__(self) -> None:
        static = _non_negative(self.static_coefficient, "static_coefficient")
        dynamic = _non_negative(self.dynamic_coefficient, "dynamic_coefficient")
        if dynamic > static:
            raise ValueError("dynamic_coefficient cannot exceed static_coefficient")
        transition = _finite(self.transition_speed_m_s, "transition_speed_m_s")
        if transition <= 0.0:
            raise ValueError("transition_speed_m_s must be positive")
        reference = _finite(self.load_reference_n, "load_reference_n")
        if reference <= 0.0:
            raise ValueError("load_reference_n must be positive")
        _finite(self.load_exponent, "load_exponent")
        _non_negative(self.wet_scale_at_saturation, "wet_scale_at_saturation")
        _non_negative(self.damage_scale_at_failure, "damage_scale_at_failure")
        minimum = _non_negative(self.minimum_coefficient, "minimum_coefficient")
        maximum = _non_negative(self.maximum_coefficient, "maximum_coefficient")
        if maximum < minimum:
            raise ValueError("maximum_coefficient must be at least minimum_coefficient")
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")

    def coefficient(
        self,
        *,
        slip_speed_m_s: float,
        normal_force_n: float,
        wet_fraction: float = 0.0,
        damage_fraction: float = 0.0,
    ) -> float:
        """Return the regularized coefficient for the current contact state."""

        speed = _non_negative(slip_speed_m_s, "slip_speed_m_s")
        normal_force = _non_negative(normal_force_n, "normal_force_n")
        wet = _fraction(wet_fraction, "wet_fraction")
        damage = _fraction(damage_fraction, "damage_fraction")
        stribeck = math.exp(-((speed / self.transition_speed_m_s) ** 2))
        coefficient = self.dynamic_coefficient + (
            self.static_coefficient - self.dynamic_coefficient
        ) * stribeck
        if self.load_exponent != 0.0:
            load_ratio = max(normal_force, _EPSILON) / self.load_reference_n
            coefficient *= load_ratio**self.load_exponent
        coefficient *= 1.0 + wet * (self.wet_scale_at_saturation - 1.0)
        coefficient *= 1.0 + damage * (self.damage_scale_at_failure - 1.0)
        return min(
            self.maximum_coefficient,
            max(self.minimum_coefficient, coefficient),
        )


@dataclass(frozen=True)
class ContactResponse:
    """Friction-limited response for one contact sample."""

    normal_force_n: float
    tangential_force_n: Vec3
    friction_coefficient: float
    sticking: bool
    dissipation_power_w: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normal_force_n",
            _non_negative(self.normal_force_n, "normal_force_n"),
        )
        object.__setattr__(
            self,
            "tangential_force_n",
            _vec3(self.tangential_force_n, "tangential_force_n"),
        )
        object.__setattr__(
            self,
            "friction_coefficient",
            _non_negative(self.friction_coefficient, "friction_coefficient"),
        )
        object.__setattr__(
            self,
            "dissipation_power_w",
            _non_negative(self.dissipation_power_w, "dissipation_power_w"),
        )


def regularized_coulomb_response(
    sample: ContactSample,
    law: CoulombFrictionLaw,
    *,
    wet_fraction: float = 0.0,
    damage_fraction: float = 0.0,
) -> ContactResponse:
    """Clamp a trial tangential force to the current Coulomb envelope."""

    speed = sample.slip_speed_m_s
    coefficient = law.coefficient(
        slip_speed_m_s=speed,
        normal_force_n=sample.normal_force_n,
        wet_fraction=wet_fraction,
        damage_fraction=damage_fraction,
    )
    trial = sample.tangential_force_n
    trial_magnitude = _norm(trial)
    trial_power = _dot(
        trial,
        sample.relative_tangential_velocity_m_s,
    )
    if trial_power > max(
        1.0e-12,
        1.0e-9 * trial_magnitude * max(speed, _EPSILON),
    ):
        raise ValueError(
            "trial tangential force does positive work under the declared "
            "body-A-relative-to-body-B sign convention"
        )
    # The sticking envelope must use the same wetness-, damage-, and
    # load-adjusted coefficient as the sliding response.  Using the dry
    # material constant here would silently ignore the observed interface
    # state exactly at the stick/slip transition.
    static_limit = coefficient * sample.normal_force_n
    sticking = (
        speed <= law.transition_speed_m_s and trial_magnitude <= static_limit
    )
    if sticking:
        tangential = trial
    elif speed > _EPSILON:
        tangential = _scale(
            sample.relative_tangential_velocity_m_s,
            -(coefficient * sample.normal_force_n / speed),
        )
    elif trial_magnitude > _EPSILON:
        limit = coefficient * sample.normal_force_n
        tangential = _scale(trial, min(1.0, limit / trial_magnitude))
    else:
        tangential = _ZERO
    dissipation = max(
        0.0,
        -_dot(tangential, sample.relative_tangential_velocity_m_s),
    )
    return ContactResponse(
        normal_force_n=sample.normal_force_n,
        tangential_force_n=tangential,
        friction_coefficient=coefficient,
        sticking=sticking,
        dissipation_power_w=dissipation,
    )


@dataclass
class ContactHistory:
    """Accumulated history for one persistent contact identity."""

    key: ContactKey
    age_s: float = 0.0
    inactive_time_s: float = 0.0
    normal_impulse_n_s: float = 0.0
    tangential_impulse_n_s: Vec3 = _ZERO
    friction_work_j: float = 0.0
    slip_distance_m: float = 0.0
    max_normal_force_n: float = 0.0
    last_point_m: Vec3 = _ZERO
    last_normal_a_to_b: Vec3 = (0.0, 0.0, 1.0)
    sticking: bool = False

    def integrate(
        self,
        sample: ContactSample,
        response: ContactResponse,
        dt_s: float,
    ) -> None:
        dt = _finite(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        if sample.key != self.key:
            raise ValueError("contact sample key does not match contact history")
        self.age_s += dt
        self.inactive_time_s = 0.0
        self.normal_impulse_n_s += response.normal_force_n * dt
        self.tangential_impulse_n_s = _add(
            self.tangential_impulse_n_s,
            _scale(response.tangential_force_n, dt),
        )
        self.friction_work_j += response.dissipation_power_w * dt
        self.slip_distance_m += sample.slip_speed_m_s * dt
        self.max_normal_force_n = max(
            self.max_normal_force_n, response.normal_force_n
        )
        self.last_point_m = sample.point_m
        self.last_normal_a_to_b = sample.normal_a_to_b
        self.sticking = response.sticking


@dataclass(frozen=True)
class ContactSnapshot:
    """Immutable observation exported by :class:`PersistentContactGraph`."""

    key: ContactKey
    age_s: float
    inactive_time_s: float
    normal_impulse_n_s: float
    tangential_impulse_n_s: Vec3
    friction_work_j: float
    slip_distance_m: float
    max_normal_force_n: float
    last_point_m: Vec3
    last_normal_a_to_b: Vec3
    sticking: bool


@dataclass
class PersistentContactGraph:
    """Deterministic contact-history registry shared by surgical mechanics."""

    histories: dict[ContactKey, ContactHistory] = field(default_factory=dict)
    _step_dt_s: float | None = field(default=None, init=False, repr=False)
    _seen: set[ContactKey] = field(default_factory=set, init=False, repr=False)

    def begin_step(self, dt_s: float) -> None:
        if self._step_dt_s is not None:
            raise RuntimeError("contact graph step is already active")
        dt = _finite(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        self._step_dt_s = dt
        self._seen.clear()

    def observe(
        self,
        sample: ContactSample,
        *,
        response: ContactResponse | None = None,
        friction_law: CoulombFrictionLaw | None = None,
        wet_fraction: float = 0.0,
        damage_fraction: float = 0.0,
    ) -> ContactResponse:
        if self._step_dt_s is None:
            raise RuntimeError("begin_step must be called before observe")
        if sample.key in self._seen:
            raise ValueError(
                "each canonical contact key may be observed only once per step; "
                "aggregate solver points before publishing the sample"
            )
        if response is None:
            if friction_law is None:
                raise ValueError("response or friction_law is required")
            response = regularized_coulomb_response(
                sample,
                friction_law,
                wet_fraction=wet_fraction,
                damage_fraction=damage_fraction,
            )
        else:
            force_scale = max(
                1.0,
                sample.normal_force_n,
                _norm(response.tangential_force_n),
            )
            tolerance = 1.0e-9 * force_scale
            if not math.isclose(
                response.normal_force_n,
                sample.normal_force_n,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "custom response normal force does not match the sample"
                )
            if abs(
                _dot(
                    response.tangential_force_n,
                    sample.normal_a_to_b,
                )
            ) > tolerance:
                raise ValueError(
                    "custom response tangential force is not tangent to "
                    "the contact normal"
                )
            if _norm(response.tangential_force_n) > (
                response.friction_coefficient
                * response.normal_force_n
                + tolerance
            ):
                raise ValueError(
                    "custom response exceeds its declared Coulomb envelope"
                )
            power = _dot(
                response.tangential_force_n,
                sample.relative_tangential_velocity_m_s,
            )
            if power > max(1.0e-12, tolerance * sample.slip_speed_m_s):
                raise ValueError(
                    "custom response tangential force injects energy"
                )
            expected_dissipation = max(0.0, -power)
            if not math.isclose(
                response.dissipation_power_w,
                expected_dissipation,
                rel_tol=1.0e-8,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "custom response dissipation does not match force power"
                )
        history = self.histories.setdefault(
            sample.key, ContactHistory(key=sample.key)
        )
        history.integrate(sample, response, self._step_dt_s)
        self._seen.add(sample.key)
        return response

    def end_step(self, *, max_inactive_s: float = 0.05) -> tuple[ContactKey, ...]:
        if self._step_dt_s is None:
            raise RuntimeError("begin_step must be called before end_step")
        retention = _non_negative(max_inactive_s, "max_inactive_s")
        dt = self._step_dt_s
        removed: list[ContactKey] = []
        for key, history in tuple(self.histories.items()):
            if key not in self._seen:
                history.inactive_time_s += dt
                history.sticking = False
                if history.inactive_time_s > retention:
                    removed.append(key)
                    del self.histories[key]
        self._step_dt_s = None
        self._seen.clear()
        return tuple(sorted(removed))

    def clear(self) -> None:
        self.histories.clear()
        self._step_dt_s = None
        self._seen.clear()

    def snapshot(self) -> tuple[ContactSnapshot, ...]:
        return tuple(
            ContactSnapshot(
                key=history.key,
                age_s=history.age_s,
                inactive_time_s=history.inactive_time_s,
                normal_impulse_n_s=history.normal_impulse_n_s,
                tangential_impulse_n_s=history.tangential_impulse_n_s,
                friction_work_j=history.friction_work_j,
                slip_distance_m=history.slip_distance_m,
                max_normal_force_n=history.max_normal_force_n,
                last_point_m=history.last_point_m,
                last_normal_a_to_b=history.last_normal_a_to_b,
                sticking=history.sticking,
            )
            for _, history in sorted(self.histories.items())
        )

    def restore(self, snapshots: Iterable[ContactSnapshot]) -> None:
        restored: dict[ContactKey, ContactHistory] = {}
        for item in snapshots:
            if item.key in restored:
                raise ValueError(f"duplicate contact snapshot for {item.key}")
            restored[item.key] = ContactHistory(
                key=item.key,
                age_s=_non_negative(item.age_s, "age_s"),
                inactive_time_s=_non_negative(
                    item.inactive_time_s, "inactive_time_s"
                ),
                normal_impulse_n_s=_non_negative(
                    item.normal_impulse_n_s, "normal_impulse_n_s"
                ),
                tangential_impulse_n_s=_vec3(
                    item.tangential_impulse_n_s, "tangential_impulse_n_s"
                ),
                friction_work_j=_non_negative(
                    item.friction_work_j, "friction_work_j"
                ),
                slip_distance_m=_non_negative(
                    item.slip_distance_m, "slip_distance_m"
                ),
                max_normal_force_n=_non_negative(
                    item.max_normal_force_n, "max_normal_force_n"
                ),
                last_point_m=_vec3(item.last_point_m, "last_point_m"),
                last_normal_a_to_b=_normalized(
                    _vec3(item.last_normal_a_to_b, "last_normal_a_to_b"),
                    "last_normal_a_to_b",
                ),
                sticking=bool(item.sticking),
            )
        self.histories = restored
        self._step_dt_s = None
        self._seen.clear()

    def by_body(self, body_id: str) -> Mapping[ContactKey, ContactHistory]:
        selected = str(body_id)
        return {
            key: history
            for key, history in self.histories.items()
            if key.body_a == selected or key.body_b == selected
        }


__all__ = [
    "ContactHistory",
    "ContactKey",
    "ContactResponse",
    "ContactSample",
    "ContactSnapshot",
    "CoulombFrictionLaw",
    "PersistentContactGraph",
    "Vec3",
    "regularized_coulomb_response",
]
