# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Reduced-order rod mechanics shared by needles, sutures, staples, and clips.

This module supplies deterministic state and constitutive integration without
claiming a particular GPU backend.  Collision impulses and instrument forces
enter as external nodal loads; all reaction and failure observations are
derived from the mutable rod state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

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


def _add(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _sub(left: Vec3, right: Vec3) -> Vec3:
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


@dataclass(frozen=True)
class RodMaterial:
    """Effective one-dimensional material parameters in SI units."""

    density_kg_m3: float
    cross_section_area_m2: float
    axial_stiffness_n: float
    bending_stiffness_n_m2: float
    torsional_stiffness_n_m2: float
    linear_damping_per_s: float
    angular_damping_per_s: float
    yield_strain: float
    failure_strain: float
    yield_curvature_1_m: float
    failure_curvature_1_m: float
    damage_rate_per_s: float = 4.0
    plastic_flow_rate_per_s: float = 12.0
    hardening_fraction: float = 0.05
    parameter_status: str = "provisional_engineering_seed"

    def __post_init__(self) -> None:
        _positive(self.density_kg_m3, "density_kg_m3")
        _positive(self.cross_section_area_m2, "cross_section_area_m2")
        _positive(self.axial_stiffness_n, "axial_stiffness_n")
        _positive(self.bending_stiffness_n_m2, "bending_stiffness_n_m2")
        _positive(self.torsional_stiffness_n_m2, "torsional_stiffness_n_m2")
        _non_negative(self.linear_damping_per_s, "linear_damping_per_s")
        _non_negative(self.angular_damping_per_s, "angular_damping_per_s")
        yield_strain = _non_negative(self.yield_strain, "yield_strain")
        failure_strain = _positive(self.failure_strain, "failure_strain")
        if failure_strain <= yield_strain:
            raise ValueError("failure_strain must exceed yield_strain")
        yield_curvature = _non_negative(
            self.yield_curvature_1_m, "yield_curvature_1_m"
        )
        failure_curvature = _positive(
            self.failure_curvature_1_m, "failure_curvature_1_m"
        )
        if failure_curvature <= yield_curvature:
            raise ValueError(
                "failure_curvature_1_m must exceed yield_curvature_1_m"
            )
        _non_negative(self.damage_rate_per_s, "damage_rate_per_s")
        _non_negative(self.plastic_flow_rate_per_s, "plastic_flow_rate_per_s")
        _fraction(self.hardening_fraction, "hardening_fraction")
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")


@dataclass(frozen=True)
class RodExternalLoad:
    """Force and torque applied to one rod node by a causal external interaction."""

    node_index: int
    force_n: Vec3 = _ZERO
    torque_n_m: float = 0.0
    source_id: str = "external"

    def __post_init__(self) -> None:
        if int(self.node_index) < 0:
            raise ValueError("node_index must be non-negative")
        object.__setattr__(self, "node_index", int(self.node_index))
        object.__setattr__(self, "force_n", _vec3(self.force_n, "force_n"))
        object.__setattr__(
            self, "torque_n_m", _finite(self.torque_n_m, "torque_n_m")
        )
        if not str(self.source_id).strip():
            raise ValueError("source_id must be non-empty")


@dataclass
class RodState:
    """Mutable maximal-coordinate state for a segmented rod."""

    positions_m: list[Vec3]
    velocities_m_s: list[Vec3]
    twist_rad: list[float]
    angular_velocity_rad_s: list[float]
    plastic_curvature_1_m: list[Vec3]
    edge_damage: list[float]
    edge_intact: list[bool]
    time_s: float = 0.0
    physics_step: int = 0

    def __post_init__(self) -> None:
        self.positions_m = [
            _vec3(value, "positions_m") for value in self.positions_m
        ]
        if len(self.positions_m) < 2:
            raise ValueError("a rod requires at least two positions")
        self.velocities_m_s = [
            _vec3(value, "velocities_m_s") for value in self.velocities_m_s
        ]
        if len(self.velocities_m_s) != len(self.positions_m):
            raise ValueError("velocities_m_s must match positions_m")
        edge_count = len(self.positions_m) - 1
        self.twist_rad = [
            _finite(value, "twist_rad") for value in self.twist_rad
        ]
        self.angular_velocity_rad_s = [
            _finite(value, "angular_velocity_rad_s")
            for value in self.angular_velocity_rad_s
        ]
        if len(self.twist_rad) != edge_count:
            raise ValueError("twist_rad must contain one value per edge")
        if len(self.angular_velocity_rad_s) != edge_count:
            raise ValueError(
                "angular_velocity_rad_s must contain one value per edge"
            )
        interior_count = max(0, len(self.positions_m) - 2)
        self.plastic_curvature_1_m = [
            _vec3(value, "plastic_curvature_1_m")
            for value in self.plastic_curvature_1_m
        ]
        if len(self.plastic_curvature_1_m) != interior_count:
            raise ValueError(
                "plastic_curvature_1_m must contain one value per interior node"
            )
        self.edge_damage = [
            _fraction(value, "edge_damage") for value in self.edge_damage
        ]
        if len(self.edge_damage) != edge_count:
            raise ValueError("edge_damage must contain one value per edge")
        self.edge_intact = [bool(value) for value in self.edge_intact]
        if len(self.edge_intact) != edge_count:
            raise ValueError("edge_intact must contain one value per edge")
        self.time_s = _non_negative(self.time_s, "time_s")
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step < 0:
            raise ValueError("physics_step must be non-negative")
        self.physics_step = physics_step
        for index, damage in enumerate(self.edge_damage):
            if damage >= 1.0 - 1.0e-9:
                self.edge_intact[index] = False

    @classmethod
    def from_rest_positions(cls, positions_m: Sequence[Iterable[float]]) -> "RodState":
        positions = [_vec3(value, "positions_m") for value in positions_m]
        if len(positions) < 2:
            raise ValueError("a rod requires at least two positions")
        edge_count = len(positions) - 1
        return cls(
            positions_m=positions,
            velocities_m_s=[_ZERO for _ in positions],
            twist_rad=[0.0 for _ in range(edge_count)],
            angular_velocity_rad_s=[0.0 for _ in range(edge_count)],
            plastic_curvature_1_m=[
                _ZERO for _ in range(max(0, len(positions) - 2))
            ],
            edge_damage=[0.0 for _ in range(edge_count)],
            edge_intact=[True for _ in range(edge_count)],
        )


@dataclass(frozen=True)
class RodStepResult:
    """Immutable observation from one rod advance."""

    component_id: str
    internal_force_n: tuple[Vec3, ...]
    reaction_force_n: tuple[Vec3, ...]
    broken_edge_indices: tuple[int, ...]
    max_abs_strain: float
    max_curvature_1_m: float
    elastic_energy_j: float
    dissipated_energy_j: float
    time_s: float
    parameter_status: str
    physics_step: int
    previous_time_s: float
    dt_s: float

    def __post_init__(self) -> None:
        if not str(self.component_id).strip():
            raise ValueError("component_id must be non-empty")
        internal = tuple(
            _vec3(value, "internal_force_n") for value in self.internal_force_n
        )
        reaction = tuple(
            _vec3(value, "reaction_force_n") for value in self.reaction_force_n
        )
        if len(internal) != len(reaction):
            raise ValueError("internal and reaction force arrays must match")
        object.__setattr__(self, "internal_force_n", internal)
        object.__setattr__(self, "reaction_force_n", reaction)
        broken = tuple(int(value) for value in self.broken_edge_indices)
        if (
            broken != tuple(sorted(set(broken)))
            or any(value < 0 for value in broken)
        ):
            raise ValueError(
                "broken_edge_indices must be sorted, unique, and non-negative"
            )
        object.__setattr__(self, "broken_edge_indices", broken)
        for field_name in (
            "max_abs_strain",
            "max_curvature_1_m",
            "elastic_energy_j",
            "dissipated_energy_j",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative(getattr(self, field_name), field_name),
            )
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step <= 0:
            raise ValueError("physics_step must be a positive integer")
        object.__setattr__(self, "physics_step", physics_step)
        previous_time = _non_negative(self.previous_time_s, "previous_time_s")
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
                "time_s must be the strictly later endpoint of the reported "
                "rod step interval"
            )
        object.__setattr__(self, "previous_time_s", previous_time)
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "dt_s", dt)
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")


@dataclass
class CosseratRod:
    """Reduced-order rod with axial, bending, twist, plasticity, and failure."""

    component_id: str
    material: RodMaterial
    state: RodState
    rest_positions_m: tuple[Vec3, ...] | None = None
    fixed_nodes: dict[int, Vec3] = field(default_factory=dict)
    gravity_m_s2: Vec3 = (0.0, 0.0, -9.81)
    fixed_node_velocities_m_s: dict[int, Vec3] = field(default_factory=dict)
    _rest_lengths_m: tuple[float, ...] = field(init=False, repr=False)
    _rest_curvature_1_m: tuple[Vec3, ...] = field(init=False, repr=False)
    _node_mass_kg: tuple[float, ...] = field(init=False, repr=False)
    _time_s: float = field(init=False, repr=False)
    _physics_step: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.component_id).strip():
            raise ValueError("component_id must be non-empty")
        if self.rest_positions_m is None:
            self.rest_positions_m = tuple(self.state.positions_m)
        else:
            self.rest_positions_m = tuple(
                _vec3(value, "rest_positions_m") for value in self.rest_positions_m
            )
        if len(self.rest_positions_m) != len(self.state.positions_m):
            raise ValueError("rest_positions_m must match rod state positions")
        self.gravity_m_s2 = _vec3(self.gravity_m_s2, "gravity_m_s2")
        self.fixed_nodes = {
            int(index): _vec3(position, "fixed node position")
            for index, position in self.fixed_nodes.items()
        }
        self.fixed_node_velocities_m_s = {
            int(index): _vec3(velocity, "fixed node velocity")
            for index, velocity in self.fixed_node_velocities_m_s.items()
        }
        for index in self.fixed_nodes:
            self._validate_node(index)
        unknown_velocity_nodes = (
            self.fixed_node_velocities_m_s.keys() - self.fixed_nodes.keys()
        )
        if unknown_velocity_nodes:
            raise ValueError(
                "fixed-node velocities require matching fixed-node positions: "
                f"{sorted(unknown_velocity_nodes)}"
            )
        self._rest_lengths_m = tuple(
            _positive(
                _norm(_sub(self.rest_positions_m[index + 1], position)),
                "rest edge length",
            )
            for index, position in enumerate(self.rest_positions_m[:-1])
        )
        self._rest_curvature_1_m = tuple(
            self._curvature_at(self.rest_positions_m, index)
            for index in range(1, len(self.rest_positions_m) - 1)
        )
        linear_density = (
            self.material.density_kg_m3 * self.material.cross_section_area_m2
        )
        masses: list[float] = []
        for node_index in range(len(self.rest_positions_m)):
            supported_length = 0.0
            if node_index > 0:
                supported_length += 0.5 * self._rest_lengths_m[node_index - 1]
            if node_index < len(self._rest_lengths_m):
                supported_length += 0.5 * self._rest_lengths_m[node_index]
            masses.append(_positive(linear_density * supported_length, "node mass"))
        self._node_mass_kg = tuple(masses)
        self._time_s = self.state.time_s
        self._physics_step = self.state.physics_step

    def _validate_node(self, node_index: int) -> None:
        if not 0 <= int(node_index) < len(self.state.positions_m):
            raise IndexError(f"rod node index outside range: {node_index}")

    @staticmethod
    def _curvature_at(positions: Sequence[Vec3], node_index: int) -> Vec3:
        left_edge = _sub(positions[node_index], positions[node_index - 1])
        right_edge = _sub(positions[node_index + 1], positions[node_index])
        left_length = _norm(left_edge)
        right_length = _norm(right_edge)
        if left_length <= _EPSILON or right_length <= _EPSILON:
            return _ZERO
        average_length = 0.5 * (left_length + right_length)
        return _scale(
            _sub(_scale(right_edge, 1.0 / right_length), _scale(left_edge, 1.0 / left_length)),
            1.0 / average_length,
        )

    def _accumulate_loads(
        self, loads: Iterable[RodExternalLoad]
    ) -> tuple[list[Vec3], list[float]]:
        forces = [_ZERO for _ in self.state.positions_m]
        torques = [0.0 for _ in self._rest_lengths_m]
        for load in loads:
            self._validate_node(load.node_index)
            forces[load.node_index] = _add(forces[load.node_index], load.force_n)
            if load.node_index == 0:
                torques[load.node_index] += load.torque_n_m
            elif load.node_index == len(self.state.positions_m) - 1:
                torques[-1] += load.torque_n_m
            else:
                torques[load.node_index - 1] += 0.5 * load.torque_n_m
                torques[load.node_index] += 0.5 * load.torque_n_m
        return forces, torques

    def _bending_energy_j(
        self,
        positions: Sequence[Vec3],
        node_index: int,
        plastic_curvature_1_m: Vec3,
        stiffness_fraction: float,
    ) -> float:
        """Return one hinge's elastic energy for a fixed internal state."""

        curvature = self._curvature_at(positions, node_index)
        elastic_curvature = _sub(
            _sub(curvature, self._rest_curvature_1_m[node_index - 1]),
            plastic_curvature_1_m,
        )
        integration_length = 0.5 * (
            self._rest_lengths_m[node_index - 1]
            + self._rest_lengths_m[node_index]
        )
        return (
            0.5
            * self.material.bending_stiffness_n_m2
            * stiffness_fraction
            * _dot(elastic_curvature, elastic_curvature)
            * integration_length
        )

    def _bending_force_triplet_n(
        self,
        node_index: int,
        plastic_curvature_1_m: Vec3,
        stiffness_fraction: float,
    ) -> tuple[Vec3, Vec3, Vec3]:
        """Differentiate hinge energy so its nodal force is restoring.

        This backend-neutral reference uses a centered directional derivative
        of the same discrete energy returned by :meth:`_bending_energy_j`.
        Consequently every component has units J/m = N and the assembled force
        is the negative energy gradient.  Production GPU backends may replace
        this with an analytic or automatic derivative of the same potential.
        """

        integration_length = 0.5 * (
            self._rest_lengths_m[node_index - 1]
            + self._rest_lengths_m[node_index]
        )
        perturbation_m = max(1.0e-9, integration_length * 1.0e-6)
        base_positions = list(self.state.positions_m)
        forces: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
        for local_index, affected_node in enumerate(
            (node_index - 1, node_index, node_index + 1)
        ):
            for axis in range(3):
                plus_positions = list(base_positions)
                minus_positions = list(base_positions)
                plus_value = list(plus_positions[affected_node])
                minus_value = list(minus_positions[affected_node])
                plus_value[axis] += perturbation_m
                minus_value[axis] -= perturbation_m
                plus_positions[affected_node] = tuple(plus_value)  # type: ignore[assignment]
                minus_positions[affected_node] = tuple(minus_value)  # type: ignore[assignment]
                positive_energy = self._bending_energy_j(
                    plus_positions,
                    node_index,
                    plastic_curvature_1_m,
                    stiffness_fraction,
                )
                negative_energy = self._bending_energy_j(
                    minus_positions,
                    node_index,
                    plastic_curvature_1_m,
                    stiffness_fraction,
                )
                forces[local_index][axis] = -(
                    positive_energy - negative_energy
                ) / (2.0 * perturbation_m)
        return tuple(tuple(value) for value in forces)  # type: ignore[return-value]

    def advance(
        self,
        dt_s: float,
        loads: Iterable[RodExternalLoad] = (),
    ) -> RodStepResult:
        """Advance one semi-implicit, provenance-bearing mechanics interval."""

        dt = _positive(dt_s, "dt_s")
        if (
            self.state.time_s != self._time_s
            or self.state.physics_step != self._physics_step
        ):
            raise RuntimeError(
                "rod time/step provenance was mutated outside CosseratRod"
            )
        previous_time = self._time_s
        next_time = previous_time + dt
        if not math.isfinite(next_time) or next_time <= previous_time:
            raise ValueError("dt_s does not advance rod time monotonically")
        next_physics_step = self._physics_step + 1
        node_count = len(self.state.positions_m)

        # Kinematic targets are applied before constitutive evaluation.  Their
        # prescribed velocity is either supplied explicitly or inferred from
        # the target displacement over this interval.  This avoids evaluating
        # forces at a stale position and then teleporting the anchor afterward.
        anchor_previous_velocity: dict[int, Vec3] = {}
        anchor_velocity: dict[int, Vec3] = {}
        for node_index, target in self.fixed_nodes.items():
            previous_position = self.state.positions_m[node_index]
            previous_velocity = self.state.velocities_m_s[node_index]
            prescribed_velocity = self.fixed_node_velocities_m_s.get(
                node_index,
                _scale(_sub(target, previous_position), 1.0 / dt),
            )
            anchor_previous_velocity[node_index] = previous_velocity
            anchor_velocity[node_index] = prescribed_velocity
            self.state.positions_m[node_index] = target
            self.state.velocities_m_s[node_index] = prescribed_velocity

        forces, torques = self._accumulate_loads(loads)
        internal = [_ZERO for _ in range(node_count)]
        reaction = [_ZERO for _ in range(node_count)]
        strains = [0.0 for _ in self._rest_lengths_m]
        edge_directions = [_ZERO for _ in self._rest_lengths_m]
        curvatures = [_ZERO for _ in range(max(0, node_count - 2))]
        elastic_curvature_magnitudes = [
            0.0 for _ in range(max(0, node_count - 2))
        ]
        undamaged_edge_energy = [
            0.0 for _ in self._rest_lengths_m
        ]
        dissipated_energy = 0.0

        for edge_index, rest_length in enumerate(self._rest_lengths_m):
            if not self.state.edge_intact[edge_index]:
                continue
            edge = _sub(
                self.state.positions_m[edge_index + 1],
                self.state.positions_m[edge_index],
            )
            length = _norm(edge)
            if length <= max(_EPSILON, rest_length * 1.0e-9):
                raise RuntimeError(
                    f"rod edge {edge_index} collapsed; constitutive direction "
                    "is undefined"
                )
            strain = (length - rest_length) / rest_length
            strains[edge_index] = strain
            edge_directions[edge_index] = _scale(edge, 1.0 / length)
            undamaged_edge_energy[edge_index] += (
                0.5
                * self.material.axial_stiffness_n
                * strain**2
                * rest_length
            )
            undamaged_edge_energy[edge_index] += (
                0.5
                * self.material.torsional_stiffness_n_m2
                * (self.state.twist_rad[edge_index] / rest_length) ** 2
                * rest_length
            )

        for interior_index, node_index in enumerate(range(1, node_count - 1)):
            left_edge = node_index - 1
            right_edge = node_index
            if not (
                self.state.edge_intact[left_edge]
                and self.state.edge_intact[right_edge]
            ):
                continue
            curvature = self._curvature_at(
                self.state.positions_m,
                node_index,
            )
            curvatures[interior_index] = curvature
            plastic = self.state.plastic_curvature_1_m[interior_index]
            elastic_curvature = _sub(
                _sub(curvature, self._rest_curvature_1_m[interior_index]),
                plastic,
            )
            integration_length = 0.5 * (
                self._rest_lengths_m[left_edge]
                + self._rest_lengths_m[right_edge]
            )
            magnitude = _norm(elastic_curvature)
            if magnitude > self.material.yield_curvature_1_m:
                flow_fraction = min(
                    1.0,
                    self.material.plastic_flow_rate_per_s * dt,
                )
                plastic_increment = _scale(
                    elastic_curvature,
                    flow_fraction
                    * (magnitude - self.material.yield_curvature_1_m)
                    / max(magnitude, _EPSILON)
                    * (1.0 - self.material.hardening_fraction),
                )
                plastic = _add(plastic, plastic_increment)
                self.state.plastic_curvature_1_m[interior_index] = plastic
                hinge_fraction = min(
                    1.0 - self.state.edge_damage[left_edge],
                    1.0 - self.state.edge_damage[right_edge],
                )
                dissipated_energy += (
                    self.material.bending_stiffness_n_m2
                    * hinge_fraction
                    * _norm(plastic_increment)
                    * max(magnitude, self.material.yield_curvature_1_m)
                    * integration_length
                )
                elastic_curvature = _sub(
                    _sub(curvature, self._rest_curvature_1_m[interior_index]),
                    plastic,
                )
            magnitude = _norm(elastic_curvature)
            elastic_curvature_magnitudes[interior_index] = magnitude
            hinge_undamaged_energy = (
                0.5
                * self.material.bending_stiffness_n_m2
                * magnitude**2
                * integration_length
            )
            undamaged_edge_energy[left_edge] += (
                0.5 * hinge_undamaged_energy
            )
            undamaged_edge_energy[right_edge] += (
                0.5 * hinge_undamaged_energy
            )

        # Damage is updated before final force assembly, so the returned force,
        # energy, topology, and damage all describe the same constitutive state.
        broken: list[int] = []
        for edge_index, strain in enumerate(strains):
            if not self.state.edge_intact[edge_index]:
                continue
            adjacent_curvature = 0.0
            if edge_index > 0:
                adjacent_curvature = max(
                    adjacent_curvature,
                    elastic_curvature_magnitudes[edge_index - 1],
                )
            if edge_index < len(elastic_curvature_magnitudes):
                adjacent_curvature = max(
                    adjacent_curvature,
                    elastic_curvature_magnitudes[edge_index],
                )
            strain_utilization = max(
                0.0,
                (abs(strain) - self.material.yield_strain)
                / (
                    self.material.failure_strain
                    - self.material.yield_strain
                ),
            )
            curvature_utilization = max(
                0.0,
                (
                    adjacent_curvature
                    - self.material.yield_curvature_1_m
                )
                / (
                    self.material.failure_curvature_1_m
                    - self.material.yield_curvature_1_m
                ),
            )
            previous_damage = self.state.edge_damage[edge_index]
            next_damage = previous_damage
            utilization = max(strain_utilization, curvature_utilization)
            if utilization > 0.0:
                next_damage = min(
                    1.0,
                    previous_damage
                    + self.material.damage_rate_per_s * utilization * dt,
                )
            if (
                abs(strain) >= self.material.failure_strain
                or adjacent_curvature
                >= self.material.failure_curvature_1_m
                or next_damage >= 1.0 - 1.0e-9
            ):
                next_damage = 1.0
            damage_increment = next_damage - previous_damage
            if damage_increment > 0.0:
                dissipated_energy += (
                    damage_increment
                    * undamaged_edge_energy[edge_index]
                )
            self.state.edge_damage[edge_index] = next_damage
            if next_damage >= 1.0 - 1.0e-9:
                self.state.edge_intact[edge_index] = False
                broken.append(edge_index)

        elastic_energy = 0.0
        for edge_index, rest_length in enumerate(self._rest_lengths_m):
            if not self.state.edge_intact[edge_index]:
                continue
            stiffness_fraction = max(
                0.0,
                1.0 - self.state.edge_damage[edge_index],
            )
            axial_force = (
                self.material.axial_stiffness_n
                * strains[edge_index]
                * stiffness_fraction
            )
            force = _scale(
                edge_directions[edge_index],
                axial_force,
            )
            internal[edge_index] = _add(internal[edge_index], force)
            internal[edge_index + 1] = _sub(
                internal[edge_index + 1],
                force,
            )
            elastic_energy += (
                0.5
                * self.material.axial_stiffness_n
                * strains[edge_index] ** 2
                * rest_length
                * stiffness_fraction
            )

        for interior_index, node_index in enumerate(range(1, node_count - 1)):
            left_edge = node_index - 1
            right_edge = node_index
            if not (
                self.state.edge_intact[left_edge]
                and self.state.edge_intact[right_edge]
            ):
                continue
            stiffness_fraction = min(
                1.0 - self.state.edge_damage[left_edge],
                1.0 - self.state.edge_damage[right_edge],
            )
            triplet = self._bending_force_triplet_n(
                node_index,
                self.state.plastic_curvature_1_m[interior_index],
                stiffness_fraction,
            )
            for affected_node, force in zip(
                (node_index - 1, node_index, node_index + 1),
                triplet,
            ):
                internal[affected_node] = _add(
                    internal[affected_node],
                    force,
                )
            elastic_energy += self._bending_energy_j(
                self.state.positions_m,
                node_index,
                self.state.plastic_curvature_1_m[interior_index],
                stiffness_fraction,
            )

        for edge_index, rest_length in enumerate(self._rest_lengths_m):
            if not self.state.edge_intact[edge_index]:
                self.state.angular_velocity_rad_s[edge_index] = 0.0
                continue
            stiffness_fraction = max(
                0.0,
                1.0 - self.state.edge_damage[edge_index],
            )
            restoring_torque = -(
                self.material.torsional_stiffness_n_m2
                * stiffness_fraction
                * self.state.twist_rad[edge_index]
                / rest_length
            )
            polar_mass_moment = (
                self.material.density_kg_m3
                * (
                    self.material.cross_section_area_m2**2
                    / (2.0 * math.pi)
                )
                * rest_length
            )
            angular_acceleration = (
                torques[edge_index] + restoring_torque
            ) / _positive(polar_mass_moment, "edge polar mass moment")
            angular_velocity = (
                self.state.angular_velocity_rad_s[edge_index]
                + angular_acceleration * dt
            )
            angular_velocity *= math.exp(
                -self.material.angular_damping_per_s * dt
            )
            self.state.angular_velocity_rad_s[edge_index] = angular_velocity
            self.state.twist_rad[edge_index] += angular_velocity * dt
            elastic_energy += (
                0.5
                * self.material.torsional_stiffness_n_m2
                * stiffness_fraction
                * (self.state.twist_rad[edge_index] / rest_length) ** 2
                * rest_length
            )

        damping = math.exp(-self.material.linear_damping_per_s * dt)
        for node_index in range(node_count):
            total_force = _add(forces[node_index], internal[node_index])
            total_force = _add(
                total_force,
                _scale(self.gravity_m_s2, self._node_mass_kg[node_index]),
            )
            if node_index in self.fixed_nodes:
                prescribed_velocity = anchor_velocity[node_index]
                prescribed_acceleration = _scale(
                    _sub(
                        prescribed_velocity,
                        anchor_previous_velocity[node_index],
                    ),
                    1.0 / dt,
                )
                reaction[node_index] = _sub(
                    _scale(
                        prescribed_acceleration,
                        self._node_mass_kg[node_index],
                    ),
                    total_force,
                )
                self.state.positions_m[node_index] = self.fixed_nodes[
                    node_index
                ]
                self.state.velocities_m_s[node_index] = prescribed_velocity
                continue
            acceleration = _scale(
                total_force,
                1.0 / self._node_mass_kg[node_index],
            )
            velocity = _add(
                self.state.velocities_m_s[node_index],
                _scale(acceleration, dt),
            )
            velocity = _scale(velocity, damping)
            self.state.velocities_m_s[node_index] = velocity
            self.state.positions_m[node_index] = _add(
                self.state.positions_m[node_index],
                _scale(velocity, dt),
            )
        self.state.time_s = next_time
        self.state.physics_step = next_physics_step
        result = RodStepResult(
            component_id=self.component_id,
            internal_force_n=tuple(internal),
            reaction_force_n=tuple(reaction),
            broken_edge_indices=tuple(broken),
            max_abs_strain=max((abs(value) for value in strains), default=0.0),
            max_curvature_1_m=max(
                (_norm(value) for value in curvatures), default=0.0
            ),
            elastic_energy_j=max(0.0, elastic_energy),
            dissipated_energy_j=max(0.0, dissipated_energy),
            time_s=next_time,
            parameter_status=self.material.parameter_status,
            physics_step=next_physics_step,
            previous_time_s=previous_time,
            dt_s=dt,
        )
        self._time_s = next_time
        self._physics_step = next_physics_step
        return result

    def connected_components(self) -> tuple[tuple[int, ...], ...]:
        """Return node components after any edge failures."""

        components: list[list[int]] = [[0]]
        for edge_index, intact in enumerate(self.state.edge_intact):
            next_node = edge_index + 1
            if intact:
                components[-1].append(next_node)
            else:
                components.append([next_node])
        return tuple(tuple(component) for component in components)

    def apply_fixed_nodes(
        self,
        nodes: Mapping[int, Iterable[float]],
        *,
        velocities_m_s: Mapping[int, Iterable[float]] | None = None,
    ) -> None:
        """Replace kinematic boundary targets and optional target velocities."""

        selected: dict[int, Vec3] = {}
        for index, position in nodes.items():
            self._validate_node(int(index))
            selected[int(index)] = _vec3(position, "fixed node position")
        selected_velocities: dict[int, Vec3] = {}
        if velocities_m_s is not None:
            for index, velocity in velocities_m_s.items():
                self._validate_node(int(index))
                if int(index) not in selected:
                    raise ValueError(
                        "fixed-node velocity requires a matching position "
                        f"for node {int(index)}"
                    )
                selected_velocities[int(index)] = _vec3(
                    velocity,
                    "fixed node velocity",
                )
        self.fixed_nodes = selected
        self.fixed_node_velocities_m_s = selected_velocities


__all__ = [
    "CosseratRod",
    "RodExternalLoad",
    "RodMaterial",
    "RodState",
    "RodStepResult",
]
