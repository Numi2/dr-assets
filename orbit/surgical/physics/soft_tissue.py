# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Layered tetrahedral soft-tissue state and constitutive foundation.

The implementation uses a compressible Neo-Hookean tetrahedral force model
with Kelvin-Voigt damping and irreversible energy-density damage.  It is a
backend-neutral reference foundation: contacts, needles, sutures, and staples
enter as physical nodal loads, while wound gaps and damage remain derived
observations of mutable tissue state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

from .contact import Vec3


Mat3 = tuple[Vec3, Vec3, Vec3]
Tet4 = tuple[int, int, int, int]
Tri3 = tuple[int, int, int]
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


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _unit(value: Vec3, label: str) -> Vec3:
    magnitude = _norm(value)
    if magnitude <= _EPSILON:
        raise ValueError(f"{label} must be non-zero")
    return _scale(value, 1.0 / magnitude)


def _columns(first: Vec3, second: Vec3, third: Vec3) -> Mat3:
    return (
        (first[0], second[0], third[0]),
        (first[1], second[1], third[1]),
        (first[2], second[2], third[2]),
    )


def _column(matrix: Mat3, index: int) -> Vec3:
    return (matrix[0][index], matrix[1][index], matrix[2][index])


def _transpose(matrix: Mat3) -> Mat3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _mat_add(left: Mat3, right: Mat3) -> Mat3:
    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _mat_sub(left: Mat3, right: Mat3) -> Mat3:
    return tuple(
        tuple(left[row][column] - right[row][column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _mat_scale(matrix: Mat3, amount: float) -> Mat3:
    return tuple(
        tuple(matrix[row][column] * amount for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _mat_mul(left: Mat3, right: Mat3) -> Mat3:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _outer(left: Vec3, right: Vec3) -> Mat3:
    return tuple(
        tuple(left[row] * right[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _determinant(matrix: Mat3) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _inverse(matrix: Mat3, label: str) -> Mat3:
    determinant = _determinant(matrix)
    if abs(determinant) <= _EPSILON:
        raise ValueError(f"{label} is singular")
    inverse_determinant = 1.0 / determinant
    return (
        (
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            * inverse_determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2])
            * inverse_determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])
            * inverse_determinant,
        ),
        (
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2])
            * inverse_determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0])
            * inverse_determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2])
            * inverse_determinant,
        ),
        (
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
            * inverse_determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1])
            * inverse_determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
            * inverse_determinant,
        ),
    )


def _frobenius_squared(matrix: Mat3) -> float:
    return sum(
        matrix[row][column] ** 2
        for row in range(3)
        for column in range(3)
    )


def _tet_shape(points: Sequence[Vec3], tet: Tet4) -> Mat3:
    origin = points[tet[0]]
    return _columns(
        _sub(points[tet[1]], origin),
        _sub(points[tet[2]], origin),
        _sub(points[tet[3]], origin),
    )


@dataclass(frozen=True)
class SoftTissueLayer:
    """One layer's effective constitutive parameters in SI units."""

    layer_id: str
    density_kg_m3: float
    shear_modulus_pa: float
    bulk_modulus_pa: float
    viscosity_pa_s: float
    damage_onset_energy_j_m3: float
    failure_energy_j_m3: float
    damage_rate_per_s: float
    fiber_direction: Vec3 = (1.0, 0.0, 0.0)
    fiber_stiffness_pa: float = 0.0
    parameter_status: str = "provisional_engineering_seed"

    def __post_init__(self) -> None:
        if not str(self.layer_id).strip():
            raise ValueError("layer_id must be non-empty")
        _positive(self.density_kg_m3, "density_kg_m3")
        shear = _positive(self.shear_modulus_pa, "shear_modulus_pa")
        bulk = _positive(self.bulk_modulus_pa, "bulk_modulus_pa")
        if bulk <= (2.0 / 3.0) * shear:
            raise ValueError("bulk_modulus_pa is too small for the shear modulus")
        _non_negative(self.viscosity_pa_s, "viscosity_pa_s")
        onset = _non_negative(
            self.damage_onset_energy_j_m3, "damage_onset_energy_j_m3"
        )
        failure = _positive(self.failure_energy_j_m3, "failure_energy_j_m3")
        if failure <= onset:
            raise ValueError(
                "failure_energy_j_m3 must exceed damage_onset_energy_j_m3"
            )
        _non_negative(self.damage_rate_per_s, "damage_rate_per_s")
        object.__setattr__(
            self,
            "fiber_direction",
            _unit(_vec3(self.fiber_direction, "fiber_direction"), "fiber_direction"),
        )
        _non_negative(self.fiber_stiffness_pa, "fiber_stiffness_pa")
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")

    @property
    def lame_lambda_pa(self) -> float:
        return self.bulk_modulus_pa - (2.0 / 3.0) * self.shear_modulus_pa


@dataclass(frozen=True)
class SoftTissueMesh:
    """Reference tetrahedral topology and layer assignment."""

    rest_positions_m: tuple[Vec3, ...]
    tetrahedra: tuple[Tet4, ...]
    tet_layer_ids: tuple[str, ...]
    boundary_faces: tuple[Tri3, ...] = ()

    def __post_init__(self) -> None:
        positions = tuple(
            _vec3(position, "rest_positions_m") for position in self.rest_positions_m
        )
        if len(positions) < 4:
            raise ValueError("soft tissue mesh requires at least four nodes")
        object.__setattr__(self, "rest_positions_m", positions)
        tetrahedra = tuple(
            tuple(int(index) for index in tet) for tet in self.tetrahedra
        )
        if not tetrahedra:
            raise ValueError("soft tissue mesh requires tetrahedra")
        for tet_index, tet in enumerate(tetrahedra):
            if len(tet) != 4 or len(set(tet)) != 4:
                raise ValueError(f"tetrahedron {tet_index} must have four unique nodes")
            if any(index < 0 or index >= len(positions) for index in tet):
                raise ValueError(f"tetrahedron {tet_index} has an invalid node index")
            determinant = _determinant(_tet_shape(positions, tet))
            if abs(determinant) <= _EPSILON:
                raise ValueError(f"tetrahedron {tet_index} has zero reference volume")
        object.__setattr__(self, "tetrahedra", tetrahedra)
        layers = tuple(str(value) for value in self.tet_layer_ids)
        if len(layers) != len(tetrahedra) or any(not value for value in layers):
            raise ValueError("tet_layer_ids must name every tetrahedron")
        object.__setattr__(self, "tet_layer_ids", layers)
        faces = tuple(tuple(int(index) for index in face) for face in self.boundary_faces)
        for face_index, face in enumerate(faces):
            if len(face) != 3 or len(set(face)) != 3:
                raise ValueError(f"boundary face {face_index} must have three nodes")
            if any(index < 0 or index >= len(positions) for index in face):
                raise ValueError(f"boundary face {face_index} has an invalid node")
        object.__setattr__(self, "boundary_faces", faces)


@dataclass(frozen=True)
class SoftTissueExternalLoad:
    node_index: int
    force_n: Vec3
    source_component_id: str

    def __post_init__(self) -> None:
        if int(self.node_index) < 0:
            raise ValueError("node_index must be non-negative")
        object.__setattr__(self, "node_index", int(self.node_index))
        object.__setattr__(self, "force_n", _vec3(self.force_n, "force_n"))
        if not str(self.source_component_id).strip():
            raise ValueError("source_component_id must be non-empty")


@dataclass
class SoftTissueState:
    """Mutable nodal and tetrahedral state."""

    positions_m: list[Vec3]
    velocities_m_s: list[Vec3]
    tet_damage_fraction: list[float]
    time_s: float = 0.0
    physics_step: int = 0

    def __post_init__(self) -> None:
        self.positions_m = [
            _vec3(position, "positions_m") for position in self.positions_m
        ]
        self.velocities_m_s = [
            _vec3(velocity, "velocities_m_s") for velocity in self.velocities_m_s
        ]
        if len(self.positions_m) != len(self.velocities_m_s):
            raise ValueError("positions_m and velocities_m_s must have equal length")
        self.tet_damage_fraction = [
            _fraction(value, "tet_damage_fraction")
            for value in self.tet_damage_fraction
        ]
        self.time_s = _non_negative(self.time_s, "time_s")
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step < 0:
            raise ValueError("physics_step must be a non-negative integer")
        self.physics_step = physics_step

    @classmethod
    def from_mesh(cls, mesh: SoftTissueMesh) -> "SoftTissueState":
        return cls(
            positions_m=list(mesh.rest_positions_m),
            velocities_m_s=[_ZERO for _ in mesh.rest_positions_m],
            tet_damage_fraction=[0.0 for _ in mesh.tetrahedra],
        )


@dataclass
class PunctureTract:
    """Persistent embedded tract created by a causal puncture interaction."""

    tract_id: str
    source_component_id: str
    entry_position_m: Vec3
    exit_position_m: Vec3
    radius_m: float
    layer_path: tuple[str, ...]
    damage_fraction: float = 0.0
    captured_filament_ids: set[str] = field(default_factory=set)
    affected_tet_indices: tuple[int, ...] = ()
    registered_physics_step: int | None = None
    registered_time_s: float | None = None

    def __post_init__(self) -> None:
        if not str(self.tract_id).strip():
            raise ValueError("tract_id must be non-empty")
        if not str(self.source_component_id).strip():
            raise ValueError("source_component_id must be non-empty")
        self.entry_position_m = _vec3(self.entry_position_m, "entry_position_m")
        self.exit_position_m = _vec3(self.exit_position_m, "exit_position_m")
        self.radius_m = _positive(self.radius_m, "radius_m")
        self.layer_path = tuple(str(value) for value in self.layer_path)
        if not self.layer_path or any(not value for value in self.layer_path):
            raise ValueError("layer_path must contain non-empty layer IDs")
        self.damage_fraction = _fraction(self.damage_fraction, "damage_fraction")
        self.captured_filament_ids = {
            str(value) for value in self.captured_filament_ids if str(value)
        }
        supplied_tets = tuple(int(value) for value in self.affected_tet_indices)
        if len(set(supplied_tets)) != len(supplied_tets):
            raise ValueError("affected_tet_indices cannot contain duplicates")
        affected_tets = tuple(sorted(supplied_tets))
        if any(value < 0 for value in affected_tets):
            raise ValueError("affected_tet_indices must be non-negative")
        self.affected_tet_indices = affected_tets
        if (
            self.registered_physics_step is None
        ) != (self.registered_time_s is None):
            raise ValueError(
                "puncture registration step and time must be supplied together"
            )
        if self.registered_physics_step is not None:
            registered_step = int(self.registered_physics_step)
            if (
                registered_step != self.registered_physics_step
                or registered_step < 0
            ):
                raise ValueError(
                    "registered_physics_step must be a non-negative integer"
                )
            self.registered_physics_step = registered_step
            self.registered_time_s = _non_negative(
                self.registered_time_s,
                "registered_time_s",
            )
        if self.length_m <= _EPSILON:
            raise ValueError("puncture tract entry and exit must be distinct")

    @property
    def length_m(self) -> float:
        return _norm(_sub(self.exit_position_m, self.entry_position_m))


@dataclass(frozen=True)
class SoftTissueStepResult:
    scene_component_id: str
    internal_force_n: tuple[Vec3, ...]
    reaction_force_n: tuple[Vec3, ...]
    max_energy_density_j_m3: float
    max_damage_fraction: float
    dissipated_energy_j: float
    time_s: float
    parameter_status_by_layer: tuple[tuple[str, str], ...]
    physics_step: int
    previous_time_s: float
    dt_s: float

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        object.__setattr__(
            self,
            "internal_force_n",
            tuple(_vec3(value, "internal_force_n") for value in self.internal_force_n),
        )
        object.__setattr__(
            self,
            "reaction_force_n",
            tuple(_vec3(value, "reaction_force_n") for value in self.reaction_force_n),
        )
        if len(self.internal_force_n) != len(self.reaction_force_n):
            raise ValueError("internal and reaction force arrays must match")
        object.__setattr__(
            self,
            "max_energy_density_j_m3",
            _non_negative(
                self.max_energy_density_j_m3,
                "max_energy_density_j_m3",
            ),
        )
        object.__setattr__(
            self,
            "max_damage_fraction",
            _fraction(self.max_damage_fraction, "max_damage_fraction"),
        )
        object.__setattr__(
            self,
            "dissipated_energy_j",
            _non_negative(self.dissipated_energy_j, "dissipated_energy_j"),
        )
        statuses = tuple(
            (str(layer_id), str(status))
            for layer_id, status in self.parameter_status_by_layer
        )
        if (
            statuses != tuple(sorted(statuses))
            or len({layer_id for layer_id, _ in statuses}) != len(statuses)
            or any(not layer_id or not status for layer_id, status in statuses)
        ):
            raise ValueError(
                "parameter_status_by_layer must contain sorted unique "
                "non-empty layer entries"
            )
        object.__setattr__(self, "parameter_status_by_layer", statuses)
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
                "step interval"
            )
        object.__setattr__(self, "previous_time_s", previous_time)
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "dt_s", dt)


@dataclass(frozen=True)
class WoundObservation:
    """Geometry and damage derived from two wound-edge node sets."""

    scene_component_id: str
    wound_id: str
    mean_gap_m: float
    maximum_gap_m: float
    edge_height_mismatch_m: float
    damaged_reference_area_m2: float
    active_puncture_tract_count: int
    mean_damage_fraction: float
    time_s: float
    physics_step: int
    localized_tet_indices: tuple[int, ...]
    active_puncture_tract_ids: tuple[str, ...]
    unlocalized_puncture_tract_count: int

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        if not str(self.wound_id).strip():
            raise ValueError("wound_id must be non-empty")
        for field_name in (
            "mean_gap_m",
            "maximum_gap_m",
            "edge_height_mismatch_m",
            "damaged_reference_area_m2",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_negative(getattr(self, field_name), field_name),
            )
        if self.maximum_gap_m < self.mean_gap_m:
            raise ValueError("maximum_gap_m cannot be below mean_gap_m")
        object.__setattr__(
            self,
            "mean_damage_fraction",
            _fraction(self.mean_damage_fraction, "mean_damage_fraction"),
        )
        object.__setattr__(self, "time_s", _non_negative(self.time_s, "time_s"))
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step < 0:
            raise ValueError("physics_step must be a non-negative integer")
        object.__setattr__(self, "physics_step", physics_step)
        localized_tets = tuple(int(value) for value in self.localized_tet_indices)
        if (
            localized_tets != tuple(sorted(set(localized_tets)))
            or any(value < 0 for value in localized_tets)
        ):
            raise ValueError(
                "localized_tet_indices must be sorted, unique, and non-negative"
            )
        object.__setattr__(self, "localized_tet_indices", localized_tets)
        tract_ids = tuple(str(value) for value in self.active_puncture_tract_ids)
        if (
            tract_ids != tuple(sorted(set(tract_ids)))
            or any(not value for value in tract_ids)
        ):
            raise ValueError(
                "active_puncture_tract_ids must be sorted, unique, and non-empty"
            )
        object.__setattr__(self, "active_puncture_tract_ids", tract_ids)
        active_count = int(self.active_puncture_tract_count)
        if (
            active_count != self.active_puncture_tract_count
            or active_count != len(tract_ids)
        ):
            raise ValueError(
                "active_puncture_tract_count must match "
                "active_puncture_tract_ids"
            )
        object.__setattr__(self, "active_puncture_tract_count", active_count)
        unlocalized_count = int(self.unlocalized_puncture_tract_count)
        if (
            unlocalized_count != self.unlocalized_puncture_tract_count
            or unlocalized_count < 0
        ):
            raise ValueError(
                "unlocalized_puncture_tract_count must be a non-negative integer"
            )
        object.__setattr__(
            self,
            "unlocalized_puncture_tract_count",
            unlocalized_count,
        )


@dataclass
class LayeredSoftTissue:
    """Mutable layered tetrahedral mechanics component."""

    scene_component_id: str
    mesh: SoftTissueMesh
    layers: Mapping[str, SoftTissueLayer]
    state: SoftTissueState | None = None
    fixed_nodes: dict[int, Vec3] = field(default_factory=dict)
    gravity_m_s2: Vec3 = (0.0, 0.0, -9.81)
    puncture_tracts: dict[str, PunctureTract] = field(default_factory=dict)
    fixed_node_velocities_m_s: dict[int, Vec3] = field(default_factory=dict)
    _rest_inverse: tuple[Mat3, ...] = field(init=False, repr=False)
    _rest_volume_m3: tuple[float, ...] = field(init=False, repr=False)
    _node_mass_kg: tuple[float, ...] = field(init=False, repr=False)
    _time_s: float = field(init=False, repr=False)
    _physics_step: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        self.layers = {str(key): value for key, value in self.layers.items()}
        missing = sorted(set(self.mesh.tet_layer_ids) - self.layers.keys())
        if missing:
            raise ValueError(f"missing tissue layer definitions: {missing}")
        if self.state is None:
            self.state = SoftTissueState.from_mesh(self.mesh)
        if len(self.state.positions_m) != len(self.mesh.rest_positions_m):
            raise ValueError("soft tissue state node count does not match mesh")
        if len(self.state.tet_damage_fraction) != len(self.mesh.tetrahedra):
            raise ValueError("soft tissue state damage count does not match mesh")
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
        self._time_s = self.state.time_s
        self._physics_step = self.state.physics_step
        rest_shapes = tuple(
            _tet_shape(self.mesh.rest_positions_m, tet)
            for tet in self.mesh.tetrahedra
        )
        self._rest_inverse = tuple(
            _inverse(shape, f"reference tetrahedron {index}")
            for index, shape in enumerate(rest_shapes)
        )
        self._rest_volume_m3 = tuple(
            abs(_determinant(shape)) / 6.0 for shape in rest_shapes
        )
        masses = [0.0 for _ in self.mesh.rest_positions_m]
        for tet_index, tet in enumerate(self.mesh.tetrahedra):
            layer = self.layers[self.mesh.tet_layer_ids[tet_index]]
            nodal_mass = (
                layer.density_kg_m3 * self._rest_volume_m3[tet_index] / 4.0
            )
            for node_index in tet:
                masses[node_index] += nodal_mass
        self._node_mass_kg = tuple(
            _positive(value, "soft tissue node mass") for value in masses
        )
        initial_tracts = dict(self.puncture_tracts)
        self.puncture_tracts = {}
        for tract_id, tract in initial_tracts.items():
            if str(tract_id) != tract.tract_id:
                raise ValueError(
                    "puncture_tracts keys must match each tract's tract_id"
                )
            self.register_puncture_tract(tract)

    def _validate_node(self, node_index: int) -> None:
        if not 0 <= int(node_index) < len(self.mesh.rest_positions_m):
            raise IndexError(f"soft tissue node index outside range: {node_index}")

    def register_puncture_tract(self, tract: PunctureTract) -> None:
        """Register a tract emitted by a needle-tissue interaction component."""

        if tract.tract_id in self.puncture_tracts:
            raise ValueError(f"duplicate puncture tract {tract.tract_id!r}")
        if any(layer not in self.layers for layer in tract.layer_path):
            raise ValueError("puncture tract references an unknown tissue layer")
        for tet_index in tract.affected_tet_indices:
            if not 0 <= tet_index < len(self.mesh.tetrahedra):
                raise IndexError(
                    f"puncture tract {tract.tract_id!r} references invalid "
                    f"tetrahedron {tet_index}"
                )
        if tract.affected_tet_indices:
            affected_layers = {
                self.mesh.tet_layer_ids[tet_index]
                for tet_index in tract.affected_tet_indices
            }
            if not affected_layers.issubset(set(tract.layer_path)):
                raise ValueError(
                    "puncture tract layer_path must include every layer bound "
                    "by affected_tet_indices"
                )
        state = self.state
        if state is None:
            raise RuntimeError("soft tissue state was not initialized")
        if tract.registered_physics_step is None:
            tract.registered_physics_step = state.physics_step
            tract.registered_time_s = state.time_s
        elif (
            tract.registered_physics_step != state.physics_step
            or tract.registered_time_s != state.time_s
        ):
            raise ValueError(
                "pre-stamped puncture registration provenance does not match "
                "the receiving soft-tissue state"
            )
        self.puncture_tracts[tract.tract_id] = tract

    def advance(
        self,
        dt_s: float,
        loads: Iterable[SoftTissueExternalLoad] = (),
    ) -> SoftTissueStepResult:
        """Advance one semi-implicit tetrahedral mechanics step."""

        dt = _positive(dt_s, "dt_s")
        state = self.state
        if state is None:
            raise RuntimeError("soft tissue state was not initialized")
        if (
            state.time_s != self._time_s
            or state.physics_step != self._physics_step
        ):
            raise RuntimeError(
                "soft-tissue time/step provenance was mutated outside "
                "LayeredSoftTissue"
            )
        previous_time = self._time_s
        next_time = previous_time + dt
        if not math.isfinite(next_time) or next_time <= previous_time:
            raise ValueError("dt_s does not advance soft-tissue time monotonically")
        next_physics_step = self._physics_step + 1
        node_count = len(state.positions_m)
        positions = list(state.positions_m)
        velocities = list(state.velocities_m_s)
        next_damage = list(state.tet_damage_fraction)
        previous_anchor_velocities: dict[int, Vec3] = {}
        prescribed_anchor_velocities: dict[int, Vec3] = {}
        for node_index, target_position in self.fixed_nodes.items():
            previous_position = positions[node_index]
            previous_velocity = velocities[node_index]
            prescribed_velocity = self.fixed_node_velocities_m_s.get(
                node_index,
                _scale(
                    _sub(target_position, previous_position),
                    1.0 / dt,
                ),
            )
            previous_anchor_velocities[node_index] = previous_velocity
            prescribed_anchor_velocities[node_index] = prescribed_velocity
            # Boundary kinematics must participate in this step's strain and
            # strain-rate evaluation, rather than being teleported afterward.
            positions[node_index] = target_position
            velocities[node_index] = prescribed_velocity

        external = [_ZERO for _ in range(node_count)]
        for load in loads:
            self._validate_node(load.node_index)
            external[load.node_index] = _add(
                external[load.node_index], load.force_n
            )
        internal = [_ZERO for _ in range(node_count)]
        reaction = [_ZERO for _ in range(node_count)]
        max_energy_density = 0.0
        dissipated_energy = 0.0

        for tet_index, tet in enumerate(self.mesh.tetrahedra):
            layer = self.layers[self.mesh.tet_layer_ids[tet_index]]
            previous_damage = state.tet_damage_fraction[tet_index]
            if previous_damage >= 1.0:
                # A fully failed element carries neither elastic nor viscous
                # load.  Its nodes remain in the observation mesh, but it no
                # longer blocks the mechanics step merely by inverting.
                next_damage[tet_index] = 1.0
                continue
            current_shape = _tet_shape(positions, tet)
            velocity_shape = _tet_shape(velocities, tet)
            deformation = _mat_mul(current_shape, self._rest_inverse[tet_index])
            deformation_rate = _mat_mul(
                velocity_shape, self._rest_inverse[tet_index]
            )
            jacobian = _determinant(deformation)
            if jacobian <= 0.0:
                raise RuntimeError(
                    f"soft tissue tetrahedron {tet_index} inverted; "
                    "the step cannot produce physically meaningful stress"
                )
            safe_jacobian = max(jacobian, 0.05)
            inverse_deformation = _inverse(
                deformation, f"current tetrahedron {tet_index}"
            )
            inverse_transpose = _transpose(inverse_deformation)
            log_jacobian = math.log(safe_jacobian)
            elastic_piola = _mat_add(
                _mat_scale(
                    _mat_sub(deformation, inverse_transpose),
                    layer.shear_modulus_pa,
                ),
                _mat_scale(
                    inverse_transpose,
                    layer.lame_lambda_pa * log_jacobian,
                ),
            )
            deformed_fiber = tuple(
                _dot(deformation[row], layer.fiber_direction)
                for row in range(3)
            )
            fiber_stretch = _norm(deformed_fiber)
            if layer.fiber_stiffness_pa > 0.0 and fiber_stretch > 1.0:
                elastic_piola = _mat_add(
                    elastic_piola,
                    _mat_scale(
                        _outer(deformed_fiber, layer.fiber_direction),
                        layer.fiber_stiffness_pa
                        * (fiber_stretch - 1.0)
                        / max(fiber_stretch, _EPSILON),
                    ),
                )
            if jacobian < 0.05:
                inversion_penalty = layer.bulk_modulus_pa * (0.05 - jacobian)
                elastic_piola = _mat_add(
                    elastic_piola,
                    _mat_scale(inverse_transpose, -inversion_penalty),
                )

            first_invariant = _frobenius_squared(deformation)
            energy_density = max(
                0.0,
                0.5
                * layer.shear_modulus_pa
                * (first_invariant - 3.0)
                - layer.shear_modulus_pa * log_jacobian
                + 0.5 * layer.lame_lambda_pa * log_jacobian**2,
            )
            if layer.fiber_stiffness_pa > 0.0 and fiber_stretch > 1.0:
                energy_density += (
                    0.5
                    * layer.fiber_stiffness_pa
                    * (fiber_stretch - 1.0) ** 2
                )
            max_energy_density = max(max_energy_density, energy_density)
            updated_damage = previous_damage
            if energy_density > layer.damage_onset_energy_j_m3:
                utilization = (
                    energy_density - layer.damage_onset_energy_j_m3
                ) / (
                    layer.failure_energy_j_m3
                    - layer.damage_onset_energy_j_m3
                )
                updated_damage = min(
                    1.0,
                    previous_damage
                    + layer.damage_rate_per_s
                    * max(0.0, utilization)
                    * dt,
                )
                if energy_density >= layer.failure_energy_j_m3:
                    updated_damage = 1.0
                dissipated_energy += (
                    updated_damage - previous_damage
                ) * energy_density * self._rest_volume_m3[tet_index]
            next_damage[tet_index] = updated_damage

            # Kelvin-Voigt damping must be objective: a rigid-body rotation
            # has a skew velocity gradient and therefore zero strain rate.
            # Damping F-dot directly invents stress and energy loss during
            # rigid motion.
            velocity_gradient = _mat_mul(
                deformation_rate,
                inverse_deformation,
            )
            rate_of_deformation = _mat_scale(
                _mat_add(velocity_gradient, _transpose(velocity_gradient)),
                0.5,
            )
            viscous_cauchy = _mat_scale(
                rate_of_deformation,
                2.0 * layer.viscosity_pa_s,
            )
            viscous_piola = _mat_scale(
                _mat_mul(viscous_cauchy, inverse_transpose),
                safe_jacobian,
            )
            # Use the damage state produced by this constitutive evaluation
            # for both elastic and viscous force.  A newly failed tetrahedron
            # therefore cannot return a pre-failure force for the same step.
            intact_fraction = max(0.0, 1.0 - updated_damage)
            piola = _mat_scale(
                _mat_add(elastic_piola, viscous_piola),
                intact_fraction,
            )
            force_matrix = _mat_scale(
                _mat_mul(piola, _transpose(self._rest_inverse[tet_index])),
                -self._rest_volume_m3[tet_index],
            )
            force_one = _column(force_matrix, 0)
            force_two = _column(force_matrix, 1)
            force_three = _column(force_matrix, 2)
            force_zero = _scale(
                _add(_add(force_one, force_two), force_three), -1.0
            )
            for node_index, force in zip(
                tet, (force_zero, force_one, force_two, force_three)
            ):
                internal[node_index] = _add(internal[node_index], force)

            viscous_power_density = (
                2.0
                * layer.viscosity_pa_s
                * _frobenius_squared(rate_of_deformation)
                * intact_fraction
            )
            dissipated_energy += (
                viscous_power_density
                * safe_jacobian
                * self._rest_volume_m3[tet_index]
                * dt
            )

        integrated_positions = list(positions)
        integrated_velocities = list(velocities)
        for node_index in range(node_count):
            total_force = _add(internal[node_index], external[node_index])
            total_force = _add(
                total_force,
                _scale(self.gravity_m_s2, self._node_mass_kg[node_index]),
            )
            if node_index in self.fixed_nodes:
                prescribed_acceleration = _scale(
                    _sub(
                        prescribed_anchor_velocities[node_index],
                        previous_anchor_velocities[node_index],
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
                integrated_positions[node_index] = self.fixed_nodes[node_index]
                integrated_velocities[node_index] = (
                    prescribed_anchor_velocities[node_index]
                )
                continue
            acceleration = _scale(
                total_force, 1.0 / self._node_mass_kg[node_index]
            )
            velocity = _add(
                velocities[node_index],
                _scale(acceleration, dt),
            )
            integrated_velocities[node_index] = velocity
            integrated_positions[node_index] = _add(
                positions[node_index],
                _scale(velocity, dt),
            )
        result = SoftTissueStepResult(
            scene_component_id=self.scene_component_id,
            internal_force_n=tuple(internal),
            reaction_force_n=tuple(reaction),
            max_energy_density_j_m3=max_energy_density,
            max_damage_fraction=max(next_damage, default=0.0),
            dissipated_energy_j=max(0.0, dissipated_energy),
            time_s=next_time,
            parameter_status_by_layer=tuple(
                sorted(
                    (layer_id, layer.parameter_status)
                    for layer_id, layer in self.layers.items()
                )
            ),
            physics_step=next_physics_step,
            previous_time_s=previous_time,
            dt_s=dt,
        )
        # Commit only after all constitutive calculations and immutable result
        # validation succeed.  A failed advance cannot leave partial damage,
        # anchor, integration, or clock mutation behind.
        state.positions_m = integrated_positions
        state.velocities_m_s = integrated_velocities
        state.tet_damage_fraction = next_damage
        state.time_s = next_time
        state.physics_step = next_physics_step
        self._time_s = next_time
        self._physics_step = next_physics_step
        return result

    def observe_wound(
        self,
        *,
        wound_id: str,
        left_edge_nodes: Sequence[int],
        right_edge_nodes: Sequence[int],
        height_axis: Iterable[float] = (0.0, 0.0, 1.0),
        affected_tet_indices: Sequence[int] | None = None,
    ) -> WoundObservation:
        """Derive geometry and damage in a wound-local tetrahedral region."""

        if not str(wound_id).strip():
            raise ValueError("wound_id must be non-empty")
        if not left_edge_nodes or not right_edge_nodes:
            raise ValueError("both wound edges require at least one node")
        left_nodes = tuple(int(value) for value in left_edge_nodes)
        right_nodes = tuple(int(value) for value in right_edge_nodes)
        if (
            len(set(left_nodes)) != len(left_nodes)
            or len(set(right_nodes)) != len(right_nodes)
        ):
            raise ValueError("wound edge node lists cannot contain duplicates")
        if set(left_nodes) & set(right_nodes):
            raise ValueError("left and right wound edges must be disjoint")
        for node_index in (*left_nodes, *right_nodes):
            self._validate_node(node_index)
        wound_nodes = set(left_nodes) | set(right_nodes)
        if affected_tet_indices is None:
            localized_tets = tuple(
                tet_index
                for tet_index, tet in enumerate(self.mesh.tetrahedra)
                if wound_nodes & set(tet)
            )
        else:
            supplied_tets = tuple(int(value) for value in affected_tet_indices)
            if len(set(supplied_tets)) != len(supplied_tets):
                raise ValueError(
                    "affected_tet_indices cannot contain duplicates"
                )
            localized_tets = tuple(sorted(supplied_tets))
        if not localized_tets:
            raise ValueError(
                "the wound edges must localize at least one tetrahedron"
            )
        for tet_index in localized_tets:
            if not 0 <= tet_index < len(self.mesh.tetrahedra):
                raise IndexError(
                    f"wound references invalid tetrahedron {tet_index}"
                )
            if not wound_nodes & set(self.mesh.tetrahedra[tet_index]):
                raise ValueError(
                    "every localized wound tetrahedron must touch a wound-edge "
                    "node"
                )
        localized_tet_set = set(localized_tets)
        axis = _unit(_vec3(height_axis, "height_axis"), "height_axis")
        state = self.state
        if state is None:
            raise RuntimeError("soft tissue state was not initialized")
        gaps: list[float] = []
        height_mismatch: list[float] = []

        def planar_distance(left: Vec3, right: Vec3) -> float:
            offset = _sub(right, left)
            return _norm(
                _sub(
                    offset,
                    _scale(axis, _dot(offset, axis)),
                )
            )

        # Use both directions so the result is not biased by which edge has
        # denser sampling.
        for source_nodes, target_nodes in (
            (left_nodes, right_nodes),
            (right_nodes, left_nodes),
        ):
            for source_index in source_nodes:
                source = state.positions_m[int(source_index)]
                nearest_index = min(
                    target_nodes,
                    key=lambda index: planar_distance(
                        source,
                        state.positions_m[int(index)],
                    ),
                )
                offset = _sub(state.positions_m[int(nearest_index)], source)
                signed_height_offset = _dot(offset, axis)
                planar_offset = _sub(
                    offset,
                    _scale(axis, signed_height_offset),
                )
                gaps.append(_norm(planar_offset))
                height_mismatch.append(abs(signed_height_offset))

        damaged_area = 0.0
        for face in self.mesh.boundary_faces:
            incident_tets = [
                tet_index
                for tet_index, tet in enumerate(self.mesh.tetrahedra)
                if set(face).issubset(tet)
            ]
            if not incident_tets:
                raise RuntimeError(
                    f"boundary face {face!r} is not incident to a tetrahedron"
                )
            localized_incident_tets = [
                tet_index
                for tet_index in incident_tets
                if tet_index in localized_tet_set
            ]
            if (
                not localized_incident_tets
                or not wound_nodes.intersection(face)
            ):
                continue
            first, second, third = (
                self.mesh.rest_positions_m[index] for index in face
            )
            face_area = 0.5 * _norm(
                _cross(_sub(second, first), _sub(third, first))
            )
            damaged_area += face_area * max(
                state.tet_damage_fraction[tet_index]
                for tet_index in localized_incident_tets
            )
        active_tract_ids = tuple(
            sorted(
                tract.tract_id
                for tract in self.puncture_tracts.values()
                if set(tract.affected_tet_indices) & localized_tet_set
            )
        )
        unlocalized_tract_count = sum(
            not tract.affected_tet_indices
            for tract in self.puncture_tracts.values()
        )
        return WoundObservation(
            scene_component_id=self.scene_component_id,
            wound_id=str(wound_id),
            mean_gap_m=sum(gaps) / len(gaps),
            maximum_gap_m=max(gaps),
            edge_height_mismatch_m=sum(height_mismatch) / len(height_mismatch),
            damaged_reference_area_m2=damaged_area,
            active_puncture_tract_count=len(active_tract_ids),
            mean_damage_fraction=(
                sum(
                    state.tet_damage_fraction[tet_index]
                    for tet_index in localized_tets
                )
                / len(localized_tets)
            ),
            time_s=state.time_s,
            physics_step=state.physics_step,
            localized_tet_indices=localized_tets,
            active_puncture_tract_ids=active_tract_ids,
            unlocalized_puncture_tract_count=unlocalized_tract_count,
        )

    def apply_fixed_nodes(
        self,
        nodes: Mapping[int, Iterable[float]],
        *,
        velocities_m_s: Mapping[int, Iterable[float]] | None = None,
    ) -> None:
        """Replace explicit physical boundary anchors, not closure outcomes."""

        selected: dict[int, Vec3] = {}
        for index, position in nodes.items():
            self._validate_node(int(index))
            selected[int(index)] = _vec3(position, "fixed node position")
        selected_velocities: dict[int, Vec3] = {}
        for index, velocity in (velocities_m_s or {}).items():
            node_index = int(index)
            if node_index not in selected:
                raise ValueError(
                    "fixed-node velocities require matching fixed-node "
                    "positions"
                )
            selected_velocities[node_index] = _vec3(
                velocity,
                "fixed node velocity",
            )
        self.fixed_nodes = selected
        self.fixed_node_velocities_m_s = selected_velocities


__all__ = [
    "LayeredSoftTissue",
    "Mat3",
    "PunctureTract",
    "SoftTissueExternalLoad",
    "SoftTissueLayer",
    "SoftTissueMesh",
    "SoftTissueState",
    "SoftTissueStepResult",
    "Tet4",
    "Tri3",
    "WoundObservation",
]
