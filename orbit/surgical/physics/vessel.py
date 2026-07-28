# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Reduced-order compliant-vessel wall and one-dimensional flow mechanics.

Boundary pressures and physical contact loads are inputs.  Lumen area,
occlusion, leakage, downstream flow, and defect growth are observations of the
mutable vessel state; callers cannot set those outcomes through this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


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


@dataclass(frozen=True)
class VesselMaterial:
    """Effective wall and fluid parameters for a vessel segment."""

    area_compliance_per_pa: float
    radial_relaxation_time_s: float
    minimum_area_fraction: float
    maximum_area_fraction: float
    damage_pressure_pa: float
    rupture_pressure_pa: float
    damage_rate_per_s: float
    defect_growth_rate_m2_s: float
    blood_dynamic_viscosity_pa_s: float
    blood_density_kg_m3: float
    leak_discharge_coefficient: float = 0.62
    parameter_status: str = "provisional_engineering_seed"

    def __post_init__(self) -> None:
        _non_negative(self.area_compliance_per_pa, "area_compliance_per_pa")
        _positive(self.radial_relaxation_time_s, "radial_relaxation_time_s")
        minimum = _positive(
            self.minimum_area_fraction, "minimum_area_fraction"
        )
        maximum = _positive(
            self.maximum_area_fraction, "maximum_area_fraction"
        )
        if minimum >= 1.0:
            raise ValueError("minimum_area_fraction must be below one")
        if maximum <= 1.0 or maximum <= minimum:
            raise ValueError(
                "maximum_area_fraction must exceed one and minimum_area_fraction"
            )
        damage_pressure = _positive(self.damage_pressure_pa, "damage_pressure_pa")
        rupture_pressure = _positive(
            self.rupture_pressure_pa, "rupture_pressure_pa"
        )
        if rupture_pressure <= damage_pressure:
            raise ValueError("rupture_pressure_pa must exceed damage_pressure_pa")
        _non_negative(self.damage_rate_per_s, "damage_rate_per_s")
        _non_negative(self.defect_growth_rate_m2_s, "defect_growth_rate_m2_s")
        _positive(
            self.blood_dynamic_viscosity_pa_s,
            "blood_dynamic_viscosity_pa_s",
        )
        _positive(self.blood_density_kg_m3, "blood_density_kg_m3")
        discharge = _finite(
            self.leak_discharge_coefficient, "leak_discharge_coefficient"
        )
        if not 0.0 < discharge <= 1.0:
            raise ValueError("leak_discharge_coefficient must be in (0, 1]")
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")


@dataclass(frozen=True)
class VesselContactLoad:
    """Physical compression applied to a named vessel segment."""

    segment_id: str
    source_component_id: str
    normal_force_n: float
    contact_area_m2: float

    def __post_init__(self) -> None:
        if not str(self.segment_id).strip():
            raise ValueError("segment_id must be non-empty")
        if not str(self.source_component_id).strip():
            raise ValueError("source_component_id must be non-empty")
        object.__setattr__(
            self, "normal_force_n", _non_negative(self.normal_force_n, "normal_force_n")
        )
        object.__setattr__(
            self, "contact_area_m2", _positive(self.contact_area_m2, "contact_area_m2")
        )

    @property
    def pressure_pa(self) -> float:
        return self.normal_force_n / self.contact_area_m2


@dataclass
class VesselSegmentState:
    """Mutable wall and defect state for one serial flow segment."""

    segment_id: str
    length_m: float
    reference_radius_m: float
    lumen_area_m2: float | None = None
    wall_damage_fraction: float = 0.0
    residual_defect_area_m2: float = 0.0
    time_s: float = 0.0
    physics_step: int = 0

    def __post_init__(self) -> None:
        if not str(self.segment_id).strip():
            raise ValueError("segment_id must be non-empty")
        self.length_m = _positive(self.length_m, "length_m")
        self.reference_radius_m = _positive(
            self.reference_radius_m, "reference_radius_m"
        )
        reference_area = math.pi * self.reference_radius_m**2
        self.lumen_area_m2 = (
            reference_area
            if self.lumen_area_m2 is None
            else _positive(self.lumen_area_m2, "lumen_area_m2")
        )
        self.wall_damage_fraction = _fraction(
            self.wall_damage_fraction, "wall_damage_fraction"
        )
        self.residual_defect_area_m2 = _non_negative(
            self.residual_defect_area_m2, "residual_defect_area_m2"
        )
        self.time_s = _non_negative(self.time_s, "time_s")
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step < 0:
            raise ValueError("physics_step must be a non-negative integer")
        self.physics_step = physics_step

    @property
    def reference_lumen_area_m2(self) -> float:
        return math.pi * self.reference_radius_m**2


@dataclass(frozen=True)
class VesselObservation:
    """Scene-addressable observation derived from mutable vessel state.

    ``measured_flow_m3_s`` is the axial segment outlet flow.  Together with
    ``inlet_flow_m3_s``, outward ``leak_flow_m3_s``, and signed wall
    ``storage_flow_m3_s`` it closes the reported segment continuity balance.
    """

    scene_component_id: str
    segment_id: str
    residual_defect_area_m2: float
    upstream_pressure_pa: float
    downstream_pressure_pa: float
    inlet_flow_m3_s: float
    measured_flow_m3_s: float
    leak_flow_m3_s: float
    storage_flow_m3_s: float
    mass_balance_residual_m3_s: float
    lumen_area_m2: float
    occlusion_fraction: float
    wall_damage_fraction: float
    contact_area_m2: float
    contact_pressure_pa: float
    parameter_status: str
    physics_step: int
    previous_time_s: float
    time_s: float
    dt_s: float

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        if not str(self.segment_id).strip():
            raise ValueError("segment_id must be non-empty")
        object.__setattr__(
            self,
            "residual_defect_area_m2",
            _non_negative(
                self.residual_defect_area_m2, "residual_defect_area_m2"
            ),
        )
        object.__setattr__(
            self,
            "upstream_pressure_pa",
            _finite(self.upstream_pressure_pa, "upstream_pressure_pa"),
        )
        object.__setattr__(
            self,
            "downstream_pressure_pa",
            _finite(self.downstream_pressure_pa, "downstream_pressure_pa"),
        )
        if self.downstream_pressure_pa > self.upstream_pressure_pa:
            raise ValueError(
                "downstream_pressure_pa cannot exceed upstream_pressure_pa "
                "in the unidirectional vessel model"
            )
        object.__setattr__(
            self,
            "inlet_flow_m3_s",
            _non_negative(self.inlet_flow_m3_s, "inlet_flow_m3_s"),
        )
        object.__setattr__(
            self,
            "measured_flow_m3_s",
            _non_negative(self.measured_flow_m3_s, "measured_flow_m3_s"),
        )
        object.__setattr__(
            self,
            "leak_flow_m3_s",
            _non_negative(self.leak_flow_m3_s, "leak_flow_m3_s"),
        )
        object.__setattr__(
            self,
            "storage_flow_m3_s",
            _finite(self.storage_flow_m3_s, "storage_flow_m3_s"),
        )
        object.__setattr__(
            self,
            "mass_balance_residual_m3_s",
            _finite(
                self.mass_balance_residual_m3_s,
                "mass_balance_residual_m3_s",
            ),
        )
        flow_scale = max(
            abs(self.inlet_flow_m3_s),
            abs(self.measured_flow_m3_s),
            abs(self.leak_flow_m3_s),
            abs(self.storage_flow_m3_s),
        )
        flow_tolerance = (
            0.0
            if flow_scale == 0.0
            else max(
                64.0 * math.ulp(flow_scale),
                1.0e-10 * flow_scale,
            )
        )
        if abs(self.mass_balance_residual_m3_s) > flow_tolerance:
            raise ValueError(
                "mass_balance_residual_m3_s exceeds the numerical continuity "
                "tolerance"
            )
        object.__setattr__(
            self, "lumen_area_m2", _positive(self.lumen_area_m2, "lumen_area_m2")
        )
        object.__setattr__(
            self,
            "occlusion_fraction",
            _fraction(self.occlusion_fraction, "occlusion_fraction"),
        )
        object.__setattr__(
            self,
            "wall_damage_fraction",
            _fraction(self.wall_damage_fraction, "wall_damage_fraction"),
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
        if not str(self.parameter_status).strip():
            raise ValueError("parameter_status must be non-empty")
        physics_step = int(self.physics_step)
        if physics_step != self.physics_step or physics_step <= 0:
            raise ValueError("physics_step must be a positive integer")
        object.__setattr__(self, "physics_step", physics_step)
        previous_time = _non_negative(self.previous_time_s, "previous_time_s")
        time = _non_negative(self.time_s, "time_s")
        dt = _positive(self.dt_s, "dt_s")
        time_tolerance = max(
            64.0 * math.ulp(max(time, previous_time, dt)),
            1.0e-12 * max(time, previous_time, dt),
        )
        if time <= previous_time or not math.isclose(
            time,
            previous_time + dt,
            rel_tol=0.0,
            abs_tol=time_tolerance,
        ):
            raise ValueError(
                "time_s must be the strictly later endpoint of "
                "[previous_time_s, previous_time_s + dt_s]"
            )
        object.__setattr__(self, "previous_time_s", previous_time)
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "dt_s", dt)


@dataclass
class VesselMechanics:
    """Serial compliant-vessel mechanics with causal leakage and damage."""

    scene_component_id: str
    material: VesselMaterial
    segments: list[VesselSegmentState]
    ambient_pressure_pa: float = 0.0
    _observations: dict[str, VesselObservation] = field(
        default_factory=dict, init=False, repr=False
    )
    _time_s: float = field(init=False, repr=False)
    _physics_step: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.scene_component_id).strip():
            raise ValueError("scene_component_id must be non-empty")
        if not self.segments:
            raise ValueError("at least one vessel segment is required")
        identifiers = [segment.segment_id for segment in self.segments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("vessel segment IDs must be unique")
        initial_time = self.segments[0].time_s
        initial_step = self.segments[0].physics_step
        for segment in self.segments:
            if (
                segment.time_s != initial_time
                or segment.physics_step != initial_step
            ):
                raise ValueError(
                    "all vessel segments must share one time and physics_step"
                )
            reference_area = segment.reference_lumen_area_m2
            minimum_area = reference_area * self.material.minimum_area_fraction
            maximum_area = reference_area * self.material.maximum_area_fraction
            if not minimum_area <= (segment.lumen_area_m2 or 0.0) <= maximum_area:
                raise ValueError(
                    f"initial lumen area for {segment.segment_id!r} lies outside "
                    "the material's admissible area range"
                )
            maximum_defect_area = (
                2.0
                * math.pi
                * segment.reference_radius_m
                * segment.length_m
            )
            if segment.residual_defect_area_m2 > maximum_defect_area:
                raise ValueError(
                    f"initial defect for {segment.segment_id!r} exceeds its "
                    "reference wall surface"
                )
        self.ambient_pressure_pa = _finite(
            self.ambient_pressure_pa, "ambient_pressure_pa"
        )
        self._time_s = initial_time
        self._physics_step = initial_step

    @classmethod
    def uniform(
        cls,
        *,
        scene_component_id: str,
        material: VesselMaterial,
        segment_count: int,
        total_length_m: float,
        reference_radius_m: float,
        ambient_pressure_pa: float = 0.0,
    ) -> "VesselMechanics":
        count = int(segment_count)
        if count <= 0:
            raise ValueError("segment_count must be positive")
        length = _positive(total_length_m, "total_length_m") / count
        return cls(
            scene_component_id=scene_component_id,
            material=material,
            segments=[
                VesselSegmentState(
                    segment_id=f"segment_{index:03d}",
                    length_m=length,
                    reference_radius_m=reference_radius_m,
                )
                for index in range(count)
            ],
            ambient_pressure_pa=ambient_pressure_pa,
        )

    def _resistance_pa_s_m3(
        self,
        segment: VesselSegmentState,
        *,
        lumen_area_m2: float | None = None,
    ) -> float:
        area = (
            segment.lumen_area_m2
            if lumen_area_m2 is None
            else _positive(lumen_area_m2, "lumen_area_m2")
        )
        if area is None or area <= 0.0:
            raise RuntimeError(
                f"segment {segment.segment_id!r} has no open lumen for the "
                "Poiseuille flow branch"
            )
        radius = math.sqrt(area / math.pi)
        denominator = math.pi * radius**4
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise RuntimeError(
                f"segment {segment.segment_id!r} has an invalid hydraulic "
                "radius"
            )
        resistance = (
            8.0
            * self.material.blood_dynamic_viscosity_pa_s
            * segment.length_m
            / denominator
        )
        if resistance <= 0.0 or not math.isfinite(resistance):
            raise RuntimeError(
                f"segment {segment.segment_id!r} has a non-finite hydraulic "
                "resistance"
            )
        return resistance

    def advance(
        self,
        dt_s: float,
        *,
        upstream_pressure_pa: float,
        downstream_pressure_pa: float,
        contact_loads: Iterable[VesselContactLoad] = (),
    ) -> tuple[VesselObservation, ...]:
        """Advance wall state and derive the serial pressure/flow observation."""

        dt = _positive(dt_s, "dt_s")
        for segment in self.segments:
            if (
                segment.time_s != self._time_s
                or segment.physics_step != self._physics_step
            ):
                raise RuntimeError(
                    "vessel segment time/step provenance was mutated outside "
                    "VesselMechanics"
                )
        previous_time = self._time_s
        next_time = previous_time + dt
        if not math.isfinite(next_time) or next_time <= previous_time:
            raise ValueError("dt_s does not advance vessel time monotonically")
        next_physics_step = self._physics_step + 1
        inlet_pressure = _finite(upstream_pressure_pa, "upstream_pressure_pa")
        outlet_pressure = _finite(downstream_pressure_pa, "downstream_pressure_pa")
        if inlet_pressure < outlet_pressure:
            raise ValueError(
                "VesselMechanics is a unidirectional reduced model; "
                "upstream_pressure_pa must be at least downstream_pressure_pa"
            )
        loads_by_segment: dict[str, list[VesselContactLoad]] = {
            segment.segment_id: [] for segment in self.segments
        }
        load_sources: set[tuple[str, str]] = set()
        for load in contact_loads:
            if load.segment_id not in loads_by_segment:
                raise KeyError(f"unknown vessel segment {load.segment_id!r}")
            identity = (load.segment_id, load.source_component_id)
            if identity in load_sources:
                raise ValueError(
                    "publish one area-weighted VesselContactLoad per source "
                    f"and segment; duplicate {identity!r}"
                )
            load_sources.add(identity)
            loads_by_segment[load.segment_id].append(load)

        contact_state: list[tuple[float, float, float]] = []
        for segment in self.segments:
            segment_loads = loads_by_segment[segment.segment_id]
            contact_area = sum(load.contact_area_m2 for load in segment_loads)
            contact_force = sum(load.normal_force_n for load in segment_loads)
            contact_pressure = (
                contact_force / contact_area if contact_area > 0.0 else 0.0
            )
            peak_contact_pressure = max(
                (load.pressure_pa for load in segment_loads),
                default=0.0,
            )
            contact_state.append(
                (contact_area, contact_pressure, peak_contact_pressure)
            )

        def solve_serial_flow(
            resistances: list[float],
            defect_areas_m2: list[float],
            storage_flows_m3_s: list[float] | None = None,
        ) -> list[
            tuple[float, float, float, float, float, float]
        ]:
            """Solve a mass-conserving chain with midpoint orifice leaks."""

            if (
                len(resistances) != len(self.segments)
                or len(defect_areas_m2) != len(self.segments)
            ):
                raise ValueError(
                    "resistance and defect area must name every vessel segment"
                )
            resistances = [
                _positive(value, "hydraulic resistance")
                for value in resistances
            ]
            defect_areas = [
                _non_negative(value, "residual defect area")
                for value in defect_areas_m2
            ]
            storage_flows = (
                [0.0 for _ in self.segments]
                if storage_flows_m3_s is None
                else [
                    _finite(value, "storage flow")
                    for value in storage_flows_m3_s
                ]
            )
            if len(storage_flows) != len(self.segments):
                raise ValueError(
                    "storage flow must name every vessel segment"
                )
            pressure_span = inlet_pressure - outlet_pressure

            def propagate(
                inlet_flow_m3_s: float,
            ) -> tuple[
                float,
                list[
                    tuple[
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                    ]
                ],
                bool,
            ]:
                pressure = inlet_pressure
                flow = _non_negative(inlet_flow_m3_s, "inlet flow")
                rows: list[
                    tuple[
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                    ]
                ] = []
                feasible = True
                for resistance, defect_area, storage_flow in zip(
                    resistances,
                    defect_areas,
                    storage_flows,
                ):
                    inlet_segment_flow = flow
                    available_after_storage = flow - storage_flow
                    if available_after_storage < 0.0:
                        feasible = False
                        available_after_storage = 0.0

                    def leak_residual(
                        candidate_leak: float,
                    ) -> tuple[float, float]:
                        candidate_outlet = (
                            available_after_storage - candidate_leak
                        )
                        candidate_pressure = pressure - (
                            0.5
                            * (flow + candidate_outlet)
                            * resistance
                        )
                        if not math.isfinite(candidate_pressure):
                            raise RuntimeError(
                                "vessel pressure propagation became non-finite"
                            )
                        mean_pressure = 0.5 * (
                            pressure + candidate_pressure
                        )
                        leak_pressure = max(
                            0.0,
                            mean_pressure - self.ambient_pressure_pa,
                        )
                        demanded = (
                            self.material.leak_discharge_coefficient
                            * defect_area
                            * math.sqrt(
                                2.0
                                * leak_pressure
                                / self.material.blood_density_kg_m3
                            )
                        )
                        if not math.isfinite(demanded):
                            raise RuntimeError(
                                "vessel orifice-flow demand became non-finite"
                            )
                        return candidate_leak - demanded, candidate_pressure

                    lower_leak = 0.0
                    upper_leak = available_after_storage
                    lower_residual, _ = leak_residual(lower_leak)
                    upper_residual, _ = leak_residual(upper_leak)
                    residual_scale = max(
                        abs(lower_residual),
                        abs(upper_residual),
                        available_after_storage,
                    )
                    residual_tolerance = (
                        0.0
                        if residual_scale == 0.0
                        else max(
                            64.0 * math.ulp(residual_scale),
                            1.0e-10 * residual_scale,
                        )
                    )
                    if lower_residual >= -residual_tolerance:
                        leak_flow = lower_leak
                    elif upper_residual < -residual_tolerance:
                        # The proposed inlet flow cannot supply the physical
                        # orifice demand.  Keep continuity in this trial row,
                        # but make the outer pressure solve reject the trial.
                        feasible = False
                        leak_flow = upper_leak
                    elif upper_residual <= residual_tolerance:
                        leak_flow = upper_leak
                    else:
                        leak_flow = 0.0
                        residual = -math.inf
                        for _ in range(96):
                            leak_flow = 0.5 * (
                                lower_leak + upper_leak
                            )
                            residual, _ = leak_residual(leak_flow)
                            if residual > 0.0:
                                upper_leak = leak_flow
                            else:
                                lower_leak = leak_flow
                            if abs(residual) <= residual_tolerance:
                                break
                            next_leak = 0.5 * (
                                lower_leak + upper_leak
                            )
                            if (
                                next_leak == lower_leak
                                or next_leak == upper_leak
                            ):
                                raise RuntimeError(
                                    "vessel leak solve stalled before "
                                    "convergence"
                                )
                        else:
                            raise RuntimeError(
                                "vessel leak solve did not converge"
                            )
                        resolved_residual, _ = leak_residual(leak_flow)
                        if abs(resolved_residual) > residual_tolerance:
                            raise RuntimeError(
                                "vessel leak solve returned an unconverged root"
                            )
                    outlet_flow = available_after_storage - leak_flow
                    if outlet_flow < 0.0:
                        raise RuntimeError(
                            "vessel leak solve produced reverse outlet flow"
                        )
                    next_pressure = pressure - (
                        0.5 * (flow + outlet_flow) * resistance
                    )
                    if not math.isfinite(next_pressure):
                        raise RuntimeError(
                            "vessel pressure propagation became non-finite"
                        )
                    rows.append(
                        (
                            pressure,
                            next_pressure,
                            inlet_segment_flow,
                            outlet_flow,
                            leak_flow,
                            storage_flow,
                        )
                    )
                    pressure = next_pressure
                    flow = outlet_flow
                return pressure, rows, feasible

            pressure_scale = max(
                abs(inlet_pressure),
                abs(outlet_pressure),
                abs(pressure_span),
            )
            pressure_tolerance = max(
                64.0 * math.ulp(pressure_scale),
                1.0e-10 * abs(pressure_span),
            )
            if pressure_span == 0.0:
                terminal_pressure, rows, feasible = propagate(0.0)
                if (
                    not feasible
                    or abs(terminal_pressure - outlet_pressure)
                    > pressure_tolerance
                ):
                    raise RuntimeError(
                        "equal endpoint pressures require bidirectional vessel "
                        "flow for the current leak/storage state"
                    )
                return rows

            (
                lower_pressure,
                lower_rows,
                lower_feasible,
            ) = propagate(0.0)
            if lower_feasible:
                lower_residual = lower_pressure - outlet_pressure
                if abs(lower_residual) <= pressure_tolerance:
                    return lower_rows
                if lower_residual < 0.0:
                    raise RuntimeError(
                        "the vessel boundary state requires reverse inlet flow, "
                        "outside this unidirectional model"
                    )

            total_resistance = sum(resistances)
            if total_resistance <= 0.0 or not math.isfinite(total_resistance):
                raise RuntimeError(
                    "serial vessel resistance must be positive and finite"
                )
            lower_flow = 0.0
            upper_flow = (
                pressure_span / total_resistance
                + sum(max(0.0, value) for value in storage_flows)
            )
            if upper_flow <= 0.0 or not math.isfinite(upper_flow):
                raise RuntimeError(
                    "could not form a finite positive vessel-flow bracket"
                )
            upper_pressure, _, upper_feasible = propagate(upper_flow)
            for _ in range(96):
                if upper_feasible and upper_pressure <= outlet_pressure:
                    break
                upper_flow *= 2.0
                if not math.isfinite(upper_flow):
                    raise RuntimeError(
                        "vessel-flow bracket overflowed before finding a "
                        "feasible endpoint"
                    )
                (
                    upper_pressure,
                    _,
                    upper_feasible,
                ) = propagate(upper_flow)
            else:
                raise RuntimeError(
                    "could not bracket the mass-conserving vessel flow solution"
                )

            rows: list[
                tuple[
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                ]
            ] = []
            terminal_pressure = math.inf
            feasible = False
            for _ in range(96):
                trial_flow = 0.5 * (lower_flow + upper_flow)
                terminal_pressure, rows, feasible = propagate(trial_flow)
                if not feasible or terminal_pressure > outlet_pressure:
                    lower_flow = trial_flow
                else:
                    upper_flow = trial_flow
                if feasible and abs(
                    terminal_pressure - outlet_pressure
                ) <= pressure_tolerance:
                    break
                next_flow = 0.5 * (lower_flow + upper_flow)
                if next_flow == lower_flow or next_flow == upper_flow:
                    raise RuntimeError(
                        "vessel endpoint-pressure solve stalled before "
                        "convergence"
                    )
            else:
                raise RuntimeError(
                    "vessel endpoint-pressure solve did not converge"
                )
            return rows

        previous_areas = [
            float(segment.lumen_area_m2)
            for segment in self.segments
        ]
        previous_damages = [
            segment.wall_damage_fraction for segment in self.segments
        ]
        previous_defects = [
            segment.residual_defect_area_m2 for segment in self.segments
        ]
        relaxation_fraction = 1.0 - math.exp(
            -dt / self.material.radial_relaxation_time_s
        )

        def wall_update(
            rows: list[
                tuple[float, float, float, float, float, float]
            ],
        ) -> tuple[list[float], list[float], list[float]]:
            updated_areas: list[float] = []
            updated_damages: list[float] = []
            updated_defects: list[float] = []
            for (
                segment,
                previous_area,
                previous_damage,
                previous_defect,
                row,
                (_, contact_pressure, peak_contact_pressure),
            ) in zip(
                self.segments,
                previous_areas,
                previous_damages,
                previous_defects,
                rows,
                contact_state,
            ):
                mean_pressure = 0.5 * (row[0] + row[1])
                transmural_pressure = mean_pressure - (
                    self.ambient_pressure_pa + contact_pressure
                )
                reference_area = segment.reference_lumen_area_m2
                target_fraction = 1.0 + (
                    self.material.area_compliance_per_pa
                    * transmural_pressure
                )
                target_fraction = min(
                    self.material.maximum_area_fraction,
                    max(
                        self.material.minimum_area_fraction,
                        target_fraction,
                    ),
                )
                target_area = reference_area * target_fraction
                updated_area = previous_area + relaxation_fraction * (
                    target_area - previous_area
                )
                updated_area = min(
                    reference_area * self.material.maximum_area_fraction,
                    max(
                        reference_area
                        * self.material.minimum_area_fraction,
                        updated_area,
                    ),
                )

                damaging_pressure = max(
                    peak_contact_pressure,
                    max(
                        0.0,
                        mean_pressure - self.ambient_pressure_pa,
                    ),
                )
                updated_damage = previous_damage
                if damaging_pressure > self.material.damage_pressure_pa:
                    utilization = (
                        damaging_pressure
                        - self.material.damage_pressure_pa
                    ) / (
                        self.material.rupture_pressure_pa
                        - self.material.damage_pressure_pa
                    )
                    updated_damage = min(
                        1.0,
                        previous_damage
                        + self.material.damage_rate_per_s
                        * max(0.0, utilization)
                        * dt,
                    )
                    if (
                        damaging_pressure
                        >= self.material.rupture_pressure_pa
                    ):
                        updated_damage = 1.0

                updated_defect = previous_defect
                if updated_damage > 0.5:
                    updated_defect += (
                        self.material.defect_growth_rate_m2_s
                        * (updated_damage - 0.5)
                        / 0.5
                        * dt
                    )
                maximum_defect_area = (
                    2.0
                    * math.pi
                    * segment.reference_radius_m
                    * segment.length_m
                )
                updated_defect = min(
                    maximum_defect_area,
                    updated_defect,
                )
                updated_areas.append(updated_area)
                updated_damages.append(updated_damage)
                updated_defects.append(updated_defect)
            return updated_areas, updated_damages, updated_defects

        def wall_state_converged(
            areas_a: list[float],
            damages_a: list[float],
            defects_a: list[float],
            areas_b: list[float],
            damages_b: list[float],
            defects_b: list[float],
        ) -> bool:
            for (
                segment,
                area_a,
                damage_a,
                defect_a,
                area_b,
                damage_b,
                defect_b,
            ) in zip(
                self.segments,
                areas_a,
                damages_a,
                defects_a,
                areas_b,
                damages_b,
                defects_b,
            ):
                reference_area = segment.reference_lumen_area_m2
                wall_surface_area = (
                    2.0
                    * math.pi
                    * segment.reference_radius_m
                    * segment.length_m
                )
                area_tolerance = max(
                    64.0 * math.ulp(reference_area),
                    1.0e-10 * reference_area,
                )
                damage_tolerance = max(
                    64.0 * math.ulp(1.0),
                    1.0e-10,
                )
                defect_tolerance = max(
                    64.0 * math.ulp(wall_surface_area),
                    1.0e-10 * wall_surface_area,
                )
                if (
                    abs(area_a - area_b) > area_tolerance
                    or abs(damage_a - damage_b) > damage_tolerance
                    or abs(defect_a - defect_b) > defect_tolerance
                ):
                    return False
            return True

        candidate_areas = list(previous_areas)
        candidate_damages = list(previous_damages)
        candidate_defects = list(previous_defects)
        final_rows: list[
            tuple[float, float, float, float, float, float]
        ] = []
        for _ in range(96):
            candidate_resistances = [
                self._resistance_pa_s_m3(
                    segment,
                    lumen_area_m2=area,
                )
                for segment, area in zip(
                    self.segments,
                    candidate_areas,
                )
            ]
            candidate_storage_flows = [
                (area - previous_area) * segment.length_m / dt
                for segment, area, previous_area in zip(
                    self.segments,
                    candidate_areas,
                    previous_areas,
                )
            ]
            candidate_rows = solve_serial_flow(
                candidate_resistances,
                candidate_defects,
                candidate_storage_flows,
            )
            next_areas, next_damages, next_defects = wall_update(
                candidate_rows
            )
            if wall_state_converged(
                candidate_areas,
                candidate_damages,
                candidate_defects,
                next_areas,
                next_damages,
                next_defects,
            ):
                # Re-solve once at the constitutive update itself so the
                # returned continuity rows use exactly the state being
                # committed, including its compliant storage.
                candidate_areas = next_areas
                candidate_damages = next_damages
                candidate_defects = next_defects
                verified_resistances = [
                    self._resistance_pa_s_m3(
                        segment,
                        lumen_area_m2=area,
                    )
                    for segment, area in zip(
                        self.segments,
                        candidate_areas,
                    )
                ]
                verified_storage_flows = [
                    (area - previous_area) * segment.length_m / dt
                    for segment, area, previous_area in zip(
                        self.segments,
                        candidate_areas,
                        previous_areas,
                    )
                ]
                verified_rows = solve_serial_flow(
                    verified_resistances,
                    candidate_defects,
                    verified_storage_flows,
                )
                (
                    verified_areas,
                    verified_damages,
                    verified_defects,
                ) = wall_update(verified_rows)
                if wall_state_converged(
                    candidate_areas,
                    candidate_damages,
                    candidate_defects,
                    verified_areas,
                    verified_damages,
                    verified_defects,
                ):
                    final_rows = verified_rows
                    break
                next_areas = verified_areas
                next_damages = verified_damages
                next_defects = verified_defects

            # Under-relax the nonlinear wall/flow fixed point.  This changes
            # only the iterative solve path; the converged state must still
            # satisfy the unrelaxed constitutive update above.
            candidate_areas = [
                0.5 * (current + updated)
                for current, updated in zip(
                    candidate_areas,
                    next_areas,
                )
            ]
            candidate_damages = [
                0.5 * (current + updated)
                for current, updated in zip(
                    candidate_damages,
                    next_damages,
                )
            ]
            candidate_defects = [
                0.5 * (current + updated)
                for current, updated in zip(
                    candidate_defects,
                    next_defects,
                )
            ]
        else:
            raise RuntimeError(
                "compliant vessel wall/flow coupling did not converge"
            )

        observations: dict[str, VesselObservation] = {}
        for (
            segment,
            lumen_area,
            wall_damage,
            residual_defect,
            row,
            contact,
        ) in zip(
            self.segments,
            candidate_areas,
            candidate_damages,
            candidate_defects,
            final_rows,
            contact_state,
        ):
            (
                segment_inlet_pressure,
                segment_outlet_pressure,
                inlet_flow,
                measured_flow,
                leak_flow,
                storage_flow,
            ) = row
            contact_area, contact_pressure, _ = contact
            reference_area = segment.reference_lumen_area_m2
            occlusion = min(
                1.0,
                max(
                    0.0,
                    1.0 - (lumen_area / reference_area),
                ),
            )
            mass_balance_residual = (
                inlet_flow
                - measured_flow
                - leak_flow
                - storage_flow
            )
            observations[segment.segment_id] = VesselObservation(
                scene_component_id=self.scene_component_id,
                segment_id=segment.segment_id,
                residual_defect_area_m2=residual_defect,
                upstream_pressure_pa=segment_inlet_pressure,
                downstream_pressure_pa=segment_outlet_pressure,
                inlet_flow_m3_s=inlet_flow,
                measured_flow_m3_s=measured_flow,
                leak_flow_m3_s=leak_flow,
                storage_flow_m3_s=storage_flow,
                mass_balance_residual_m3_s=mass_balance_residual,
                lumen_area_m2=lumen_area,
                occlusion_fraction=occlusion,
                wall_damage_fraction=wall_damage,
                contact_area_m2=contact_area,
                contact_pressure_pa=contact_pressure,
                parameter_status=self.material.parameter_status,
                physics_step=next_physics_step,
                previous_time_s=previous_time,
                time_s=next_time,
                dt_s=dt,
            )

        # Commit only after both nonlinear solves and every immutable
        # observation have validated, so a failed advance leaves no partial
        # wall, defect, clock, or evidence mutation.
        for (
            segment,
            lumen_area,
            wall_damage,
            residual_defect,
        ) in zip(
            self.segments,
            candidate_areas,
            candidate_damages,
            candidate_defects,
        ):
            segment.lumen_area_m2 = lumen_area
            segment.wall_damage_fraction = wall_damage
            segment.residual_defect_area_m2 = residual_defect
            segment.time_s = next_time
            segment.physics_step = next_physics_step
        self._time_s = next_time
        self._physics_step = next_physics_step
        self._observations = observations
        return tuple(observations[segment.segment_id] for segment in self.segments)

    def observe(self, segment_id: str) -> VesselObservation:
        """Return the most recent derived observation for a named segment."""

        try:
            return self._observations[str(segment_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"no vessel observation is available for {segment_id!r}; "
                "advance the mechanics first"
            ) from exc

    def observations(self) -> tuple[VesselObservation, ...]:
        return tuple(
            self._observations[segment.segment_id]
            for segment in self.segments
            if segment.segment_id in self._observations
        )


__all__ = [
    "VesselContactLoad",
    "VesselMaterial",
    "VesselMechanics",
    "VesselObservation",
    "VesselSegmentState",
]
