# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Scene-evidence-owned circulation and ventilation effects for the rescue OR."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .resuscitation_scene_evidence import (
    PumpSceneEvidence,
    VentilationSceneEvidence,
)

CHANNELS: Final = frozenset({"crystalloid", "blood_product", "vasopressor"})


def _finite(value: float, name: str) -> float:
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be finite")
    return rendered


def _nonnegative(value: float, name: str) -> float:
    rendered = _finite(value, name)
    if rendered < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return rendered


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _required_text(value: str, name: str) -> str:
    rendered = str(value).strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered


def _attachment_prim_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    rendered = tuple(
        sorted(
            _required_text(value, "attachment_prim_id")
            for value in values
        )
    )
    if any(not value.startswith("/") for value in rendered):
        raise ValueError("attachment prim IDs must be absolute scene paths")
    if len(rendered) != len(set(rendered)):
        raise ValueError("attachment prim IDs must be unique")
    return rendered


@dataclass(frozen=True)
class _PumpEvidenceFrame:
    """Internal projection of a validated pump scene-evidence envelope.

    This type cannot be exported as a caller-authored outcome surface.  The
    owning adapter constructs it only from :class:`PumpSceneEvidence`.
    """

    physics_step: int
    simulation_time_s: float
    dt_s: float
    episode_id: str
    environment_id: str
    topology_revision: str
    channel_id: str
    access_attachment_prim_ids: tuple[str, ...]
    delivery_coordinate_m: float
    downstream_flow_ml_s: float
    vascular_inflow_ml_s: float
    reservoir_mass_g: float
    line_pressure_kpa: float
    evidence_digest_sha256: str

    def __post_init__(self) -> None:
        if self.physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        for name in (
            "episode_id",
            "environment_id",
            "topology_revision",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name),
            )
        if self.channel_id not in CHANNELS:
            raise ValueError(f"unsupported resuscitation channel {self.channel_id!r}")
        object.__setattr__(
            self,
            "access_attachment_prim_ids",
            _attachment_prim_ids(self.access_attachment_prim_ids),
        )
        for name in (
            "simulation_time_s",
            "dt_s",
            "downstream_flow_ml_s",
            "vascular_inflow_ml_s",
            "reservoir_mass_g",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "delivery_coordinate_m",
            _finite(self.delivery_coordinate_m, "delivery_coordinate_m"),
        )
        object.__setattr__(
            self,
            "line_pressure_kpa",
            _finite(self.line_pressure_kpa, "line_pressure_kpa"),
        )
        digest = _required_text(
            self.evidence_digest_sha256,
            "evidence_digest_sha256",
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest.lower()
        ):
            raise ValueError(
                "evidence_digest_sha256 must be a hexadecimal SHA-256 digest"
            )
        object.__setattr__(self, "evidence_digest_sha256", digest.lower())
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be greater than zero")
        if (
            self.vascular_inflow_ml_s > 1.0e-9
            and not self.access_attachment_prim_ids
        ):
            raise ValueError(
                "vascular inflow requires a live access attachment prim"
            )

    @classmethod
    def from_scene_evidence(
        cls,
        evidence: PumpSceneEvidence,
    ) -> _PumpEvidenceFrame:
        return cls(
            physics_step=evidence.physics_step,
            simulation_time_s=evidence.simulation_time_s,
            dt_s=evidence.dt_s,
            episode_id=evidence.episode_id,
            environment_id=evidence.environment_id,
            topology_revision=evidence.topology_revision,
            channel_id=evidence.channel_id,
            access_attachment_prim_ids=evidence.attachment_prim_ids,
            delivery_coordinate_m=evidence.delivery_coordinate_m,
            downstream_flow_ml_s=evidence.downstream_flow_ml_s,
            vascular_inflow_ml_s=evidence.vascular_inflow_ml_s,
            reservoir_mass_g=evidence.reservoir_mass_g,
            line_pressure_kpa=evidence.line_pressure_kpa,
            evidence_digest_sha256=evidence.evidence_digest_sha256,
        )


@dataclass(frozen=True)
class _VentilationEvidenceFrame:
    """Internal projection of validated ventilation scene evidence."""

    physics_step: int
    simulation_time_s: float
    dt_s: float
    episode_id: str
    environment_id: str
    topology_revision: str
    airway_attachment_prim_ids: tuple[str, ...]
    valve_opening_angle_deg: float
    inspiratory_flow_l_min: float
    airway_interface_flow_l_min: float
    airway_pressure_cmh2o: float
    measured_fio2_fraction: float
    chest_excursion_m: float
    evidence_digest_sha256: str

    def __post_init__(self) -> None:
        if self.physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        for name in (
            "episode_id",
            "environment_id",
            "topology_revision",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "airway_attachment_prim_ids",
            _attachment_prim_ids(self.airway_attachment_prim_ids),
        )
        for name in (
            "simulation_time_s",
            "dt_s",
            "valve_opening_angle_deg",
            "inspiratory_flow_l_min",
            "airway_interface_flow_l_min",
            "chest_excursion_m",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "airway_pressure_cmh2o",
            _finite(self.airway_pressure_cmh2o, "airway_pressure_cmh2o"),
        )
        fio2 = _finite(self.measured_fio2_fraction, "measured_fio2_fraction")
        if not 0.21 <= fio2 <= 1.0:
            raise ValueError("measured_fio2_fraction must be within [0.21, 1]")
        object.__setattr__(self, "measured_fio2_fraction", fio2)
        digest = _required_text(
            self.evidence_digest_sha256,
            "evidence_digest_sha256",
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest.lower()
        ):
            raise ValueError(
                "evidence_digest_sha256 must be a hexadecimal SHA-256 digest"
            )
        object.__setattr__(self, "evidence_digest_sha256", digest.lower())
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be greater than zero")
        if (
            self.airway_interface_flow_l_min > 1.0e-9
            and not self.airway_attachment_prim_ids
        ):
            raise ValueError(
                "airway-interface flow requires a live attachment prim"
            )

    @classmethod
    def from_scene_evidence(
        cls,
        evidence: VentilationSceneEvidence,
    ) -> _VentilationEvidenceFrame:
        return cls(
            physics_step=evidence.physics_step,
            simulation_time_s=evidence.simulation_time_s,
            dt_s=evidence.dt_s,
            episode_id=evidence.episode_id,
            environment_id=evidence.environment_id,
            topology_revision=evidence.topology_revision,
            airway_attachment_prim_ids=evidence.attachment_prim_ids,
            valve_opening_angle_deg=evidence.valve_opening_angle_deg,
            inspiratory_flow_l_min=evidence.inspiratory_flow_l_min,
            airway_interface_flow_l_min=evidence.airway_interface_flow_l_min,
            airway_pressure_cmh2o=evidence.airway_pressure_cmh2o,
            measured_fio2_fraction=evidence.measured_fio2_fraction,
            chest_excursion_m=evidence.chest_excursion_m,
            evidence_digest_sha256=evidence.evidence_digest_sha256,
        )


@dataclass(frozen=True)
class ResuscitationCalibration:
    plunger_stroke_m: float = 0.18
    crystalloid_volume_per_stroke_ml: float = 500.0
    blood_volume_per_stroke_ml: float = 300.0
    vasopressor_volume_per_stroke_ml: float = 10.0
    crystalloid_density_g_ml: float = 1.00
    blood_density_g_ml: float = 1.06
    vasopressor_density_g_ml: float = 1.00
    minimum_line_pressure_kpa: float = 1.0
    maximum_line_pressure_kpa: float = 40.0
    hard_line_pressure_kpa: float = 65.0
    initial_crystalloid_inventory_ml: float = 3000.0
    initial_blood_inventory_ml: float = 1200.0
    initial_vasopressor_inventory_ml: float = 40.0
    crystalloid_intravascular_retention_fraction: float = 0.24
    ventilation_valve_full_open_deg: float = 90.0
    ventilation_target_chest_excursion_m: float = 0.02
    ventilation_minimum_airway_pressure_cmh2o: float = 4.0
    ventilation_maximum_airway_pressure_cmh2o: float = 30.0
    ventilation_hard_airway_pressure_cmh2o: float = 45.0
    ventilation_reporting_window_s: float = 60.0
    parameter_status: str = "provisional_engineering_seeds"

    def __post_init__(self) -> None:
        if self.parameter_status != "provisional_engineering_seeds":
            raise ValueError(
                "resuscitation calibration must retain its provisional "
                "engineering status"
            )
        for name, value in self.__dict__.items():
            if name == "parameter_status":
                continue
            _nonnegative(value, name)
        if self.plunger_stroke_m <= 0.0:
            raise ValueError("plunger_stroke_m must be positive")
        for name in (
            "crystalloid_volume_per_stroke_ml",
            "blood_volume_per_stroke_ml",
            "vasopressor_volume_per_stroke_ml",
            "crystalloid_density_g_ml",
            "blood_density_g_ml",
            "vasopressor_density_g_ml",
            "ventilation_valve_full_open_deg",
            "ventilation_target_chest_excursion_m",
            "ventilation_reporting_window_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_line_pressure_kpa > self.maximum_line_pressure_kpa:
            raise ValueError("minimum_line_pressure_kpa must not exceed maximum_line_pressure_kpa")
        if self.hard_line_pressure_kpa <= self.maximum_line_pressure_kpa:
            raise ValueError("hard_line_pressure_kpa must exceed maximum_line_pressure_kpa")
        if (
            self.ventilation_minimum_airway_pressure_cmh2o
            > self.ventilation_maximum_airway_pressure_cmh2o
        ):
            raise ValueError("ventilation minimum pressure must not exceed maximum pressure")
        if (
            self.ventilation_hard_airway_pressure_cmh2o
            <= self.ventilation_maximum_airway_pressure_cmh2o
        ):
            raise ValueError("ventilation hard pressure must exceed maximum pressure")
        if not 0.0 <= self.crystalloid_intravascular_retention_fraction <= 1.0:
            raise ValueError("crystalloid_intravascular_retention_fraction must be a fraction")


@dataclass
class _PumpChannelState:
    channel_id: str
    remaining_inventory_ml: float
    delivered_to_patient_ml: float = 0.0
    withdrawn_from_reservoir_ml: float = 0.0
    wasted_or_extravasated_ml: float = 0.0
    measured_extravasated_ml: float = 0.0
    line_storage_or_unreconciled_ml: float = 0.0
    pressure_damage_fraction: float = 0.0
    last_delivery_coordinate_m: float | None = None
    last_reservoir_mass_g: float | None = None
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0
    last_evidence_digest_sha256: str | None = None
    last_attachment_prim_ids: tuple[str, ...] = ()
    last_topology_revision: str | None = None


@dataclass
class _VentilationState:
    airway_connected: bool = False
    effective_minute_ventilation_l_min: float = 0.0
    delivered_fio2_fraction: float = 0.21
    airway_pressure_cmh2o: float = 0.0
    chest_excursion_m: float = 0.0
    circuit_leak_fraction: float = 0.0
    pressure_damage_fraction: float = 0.0
    cumulative_effective_ventilation_l: float = 0.0
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0
    last_evidence_digest_sha256: str | None = None
    last_attachment_prim_ids: tuple[str, ...] = ()
    last_topology_revision: str | None = None


@dataclass(frozen=True)
class ResuscitationSnapshot:
    physics_step: int
    simulation_time_s: float
    episode_id: str | None
    environment_id: str | None
    channels: Mapping[
        str,
        Mapping[str, float | int | str | tuple[str, ...] | None],
    ]
    ventilation: Mapping[
        str,
        float | int | bool | str | tuple[str, ...] | None,
    ]
    effective_circulating_volume_gain_ml: float
    evidence_frames: int
    rejected_frames: int
    parameter_status: str
    validation_status: str


class _PumpAuthority:
    __slots__ = ()


class _PumpEvidenceAdapter:
    __slots__ = ("_authority", "_effects")

    def __init__(
        self,
        effects: ContactDrivenResuscitationEffects,
        authority: _PumpAuthority,
    ) -> None:
        self._effects = effects
        self._authority = authority

    def publish(
        self,
        evidence: PumpSceneEvidence,
    ) -> ResuscitationSnapshot:
        frame = _PumpEvidenceFrame.from_scene_evidence(evidence)
        return self._effects._ingest(frame, self._authority)

    def projected_reservoir_withdrawal_ml(
        self,
        evidence: PumpSceneEvidence,
    ) -> float:
        frame = _PumpEvidenceFrame.from_scene_evidence(evidence)
        return self._effects._projected_reservoir_withdrawal_ml(frame)

    def publish_ventilation(
        self,
        evidence: VentilationSceneEvidence,
    ) -> ResuscitationSnapshot:
        frame = _VentilationEvidenceFrame.from_scene_evidence(evidence)
        return self._effects._ingest_ventilation(frame, self._authority)


class ContactDrivenResuscitationEffects:
    """Integrate conserved fluid delivery and measured ventilation support."""

    def __init__(
        self,
        *,
        calibration: ResuscitationCalibration | None = None,
    ) -> None:
        self.calibration = calibration or ResuscitationCalibration()
        self._authority = _PumpAuthority()
        self._physics_step = -1
        self._simulation_time_s = 0.0
        self._episode_id: str | None = None
        self._environment_id: str | None = None
        self._evidence_frames = 0
        self._rejected_frames = 0
        self._channels = self._new_channels()
        self._ventilation = _VentilationState()

    def _new_channels(self) -> dict[str, _PumpChannelState]:
        cfg = self.calibration
        return {
            "crystalloid": _PumpChannelState(
                "crystalloid",
                cfg.initial_crystalloid_inventory_ml,
            ),
            "blood_product": _PumpChannelState(
                "blood_product",
                cfg.initial_blood_inventory_ml,
            ),
            "vasopressor": _PumpChannelState(
                "vasopressor",
                cfg.initial_vasopressor_inventory_ml,
            ),
        }

    def _create_scene_adapter(self) -> _PumpEvidenceAdapter:
        return _PumpEvidenceAdapter(self, self._authority)

    def reset(self) -> ResuscitationSnapshot:
        self._physics_step = -1
        self._simulation_time_s = 0.0
        self._episode_id = None
        self._environment_id = None
        self._evidence_frames = 0
        self._rejected_frames = 0
        self._channels = self._new_channels()
        self._ventilation = _VentilationState()
        return self.snapshot()

    def _channel_volume_per_stroke(self, channel_id: str) -> float:
        return {
            "crystalloid": self.calibration.crystalloid_volume_per_stroke_ml,
            "blood_product": self.calibration.blood_volume_per_stroke_ml,
            "vasopressor": self.calibration.vasopressor_volume_per_stroke_ml,
        }[channel_id]

    def _channel_density(self, channel_id: str) -> float:
        return {
            "crystalloid": self.calibration.crystalloid_density_g_ml,
            "blood_product": self.calibration.blood_density_g_ml,
            "vasopressor": self.calibration.vasopressor_density_g_ml,
        }[channel_id]

    def _validate_runtime_identity(
        self,
        *,
        episode_id: str,
        environment_id: str,
        bind: bool,
    ) -> None:
        if self._episode_id is None:
            if bind:
                self._episode_id = episode_id
                self._environment_id = environment_id
            return
        if (
            episode_id != self._episode_id
            or environment_id != self._environment_id
        ):
            if bind:
                self._rejected_frames += 1
            raise ValueError(
                "resuscitation evidence changed episode or environment "
                "without an effects reset"
            )

    def _validate_pump_sequence(
        self,
        frame: _PumpEvidenceFrame,
        state: _PumpChannelState,
        *,
        record_rejection: bool,
    ) -> None:
        if frame.physics_step <= state.last_physics_step:
            if record_rejection:
                self._rejected_frames += 1
            raise ValueError("pump evidence must use a strictly increasing step per channel")
        if frame.simulation_time_s <= state.last_simulation_time_s and state.last_physics_step >= 0:
            if record_rejection:
                self._rejected_frames += 1
            raise ValueError("pump evidence must use increasing simulation time per channel")

    def _validate_global_clock(
        self,
        *,
        physics_step: int,
        simulation_time_s: float,
        record_rejection: bool,
    ) -> None:
        if physics_step < self._physics_step:
            if record_rejection:
                self._rejected_frames += 1
            raise ValueError(
                "resuscitation evidence may not move the global physics "
                "clock backward"
            )
        if simulation_time_s < self._simulation_time_s:
            if record_rejection:
                self._rejected_frames += 1
            raise ValueError(
                "resuscitation evidence may not move simulation time backward"
            )

    def _reservoir_withdrawal_ml(
        self,
        frame: _PumpEvidenceFrame,
        state: _PumpChannelState,
    ) -> float:
        if state.last_reservoir_mass_g is None:
            return 0.0
        mass_volume_ml = max(
            0.0,
            state.last_reservoir_mass_g - frame.reservoir_mass_g,
        ) / self._channel_density(frame.channel_id)
        return min(mass_volume_ml, state.remaining_inventory_ml)

    def _projected_reservoir_withdrawal_ml(
        self,
        frame: _PumpEvidenceFrame,
    ) -> float:
        """Preview conserved withdrawal for the owning runtime."""

        state = self._channels[frame.channel_id]
        self._validate_runtime_identity(
            episode_id=frame.episode_id,
            environment_id=frame.environment_id,
            bind=False,
        )
        self._validate_global_clock(
            physics_step=frame.physics_step,
            simulation_time_s=frame.simulation_time_s,
            record_rejection=False,
        )
        self._validate_pump_sequence(
            frame,
            state,
            record_rejection=False,
        )
        return self._reservoir_withdrawal_ml(frame, state)

    def _ingest(
        self,
        frame: _PumpEvidenceFrame,
        authority: _PumpAuthority,
    ) -> ResuscitationSnapshot:
        if authority is not self._authority:
            self._rejected_frames += 1
            raise PermissionError("resuscitation effects accept scene evidence only")
        self._validate_runtime_identity(
            episode_id=frame.episode_id,
            environment_id=frame.environment_id,
            bind=True,
        )
        self._validate_global_clock(
            physics_step=frame.physics_step,
            simulation_time_s=frame.simulation_time_s,
            record_rejection=True,
        )
        state = self._channels[frame.channel_id]
        self._validate_pump_sequence(
            frame,
            state,
            record_rejection=True,
        )

        self._physics_step = max(self._physics_step, frame.physics_step)
        self._simulation_time_s = max(
            self._simulation_time_s,
            frame.simulation_time_s,
        )
        self._evidence_frames += 1
        if state.last_delivery_coordinate_m is None or state.last_reservoir_mass_g is None:
            state.last_delivery_coordinate_m = frame.delivery_coordinate_m
            state.last_reservoir_mass_g = frame.reservoir_mass_g
            state.last_physics_step = frame.physics_step
            state.last_simulation_time_s = frame.simulation_time_s
            state.last_evidence_digest_sha256 = (
                frame.evidence_digest_sha256
            )
            state.last_attachment_prim_ids = (
                frame.access_attachment_prim_ids
            )
            state.last_topology_revision = frame.topology_revision
            return self.snapshot()

        plunger_delta_m = max(
            0.0,
            frame.delivery_coordinate_m - state.last_delivery_coordinate_m,
        )
        plunger_volume_ml = (
            plunger_delta_m
            / self.calibration.plunger_stroke_m
            * self._channel_volume_per_stroke(frame.channel_id)
        )
        flow_volume_ml = frame.downstream_flow_ml_s * frame.dt_s
        vascular_inflow_volume_ml = (
            frame.vascular_inflow_ml_s * frame.dt_s
        )
        reservoir_withdrawal_ml = self._reservoir_withdrawal_ml(
            frame,
            state,
        )
        supported_delivery_ml = min(
            plunger_volume_ml,
            flow_volume_ml,
            vascular_inflow_volume_ml,
            reservoir_withdrawal_ml,
        )
        pressure_valid = (
            self.calibration.minimum_line_pressure_kpa
            <= frame.line_pressure_kpa
            <= self.calibration.maximum_line_pressure_kpa
        )
        connected = bool(frame.access_attachment_prim_ids)
        delivered_ml = (
            supported_delivery_ml
            if connected and pressure_valid
            else 0.0
        )
        wasted_ml = reservoir_withdrawal_ml - delivered_ml
        measured_extravasated_ml = min(
            max(0.0, flow_volume_ml - vascular_inflow_volume_ml),
            wasted_ml,
        )
        unreconciled_ml = max(
            0.0,
            wasted_ml - measured_extravasated_ml,
        )
        state.withdrawn_from_reservoir_ml += reservoir_withdrawal_ml
        state.delivered_to_patient_ml += delivered_ml
        state.wasted_or_extravasated_ml += wasted_ml
        state.measured_extravasated_ml += measured_extravasated_ml
        state.line_storage_or_unreconciled_ml += unreconciled_ml
        state.remaining_inventory_ml -= reservoir_withdrawal_ml
        if (
            connected
            and frame.line_pressure_kpa
            > self.calibration.maximum_line_pressure_kpa
        ):
            overload = _clamp(
                (frame.line_pressure_kpa - self.calibration.maximum_line_pressure_kpa)
                / (
                    self.calibration.hard_line_pressure_kpa
                    - self.calibration.maximum_line_pressure_kpa
                )
            )
            state.pressure_damage_fraction = _clamp(
                state.pressure_damage_fraction + overload * frame.dt_s * 0.04
            )
        state.last_delivery_coordinate_m = frame.delivery_coordinate_m
        state.last_reservoir_mass_g = frame.reservoir_mass_g
        state.last_physics_step = frame.physics_step
        state.last_simulation_time_s = frame.simulation_time_s
        state.last_evidence_digest_sha256 = frame.evidence_digest_sha256
        state.last_attachment_prim_ids = frame.access_attachment_prim_ids
        state.last_topology_revision = frame.topology_revision
        return self.snapshot()

    def _ingest_ventilation(
        self,
        frame: _VentilationEvidenceFrame,
        authority: _PumpAuthority,
    ) -> ResuscitationSnapshot:
        if authority is not self._authority:
            self._rejected_frames += 1
            raise PermissionError("ventilation effects accept scene evidence only")
        self._validate_runtime_identity(
            episode_id=frame.episode_id,
            environment_id=frame.environment_id,
            bind=True,
        )
        self._validate_global_clock(
            physics_step=frame.physics_step,
            simulation_time_s=frame.simulation_time_s,
            record_rejection=True,
        )
        state = self._ventilation
        if frame.physics_step <= state.last_physics_step:
            self._rejected_frames += 1
            raise ValueError("ventilation evidence must use a strictly increasing step")
        if frame.simulation_time_s <= state.last_simulation_time_s and state.last_physics_step >= 0:
            self._rejected_frames += 1
            raise ValueError("ventilation evidence must use increasing simulation time")
        cfg = self.calibration
        self._physics_step = max(self._physics_step, frame.physics_step)
        self._simulation_time_s = max(
            self._simulation_time_s,
            frame.simulation_time_s,
        )
        self._evidence_frames += 1
        state.airway_connected = bool(frame.airway_attachment_prim_ids)
        valve_fraction = _clamp(
            frame.valve_opening_angle_deg
            / cfg.ventilation_valve_full_open_deg
        )
        net_flow_l_min = frame.airway_interface_flow_l_min
        leaked_flow_l_min = max(
            0.0,
            frame.inspiratory_flow_l_min
            - frame.airway_interface_flow_l_min,
        )
        state.circuit_leak_fraction = _clamp(
            leaked_flow_l_min
            / max(frame.inspiratory_flow_l_min, 1.0e-9)
        )
        chest_fraction = _clamp(frame.chest_excursion_m / cfg.ventilation_target_chest_excursion_m)
        pressure_valid = (
            cfg.ventilation_minimum_airway_pressure_cmh2o
            <= frame.airway_pressure_cmh2o
            <= cfg.ventilation_maximum_airway_pressure_cmh2o
        )
        delivery_fraction = (
            min(valve_fraction, chest_fraction)
            if state.airway_connected and pressure_valid
            else 0.0
        )
        instantaneous_effective_flow_l_min = (
            net_flow_l_min * delivery_fraction
        )
        reporting_alpha = 1.0 - math.exp(
            -frame.dt_s / cfg.ventilation_reporting_window_s
        )
        state.effective_minute_ventilation_l_min += (
            reporting_alpha
            * (
                instantaneous_effective_flow_l_min
                - state.effective_minute_ventilation_l_min
            )
        )
        state.delivered_fio2_fraction = (
            0.21 + (frame.measured_fio2_fraction - 0.21) * delivery_fraction
        )
        state.airway_pressure_cmh2o = frame.airway_pressure_cmh2o
        state.chest_excursion_m = frame.chest_excursion_m
        state.cumulative_effective_ventilation_l += (
            instantaneous_effective_flow_l_min * frame.dt_s / 60.0
        )
        if (
            state.airway_connected
            and frame.airway_pressure_cmh2o
            > cfg.ventilation_maximum_airway_pressure_cmh2o
        ):
            overload = _clamp(
                (frame.airway_pressure_cmh2o - cfg.ventilation_maximum_airway_pressure_cmh2o)
                / (
                    cfg.ventilation_hard_airway_pressure_cmh2o
                    - cfg.ventilation_maximum_airway_pressure_cmh2o
                )
            )
            state.pressure_damage_fraction = _clamp(
                state.pressure_damage_fraction + overload * frame.dt_s * 0.03
            )
        state.last_physics_step = frame.physics_step
        state.last_simulation_time_s = frame.simulation_time_s
        state.last_evidence_digest_sha256 = frame.evidence_digest_sha256
        state.last_attachment_prim_ids = frame.airway_attachment_prim_ids
        state.last_topology_revision = frame.topology_revision
        return self.snapshot()

    def snapshot(self) -> ResuscitationSnapshot:
        channels = MappingProxyType(
            {
                channel_id: MappingProxyType(
                    {
                        "remaining_inventory_ml": state.remaining_inventory_ml,
                        "delivered_to_patient_ml": state.delivered_to_patient_ml,
                        "withdrawn_from_reservoir_ml": (state.withdrawn_from_reservoir_ml),
                        "wasted_or_extravasated_ml": (state.wasted_or_extravasated_ml),
                        "measured_extravasated_ml": (
                            state.measured_extravasated_ml
                        ),
                        "line_storage_or_unreconciled_ml": (
                            state.line_storage_or_unreconciled_ml
                        ),
                        "pressure_damage_fraction": (state.pressure_damage_fraction),
                        "last_physics_step": state.last_physics_step,
                        "last_evidence_digest_sha256": (
                            state.last_evidence_digest_sha256
                        ),
                        "last_attachment_prim_ids": (
                            state.last_attachment_prim_ids
                        ),
                        "last_topology_revision": (
                            state.last_topology_revision
                        ),
                    }
                )
                for channel_id, state in self._channels.items()
            }
        )
        cfg = self.calibration
        ventilation = MappingProxyType(
            {
                "airway_connected": self._ventilation.airway_connected,
                "effective_minute_ventilation_l_min": (
                    self._ventilation.effective_minute_ventilation_l_min
                ),
                "delivered_fio2_fraction": (
                    self._ventilation.delivered_fio2_fraction
                ),
                "airway_pressure_cmh2o": (
                    self._ventilation.airway_pressure_cmh2o
                ),
                "chest_excursion_m": self._ventilation.chest_excursion_m,
                "circuit_leak_fraction": (
                    self._ventilation.circuit_leak_fraction
                ),
                "pressure_damage_fraction": (
                    self._ventilation.pressure_damage_fraction
                ),
                "cumulative_effective_ventilation_l": (
                    self._ventilation.cumulative_effective_ventilation_l
                ),
                "last_physics_step": self._ventilation.last_physics_step,
                "last_evidence_digest_sha256": (
                    self._ventilation.last_evidence_digest_sha256
                ),
                "last_attachment_prim_ids": (
                    self._ventilation.last_attachment_prim_ids
                ),
                "last_topology_revision": (
                    self._ventilation.last_topology_revision
                ),
            }
        )
        effective_gain = (
            self._channels["blood_product"].delivered_to_patient_ml
            + cfg.crystalloid_intravascular_retention_fraction
            * self._channels["crystalloid"].delivered_to_patient_ml
        )
        return ResuscitationSnapshot(
            physics_step=self._physics_step,
            simulation_time_s=self._simulation_time_s,
            episode_id=self._episode_id,
            environment_id=self._environment_id,
            channels=channels,
            ventilation=ventilation,
            effective_circulating_volume_gain_ml=effective_gain,
            evidence_frames=self._evidence_frames,
            rejected_frames=self._rejected_frames,
            parameter_status=self.calibration.parameter_status,
            validation_status=(
                "engineering_only_not_clinically_validated_or_approved_"
                "for_patient_care"
            ),
        )


__all__ = [
    "ContactDrivenResuscitationEffects",
    "ResuscitationCalibration",
    "ResuscitationSnapshot",
]
