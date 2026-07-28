# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Single-owner blood and fluid authority for DrAnmar patient runtimes.

The authority deliberately owns quantities that must be conserved across
procedure modules.  Procedure-specific models may request a debit or credit,
but they must not keep an independent blood-volume or haemoglobin ledger.

This remains a research engineering model.  The default product composition
and crystalloid distribution are explicit provisional parameters, not clinical
reference values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Mapping


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


def _fraction(value: float, name: str) -> float:
    rendered = _finite(value, name)
    if not 0.0 <= rendered <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return rendered


@dataclass(frozen=True)
class PatientFluidSnapshot:
    """Immutable conserved-fluid view shared with procedure modules."""

    time_s: float
    baseline_blood_volume_ml: float
    intravascular_volume_ml: float
    interstitial_volume_ml: float
    plasma_excess_ml: float
    interstitial_excess_ml: float
    hemoglobin_mass_g: float
    hemoglobin_g_dl: float
    blood_volume_fraction: float
    crystalloid_input_ml: float
    colloid_input_ml: float
    transfused_red_cell_ml: float
    transfused_hemoglobin_g: float
    cumulative_blood_loss_ml: float
    cumulative_hemoglobin_loss_g: float
    urine_output_ml: float
    bile_output_ml: float
    suction_output_ml: float
    irrigation_input_ml: float
    irrigation_recovered_ml: float


class PatientAuthority:
    """Own conserved patient blood composition and fluid ledgers.

    Haemoglobin is represented as mass.  Acute blood loss removes circulating
    volume and haemoglobin at the current concentration, so haemoglobin
    concentration does not spuriously fall from haemorrhage alone.  Crystalloid
    changes volume without adding haemoglobin and therefore produces dilution.
    """

    def __init__(
        self,
        *,
        baseline_blood_volume_ml: float,
        baseline_hemoglobin_g_dl: float,
        interstitial_volume_ml: float = 11_000.0,
        crystalloid_intravascular_fraction: float = 0.24,
        transfusion_product_hemoglobin_g_dl: float = 20.0,
        plasma_to_interstitial_rate_s: float = 1.0 / 1_200.0,
        interstitial_to_plasma_rate_s: float = 1.0 / 3_600.0,
        lymph_return_rate_s: float = 1.0 / 7_200.0,
    ) -> None:
        baseline_volume = _nonnegative(
            baseline_blood_volume_ml,
            "baseline_blood_volume_ml",
        )
        if baseline_volume <= 0.0:
            raise ValueError("baseline_blood_volume_ml must be positive")
        baseline_hb = _nonnegative(
            baseline_hemoglobin_g_dl,
            "baseline_hemoglobin_g_dl",
        )
        if baseline_hb <= 0.0:
            raise ValueError("baseline_hemoglobin_g_dl must be positive")

        self._baseline_blood_volume_ml = baseline_volume
        self._intravascular_volume_ml = baseline_volume
        self._baseline_interstitial_volume_ml = _nonnegative(
            interstitial_volume_ml,
            "interstitial_volume_ml",
        )
        self._interstitial_volume_ml = self._baseline_interstitial_volume_ml
        self._plasma_excess_ml = 0.0
        self._interstitial_excess_ml = 0.0
        self._hemoglobin_mass_g = baseline_hb * baseline_volume / 100.0
        self._crystalloid_intravascular_fraction = _fraction(
            crystalloid_intravascular_fraction,
            "crystalloid_intravascular_fraction",
        )
        self._transfusion_product_hemoglobin_g_dl = _nonnegative(
            transfusion_product_hemoglobin_g_dl,
            "transfusion_product_hemoglobin_g_dl",
        )
        self._plasma_to_interstitial_rate_s = _nonnegative(
            plasma_to_interstitial_rate_s,
            "plasma_to_interstitial_rate_s",
        )
        self._interstitial_to_plasma_rate_s = _nonnegative(
            interstitial_to_plasma_rate_s,
            "interstitial_to_plasma_rate_s",
        )
        self._lymph_return_rate_s = _nonnegative(
            lymph_return_rate_s,
            "lymph_return_rate_s",
        )
        self._time_s = 0.0

        self._crystalloid_input_ml = 0.0
        self._colloid_input_ml = 0.0
        self._transfused_red_cell_ml = 0.0
        self._transfused_hemoglobin_g = 0.0
        self._cumulative_blood_loss_ml = 0.0
        self._cumulative_hemoglobin_loss_g = 0.0
        self._urine_output_ml = 0.0
        self._bile_output_ml = 0.0
        self._suction_output_ml = 0.0
        self._irrigation_input_ml = 0.0
        self._irrigation_recovered_ml = 0.0

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def baseline_blood_volume_ml(self) -> float:
        return self._baseline_blood_volume_ml

    @property
    def intravascular_volume_ml(self) -> float:
        return self._intravascular_volume_ml

    @property
    def interstitial_volume_ml(self) -> float:
        return self._interstitial_volume_ml

    @property
    def plasma_excess_ml(self) -> float:
        return self._plasma_excess_ml

    @property
    def interstitial_excess_ml(self) -> float:
        return self._interstitial_excess_ml

    @property
    def hemoglobin_mass_g(self) -> float:
        return self._hemoglobin_mass_g

    @property
    def hemoglobin_g_dl(self) -> float:
        if self._intravascular_volume_ml <= 1.0e-9:
            return 0.0
        return 100.0 * self._hemoglobin_mass_g / self._intravascular_volume_ml

    @property
    def blood_volume_fraction(self) -> float:
        return max(
            0.0,
            min(
                2.0,
                self._intravascular_volume_ml
                / self._baseline_blood_volume_ml,
            ),
        )

    @property
    def crystalloid_input_ml(self) -> float:
        return self._crystalloid_input_ml

    @property
    def colloid_input_ml(self) -> float:
        return self._colloid_input_ml

    @property
    def transfused_red_cell_ml(self) -> float:
        return self._transfused_red_cell_ml

    @property
    def transfused_hemoglobin_g(self) -> float:
        return self._transfused_hemoglobin_g

    @property
    def cumulative_blood_loss_ml(self) -> float:
        return self._cumulative_blood_loss_ml

    @property
    def cumulative_hemoglobin_loss_g(self) -> float:
        return self._cumulative_hemoglobin_loss_g

    @property
    def urine_output_ml(self) -> float:
        return self._urine_output_ml

    @property
    def bile_output_ml(self) -> float:
        return self._bile_output_ml

    @property
    def suction_output_ml(self) -> float:
        return self._suction_output_ml

    @property
    def irrigation_input_ml(self) -> float:
        return self._irrigation_input_ml

    @property
    def irrigation_recovered_ml(self) -> float:
        return self._irrigation_recovered_ml

    def advance_time(self, dt_s: float) -> float:
        dt = _finite(dt_s, "dt_s")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")

        plasma_to_interstitial_ml = self._plasma_excess_ml * (
            1.0
            - math.exp(-self._plasma_to_interstitial_rate_s * dt)
        )
        combined_return_rate = (
            self._interstitial_to_plasma_rate_s
            + self._lymph_return_rate_s
        )
        interstitial_to_plasma_ml = self._interstitial_excess_ml * (
            1.0 - math.exp(-combined_return_rate * dt)
        )
        net_to_plasma_ml = (
            interstitial_to_plasma_ml - plasma_to_interstitial_ml
        )
        self._plasma_excess_ml = max(
            0.0,
            self._plasma_excess_ml + net_to_plasma_ml,
        )
        self._interstitial_excess_ml = max(
            0.0,
            self._interstitial_excess_ml - net_to_plasma_ml,
        )
        self._intravascular_volume_ml = max(
            0.0,
            self._intravascular_volume_ml + net_to_plasma_ml,
        )
        self._interstitial_volume_ml = max(
            self._baseline_interstitial_volume_ml,
            self._baseline_interstitial_volume_ml
            + self._interstitial_excess_ml,
        )
        self._time_s += dt
        return self._time_s

    def withdraw_blood(self, requested_volume_ml: float) -> float:
        """Debit blood and proportional haemoglobin; return the actual debit."""

        requested = _nonnegative(
            requested_volume_ml,
            "requested blood-loss volume",
        )
        actual = min(requested, self._intravascular_volume_ml)
        starting_volume_ml = self._intravascular_volume_ml
        concentration_g_ml = (
            self._hemoglobin_mass_g / self._intravascular_volume_ml
            if self._intravascular_volume_ml > 1.0e-9
            else 0.0
        )
        hemoglobin_loss_g = min(
            self._hemoglobin_mass_g,
            concentration_g_ml * actual,
        )
        self._intravascular_volume_ml -= actual
        if starting_volume_ml > 1.0e-9:
            self._plasma_excess_ml *= max(
                0.0,
                1.0 - actual / starting_volume_ml,
            )
        self._hemoglobin_mass_g -= hemoglobin_loss_g
        self._cumulative_blood_loss_ml += actual
        self._cumulative_hemoglobin_loss_g += hemoglobin_loss_g
        return actual

    def infuse_crystalloid(self, volume_ml: float) -> float:
        volume = _nonnegative(volume_ml, "crystalloid volume")
        retained = self._crystalloid_intravascular_fraction * volume
        self._crystalloid_input_ml += volume
        self._intravascular_volume_ml += retained
        self._plasma_excess_ml += retained
        self._interstitial_excess_ml += volume - retained
        self._interstitial_volume_ml = (
            self._baseline_interstitial_volume_ml
            + self._interstitial_excess_ml
        )
        return retained

    def infuse_colloid(
        self,
        volume_ml: float,
        *,
        intravascular_fraction: float = 0.80,
    ) -> float:
        volume = _nonnegative(volume_ml, "colloid volume")
        retained_fraction = _fraction(
            intravascular_fraction,
            "colloid intravascular_fraction",
        )
        retained = retained_fraction * volume
        self._colloid_input_ml += volume
        self._intravascular_volume_ml += retained
        self._plasma_excess_ml += retained
        self._interstitial_excess_ml += volume - retained
        self._interstitial_volume_ml = (
            self._baseline_interstitial_volume_ml
            + self._interstitial_excess_ml
        )
        return retained

    def transfuse_blood(
        self,
        volume_ml: float,
        *,
        product_hemoglobin_g_dl: float | None = None,
    ) -> float:
        volume = _nonnegative(volume_ml, "transfusion volume")
        product_hb = (
            self._transfusion_product_hemoglobin_g_dl
            if product_hemoglobin_g_dl is None
            else _nonnegative(
                product_hemoglobin_g_dl,
                "product_hemoglobin_g_dl",
            )
        )
        added_hb_g = product_hb * volume / 100.0
        self._transfused_red_cell_ml += volume
        self._transfused_hemoglobin_g += added_hb_g
        self._intravascular_volume_ml += volume
        self._hemoglobin_mass_g += added_hb_g
        return volume

    def record_urine_output(self, requested_volume_ml: float) -> float:
        requested = _nonnegative(requested_volume_ml, "urine volume")
        actual = min(requested, self._intravascular_volume_ml)
        self._intravascular_volume_ml -= actual
        excess_removed = min(actual, self._plasma_excess_ml)
        self._plasma_excess_ml -= excess_removed
        self._urine_output_ml += actual
        return actual

    def record_bile_output(self, volume_ml: float) -> float:
        volume = _nonnegative(volume_ml, "bile volume")
        self._bile_output_ml += volume
        return volume

    def collect_suction(self, volume_ml: float) -> float:
        volume = _nonnegative(volume_ml, "suction volume")
        self._suction_output_ml += volume
        return volume

    def add_irrigation(self, volume_ml: float) -> float:
        volume = _nonnegative(volume_ml, "irrigation volume")
        self._irrigation_input_ml += volume
        return volume

    def recover_irrigation(self, requested_volume_ml: float) -> float:
        requested = _nonnegative(
            requested_volume_ml,
            "recovered irrigation volume",
        )
        available = max(
            0.0,
            self._irrigation_input_ml - self._irrigation_recovered_ml,
        )
        actual = min(requested, available)
        self._irrigation_recovered_ml += actual
        return actual

    def fluid_snapshot(self) -> PatientFluidSnapshot:
        return PatientFluidSnapshot(
            time_s=self._time_s,
            baseline_blood_volume_ml=self._baseline_blood_volume_ml,
            intravascular_volume_ml=self._intravascular_volume_ml,
            interstitial_volume_ml=self._interstitial_volume_ml,
            plasma_excess_ml=self._plasma_excess_ml,
            interstitial_excess_ml=self._interstitial_excess_ml,
            hemoglobin_mass_g=self._hemoglobin_mass_g,
            hemoglobin_g_dl=self.hemoglobin_g_dl,
            blood_volume_fraction=self.blood_volume_fraction,
            crystalloid_input_ml=self._crystalloid_input_ml,
            colloid_input_ml=self._colloid_input_ml,
            transfused_red_cell_ml=self._transfused_red_cell_ml,
            transfused_hemoglobin_g=self._transfused_hemoglobin_g,
            cumulative_blood_loss_ml=self._cumulative_blood_loss_ml,
            cumulative_hemoglobin_loss_g=self._cumulative_hemoglobin_loss_g,
            urine_output_ml=self._urine_output_ml,
            bile_output_ml=self._bile_output_ml,
            suction_output_ml=self._suction_output_ml,
            irrigation_input_ml=self._irrigation_input_ml,
            irrigation_recovered_ml=self._irrigation_recovered_ml,
        )

    def snapshot(self) -> Mapping[str, float | str]:
        payload: dict[str, float | str] = {
            "schema": "dr.anmar.patient-authority.v1",
            **asdict(self.fluid_snapshot()),
        }
        return MappingProxyType(payload)


__all__ = [
    "PatientAuthority",
    "PatientFluidSnapshot",
]
