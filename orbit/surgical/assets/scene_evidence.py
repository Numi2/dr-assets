# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Immutable provenance records for post-physics scene evidence.

This module contains data contracts and registry validation only.  It does not
provide a public ``publish(values)`` adapter that could be used to write patient
outcomes.  Environment integrations are expected to sample registered prims
after physics and submit the resulting envelope directly to their owning
mechanics subsystem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Iterable, Mapping


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


def _finite(value: float, name: str) -> float:
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be finite")
    return rendered


@dataclass(frozen=True)
class SceneEvidenceSource:
    """One quantity registered against an exact scene prim."""

    source_id: str
    prim_path: str
    quantity: str
    unit: str
    coordinate_frame: str
    calibration_profile_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )
        object.__setattr__(
            self,
            "prim_path",
            _scene_path(self.prim_path, "prim_path"),
        )
        for name in (
            "quantity",
            "unit",
            "coordinate_frame",
            "calibration_profile_id",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name),
            )


@dataclass(frozen=True)
class SceneMeasurement:
    """A raw scalar or vector read from one registered source."""

    source_id: str
    value: tuple[float, ...]
    source_prim_path: str
    target_prim_path: str | None = None
    contact_pair_id: str | None = None
    attachment_prim_ids: tuple[str, ...] = ()
    raw_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _required_text(self.source_id, "source_id"),
        )
        values = tuple(
            _finite(component, "measurement component")
            for component in self.value
        )
        if not values:
            raise ValueError("measurement value must not be empty")
        object.__setattr__(self, "value", values)
        object.__setattr__(
            self,
            "source_prim_path",
            _scene_path(self.source_prim_path, "source_prim_path"),
        )
        if self.target_prim_path is not None:
            object.__setattr__(
                self,
                "target_prim_path",
                _scene_path(self.target_prim_path, "target_prim_path"),
            )
        if self.contact_pair_id is not None:
            object.__setattr__(
                self,
                "contact_pair_id",
                _required_text(self.contact_pair_id, "contact_pair_id"),
            )
        attachment_ids = tuple(
            sorted(
                _scene_path(path, "attachment_prim_id")
                for path in self.attachment_prim_ids
            )
        )
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("attachment_prim_ids must not contain duplicates")
        object.__setattr__(self, "attachment_prim_ids", attachment_ids)
        raw_sample_ids = tuple(
            sorted(
                _required_text(sample_id, "raw_sample_id")
                for sample_id in self.raw_sample_ids
            )
        )
        if len(raw_sample_ids) != len(set(raw_sample_ids)):
            raise ValueError("raw_sample_ids must not contain duplicates")
        object.__setattr__(self, "raw_sample_ids", raw_sample_ids)


@dataclass(frozen=True)
class EvidenceProvenance:
    """Identity and clock information for one post-physics interval."""

    episode_id: str
    environment_id: str
    physics_step: int
    simulation_time_s: float
    dt_s: float
    adapter_id: str
    adapter_version: str
    topology_revision: str

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "environment_id",
            "adapter_id",
            "adapter_version",
            "topology_revision",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name),
            )
        if self.physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        simulation_time = _finite(
            self.simulation_time_s,
            "simulation_time_s",
        )
        dt = _finite(self.dt_s, "dt_s")
        if simulation_time < 0.0:
            raise ValueError("simulation_time_s must be nonnegative")
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")
        object.__setattr__(self, "simulation_time_s", simulation_time)
        object.__setattr__(self, "dt_s", dt)


@dataclass(frozen=True)
class SceneEvidenceEnvelope:
    """Immutable, digestible measurements from one physics interval."""

    provenance: EvidenceProvenance
    measurements: tuple[SceneMeasurement, ...]

    def __post_init__(self) -> None:
        measurements = tuple(
            sorted(self.measurements, key=lambda item: item.source_id)
        )
        if not measurements:
            raise ValueError("scene evidence requires at least one measurement")
        source_ids = [item.source_id for item in measurements]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "a scene evidence envelope may contain each source only once"
            )
        object.__setattr__(self, "measurements", measurements)

    @property
    def digest_sha256(self) -> str:
        canonical = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


class SceneEvidenceRegistry:
    """Validate that evidence names the prims registered at scene creation."""

    def __init__(self, sources: Iterable[SceneEvidenceSource] = ()) -> None:
        self._sources: dict[str, SceneEvidenceSource] = {}
        for source in sources:
            self.register(source)

    def register(self, source: SceneEvidenceSource) -> None:
        if source.source_id in self._sources:
            raise ValueError(
                f"scene evidence source {source.source_id!r} is already registered"
            )
        self._sources[source.source_id] = source

    @property
    def sources(self) -> Mapping[str, SceneEvidenceSource]:
        return MappingProxyType(dict(self._sources))

    def validate(self, envelope: SceneEvidenceEnvelope) -> None:
        for measurement in envelope.measurements:
            try:
                source = self._sources[measurement.source_id]
            except KeyError as error:
                raise ValueError(
                    f"unregistered scene evidence source "
                    f"{measurement.source_id!r}"
                ) from error
            if measurement.source_prim_path != source.prim_path:
                raise ValueError(
                    f"scene evidence source {measurement.source_id!r} "
                    "does not match its registered prim"
                )


__all__ = [
    "EvidenceProvenance",
    "SceneEvidenceEnvelope",
    "SceneEvidenceRegistry",
    "SceneEvidenceSource",
    "SceneMeasurement",
]
