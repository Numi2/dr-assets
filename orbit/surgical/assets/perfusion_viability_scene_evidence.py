# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Prim-bound evidence contracts for perfusion-assessment runtimes.

This module deliberately does not implement an Isaac, RTX, ultrasound, Doppler,
or DynamicSurgicalPatient sampling bridge.  It defines the fail-closed ingress
that such a bridge must populate from post-physics records.  Synthetic vascular
and modality generators do not satisfy this contract.

The resulting scores remain provisional simulation-training outputs.  They are
not clinically validated, physically calibrated, or approved for patient-care
use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from .scene_evidence import (
    SceneEvidenceEnvelope,
    SceneEvidenceRegistry,
    SceneEvidenceSource,
)


PERFUSION_SCENE_ADAPTER_ID = "dr.anmar.perfusion.native-scene-bridge"
PERFUSION_SCENE_ADAPTER_VERSION = "1"

PERFUSION_SCAN_CHANNELS = (
    "rgb_left",
    "rgb_right",
    "nir_icg",
    "laser_speckle",
    "thermal",
    "surface_oxygenation",
    "depth",
    "doppler",
    "ultrasound",
)

_MAP_CHANNELS = frozenset(
    {
        "nir_icg",
        "laser_speckle",
        "thermal",
        "surface_oxygenation",
        "doppler",
        "ultrasound",
    }
)
_IMAGE_CHANNELS = frozenset({"rgb_left", "rgb_right"})
_PATIENT_SOURCE_ID = "perfusion.scan.dynamic_patient"


def _required_text(value: str, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _scene_path(value: str, label: str) -> str:
    result = _required_text(value, label)
    if not result.startswith("/"):
        raise ValueError(f"{label} must be an absolute scene path")
    return result.rstrip("/") or "/"


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _is_at_or_below(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _array_digest(value: Any) -> str:
    """Return a shape/dtype/value digest for one native bridge payload."""

    import numpy as np

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("perfusion evidence payloads cannot use object dtype")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(tuple(int(item) for item in contiguous.shape)).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _composite_digest(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_array_digest(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def payload_digest_token(value: Any) -> str:
    """Return the raw-sample token required for a payload measurement."""

    return f"payload-sha256:{_array_digest(value)}"


@dataclass(frozen=True, init=False)
class DynamicSurgicalPatientBinding:
    """Identity and immutable snapshot of one live patient runtime instance.

    Construction is intentionally limited to :meth:`capture`; callers cannot
    substitute a condition label, recovery scalar, or authored metadata for an
    actual :class:`DynamicSurgicalPatient` object.
    """

    patient_prim_path: str
    snapshot_time_s: float
    snapshot_digest_sha256: str
    runtime_instance_id: str
    _patient: Any = field(repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        patient: Any,
        *,
        patient_prim_path: str,
    ) -> "DynamicSurgicalPatientBinding":
        from .dynamic_abdominal_patient import DynamicSurgicalPatient

        if not isinstance(patient, DynamicSurgicalPatient):
            raise TypeError(
                "perfusion evidence requires a live DynamicSurgicalPatient"
            )
        root = _scene_path(patient_prim_path, "patient_prim_path")
        snapshot = patient.snapshot()
        snapshot_time = _finite(patient.time_s, "patient.time_s")
        digest = _snapshot_digest(snapshot)
        instance_material = f"{id(patient):x}:{digest}".encode("ascii")
        instance_id = hashlib.sha256(instance_material).hexdigest()
        result = object.__new__(cls)
        object.__setattr__(result, "patient_prim_path", root)
        object.__setattr__(result, "snapshot_time_s", snapshot_time)
        object.__setattr__(result, "snapshot_digest_sha256", digest)
        object.__setattr__(result, "runtime_instance_id", instance_id)
        object.__setattr__(result, "_patient", patient)
        return result

    @property
    def patient(self) -> Any:
        """Return the exact bound runtime; never a deserialized substitute."""

        return self._patient

    @property
    def snapshot_raw_sample_id(self) -> str:
        return f"patient-snapshot-sha256:{self.snapshot_digest_sha256}"

    @property
    def instance_raw_sample_id(self) -> str:
        return f"dynamic-patient-instance:{self.runtime_instance_id}"

    def validate_current(self) -> None:
        """Fail if the bound patient advanced after the snapshot was captured."""

        from .dynamic_abdominal_patient import DynamicSurgicalPatient

        if not isinstance(self._patient, DynamicSurgicalPatient):
            raise TypeError("bound patient is no longer a DynamicSurgicalPatient")
        if abs(float(self._patient.time_s) - self.snapshot_time_s) > 1.0e-9:
            raise ValueError(
                "DynamicSurgicalPatient advanced after evidence binding"
            )
        if _snapshot_digest(self._patient.snapshot()) != self.snapshot_digest_sha256:
            raise ValueError(
                "DynamicSurgicalPatient state changed after evidence binding"
            )

    def is_same_runtime_patient(
        self,
        other: "DynamicSurgicalPatientBinding",
    ) -> bool:
        return (
            isinstance(other, DynamicSurgicalPatientBinding)
            and self._patient is other._patient
            and self.patient_prim_path == other.patient_prim_path
        )


@dataclass(frozen=True)
class PerfusionSceneChannelSource:
    channel: str
    evidence_source: SceneEvidenceSource
    target_prim_path: str
    attachment_prim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.channel not in PERFUSION_SCAN_CHANNELS:
            raise ValueError(f"unsupported perfusion channel {self.channel!r}")
        object.__setattr__(
            self,
            "target_prim_path",
            _scene_path(self.target_prim_path, "target_prim_path"),
        )
        attachments = tuple(
            sorted(
                _scene_path(item, "attachment_prim_id")
                for item in self.attachment_prim_ids
            )
        )
        if len(attachments) != len(set(attachments)):
            raise ValueError("attachment_prim_ids must be unique")
        if self.channel not in {"doppler", "ultrasound"} and attachments:
            raise ValueError(
                f"{self.channel} must not claim probe attachments"
            )
        object.__setattr__(self, "attachment_prim_ids", attachments)


@dataclass(frozen=True)
class PerfusionScanSceneSources:
    """Exact modality and patient prims registered at scene creation."""

    channels: tuple[PerfusionSceneChannelSource, ...]
    patient_source: SceneEvidenceSource
    patient_prim_path: str

    def __post_init__(self) -> None:
        root = _scene_path(self.patient_prim_path, "patient_prim_path")
        object.__setattr__(self, "patient_prim_path", root)
        channels = tuple(sorted(self.channels, key=lambda item: item.channel))
        names = tuple(item.channel for item in channels)
        if set(names) != set(PERFUSION_SCAN_CHANNELS):
            raise ValueError(
                "perfusion scan sources must register every canonical channel"
            )
        if len(names) != len(set(names)):
            raise ValueError("perfusion scan channels must be unique")
        source_ids = tuple(
            item.evidence_source.source_id for item in channels
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("perfusion scan source IDs must be unique")
        for item in channels:
            if not _is_at_or_below(item.target_prim_path, root):
                raise ValueError(
                    f"{item.channel} target is outside the bound patient prim"
                )
        if self.patient_source.source_id != _PATIENT_SOURCE_ID:
            raise ValueError("perfusion patient source has the wrong source_id")
        if self.patient_source.prim_path != root:
            raise ValueError(
                "perfusion patient source is not bound to the patient root"
            )
        object.__setattr__(self, "channels", channels)

    @property
    def registry(self) -> SceneEvidenceRegistry:
        return SceneEvidenceRegistry(
            tuple(item.evidence_source for item in self.channels)
            + (self.patient_source,)
        )

    def channel_source(self, channel: str) -> PerfusionSceneChannelSource:
        try:
            return next(item for item in self.channels if item.channel == channel)
        except StopIteration as error:
            raise KeyError(channel) from error


def make_perfusion_scan_scene_sources(
    *,
    patient_binding: DynamicSurgicalPatientBinding,
    source_prim_paths: Mapping[str, str],
    target_prim_paths: Mapping[str, str],
    calibration_profile_ids: Mapping[str, str],
    contact_attachment_prim_ids: Mapping[str, tuple[str, ...]],
) -> PerfusionScanSceneSources:
    """Register source paths supplied by the real environment integration.

    No paths or calibration profiles are guessed.  A fixture generator may not
    call this function as a substitute for a native sensor bridge.
    """

    required = set(PERFUSION_SCAN_CHANNELS)
    for name, values in (
        ("source_prim_paths", source_prim_paths),
        ("target_prim_paths", target_prim_paths),
        ("calibration_profile_ids", calibration_profile_ids),
    ):
        if set(values) != required:
            missing = sorted(required.difference(values))
            extra = sorted(set(values).difference(required))
            raise ValueError(
                f"{name} must exactly match canonical channels; "
                f"missing={missing}, extra={extra}"
            )
    if set(contact_attachment_prim_ids) != {"doppler", "ultrasound"}:
        raise ValueError(
            "contact_attachment_prim_ids must exactly register Doppler and "
            "ultrasound probe attachments"
        )
    channels = []
    for channel in PERFUSION_SCAN_CHANNELS:
        source_path = _scene_path(
            source_prim_paths[channel],
            f"{channel} source prim",
        )
        channels.append(
            PerfusionSceneChannelSource(
                channel=channel,
                evidence_source=SceneEvidenceSource(
                    source_id=f"perfusion.scan.{channel}",
                    prim_path=source_path,
                    quantity=f"{channel}_native_payload",
                    unit="native_sample",
                    coordinate_frame=source_path,
                    calibration_profile_id=_required_text(
                        calibration_profile_ids[channel],
                        f"{channel} calibration_profile_id",
                    ),
                ),
                target_prim_path=target_prim_paths[channel],
                attachment_prim_ids=(
                    contact_attachment_prim_ids[channel]
                    if channel in {"doppler", "ultrasound"}
                    else ()
                ),
            )
        )
    return PerfusionScanSceneSources(
        channels=tuple(channels),
        patient_source=SceneEvidenceSource(
            source_id=_PATIENT_SOURCE_ID,
            prim_path=patient_binding.patient_prim_path,
            quantity="dynamic_patient_snapshot_binding",
            unit="sha256",
            coordinate_frame=patient_binding.patient_prim_path,
            calibration_profile_id="dr.anmar.dynamic-patient-snapshot.v1",
        ),
        patient_prim_path=patient_binding.patient_prim_path,
    )


@dataclass(frozen=True)
class PerfusionObservationPayload:
    """Native bridge payloads used for one registered fusion interval."""

    rgb_left: Any
    rgb_right: Any
    nir_icg_intensity: Any
    nir_icg_extravascular: Any
    laser_speckle: Any
    thermal_c: Any
    surface_oxygenation_fraction: Any
    depth_m: Any
    doppler_speed_m_s: Any
    ultrasound_relative_lumen_signal: Any

    def for_channel(self, channel: str) -> Any:
        if channel == "nir_icg":
            return (
                self.nir_icg_intensity,
                self.nir_icg_extravascular,
            )
        mapping = {
            "rgb_left": self.rgb_left,
            "rgb_right": self.rgb_right,
            "laser_speckle": self.laser_speckle,
            "thermal": self.thermal_c,
            "surface_oxygenation": self.surface_oxygenation_fraction,
            "depth": self.depth_m,
            "doppler": self.doppler_speed_m_s,
            "ultrasound": self.ultrasound_relative_lumen_signal,
        }
        try:
            return mapping[channel]
        except KeyError as error:
            raise KeyError(channel) from error

    def digest_token_for_channel(self, channel: str) -> str:
        value = self.for_channel(channel)
        if channel == "nir_icg":
            digest = _composite_digest(*value)
        else:
            digest = _array_digest(value)
        return f"payload-sha256:{digest}"


@dataclass(frozen=True)
class PerfusionScanSceneEvidence:
    """One fused scan interval bound to native payloads and a live patient."""

    envelope: SceneEvidenceEnvelope
    sources: PerfusionScanSceneSources
    patient_binding: DynamicSurgicalPatientBinding
    payload: PerfusionObservationPayload

    def __post_init__(self) -> None:
        import numpy as np

        provenance = self.envelope.provenance
        if provenance.adapter_id != PERFUSION_SCENE_ADAPTER_ID:
            raise ValueError("perfusion scan evidence has the wrong adapter_id")
        if provenance.adapter_version != PERFUSION_SCENE_ADAPTER_VERSION:
            raise ValueError(
                "perfusion scan evidence has the wrong adapter_version"
            )
        if self.patient_binding.patient_prim_path != self.sources.patient_prim_path:
            raise ValueError("scan sources and patient binding disagree")
        self.patient_binding.validate_current()
        if (
            abs(provenance.simulation_time_s - self.patient_binding.snapshot_time_s)
            > 1.0e-9
        ):
            raise ValueError(
                "scan provenance time does not match DynamicSurgicalPatient time"
            )
        self.sources.registry.validate(self.envelope)
        measurements = {
            item.source_id: item for item in self.envelope.measurements
        }
        expected = {
            item.evidence_source.source_id for item in self.sources.channels
        } | {_PATIENT_SOURCE_ID}
        if set(measurements) != expected:
            raise ValueError(
                "perfusion scan envelope source set does not match registration"
            )
        patient_measurement = measurements[_PATIENT_SOURCE_ID]
        if patient_measurement.source_prim_path != self.sources.patient_prim_path:
            raise ValueError("patient snapshot measurement changed source prim")
        if patient_measurement.target_prim_path != self.sources.patient_prim_path:
            raise ValueError("patient snapshot measurement changed target prim")
        if patient_measurement.value != (self.patient_binding.snapshot_time_s,):
            raise ValueError("patient snapshot measurement changed capture time")
        required_patient_ids = {
            self.patient_binding.snapshot_raw_sample_id,
            self.patient_binding.instance_raw_sample_id,
        }
        if not required_patient_ids.issubset(
            patient_measurement.raw_sample_ids
        ):
            raise ValueError(
                "scan envelope is not bound to the exact patient snapshot"
            )

        map_shape: tuple[int, int] | None = None
        for channel_source in self.sources.channels:
            channel = channel_source.channel
            measurement = measurements[
                channel_source.evidence_source.source_id
            ]
            if measurement.target_prim_path != channel_source.target_prim_path:
                raise ValueError(
                    f"{channel} measurement changed its patient target prim"
                )
            if (
                measurement.attachment_prim_ids
                != channel_source.attachment_prim_ids
            ):
                raise ValueError(
                    f"{channel} measurement changed attachment identity"
                )
            if channel in {"doppler", "ultrasound"}:
                expected_pair = (
                    f"{channel_source.evidence_source.prim_path}|"
                    f"{channel_source.target_prim_path}"
                )
                if measurement.contact_pair_id != expected_pair:
                    raise ValueError(
                        f"{channel} measurement changed probe contact pair"
                    )
            elif measurement.contact_pair_id is not None:
                raise ValueError(
                    f"{channel} must not claim a contact-pair measurement"
                )
            if len(measurement.value) != 3:
                raise ValueError(
                    f"{channel} measurement must carry "
                    "(sample_time_s, valid_flag, registration_error_m)"
                )
            sample_time, valid_flag, registration_error = measurement.value
            if valid_flag not in (0.0, 1.0):
                raise ValueError(f"{channel} valid_flag must be zero or one")
            if registration_error < 0.0:
                raise ValueError(
                    f"{channel} registration error must be nonnegative"
                )
            expected_digest = self.payload.digest_token_for_channel(channel)
            if expected_digest not in measurement.raw_sample_ids:
                raise ValueError(
                    f"{channel} payload is not bound to its envelope measurement"
                )
            if len(measurement.raw_sample_ids) < 2:
                raise ValueError(
                    f"{channel} requires a native raw record ID in addition "
                    "to its payload digest"
                )
            if abs(sample_time - provenance.simulation_time_s) > 0.050:
                # Preserve the record for abstention, but do not allow a source
                # to masquerade as evidence from an unrelated interval.
                raise ValueError(
                    f"{channel} sample is outside the 50 ms fusion interval"
                )
            payload_values = self.payload.for_channel(channel)
            arrays = (
                tuple(np.asarray(item) for item in payload_values)
                if channel == "nir_icg"
                else (np.asarray(payload_values),)
            )
            if valid_flag == 1.0:
                for array in arrays:
                    if array.size == 0 or not np.isfinite(array).all():
                        raise ValueError(
                            f"valid {channel} payload must be finite and nonempty"
                        )
                if channel in _IMAGE_CHANNELS:
                    if arrays[0].ndim != 3 or arrays[0].shape[2] not in (3, 4):
                        raise ValueError(
                            f"valid {channel} payload must be an RGB/RGBA image"
                        )
                elif channel == "depth":
                    if arrays[0].ndim != 2:
                        raise ValueError(
                            "valid depth payload must be a two-dimensional map"
                        )
                elif channel in _MAP_CHANNELS:
                    for array in arrays:
                        if array.ndim != 2:
                            raise ValueError(
                                f"valid {channel} payload must be a 2-D map"
                            )
                        shape = tuple(int(item) for item in array.shape)
                        if map_shape is None:
                            map_shape = shape
                        elif shape != map_shape:
                            raise ValueError(
                                "registered perfusion maps disagree on shape"
                            )
        if map_shape is None:
            raise ValueError("scan contains no valid perfusion map")
        if map_shape != (4, 6):
            raise ValueError(
                "perfusion fusion contract requires the registered 4 x 6 "
                "tissue grid"
            )

    def measurement_for_channel(self, channel: str):
        source_id = self.sources.channel_source(
            channel
        ).evidence_source.source_id
        return next(
            item for item in self.envelope.measurements
            if item.source_id == source_id
        )

    def validate_payload_bindings(self) -> None:
        """Re-hash mutable array objects before every assessment."""

        for channel in PERFUSION_SCAN_CHANNELS:
            measurement = self.measurement_for_channel(channel)
            expected_digest = self.payload.digest_token_for_channel(channel)
            if expected_digest not in measurement.raw_sample_ids:
                raise ValueError(
                    f"{channel} payload changed after evidence capture"
                )

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256

    @property
    def valid_channels(self) -> tuple[str, ...]:
        return tuple(
            channel
            for channel in PERFUSION_SCAN_CHANNELS
            if self.measurement_for_channel(channel).value[1] == 1.0
        )

    @property
    def timestamp_skew_s(self) -> float:
        times = [
            self.measurement_for_channel(channel).value[0]
            for channel in PERFUSION_SCAN_CHANNELS
        ]
        return max(times) - min(times)

    @property
    def registration_error_m(self) -> float:
        return max(
            self.measurement_for_channel(channel).value[2]
            for channel in PERFUSION_SCAN_CHANNELS
        )


@dataclass(frozen=True)
class PerfusionInterventionSceneSources:
    """Exact mechanics sources required between two registered scans."""

    displacement_source: SceneEvidenceSource
    contact_force_source: SceneEvidenceSource
    patient_source: SceneEvidenceSource
    target_prim_path: str
    attachment_prim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        target = _scene_path(self.target_prim_path, "target_prim_path")
        attachments = tuple(
            sorted(
                _scene_path(item, "attachment_prim_id")
                for item in self.attachment_prim_ids
            )
        )
        if len(attachments) != len(set(attachments)):
            raise ValueError("attachment_prim_ids must be unique")
        object.__setattr__(self, "target_prim_path", target)
        object.__setattr__(self, "attachment_prim_ids", attachments)

    @property
    def registry(self) -> SceneEvidenceRegistry:
        return SceneEvidenceRegistry(
            (
                self.displacement_source,
                self.contact_force_source,
                self.patient_source,
            )
        )


def make_perfusion_intervention_scene_sources(
    *,
    patient_binding: DynamicSurgicalPatientBinding,
    tracking_prim_path: str,
    contact_prim_path: str,
    target_prim_path: str,
    attachment_prim_ids: tuple[str, ...],
    tracking_calibration_profile_id: str,
    force_calibration_profile_id: str,
) -> PerfusionInterventionSceneSources:
    """Register real post-physics mechanics sources without outcome scalars."""

    tracking_path = _scene_path(tracking_prim_path, "tracking_prim_path")
    contact_path = _scene_path(contact_prim_path, "contact_prim_path")
    target = _scene_path(target_prim_path, "target_prim_path")
    if not _is_at_or_below(target, patient_binding.patient_prim_path):
        raise ValueError("intervention target is outside the bound patient")
    return PerfusionInterventionSceneSources(
        displacement_source=SceneEvidenceSource(
            source_id="perfusion.intervention.tool_displacement_w_m",
            prim_path=tracking_path,
            quantity="post_physics_displacement_w",
            unit="m",
            coordinate_frame="world",
            calibration_profile_id=_required_text(
                tracking_calibration_profile_id,
                "tracking_calibration_profile_id",
            ),
        ),
        contact_force_source=SceneEvidenceSource(
            source_id="perfusion.intervention.contact_force_w_n",
            prim_path=contact_path,
            quantity="post_physics_contact_force_w",
            unit="N",
            coordinate_frame="world",
            calibration_profile_id=_required_text(
                force_calibration_profile_id,
                "force_calibration_profile_id",
            ),
        ),
        patient_source=SceneEvidenceSource(
            source_id="perfusion.intervention.dynamic_patient",
            prim_path=patient_binding.patient_prim_path,
            quantity="dynamic_patient_snapshot_binding",
            unit="sha256",
            coordinate_frame=patient_binding.patient_prim_path,
            calibration_profile_id="dr.anmar.dynamic-patient-snapshot.v1",
        ),
        target_prim_path=target,
        attachment_prim_ids=attachment_prim_ids,
    )


@dataclass(frozen=True)
class PerfusionInterventionSceneEvidence:
    """Observed mechanics interval; it does not contain a success scalar."""

    envelope: SceneEvidenceEnvelope
    sources: PerfusionInterventionSceneSources
    patient_binding: DynamicSurgicalPatientBinding

    def __post_init__(self) -> None:
        provenance = self.envelope.provenance
        if provenance.adapter_id != PERFUSION_SCENE_ADAPTER_ID:
            raise ValueError(
                "perfusion intervention evidence has the wrong adapter_id"
            )
        if provenance.adapter_version != PERFUSION_SCENE_ADAPTER_VERSION:
            raise ValueError(
                "perfusion intervention evidence has the wrong adapter_version"
            )
        self.patient_binding.validate_current()
        if (
            abs(provenance.simulation_time_s - self.patient_binding.snapshot_time_s)
            > 1.0e-9
        ):
            raise ValueError(
                "intervention provenance time does not match patient time"
            )
        if self.sources.patient_source.prim_path != (
            self.patient_binding.patient_prim_path
        ):
            raise ValueError("intervention sources changed patient binding")
        if not _is_at_or_below(
            self.sources.target_prim_path,
            self.patient_binding.patient_prim_path,
        ):
            raise ValueError("intervention target is outside bound patient")
        self.sources.registry.validate(self.envelope)
        measurements = {
            item.source_id: item for item in self.envelope.measurements
        }
        expected = {
            self.sources.displacement_source.source_id,
            self.sources.contact_force_source.source_id,
            self.sources.patient_source.source_id,
        }
        if set(measurements) != expected:
            raise ValueError(
                "perfusion intervention source set does not match registration"
            )
        displacement = measurements[
            self.sources.displacement_source.source_id
        ]
        force = measurements[self.sources.contact_force_source.source_id]
        patient = measurements[self.sources.patient_source.source_id]
        if len(displacement.value) != 3 or len(force.value) != 3:
            raise ValueError(
                "intervention displacement and force must be world 3-vectors"
            )
        for name, measurement in (
            ("displacement", displacement),
            ("contact force", force),
        ):
            if measurement.target_prim_path != self.sources.target_prim_path:
                raise ValueError(
                    f"{name} changed the registered patient target"
                )
            if (
                measurement.attachment_prim_ids
                != self.sources.attachment_prim_ids
            ):
                raise ValueError(
                    f"{name} changed live attachment identity"
                )
            if not measurement.raw_sample_ids:
                raise ValueError(f"{name} requires exact raw record identity")
        expected_pair = (
            f"{self.sources.contact_force_source.prim_path}|"
            f"{self.sources.target_prim_path}"
        )
        if force.contact_pair_id != expected_pair:
            raise ValueError(
                "intervention force does not name the registered contact pair"
            )
        if patient.source_prim_path != self.patient_binding.patient_prim_path:
            raise ValueError("intervention patient source changed prim")
        if patient.target_prim_path != self.patient_binding.patient_prim_path:
            raise ValueError("intervention patient target changed prim")
        if patient.value != (self.patient_binding.snapshot_time_s,):
            raise ValueError("intervention patient capture time changed")
        required_patient_ids = {
            self.patient_binding.snapshot_raw_sample_id,
            self.patient_binding.instance_raw_sample_id,
        }
        if not required_patient_ids.issubset(patient.raw_sample_ids):
            raise ValueError(
                "intervention is not bound to the exact patient snapshot"
            )

    def _measurement(self, source_id: str):
        return next(
            item for item in self.envelope.measurements
            if item.source_id == source_id
        )

    @property
    def displacement_m(self) -> float:
        values = self._measurement(
            self.sources.displacement_source.source_id
        ).value
        return math.sqrt(sum(value * value for value in values))

    @property
    def contact_force_n(self) -> float:
        values = self._measurement(
            self.sources.contact_force_source.source_id
        ).value
        return math.sqrt(sum(value * value for value in values))

    @property
    def evidence_digest_sha256(self) -> str:
        return self.envelope.digest_sha256


__all__ = [
    "DynamicSurgicalPatientBinding",
    "PERFUSION_SCAN_CHANNELS",
    "PERFUSION_SCENE_ADAPTER_ID",
    "PERFUSION_SCENE_ADAPTER_VERSION",
    "PerfusionInterventionSceneEvidence",
    "PerfusionInterventionSceneSources",
    "PerfusionObservationPayload",
    "PerfusionScanSceneEvidence",
    "PerfusionScanSceneSources",
    "PerfusionSceneChannelSource",
    "make_perfusion_intervention_scene_sources",
    "make_perfusion_scan_scene_sources",
    "payload_digest_token",
]
