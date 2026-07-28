# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Shared causal mechanics foundations for DrAnmar simulation assets.

These modules expose mutable simulation state and scene-derived observations.
Default material values are explicitly provisional engineering seeds; this
package makes no clinical-validation or patient-care claim.
"""

from .clip import (
    ClipContactSample,
    ClipMaterial,
    ClipMechanics,
    ClipObservation,
    ClipState,
)
from .cohesive import (
    CohesiveInterface,
    CohesiveLaw,
    CohesiveResponse,
    CohesiveState,
)
from .contact import (
    ContactHistory,
    ContactKey,
    ContactResponse,
    ContactSample,
    ContactSnapshot,
    CoulombFrictionLaw,
    PersistentContactGraph,
    Vec3,
    regularized_coulomb_response,
)
from .rod import (
    CosseratRod,
    RodExternalLoad,
    RodMaterial,
    RodState,
    RodStepResult,
)
from .soft_tissue import (
    LayeredSoftTissue,
    Mat3,
    PunctureTract,
    SoftTissueExternalLoad,
    SoftTissueLayer,
    SoftTissueMesh,
    SoftTissueState,
    SoftTissueStepResult,
    Tet4,
    Tri3,
    WoundObservation,
)
from .vessel import (
    VesselContactLoad,
    VesselMaterial,
    VesselMechanics,
    VesselObservation,
    VesselSegmentState,
)


__all__ = [
    "ClipContactSample",
    "ClipMaterial",
    "ClipMechanics",
    "ClipObservation",
    "ClipState",
    "CohesiveInterface",
    "CohesiveLaw",
    "CohesiveResponse",
    "CohesiveState",
    "ContactHistory",
    "ContactKey",
    "ContactResponse",
    "ContactSample",
    "ContactSnapshot",
    "CosseratRod",
    "CoulombFrictionLaw",
    "LayeredSoftTissue",
    "Mat3",
    "PersistentContactGraph",
    "PunctureTract",
    "RodExternalLoad",
    "RodMaterial",
    "RodState",
    "RodStepResult",
    "SoftTissueExternalLoad",
    "SoftTissueLayer",
    "SoftTissueMesh",
    "SoftTissueState",
    "SoftTissueStepResult",
    "Tet4",
    "Tri3",
    "Vec3",
    "VesselContactLoad",
    "VesselMaterial",
    "VesselMechanics",
    "VesselObservation",
    "VesselSegmentState",
    "WoundObservation",
    "regularized_coulomb_response",
]
