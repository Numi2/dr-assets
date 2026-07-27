# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Catalog helpers for the canonical DrAnmar needle-ready tissue unit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from . import ORBITSURGICAL_ASSETS_DATA_DIR


CATALOG_SUBPATH: Final = "Props/SurgicalTissue/NeedleReadyTissueUnit"
ASSET_ROOT: Final = (
    Path(ORBITSURGICAL_ASSETS_DATA_DIR) / CATALOG_SUBPATH
)
TISSUE_UNIT_USD: Final = ASSET_ROOT / "needle_ready_tissue_unit.usda"
GEOMETRY_CONTRACT_PATH: Final = ASSET_ROOT / "geometry_contract.json"
PHYSICS_PROFILE_PATH: Final = ASSET_ROOT / "physics_profile.json"
QUALIFICATION_CONTRACT_PATH: Final = (
    ASSET_ROOT / "qualification_contract.json"
)
LOD_USD: Final[dict[str, Path]] = {
    lod: ASSET_ROOT / f"needle_ready_tissue_{lod}.usda"
    for lod in ("training", "contact", "validation")
}


def needle_ready_tissue_usd(lod: str = "contact") -> Path:
    """Return one explicit TetMesh LOD path.

    Direct LOD files are preferred for solver spawning because each stage
    contains exactly one TetMesh and requires no variant-selection mutation.
    """

    normalized = str(lod).strip().lower()
    try:
        return LOD_USD[normalized]
    except KeyError as error:
        raise ValueError(
            f"lod must be one of {sorted(LOD_USD)}, received {lod!r}"
        ) from error


def load_needle_ready_tissue_geometry_contract() -> dict[str, Any]:
    return json.loads(GEOMETRY_CONTRACT_PATH.read_text(encoding="utf-8"))


def load_needle_ready_tissue_physics_profile() -> dict[str, Any]:
    return json.loads(PHYSICS_PROFILE_PATH.read_text(encoding="utf-8"))


def load_needle_ready_tissue_qualification_contract() -> dict[str, Any]:
    return json.loads(
        QUALIFICATION_CONTRACT_PATH.read_text(encoding="utf-8")
    )


def make_needle_ready_tissue_cfg(
    *,
    lod: str = "contact",
    prim_path: str = "{ENV_REGEX_NS}/NeedleReadyTissue",
    position: tuple[float, float, float] = (0.0, 0.0, 0.08),
    youngs_modulus_pa: float = 180_000.0,
    poissons_ratio: float = 0.47,
    density_kg_m3: float = 1_050.0,
):
    """Build the current Isaac Lab Newton deformable configuration lazily."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import DeformableObjectCfg  # type: ignore
    from isaaclab_newton.sim.schemas import (  # type: ignore
        NewtonDeformableBodyPropertiesCfg,
    )
    from isaaclab_newton.sim.spawners.materials import (  # type: ignore
        NewtonDeformableBodyMaterialCfg,
    )

    physics_profile = load_needle_ready_tissue_physics_profile()
    intact = physics_profile["intact_deformation"]
    poisson = float(poissons_ratio)
    if not -0.999 < poisson < 0.5:
        raise ValueError("poissons_ratio must be between -0.999 and 0.5")
    youngs = float(youngs_modulus_pa)
    shear_modulus = youngs / (2.0 * (1.0 + poisson))
    lame_first_parameter = youngs * poisson / (
        (1.0 + poisson) * (1.0 - 2.0 * poisson)
    )
    material = NewtonDeformableBodyMaterialCfg(
        density=float(density_kg_m3),
        particle_radius=float(
            intact["particle_radius_m_by_lod"][str(lod).strip().lower()]
        ),
        k_mu=shear_modulus,
        k_lambda=lame_first_parameter,
        k_damp=float(intact["newton_k_damp_adapter_seed"]),
    )
    return DeformableObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(needle_ready_tissue_usd(lod).resolve()),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            physics_material=material,
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=position),
    )


__all__ = [
    "ASSET_ROOT",
    "CATALOG_SUBPATH",
    "GEOMETRY_CONTRACT_PATH",
    "LOD_USD",
    "PHYSICS_PROFILE_PATH",
    "QUALIFICATION_CONTRACT_PATH",
    "TISSUE_UNIT_USD",
    "load_needle_ready_tissue_geometry_contract",
    "load_needle_ready_tissue_physics_profile",
    "load_needle_ready_tissue_qualification_contract",
    "make_needle_ready_tissue_cfg",
    "needle_ready_tissue_usd",
]
