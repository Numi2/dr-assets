# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar atraumatic surgical exposure robot.

The tool replaces the Panda hand at its verified stock joint frame. Its bilateral
compliant pads use independent, overlap-prioritized vertex attachments. The control helpers
maintain ROI exposure while enforcing provisional pad-force limits. This module
is an engineering research interface, not clinical control software.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math

from .atraumatic_exposure_scene_evidence import (
    AtraumaticExposureEvidenceCursor,
    AtraumaticExposureSceneEvidence,
)

CATALOG_SUBPATH = "Props/SurgicalExposure/AtraumaticExposureRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_atraumatic_exposure_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_atraumatic_exposure_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_atraumatic_exposure_tool_rigid_proxy.usda"
FENESTRATED_PAD_USD = ROOT / "dranmar_fenestrated_retraction_pad.usda"
MICROCUP_PAD_USD = ROOT / "dranmar_microcup_retraction_pad.usda"
TISSUE_DEMO_USD = ROOT / "dranmar_exposure_tissue_demo.usda"

VALID_PAD_TYPES = frozenset({"fenestrated", "microcup"})
CAPTURE_CELL_COUNT = 6

TOOL_JOINTS = {
    "left_carriage": "left_carriage_joint",
    "right_carriage": "right_carriage_joint",
    "left_lift": "left_lift_joint",
    "right_lift": "right_lift_joint",
    "left_pitch": "left_pitch_joint",
    "right_pitch": "right_pitch_joint",
    "left_compliance": "left_compliance_joint",
    "right_compliance": "right_compliance_joint",
}

TOOL_FRAME_PATHS = {
    "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
    "exposure_tcp": "Links/Mount/Frames/exposure_tcp",
    "roi_camera": "Links/Mount/Frames/roi_camera",
    "illumination_center": "Links/Mount/Frames/illumination_center",
    "exposure_center": "Links/Mount/Frames/exposure_center",
    "count_reference": "Links/Mount/Frames/count_reference",
    "left_pad_center": "Links/LeftPad/Frames/left_pad_center",
    "right_pad_center": "Links/RightPad/Frames/right_pad_center",
    "left_pad_normal": "Links/LeftPad/Frames/left_pad_normal",
    "right_pad_normal": "Links/RightPad/Frames/right_pad_normal",
    "left_force_sensor": "Links/LeftPad/Frames/left_force_sensor",
    "right_force_sensor": "Links/RightPad/Frames/right_force_sensor",
}
REGISTERED_CAMERA_FRAMES = ("roi_camera",)
for _side in ("left", "right"):
    for _index in range(CAPTURE_CELL_COUNT):
        TOOL_FRAME_PATHS[f"{_side}_capture_{_index:02d}"] = (
            f"Links/{_side.capitalize()}Pad/Frames/{_side}_capture_{_index:02d}"
        )


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown exposure-tool frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    """Return a native tensor from Isaac 6 tensor proxy objects when required."""
    return value.torch if hasattr(value, "torch") else value


def _xyzw_from_wxyz(orientation_wxyz) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in orientation_wxyz)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("orientation_wxyz must contain four finite values")
    if abs(math.sqrt(sum(value * value for value in values)) - 1.0) > 1.0e-4:
        raise ValueError("orientation_wxyz must be a unit quaternion")
    w, x, y, z = values
    return x, y, z, w


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_finite(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _deformable_attachment_body_path(stage: Any, prim_path: str) -> str:
    """Resolve an actor root to the exact mesh body used by attachments."""
    path = str(prim_path).rstrip("/")
    if not path.startswith("/") or not path:
        raise ValueError("deformable attachment source must be an absolute scene path")
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"No deformable attachment source at {path}")
    if prim.GetTypeName() == "Mesh":
        return path
    mesh_path = f"{path}/SimulationMesh"
    mesh = stage.GetPrimAtPath(mesh_path)
    if not mesh or not mesh.IsValid() or mesh.GetTypeName() != "Mesh":
        raise ValueError(
            f"Deformable actor {path} has no exact SimulationMesh attachment body"
        )
    return mesh_path


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarAtraumaticExposureTool",
    *,
    pad_type: str = "fenestrated",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    """Return a standalone Isaac Lab articulation configuration."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    _check(pad_type, VALID_PAD_TYPES, "pad_type")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"pad_type": pad_type},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=_xyzw_from_wxyz(orientation_wxyz),
            joint_pos={
                "left_carriage_joint": 0.0,
                "right_carriage_joint": 0.0,
                "left_lift_joint": -0.012,
                "right_lift_joint": -0.012,
                "left_pitch_joint": math.radians(58.0),
                "right_pitch_joint": math.radians(-58.0),
                "left_compliance_joint": 0.0,
                "right_compliance_joint": 0.0,
            },
        ),
        actuators={
            "lateral_retraction": ImplicitActuatorCfg(
                joint_names_expr=[".*_carriage_joint"],
                effort_limit_sim=95.0,
                velocity_limit_sim=0.12,
                stiffness=5200.0,
                damping=190.0,
            ),
            "independent_lift": ImplicitActuatorCfg(
                joint_names_expr=[".*_lift_joint"],
                effort_limit_sim=110.0,
                velocity_limit_sim=0.10,
                stiffness=6200.0,
                damping=210.0,
            ),
            "pad_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*_pitch_joint"],
                effort_limit_sim=7.0,
                velocity_limit_sim=2.0,
                stiffness=52.0,
                damping=2.5,
            ),
            "pad_compliance": ImplicitActuatorCfg(
                joint_names_expr=[".*_compliance_joint"],
                effort_limit_sim=16.0,
                velocity_limit_sim=0.08,
                stiffness=1250.0,
                damping=38.0,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarAtraumaticExposureProxy",
    *,
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_RIGID_PROXY_USD),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position, rot=_xyzw_from_wxyz(orientation_wxyz)
        ),
    )


def _spawn_single_franka_with_exposure_tool(
    prim_path: str,
    cfg: Any,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn stock Franka, remove Panda hand, and mount the exposure payload."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics

    robot = spawn_from_usd(prim_path, cfg, translation, orientation)
    stage = get_current_stage()
    names_to_disable = {
        "panda_hand_joint", "panda_hand", "panda_finger_joint1", "panda_finger_joint2",
        "panda_leftfinger", "panda_rightfinger",
    }
    robot_path = Sdf.Path(prim_path)
    hand_joint_prims = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_hand_joint"
    ]
    if len(hand_joint_prims) == 1:
        stock_hand_joint = UsdPhysics.Joint(hand_joint_prims[0])
        mount_body_paths = stock_hand_joint.GetBody0Rel().GetTargets()
        mount_local_pos0 = stock_hand_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        mount_local_rot0 = stock_hand_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
    else:
        link8_paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_link8"
        ]
        if len(link8_paths) != 1:
            raise RuntimeError(
                "Could not resolve the Franka hand mount from panda_hand_joint or panda_link8"
            )
        mount_body_paths = link8_paths
        mount_local_pos0 = Gf.Vec3f(0, 0, 0)
        half_angle = math.radians(-45.0) / 2.0
        mount_local_rot0 = Gf.Quatf(
            math.cos(half_angle), 0, 0, math.sin(half_angle)
        )
    if len(mount_body_paths) != 1 or not stage.GetPrimAtPath(mount_body_paths[0]).IsValid():
        raise RuntimeError(f"Invalid Franka hand mount target: {mount_body_paths}")

    candidate_paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in names_to_disable
    ]
    paths_to_disable = []
    for path in sorted(candidate_paths, key=lambda item: str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)

    tool_path = f"{prim_path}/DrAnmarAtraumaticExposureTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(tool_path, {"pad_type": cfg.pad_type})

    mount_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_exposure_mount_joint")
    mount_joint.CreateBody0Rel().SetTargets(mount_body_paths)
    mount_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    mount_joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    mount_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    mount_joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    mount_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_exposure_tool(
    prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs
):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_exposure_tool)(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


def make_franka_exposure_robot_cfg(
    *,
    prim_path: str = "/World/Robot",
    pad_type: str = "fenestrated",
):
    """Return the stock Isaac Lab Franka with the Panda hand replaced."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    _check(pad_type, VALID_PAD_TYPES, "pad_type")

    @configclass
    class FrankaExposureUsdCfg(sim_utils.UsdFileCfg):
        pad_type: str = "fenestrated"
        func = spawn_franka_with_exposure_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaExposureUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper": "Default", "Mesh": "Performance"},
        pad_type=pad_type,
        activate_contact_sensors=True,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {
        key: value for key, value in cfg.init_state.joint_pos.items() if "finger" not in key
    }
    cfg.init_state.joint_pos.update({
        "left_carriage_joint": 0.0,
        "right_carriage_joint": 0.0,
        "left_lift_joint": -0.012,
        "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(58.0),
        "right_pitch_joint": math.radians(-58.0),
        "left_compliance_joint": 0.0,
        "right_compliance_joint": 0.0,
    })
    cfg.actuators.pop("panda_hand", None)
    cfg.actuators.update({
        "exposure_lateral": ImplicitActuatorCfg(
            joint_names_expr=[".*_carriage_joint"], effort_limit_sim=95.0,
            velocity_limit_sim=0.12, stiffness=5200.0, damping=190.0,
        ),
        "exposure_lift": ImplicitActuatorCfg(
            joint_names_expr=[".*_lift_joint"], effort_limit_sim=110.0,
            velocity_limit_sim=0.10, stiffness=6200.0, damping=210.0,
        ),
        "exposure_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_pitch_joint"], effort_limit_sim=7.0,
            velocity_limit_sim=2.0, stiffness=52.0, damping=2.5,
        ),
        "exposure_compliance": ImplicitActuatorCfg(
            joint_names_expr=[".*_compliance_joint"], effort_limit_sim=16.0,
            velocity_limit_sim=0.08, stiffness=1250.0, damping=38.0,
        ),
    })
    return cfg


def spawn_exposure_tissue_demo(
    prim_path: str = "/World/DrAnmarExposureTissue",
    *,
    translation=(0.0, 0.0, 0.0),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(TISSUE_DEMO_USD))
    return cfg.func(
        prim_path, cfg, translation=translation,
        orientation=_xyzw_from_wxyz(orientation_wxyz),
    )


def _current_stage(stage=None):
    if stage is not None:
        return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def apply_exposure_tissue_surface_deformables(
    tissue_root_path: str = "/World/DrAnmarExposureTissue",
    *,
    stage=None,
    material_path: str = "/World/Materials/DrAnmarExposureTissueSurface",
    youngs_modulus_pa: float = 60_000.0,
    poissons_ratio: float = 0.45,
    surface_thickness_m: float = 0.006,
    density_kg_m3: float = 1_050.0,
    dynamic_friction: float = 0.58,
    elasticity_damping: float = 0.16,
    bend_damping: float = 0.14,
    self_collision: bool = True,
) -> dict[str, Any]:
    """Cook both portable flap meshes with the current surface-deformable API."""
    stage = _current_stage(stage)
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()
    prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    prim.GetAttribute("omniphysics:dynamicFriction").Set(float(dynamic_friction))
    prim.GetAttribute("omniphysics:density").Set(float(density_kg_m3))
    prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:youngsModulus").Set(float(youngs_modulus_pa))
    prim.GetAttribute("omniphysics:poissonsRatio").Set(float(poissons_ratio))
    prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:surfaceThickness").Set(float(surface_thickness_m))
    prim.GetAttribute("omniphysics:surfaceBendStiffness").Set(0.0)
    prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(float(elasticity_damping))
    prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(float(bend_damping))

    result = {"root_path": tissue_root_path, "material_path": material_path, "flaps": {}}
    for side in ("LeftFlap", "RightFlap"):
        actor_path = f"{tissue_root_path.rstrip('/')}/{side}"
        mesh_path = f"{actor_path}/SimulationMesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim or not mesh_prim.IsValid():
            raise ValueError(f"No exposure tissue mesh at {mesh_path}")
        success = deformableUtils.set_physics_surface_deformable_body(stage, mesh_prim.GetPath())
        if success is False:
            raise RuntimeError(f"PhysX could not create a surface deformable at {mesh_path}")
        mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        if mesh_prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
            mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        result["flaps"][side] = {"actor_path": actor_path, "mesh_path": mesh_path}
    result["parameters"] = {
        "youngs_modulus_pa": youngs_modulus_pa,
        "poissons_ratio": poissons_ratio,
        "surface_thickness_m": surface_thickness_m,
        "density_kg_m3": density_kg_m3,
        "dynamic_friction": dynamic_friction,
        "elasticity_damping": elasticity_damping,
        "bend_damping": bend_damping,
        "self_collision": self_collision,
        "status": "provisional_engineering_seed",
    }
    return result


def create_deformable_attachment(
    deformable_prim_path: str,
    rigid_prim_path: str,
    attachment_path: str,
    *,
    stage=None,
) -> str:
    """Create a verified rigid/deformable attachment across Isaac generations."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    stage = _current_stage(stage)
    deformable_prim_path = _deformable_attachment_body_path(
        stage,
        deformable_prim_path,
    )
    if stage.GetPrimAtPath(attachment_path).IsValid():
        remove_prims([attachment_path], stage=stage)

    prim_definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if prim_definition:
        deformable_prim = stage.GetPrimAtPath(deformable_prim_path)
        rigid_prim = stage.GetPrimAtPath(rigid_prim_path)
        mesh = UsdGeom.Mesh(deformable_prim)
        points = list(mesh.GetPointsAttr().Get() or [])
        if not deformable_prim.IsValid() or not mesh or not points:
            raise ValueError(f"Attachment source is not a populated mesh: {deformable_prim_path}")
        if not rigid_prim.IsValid() or not UsdGeom.Xformable(rigid_prim):
            raise ValueError(f"Attachment target is not xformable: {rigid_prim_path}")

        mesh_to_world = UsdGeom.Xformable(deformable_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rigid_to_world = UsdGeom.Xformable(rigid_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_to_rigid = rigid_to_world.GetInverse()
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        ).ComputeWorldBound(rigid_prim).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        center = (minimum + maximum) * 0.5
        margin = 0.0025
        ranked: list[tuple[float, int, Gf.Vec3d, bool]] = []
        for index, point in enumerate(points):
            world = mesh_to_world.Transform(Gf.Vec3d(point))
            delta = world - center
            distance_sq = float(Gf.Dot(delta, delta))
            overlaps = all(
                minimum[axis] - margin <= world[axis] <= maximum[axis] + margin
                for axis in range(3)
            )
            ranked.append((distance_sq, index, world, overlaps))
        ranked.sort(key=lambda item: item[0])
        selected = [item for item in ranked if item[3]][:12]
        if len(selected) < 4:
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_prim_path}, "
                f"target={rigid_prim_path}, overlapping={len(selected)}, "
                "required=4, overlap_margin_m=0.0025"
            )

        attachment = stage.DefinePrim(attachment_path, "OmniPhysicsVtxXformAttachment")
        attachment.CreateRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(deformable_prim_path)]
        )
        attachment.CreateRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(rigid_prim_path)]
        )
        attachment.CreateAttribute(
            "omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray
        ).Set(Vt.IntArray([item[1] for item in selected]))
        attachment.CreateAttribute(
            "omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray
        ).Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(world_to_rigid.Transform(item[2])) for item in selected]
            )
        )
        attachment.CreateAttribute(
            "omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)
        if (
            not attachment.IsValid()
            or attachment.GetTypeName() != "OmniPhysicsVtxXformAttachment"
            or not attachment.GetRelationship("omniphysics:src0").GetTargets()
            or not attachment.GetRelationship("omniphysics:src1").GetTargets()
        ):
            raise RuntimeError(f"Could not author current attachment schema at {attachment_path}")
        return "OmniPhysicsVtxXformAttachment"

    import omni.kit.commands

    def execute_and_verify(command: str, **kwargs) -> str:
        omni.kit.commands.execute(command, **kwargs)
        attachment = stage.GetPrimAtPath(attachment_path)
        if not attachment.IsValid():
            raise RuntimeError(f"{command} did not author {attachment_path}")
        return command

    try:
        return execute_and_verify(
            "CreateAutoDeformableAttachment",
            target_attachment_path=Sdf.Path(attachment_path),
            attachable0_path=Sdf.Path(deformable_prim_path),
            attachable1_path=Sdf.Path(rigid_prim_path),
        )
    except Exception as current_error:
        if stage.GetPrimAtPath(attachment_path).IsValid():
            remove_prims([attachment_path], stage=stage)
        try:
            return execute_and_verify(
                "CreatePhysicsAttachment",
                target_attachment_path=Sdf.Path(attachment_path),
                actor0_path=Sdf.Path(deformable_prim_path),
                actor1_path=Sdf.Path(rigid_prim_path),
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Could not create attachment {attachment_path}: "
                f"current={current_error!r}; legacy={legacy_error!r}"
            ) from legacy_error


def remove_prims(paths: Iterable[str], *, stage=None) -> None:
    stage = _current_stage(stage)
    normalized = tuple(dict.fromkeys(str(path) for path in paths))
    for path in normalized:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            removed = stage.RemovePrim(path)
            if removed is False:
                raise RuntimeError(f"Scene refused to remove prim {path}")
            remaining = stage.GetPrimAtPath(path)
            if remaining and remaining.IsValid():
                raise RuntimeError(f"Scene prim remained after removal: {path}")


def anchor_tissue_outer_bands(
    tissue_root_path: str = "/World/DrAnmarExposureTissue",
    *,
    stage=None,
) -> list[str]:
    """Attach one local outer-edge vertex cluster per flap to a fixture proxy."""
    stage = _current_stage(stage)
    attachments = []
    for side in ("Left", "Right"):
        path = f"{tissue_root_path}/Attachments/{side}OuterAnchor"
        create_deformable_attachment(
            f"{tissue_root_path}/{side}Flap",
            f"{tissue_root_path}/Fixture/{side}Anchor",
            path,
            stage=stage,
        )
        attachments.append(path)
    return attachments


@dataclass
class CaptureCell:
    side: str
    index: int
    attachment_path: str
    rigid_cell_path: str
    active: bool = True
    released_reason: str | None = None


@dataclass
class DistributedPadCaptureController:
    """Manage six independent tissue bonds per pad.

    Multiple small attachments distribute pad traction over the contact area.
    Exact evidence can release a local cell or an entire pad, after which the
    controller latches safe relief until a complete recapture is confirmed.
    This is a research proxy for local loss of contact, not a calibrated
    tissue-injury or vacuum model.
    """

    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    stage: Any = None
    soft_cell_release_force_n: float = 0.75
    hard_pad_release_force_n: float = 4.0
    max_cell_slip_speed_m_s: float = 0.012
    capture_linear_position_tolerance_m: float = 0.0015
    capture_angular_position_tolerance_rad: float = math.radians(1.0)
    minimum_capture_cell_normal_force_n: float = 0.02
    minimum_capture_cell_contact_area_m2: float = 1.0e-6
    capture_confirmation_intervals_required: int = 3
    capture_confirmation_min_dwell_s: float = 1.0 / 30.0
    capture_maximum_evidence_interval_s: float = 1.0 / 30.0
    evidence_cursor: AtraumaticExposureEvidenceCursor = field(
        default_factory=AtraumaticExposureEvidenceCursor
    )
    cells: list[CaptureCell] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    safety_latched: bool = field(default=True, init=False)
    latch_reason: str = field(default="capture_not_established", init=False)
    capture_epoch_confirmed: bool = field(default=False, init=False)
    _capture_preflight_digest_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_preflight_topology_revision: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_preflight_episode_id: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_preflight_environment_id: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_preflight_source_registration: Any = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_preflight_physics_step: int | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_preflight_simulation_time_s: float | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_confirmation_topology_revision: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_confirmation_interval_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _capture_confirmation_started_at_s: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        for name in ("tool_path", "left_tissue_path", "right_tissue_path"):
            value = str(getattr(self, name)).rstrip("/")
            if not value.startswith("/") or value == "":
                raise ValueError(f"{name} must be an absolute scene path")
            setattr(self, name, value)
        soft = _nonnegative_finite(
            self.soft_cell_release_force_n,
            "soft_cell_release_force_n",
        )
        hard = _nonnegative_finite(
            self.hard_pad_release_force_n,
            "hard_pad_release_force_n",
        )
        slip = _nonnegative_finite(
            self.max_cell_slip_speed_m_s,
            "max_cell_slip_speed_m_s",
        )
        linear_tolerance = _nonnegative_finite(
            self.capture_linear_position_tolerance_m,
            "capture_linear_position_tolerance_m",
        )
        angular_tolerance = _nonnegative_finite(
            self.capture_angular_position_tolerance_rad,
            "capture_angular_position_tolerance_rad",
        )
        minimum_normal_force = _nonnegative_finite(
            self.minimum_capture_cell_normal_force_n,
            "minimum_capture_cell_normal_force_n",
        )
        minimum_contact_area = _nonnegative_finite(
            self.minimum_capture_cell_contact_area_m2,
            "minimum_capture_cell_contact_area_m2",
        )
        confirmation_dwell = _nonnegative_finite(
            self.capture_confirmation_min_dwell_s,
            "capture_confirmation_min_dwell_s",
        )
        maximum_capture_interval = _nonnegative_finite(
            self.capture_maximum_evidence_interval_s,
            "capture_maximum_evidence_interval_s",
        )
        if (
            soft <= 0.0
            or hard <= soft
            or slip <= 0.0
            or linear_tolerance <= 0.0
            or angular_tolerance <= 0.0
            or minimum_normal_force <= 0.0
            or minimum_normal_force >= soft
            or minimum_contact_area <= 0.0
            or confirmation_dwell <= 0.0
            or maximum_capture_interval <= 0.0
        ):
            raise ValueError(
                "release thresholds require 0 < soft < hard, positive slip, "
                "positive linear/angular capture-position tolerances, "
                "nontrivial capture force/area floors, and positive "
                "confirmation dwell/maximum evidence interval"
            )
        if (
            isinstance(self.capture_confirmation_intervals_required, bool)
            or not isinstance(self.capture_confirmation_intervals_required, int)
            or self.capture_confirmation_intervals_required < 2
        ):
            raise ValueError(
                "capture_confirmation_intervals_required must be an integer >= 2"
            )
        self.soft_cell_release_force_n = soft
        self.hard_pad_release_force_n = hard
        self.max_cell_slip_speed_m_s = slip
        self.capture_linear_position_tolerance_m = linear_tolerance
        self.capture_angular_position_tolerance_rad = angular_tolerance
        self.minimum_capture_cell_normal_force_n = minimum_normal_force
        self.minimum_capture_cell_contact_area_m2 = minimum_contact_area
        self.capture_confirmation_min_dwell_s = confirmation_dwell
        self.capture_maximum_evidence_interval_s = maximum_capture_interval
        self.stage = _current_stage(self.stage)
        self.left_tissue_path = _deformable_attachment_body_path(
            self.stage,
            self.left_tissue_path,
        )
        self.right_tissue_path = _deformable_attachment_body_path(
            self.stage,
            self.right_tissue_path,
        )

    def _clear_capture_handshake(self) -> None:
        self._capture_preflight_digest_sha256 = None
        self._capture_preflight_topology_revision = None
        self._capture_preflight_episode_id = None
        self._capture_preflight_environment_id = None
        self._capture_preflight_source_registration = None
        self._capture_preflight_physics_step = None
        self._capture_preflight_simulation_time_s = None
        self._capture_confirmation_topology_revision = None
        self._capture_confirmation_interval_count = 0
        self._capture_confirmation_started_at_s = None

    def _require_current_capture_preflight(self) -> None:
        expected_cursor_state = (
            self._capture_preflight_episode_id,
            self._capture_preflight_environment_id,
            self._capture_preflight_topology_revision,
            self._capture_preflight_source_registration,
            self._capture_preflight_physics_step,
            self._capture_preflight_simulation_time_s,
        )
        current_cursor_state = (
            self.evidence_cursor.episode_id,
            self.evidence_cursor.environment_id,
            self.evidence_cursor.topology_revision,
            self.evidence_cursor.source_registration,
            self.evidence_cursor.last_physics_step,
            self.evidence_cursor.last_simulation_time_s,
        )
        if (
            self._capture_preflight_digest_sha256 is None
            or self._capture_preflight_digest_sha256
            not in self.evidence_cursor.consumed_digests
            or expected_cursor_state != current_cursor_state
        ):
            self._clear_capture_handshake()
            self.latch_reason = "capture_preflight_stale_requires_new_preflight"
            raise RuntimeError(
                "capture preflight is stale: episode, environment, source, "
                "topology, or evidence clock advanced before authoring"
            )

    def capture(self) -> list[CaptureCell]:
        if (
            self.latch_reason
            != "capture_preflight_confirmed_pending_authoring"
            or self._capture_preflight_digest_sha256 is None
            or self._capture_preflight_topology_revision is None
        ):
            raise RuntimeError(
                "capture authoring requires a fresh measured-pose/contact "
                "preflight from the current episode"
            )
        self._require_current_capture_preflight()
        specifications: list[tuple[str, int, str, str, str]] = []
        for side, tissue_path in (("left", self.left_tissue_path), ("right", self.right_tissue_path)):
            side_title = side.capitalize()
            for index in range(CAPTURE_CELL_COUNT):
                rigid_path = f"{self.tool_path}/Links/{side_title}Pad/Collisions/TissueCaptureCell_{index:02d}"
                attachment_path = f"{self.tool_path}/RuntimeAttachments/{side_title}Capture_{index:02d}"
                rigid = self.stage.GetPrimAtPath(rigid_path)
                if not rigid or not rigid.IsValid():
                    raise ValueError(
                        f"No registered capture-cell rigid body at {rigid_path}"
                    )
                specifications.append(
                    (side, index, tissue_path, rigid_path, attachment_path)
                )

        if self.active_cells():
            raise RuntimeError(
                "capture preflight must begin with no controller-owned "
                "attachments"
            )
        self.cells.clear()
        for _, _, _, _, attachment_path in specifications:
            existing = self.stage.GetPrimAtPath(attachment_path)
            if existing and existing.IsValid():
                raise RuntimeError(
                    "refusing to overwrite an attachment prim that is not "
                    f"owned by the current capture state: {attachment_path}"
                )
        created_paths: list[str] = []
        pending_cells: list[CaptureCell] = []
        pending_events: list[dict[str, Any]] = []
        try:
            for side, index, tissue_path, rigid_path, attachment_path in specifications:
                created_paths.append(attachment_path)
                method = create_deformable_attachment(
                    tissue_path,
                    rigid_path,
                    attachment_path,
                    stage=self.stage,
                )
                pending_cells.append(
                    CaptureCell(side, index, attachment_path, rigid_path)
                )
                pending_events.append(
                    {
                        "event": "capture",
                        "side": side,
                        "index": index,
                        "method": method,
                        "tissue_body_path": tissue_path,
                    }
                )
        except Exception:
            try:
                remove_prims(created_paths, stage=self.stage)
            except Exception as rollback_error:
                self.cells.clear()
                self.safety_latched = True
                self.capture_epoch_confirmed = False
                self.latch_reason = "recapture_rollback_failed"
                self._clear_capture_handshake()
                self.events.append(
                    {
                        "event": "capture_rollback_failed",
                        "attempted_attachment_count": len(created_paths),
                    }
                )
                raise RuntimeError(
                    "capture authoring failed and partial attachment rollback "
                    "could not be verified"
                ) from rollback_error
            self.cells.clear()
            self.safety_latched = True
            self.capture_epoch_confirmed = False
            self.latch_reason = "recapture_failed"
            self._clear_capture_handshake()
            self.events.append(
                {
                    "event": "capture_failed",
                    "created_attachment_count_rolled_back": len(created_paths),
                }
            )
            raise

        self.cells[:] = pending_cells
        self.events.extend(pending_events)
        self.capture_epoch_confirmed = False
        self.safety_latched = True
        self._capture_confirmation_topology_revision = None
        self._capture_confirmation_interval_count = 0
        self._capture_confirmation_started_at_s = None
        self.latch_reason = "recapture_pending_post_physics_confirmation"
        return list(self.cells)

    def active_cells(self, side: str | None = None) -> list[CaptureCell]:
        return [cell for cell in self.cells if cell.active and (side is None or cell.side == side)]

    def _capture_pose_tolerance(self, joint_name: str) -> float:
        return (
            self.capture_angular_position_tolerance_rad
            if "pitch" in joint_name
            else self.capture_linear_position_tolerance_m
        )

    def _validate_capture_scene(
        self,
        evidence: AtraumaticExposureSceneEvidence,
        *,
        attachments_expected: bool,
    ) -> dict[str, dict[str, float]]:
        capture_targets = phase_targets("capture")
        for joint_name, measured_position in (
            evidence.measured_joint_positions_m.items()
        ):
            if (
                abs(measured_position - capture_targets[joint_name])
                > self._capture_pose_tolerance(joint_name)
            ):
                raise RuntimeError(
                    f"{joint_name} has not reached the registered capture "
                    "pose within tolerance"
                )

        active = {
            (cell.side, cell.index): cell
            for cell in self.active_cells()
        }
        expected_active_count = (
            2 * CAPTURE_CELL_COUNT if attachments_expected else 0
        )
        if len(active) != expected_active_count:
            raise RuntimeError(
                "capture evidence does not match controller attachment "
                f"ownership: expected {expected_active_count}, got {len(active)}"
            )

        totals: dict[str, dict[str, float]] = {}
        for side, tissue_path in (
            ("left", self.left_tissue_path),
            ("right", self.right_tissue_path),
        ):
            side_title = side.capitalize()
            samples = evidence.cells(side)
            normal_total = _nonnegative_finite(
                evidence.total_force_n(side),
                f"{side}_total_normal_force_n",
            )
            transmitted_total = _nonnegative_finite(
                evidence.total_transmitted_load_n(side),
                f"{side}_total_transmitted_load_n",
            )
            if (
                normal_total
                < self.minimum_capture_cell_normal_force_n * CAPTURE_CELL_COUNT
                or transmitted_total
                < self.minimum_capture_cell_normal_force_n * CAPTURE_CELL_COUNT
                or transmitted_total >= self.hard_pad_release_force_n
            ):
                raise RuntimeError(
                    f"{side} capture precondition load is absent or unsafe"
                )
            totals[side] = {
                "normal_force_n": normal_total,
                "transmitted_load_n": transmitted_total,
            }
            for sample in samples:
                expected_rigid_path = (
                    f"{self.tool_path}/Links/{side_title}Pad/Collisions/"
                    f"TissueCaptureCell_{sample.index:02d}"
                )
                expected_attachment_path = (
                    f"{self.tool_path}/RuntimeAttachments/"
                    f"{side_title}Capture_{sample.index:02d}"
                )
                cell = active.get((side, sample.index))
                if (
                    sample.source.contact_cell_prim_path
                    != expected_rigid_path
                    or sample.source.tissue_prim_path != tissue_path
                    or sample.source.attachment_prim_path
                    != expected_attachment_path
                ):
                    raise ValueError(
                        f"{side} capture cell {sample.index} is not bound to "
                        "the controller's exact scene bodies"
                    )
                if attachments_expected and (
                    cell is None
                    or cell.rigid_cell_path != expected_rigid_path
                    or cell.attachment_path != expected_attachment_path
                ):
                    raise ValueError(
                        f"{side} capture cell {sample.index} is not owned by "
                        "this controller"
                    )
                if sample.attachment_present != attachments_expected:
                    raise RuntimeError(
                        f"{side} capture cell {sample.index} attachment state "
                        "does not match the capture handshake"
                    )
                if (
                    sample.normal_force_n
                    < self.minimum_capture_cell_normal_force_n
                    or sample.contact_area_m2
                    < self.minimum_capture_cell_contact_area_m2
                    or sample.transmitted_load_n
                    >= self.soft_cell_release_force_n
                    or sample.slip_speed_m_s
                    >= self.max_cell_slip_speed_m_s
                ):
                    raise RuntimeError(
                        f"{side} capture cell {sample.index} is not in a "
                        "stable capture contact state"
                    )
        return totals

    def preflight_capture_from_scene(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> dict[str, Any]:
        """Consume a no-attachment contact interval before authoring capture."""

        if self.active_cells():
            raise RuntimeError(
                "capture preflight requires all prior attachments to be "
                "released first"
            )
        self._clear_capture_handshake()
        self.safety_latched = True
        self.capture_epoch_confirmed = False
        self.latch_reason = "capture_preflight_in_progress"
        provenance = evidence.envelope.provenance
        topology_revision = provenance.topology_revision
        if provenance.dt_s > self.capture_maximum_evidence_interval_s:
            self.latch_reason = "capture_preflight_interval_not_fresh"
            raise ValueError(
                "capture preflight interval exceeds "
                "capture_maximum_evidence_interval_s"
            )
        if (
            self.evidence_cursor.last_physics_step >= 0
            and provenance.physics_step
            != self.evidence_cursor.last_physics_step + 1
        ):
            self.latch_reason = "capture_preflight_not_adjacent"
            raise ValueError(
                "capture preflight must be the next post-physics interval "
                "after the last consumed exposure evidence"
            )
        topology_changed = (
            self.evidence_cursor.topology_revision is not None
            and topology_revision != self.evidence_cursor.topology_revision
        )
        self.evidence_cursor.validate(
            evidence,
            allow_topology_change=topology_changed,
        )
        totals = self._validate_capture_scene(
            evidence,
            attachments_expected=False,
        )
        if topology_changed:
            self.evidence_cursor.commit_topology_transition(evidence)
        else:
            self.evidence_cursor.commit(evidence)
        self._capture_preflight_digest_sha256 = (
            evidence.evidence_digest_sha256
        )
        self._capture_preflight_topology_revision = topology_revision
        self._capture_preflight_episode_id = provenance.episode_id
        self._capture_preflight_environment_id = provenance.environment_id
        self._capture_preflight_source_registration = evidence.sources
        self._capture_preflight_physics_step = provenance.physics_step
        self._capture_preflight_simulation_time_s = provenance.simulation_time_s
        self.latch_reason = "capture_preflight_confirmed_pending_authoring"
        report = {
            "capture_preflight_confirmed": True,
            "left_total_normal_force_n": totals["left"]["normal_force_n"],
            "right_total_normal_force_n": totals["right"]["normal_force_n"],
            "left_total_transmitted_load_n": (
                totals["left"]["transmitted_load_n"]
            ),
            "right_total_transmitted_load_n": (
                totals["right"]["transmitted_load_n"]
            ),
            "minimum_capture_cell_normal_force_n": (
                self.minimum_capture_cell_normal_force_n
            ),
            "minimum_capture_cell_contact_area_m2": (
                self.minimum_capture_cell_contact_area_m2
            ),
            "capture_maximum_evidence_interval_s": (
                self.capture_maximum_evidence_interval_s
            ),
            "topology_revision": topology_revision,
            "physics_step": evidence.physics_step,
            "simulation_time_s": evidence.simulation_time_s,
            "evidence_digest_sha256": evidence.evidence_digest_sha256,
        }
        self.events.append({"event": "capture_preflight_confirmed", **report})
        return report

    def release_cell(self, side: str, index: int, reason: str) -> bool:
        for cell in self.cells:
            if cell.side == side and cell.index == index and cell.active:
                self.safety_latched = True
                self.capture_epoch_confirmed = False
                self.latch_reason = (
                    f"attachment_removal_pending_{side}_{reason}"
                )
                try:
                    remove_prims([cell.attachment_path], stage=self.stage)
                except Exception:
                    self.latch_reason = (
                        f"attachment_removal_failed_{side}_{reason}"
                    )
                    raise
                cell.active = False
                cell.released_reason = reason
                self.events.append({"event": "release_cell", "side": side, "index": index, "reason": reason})
                self.latch_reason = (
                    f"capture_topology_changed_after_{side}_{reason}"
                )
                self._clear_capture_handshake()
                return True
        return False

    def release_all(self, reason: str = "commanded_release") -> None:
        active = self.active_cells()
        self.safety_latched = True
        self.capture_epoch_confirmed = False
        self.latch_reason = f"attachment_removal_pending_{reason}"
        try:
            remove_prims(
                [cell.attachment_path for cell in active],
                stage=self.stage,
            )
        except Exception:
            self.latch_reason = f"attachment_removal_failed_{reason}"
            raise
        for cell in active:
            cell.active = False
            cell.released_reason = reason
        if active:
            self.events.append({"event": "release_all", "reason": reason, "count": len(active)})
        self.latch_reason = reason
        self._clear_capture_handshake()

    def confirm_capture_from_scene(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> dict[str, Any]:
        """Confirm capture across adjacent, stable post-physics intervals."""
        initial_confirmation = (
            self.latch_reason
            == "recapture_pending_post_physics_confirmation"
        )
        continuing_confirmation = (
            self.latch_reason
            == "recapture_pending_dwell_confirmation"
        )
        if not initial_confirmation and not continuing_confirmation:
            raise RuntimeError(
                "capture confirmation requires a newly authored capture or an "
                "in-progress adjacent dwell sequence"
            )
        provenance = evidence.envelope.provenance
        topology_revision = provenance.topology_revision
        try:
            if provenance.dt_s > self.capture_maximum_evidence_interval_s:
                raise ValueError(
                    "capture confirmation interval exceeds "
                    "capture_maximum_evidence_interval_s"
                )
            if (
                provenance.episode_id != self._capture_preflight_episode_id
                or provenance.environment_id
                != self._capture_preflight_environment_id
                or evidence.sources
                != self._capture_preflight_source_registration
            ):
                raise ValueError(
                    "capture confirmation must preserve the preflight episode, "
                    "environment, and exact source registration"
                )
            if initial_confirmation:
                self._require_current_capture_preflight()
                if (
                    self._capture_preflight_physics_step is None
                    or provenance.physics_step
                    != self._capture_preflight_physics_step + 1
                ):
                    raise ValueError(
                        "first capture confirmation must be the physics interval "
                        "immediately adjacent to preflight"
                    )
                if topology_revision == self._capture_preflight_topology_revision:
                    raise ValueError(
                        "recapture evidence must publish a new topology revision"
                    )
                self.evidence_cursor.validate(
                    evidence,
                    allow_topology_change=True,
                )
                totals = self._validate_capture_scene(
                    evidence,
                    attachments_expected=True,
                )
                self.evidence_cursor.commit_topology_transition(evidence)
                self._capture_confirmation_topology_revision = (
                    topology_revision
                )
                self._capture_confirmation_interval_count = 1
                self._capture_confirmation_started_at_s = (
                    provenance.simulation_time_s
                )
            else:
                if (
                    self._capture_confirmation_topology_revision is None
                    or self._capture_confirmation_started_at_s is None
                    or topology_revision
                    != self._capture_confirmation_topology_revision
                ):
                    raise ValueError(
                        "capture confirmation topology changed during the "
                        "required dwell sequence"
                    )
                if (
                    self.evidence_cursor.episode_id
                    != self._capture_preflight_episode_id
                    or self.evidence_cursor.environment_id
                    != self._capture_preflight_environment_id
                    or self.evidence_cursor.source_registration
                    != self._capture_preflight_source_registration
                    or self.evidence_cursor.topology_revision
                    != self._capture_confirmation_topology_revision
                ):
                    raise ValueError(
                        "capture confirmation cursor departed from the "
                        "preflight lineage"
                    )
                if (
                    provenance.physics_step
                    != self.evidence_cursor.last_physics_step + 1
                ):
                    raise ValueError(
                        "capture confirmation intervals must be consecutive"
                    )
                self.evidence_cursor.validate(evidence)
                totals = self._validate_capture_scene(
                    evidence,
                    attachments_expected=True,
                )
                self.evidence_cursor.commit(evidence)
                self._capture_confirmation_interval_count += 1
        except Exception:
            self.capture_epoch_confirmed = False
            self.safety_latched = True
            self.latch_reason = (
                "capture_confirmation_failed_requires_release"
            )
            raise

        confirmation_dwell_s = (
            provenance.simulation_time_s
            - self._capture_confirmation_started_at_s
        )
        confirmation_complete = (
            self._capture_confirmation_interval_count
            >= self.capture_confirmation_intervals_required
            and confirmation_dwell_s + 1.0e-12
            >= self.capture_confirmation_min_dwell_s
        )
        self.capture_epoch_confirmed = confirmation_complete
        self.safety_latched = not confirmation_complete
        self.latch_reason = (
            ""
            if confirmation_complete
            else "recapture_pending_dwell_confirmation"
        )
        report = {
            "capture_epoch_confirmed": confirmation_complete,
            "capture_confirmation_interval_count": (
                self._capture_confirmation_interval_count
            ),
            "capture_confirmation_intervals_required": (
                self.capture_confirmation_intervals_required
            ),
            "capture_confirmation_dwell_s": confirmation_dwell_s,
            "capture_confirmation_min_dwell_s": (
                self.capture_confirmation_min_dwell_s
            ),
            "capture_maximum_evidence_interval_s": (
                self.capture_maximum_evidence_interval_s
            ),
            "minimum_capture_cell_normal_force_n": (
                self.minimum_capture_cell_normal_force_n
            ),
            "minimum_capture_cell_contact_area_m2": (
                self.minimum_capture_cell_contact_area_m2
            ),
            "active_left": len(self.active_cells("left")),
            "active_right": len(self.active_cells("right")),
            "left_total_normal_force_n": totals["left"]["normal_force_n"],
            "right_total_normal_force_n": totals["right"]["normal_force_n"],
            "left_total_transmitted_load_n": (
                totals["left"]["transmitted_load_n"]
            ),
            "right_total_transmitted_load_n": (
                totals["right"]["transmitted_load_n"]
            ),
            "topology_revision": topology_revision,
            "physics_step": evidence.physics_step,
            "simulation_time_s": evidence.simulation_time_s,
            "evidence_digest_sha256": evidence.evidence_digest_sha256,
        }
        self.events.append(
            {
                "event": (
                    "capture_confirmed"
                    if confirmation_complete
                    else "capture_confirmation_sample"
                ),
                **report,
            }
        )
        if confirmation_complete:
            self._clear_capture_handshake()
        return report

    def update_loads(
        self,
        *,
        left_total_force_n: float,
        right_total_force_n: float,
        left_cell_forces_n: Sequence[float] | None = None,
        right_cell_forces_n: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        del (
            left_total_force_n,
            right_total_force_n,
            left_cell_forces_n,
            right_cell_forces_n,
        )
        raise RuntimeError(
            "caller-authored pad loads are not admissible mechanics evidence; "
            "use update_from_scene() with a prim-bound exposure envelope"
        )

    def update_from_scene(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> dict[str, Any]:
        """Release capture cells only from exact post-physics contact evidence."""
        if not isinstance(evidence, AtraumaticExposureSceneEvidence):
            raise TypeError(
                "capture control requires AtraumaticExposureSceneEvidence"
            )
        if self.safety_latched:
            raise RuntimeError(
                "capture safety latch is active; explicitly recapture before "
                "processing another hold interval"
            )
        # Pessimistically latch before any validation or mutation.  Every
        # successful no-release path below explicitly restores the confirmed
        # capture; any exception therefore fails closed.
        self.safety_latched = True
        self.capture_epoch_confirmed = False
        self.latch_reason = "hold_evidence_preflight_in_progress"
        topology_revision = evidence.envelope.provenance.topology_revision
        topology_changed = (
            self.evidence_cursor.topology_revision is not None
            and topology_revision != self.evidence_cursor.topology_revision
        )
        self.evidence_cursor.validate(
            evidence,
            allow_topology_change=topology_changed,
        )
        result: dict[str, Any] = {
            "released": [],
            "hard_release": [],
        }
        totals: dict[str, float] = {}
        transmitted_totals: dict[str, float] = {}
        active_by_key: dict[tuple[str, int], CaptureCell] = {}
        for cell in self.active_cells():
            key = (cell.side, cell.index)
            if key in active_by_key:
                raise RuntimeError(
                    f"duplicate active capture-cell ownership for {key!r}"
                )
            active_by_key[key] = cell
        release_plan: dict[tuple[str, int], str] = {}
        hard_release_keys: set[tuple[str, int]] = set()

        for side, tissue_path in (
            ("left", self.left_tissue_path),
            ("right", self.right_tissue_path),
        ):
            side_title = side.capitalize()
            samples = evidence.cells(side)
            total = _nonnegative_finite(
                evidence.total_force_n(side),
                f"{side}_total_force_n",
            )
            totals[side] = total
            transmitted_total = _nonnegative_finite(
                evidence.total_transmitted_load_n(side),
                f"{side}_total_transmitted_load_n",
            )
            transmitted_totals[side] = transmitted_total
            for sample in samples:
                expected_rigid_path = (
                    f"{self.tool_path}/Links/{side_title}Pad/Collisions/"
                    f"TissueCaptureCell_{sample.index:02d}"
                )
                expected_attachment_path = (
                    f"{self.tool_path}/RuntimeAttachments/"
                    f"{side_title}Capture_{sample.index:02d}"
                )
                source = sample.source
                if (
                    source.contact_cell_prim_path != expected_rigid_path
                    or source.tissue_prim_path != tissue_path
                    or source.attachment_prim_path
                    != expected_attachment_path
                ):
                    raise ValueError(
                        f"{side} capture cell {sample.index} is not bound to "
                        "this controller's exact scene prims"
                    )
                active = active_by_key.get((side, sample.index))
                if active is not None and (
                    active.rigid_cell_path != expected_rigid_path
                    or active.attachment_path != expected_attachment_path
                ):
                    raise RuntimeError(
                        f"controller ownership for {side} cell {sample.index} "
                        "does not match its registered scene paths"
                    )
                if sample.attachment_present and active is None:
                    raise RuntimeError(
                        f"scene reports unowned active {side} attachment "
                        f"{sample.index}"
                    )
                if active is not None and not sample.attachment_present:
                    release_plan[(side, sample.index)] = (
                        "attachment_missing_from_scene"
                    )

            if transmitted_total >= self.hard_pad_release_force_n:
                for key in active_by_key:
                    if key[0] != side:
                        continue
                    release_plan[key] = "measured_hard_pad_overload"
                    hard_release_keys.add(key)
                continue
            for sample in samples:
                if not sample.attachment_present:
                    continue
                release_reason: str | None = None
                if (
                    sample.normal_force_n
                    < self.minimum_capture_cell_normal_force_n
                    or sample.contact_area_m2
                    < self.minimum_capture_cell_contact_area_m2
                ):
                    release_reason = "measured_contact_below_retention_floor"
                elif sample.slip_speed_m_s >= self.max_cell_slip_speed_m_s:
                    release_reason = "measured_local_cell_slip"
                elif (
                    sample.transmitted_load_n
                    >= self.soft_cell_release_force_n
                ):
                    release_reason = "measured_local_transmitted_overload"
                if release_reason is not None:
                    release_plan[(side, sample.index)] = release_reason

        cells_to_release = [
            active_by_key[key]
            for key in release_plan
            if key in active_by_key
        ]
        missing_attachment_observed = any(
            reason == "attachment_missing_from_scene"
            for reason in release_plan.values()
        )
        if topology_changed and not missing_attachment_observed:
            self.latch_reason = "unexpected_capture_topology_change"
            raise RuntimeError(
                "exposure topology changed without a corresponding exact "
                "attachment-loss observation"
            )
        if topology_changed:
            self.evidence_cursor.commit_topology_transition(evidence)
        else:
            self.evidence_cursor.commit(evidence)
        try:
            remove_prims(
                [cell.attachment_path for cell in cells_to_release],
                stage=self.stage,
            )
        except Exception:
            self.safety_latched = True
            self.capture_epoch_confirmed = False
            self.latch_reason = (
                "attachment_removal_failed_after_evidence_commit"
            )
            self.events.append(
                {
                    "event": "release_commit_failed",
                    "physics_step": evidence.physics_step,
                    "evidence_digest_sha256": evidence.evidence_digest_sha256,
                }
            )
            raise
        for cell in cells_to_release:
            key = (cell.side, cell.index)
            reason = release_plan[key]
            cell.active = False
            cell.released_reason = reason
            self.events.append(
                {
                    "event": "release_cell",
                    "side": cell.side,
                    "index": cell.index,
                    "reason": reason,
                    "physics_step": evidence.physics_step,
                    "evidence_digest_sha256": evidence.evidence_digest_sha256,
                }
            )
            if key in hard_release_keys:
                result["hard_release"].append(key)
            else:
                result["released"].append(key)

        if cells_to_release:
            self.safety_latched = True
            self.capture_epoch_confirmed = False
            self.latch_reason = (
                "measured_hard_pad_overload"
                if hard_release_keys
                else "capture_topology_changed_requires_explicit_recapture"
            )
            self._clear_capture_handshake()
        else:
            self.safety_latched = False
            self.capture_epoch_confirmed = True
            self.latch_reason = ""

        result.update(
            {
                "active_left": len(self.active_cells("left")),
                "active_right": len(self.active_cells("right")),
                "left_total_force_n": totals["left"],
                "right_total_force_n": totals["right"],
                "left_total_transmitted_load_n": transmitted_totals["left"],
                "right_total_transmitted_load_n": transmitted_totals["right"],
                "physics_step": evidence.physics_step,
                "simulation_time_s": evidence.simulation_time_s,
                "evidence_digest_sha256": evidence.evidence_digest_sha256,
                "safety_latched": self.safety_latched,
                "latch_reason": self.latch_reason,
                "scope": (
                    "simulator_mechanics_only_not_tissue_injury_or_clinical_"
                    "safety"
                ),
            }
        )
        return result


def estimate_pad_force_n(
    compression_m: float,
    compression_velocity_m_s: float = 0.0,
    *,
    stiffness_n_m: float = 1_250.0,
    damping_n_s_m: float = 38.0,
) -> float:
    """Return a task proxy; this estimate is not admissible force evidence."""
    compression = max(0.0, -_finite(compression_m, "compression_m"))
    closing_velocity = max(
        0.0, -_finite(compression_velocity_m_s, "compression_velocity_m_s")
    )
    stiffness = _nonnegative_finite(stiffness_n_m, "stiffness_n_m")
    damping = _nonnegative_finite(damping_n_s_m, "damping_n_s_m")
    return stiffness * compression + damping * closing_velocity


@dataclass
class ForceControlOutput:
    joint_targets: dict[str, float]
    force_error_n: dict[str, float]
    exposure_error: float
    overload: dict[str, bool]
    mode: str


@dataclass
class ForceControlledRetractionController:
    """Outer-loop ROI controller with independent force-limited pad motion."""

    target_visible_fraction: float = 0.88
    target_force_per_pad_n: float = 1.25
    soft_force_limit_n: float = 2.5
    hard_force_limit_n: float = 4.0
    max_force_asymmetry_n: float = 1.0
    lateral_gain_m_per_fraction: float = 0.010
    lift_gain_m_per_fraction: float = 0.006
    force_gain_m_per_n: float = 0.0018
    integral_gain_m_per_fraction_s: float = 0.0012
    max_integral_m: float = 0.008
    nominal_update_hz: float = 120.0
    maximum_update_interval_s: float = 1.0 / 30.0
    left_carriage_m: float = 0.006
    right_carriage_m: float = -0.006
    left_lift_m: float = 0.017
    right_lift_m: float = 0.017
    integral_error: float = 0.0

    def __post_init__(self) -> None:
        visible = _finite(
            self.target_visible_fraction,
            "target_visible_fraction",
        )
        target_force = _nonnegative_finite(
            self.target_force_per_pad_n,
            "target_force_per_pad_n",
        )
        soft = _nonnegative_finite(
            self.soft_force_limit_n,
            "soft_force_limit_n",
        )
        hard = _nonnegative_finite(
            self.hard_force_limit_n,
            "hard_force_limit_n",
        )
        asymmetry = _nonnegative_finite(
            self.max_force_asymmetry_n,
            "max_force_asymmetry_n",
        )
        if not 0.0 < visible <= 1.0:
            raise ValueError("target_visible_fraction must be in (0, 1]")
        if not 0.0 < target_force < soft < hard:
            raise ValueError(
                "force thresholds require 0 < target < soft < hard"
            )
        if not 0.0 < asymmetry < hard:
            raise ValueError(
                "max_force_asymmetry_n must be positive and below hard limit"
            )
        self.target_visible_fraction = visible
        self.target_force_per_pad_n = target_force
        self.soft_force_limit_n = soft
        self.hard_force_limit_n = hard
        self.max_force_asymmetry_n = asymmetry

        for name in (
            "lateral_gain_m_per_fraction",
            "lift_gain_m_per_fraction",
            "force_gain_m_per_n",
            "integral_gain_m_per_fraction_s",
            "max_integral_m",
            "nominal_update_hz",
            "maximum_update_interval_s",
        ):
            value = _nonnegative_finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        self.synchronize(
            {
                "left_carriage_joint": self.left_carriage_m,
                "right_carriage_joint": self.right_carriage_m,
                "left_lift_joint": self.left_lift_m,
                "right_lift_joint": self.right_lift_m,
            },
            reset_integral=False,
        )
        integral = _finite(self.integral_error, "integral_error")
        maximum_error = (
            self.max_integral_m / self.integral_gain_m_per_fraction_s
        )
        if abs(integral) > maximum_error:
            raise ValueError(
                "integral_error exceeds the configured anti-windup bound"
            )
        self.integral_error = integral

    def synchronize(
        self,
        joint_targets: Mapping[str, float],
        *,
        reset_integral: bool = True,
    ) -> None:
        required = {
            "left_carriage_joint": (0.0, 0.040),
            "right_carriage_joint": (-0.040, 0.0),
            "left_lift_joint": (-0.025, 0.030),
            "right_lift_joint": (-0.025, 0.030),
        }
        values: dict[str, float] = {}
        for name, (low, high) in required.items():
            if name not in joint_targets:
                raise ValueError(
                    f"controller synchronization is missing {name}"
                )
            value = _finite(joint_targets[name], name)
            if not low <= value <= high:
                raise ValueError(
                    f"{name}={value} is outside [{low}, {high}]"
                )
            values[name] = value
        self.left_carriage_m = values["left_carriage_joint"]
        self.right_carriage_m = values["right_carriage_joint"]
        self.left_lift_m = values["left_lift_joint"]
        self.right_lift_m = values["right_lift_joint"]
        if reset_integral:
            self.integral_error = 0.0

    def commit_state_from(
        self,
        candidate: "ForceControlledRetractionController",
    ) -> None:
        if not isinstance(candidate, ForceControlledRetractionController):
            raise TypeError("candidate must be a force controller")
        self.synchronize(
            {
                "left_carriage_joint": candidate.left_carriage_m,
                "right_carriage_joint": candidate.right_carriage_m,
                "left_lift_joint": candidate.left_lift_m,
                "right_lift_joint": candidate.right_lift_m,
            },
            reset_integral=False,
        )
        self.integral_error = candidate.integral_error

    def reset(self) -> None:
        self.synchronize(
            {
                "left_carriage_joint": 0.006,
                "right_carriage_joint": -0.006,
                "left_lift_joint": 0.017,
                "right_lift_joint": 0.017,
            }
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def update(
        self,
        *,
        dt: float,
        visible_fraction: float,
        left_force_n: float,
        right_force_n: float,
    ) -> ForceControlOutput:
        dt = _finite(dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if dt > self.maximum_update_interval_s:
            raise ValueError(
                "dt exceeds maximum_update_interval_s; stale control "
                "intervals cannot be extrapolated"
            )
        update_scale = dt * _nonnegative_finite(
            self.nominal_update_hz, "nominal_update_hz"
        )
        if update_scale <= 0.0:
            raise ValueError("nominal_update_hz must be positive")
        visible = self._clamp(_finite(visible_fraction, "visible_fraction"), 0.0, 1.0)
        left_force = _nonnegative_finite(left_force_n, "left_force_n")
        right_force = _nonnegative_finite(right_force_n, "right_force_n")
        exposure_error = self.target_visible_fraction - visible
        self.integral_error = self._clamp(
            self.integral_error + exposure_error * dt,
            -self.max_integral_m / max(self.integral_gain_m_per_fraction_s, 1e-9),
            self.max_integral_m / max(self.integral_gain_m_per_fraction_s, 1e-9),
        )

        left_over = left_force >= self.soft_force_limit_n
        right_over = right_force >= self.soft_force_limit_n
        hard_left = left_force >= self.hard_force_limit_n
        hard_right = right_force >= self.hard_force_limit_n

        if hard_left or hard_right:
            # Immediate commanded unloading; capture controller handles bond release.
            relief_step = 0.004 * update_scale
            self.left_carriage_m = max(0.0, self.left_carriage_m - relief_step)
            self.right_carriage_m = min(0.0, self.right_carriage_m + relief_step)
            self.left_lift_m = min(0.030, self.left_lift_m + relief_step)
            self.right_lift_m = min(0.030, self.right_lift_m + relief_step)
            mode = "hard_overload_relief"
        else:
            visibility_step = (
                self.lateral_gain_m_per_fraction * exposure_error
                + self.integral_gain_m_per_fraction_s * self.integral_error
            )
            left_force_error = self.target_force_per_pad_n - left_force
            right_force_error = self.target_force_per_pad_n - right_force
            left_step = (
                visibility_step + self.force_gain_m_per_n * left_force_error
            ) * update_scale
            right_step = (
                visibility_step + self.force_gain_m_per_n * right_force_error
            ) * update_scale
            if left_over:
                left_step = min(
                    left_step,
                    -self.force_gain_m_per_n
                    * (left_force - self.soft_force_limit_n)
                    * update_scale,
                )
            if right_over:
                right_step = min(
                    right_step,
                    -self.force_gain_m_per_n
                    * (right_force - self.soft_force_limit_n)
                    * update_scale,
                )
            self.left_carriage_m = self._clamp(self.left_carriage_m + left_step, 0.0, 0.040)
            self.right_carriage_m = self._clamp(self.right_carriage_m - right_step, -0.040, 0.0)

            # Lift assists exposure but unloads a pad that is already force limited.
            lift_step = (
                self.lift_gain_m_per_fraction * exposure_error * update_scale
            )
            overload_lift_step = 0.002 * update_scale
            self.left_lift_m = self._clamp(
                self.left_lift_m
                - lift_step
                + (overload_lift_step if left_over else 0.0),
                -0.025,
                0.030,
            )
            self.right_lift_m = self._clamp(
                self.right_lift_m
                - lift_step
                + (overload_lift_step if right_over else 0.0),
                -0.025,
                0.030,
            )
            mode = "force_limited_exposure_control"

        # Differential correction reduces excessive bilateral force asymmetry.
        asymmetry = left_force - right_force
        if abs(asymmetry) > self.max_force_asymmetry_n:
            correction = min(0.0025, 0.0012 * abs(asymmetry)) * update_scale
            if asymmetry > 0:
                self.left_carriage_m = max(0.0, self.left_carriage_m - correction)
            else:
                self.right_carriage_m = min(0.0, self.right_carriage_m + correction)

        return ForceControlOutput(
            joint_targets={
                "left_carriage_joint": self.left_carriage_m,
                "right_carriage_joint": self.right_carriage_m,
                "left_lift_joint": self.left_lift_m,
                "right_lift_joint": self.right_lift_m,
                "left_pitch_joint": math.radians(-16.0),
                "right_pitch_joint": math.radians(16.0),
                "left_compliance_joint": 0.0,
                "right_compliance_joint": 0.0,
            },
            force_error_n={
                "left": self.target_force_per_pad_n - left_force,
                "right": self.target_force_per_pad_n - right_force,
                "asymmetry": asymmetry,
            },
            exposure_error=exposure_error,
            overload={"left_soft": left_over, "right_soft": right_over, "left_hard": hard_left, "right_hard": hard_right},
            mode=mode,
        )


class ROIExposureEstimator:
    """Visibility metrics usable with segmentation masks or geometric flap edges."""

    @staticmethod
    def from_masks(roi_mask: Any, occluder_mask: Any) -> float:
        import numpy as np
        roi = np.asarray(tensor_value(roi_mask), dtype=bool)
        occluder = np.asarray(tensor_value(occluder_mask), dtype=bool)
        if roi.shape != occluder.shape:
            raise ValueError(f"mask shape mismatch: roi={roi.shape}, occluder={occluder.shape}")
        total = int(roi.sum())
        if total == 0:
            return 0.0
        visible = np.logical_and(roi, np.logical_not(occluder)).sum()
        return float(visible / total)

    @staticmethod
    def from_edge_gap(gap_width_m: float, target_width_m: float = 0.044) -> float:
        gap = _finite(gap_width_m, "gap_width_m")
        target = _finite(target_width_m, "target_width_m")
        if target <= 0:
            raise ValueError("target_width_m must be positive")
        return max(0.0, min(1.0, gap / target))

    @staticmethod
    def bilateral_balance(left_visible_fraction: float, right_visible_fraction: float) -> float:
        left = max(
            0.0,
            min(1.0, _finite(left_visible_fraction, "left_visible_fraction")),
        )
        right = max(
            0.0,
            min(1.0, _finite(right_visible_fraction, "right_visible_fraction")),
        )
        return 1.0 - abs(left - right)


PHASE_TARGETS = {
    "stowed": {
        "left_carriage_joint": 0.0, "right_carriage_joint": 0.0,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(58.0), "right_pitch_joint": math.radians(-58.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "approach": {
        "left_carriage_joint": 0.002, "right_carriage_joint": -0.002,
        "left_lift_joint": 0.004, "right_lift_joint": 0.004,
        "left_pitch_joint": math.radians(35.0), "right_pitch_joint": math.radians(-35.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "deploy": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.014, "right_lift_joint": 0.014,
        "left_pitch_joint": math.radians(12.0), "right_pitch_joint": math.radians(-12.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "contact": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.017, "right_lift_joint": 0.017,
        "left_pitch_joint": math.radians(2.0), "right_pitch_joint": math.radians(-2.0),
        "left_compliance_joint": -0.002, "right_compliance_joint": -0.002,
    },
    "capture": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.017, "right_lift_joint": 0.017,
        "left_pitch_joint": math.radians(2.0), "right_pitch_joint": math.radians(-2.0),
        "left_compliance_joint": -0.002, "right_compliance_joint": -0.002,
    },
    "retract": {
        "left_carriage_joint": 0.032, "right_carriage_joint": -0.032,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(-16.0), "right_pitch_joint": math.radians(16.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "hold": {
        "left_carriage_joint": 0.032, "right_carriage_joint": -0.032,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(-16.0), "right_pitch_joint": math.radians(16.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "overload_relief": {
        "left_carriage_joint": 0.020, "right_carriage_joint": -0.020,
        "left_lift_joint": -0.004, "right_lift_joint": -0.004,
        "left_pitch_joint": math.radians(-5.0), "right_pitch_joint": math.radians(5.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "release": {
        "left_carriage_joint": 0.010, "right_carriage_joint": -0.010,
        "left_lift_joint": 0.010, "right_lift_joint": 0.010,
        "left_pitch_joint": math.radians(20.0), "right_pitch_joint": math.radians(-20.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
}

LEGAL_PHASE_TRANSITIONS = {
    "stowed": frozenset({"approach"}),
    "approach": frozenset({"deploy", "stowed"}),
    "deploy": frozenset({"contact", "approach", "release"}),
    "contact": frozenset({"capture", "release"}),
    "capture": frozenset({"retract", "release"}),
    "retract": frozenset({"hold", "overload_relief", "release"}),
    "hold": frozenset({"overload_relief", "release"}),
    "overload_relief": frozenset({"release"}),
    "release": frozenset({"stowed", "approach"}),
}


def phase_targets(phase: str) -> dict[str, float]:
    try:
        return dict(PHASE_TARGETS[phase])
    except KeyError as exc:
        raise KeyError(f"Unknown exposure phase {phase!r}; expected one of {sorted(PHASE_TARGETS)}") from exc


@dataclass
class ExposureSequenceController:
    """Discrete workflow coordinator around capture and force-aware hold control."""

    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    stage: Any = None
    phase: str = "stowed"
    capture: DistributedPadCaptureController = field(init=False)
    force_controller: ForceControlledRetractionController = field(default_factory=ForceControlledRetractionController)
    evidence_cursor: AtraumaticExposureEvidenceCursor = field(
        default_factory=AtraumaticExposureEvidenceCursor
    )
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.stage = _current_stage(self.stage)
        self.capture = DistributedPadCaptureController(
            tool_path=self.tool_path,
            left_tissue_path=self.left_tissue_path,
            right_tissue_path=self.right_tissue_path,
            stage=self.stage,
            evidence_cursor=self.evidence_cursor,
        )

    def set_phase(self, phase: str) -> dict[str, float]:
        targets = phase_targets(phase)
        if phase == self.phase:
            if phase not in {"stowed", "hold"}:
                raise RuntimeError(
                    f"phase {phase!r} cannot be re-entered implicitly"
                )
        elif phase not in LEGAL_PHASE_TRANSITIONS[self.phase]:
            raise RuntimeError(
                f"unsafe exposure phase transition {self.phase!r} -> "
                f"{phase!r}"
            )
        if phase in {"release", "stowed"}:
            self.capture.release_all("sequence_release")
        elif phase in {"retract", "hold"} and self.capture.safety_latched:
            raise RuntimeError(
                f"cannot enter {phase!r} while capture safety is latched: "
                f"{self.capture.latch_reason}"
            )
        self.force_controller.synchronize(targets)
        self.phase = phase
        self.history.append({"event": "phase", "phase": phase, "targets": targets})
        return targets

    def establish_capture(self) -> dict[str, Any]:
        if self.phase != "capture":
            raise RuntimeError(
                "establish_capture requires the commanded capture phase"
            )
        cells = self.capture.capture()
        result = {
            "capture_authored_pending_confirmation": True,
            "cell_count": len(cells),
            "safety_latched": self.capture.safety_latched,
            "latch_reason": self.capture.latch_reason,
        }
        self.history.append({"event": "capture_authored", **result})
        return result

    def preflight_capture(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> dict[str, Any]:
        if self.phase != "capture":
            raise RuntimeError(
                "preflight_capture requires the commanded capture phase"
            )
        result = self.capture.preflight_capture_from_scene(evidence)
        self.force_controller.synchronize(
            evidence.measured_joint_positions_m
        )
        self.history.append({"event": "capture_preflight", **result})
        return result

    def confirm_capture(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> dict[str, Any]:
        if self.phase != "capture":
            raise RuntimeError(
                "confirm_capture requires the commanded capture phase"
            )
        result = self.capture.confirm_capture_from_scene(evidence)
        self.force_controller.synchronize(
            evidence.measured_joint_positions_m
        )
        self.history.append(
            {
                "event": (
                    "capture_confirmed"
                    if result["capture_epoch_confirmed"]
                    else "capture_confirmation_sample"
                ),
                **result,
            }
        )
        return result

    def hold_update(
        self,
        evidence: AtraumaticExposureSceneEvidence,
    ) -> dict[str, Any]:
        if self.phase not in {"retract", "hold"}:
            raise RuntimeError(
                "hold_update requires an explicit retract or hold phase"
            )
        if self.capture.safety_latched:
            raise RuntimeError(
                "capture safety latch is active; explicit recapture is "
                f"required ({self.capture.latch_reason})"
            )
        try:
            left_force = evidence.total_force_n("left")
            right_force = evidence.total_force_n("right")
            candidate = replace(self.force_controller)
            candidate.synchronize(
                evidence.measured_joint_positions_m,
                reset_integral=False,
            )
            control = candidate.update(
                dt=evidence.dt_s,
                visible_fraction=evidence.visible_fraction,
                left_force_n=left_force,
                right_force_n=right_force,
            )
        except Exception:
            self.capture.safety_latched = True
            self.capture.capture_epoch_confirmed = False
            self.capture.latch_reason = (
                "hold_control_preflight_failed_requires_release"
            )
            raise
        try:
            release = self.capture.update_from_scene(evidence)
        except Exception:
            if not self.capture.safety_latched:
                self.capture.safety_latched = True
                self.capture.capture_epoch_confirmed = False
                self.capture.latch_reason = (
                    "hold_evidence_rejected_requires_release"
                )
            raise
        if self.capture.safety_latched:
            joint_targets = phase_targets("overload_relief")
            candidate.synchronize(joint_targets)
            self.phase = "overload_relief"
            mode = f"{control.mode}_capture_latched"
        else:
            joint_targets = control.joint_targets
            self.phase = "hold"
            mode = control.mode
        self.force_controller.commit_state_from(candidate)
        result = {
            "phase": self.phase,
            "joint_targets": joint_targets,
            "visible_fraction": evidence.visible_fraction,
            "left_force_n": left_force,
            "right_force_n": right_force,
            "force_error_n": control.force_error_n,
            "exposure_error": control.exposure_error,
            "overload": control.overload,
            "capture_release": release,
            "capture_safety_latched": self.capture.safety_latched,
            "capture_latch_reason": self.capture.latch_reason,
            "mode": mode,
            "physics_step": evidence.physics_step,
            "simulation_time_s": evidence.simulation_time_s,
            "evidence_digest_sha256": evidence.evidence_digest_sha256,
            "scope": (
                "simulator_mechanics_only_not_tissue_injury_or_clinical_safety"
            ),
        }
        self.history.append({"event": "hold_update", **result})
        return result
