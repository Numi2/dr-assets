# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar Adaptive Hemostasis Robot.

The payload replaces the Panda hand at ``panda_link8``. Runtime helpers provide
conserved blood-volume bookkeeping, PhysX particle emission and suction,
temporary vessel compression, physical clip retention, staged patch bonding,
and a reduced-order pressure/flow verification model. All parameters are
provisional research values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import math

from orbit.surgical.assets.adaptive_hemostasis_scene_evidence import (
    ClipMechanicsSample,
    CompressionMechanicsSample,
    HemostasisEvidenceCursor,
    HemostasisSceneEvidence,
    HemostasisSceneEvidenceSource,
    PatchMechanicsSample,
    VesselMechanicsSample,
)

CATALOG_SUBPATH = "Props/SurgicalHemostasis/AdaptiveHemostasisRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_adaptive_hemostasis_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_adaptive_hemostasis_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_adaptive_hemostasis_tool_rigid_proxy.usda"
CLIP_USD = ROOT / "dranmar_hemostatic_clip.usda"
PATCH_USD = ROOT / "dranmar_hemostatic_patch.usda"
PATCH_PROXY_USD = ROOT / "dranmar_hemostatic_patch_rigid_proxy.usda"
VESSEL_USD = ROOT / "dranmar_bleeding_vessel_demo.usda"
DROPLET_USD = ROOT / "dranmar_blood_droplet.usda"

VALID_BINARY_STATES = frozenset({"loaded", "empty"})
VALID_IRRIGATION_STATES = frozenset({"full", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
BLOOD_PARTICLE_RADIUS_M = 0.00065
BLOOD_PARTICLE_VOLUME_ML = 0.002
TOOL_JOINTS = {
    "left_compression":"left_compression_joint","right_compression":"right_compression_joint",
    "left_pad_compliance":"left_pad_compliance_joint","right_pad_compliance":"right_pad_compliance_joint",
    "left_clip_jaw":"left_clip_jaw_joint","right_clip_jaw":"right_clip_jaw_joint",
    "clip_driver":"clip_driver_joint","patch_carousel":"patch_carousel_joint",
    "patch_applicator":"patch_applicator_joint","suction_valve":"suction_valve_joint","irrigation_valve":"irrigation_valve_joint",
}
TOOL_FRAME_PATHS = {
    "panda_link8_mount":"Links/Mount/Frames/panda_link8_mount","hemostasis_tcp":"Links/Mount/Frames/hemostasis_tcp",
    "bleeding_source_reference":"Links/Mount/Frames/bleeding_source_reference","suction_center":"Links/Mount/Frames/suction_center",
    "irrigation_center":"Links/Mount/Frames/irrigation_center","clip_forming_center":"Links/Mount/Frames/clip_forming_center",
    "clip_exit":"Links/ClipDriver/Frames/clip_exit","patch_application":"Links/PatchPlaten/Frames/patch_application",
    "left_compression_contact":"Links/LeftPad/Frames/left_compression_contact","right_compression_contact":"Links/RightPad/Frames/right_compression_contact",
    "left_clip_contact":"Links/LeftClipJaw/Frames/left_clip_contact","right_clip_contact":"Links/RightClipJaw/Frames/right_clip_contact",
    "flow_probe":"Links/Mount/Frames/flow_probe","count_reference":"Links/Mount/Frames/count_reference","disposal_reference":"Links/Mount/Frames/disposal_reference",
    "fluorescence_camera":"Links/Mount/Frames/fluorescence_camera",
    "rgb_camera_left":"Links/Mount/Frames/rgb_camera_left",
    "rgb_camera_right":"Links/Mount/Frames/rgb_camera_right",
}
REGISTERED_CAMERA_FRAMES = (
    "rgb_camera_left", "rgb_camera_right", "fluorescence_camera",
)


def frame_path(tool_path: str, name: str) -> str:
    try: suffix=TOOL_FRAME_PATHS[name]
    except KeyError as exc: raise KeyError(f"Unknown hemostasis frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    return value.torch if hasattr(value,"torch") else value


def _xyzw_from_wxyz(orientation_wxyz) -> tuple[float, float, float, float]:
    values=tuple(float(value) for value in orientation_wxyz)
    if len(values)!=4 or not all(math.isfinite(value) for value in values):raise ValueError("orientation_wxyz must contain four finite values")
    if abs(math.sqrt(sum(value*value for value in values))-1.0)>1.0e-4:raise ValueError("orientation_wxyz must be a unit quaternion")
    w,x,y,z=values
    return x,y,z,w


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed: raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def make_tool_cfg(prim_path: str="/World/DrAnmarAdaptiveHemostasisTool", *, clip_state="loaded", patch_state="loaded", irrigation_state="full", collection_state="empty", position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    _check(clip_state,VALID_BINARY_STATES,"clip_state");_check(patch_state,VALID_BINARY_STATES,"patch_state");_check(irrigation_state,VALID_IRRIGATION_STATES,"irrigation_state");_check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_STANDALONE_USD),variants={"clip_state":clip_state,"patch_state":patch_state,"irrigation_state":irrigation_state,"collection_state":collection_state},activate_contact_sensors=True,articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=20,solver_velocity_iteration_count=6)),
        init_state=ArticulationCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz),joint_pos={name:0.0 for name in TOOL_JOINTS.values()}),
        actuators={
            "compression":ImplicitActuatorCfg(joint_names_expr=[".*compression_joint",".*pad_compliance_joint"],effort_limit_sim=120.0,velocity_limit_sim=0.18,stiffness=5200.0,damping=190.0),
            "clip":ImplicitActuatorCfg(joint_names_expr=[".*clip_jaw_joint","clip_driver_joint"],effort_limit_sim=240.0,velocity_limit_sim=0.30,stiffness=14000.0,damping=300.0),
            "patch":ImplicitActuatorCfg(joint_names_expr=["patch_.*_joint"],effort_limit_sim=90.0,velocity_limit_sim=1.5,stiffness=6500.0,damping=180.0),
            "valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
        },
    )


def make_rigid_proxy_cfg(prim_path="/World/DrAnmarAdaptiveHemostasisProxy", *, position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(prim_path=prim_path,spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD),activate_contact_sensors=True),init_state=RigidObjectCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz)))


def _spawn_single_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics
    robot = spawn_from_usd(prim_path, cfg, translation, orientation)
    stage = get_current_stage()
    robot_path = Sdf.Path(prim_path)
    names_to_disable = {
        "panda_hand_joint", "panda_hand", "panda_finger_joint1",
        "panda_finger_joint2", "panda_leftfinger", "panda_rightfinger",
    }
    hand_joints = [
        prim for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_hand_joint"
    ]
    if len(hand_joints) == 1:
        stock_joint = UsdPhysics.Joint(hand_joints[0])
        mount_body_paths = stock_joint.GetBody0Rel().GetTargets()
        mount_local_pos0 = stock_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        mount_local_rot0 = stock_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
    else:
        link8_paths = [
            prim.GetPath() for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_link8"
        ]
        if len(link8_paths) != 1:
            raise RuntimeError("Could not resolve the Franka hand mount")
        mount_body_paths = link8_paths
        mount_local_pos0 = Gf.Vec3f(0, 0, 0)
        half_angle = math.radians(-45.0) / 2.0
        mount_local_rot0 = Gf.Quatf(math.cos(half_angle), 0, 0, math.sin(half_angle))
    if len(mount_body_paths) != 1 or not stage.GetPrimAtPath(mount_body_paths[0]).IsValid():
        raise RuntimeError(f"Invalid Franka hand mount target: {mount_body_paths}")

    candidate_paths = [
        prim.GetPath() for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in names_to_disable
    ]
    paths_to_disable = []
    for path in sorted(candidate_paths, key=lambda item: str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)

    tool_path = f"{prim_path}/DrAnmarAdaptiveHemostasisTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(
        tool_path,
        {
            "clip_state": cfg.clip_state,
            "patch_state": cfg.patch_state,
            "irrigation_state": cfg.irrigation_state,
            "collection_state": cfg.collection_state,
        },
    )
    joint = UsdPhysics.FixedJoint.Define(
        stage, f"{prim_path}/dranmar_hemostasis_mount_joint"
    )
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_tool(prim_path: str,cfg: Any,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)


def make_franka_adaptive_hemostasis_robot_cfg(*, prim_path="/World/Robot", clip_state="loaded", patch_state="loaded", irrigation_state="full", collection_state="empty"):
    _check(clip_state,VALID_BINARY_STATES,"clip_state");_check(patch_state,VALID_BINARY_STATES,"patch_state");_check(irrigation_state,VALID_IRRIGATION_STATES,"irrigation_state");_check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    @configclass
    class FrankaHemostasisUsdCfg(sim_utils.UsdFileCfg):
        clip_state: str="loaded";patch_state: str="loaded";irrigation_state: str="full";collection_state: str="empty";func=spawn_franka_with_tool
    cfg=FRANKA_PANDA_CFG.copy();cfg.prim_path=prim_path
    cfg.spawn=FrankaHemostasisUsdCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",variants={"Gripper":"Default","Mesh":"Performance"},clip_state=clip_state,patch_state=patch_state,irrigation_state=irrigation_state,collection_state=collection_state,activate_contact_sensors=True,rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props)
    cfg.init_state.joint_pos={k:v for k,v in cfg.init_state.joint_pos.items() if "finger" not in k};cfg.init_state.joint_pos.update({name:0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators={k:v for k,v in cfg.actuators.items() if k!="panda_hand"}
    cfg.actuators.update({
        "hemostasis_compression":ImplicitActuatorCfg(joint_names_expr=[".*compression_joint",".*pad_compliance_joint"],effort_limit_sim=120.0,velocity_limit_sim=0.18,stiffness=5200.0,damping=190.0),
        "hemostasis_clip":ImplicitActuatorCfg(joint_names_expr=[".*clip_jaw_joint","clip_driver_joint"],effort_limit_sim=240.0,velocity_limit_sim=0.30,stiffness=14000.0,damping=300.0),
        "hemostasis_patch":ImplicitActuatorCfg(joint_names_expr=["patch_.*_joint"],effort_limit_sim=90.0,velocity_limit_sim=1.5,stiffness=6500.0,damping=180.0),
        "hemostasis_valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
    })
    return cfg


def _current_stage(stage=None):
    if stage is not None:return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def spawn_vessel_demo(prim_path="/World/DrAnmarBleedingVessel", *, translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    cfg=sim_utils.UsdFileCfg(usd_path=str(VESSEL_USD));return cfg.func(prim_path,cfg,translation=translation,orientation=_xyzw_from_wxyz(orientation_wxyz))


def apply_vessel_surface_deformable(root_path: str, *, self_collision=True, stage=None):
    stage=_current_stage(stage);mesh_path=f"{root_path.rstrip('/')}/VesselWall";mesh=stage.GetPrimAtPath(mesh_path)
    if not mesh or not mesh.IsValid():raise ValueError(f"No vessel wall at {mesh_path}")
    from omni.physx.scripts import deformableUtils
    ok=deformableUtils.set_physics_surface_deformable_body(stage,mesh.GetPath())
    if ok is False:raise RuntimeError(f"Failed to cook vessel surface deformable at {mesh_path}")
    mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    if mesh.HasAPI("PhysxSurfaceDeformableBodyAPI"):mesh.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
    return {"root_path":root_path,"mesh_path":mesh_path,"self_collision":bool(self_collision)}


def create_deformable_attachment(deformable_path: str, target_path: str, attachment_path: str, *, stage=None):
    """Create an overlap-prioritized attachment across Isaac generations."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    stage = _current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)
    definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if definition:
        deformable = stage.GetPrimAtPath(deformable_path)
        target = stage.GetPrimAtPath(target_path)
        mesh = UsdGeom.Mesh(deformable)
        points = list(mesh.GetPointsAttr().Get() or [])
        if not deformable.IsValid() or not mesh or not points:
            raise ValueError(f"Attachment source is not a populated mesh: {deformable_path}")
        if not target.IsValid() or not UsdGeom.Xformable(target):
            raise ValueError(f"Attachment target is not xformable: {target_path}")
        mesh_to_world = UsdGeom.Xformable(deformable).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        target_to_world = UsdGeom.Xformable(target).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_to_target = target_to_world.GetInverse()
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        ).ComputeWorldBound(target).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        center = (minimum + maximum) * 0.5
        ranked = []
        for index, point in enumerate(points):
            world = mesh_to_world.Transform(Gf.Vec3d(point))
            delta = world - center
            overlaps = all(
                minimum[axis] - 0.0025 <= world[axis] <= maximum[axis] + 0.0025
                for axis in range(3)
            )
            ranked.append((float(Gf.Dot(delta, delta)), index, world, overlaps))
        ranked.sort(key=lambda item: item[0])
        selected = [item for item in ranked if item[3]][:12]
        if len(selected) < 4:
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_path}, "
                f"target={target_path}, overlapping={len(selected)}, "
                "required=4, overlap_margin_m=0.0025"
            )
        attachment = stage.DefinePrim(
            attachment_path, "OmniPhysicsVtxXformAttachment"
        )
        attachment.CreateRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(deformable_path)]
        )
        attachment.CreateRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(target_path)]
        )
        attachment.CreateAttribute(
            "omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray
        ).Set(Vt.IntArray([item[1] for item in selected]))
        attachment.CreateAttribute(
            "omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray
        ).Set(Vt.Vec3fArray([
            Gf.Vec3f(world_to_target.Transform(item[2])) for item in selected
        ]))
        attachment.CreateAttribute(
            "omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)
        if (
            not attachment.IsValid()
            or attachment.GetTypeName() != "OmniPhysicsVtxXformAttachment"
            or not attachment.GetRelationship("omniphysics:src0").GetTargets()
            or not attachment.GetRelationship("omniphysics:src1").GetTargets()
        ):
            raise RuntimeError(f"Could not author attachment {attachment_path}")
        return "OmniPhysicsVtxXformAttachment"

    import omni.kit.commands

    def execute_and_verify(command: str, **kwargs) -> str:
        omni.kit.commands.execute(command, **kwargs)
        if not stage.GetPrimAtPath(attachment_path).IsValid():
            raise RuntimeError(f"{command} did not author {attachment_path}")
        return command
    try:
        return execute_and_verify(
            "CreateAutoDeformableAttachment",
            target_attachment_path=Sdf.Path(attachment_path),
            attachable0_path=Sdf.Path(deformable_path),
            attachable1_path=Sdf.Path(target_path),
        )
    except Exception as current_error:
        if stage.GetPrimAtPath(attachment_path).IsValid():
            stage.RemovePrim(attachment_path)
        try:
            return execute_and_verify(
                "CreatePhysicsAttachment",
                target_attachment_path=Sdf.Path(attachment_path),
                actor0_path=Sdf.Path(deformable_path),
                actor1_path=Sdf.Path(target_path),
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Could not create {attachment_path}: "
                f"current={current_error!r}; legacy={legacy_error!r}"
            ) from legacy_error


def anchor_vessel_fixture(root_path: str, *, stage=None) -> list[str]:
    """Attach the deformable vessel endpoints to the two fixture anchors."""
    from pxr import UsdPhysics
    stage = _current_stage(stage)
    root_path = root_path.rstrip("/")
    attachments_root = f"{root_path}/RuntimeFixtureAttachments"
    stage.DefinePrim(attachments_root, "Scope")
    created = []
    try:
        for label, target in (("min", "AnchorMin"), ("max", "AnchorMax")):
            path = f"{attachments_root}/{label}"
            target_path = f"{root_path}/{target}"
            target_prim = stage.GetPrimAtPath(target_path)
            if not target_prim.IsValid():
                raise ValueError(f"Vessel fixture anchor is missing: {target_path}")
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(target_prim)
            rigid_body.CreateRigidBodyEnabledAttr(True)
            rigid_body.CreateKinematicEnabledAttr(True)
            create_deformable_attachment(
                f"{root_path}/VesselWall", target_path, path, stage=stage
            )
            created.append(path)
    except Exception:
        remove_prims(created, stage=stage)
        raise
    return created


def remove_prims(paths: Iterable[str], *, stage=None):
    stage=_current_stage(stage)
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():stage.RemovePrim(path)


def _nonnegative_finite(value: float, label: str) -> float:
    amount = float(value)
    if not math.isfinite(amount) or amount < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return amount


def _fraction(value: float, label: str) -> float:
    amount = float(value)
    if not math.isfinite(amount):
        raise ValueError(f"{label} must be finite")
    return max(0.0, min(1.0, amount))


@dataclass
class HemorrhageLedger:
    initial_reservoir_ml: float=250.0
    reservoir_ml: float=250.0
    emitted_ml: float=0.0
    active_particle_ml: float=0.0
    suctioned_ml: float=0.0
    spilled_ml: float=0.0
    discarded_ml: float=0.0
    def __post_init__(self):
        for name in (
            "initial_reservoir_ml", "reservoir_ml", "emitted_ml",
            "active_particle_ml", "suctioned_ml", "spilled_ml", "discarded_ml",
        ):
            setattr(self, name, _nonnegative_finite(getattr(self, name), name))
        if self.reservoir_ml > self.initial_reservoir_ml:
            raise ValueError("reservoir_ml cannot exceed initial_reservoir_ml")
    def emit(self,volume_ml: float):
        amount=min(_nonnegative_finite(volume_ml,"volume_ml"),self.reservoir_ml);self.reservoir_ml-=amount;self.emitted_ml+=amount;self.active_particle_ml+=amount;return amount
    def suction(self,volume_ml: float):
        amount=min(_nonnegative_finite(volume_ml,"volume_ml"),self.active_particle_ml);self.active_particle_ml-=amount;self.suctioned_ml+=amount;return amount
    def spill(self,volume_ml: float):
        amount=min(_nonnegative_finite(volume_ml,"volume_ml"),self.active_particle_ml);self.active_particle_ml-=amount;self.spilled_ml+=amount;return amount
    def discard(self,volume_ml: float):
        amount=min(_nonnegative_finite(volume_ml,"volume_ml"),self.active_particle_ml);self.active_particle_ml-=amount;self.discarded_ml+=amount;return amount
    @property
    def conservation_error_ml(self):return self.initial_reservoir_ml-(self.reservoir_ml+self.active_particle_ml+self.suctioned_ml+self.spilled_ml+self.discarded_ml)
    def snapshot(self):
        return {
            "initial_reservoir_ml":self.initial_reservoir_ml,
            "reservoir_ml":self.reservoir_ml,
            "emitted_ml":self.emitted_ml,
            "active_particle_ml":self.active_particle_ml,
            "suctioned_ml":self.suctioned_ml,
            "spilled_ml":self.spilled_ml,
            "discarded_ml":self.discarded_ml,
            "conservation_error_ml":self.conservation_error_ml,
        }


@dataclass
class ReducedOrderBleedModel:
    """Compatibility-named view over the shared vessel observation.

    This class does not evolve vessel state or recompute leakage.  Flow and
    residual defect geometry remain authoritative outputs of shared
    ``VesselMechanics``.
    """

    last_vessel_sample: VesselMechanicsSample | None = field(default=None, init=False)

    def update_from_scene(self, vessel: VesselMechanicsSample) -> None:
        if not isinstance(vessel, VesselMechanicsSample):
            raise TypeError("bleed model requires a VesselMechanicsSample")
        self.last_vessel_sample = vessel

    def _sample(self, vessel: VesselMechanicsSample | None) -> VesselMechanicsSample:
        sample = vessel if vessel is not None else self.last_vessel_sample
        if sample is None:
            raise RuntimeError("scene vessel mechanics have not been sampled")
        return sample

    def effective_area_m2(self, vessel: VesselMechanicsSample | None = None):
        return self._sample(vessel).residual_defect_area_m2

    def observed_flow_ml_min(self, vessel: VesselMechanicsSample | None = None):
        return self._sample(vessel).measured_leak_flow_ml_min


def ensure_blood_particle_system(*, physics_scene_path="/physicsScene", root_path="/World/DrAnmarBlood", system_path=None, particles_path=None, material_path=None, stage=None):
    stage=_current_stage(stage)
    from pxr import Sdf, UsdGeom, UsdPhysics
    from omni.physx.scripts import particleUtils, physicsUtils
    stage.DefinePrim(root_path,"Scope")
    system_path=system_path or f"{root_path}/ParticleSystem";particles_path=particles_path or f"{root_path}/Particles";material_path=material_path or f"{root_path}/PBDMaterial"
    scene_path=Sdf.Path(physics_scene_path)
    if not stage.GetPrimAtPath(scene_path).IsValid():UsdPhysics.Scene.Define(stage,scene_path)
    if not stage.GetPrimAtPath(material_path).IsValid():particleUtils.add_pbd_particle_material(stage,Sdf.Path(material_path),friction=0.08,viscosity=0.0035,cohesion=0.01,surface_tension=0.02)
    if not stage.GetPrimAtPath(system_path).IsValid():particleUtils.add_physx_particle_system(stage=stage,particle_system_path=Sdf.Path(system_path),simulation_owner=scene_path,particle_contact_offset=BLOOD_PARTICLE_RADIUS_M*1.15,rest_offset=BLOOD_PARTICLE_RADIUS_M*0.9,solid_rest_offset=BLOOD_PARTICLE_RADIUS_M*1.8,fluid_rest_offset=BLOOD_PARTICLE_RADIUS_M*0.92)
    physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system_path),Sdf.Path(material_path))
    if not stage.GetPrimAtPath(particles_path).IsValid():
        particleUtils.add_physx_particleset_points(stage,Sdf.Path(particles_path),[],[],[],Sdf.Path(system_path),True,True,0,1.0,BLOOD_PARTICLE_RADIUS_M*2.0)
        UsdGeom.Points(stage.GetPrimAtPath(particles_path)).GetWidthsAttr().Set([])
    return {"root_path":root_path,"particle_system_path":system_path,"particles_path":particles_path,"material_path":material_path}


def emit_blood_burst(positions: Sequence[Sequence[float]], velocities: Sequence[Sequence[float]], *, particle_volume_ml=BLOOD_PARTICLE_VOLUME_ML, system_path="/World/DrAnmarBlood/ParticleSystem", particles_path="/World/DrAnmarBlood/Particles", ledger: HemorrhageLedger|None=None, stage=None):
    stage=_current_stage(stage);ensure_blood_particle_system(system_path=system_path,particles_path=particles_path,stage=stage)
    from pxr import Gf, UsdGeom
    particle_volume_ml=_nonnegative_finite(particle_volume_ml,"particle_volume_ml")
    if particle_volume_ml == 0.0:raise ValueError("particle_volume_ml must be positive")
    positions=list(positions);velocities=list(velocities)
    if len(positions)!=len(velocities):raise ValueError("positions and velocities must have equal lengths")
    converted_positions=[];converted_velocities=[]
    for position,velocity in zip(positions,velocities):
        if len(position)!=3 or len(velocity)!=3:raise ValueError("particle vectors must be length three")
        values=[float(v) for v in (*position,*velocity)]
        if not all(math.isfinite(v) for v in values):raise ValueError("particle vectors must be finite")
        converted_positions.append(Gf.Vec3f(*values[:3]));converted_velocities.append(Gf.Vec3f(*values[3:]))
    allowed=len(converted_positions);remainder=0.0
    if ledger is not None:
        debited=ledger.emit(allowed*particle_volume_ml);allowed=int(debited/particle_volume_ml);actual=allowed*particle_volume_ml;remainder=debited-actual
        ledger.reservoir_ml+=remainder;ledger.emitted_ml-=remainder;ledger.active_particle_ml-=remainder
    converted_positions=converted_positions[:allowed];converted_velocities=converted_velocities[:allowed]
    if not converted_positions:return {"particle_count":0,"emitted_ml":0.0,"quantization_remainder_ml":remainder}
    points=UsdGeom.Points(stage.GetPrimAtPath(particles_path))
    current_positions=list(points.GetPointsAttr().Get() or []);current_velocities=list(points.GetVelocitiesAttr().Get() or []);current_widths=list(points.GetWidthsAttr().Get() or [])
    current_positions.extend(converted_positions);current_velocities.extend(converted_velocities);current_widths.extend([BLOOD_PARTICLE_RADIUS_M*2.0]*allowed)
    points.GetPointsAttr().Set(current_positions);points.GetVelocitiesAttr().Set(current_velocities);points.GetWidthsAttr().Set(current_widths)
    return {"particle_count":allowed,"emitted_ml":allowed*particle_volume_ml,"quantization_remainder_ml":remainder,"total_particle_count":len(current_positions)}


@dataclass
class AnnularSuctionController:
    center_world: tuple[float,float,float]
    capture_radius_m: float=0.010
    attraction_radius_m: float=0.050
    acceleration_m_s2: float=14.0
    particle_volume_ml: float=0.002
    def __post_init__(self):
        if len(self.center_world)!=3 or not all(math.isfinite(float(v)) for v in self.center_world):raise ValueError("center_world must contain three finite values")
        self.capture_radius_m=_nonnegative_finite(self.capture_radius_m,"capture_radius_m");self.attraction_radius_m=_nonnegative_finite(self.attraction_radius_m,"attraction_radius_m");self.acceleration_m_s2=_nonnegative_finite(self.acceleration_m_s2,"acceleration_m_s2");self.particle_volume_ml=_nonnegative_finite(self.particle_volume_ml,"particle_volume_ml")
        if self.capture_radius_m>self.attraction_radius_m:raise ValueError("capture radius cannot exceed attraction radius")
    def update_positions_velocities(self,positions,velocities,dt: float,ledger: HemorrhageLedger|None=None):
        import numpy as np
        pos=np.asarray(tensor_value(positions),dtype=float);vel=np.asarray(tensor_value(velocities),dtype=float)
        if pos.ndim!=2 or pos.shape[1]!=3 or vel.shape!=pos.shape:raise ValueError("positions and velocities must be matching Nx3 arrays")
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():raise ValueError("positions and velocities must be finite")
        center=np.asarray(self.center_world,dtype=float);delta=center-pos;dist=np.linalg.norm(delta,axis=-1);direction=delta/np.maximum(dist[...,None],1e-9);mask=dist<self.attraction_radius_m;vel[mask]+=direction[mask]*self.acceleration_m_s2*_nonnegative_finite(dt,"dt");captured=dist<self.capture_radius_m
        if ledger is not None:
            maximum_captures=int(math.floor((ledger.active_particle_ml+1.0e-12)/self.particle_volume_ml))
            candidate_indices=np.flatnonzero(captured)
            if len(candidate_indices)>maximum_captures:
                captured[:]=False
                captured[candidate_indices[:maximum_captures]]=True
        count=int(np.count_nonzero(captured))
        if ledger is not None and count:
            requested=count*self.particle_volume_ml
            accounted=ledger.suction(requested)
            if not math.isclose(accounted,requested,rel_tol=0.0,abs_tol=1.0e-12):
                raise RuntimeError(
                    "Particle removal and hemorrhage ledger diverged: "
                    f"removed_ml={requested}, accounted_ml={accounted}"
                )
        return pos[~captured],vel[~captured],captured
    def update_particle_set(self,dt: float,ledger: HemorrhageLedger,*,stage=None,particles_path="/World/DrAnmarBlood/Particles"):
        from pxr import UsdGeom,Vt
        stage=_current_stage(stage);points=UsdGeom.Points(stage.GetPrimAtPath(particles_path))
        if not points:
            raise ValueError(f"No blood particle set at {particles_path}")
        positions=list(points.GetPointsAttr().Get() or [])
        velocities=list(points.GetVelocitiesAttr().Get() or [])
        if len(velocities)!=len(positions):
            raise RuntimeError(
                f"Blood particle position/velocity count mismatch at {particles_path}: "
                f"positions={len(positions)}, velocities={len(velocities)}"
            )
        widths=list(points.GetWidthsAttr().Get() or [BLOOD_PARTICLE_RADIUS_M*2.0]*len(positions))
        if len(widths)!=len(positions):
            raise RuntimeError(
                f"Blood particle position/width count mismatch at {particles_path}: "
                f"positions={len(positions)}, widths={len(widths)}"
            )
        kept_positions,kept_velocities,captured=self.update_positions_velocities(
            positions,velocities,dt,ledger
        )
        kept_widths=[width for width,was_captured in zip(widths,captured) if not was_captured]
        points.GetPointsAttr().Set(Vt.Vec3fArray(kept_positions))
        points.GetVelocitiesAttr().Set(Vt.Vec3fArray(kept_velocities))
        points.GetWidthsAttr().Set(kept_widths)
        return {
            "active_particle_count":len(kept_positions),
            "captured_particle_count":int(captured.sum()),
            "suctioned_ml":int(captured.sum())*self.particle_volume_ml,
            "particles_path":particles_path,
        }


@dataclass
class TemporaryCompressionController:
    tool_path: str
    vessel_path: str
    target_force_per_pad_n: float=1.8
    soft_force_limit_n: float=4.0
    hard_release_limit_n: float=7.0
    attachment_paths: list[str]=field(default_factory=list)
    engaged: bool=False
    def __post_init__(self):
        self.target_force_per_pad_n=_nonnegative_finite(self.target_force_per_pad_n,"target_force_per_pad_n");self.soft_force_limit_n=_nonnegative_finite(self.soft_force_limit_n,"soft_force_limit_n");self.hard_release_limit_n=_nonnegative_finite(self.hard_release_limit_n,"hard_release_limit_n")
        if not self.target_force_per_pad_n<=self.soft_force_limit_n<=self.hard_release_limit_n:raise ValueError("compression limits must be target <= soft <= hard")
    def engage(self,*,stage=None):
        if self.engaged:return list(self.attachment_paths)
        stage=_current_stage(stage);pairs=[("LeftPad","left"),("RightPad","right")]
        created=[]
        try:
            for link,label in pairs:
                path=f"{self.tool_path}/RuntimeAttachments/compression_{label}"
                stage.DefinePrim(f"{self.tool_path}/RuntimeAttachments","Scope")
                create_deformable_attachment(self.vessel_path,f"{self.tool_path}/Links/{link}/Collisions/VesselCaptureVolume",path,stage=stage);created.append(path)
        except Exception:
            remove_prims(created,stage=stage);raise
        self.attachment_paths=created;self.engaged=True;return list(created)
    def release(self,*,stage=None):remove_prims(self.attachment_paths,stage=stage);self.attachment_paths.clear();self.engaged=False
    def update_from_scene(self,mechanics: CompressionMechanicsSample,*,stage=None):
        if not isinstance(mechanics, CompressionMechanicsSample):
            raise TypeError("compression controller requires CompressionMechanicsSample")
        left=mechanics.left_normal_force_n;right=mechanics.right_normal_force_n
        hard=max(left,right)>self.hard_release_limit_n;soft=max(left,right)>self.soft_force_limit_n
        if hard and self.engaged:self.release(stage=stage)
        return {"mode":"hard_release" if hard else "soft_limit" if soft else "controlled","left_force_n":left,"right_force_n":right,"left_contact_area_m2":mechanics.left_contact_area_m2,"right_contact_area_m2":mechanics.right_contact_area_m2,"left_mean_pressure_pa":mechanics.left_mean_pressure_pa,"right_mean_pressure_pa":mechanics.right_mean_pressure_pa,"left_target_error_n":left-self.target_force_per_pad_n,"right_target_error_n":right-self.target_force_per_pad_n,"engaged":self.engaged}


def _spawn_reference_at_transform(stage,prim_path: str,usd_path: Path,world_transform: Any,variants: dict[str,str]|None=None):
    from pxr import Gf, UsdGeom
    prim=stage.DefinePrim(prim_path,"Xform");prim.GetReferences().AddReference(str(usd_path))
    matrix=Gf.Matrix4d(world_transform);UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)
    if variants:
        for name,value in variants.items():prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(value)
    return prim


def deploy_formed_clip(prim_path: str, world_transform: Any, vessel_path: str, *, stage=None):
    stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,CLIP_USD,world_transform,{"state":"formed"});stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
    try:
        for name in ("LeftLegAttachment","RightLegAttachment"):
            ap=f"{prim_path}/Attachments/{name}";create_deformable_attachment(vessel_path,f"{prim_path}/Collisions/{name}",ap,stage=stage);created.append(ap)
    except Exception:
        remove_prims(created+[prim_path],stage=stage);raise
    return {"clip_path":prim_path,"attachment_paths":created,"state":"formed"}


@dataclass
class ClipRetentionBond:
    clip_path: str
    attachment_paths: list[str]
    last_scene_step: int=-1


@dataclass
class ClipRetentionController:
    minimum_contact_area_m2: float=2.0e-6
    minimum_contact_pressure_pa: float=500.0
    maximum_residual_gap_m: float=0.0010
    minimum_formed_span_m: float=0.0020
    maximum_damage_fraction: float=0.95
    maximum_retention_slip_speed_m_s: float=0.001
    bonds: list[ClipRetentionBond]=field(default_factory=list)
    def __post_init__(self):
        self.minimum_contact_area_m2=_nonnegative_finite(self.minimum_contact_area_m2,"minimum_contact_area_m2")
        self.minimum_contact_pressure_pa=_nonnegative_finite(self.minimum_contact_pressure_pa,"minimum_contact_pressure_pa")
        self.maximum_residual_gap_m=_nonnegative_finite(self.maximum_residual_gap_m,"maximum_residual_gap_m")
        self.minimum_formed_span_m=_nonnegative_finite(self.minimum_formed_span_m,"minimum_formed_span_m")
        self.maximum_damage_fraction=_fraction(self.maximum_damage_fraction,"maximum_damage_fraction")
        self.maximum_retention_slip_speed_m_s=_nonnegative_finite(self.maximum_retention_slip_speed_m_s,"maximum_retention_slip_speed_m_s")
        if self.minimum_contact_pressure_pa==0.0:raise ValueError("minimum_contact_pressure_pa must be positive")
    def register(self,deployment):
        b=ClipRetentionBond(str(deployment["clip_path"]),list(deployment["attachment_paths"]));self.bonds.append(b);return b
    def update_from_scene(self,bond: ClipRetentionBond,mechanics: ClipMechanicsSample,*,stage=None):
        if mechanics.clip_path != bond.clip_path:
            raise ValueError(f"Clip evidence path mismatch: bond={bond.clip_path}, evidence={mechanics.clip_path}")
        if mechanics.step_index <= bond.last_scene_step:
            raise ValueError("clip mechanics evidence must advance monotonically")
        bond.last_scene_step=mechanics.step_index
        contact_qualified=(
            mechanics.contact_area_m2>=self.minimum_contact_area_m2
            and mechanics.contact_pressure_pa>=self.minimum_contact_pressure_pa
        )
        form_qualified=(
            mechanics.residual_gap_m<=self.maximum_residual_gap_m
            and mechanics.formed_span_m>=self.minimum_formed_span_m
        )
        interface_resultant_force_n=mechanics.interface_traction_pa*mechanics.contact_area_m2
        mechanically_failed=mechanics.damage_fraction>self.maximum_damage_fraction
        capacity_qualified=(
            mechanics.retention_capacity_n>0.0
            and mechanics.retention_load_n<=mechanics.retention_capacity_n
        )
        slip_qualified=(
            mechanics.max_relative_slip_speed_m_s
            <= self.maximum_retention_slip_speed_m_s
        )
        loaded_retention_observed=mechanics.retention_load_n>0.0
        expected_attachment_ids=tuple(sorted(bond.attachment_paths))
        live_attachment_ids=tuple(sorted(mechanics.live_attachment_prim_ids))
        attachment_evidence_matches=bool(expected_attachment_ids) and (
            live_attachment_ids==expected_attachment_ids
        )
        if mechanically_failed and bond.attachment_paths:
            remove_prims(bond.attachment_paths,stage=stage)
            bond.attachment_paths.clear()
        mechanically_qualified=(
            contact_qualified
            and form_qualified
            and capacity_qualified
            and slip_qualified
            and not mechanically_failed
        )
        retained=(
            mechanically_qualified
            and loaded_retention_observed
            and attachment_evidence_matches
        )
        return {
            "retained":retained,
            "mechanically_qualified":mechanically_qualified,
            "contact_qualified":contact_qualified,
            "form_qualified":form_qualified,
            "capacity_qualified":capacity_qualified,
            "slip_qualified":slip_qualified,
            "loaded_retention_observed":loaded_retention_observed,
            "mechanically_failed":mechanically_failed,
            "residual_gap_m":mechanics.residual_gap_m,
            "formed_span_m":mechanics.formed_span_m,
            "contact_area_m2":mechanics.contact_area_m2,
            "contact_pressure_pa":mechanics.contact_pressure_pa,
            "interface_resultant_force_n":interface_resultant_force_n,
            "retention_load_n":mechanics.retention_load_n,
            "retention_capacity_n":mechanics.retention_capacity_n,
            "retention_utilization":mechanics.retention_utilization,
            "measured_tangential_force_n":mechanics.measured_tangential_force_n,
            "max_relative_slip_speed_m_s":mechanics.max_relative_slip_speed_m_s,
            "plastic_curvature_1_m":mechanics.plastic_curvature_1_m,
            "damage_fraction":mechanics.damage_fraction,
            "expected_attachment_prim_ids":expected_attachment_ids,
            "live_attachment_prim_ids":live_attachment_ids,
            "attachment_evidence_matches":attachment_evidence_matches,
            "scene_step":mechanics.step_index,
        }


@dataclass
class PatchBond:
    patch_path: str
    attachment_paths: list[str]
    last_scene_step: int=-1


@dataclass
class HemostaticPatchBondController:
    minimum_contact_fraction: float=0.60
    minimum_contact_pressure_pa: float=500.0
    maximum_interface_separation_m: float=0.0015
    bonds: list[PatchBond]=field(default_factory=list)
    def __post_init__(self):
        self.minimum_contact_fraction=_fraction(self.minimum_contact_fraction,"minimum_contact_fraction");self.minimum_contact_pressure_pa=_nonnegative_finite(self.minimum_contact_pressure_pa,"minimum_contact_pressure_pa");self.maximum_interface_separation_m=_nonnegative_finite(self.maximum_interface_separation_m,"maximum_interface_separation_m")
        if self.minimum_contact_pressure_pa==0.0:raise ValueError("minimum_contact_pressure_pa must be positive")
    def deploy(self,prim_path: str,world_transform: Any,vessel_path: str,*,stage=None):
        stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,PATCH_PROXY_USD,world_transform);stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
        try:
            for i in range(8):
                ap=f"{prim_path}/Attachments/bond_{i:02d}";create_deformable_attachment(vessel_path,f"{prim_path}/Collisions/BondCell_{i:02d}",ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created+[prim_path],stage=stage);raise
        b=PatchBond(prim_path,created);self.bonds.append(b);return b
    def update_from_scene(self,bond: PatchBond,mechanics: PatchMechanicsSample,*,stage=None):
        if mechanics.patch_path != bond.patch_path:
            raise ValueError(f"Patch evidence path mismatch: bond={bond.patch_path}, evidence={mechanics.patch_path}")
        if mechanics.step_index <= bond.last_scene_step:
            raise ValueError("patch mechanics evidence must advance monotonically")
        bond.last_scene_step=mechanics.step_index
        expected_attachment_ids=tuple(sorted(bond.attachment_paths))
        live_attachment_ids=tuple(sorted(mechanics.live_attachment_prim_ids))
        attachment_evidence_matches=bool(expected_attachment_ids) and (
            live_attachment_ids==expected_attachment_ids
        )
        contact_qualified=(
            mechanics.contact_fraction>=self.minimum_contact_fraction
            and mechanics.mean_contact_pressure_pa>=self.minimum_contact_pressure_pa
            and mechanics.interface_separation_m<=self.maximum_interface_separation_m
            and not mechanics.cohesive_failed
        )
        failed=(
            mechanics.cohesive_failed
            or mechanics.interface_separation_m>self.maximum_interface_separation_m
        )
        if failed and bond.attachment_paths:
            remove_prims(bond.attachment_paths,stage=stage);bond.attachment_paths.clear()
        return {
            "broken":failed or not attachment_evidence_matches,
            "mechanically_qualified":contact_qualified and attachment_evidence_matches,
            "contact_fraction":mechanics.contact_fraction,
            "mean_contact_pressure_pa":mechanics.mean_contact_pressure_pa,
            "interface_traction_n":mechanics.interface_traction_n,
            "interface_separation_m":mechanics.interface_separation_m,
            "cohesive_damage_fraction":mechanics.cohesive_damage_fraction,
            "cohesive_failed":mechanics.cohesive_failed,
            "surface_wetness_fraction":mechanics.surface_wetness_fraction,
            "interface_temperature_c":mechanics.interface_temperature_c,
            "expected_attachment_prim_ids":expected_attachment_ids,
            "live_attachment_prim_ids":live_attachment_ids,
            "attachment_evidence_matches":attachment_evidence_matches,
            "scene_step":mechanics.step_index,
        }


@dataclass
class SealVerificationController:
    maximum_flow_ml_min: float=0.1
    observation_window_s: float=5.0
    target_pressure_pa: float=26664.5
    minimum_pressure_fraction: float=0.90
    elapsed_s: float=0.0
    integrated_volume_ml: float=0.0
    integrated_upstream_pressure_pa_s: float=0.0
    peak_flow_ml_min: float=0.0
    def __post_init__(self):
        self.maximum_flow_ml_min=_nonnegative_finite(self.maximum_flow_ml_min,"maximum_flow_ml_min");self.observation_window_s=_nonnegative_finite(self.observation_window_s,"observation_window_s");self.target_pressure_pa=_nonnegative_finite(self.target_pressure_pa,"target_pressure_pa");self.minimum_pressure_fraction=_fraction(self.minimum_pressure_fraction,"minimum_pressure_fraction")
        if self.observation_window_s==0.0:raise ValueError("observation_window_s must be positive")
        if self.target_pressure_pa==0.0:raise ValueError("target_pressure_pa must be positive")
    def reset(self):self.elapsed_s=0.0;self.integrated_volume_ml=0.0;self.integrated_upstream_pressure_pa_s=0.0;self.peak_flow_ml_min=0.0
    def _integrate(self,flow_ml_min: float,dt: float):
        flow=_nonnegative_finite(flow_ml_min,"flow_ml_min");dt=_nonnegative_finite(dt,"dt");self.elapsed_s+=dt;self.integrated_volume_ml+=flow*dt/60.0;self.peak_flow_ml_min=max(self.peak_flow_ml_min,flow)
    def update_from_scene(self,evidence: HemostasisSceneEvidence,bleed_model: ReducedOrderBleedModel):
        bleed_model.update_from_scene(evidence.vessel)
        flow=bleed_model.observed_flow_ml_min(evidence.vessel)
        dt=evidence.dt_s;self._integrate(flow,dt);self.integrated_upstream_pressure_pa_s+=evidence.vessel.upstream_pressure_pa*dt
        return flow
    @property
    def average_flow_ml_min(self):return 0.0 if self.elapsed_s<=0 else self.integrated_volume_ml*60.0/self.elapsed_s
    @property
    def average_upstream_pressure_pa(self):return 0.0 if self.elapsed_s<=0 else self.integrated_upstream_pressure_pa_s/self.elapsed_s
    @property
    def complete(self):return self.elapsed_s>=self.observation_window_s
    @property
    def passed(self):return self.complete and self.average_flow_ml_min<=self.maximum_flow_ml_min and self.average_upstream_pressure_pa>=self.target_pressure_pa*self.minimum_pressure_fraction


PHASE_TARGETS={
    "inspect":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "clear":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.008,"irrigation_valve_joint":0.005},
    "compress":{"left_compression_joint":0.026,"right_compression_joint":-0.026,"left_pad_compliance_joint":-0.005,"right_pad_compliance_joint":-0.005,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.006,"irrigation_valve_joint":0.0},
    "temporary_control_check":{"left_compression_joint":0.026,"right_compression_joint":-0.026,"left_pad_compliance_joint":-0.005,"right_pad_compliance_joint":-0.005,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "clip":{"left_compression_joint":0.026,"right_compression_joint":-0.026,"left_pad_compliance_joint":-0.005,"right_pad_compliance_joint":-0.005,"left_clip_jaw_joint":0.007,"right_clip_jaw_joint":-0.007,"clip_driver_joint":0.017,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "release_compression":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.003,"irrigation_valve_joint":0.0},
    "patch":{"left_compression_joint":0.012,"right_compression_joint":-0.012,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":math.radians(90.0),"patch_applicator_joint":0.034,"suction_valve_joint":0.002,"irrigation_valve_joint":0.0},
    "pressure_challenge":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":math.radians(90.0),"patch_applicator_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "verify":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":math.radians(90.0),"patch_applicator_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "complete":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "abort":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.008,"irrigation_valve_joint":0.0},
}

def phase_targets(phase: str):
    try:return dict(PHASE_TARGETS[phase])
    except KeyError as exc:raise KeyError(f"Unknown hemostasis phase {phase!r}") from exc


@dataclass
class AdaptiveHemostasisSequenceController:
    phase: str="inspect"
    bleed_model: ReducedOrderBleedModel=field(default_factory=ReducedOrderBleedModel)
    verifier: SealVerificationController=field(default_factory=SealVerificationController)
    clip_retention: ClipRetentionController=field(default_factory=ClipRetentionController)
    patch_bonds: HemostaticPatchBondController=field(default_factory=HemostaticPatchBondController)
    evidence_source: HemostasisSceneEvidenceSource|None=None
    evidence_cursor: HemostasisEvidenceCursor=field(default_factory=HemostasisEvidenceCursor)
    history: list[str]=field(default_factory=list)
    baseline_pressure_pa: float=10665.8
    challenge_pressure_pa: float=26664.5
    requested_pressure_pa: float=field(default=10665.8,init=False)
    last_evidence: HemostasisSceneEvidence|None=field(default=None,init=False)
    def __post_init__(self):
        phase_targets(self.phase);self.baseline_pressure_pa=_nonnegative_finite(self.baseline_pressure_pa,"baseline_pressure_pa");self.challenge_pressure_pa=_nonnegative_finite(self.challenge_pressure_pa,"challenge_pressure_pa")
        if self.challenge_pressure_pa<=self.baseline_pressure_pa:raise ValueError("challenge_pressure_pa must exceed baseline_pressure_pa")
        self.requested_pressure_pa=self.baseline_pressure_pa;self.verifier.target_pressure_pa=self.challenge_pressure_pa
    def transition(self,phase: str):
        phase_targets(phase);self.phase=phase;self.history.append(phase)
        if phase=="pressure_challenge":self.requested_pressure_pa=self.challenge_pressure_pa
        elif phase in {"complete","abort"}:self.requested_pressure_pa=self.baseline_pressure_pa
        if phase=="verify":self.verifier.reset()
        return phase_targets(phase)
    def _resolve_evidence(self,evidence: HemostasisSceneEvidence|None):
        if evidence is None:
            if self.evidence_source is None:
                raise RuntimeError("HemostasisSceneEvidence or an evidence_source is required")
            evidence=self.evidence_source.sample_hemostasis_scene()
        return evidence
    def consume_scene_evidence(self,evidence: HemostasisSceneEvidence|None=None,*,dt: float|None=None,stage=None):
        evidence=self._resolve_evidence(evidence)
        if dt is not None:
            supplied_dt=_nonnegative_finite(dt,"dt")
            if not math.isclose(supplied_dt,evidence.dt_s,rel_tol=0.0,abs_tol=1.0e-9):
                raise ValueError("caller dt must match the scene evidence provenance dt")
        evidence=self.evidence_cursor.consume(evidence);self.last_evidence=evidence
        self.bleed_model.update_from_scene(evidence.vessel)
        clip_reports={}
        for bond in self.clip_retention.bonds:
            mechanics=evidence.clip_for(bond.clip_path)
            if mechanics is not None:
                clip_reports[bond.clip_path]=self.clip_retention.update_from_scene(bond,mechanics,stage=stage)
            else:
                clip_reports[bond.clip_path]={"retained":False,"mechanically_qualified":False,"reason":"missing_scene_clip_evidence"}
        patch_reports={}
        for bond in self.patch_bonds.bonds:
            mechanics=evidence.patch_for(bond.patch_path)
            if mechanics is not None:
                patch_reports[bond.patch_path]=self.patch_bonds.update_from_scene(bond,mechanics,stage=stage)
            else:
                patch_reports[bond.patch_path]={"broken":True,"mechanically_qualified":False,"reason":"missing_scene_patch_evidence"}
        return {"scene_step":evidence.step_index,"scene_time_s":evidence.time_s,"scene_dt_s":evidence.dt_s,"evidence_digest_sha256":evidence.digest_sha256,"source":evidence.source,"residual_defect_area_m2":evidence.vessel.residual_defect_area_m2,"observed_flow_ml_min":self.bleed_model.observed_flow_ml_min(evidence.vessel),"clip_reports":clip_reports,"patch_reports":patch_reports}
    def update_verification(self,dt: float|None=None,evidence: HemostasisSceneEvidence|None=None,*,stage=None):
        if self.phase!="verify":raise RuntimeError("verification evidence may only be integrated during the verify phase")
        scene_report=self.consume_scene_evidence(evidence,dt=dt,stage=stage)
        flow=self.verifier.update_from_scene(self.last_evidence,self.bleed_model)
        return {"flow_ml_min":flow,"average_flow_ml_min":self.verifier.average_flow_ml_min,"average_upstream_pressure_pa":self.verifier.average_upstream_pressure_pa,"required_upstream_pressure_pa":self.verifier.target_pressure_pa*self.verifier.minimum_pressure_fraction,"complete":self.verifier.complete,"passed":self.verifier.passed,"scene_step":scene_report["scene_step"],"scene_time_s":scene_report["scene_time_s"],"scene_dt_s":scene_report["scene_dt_s"],"evidence_digest_sha256":scene_report["evidence_digest_sha256"],"evidence_source":scene_report["source"],"residual_defect_area_m2":scene_report["residual_defect_area_m2"],"clip_reports":scene_report["clip_reports"],"patch_reports":scene_report["patch_reports"]}
