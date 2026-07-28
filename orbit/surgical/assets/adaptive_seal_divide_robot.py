# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac integration for the DrAnmar Adaptive Seal-and-Divide Robot.

The payload replaces the Panda hand at ``panda_link8``. Runtime helpers
center and compress a vascular pedicle, create two physical stump seal
bands, consume prim-bound thermal/electrical/contact evidence plus shared
cohesive and vessel observations, enforce a blade interlock, and release the
pre-authored mechanical tissue bridge from measured blade motion and contact.
All values are provisional research parameters and are not patient-care
settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable
import copy
import math

from .adaptive_seal_divide_scene_evidence import (
    AdaptiveSealDivideSceneSources,
    SealDivideEvidenceCursor,
    SealDivideSceneEvidence,
    SealDivideSceneEvidenceSource,
    SealDivideZoneMechanicsSample,
    SealDivideZoneSources,
)

CATALOG_SUBPATH = "Props/SurgicalDivision/AdaptiveSealDivideRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_adaptive_seal_divide_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_adaptive_seal_divide_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_adaptive_seal_divide_tool_rigid_proxy.usda"
VESSEL_USD = ROOT / "dranmar_seal_divide_vessel_demo.usda"
SEAL_BAND_USD = ROOT / "dranmar_tissue_seal_band.usda"
BLADE_USD = ROOT / "dranmar_division_blade_cartridge.usda"
VAPOR_USD = ROOT / "dranmar_seal_vapor_particle.usda"

# Isotropic small-strain research baseline for the demo vessel shell.  The
# modulus is the reported fresh ex-vivo porcine aorta mechanical-test mean
# (202.4 kPa), Poisson ratio 0.35 is the midpoint of the measured 0.3-0.4
# in-plane porcine arterial-wall range, and density 1060 kg/m^3 follows a
# published isotropic arterial-wall structural model.  This is not a
# calibrated patient, vessel-type, or electrosurgical tissue model.
VESSEL_SURFACE_MATERIAL = {
    "density_kg_m3":1060.0,
    "youngs_modulus_pa":202_400.0,
    "poissons_ratio":0.35,
    "surface_thickness_m":0.00068,
    "dynamic_friction":0.40,
}

VALID_CARTRIDGE_STATES = frozenset({"fresh", "spent"})
VALID_SALINE_STATES = frozenset({"full", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
VALID_ENERGY_STATES = frozenset({"ready", "fault"})
BRIDGE_PIN_COUNT = 8

TOOL_JOINTS = {
    "left_centering":"left_centering_joint",
    "right_centering":"right_centering_joint",
    "upper_jaw":"upper_jaw_joint",
    "lower_jaw":"lower_jaw_joint",
    "blade_guard":"blade_guard_joint",
    "blade":"blade_joint",
    "suction_valve":"suction_valve_joint",
    "irrigation_valve":"irrigation_valve_joint",
}
TOOL_FRAME_PATHS = {
    "panda_link8_mount":"Links/Mount/Frames/panda_link8_mount",
    "seal_divide_tcp":"Links/Mount/Frames/seal_divide_tcp",
    "tissue_center_reference":"Links/Mount/Frames/tissue_center_reference",
    "left_seal_zone":"Links/Mount/Frames/left_seal_zone",
    "right_seal_zone":"Links/Mount/Frames/right_seal_zone",
    "cut_plane":"Links/Mount/Frames/cut_plane",
    "suction_center":"Links/Mount/Frames/suction_center",
    "irrigation_center":"Links/Mount/Frames/irrigation_center",
    "thermal_camera":"Links/Mount/Frames/thermal_camera",
    "impedance_probe":"Links/Mount/Frames/impedance_probe",
    "seal_verification_probe":"Links/Mount/Frames/seal_verification_probe",
    "left_centering_contact":"Links/LeftCentering/Frames/left_centering_contact",
    "right_centering_contact":"Links/RightCentering/Frames/right_centering_contact",
    "upper_jaw_contact":"Links/UpperJaw/Frames/upper_jaw_contact",
    "lower_jaw_contact":"Links/LowerJaw/Frames/lower_jaw_contact",
    "blade_tip":"Links/BladeCarriage/Frames/blade_tip",
    "blade_guard_reference":"Links/BladeGuard/Frames/blade_guard_reference",
    "count_reference":"Links/Mount/Frames/count_reference",
    "disposal_reference":"Links/Mount/Frames/disposal_reference",
}
REGISTERED_CAMERA_FRAMES = ("thermal_camera",)

def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown seal/divide frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"

def tensor_value(value: Any):
    return value.torch if hasattr(value, "torch") else value


def _xyzw_from_wxyz(orientation_wxyz) -> tuple[float, float, float, float]:
    values=tuple(float(value) for value in orientation_wxyz)
    if len(values)!=4 or not all(math.isfinite(value) for value in values):raise ValueError("orientation_wxyz must contain four finite values")
    if abs(math.sqrt(sum(value*value for value in values))-1.0)>1.0e-4:raise ValueError("orientation_wxyz must be a unit quaternion")
    w,x,y,z=values
    return x,y,z,w


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def _finite(value: float,label: str) -> float:
    result=float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_nonnegative(value: float,label: str) -> float:
    result=_finite(value,label)
    if result<0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result

def _unit_fraction(value: float,label: str) -> float:
    result=_finite(value,label)
    if not 0.0<=result<=1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result

def make_tool_cfg(
    prim_path: str = "/World/DrAnmarAdaptiveSealDivideTool",
    *,
    cartridge_state: str = "fresh",
    saline_state: str = "full",
    collection_state: str = "empty",
    energy_state: str = "ready",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    _check(cartridge_state, VALID_CARTRIDGE_STATES, "cartridge_state")
    _check(saline_state, VALID_SALINE_STATES, "saline_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
    _check(energy_state, VALID_ENERGY_STATES, "energy_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"cartridge_state":cartridge_state,"saline_state":saline_state,"collection_state":collection_state,"energy_state":energy_state},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=24,solver_velocity_iteration_count=8),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz),joint_pos={name:0.0 for name in TOOL_JOINTS.values()}),
        actuators={
            "centering":ImplicitActuatorCfg(joint_names_expr=[".*centering_joint"],effort_limit_sim=90.0,velocity_limit_sim=0.16,stiffness=4200.0,damping=145.0),
            "seal_jaws":ImplicitActuatorCfg(joint_names_expr=[".*jaw_joint"],effort_limit_sim=360.0,velocity_limit_sim=0.10,stiffness=18000.0,damping=420.0),
            "blade_system":ImplicitActuatorCfg(joint_names_expr=["blade_guard_joint","blade_joint"],effort_limit_sim=280.0,velocity_limit_sim=0.25,stiffness=16000.0,damping=360.0),
            "valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
        },
    )

def make_rigid_proxy_cfg(prim_path="/World/DrAnmarAdaptiveSealDivideProxy", *, position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(prim_path=prim_path,spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD),activate_contact_sensors=True),init_state=RigidObjectCfg.InitialStateCfg(pos=position,rot=_xyzw_from_wxyz(orientation_wxyz)))

def _spawn_single_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics
    robot=spawn_from_usd(prim_path,cfg,translation,orientation)
    stage=get_current_stage()
    robot_path=Sdf.Path(prim_path)
    names_to_disable={
        "panda_hand_joint","panda_hand","panda_finger_joint1",
        "panda_finger_joint2","panda_leftfinger","panda_rightfinger",
    }
    hand_joints=[
        prim for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName()=="panda_hand_joint"
    ]
    if len(hand_joints)==1:
        stock_joint=UsdPhysics.Joint(hand_joints[0])
        mount_body_paths=stock_joint.GetBody0Rel().GetTargets()
        mount_local_pos0=stock_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0,0,0)
        mount_local_rot0=stock_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1,0,0,0)
    else:
        link8_paths=[
            prim.GetPath() for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.GetName()=="panda_link8"
        ]
        if len(link8_paths)!=1:
            raise RuntimeError("Could not resolve the Franka hand mount")
        mount_body_paths=link8_paths
        mount_local_pos0=Gf.Vec3f(0,0,0)
        half_angle=math.radians(-45.0)/2.0
        mount_local_rot0=Gf.Quatf(
            math.cos(half_angle),0,0,math.sin(half_angle)
        )
    if len(mount_body_paths)!=1 or not stage.GetPrimAtPath(mount_body_paths[0]).IsValid():
        raise RuntimeError(f"Invalid Franka hand mount target: {mount_body_paths}")

    candidate_paths=[
        prim.GetPath() for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in names_to_disable
    ]
    paths_to_disable=[]
    for path in sorted(candidate_paths,key=lambda item:str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)
    tool_path=f"{prim_path}/DrAnmarAdaptiveSealDivideTool"
    create_prim(tool_path,usd_path=str(TOOL_PAYLOAD_USD),stage=stage)
    select_usd_variants(tool_path,{"cartridge_state":cfg.cartridge_state,"saline_state":cfg.saline_state,"collection_state":cfg.collection_state,"energy_state":cfg.energy_state})
    joint=UsdPhysics.FixedJoint.Define(stage,f"{prim_path}/dranmar_seal_divide_mount_joint")
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
    return robot

def spawn_franka_with_tool(prim_path: str,cfg: Any,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)

def make_franka_adaptive_seal_divide_robot_cfg(*, prim_path="/World/Robot", cartridge_state="fresh", saline_state="full", collection_state="empty", energy_state="ready"):
    _check(cartridge_state,VALID_CARTRIDGE_STATES,"cartridge_state");_check(saline_state,VALID_SALINE_STATES,"saline_state");_check(collection_state,VALID_COLLECTION_STATES,"collection_state");_check(energy_state,VALID_ENERGY_STATES,"energy_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    @configclass
    class FrankaSealDivideUsdCfg(sim_utils.UsdFileCfg):
        cartridge_state: str="fresh";saline_state: str="full";collection_state: str="empty";energy_state: str="ready";func=spawn_franka_with_tool
    cfg=FRANKA_PANDA_CFG.copy();cfg.prim_path=prim_path
    cfg.spawn=FrankaSealDivideUsdCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",variants={"Gripper":"Default","Mesh":"Performance"},cartridge_state=cartridge_state,saline_state=saline_state,collection_state=collection_state,energy_state=energy_state,activate_contact_sensors=True,rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props)
    cfg.init_state.joint_pos={k:v for k,v in cfg.init_state.joint_pos.items() if "finger" not in k};cfg.init_state.joint_pos.update({name:0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators={k:v for k,v in cfg.actuators.items() if k!="panda_hand"}
    cfg.actuators.update({
        "seal_divide_centering":ImplicitActuatorCfg(joint_names_expr=[".*centering_joint"],effort_limit_sim=90.0,velocity_limit_sim=0.16,stiffness=4200.0,damping=145.0),
        "seal_divide_jaws":ImplicitActuatorCfg(joint_names_expr=[".*jaw_joint"],effort_limit_sim=360.0,velocity_limit_sim=0.10,stiffness=18000.0,damping=420.0),
        "seal_divide_blade":ImplicitActuatorCfg(joint_names_expr=["blade_guard_joint","blade_joint"],effort_limit_sim=280.0,velocity_limit_sim=0.25,stiffness=16000.0,damping=360.0),
        "seal_divide_valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
    })
    return cfg

def _current_stage(stage=None):
    if stage is not None:return stage
    import omni.usd
    return omni.usd.get_context().get_stage()

def spawn_vessel_demo(prim_path="/World/DrAnmarSealDivideVessel", *, translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    cfg=sim_utils.UsdFileCfg(usd_path=str(VESSEL_USD));return cfg.func(prim_path,cfg,translation=translation,orientation=_xyzw_from_wxyz(orientation_wxyz))

def apply_vessel_surface_deformables(root_path: str, *, self_collision=False, stage=None):
    stage=_current_stage(stage);results=[]
    from omni.physx.scripts import deformableUtils
    from pxr import Sdf,UsdPhysics,UsdShade
    root_path=root_path.rstrip("/")
    material_path=f"{root_path}/RuntimeMaterials/VesselWallSurface"
    material=UsdShade.Material.Define(stage,material_path)
    material_prim=material.GetPrim()
    for schema in (
        "OmniPhysicsBaseMaterialAPI",
        "OmniPhysicsDeformableMaterialAPI",
        "OmniPhysicsSurfaceDeformableMaterialAPI",
        "PhysxDeformableMaterialAPI",
        "PhysxSurfaceDeformableMaterialAPI",
    ):
        if schema not in material_prim.GetAppliedSchemas():
            material_prim.AddAppliedSchema(schema)
    values=VESSEL_SURFACE_MATERIAL
    material_prim.CreateAttribute(
        "omniphysics:density",Sdf.ValueTypeNames.Float
    ).Set(values["density_kg_m3"])
    material_prim.CreateAttribute(
        "omniphysics:dynamicFriction",Sdf.ValueTypeNames.Float
    ).Set(values["dynamic_friction"])
    material_prim.CreateAttribute(
        "omniphysics:youngsModulus",Sdf.ValueTypeNames.Float
    ).Set(values["youngs_modulus_pa"])
    material_prim.CreateAttribute(
        "omniphysics:poissonsRatio",Sdf.ValueTypeNames.Float
    ).Set(values["poissons_ratio"])
    material_prim.CreateAttribute(
        "omniphysics:surfaceThickness",Sdf.ValueTypeNames.Float
    ).Set(values["surface_thickness_m"])
    material_prim.CreateAttribute(
        "omniphysics:surfaceBendStiffness",Sdf.ValueTypeNames.Float
    ).Set(
        values["youngs_modulus_pa"]
        /(12.0*(1.0-values["poissons_ratio"]**2))
    )
    for child in ("LeftVesselWall","RightVesselWall"):
        mesh_path=f"{root_path}/{child}";mesh=stage.GetPrimAtPath(mesh_path)
        if not mesh or not mesh.IsValid():raise ValueError(f"No vessel wall at {mesh_path}")
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
            material,UsdShade.Tokens.weakerThanDescendants,"physics"
        )
        ok=deformableUtils.set_physics_surface_deformable_body(stage,mesh.GetPath())
        if ok is False:raise RuntimeError(f"Failed to cook vessel surface deformable at {mesh_path}")
        mesh.CreateAttribute(
            "omniphysics:restBendAnglesDefault",Sdf.ValueTypeNames.Token
        ).Set("restShapeDefault")
        mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        if mesh.HasAPI("PhysxSurfaceDeformableBodyAPI"):
            mesh.GetAttribute("physxDeformableBody:selfCollision").Set(
                bool(self_collision)
            )
            # The two vessel halves are only 0.4 mm apart.  Keep the contact
            # envelope below half that authored clearance so the solver does
            # not begin by depenetrating otherwise separated stump surfaces.
            mesh.CreateAttribute(
                "physxCollision:contactOffset",Sdf.ValueTypeNames.Float
            ).Set(0.0001)
            mesh.CreateAttribute(
                "physxCollision:restOffset",Sdf.ValueTypeNames.Float
            ).Set(0.0)
        results.append(mesh_path)
    # NVIDIA documents that element-level collision filtering is not
    # supported between two surface deformables.  These halves meet at an
    # attached seam, so use the supported prim-pair filter and let their
    # explicit bridge attachments carry the cross-seam load.
    left_prim=stage.GetPrimAtPath(f"{root_path}/LeftVesselWall")
    UsdPhysics.FilteredPairsAPI.Apply(left_prim).CreateFilteredPairsRel().AddTarget(
        f"{root_path}/RightVesselWall"
    )
    return {"root_path":root_path,"mesh_paths":results,"self_collision":bool(self_collision)}

def create_deformable_attachment(
    deformable_path: str,
    target_path: str,
    attachment_path: str,
    *,
    stage=None,
    deformable_points_world=None,
    target_to_world=None,
    attachment_frame_path=None,
    attachment_frame_to_world=None,
    excluded_vertex_indices=None,
    maximum_vertices=12,
    selected_vertex_indices_out=None,
):
    """Create and verify an overlap-prioritized attachment across Isaac versions."""
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,Vt
    stage=_current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)
    definition=Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if definition:
        deformable=stage.GetPrimAtPath(deformable_path)
        target=stage.GetPrimAtPath(target_path)
        mesh=UsdGeom.Mesh(deformable)
        points=list(mesh.GetPointsAttr().Get() or [])
        if not deformable.IsValid() or not mesh or not points:
            raise ValueError(f"Attachment source is not a populated mesh: {deformable_path}")
        if not target.IsValid() or not UsdGeom.Xformable(target):
            raise ValueError(f"Attachment target is not xformable: {target_path}")
        runtime_geometry=(
            deformable_points_world is not None
            or target_to_world is not None
            or attachment_frame_to_world is not None
        )
        if runtime_geometry and (
            deformable_points_world is None
            or target_to_world is None
            or attachment_frame_to_world is None
        ):
            raise ValueError(
                "Runtime attachment selection requires current deformable "
                "world points, target-to-world, and attachment-frame-to-world"
            )
        if attachment_frame_path is None:
            frame=target
            while frame.IsValid() and not frame.HasAPI(UsdPhysics.RigidBodyAPI):
                frame=frame.GetParent()
            if not frame.IsValid():
                raise RuntimeError(
                    f"Attachment target has no rigid-body frame: {target_path}"
                )
            attachment_frame_path=str(frame.GetPath())
        attachment_frame=stage.GetPrimAtPath(attachment_frame_path)
        if not attachment_frame.IsValid():
            raise RuntimeError(
                f"Attachment frame is missing: {attachment_frame_path}"
            )
        mesh_to_world=UsdGeom.Xformable(deformable).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        if target_to_world is None:
            target_to_world=UsdGeom.Xformable(target).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            bounds=UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_,UsdGeom.Tokens.guide],
            ).ComputeWorldBound(target).ComputeAlignedRange()
        else:
            untransformed=UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_,UsdGeom.Tokens.guide],
            ).ComputeUntransformedBound(target)
            untransformed.Transform(target_to_world)
            bounds=untransformed.ComputeAlignedRange()
        if attachment_frame_to_world is None:
            attachment_frame_to_world=UsdGeom.Xformable(
                attachment_frame
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_to_attachment_frame=attachment_frame_to_world.GetInverse()
        minimum,maximum=bounds.GetMin(),bounds.GetMax()
        center=(minimum+maximum)*0.5
        if deformable_points_world is not None:
            if len(deformable_points_world)!=len(points):
                raise RuntimeError(
                    f"Runtime deformable topology changed for {deformable_path}: "
                    f"usd_vertices={len(points)}, simulation_vertices="
                    f"{len(deformable_points_world)}"
                )
            world_points=[
                Gf.Vec3d(float(point[0]),float(point[1]),float(point[2]))
                for point in deformable_points_world
            ]
        else:
            world_points=[
                mesh_to_world.Transform(Gf.Vec3d(point)) for point in points
            ]
        if maximum_vertices<4:
            raise ValueError("maximum_vertices must be at least four")
        excluded=set(excluded_vertex_indices or ())
        ranked=[]
        for index,world in enumerate(world_points):
            delta=world-center
            overlaps=all(
                minimum[axis]-0.0025<=world[axis]<=maximum[axis]+0.0025
                for axis in range(3)
            )
            ranked.append((float(Gf.Dot(delta,delta)),index,world,overlaps))
        ranked.sort(key=lambda item:item[0])
        selected=[
            item for item in ranked if item[3] and item[1] not in excluded
        ][:maximum_vertices]
        if len(selected)<4:
            nearest_center_distance_m=(
                math.sqrt(ranked[0][0]) if ranked else math.inf
            )
            source_min=tuple(
                min(point[axis] for point in world_points) for axis in range(3)
            )
            source_max=tuple(
                max(point[axis] for point in world_points) for axis in range(3)
            )
            nearest_world=(
                tuple(ranked[0][2]) if ranked else None
            )
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_path}, "
                f"target={target_path}, overlapping={len(selected)}, "
                f"required=4, overlap_margin_m=0.0025, "
                f"target_bounds_world=({tuple(minimum)}, {tuple(maximum)}), "
                f"source_bounds_world=({source_min}, {source_max}), "
                f"nearest_vertex_world={nearest_world}, "
                f"nearest_vertex_to_target_center_m={nearest_center_distance_m}"
            )
        attachment=stage.DefinePrim(
            attachment_path,"OmniPhysicsVtxXformAttachment"
        )
        attachment.CreateRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(deformable_path)]
        )
        attachment.CreateRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(attachment_frame_path)]
        )
        attachment.CreateAttribute(
            "omniphysics:vtxIndicesSrc0",Sdf.ValueTypeNames.IntArray
        ).Set(Vt.IntArray([item[1] for item in selected]))
        attachment.CreateAttribute(
            "omniphysics:localPositionsSrc1",Sdf.ValueTypeNames.Point3fArray
        ).Set(Vt.Vec3fArray([
            Gf.Vec3f(world_to_attachment_frame.Transform(item[2]))
            for item in selected
        ]))
        if selected_vertex_indices_out is not None:
            selected_vertex_indices_out.extend(item[1] for item in selected)
        attachment.CreateAttribute(
            "omniphysics:attachmentEnabled",Sdf.ValueTypeNames.Bool
        ).Set(True)
        if (
            not attachment.IsValid()
            or attachment.GetTypeName()!="OmniPhysicsVtxXformAttachment"
            or not attachment.GetRelationship("omniphysics:src0").GetTargets()
            or not attachment.GetRelationship("omniphysics:src1").GetTargets()
        ):
            raise RuntimeError(f"Could not author attachment {attachment_path}")
        return "OmniPhysicsVtxXformAttachment"

    import omni.kit.commands
    def execute_and_verify(command: str,**kwargs) -> str:
        omni.kit.commands.execute(command,**kwargs)
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
        if stage.GetPrimAtPath(attachment_path).IsValid():stage.RemovePrim(attachment_path)
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

def remove_prims(paths: Iterable[str], *, stage=None):
    stage=_current_stage(stage)
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():stage.RemovePrim(path)


def anchor_vessel_distal_ends(root_path: str,*,stage=None) -> list[str]:
    """Attach both cooked vessel halves to explicit kinematic fixtures."""
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    stage=_current_stage(stage);root_path=root_path.rstrip("/")
    attachments_root=f"{root_path}/RuntimeFixtureAttachments"
    frames_root=f"{root_path}/RuntimeFixtureFrames"
    stage.DefinePrim(attachments_root,"Scope")
    stage.DefinePrim(frames_root,"Scope");created=[]
    try:
        for label,vessel,target in (
            ("left","LeftVesselWall","LeftFixtureAnchor"),
            ("right","RightVesselWall","RightFixtureAnchor"),
        ):
            target_path=f"{root_path}/{target}"
            target_prim=stage.GetPrimAtPath(target_path)
            if not target_prim.IsValid():
                raise ValueError(f"Vessel fixture anchor is missing: {target_path}")
            frame_path=f"{frames_root}/{label}"
            frame=UsdGeom.Xform.Define(stage,frame_path)
            root_to_world=UsdGeom.Xformable(
                stage.GetPrimAtPath(root_path)
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            target_to_world=UsdGeom.Xformable(
                target_prim
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            frame_to_world=Gf.Matrix4d(1.0)
            frame_to_world.SetRotate(target_to_world.ExtractRotationMatrix())
            frame_to_world.SetTranslateOnly(target_to_world.ExtractTranslation())
            frame.MakeMatrixXform().Set(
                frame_to_world*root_to_world.GetInverse()
            )
            rigid=UsdPhysics.RigidBodyAPI.Apply(frame.GetPrim())
            rigid.CreateRigidBodyEnabledAttr(True)
            rigid.CreateKinematicEnabledAttr(True)
            attachment_path=f"{attachments_root}/{label}"
            create_deformable_attachment(
                f"{root_path}/{vessel}",
                target_path,
                attachment_path,
                stage=stage,
                attachment_frame_path=frame_path,
            )
            created.append(attachment_path)
    except Exception:
        remove_prims(created,stage=stage);raise
    return created


@dataclass
class DualZoneCompressionController:
    tool_root: str
    vessel_root: str
    minimum_total_force_n: float=8.0
    target_total_force_n: float=18.0
    soft_force_limit_n: float=32.0
    hard_release_limit_n: float=45.0
    _attachment_paths: list[str]=field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _engaged: bool=field(default=False,init=False,repr=False)
    @property
    def attachment_prim_ids(self):return tuple(self._attachment_paths)
    @property
    def engaged(self):return self._engaged
    def __post_init__(self):
        values=(
            self.minimum_total_force_n,
            self.target_total_force_n,
            self.soft_force_limit_n,
            self.hard_release_limit_n,
        )
        for value,label in zip(
            values,
            (
                "minimum_total_force_n",
                "target_total_force_n",
                "soft_force_limit_n",
                "hard_release_limit_n",
            ),
            strict=True,
        ):
            _finite_nonnegative(value,label)
        if tuple(values)!=tuple(sorted(values)):
            raise ValueError(
                "compression force thresholds must be ordered minimum, "
                "target, soft limit, hard release"
            )
    def engage(self,*,stage=None,runtime_geometry=None):
        stage=_current_stage(stage)
        parent=f"{self.vessel_root}/RuntimeJawCompressionAttachments"
        stage.DefinePrim(parent,"Scope");created=[]
        try:
            for side,vessel,contact in (
                ("left_upper","LeftVesselWall","UpperJaw/Collisions/LeftSealContact"),
                ("left_lower","LeftVesselWall","LowerJaw/Collisions/LeftSealContact"),
                ("right_upper","RightVesselWall","UpperJaw/Collisions/RightSealContact"),
                ("right_lower","RightVesselWall","LowerJaw/Collisions/RightSealContact"),
            ):
                attachment=f"{parent}/{side}"
                geometry=(
                    {} if runtime_geometry is None
                    else runtime_geometry.get(side,{})
                )
                create_deformable_attachment(
                    f"{self.vessel_root}/{vessel}",
                    f"{self.tool_root}/Links/{contact}",
                    attachment,stage=stage,**geometry,
                )
                created.append(attachment)
        except Exception:
            remove_prims(created,stage=stage);raise
        self._attachment_paths=created;self._engaged=True
        return list(created)
    def release(self,*,stage=None):
        stage=_current_stage(stage)
        remove_prims(self._attachment_paths,stage=stage)
        if any(
            stage.GetPrimAtPath(path).IsValid()
            for path in self._attachment_paths
        ):
            raise RuntimeError(
                "jaw-compression attachment release did not commit to stage"
            )
        self._engaged=False
    def update_from_scene(
        self,
        evidence: SealDivideSceneEvidence,
        *,
        stage=None,
    ):
        if not isinstance(evidence,SealDivideSceneEvidence):
            raise TypeError(
                "compression control requires SealDivideSceneEvidence"
            )
        registered_attachment_ids=tuple(sorted(
            (
                *evidence.sources.left.upper_compression_attachment_prim_ids,
                *evidence.sources.left.lower_compression_attachment_prim_ids,
                *evidence.sources.right.upper_compression_attachment_prim_ids,
                *evidence.sources.right.lower_compression_attachment_prim_ids,
            )
        ))
        if (
            self.engaged
            and tuple(sorted(self._attachment_paths))
            != registered_attachment_ids
        ):
            raise ValueError(
                "compression controller topology does not match the "
                "registered evidence sources"
            )
        upper=(
            evidence.left.upper_contact.normal_force_n
            + evidence.right.upper_contact.normal_force_n
        )
        lower=(
            evidence.left.lower_contact.normal_force_n
            + evidence.right.lower_contact.normal_force_n
        )
        left=evidence.left.compression_force_n
        right=evidence.right.compression_force_n
        total=left+right
        if total>self.hard_release_limit_n and self.engaged:self.release(stage=stage)
        return {
            "mode":(
                "hard_release" if total>self.hard_release_limit_n
                else "soft_limit" if total>self.soft_force_limit_n
                else "insufficient" if total<self.minimum_total_force_n
                else "controlled"
            ),
            "upper_force_n":upper,
            "lower_force_n":lower,
            "left_effective_force_n":left,
            "right_effective_force_n":right,
            "total_force_n":total,
            "target_error_n":total-self.target_total_force_n,
            "engaged":self.engaged,
            "evidence_digest_sha256":evidence.evidence_digest_sha256,
            "attachment_prim_ids":tuple(sorted({
                *evidence.left.upper_contact.attachment_prim_ids,
                *evidence.left.lower_contact.attachment_prim_ids,
                *evidence.right.upper_contact.attachment_prim_ids,
                *evidence.right.lower_contact.attachment_prim_ids,
            })),
        }

@dataclass(frozen=True)
class BridgeCell:
    index: int
    pin_path: str
    attachment_paths: tuple[str,...]

@dataclass
class BridgeAttachmentController:
    vessel_root: str
    release_order: tuple[int,...]=(2,1,3,0,4,7,5,6)
    _cells: list[BridgeCell]=field(default_factory=list,init=False,repr=False)
    _released_indices: set[int]=field(
        default_factory=set,
        init=False,
        repr=False,
    )
    @property
    def cells(self):return tuple(self._cells)
    @property
    def attachment_prim_ids(self):
        return tuple(sorted(
            path
            for cell in self._cells
            for path in cell.attachment_paths
        ))
    def engage(self, *, stage=None):
        if tuple(sorted(self.release_order))!=tuple(range(BRIDGE_PIN_COUNT)):
            raise ValueError("release_order must contain each bridge index exactly once")
        stage=_current_stage(stage);stage.DefinePrim(f"{self.vessel_root}/RuntimeBridgeAttachments","Scope");created=[];cells=[]
        left=f"{self.vessel_root}/LeftVesselWall";right=f"{self.vessel_root}/RightVesselWall"
        used_vertices={"left":set(),"right":set()}
        try:
            for i in range(BRIDGE_PIN_COUNT):
                pin=f"{self.vessel_root}/BridgePins/BridgePin_{i:02d}/Capture";paths=[]
                for side,actor in (("left",left),("right",right)):
                    ap=f"{self.vessel_root}/RuntimeBridgeAttachments/pin_{i:02d}_{side}"
                    selected=[]
                    create_deformable_attachment(
                        actor,pin,ap,stage=stage,
                        excluded_vertex_indices=used_vertices[side],
                        maximum_vertices=4,
                        selected_vertex_indices_out=selected,
                    )
                    used_vertices[side].update(selected)
                    paths.append(ap);created.append(ap)
                cells.append(BridgeCell(i,pin,tuple(paths)))
        except Exception:
            remove_prims(created,stage=stage);raise
        self._cells=cells
        self._released_indices.clear()
        return self.cells
    @property
    def _released_fraction(self):
        return (
            0.0
            if not self._cells
            else len(self._released_indices)/len(self._cells)
        )
    def _reconcile_scene_attachments(
        self,
        active_attachment_prim_ids: tuple[str,...],
    ):
        candidate_released_indices=(
            self._candidate_released_indices_from_scene(
                active_attachment_prim_ids
            )
        )
        self._released_indices=candidate_released_indices
        return self._released_fraction
    def _validate_candidate_commit(
        self,
        candidate: "BridgeAttachmentController",
    ):
        if not isinstance(candidate,BridgeAttachmentController):
            raise TypeError(
                "bridge candidate must be BridgeAttachmentController"
            )
        if (
            candidate.vessel_root!=self.vessel_root
            or candidate.release_order!=self.release_order
            or candidate.cells!=self.cells
        ):
            raise RuntimeError(
                "bridge candidate topology or release policy changed"
            )
        registered_indices={cell.index for cell in self._cells}
        if not candidate._released_indices.issubset(registered_indices):
            raise RuntimeError(
                "bridge candidate contains an unregistered released cell"
            )
    def _commit_candidate(
        self,
        candidate: "BridgeAttachmentController",
    ):
        self._validate_candidate_commit(candidate)
        self._released_indices=set(candidate._released_indices)
    def _candidate_released_indices_from_scene(
        self,
        active_attachment_prim_ids: tuple[str,...],
    ):
        if not self._cells:
            raise RuntimeError(
                "bridge cells must be engaged before scene reconciliation"
            )
        active=set(active_attachment_prim_ids)
        registered={
            path for cell in self._cells for path in cell.attachment_paths
        }
        if not active.issubset(registered):
            raise ValueError(
                "scene evidence contains an unregistered bridge attachment"
            )
        candidate_released_indices=set(self._released_indices)
        for cell in self._cells:
            live=[path in active for path in cell.attachment_paths]
            if any(live) and not all(live):
                raise RuntimeError(
                    f"bridge cell {cell.index} has a partial attachment pair"
                )
            if cell.index in candidate_released_indices and all(live):
                raise RuntimeError(
                    f"released bridge cell {cell.index} reappeared in scene"
                )
            if not any(live):
                candidate_released_indices.add(cell.index)
        return candidate_released_indices
    def _release_for_blade_progress(self,progress: float,*,stage=None):
        if tuple(sorted(self.release_order))!=tuple(range(BRIDGE_PIN_COUNT)):
            raise ValueError(
                "release_order must contain each bridge index exactly once"
            )
        progress=max(0.0,min(1.0,_finite(progress,"progress")));target=int(math.floor(progress*len(self.release_order)+1e-9));stage=_current_stage(stage)
        by_index={c.index:c for c in self._cells}
        for idx in self.release_order[:target]:
            cell=by_index.get(idx)
            if cell is not None and idx not in self._released_indices:
                remove_prims(cell.attachment_paths,stage=stage)
                if any(
                    stage.GetPrimAtPath(path).IsValid()
                    for path in cell.attachment_paths
                ):
                    raise RuntimeError(
                        f"bridge cell {idx} attachment removal was not "
                        "committed to the stage"
                    )
                self._released_indices.add(idx)
        if progress>=1.0:
            bridge=stage.GetPrimAtPath(f"{self.vessel_root}/BridgeVisual")
            if bridge and bridge.IsValid():
                from pxr import UsdGeom
                UsdGeom.Imageable(bridge).MakeInvisible()
        return self._released_fraction

@dataclass
class _SealZoneState:
    name: str
    temperature_c: float=37.0
    impedance_ohm: float=120.0
    measured_energy_j: float=0.0
    thermal_dose: float=0.0
    conditioning_fraction: float=0.0
    compression_force_n: float=0.0
    maximum_contact_force_n: float=0.0
    compression_force_imbalance_fraction: float=0.0
    compression_pressure_pa: float=0.0
    maximum_slip_speed_m_s: float=0.0
    compression_attached: bool=False
    cohesive_integrity_fraction: float=0.0
    cohesive_damage_fraction: float=0.0
    cohesive_failed: bool=False
    observed_leak_ml_min: float=math.inf
    upstream_pressure_pa: float=0.0
    wall_damage_fraction: float=0.0
    attachment_prim_ids: tuple[str,...]=()
    baseline_impedance_ohm: float|None=None
    overtemperature: bool=False
    impedance_fault: bool=False
    last_scene_step: int=-1
    last_evidence_digest_sha256: str|None=None

@dataclass(frozen=True)
class SealZoneSnapshot:
    name: str
    temperature_c: float
    impedance_ohm: float
    measured_energy_j: float
    thermal_dose: float
    conditioning_fraction: float
    compression_force_n: float
    maximum_contact_force_n: float
    compression_force_imbalance_fraction: float
    compression_pressure_pa: float
    maximum_slip_speed_m_s: float
    compression_attached: bool
    cohesive_integrity_fraction: float
    cohesive_damage_fraction: float
    cohesive_failed: bool
    observed_leak_ml_min: float
    upstream_pressure_pa: float
    wall_damage_fraction: float
    attachment_prim_ids: tuple[str,...]
    overtemperature: bool
    impedance_fault: bool
    last_scene_step: int
    last_evidence_digest_sha256: str|None

@dataclass
class AdaptiveSealEnergyController:
    target_temperature_c: float=78.0
    maximum_temperature_c: float=105.0
    maximum_power_w: float=45.0
    minimum_compression_force_n: float=8.0
    maximum_compression_force_n: float=32.0
    maximum_compression_force_imbalance_fraction: float=0.35
    maximum_slip_speed_m_s: float=0.002
    minimum_impedance_rise_fraction: float=0.20
    maximum_cohesive_damage_fraction: float=0.35
    minimum_cohesive_integrity_fraction: float=0.65
    maximum_observed_leak_ml_min: float=0.1
    maximum_wall_damage_fraction: float=0.35
    minimum_verification_pressure_pa: float=12000.0
    _left: _SealZoneState=field(
        default_factory=lambda:_SealZoneState("left"),
        init=False,
        repr=False,
    )
    _right: _SealZoneState=field(
        default_factory=lambda:_SealZoneState("right"),
        init=False,
        repr=False,
    )
    _locked_policy_signature: tuple[float,...]|None=field(
        default=None,
        init=False,
        repr=False,
    )
    def __post_init__(self):
        target=_finite(self.target_temperature_c,"target_temperature_c")
        maximum=_finite(
            self.maximum_temperature_c,
            "maximum_temperature_c",
        )
        if target<=0.0 or maximum<target:
            raise ValueError(
                "temperature policy must be positive and maximum must not "
                "be below target"
            )
        for name in (
            "maximum_power_w",
            "minimum_compression_force_n",
            "maximum_compression_force_n",
            "maximum_slip_speed_m_s",
            "maximum_observed_leak_ml_min",
            "minimum_verification_pressure_pa",
        ):
            _finite_nonnegative(getattr(self,name),name)
        if (
            self.maximum_compression_force_n
            < self.minimum_compression_force_n
        ):
            raise ValueError(
                "maximum compression force must not be below minimum"
            )
        for name in (
            "maximum_compression_force_imbalance_fraction",
            "minimum_impedance_rise_fraction",
            "maximum_cohesive_damage_fraction",
            "minimum_cohesive_integrity_fraction",
            "maximum_wall_damage_fraction",
        ):
            _unit_fraction(getattr(self,name),name)
    def _snapshot(self,zone: _SealZoneState):
        return SealZoneSnapshot(
            name=zone.name,
            temperature_c=zone.temperature_c,
            impedance_ohm=zone.impedance_ohm,
            measured_energy_j=zone.measured_energy_j,
            thermal_dose=zone.thermal_dose,
            conditioning_fraction=zone.conditioning_fraction,
            compression_force_n=zone.compression_force_n,
            maximum_contact_force_n=zone.maximum_contact_force_n,
            compression_force_imbalance_fraction=(
                zone.compression_force_imbalance_fraction
            ),
            compression_pressure_pa=zone.compression_pressure_pa,
            maximum_slip_speed_m_s=zone.maximum_slip_speed_m_s,
            compression_attached=zone.compression_attached,
            cohesive_integrity_fraction=(
                zone.cohesive_integrity_fraction
            ),
            cohesive_damage_fraction=zone.cohesive_damage_fraction,
            cohesive_failed=zone.cohesive_failed,
            observed_leak_ml_min=zone.observed_leak_ml_min,
            upstream_pressure_pa=zone.upstream_pressure_pa,
            wall_damage_fraction=zone.wall_damage_fraction,
            attachment_prim_ids=zone.attachment_prim_ids,
            overtemperature=zone.overtemperature,
            impedance_fault=zone.impedance_fault,
            last_scene_step=zone.last_scene_step,
            last_evidence_digest_sha256=(
                zone.last_evidence_digest_sha256
            ),
        )
    @property
    def left(self):return self._snapshot(self._left)
    @property
    def right(self):return self._snapshot(self._right)
    def _energy_delivery_safe(self,zone: _SealZoneState):
        return (
            zone.last_scene_step>=0
            and not zone.overtemperature
            and not zone.impedance_fault
            and zone.compression_attached
            and bool(zone.attachment_prim_ids)
            and self.minimum_compression_force_n
            <=zone.compression_force_n
            and zone.maximum_contact_force_n
            <=self.maximum_compression_force_n
            and zone.compression_force_imbalance_fraction
            <=self.maximum_compression_force_imbalance_fraction
            and zone.maximum_slip_speed_m_s
            <=self.maximum_slip_speed_m_s
            and not zone.cohesive_failed
            and zone.cohesive_damage_fraction
            <=self.maximum_cohesive_damage_fraction
            and zone.cohesive_integrity_fraction
            >=self.minimum_cohesive_integrity_fraction
            and zone.wall_damage_fraction
            <=self.maximum_wall_damage_fraction
        )
    def _recommended_power_w(self,zone: _SealZoneState):
        if not self._energy_delivery_safe(zone):return 0.0
        compression_scale=max(0.0,min(1.0,(zone.compression_force_n-self.minimum_compression_force_n)/max(self.maximum_compression_force_n-self.minimum_compression_force_n,1e-9)))
        return max(0.0,min(self.maximum_power_w,(self.target_temperature_c-zone.temperature_c)*1.25*(0.55+0.45*compression_scale)))
    def recommended_power_w(self,side: str):
        self._assert_policy_unchanged()
        if side=="left":return self._recommended_power_w(self._left)
        if side=="right":return self._recommended_power_w(self._right)
        raise ValueError("side must be 'left' or 'right'")
    def readiness_policy(self):
        return {
            "target_temperature_c":self.target_temperature_c,
            "maximum_temperature_c":self.maximum_temperature_c,
            "maximum_power_w":self.maximum_power_w,
            "minimum_compression_force_n":(
                self.minimum_compression_force_n
            ),
            "maximum_compression_force_n":(
                self.maximum_compression_force_n
            ),
            "maximum_compression_force_imbalance_fraction":(
                self.maximum_compression_force_imbalance_fraction
            ),
            "maximum_slip_speed_m_s":self.maximum_slip_speed_m_s,
            "minimum_impedance_rise_fraction":(
                self.minimum_impedance_rise_fraction
            ),
            "maximum_cohesive_damage_fraction":(
                self.maximum_cohesive_damage_fraction
            ),
            "minimum_cohesive_integrity_fraction":(
                self.minimum_cohesive_integrity_fraction
            ),
            "maximum_observed_leak_ml_min":(
                self.maximum_observed_leak_ml_min
            ),
            "maximum_wall_damage_fraction":(
                self.maximum_wall_damage_fraction
            ),
            "minimum_verification_pressure_pa":(
                self.minimum_verification_pressure_pa
            ),
            "parameter_status":"provisional_engineering_parameters",
        }
    def _policy_signature(self):
        policy=self.readiness_policy()
        return tuple(
            float(value)
            for key,value in policy.items()
            if key!="parameter_status"
        )
    def _assert_policy_unchanged(self):
        if (
            self._locked_policy_signature is not None
            and self._policy_signature()!=self._locked_policy_signature
        ):
            raise RuntimeError(
                "seal-readiness policy changed after evidence consumption"
            )
    def _update_zone_from_scene(
        self,
        zone: _SealZoneState,
        mechanics: SealDivideZoneMechanicsSample,
    ):
        if mechanics.side!=zone.name:
            raise ValueError(
                f"zone mismatch: state={zone.name!r}, evidence={mechanics.side!r}"
            )
        if mechanics.physics_step<=zone.last_scene_step:
            raise ValueError(
                "seal-zone evidence must advance monotonically"
            )
        zone.temperature_c=mechanics.interface_temperature_c
        zone.impedance_ohm=mechanics.interface_impedance_ohm
        zone.compression_force_n=mechanics.compression_force_n
        zone.maximum_contact_force_n=mechanics.maximum_contact_force_n
        zone.compression_force_imbalance_fraction=(
            mechanics.compression_force_imbalance_fraction
        )
        zone.compression_pressure_pa=mechanics.compression_pressure_pa
        zone.maximum_slip_speed_m_s=mechanics.maximum_slip_speed_m_s
        zone.compression_attached=mechanics.compression_attached
        zone.measured_energy_j += mechanics.measured_power_w*mechanics.dt_s
        dose_rate=(
            0.0
            if zone.temperature_c<45.0
            else math.exp(
                min(50.0,(zone.temperature_c-62.0)/8.0)
            )
        )
        zone.thermal_dose += dose_rate*mechanics.dt_s
        if (
            zone.baseline_impedance_ohm is None
            and zone.impedance_ohm>0.0
        ):
            zone.baseline_impedance_ohm=zone.impedance_ohm
        impedance_rise=0.0
        if zone.baseline_impedance_ohm:
            impedance_rise=max(
                0.0,
                zone.impedance_ohm/zone.baseline_impedance_ohm-1.0,
            )
        thermal_fraction=1.0-math.exp(-zone.thermal_dose/8.0)
        impedance_fraction=max(
            0.0,
            min(
                1.0,
                impedance_rise
                / max(self.minimum_impedance_rise_fraction,1e-9),
            ),
        )
        zone.conditioning_fraction=max(
            zone.conditioning_fraction,
            min(
                thermal_fraction,
                impedance_fraction,
            ),
        )
        zone.cohesive_damage_fraction=(
            mechanics.cohesive_damage_fraction
        )
        zone.cohesive_failed=mechanics.cohesive_failed
        zone.cohesive_integrity_fraction=mechanics.seal_integrity_fraction
        zone.observed_leak_ml_min=mechanics.observed_leak_ml_min
        zone.upstream_pressure_pa=mechanics.upstream_pressure_pa
        zone.wall_damage_fraction=mechanics.wall_damage_fraction
        zone.attachment_prim_ids=mechanics.seal_attachment_prim_ids
        zone.overtemperature=(
            zone.overtemperature
            or zone.temperature_c>self.maximum_temperature_c
        )
        zone.impedance_fault=(
            zone.impedance_fault
            or not math.isfinite(zone.impedance_ohm)
            or zone.impedance_ohm<10.0
            or zone.impedance_ohm>500.0
        )
        zone.last_scene_step=mechanics.physics_step
        zone.last_evidence_digest_sha256=(
            mechanics.evidence_digest_sha256
        )
        return zone
    def update_from_scene(self,evidence: SealDivideSceneEvidence):
        if not isinstance(evidence,SealDivideSceneEvidence):
            raise TypeError(
                "seal energy control requires SealDivideSceneEvidence"
            )
        signature=self._policy_signature()
        if (
            self._locked_policy_signature is not None
            and signature!=self._locked_policy_signature
        ):
            raise RuntimeError(
                "seal-readiness policy changed after evidence consumption"
            )
        candidate_left=replace(self._left)
        candidate_right=replace(self._right)
        self._update_zone_from_scene(candidate_left,evidence.left)
        self._update_zone_from_scene(candidate_right,evidence.right)
        left_snapshot=self._snapshot(candidate_left)
        right_snapshot=self._snapshot(candidate_right)
        self._left=candidate_left
        self._right=candidate_right
        if self._locked_policy_signature is None:
            self._locked_policy_signature=signature
        return left_snapshot,right_snapshot
    def _commit_candidate(
        self,
        candidate: "AdaptiveSealEnergyController",
    ):
        if not isinstance(candidate,AdaptiveSealEnergyController):
            raise TypeError(
                "seal energy candidate must be AdaptiveSealEnergyController"
            )
        self._left=replace(candidate._left)
        self._right=replace(candidate._right)
        self._locked_policy_signature=(
            candidate._locked_policy_signature
        )
    def _zone_ready(self,zone: _SealZoneState):
        return (
            zone.conditioning_fraction>=0.90
            and not zone.overtemperature
            and not zone.impedance_fault
            and self.minimum_compression_force_n
            <=zone.compression_force_n
            and zone.maximum_contact_force_n
            <=self.maximum_compression_force_n
            and zone.compression_force_imbalance_fraction
            <=self.maximum_compression_force_imbalance_fraction
            and zone.maximum_slip_speed_m_s
            <=self.maximum_slip_speed_m_s
            and zone.compression_attached
            and not zone.cohesive_failed
            and zone.cohesive_damage_fraction
            <=self.maximum_cohesive_damage_fraction
            and zone.cohesive_integrity_fraction
            >=self.minimum_cohesive_integrity_fraction
            and bool(zone.attachment_prim_ids)
            and zone.observed_leak_ml_min
            <=self.maximum_observed_leak_ml_min
            and zone.upstream_pressure_pa
            >=self.minimum_verification_pressure_pa
            and zone.wall_damage_fraction
            <=self.maximum_wall_damage_fraction
        )
    def zone_ready(self,side: str):
        self._assert_policy_unchanged()
        if side=="left":return self._zone_ready(self._left)
        if side=="right":return self._zone_ready(self._right)
        raise ValueError("side must be 'left' or 'right'")
    @property
    def both_ready(self):
        self._assert_policy_unchanged()
        return self._zone_ready(self._left) and self._zone_ready(self._right)

def _spawn_reference_at_transform(stage,prim_path: str,usd_path: Path,world_transform: Any,variants: dict[str,str]|None=None):
    from pxr import Gf,UsdGeom
    prim=stage.DefinePrim(prim_path,"Xform");prim.GetReferences().AddReference(str(usd_path))
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(world_transform))
    if variants:
        for name,value in variants.items():prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(value)
    return prim

@dataclass
class SealBandBond:
    band_path: str
    vessel_path: str
    _attachment_paths: list[str]=field(default_factory=list,init=False,repr=False)
    _failed: bool=field(default=False,init=False,repr=False)
    _last_scene_step: int=field(default=-1,init=False,repr=False)
    _last_evidence_digest_sha256: str|None=field(
        default=None,
        init=False,
        repr=False,
    )
    @property
    def attachment_prim_ids(self):return tuple(self._attachment_paths)
    @property
    def failed(self):return self._failed
    @property
    def last_scene_step(self):return self._last_scene_step
    @property
    def last_evidence_digest_sha256(self):
        return self._last_evidence_digest_sha256
    def _validate_candidate_commit(self,candidate: "SealBandBond"):
        if not isinstance(candidate,SealBandBond):
            raise TypeError("seal-band candidate must be SealBandBond")
        if (
            candidate.band_path!=self.band_path
            or candidate.vessel_path!=self.vessel_path
        ):
            raise RuntimeError(
                "seal-band candidate identity does not match the live bond"
            )
    def _commit_candidate(self,candidate: "SealBandBond"):
        self._validate_candidate_commit(candidate)
        self._attachment_paths=list(candidate._attachment_paths)
        self._failed=candidate._failed
        self._last_scene_step=candidate._last_scene_step
        self._last_evidence_digest_sha256=(
            candidate._last_evidence_digest_sha256
        )

@dataclass
class TissueSealBandController:
    _bonds: list[SealBandBond]=field(
        default_factory=list,
        init=False,
        repr=False,
    )
    @property
    def bonds(self):return tuple(self._bonds)
    def _validate_candidate_commit(
        self,
        candidate: "TissueSealBandController",
    ):
        if not isinstance(candidate,TissueSealBandController):
            raise TypeError(
                "seal-band candidate must be TissueSealBandController"
            )
        live_by_identity={
            (bond.band_path,bond.vessel_path):bond
            for bond in self._bonds
        }
        candidate_by_identity={
            (bond.band_path,bond.vessel_path):bond
            for bond in candidate._bonds
        }
        if (
            len(live_by_identity)!=len(self._bonds)
            or len(candidate_by_identity)!=len(candidate._bonds)
            or live_by_identity.keys()!=candidate_by_identity.keys()
        ):
            raise RuntimeError(
                "seal-band candidate topology does not match live bonds"
            )
        for identity,bond in live_by_identity.items():
            bond._validate_candidate_commit(
                candidate_by_identity[identity]
            )
        return candidate_by_identity
    def _commit_candidate(
        self,
        candidate: "TissueSealBandController",
    ):
        candidate_by_identity=self._validate_candidate_commit(candidate)
        for bond in self._bonds:
            bond._commit_candidate(
                candidate_by_identity[(bond.band_path,bond.vessel_path)]
            )
    def deploy(self,prim_path: str,world_transform: Any,vessel_path: str,*,stage=None):
        stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,SEAL_BAND_USD,world_transform,{"state":"fresh"});stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
        try:
            for name in ("UpperBondVolume","LowerBondVolume"):
                ap=f"{prim_path}/Attachments/{name}";create_deformable_attachment(vessel_path,f"{prim_path}/Collisions/{name}",ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created+[prim_path],stage=stage);raise
        bond=SealBandBond(prim_path,vessel_path)
        bond._attachment_paths.extend(created)
        self._bonds.append(bond);return bond
    def _validate_update_from_scene(
        self,
        bond: SealBandBond,
        mechanics: SealDivideZoneMechanicsSample,
        energy: AdaptiveSealEnergyController,
    ):
        zone_state=(
            energy.left if mechanics.side=="left" else energy.right
        )
        if (
            zone_state.last_scene_step!=mechanics.physics_step
            or zone_state.last_evidence_digest_sha256
            != mechanics.evidence_digest_sha256
        ):
            raise ValueError(
                "seal-band update requires energy state from the same "
                "evidence interval"
            )
        seal_ready=energy.zone_ready(mechanics.side)
        expected_zone=mechanics.sources.zone(mechanics.side)
        if expected_zone.seal_band_prim_path!=bond.band_path:
            raise ValueError(
                f"seal-band path mismatch: bond={bond.band_path}, "
                f"evidence={expected_zone.seal_band_prim_path}"
            )
        if expected_zone.vessel_prim_path!=bond.vessel_path:
            raise ValueError(
                f"seal-band vessel mismatch: bond={bond.vessel_path}, "
                f"evidence={expected_zone.vessel_prim_path}"
            )
        if mechanics.physics_step<=bond.last_scene_step:
            raise ValueError(
                "seal-band mechanics evidence must advance monotonically"
            )
        if (
            not bond.failed
            and tuple(sorted(bond._attachment_paths))
            != expected_zone.seal_attachment_prim_ids
        ):
            raise ValueError(
                "seal-band controller topology does not match the registered "
                "evidence sources"
            )
        live=set(mechanics.seal_attachment_prim_ids)
        registered=set(bond._attachment_paths)
        if not live.issubset(registered):
            raise ValueError(
                "seal-band evidence contains an unregistered attachment prim"
            )
        failed=(
            mechanics.cohesive_failed
            or mechanics.cohesive_damage_fraction>=1.0-1.0e-9
            or live!=registered
        )
        return zone_state,seal_ready,failed
    def update_from_scene(
        self,
        bond: SealBandBond,
        mechanics: SealDivideZoneMechanicsSample,
        energy: AdaptiveSealEnergyController,
        *,
        stage=None,
    ):
        zone_state,seal_ready,failed=self._validate_update_from_scene(
            bond,
            mechanics,
            energy,
        )
        next_failed=bond.failed or failed
        next_attachment_paths=(
            [] if failed else list(bond._attachment_paths)
        )
        stage=_current_stage(stage)
        if failed and bond._attachment_paths:
            remove_prims(bond._attachment_paths,stage=stage)
            if any(
                stage.GetPrimAtPath(path).IsValid()
                for path in bond._attachment_paths
            ):
                raise RuntimeError(
                    "failed seal-band attachments were not removed from "
                    "the stage"
                )
        prim=stage.GetPrimAtPath(bond.band_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                f"registered seal-band prim is missing: {bond.band_path}"
            )
        state=(
            "failed"
            if next_failed
            else "mature"
            if seal_ready
            else "fresh"
        )
        if not (
            prim.GetVariantSets()
            .GetVariantSet("state")
            .SetVariantSelection(state)
        ):
            raise RuntimeError(
                f"could not commit seal-band state {state!r} "
                f"for {bond.band_path}"
            )
        bond._attachment_paths=next_attachment_paths
        bond._failed=next_failed
        bond._last_scene_step=mechanics.physics_step
        bond._last_evidence_digest_sha256=(
            mechanics.evidence_digest_sha256
        )
        return {
            "failed":next_failed,
            "mechanically_qualified":seal_ready and not next_failed,
            "conditioning_fraction":zone_state.conditioning_fraction,
            "cohesive_integrity_fraction":(
                mechanics.seal_integrity_fraction
            ),
            "cohesive_damage_fraction":(
                mechanics.cohesive_damage_fraction
            ),
            "cohesive_resultant_force_n":(
                mechanics.cohesive_resultant_force_n
            ),
            "attachment_prim_ids":mechanics.seal_attachment_prim_ids,
            "scene_step":mechanics.physics_step,
            "evidence_digest_sha256":mechanics.evidence_digest_sha256,
        }


def make_adaptive_seal_divide_scene_sources(
    *,
    workcell_id: str,
    compression: DualZoneCompressionController,
    bridge: BridgeAttachmentController,
    left_bond: SealBandBond,
    right_bond: SealBandBond,
    left_cohesive_interface_id: str,
    right_cohesive_interface_id: str,
    left_vessel_segment_id: str,
    right_vessel_segment_id: str,
    calibration_profile_id: str=(
        "dranmar-seal-divide-engineering-v1"
    ),
) -> AdaptiveSealDivideSceneSources:
    """Bind evidence sources to topology created by the runtime controllers."""

    if not compression.engaged:
        raise RuntimeError(
            "jaw-compression attachments must be engaged before source "
            "registration"
        )
    if not bridge.cells:
        raise RuntimeError(
            "bridge attachments must be engaged before source registration"
        )
    if (
        left_bond.failed
        or right_bond.failed
        or not left_bond.attachment_prim_ids
        or not right_bond.attachment_prim_ids
    ):
        raise RuntimeError(
            "fresh left and right seal-band attachments are required for "
            "source registration"
        )
    if compression.vessel_root.rstrip("/") != bridge.vessel_root.rstrip("/"):
        raise ValueError(
            "compression and bridge controllers must share one vessel root"
        )
    tool_root=compression.tool_root.rstrip("/")
    vessel_root=compression.vessel_root.rstrip("/")
    expected_vessels={
        "left":f"{vessel_root}/LeftVesselWall",
        "right":f"{vessel_root}/RightVesselWall",
    }
    if (
        left_bond.vessel_path!=expected_vessels["left"]
        or right_bond.vessel_path!=expected_vessels["right"]
    ):
        raise ValueError(
            "seal bands must bind the canonical left and right vessel walls"
        )
    compression_by_name={
        path.rsplit("/",1)[-1]:path
        for path in compression.attachment_prim_ids
    }
    expected_compression_names={
        "left_upper",
        "left_lower",
        "right_upper",
        "right_lower",
    }
    if set(compression_by_name)!=expected_compression_names:
        raise ValueError(
            "compression topology does not contain the four canonical "
            "zone/jaw attachments"
        )
    thermal_path=frame_path(tool_root,"thermal_camera")
    impedance_path=frame_path(tool_root,"impedance_probe")
    electrical_path=f"{tool_root}/Links/Mount/Visuals/EnergyModule"
    zones={}
    for side,bond,cohesive_id,segment_id in (
        (
            "left",
            left_bond,
            left_cohesive_interface_id,
            left_vessel_segment_id,
        ),
        (
            "right",
            right_bond,
            right_cohesive_interface_id,
            right_vessel_segment_id,
        ),
    ):
        zones[side]=SealDivideZoneSources(
            side=side,
            vessel_prim_path=expected_vessels[side],
            upper_contact_prim_path=(
                f"{tool_root}/Links/UpperJaw/Collisions/"
                f"{side.title()}SealContact"
            ),
            lower_contact_prim_path=(
                f"{tool_root}/Links/LowerJaw/Collisions/"
                f"{side.title()}SealContact"
            ),
            temperature_sensor_prim_path=thermal_path,
            impedance_sensor_prim_path=impedance_path,
            electrical_sensor_prim_path=electrical_path,
            seal_band_prim_path=bond.band_path,
            cohesive_interface_id=cohesive_id,
            vessel_segment_id=segment_id,
            upper_compression_attachment_prim_ids=(
                compression_by_name[f"{side}_upper"],
            ),
            lower_compression_attachment_prim_ids=(
                compression_by_name[f"{side}_lower"],
            ),
            seal_attachment_prim_ids=bond.attachment_prim_ids,
        )
    return AdaptiveSealDivideSceneSources(
        workcell_id=workcell_id,
        tool_root_prim_path=tool_root,
        left=zones["left"],
        right=zones["right"],
        blade_prim_path=f"{tool_root}/Links/BladeCarriage",
        blade_joint_prim_path=f"{tool_root}/Joints/blade_joint",
        blade_tip_prim_path=frame_path(tool_root,"blade_tip"),
        blade_contact_prim_path=(
            f"{tool_root}/Links/BladeCarriage/Collisions/BladeCollider"
        ),
        blade_guard_prim_path=f"{tool_root}/Links/BladeGuard",
        blade_guard_joint_prim_path=(
            f"{tool_root}/Joints/blade_guard_joint"
        ),
        bridge_topology_prim_path=(
            f"{vessel_root}/RuntimeBridgeAttachments"
        ),
        cut_plane_prim_path=frame_path(tool_root,"cut_plane"),
        tissue_center_reference_prim_path=frame_path(
            tool_root,
            "tissue_center_reference",
        ),
        vessel_center_prim_path=f"{vessel_root}/Frames/cut_center",
        bridge_attachment_prim_ids=bridge.attachment_prim_ids,
        calibration_profile_id=calibration_profile_id,
    )


@dataclass
class DualStumpLeakObservationTracker:
    """Retain only vessel-mechanics observations from the latest scene step."""

    _left_ml_min: float=field(default=math.inf,init=False,repr=False)
    _right_ml_min: float=field(default=math.inf,init=False,repr=False)
    _left_upstream_pressure_pa: float=field(default=0.0,init=False,repr=False)
    _right_upstream_pressure_pa: float=field(default=0.0,init=False,repr=False)
    _left_wall_damage_fraction: float=field(default=0.0,init=False,repr=False)
    _right_wall_damage_fraction: float=field(default=0.0,init=False,repr=False)
    _last_scene_step: int=field(default=-1,init=False,repr=False)
    _last_evidence_digest_sha256: str|None=field(default=None,init=False,repr=False)
    def update_from_scene(self,evidence: SealDivideSceneEvidence):
        if not isinstance(evidence,SealDivideSceneEvidence):
            raise TypeError(
                "stump leak tracker requires SealDivideSceneEvidence"
            )
        if evidence.physics_step<=self._last_scene_step:
            raise ValueError(
                "stump-flow evidence must advance monotonically"
            )
        left_ml_min=evidence.left.observed_leak_ml_min
        right_ml_min=evidence.right.observed_leak_ml_min
        left_upstream_pressure_pa=evidence.left.upstream_pressure_pa
        right_upstream_pressure_pa=evidence.right.upstream_pressure_pa
        left_wall_damage_fraction=evidence.left.wall_damage_fraction
        right_wall_damage_fraction=evidence.right.wall_damage_fraction
        evidence_digest_sha256=evidence.evidence_digest_sha256
        self._left_ml_min=left_ml_min
        self._right_ml_min=right_ml_min
        self._left_upstream_pressure_pa=left_upstream_pressure_pa
        self._right_upstream_pressure_pa=right_upstream_pressure_pa
        self._left_wall_damage_fraction=left_wall_damage_fraction
        self._right_wall_damage_fraction=right_wall_damage_fraction
        self._last_scene_step=evidence.physics_step
        self._last_evidence_digest_sha256=evidence_digest_sha256
        return self.flows()
    def _commit_candidate(
        self,
        candidate: "DualStumpLeakObservationTracker",
    ):
        if not isinstance(candidate,DualStumpLeakObservationTracker):
            raise TypeError(
                "stump-flow candidate must be "
                "DualStumpLeakObservationTracker"
            )
        self._left_ml_min=candidate._left_ml_min
        self._right_ml_min=candidate._right_ml_min
        self._left_upstream_pressure_pa=(
            candidate._left_upstream_pressure_pa
        )
        self._right_upstream_pressure_pa=(
            candidate._right_upstream_pressure_pa
        )
        self._left_wall_damage_fraction=(
            candidate._left_wall_damage_fraction
        )
        self._right_wall_damage_fraction=(
            candidate._right_wall_damage_fraction
        )
        self._last_scene_step=candidate._last_scene_step
        self._last_evidence_digest_sha256=(
            candidate._last_evidence_digest_sha256
        )
    def flows(self):
        return {
            "left_ml_min":self._left_ml_min,
            "right_ml_min":self._right_ml_min,
        }
    @property
    def left_upstream_pressure_pa(self):
        return self._left_upstream_pressure_pa
    @property
    def right_upstream_pressure_pa(self):
        return self._right_upstream_pressure_pa
    @property
    def last_scene_step(self):
        return self._last_scene_step
    @property
    def last_evidence_digest_sha256(self):
        return self._last_evidence_digest_sha256

@dataclass
class BladeInterlockController:
    minimum_jaw_force_n: float=8.0
    maximum_jaw_force_n: float=32.0
    maximum_stump_flow_ml_min: float=0.1
    minimum_verification_pressure_pa: float=12000.0
    _locked_policy_signature: tuple[float,...]|None=field(
        default=None,
        init=False,
        repr=False,
    )
    def __post_init__(self):
        for name in (
            "minimum_jaw_force_n",
            "maximum_jaw_force_n",
            "maximum_stump_flow_ml_min",
            "minimum_verification_pressure_pa",
        ):
            _finite_nonnegative(getattr(self,name),name)
        if self.maximum_jaw_force_n<self.minimum_jaw_force_n:
            raise ValueError(
                "maximum jaw force must not be below minimum jaw force"
            )
    def policy_snapshot(self):
        return {
            "minimum_jaw_force_n":self.minimum_jaw_force_n,
            "maximum_jaw_force_n":self.maximum_jaw_force_n,
            "maximum_stump_flow_ml_min":(
                self.maximum_stump_flow_ml_min
            ),
            "minimum_verification_pressure_pa":(
                self.minimum_verification_pressure_pa
            ),
            "parameter_status":"provisional_engineering_parameters",
        }
    def _validate_policy_unchanged(self):
        policy=self.policy_snapshot()
        signature=tuple(
            float(value)
            for key,value in policy.items()
            if key!="parameter_status"
        )
        if (
            self._locked_policy_signature is not None
            and signature!=self._locked_policy_signature
        ):
            raise RuntimeError(
                "blade-interlock policy changed after evidence consumption"
            )
        return signature
    def _lock_policy(self):
        signature=self._validate_policy_unchanged()
        if self._locked_policy_signature is None:
            self._locked_policy_signature=signature
    def evaluate_from_scene(
        self,
        energy: AdaptiveSealEnergyController,
        leak: DualStumpLeakObservationTracker,
        evidence: SealDivideSceneEvidence,
    ):
        self._lock_policy()
        reasons=[]
        evidence_digest_sha256=evidence.evidence_digest_sha256
        energy_receipts=(
            ("left",energy.left),
            ("right",energy.right),
        )
        for side,receipt in energy_receipts:
            if (
                receipt.last_scene_step!=evidence.physics_step
                or receipt.last_evidence_digest_sha256
                !=evidence_digest_sha256
            ):
                reasons.append(f"{side}_energy_evidence_not_current")
        if (
            leak.last_scene_step!=evidence.physics_step
            or leak.last_evidence_digest_sha256
            !=evidence_digest_sha256
        ):
            reasons.append("stump_flow_evidence_not_current")
        if evidence.tissue_centered is not True:reasons.append("tissue_not_centered")
        for side,zone in (
            ("left",evidence.left),
            ("right",evidence.right),
        ):
            if zone.compression_force_n<self.minimum_jaw_force_n:
                reasons.append(f"{side}_insufficient_compression")
            if zone.maximum_contact_force_n>self.maximum_jaw_force_n:
                reasons.append(f"{side}_excess_compression")
        if not energy.zone_ready("left"):reasons.append("left_seal_not_ready")
        if not energy.zone_ready("right"):reasons.append("right_seal_not_ready")
        flows=leak.flows()
        if flows["left_ml_min"]>self.maximum_stump_flow_ml_min:reasons.append("left_observed_leak")
        if flows["right_ml_min"]>self.maximum_stump_flow_ml_min:reasons.append("right_observed_leak")
        if (
            leak.left_upstream_pressure_pa
            <self.minimum_verification_pressure_pa
        ):
            reasons.append("left_pressure_challenge_insufficient")
        if (
            leak.right_upstream_pressure_pa
            <self.minimum_verification_pressure_pa
        ):
            reasons.append("right_pressure_challenge_insufficient")
        if not evidence.kinematics.guard_retracted:reasons.append("blade_guard_not_retracted")
        return {
            "authorized":not reasons,
            "reasons":reasons,
            "observed_flows":flows,
            "tissue_centered":evidence.tissue_centered,
            "guard_retracted":evidence.kinematics.guard_retracted,
            "interlock_policy":self.policy_snapshot(),
            "scene_step":evidence.physics_step,
            "evidence_digest_sha256":evidence_digest_sha256,
        }

@dataclass
class TissueDivisionController:
    bridge: BridgeAttachmentController
    interlock: BladeInterlockController=field(default_factory=BladeInterlockController)
    _blade_progress: float=field(default=0.0,init=False,repr=False)
    _violations: int=field(default=0,init=False,repr=False)
    _last_scene_step: int=field(default=-1,init=False,repr=False)
    _last_evidence_digest_sha256: str|None=field(
        default=None,
        init=False,
        repr=False,
    )
    _initial_bridge_topology_confirmed: bool=field(
        default=False,
        init=False,
        repr=False,
    )
    _maximum_authorized_blade_progress: float=field(
        default=0.0,
        init=False,
        repr=False,
    )
    def _validate_candidate_commit(
        self,
        candidate: "TissueDivisionController",
    ):
        if not isinstance(candidate,TissueDivisionController):
            raise TypeError(
                "division candidate must be TissueDivisionController"
            )
        self.bridge._validate_candidate_commit(candidate.bridge)
        live_policy=(
            self.interlock.minimum_jaw_force_n,
            self.interlock.maximum_jaw_force_n,
            self.interlock.maximum_stump_flow_ml_min,
            self.interlock.minimum_verification_pressure_pa,
        )
        candidate_policy=(
            candidate.interlock.minimum_jaw_force_n,
            candidate.interlock.maximum_jaw_force_n,
            candidate.interlock.maximum_stump_flow_ml_min,
            candidate.interlock.minimum_verification_pressure_pa,
        )
        if candidate_policy!=live_policy:
            raise RuntimeError(
                "division candidate interlock policy changed"
            )
    def _commit_candidate(
        self,
        candidate: "TissueDivisionController",
    ):
        self._validate_candidate_commit(candidate)
        self.bridge._commit_candidate(candidate.bridge)
        self.interlock._locked_policy_signature=(
            candidate.interlock._locked_policy_signature
        )
        self._blade_progress=candidate._blade_progress
        self._violations=candidate._violations
        self._last_scene_step=candidate._last_scene_step
        self._last_evidence_digest_sha256=(
            candidate._last_evidence_digest_sha256
        )
        self._initial_bridge_topology_confirmed=(
            candidate._initial_bridge_topology_confirmed
        )
        self._maximum_authorized_blade_progress=(
            candidate._maximum_authorized_blade_progress
        )
    def _validate_advance_from_scene(
        self,
        evidence: SealDivideSceneEvidence,
        *,
        energy: AdaptiveSealEnergyController,
        leak: DualStumpLeakObservationTracker,
    ):
        if not isinstance(evidence,SealDivideSceneEvidence):
            raise TypeError(
                "division control requires SealDivideSceneEvidence"
            )
        if not isinstance(energy,AdaptiveSealEnergyController):
            raise TypeError(
                "division control requires AdaptiveSealEnergyController"
            )
        if not isinstance(leak,DualStumpLeakObservationTracker):
            raise TypeError(
                "division control requires DualStumpLeakObservationTracker"
            )
        if evidence.physics_step<=self._last_scene_step:
            raise ValueError(
                "division evidence must advance monotonically"
            )
        kinematics=evidence.kinematics
        registered_bridge_ids=self.bridge.attachment_prim_ids
        if (
            registered_bridge_ids
            != evidence.sources.bridge_attachment_prim_ids
        ):
            raise ValueError(
                "division controller bridge topology does not match the "
                "registered evidence sources"
            )
        initial_bridge_topology_confirmed=(
            self._initial_bridge_topology_confirmed
        )
        if self._last_scene_step<0:
            if (
                kinematics.active_bridge_attachment_prim_ids
                != registered_bridge_ids
            ):
                raise RuntimeError(
                    "first division evidence must confirm the complete "
                    "registered bridge topology"
                )
            if (
                abs(
                    kinematics.blade_joint_position_m
                    - evidence.sources.blade_start_position_m
                )
                > evidence.sources.blade_position_tolerance_m
            ):
                raise RuntimeError(
                    "first division evidence must confirm the blade at its "
                    "registered start position"
                )
            initial_bridge_topology_confirmed=True
        self.interlock._validate_policy_unchanged()
        energy._assert_policy_unchanged()
        self.bridge._candidate_released_indices_from_scene(
            kinematics.active_bridge_attachment_prim_ids
        )
        return initial_bridge_topology_confirmed
    def advance_from_scene(
        self,
        evidence: SealDivideSceneEvidence,
        *,
        energy: AdaptiveSealEnergyController,
        leak: DualStumpLeakObservationTracker,
        stage=None,
    ):
        initial_bridge_topology_confirmed=(
            self._validate_advance_from_scene(
                evidence,
                energy=energy,
                leak=leak,
            )
        )
        kinematics=evidence.kinematics
        result=self.interlock.evaluate_from_scene(
            energy,
            leak,
            evidence,
        )
        observed_release=self.bridge._reconcile_scene_attachments(
            kinematics.active_bridge_attachment_prim_ids
        )
        requested=kinematics.blade_progress
        moved_forward=requested>self._blade_progress+1.0e-9
        reasons=list(result["reasons"])
        if not kinematics.blade_position_within_limits:
            reasons.append("blade_position_out_of_range")
        if not kinematics.blade_tip_position_consistent:
            reasons.append("blade_joint_tip_position_mismatch")
        authorized_release_limit=(
            math.floor(
                self._maximum_authorized_blade_progress
                * BRIDGE_PIN_COUNT
                + 1.0e-9
            )
            / BRIDGE_PIN_COUNT
        )
        if observed_release>authorized_release_limit+1.0e-9:
            reasons.append("bridge_release_exceeds_authorized_blade_progress")
        if moved_forward and not kinematics.blade_in_contact:
            reasons.append("blade_not_in_contact")
        authorized=not reasons
        commanded_release=observed_release
        violations=self._violations
        maximum_authorized_blade_progress=(
            self._maximum_authorized_blade_progress
        )
        if moved_forward and not authorized:
            violations+=1
        elif moved_forward and authorized and kinematics.blade_in_contact:
            candidate_authorized_blade_progress=max(
                maximum_authorized_blade_progress,
                requested,
            )
            commanded_release=self.bridge._release_for_blade_progress(
                requested,
                stage=stage,
            )
            maximum_authorized_blade_progress=(
                candidate_authorized_blade_progress
            )
        maximum_observed_blade_progress=max(
            self._blade_progress,
            requested,
        )
        evidence_digest_sha256=evidence.evidence_digest_sha256
        self._initial_bridge_topology_confirmed=(
            initial_bridge_topology_confirmed
        )
        self._violations=violations
        self._maximum_authorized_blade_progress=(
            maximum_authorized_blade_progress
        )
        self._blade_progress=maximum_observed_blade_progress
        self._last_scene_step=evidence.physics_step
        self._last_evidence_digest_sha256=evidence_digest_sha256
        return {
            **result,
            "authorized":authorized,
            "reasons":reasons,
            "blade_progress":requested,
            "maximum_observed_blade_progress":(
                maximum_observed_blade_progress
            ),
            "maximum_authorized_blade_progress":(
                self._maximum_authorized_blade_progress
            ),
            "blade_tip_position_tool_m":(
                kinematics.blade_tip_position_tool_m
            ),
            "blade_tip_position_error_m":(
                kinematics.blade_tip_position_error_m
            ),
            "blade_position_within_limits":(
                kinematics.blade_position_within_limits
            ),
            "blade_in_contact":kinematics.blade_in_contact,
            "blade_contact_force_n":kinematics.blade_contact_force_n,
            "active_bridge_attachment_prim_ids":(
                kinematics.active_bridge_attachment_prim_ids
            ),
            "bridge_release_fraction":observed_release,
            "commanded_bridge_release_fraction":commanded_release,
            "observed_bridge_empty":kinematics.division_complete,
            "division_complete":(
                kinematics.division_complete
                and initial_bridge_topology_confirmed
                and maximum_authorized_blade_progress
                >=1.0-1.0e-9
            ),
            "violations":violations,
        }

_PHASE_TARGETS={
    "inspect":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "center":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.002,"irrigation_valve_joint":0.0},
    "compress":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    # The inward electrode faces meet at +/-10 mm jaw travel.  Holding that
    # geometry through seal and division avoids the former 3 mm per-jaw
    # crossover, which pulled captured tissue apart instead of compressing it.
    "seal":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.005,"irrigation_valve_joint":0.0},
    "verify_seal":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.003,"irrigation_valve_joint":0.0},
    "retract_guard":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":-0.011,"blade_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "divide":{"left_centering_joint":0.022,"right_centering_joint":-0.022,"upper_jaw_joint":0.010,"lower_jaw_joint":-0.010,"blade_guard_joint":-0.011,"blade_joint":0.041,"suction_valve_joint":0.006,"irrigation_valve_joint":0.0},
    "release":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.003,"irrigation_valve_joint":0.003},
    "verify_stumps":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.002,"irrigation_valve_joint":0.0},
    "complete":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "abort":{"left_centering_joint":0.0,"right_centering_joint":0.0,"upper_jaw_joint":0.0,"lower_jaw_joint":0.0,"blade_guard_joint":0.0,"blade_joint":0.0,"suction_valve_joint":0.008,"irrigation_valve_joint":0.004},
}

def _phase_targets_unchecked(phase: str):
    try:return dict(_PHASE_TARGETS[phase])
    except KeyError as exc:raise KeyError(f"Unknown seal/divide phase {phase!r}") from exc

def phase_targets(phase: str):
    if phase=="divide":
        raise RuntimeError(
            "divide targets require "
            "AdaptiveSealDivideSequenceController.set_command_phase()"
        )
    return _phase_targets_unchecked(phase)

@dataclass
class AdaptiveSealDivideSequenceController:
    energy: AdaptiveSealEnergyController=field(default_factory=AdaptiveSealEnergyController)
    leak: DualStumpLeakObservationTracker=field(
        default_factory=DualStumpLeakObservationTracker
    )
    seal_bands: TissueSealBandController=field(default_factory=TissueSealBandController)
    division: TissueDivisionController|None=None
    evidence_source: SealDivideSceneEvidenceSource|None=None
    _evidence_cursor: SealDivideEvidenceCursor=field(
        default_factory=SealDivideEvidenceCursor,
        init=False,
        repr=False,
    )
    _commanded_phase: str=field(
        default="inspect",
        init=False,
        repr=False,
    )
    _command_history: list[str]=field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _last_evidence: SealDivideSceneEvidence|None=field(
        default=None,
        init=False,
        repr=False,
    )
    _last_division_report: dict[str,Any]|None=field(
        default=None,
        init=False,
        repr=False,
    )
    @property
    def last_evidence(self):return self._last_evidence
    @property
    def commanded_phase(self):return self._commanded_phase
    @property
    def command_history(self):return tuple(self._command_history)
    def set_command_phase(self,phase: str):
        targets=_phase_targets_unchecked(phase)
        if phase=="divide":
            evidence=self._last_evidence
            receipt=self._last_division_report
            if self.division is None:
                raise RuntimeError(
                    "divide command requires a configured division controller"
                )
            if evidence is None or receipt is None:
                raise RuntimeError(
                    "divide command requires a current scene-evidence "
                    "authorization receipt"
                )
            if (
                receipt.get("authorized") is not True
                or receipt.get("scene_step")!=evidence.physics_step
                or receipt.get("evidence_digest_sha256")
                !=evidence.evidence_digest_sha256
            ):
                raise RuntimeError(
                    "divide command is not authorized by the exact latest "
                    "scene-evidence interval"
                )
        else:
            # A different issued target can invalidate the geometry and
            # compression state behind an earlier blade authorization.
            self._last_division_report=None
        self._commanded_phase=phase
        self._command_history.append(phase)
        return targets
    def _resolve_evidence(
        self,
        evidence: SealDivideSceneEvidence|None,
    ):
        if evidence is None:
            if self.evidence_source is None:
                raise RuntimeError(
                    "SealDivideSceneEvidence or an evidence_source is required"
                )
            evidence=self.evidence_source.sample_seal_divide_scene()
        if not isinstance(evidence,SealDivideSceneEvidence):
            raise TypeError(
                "seal/divide sequence requires SealDivideSceneEvidence"
            )
        return evidence
    def consume_scene_evidence(
        self,
        evidence: SealDivideSceneEvidence|None=None,
        *,
        stage=None,
    ):
        evidence=self._evidence_cursor.validate(
            self._resolve_evidence(evidence)
        )
        # Keep the interval-derived controller state provisional until every
        # downstream consumer has accepted this exact evidence envelope.  The
        # USD stage itself is an external mutation boundary and cannot be
        # rolled back here, but a failed band/division update must not advance
        # controller state, energy/leak receipts, or the evidence cursor.
        candidate_energy=copy.deepcopy(self.energy)
        candidate_leak=copy.deepcopy(self.leak)
        candidate_seal_bands=copy.deepcopy(self.seal_bands)
        candidate_division=copy.deepcopy(self.division)
        left_state,right_state=candidate_energy.update_from_scene(evidence)
        flows=candidate_leak.update_from_scene(evidence)
        expected_band_paths={
            evidence.sources.left.seal_band_prim_path,
            evidence.sources.right.seal_band_prim_path,
        }
        controlled_band_paths=[
            bond.band_path for bond in candidate_seal_bands.bonds
        ]
        if (
            len(controlled_band_paths)!=len(expected_band_paths)
            or set(controlled_band_paths)!=expected_band_paths
        ):
            raise ValueError(
                "seal-band controller must contain exactly the two bands "
                "registered by the scene evidence"
            )
        band_mechanics={}
        for bond in candidate_seal_bands.bonds:
            if bond.band_path==evidence.sources.left.seal_band_prim_path:
                mechanics=evidence.left
            else:
                mechanics=evidence.right
            candidate_seal_bands._validate_update_from_scene(
                bond,
                mechanics,
                candidate_energy,
            )
            band_mechanics[bond.band_path]=mechanics
        self.seal_bands._validate_candidate_commit(
            candidate_seal_bands
        )
        if (self.division is None)!=(candidate_division is None):
            raise RuntimeError(
                "division candidate does not match live controller topology"
            )
        if (
            self.division is not None
            and candidate_division is not None
        ):
            self.division._validate_candidate_commit(candidate_division)
            candidate_division._validate_advance_from_scene(
                evidence,
                energy=candidate_energy,
                leak=candidate_leak,
            )
        band_reports={}
        for bond in candidate_seal_bands.bonds:
            band_reports[bond.band_path]=(
                candidate_seal_bands.update_from_scene(
                    bond,
                    band_mechanics[bond.band_path],
                    candidate_energy,
                    stage=stage,
                )
            )
        division_report=None
        if candidate_division is not None:
            division_report=candidate_division.advance_from_scene(
                evidence,
                energy=candidate_energy,
                leak=candidate_leak,
                stage=stage,
            )
        self.energy._commit_candidate(candidate_energy)
        self.leak._commit_candidate(candidate_leak)
        self.seal_bands._commit_candidate(candidate_seal_bands)
        if (
            self.division is not None
            and candidate_division is not None
        ):
            self.division._commit_candidate(candidate_division)
        self._evidence_cursor.consume(evidence)
        self._last_evidence=evidence
        self._last_division_report=copy.deepcopy(division_report)
        return {
            "scene_step":evidence.physics_step,
            "scene_time_s":evidence.simulation_time_s,
            "scene_dt_s":evidence.dt_s,
            "commanded_phase":self.commanded_phase,
            "episode_id":evidence.episode_id,
            "environment_id":evidence.environment_id,
            "topology_revision":evidence.topology_revision,
            "evidence_source":evidence.source,
            "envelope_digest_sha256":(
                evidence.envelope_digest_sha256
            ),
            "registration_digest_sha256":(
                evidence.registration_digest_sha256
            ),
            "evidence_digest_sha256":(
                evidence.evidence_digest_sha256
            ),
            "parameter_status":"provisional_engineering_parameters",
            "validation_status":(
                "not_clinically_validated_or_approved_for_patient_care"
            ),
            "left":left_state,
            "right":right_state,
            "observed_flows":flows,
            "both_ready":self.energy.both_ready,
            "seal_readiness_policy":self.energy.readiness_policy(),
            "tissue_centered":evidence.tissue_centered,
            "band_reports":band_reports,
            "division":copy.deepcopy(division_report),
        }
