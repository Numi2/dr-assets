#!/usr/bin/env python3
"""Render the real Dr.Anmar GLB inspection payload as a GitHub showcase.

Run with Blender:

    blender --background --python tools/render_showcase.py

The script deliberately renders shipped geometry rather than concept art.
It uses a neutral clinical-studio light rig, restrained physically based
materials, and per-asset camera angles selected to reveal the procedural work
surface without tinting the assets with brand colors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"


@dataclass(frozen=True)
class ShowcaseRender:
    source: str
    output: str
    azimuth_deg: float
    elevation_deg: float
    width: int = 1400
    height: int = 900
    margin: float = 1.32


RENDERS = (
    ShowcaseRender(
        "data/Environments/SurgicalAutonomy/AutonomousRescueOR/glb/"
        "dranmar_autonomous_rescue_or_procedure.glb",
        "hero-autonomous-rescue-or.png",
        34.0,
        27.0,
        2400,
        1000,
        1.10,
    ),
    ShowcaseRender(
        "data/Environments/SurgicalAutonomy/AutonomousRescueOR/glb/"
        "dranmar_deformable_rescue_suite.glb",
        "contact-driven-rescue-suite.png",
        35.0,
        24.0,
    ),
    ShowcaseRender(
        "data/Props/Patients/DynamicAbdominalPatient/glb/"
        "dranmar_patient_open_abdomen.glb",
        "dynamic-abdominal-patient.png",
        35.0,
        54.0,
        margin=1.18,
    ),
    ShowcaseRender(
        "data/Props/SurgicalHemostasis/AdaptiveHemostasisRobot/glb/"
        "dranmar_hemostasis_tool_compress.glb",
        "adaptive-hemostasis.png",
        38.0,
        22.0,
    ),
    ShowcaseRender(
        "data/Props/SurgicalDissection/SafePlaneDissectionRobot/glb/"
        "dranmar_safeplane_tool_exploded.glb",
        "safeplane-dissection.png",
        34.0,
        24.0,
    ),
    ShowcaseRender(
        "data/Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/glb/"
        "dranmar_anastomosis_tool_complete.glb",
        "adaptive-anastomosis.png",
        36.0,
        23.0,
    ),
    ShowcaseRender(
        "data/Props/SurgicalAssessment/PerfusionViabilityRobot/glb/"
        "dranmar_perfusion_tool_fused.glb",
        "perfusion-viability.png",
        38.0,
        22.0,
    ),
    ShowcaseRender(
        "data/Props/SurgicalOncology/OncoSurgeryCell/glb/"
        "dranmar_oncosurgery_cell.glb",
        "oncologic-resection.png",
        36.0,
        28.0,
    ),
    ShowcaseRender(
        "data/Props/SurgicalClosure/SkinStapler/glb/"
        "skin_stapler_articulated_loaded.glb",
        "skin-stapler.png",
        35.0,
        18.0,
    ),
    ShowcaseRender(
        "data/Environments/SurgicalAutonomy/AutonomousRescueOR/glb/"
        "dranmar_rescue_vessel_bleeding.glb",
        "effect-vessel-bleeding.png",
        35.0,
        23.0,
        1000,
        700,
    ),
    ShowcaseRender(
        "data/Environments/SurgicalAutonomy/AutonomousRescueOR/glb/"
        "dranmar_rescue_vessel_compressed.glb",
        "effect-vessel-compressed.png",
        35.0,
        23.0,
        1000,
        700,
    ),
    ShowcaseRender(
        "data/Environments/SurgicalAutonomy/AutonomousRescueOR/glb/"
        "dranmar_rescue_vessel_clipped_patched.glb",
        "effect-vessel-repaired.png",
        35.0,
        23.0,
        1000,
        700,
    ),
    ShowcaseRender(
        "data/Environments/SurgicalAutonomy/AutonomousRescueOR/glb/"
        "dranmar_rescue_bowel_leaking.glb",
        "effect-bowel-leaking.png",
        35.0,
        25.0,
        1000,
        700,
    ),
    ShowcaseRender(
        "data/Environments/SurgicalAutonomy/AutonomousRescueOR/glb/"
        "dranmar_rescue_bowel_repaired.glb",
        "effect-bowel-repaired.png",
        35.0,
        25.0,
        1000,
        700,
    ),
)


def _rgb(hex_color: str) -> tuple[float, float, float, float]:
    value = hex_color.lstrip("#")
    srgb = (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )

    def to_linear(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    return (
        *(to_linear(channel) for channel in srgb),
        1.0,
    )


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _bounds() -> tuple[Vector, Vector]:
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
        for corner in obj.bound_box
    ]
    if not points:
        raise RuntimeError("imported GLB contains no renderable mesh")
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    return low, high


def _material(
    name: str,
    color: str,
    roughness: float,
    metallic: float = 0.0,
    *,
    subsurface_weight: float = 0.0,
    ior_level: float = 0.5,
):
    material = bpy.data.materials.new(name)
    material.diffuse_color = _rgb(color)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = _rgb(color)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if "Subsurface Weight" in shader.inputs:
        shader.inputs["Subsurface Weight"].default_value = subsurface_weight
    if "IOR Level" in shader.inputs:
        shader.inputs["IOR Level"].default_value = ior_level
    return material


def _realistic_materials() -> dict[str, bpy.types.Material]:
    return {
        "aluminum": _material(
            "Brushed Aluminum", "#8c9499", 0.30, 0.72
        ),
        "blood": _material(
            "Blood",
            "#4a0709",
            0.52,
            subsurface_weight=0.03,
            ior_level=0.28,
        ),
        "bowel": _material(
            "Bowel Tissue",
            "#a95f58",
            0.72,
            subsurface_weight=0.12,
            ior_level=0.22,
        ),
        "drape": _material("Surgical Drape", "#1f6670", 0.72),
        "fat": _material(
            "Subcutaneous Fat",
            "#cba66d",
            0.76,
            subsurface_weight=0.08,
            ior_level=0.18,
        ),
        "floor": _material("Clinical Floor", "#343a3e", 0.62, 0.04),
        "gallbladder": _material(
            "Gallbladder",
            "#4b5632",
            0.68,
            subsurface_weight=0.07,
            ior_level=0.22,
        ),
        "graphite": _material(
            "Graphite Mechanism", "#20262b", 0.34, 0.30
        ),
        "liver": _material(
            "Liver Tissue",
            "#651c1c",
            0.68,
            subsurface_weight=0.10,
            ior_level=0.24,
        ),
        "nerve": _material("Peripheral Nerve", "#c6a858", 0.48),
        "monitor": _material(
            "Monitor Housing", "#151b20", 0.30, 0.22
        ),
        "polymer": _material(
            "Medical White Polymer", "#bfc4c7", 0.36, 0.02
        ),
        "screen": _material(
            "Inactive Clinical Display", "#071218", 0.22, 0.12
        ),
        "skin": _material(
            "Skin",
            "#9a5b49",
            0.74,
            subsurface_weight=0.08,
            ior_level=0.20,
        ),
        "spleen": _material(
            "Spleen",
            "#562839",
            0.70,
            subsurface_weight=0.09,
            ior_level=0.22,
        ),
        "stomach": _material(
            "Stomach",
            "#9a554f",
            0.72,
            subsurface_weight=0.11,
            ior_level=0.22,
        ),
        "steel": _material(
            "Surgical Stainless Steel", "#798287", 0.22, 0.86
        ),
        "tissue": _material(
            "Exposed Soft Tissue",
            "#81372f",
            0.72,
            subsurface_weight=0.10,
            ior_level=0.22,
        ),
        "tubing": _material("Medical Tubing", "#9eb8b5", 0.46),
        "urinary": _material(
            "Urinary Tissue",
            "#a96e68",
            0.72,
            subsurface_weight=0.09,
            ior_level=0.22,
        ),
        "vessel": _material(
            "Vessel Wall",
            "#731019",
            0.70,
            subsurface_weight=0.05,
            ior_level=0.22,
        ),
    }


def _material_key(object_name: str) -> str:
    name = object_name.lower()
    if name == "floor" or name.endswith("_floor"):
        return "floor"
    if "blood" in name or "bleed" in name or "leak" in name:
        return "blood"
    if "subcutaneous_fat" in name or name.endswith("_fat"):
        return "fat"
    if "liver" in name:
        return "liver"
    if any(
        token in name
        for token in ("bowel", "colon", "intestin", "mesentery")
    ) or name in {"proximal", "distal"}:
        return "bowel"
    if "gallbladder" in name:
        return "gallbladder"
    if "stomach" in name or "pancreas" in name:
        return "stomach"
    if "spleen" in name or "kidney" in name or "diaphragm" in name:
        return "spleen"
    if "bladder" in name or "ureter" in name:
        return "urinary"
    if "nerve" in name:
        return "nerve"
    if "vessel" in name or "arter" in name or "vein" in name:
        return "vessel"
    if any(token in name for token in ("skin", "patient_torso", "abdomen")):
        return "skin"
    if any(
        token in name
        for token in (
            "aperture",
            "fascia",
            "lesion",
            "organ",
            "tissue",
            "tumor",
            "wound",
        )
    ):
        return "tissue"
    if any(token in name for token in ("drape", "mattress", "pad")):
        return "drape"
    if "monitor_screen" in name or name.endswith("_screen"):
        return "screen"
    if any(token in name for token in ("monitor", "pump_rack")):
        return "monitor"
    if name.startswith("trace_"):
        return "tubing"
    if name.startswith("line_") or "tube" in name or "catheter" in name:
        return "tubing"
    if any(
        token in name
        for token in (
            "blade",
            "clip",
            "coupler",
            "jaw",
            "needle",
            "pusher",
            "shaft",
            "staple",
            "tool_",
            "wire",
        )
    ):
        return "steel"
    if any(
        token in name
        for token in ("joint", "motor", "dock_", "grip", "rubber")
    ):
        return "graphite"
    if any(
        token in name
        for token in (
            "base",
            "carriage",
            "column",
            "frame",
            "housing",
            "pedestal",
            "rail",
            "ring",
            "rotor",
            "table",
        )
    ):
        return "aluminum"
    return "polymer"


def _apply_realistic_materials() -> None:
    materials = _realistic_materials()
    mesh_objects = [
        obj for obj in bpy.context.scene.objects if obj.type == "MESH"
    ]
    for obj in mesh_objects:
        material = materials[_material_key(obj.name)]
        if obj.data.materials:
            obj.data.materials[0] = material
            while len(obj.data.materials) > 1:
                obj.data.materials.pop(index=1)
        else:
            obj.data.materials.append(material)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    if mesh_objects:
        bpy.context.view_layer.objects.active = mesh_objects[0]
        bpy.ops.object.shade_smooth_by_angle(
            angle=math.radians(32.0),
            keep_sharp_edges=True,
        )
    bpy.ops.object.select_all(action="DESELECT")


def _area_light(
    name: str,
    location: Vector,
    target: Vector,
    *,
    energy: float,
    size: float,
    color: str,
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = _rgb(color)[:3]
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    _look_at(light, target)


def _configure_scene(render: ShowcaseRender) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = render.width
    scene.render.resolution_y = render.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(MEDIA / render.output)
    scene.render.image_settings.color_depth = "8"

    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.55

    world = bpy.data.worlds.new("Dr.Anmar Neutral Clinical Studio")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = _rgb("#15191d")
    background.inputs["Strength"].default_value = 0.08


def _render(spec: ShowcaseRender) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source = ROOT / spec.source
    if not source.is_file():
        raise FileNotFoundError(source)

    bpy.ops.import_scene.gltf(filepath=str(source))
    for obj in tuple(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    _apply_realistic_materials()

    # The inspection GLBs preserve the simulation's Z-up vertex convention.
    # Rotate their imported Y-up scene root into Blender's Z-up world.
    imported_root = bpy.data.objects.new("Dr.Anmar Asset Root", None)
    bpy.context.collection.objects.link(imported_root)
    imported_root.rotation_euler.x = math.radians(-90.0)
    for obj in tuple(bpy.context.scene.objects):
        if obj is imported_root or obj.parent is not None:
            continue
        obj.parent = imported_root
    bpy.context.view_layer.update()

    low, high = _bounds()
    center = (low + high) * 0.5
    extent = high - low
    radius = max(extent.length * 0.5, 0.01)

    _configure_scene(spec)

    floor_data = bpy.data.meshes.new("Studio Floor")
    floor = bpy.data.objects.new("Studio Floor", floor_data)
    bpy.context.collection.objects.link(floor)
    floor_size = radius * 5.0
    floor_data.from_pydata(
        [
            (-floor_size, -floor_size, 0.0),
            (floor_size, -floor_size, 0.0),
            (floor_size, floor_size, 0.0),
            (-floor_size, floor_size, 0.0),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    floor.location = (center.x, center.y, low.z - radius * 0.035)
    floor.data.materials.append(
        _material("Neutral Studio Floor", "#2b3035", 0.52, 0.02)
    )

    azimuth = math.radians(spec.azimuth_deg)
    elevation = math.radians(spec.elevation_deg)
    direction = Vector(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        )
    )

    camera_data = bpy.data.cameras.new("Showcase Camera")
    camera = bpy.data.objects.new("Showcase Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.lens = 56.0
    camera_data.sensor_width = 36.0
    camera.location = center + direction * radius * 2.45 * spec.margin
    _look_at(camera, center + Vector((0.0, 0.0, extent.z * 0.03)))

    key_direction = Vector((-0.55, -0.40, 1.00)).normalized()
    fill_direction = Vector((0.75, -0.25, 0.55)).normalized()
    rim_direction = Vector((0.20, 0.85, 0.90)).normalized()
    light_scale = max(radius * radius, 0.0025)
    _area_light(
        "Key",
        center + key_direction * radius * 2.4,
        center,
        energy=220.0 * light_scale,
        size=radius * 1.35,
        color="#fff7ed",
    )
    _area_light(
        "Fill",
        center + fill_direction * radius * 2.0,
        center,
        energy=70.0 * light_scale,
        size=radius * 1.65,
        color="#eef4f8",
    )
    _area_light(
        "Rim",
        center + rim_direction * radius * 2.2,
        center,
        energy=105.0 * light_scale,
        size=radius * 1.0,
        color="#f8fbff",
    )

    MEDIA.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {spec.source} -> {spec.output}")


only = os.environ.get("DRANMAR_RENDER_ONLY")
for item in RENDERS:
    if only and item.output != only:
        continue
    _render(item)
