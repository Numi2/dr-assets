#!/usr/bin/env python3
"""Render the real Dr.Anmar GLB inspection payload as a GitHub showcase.

Run with Blender:

    blender --background --python tools/render_showcase.py

The script deliberately renders shipped geometry rather than concept art.
It uses a consistent dark studio, a three-point light rig, and per-asset camera
angles selected to reveal the procedural work surface.
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
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
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


def _material(name: str, color: str, roughness: float, metallic: float = 0.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = _rgb(color)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = _rgb(color)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return material


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
    scene.view_settings.exposure = -0.65

    world = bpy.data.worlds.new("Dr.Anmar Studio")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = _rgb("#07111b")
    background.inputs["Strength"].default_value = 0.16


def _render(spec: ShowcaseRender) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source = ROOT / spec.source
    if not source.is_file():
        raise FileNotFoundError(source)

    bpy.ops.import_scene.gltf(filepath=str(source))
    for obj in tuple(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

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
    floor.data.materials.append(_material("Midnight Floor", "#091722", 0.3, 0.08))

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

    key_direction = Vector((-0.65, -0.45, 0.95)).normalized()
    fill_direction = Vector((0.70, -0.30, 0.45)).normalized()
    rim_direction = Vector((0.25, 0.85, 0.80)).normalized()
    light_scale = max(radius * radius, 0.0025)
    _area_light(
        "Key",
        center + key_direction * radius * 2.4,
        center,
        energy=360.0 * light_scale,
        size=radius * 1.15,
        color="#d9efff",
    )
    _area_light(
        "Fill",
        center + fill_direction * radius * 2.0,
        center,
        energy=125.0 * light_scale,
        size=radius * 1.4,
        color="#76b900",
    )
    _area_light(
        "Rim",
        center + rim_direction * radius * 2.2,
        center,
        energy=185.0 * light_scale,
        size=radius * 0.8,
        color="#52c7ff",
    )

    MEDIA.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {spec.source} -> {spec.output}")


only = os.environ.get("DRANMAR_RENDER_ONLY")
for item in RENDERS:
    if only and item.output != only:
        continue
    _render(item)
