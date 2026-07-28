# DrAnmar 22 mm Half-Circle Taper-Point Needle Candidate

This is a new, inactive, category-level closure-needle research asset. It is
not registered in `needle_thread.py`, task configuration, the learning path,
or a runtime catalog. The entry layer defaults to `Physics=none`; `physics`
and `physx` must be selected deliberately during later qualification.

The nominal 22 mm dimension means **centerline arc length**. A half-circle
therefore has radius `0.022 / pi = 0.007002817496 m`. The
0.530 mm body is adopted from the existing DrAnmar 22 mm asset profile, whose
SHA-256 is pinned in `geometry_contract.json`. The separate newer
`physics_next` 0.520 mm profile is recorded as comparison-only and is not
silently averaged.

## Geometry and contact

- one connected, watertight swept body at final metric scale;
- finite-curvature taper-point apex;
- reduced swage body with a real 0.82 mm blind recess in the mass mesh;
- smoothly flattened driver land centered on the preferred 65% arc frame;
- three shallow longitudinal ribs on each broad grasp face;
- authored smooth normals, repeating UVs, and metric `primvars:stMeters`;
- 48 adaptive collision segments: 32 capsules and 16 convex grasp-land
  segments, plus one separate precise taper-tip convex shape;
- mass, center of mass, full inertia tensor, principal inertia and principal
  axes integrated from the actual watertight recessed render solid;
- ordered friction (`static >= dynamic`) with PhysX `min` combine, so this
  candidate cannot unexpectedly select the jaw's larger coefficient through
  `max` combine.

## Appearance

The satin-steel look uses deterministic 2048 px base-color, roughness and
normal maps. NVIDIA PhysicalAI SimReady Materials v0.2.0 provides the pinned
MIT-0 OpenPBR 1.1 MaterialX base class. `UsdPreviewSurface` is a portable
fallback. Roughness is restrained to 0.36-0.52, anisotropy is 0.12, and
clearcoat is disabled. Native RTX appearance is not yet qualified.

## Manufacturer-category boundary

The J&J MedTech / Ethicon catalog is used only to ground the published
category facts “22 mm”, “1/2 circle”, “taper point”, longitudinal ribbing and
factory swaging. DrAnmar independently authored all dimensions not listed as
catalog facts, all geometry, collision, mass properties and textures. This is
not an SH-1 or CT-3 replica, manufacturer digital twin, clinically validated
device, physics-calibrated model, or patient-care asset.

Regenerate from the asset-extension root:

```bash
uv run --no-project \
  --with numpy==2.2.6 --with Pillow==11.3.0 --with usd-core==25.11 \
  python tools/generate_needle_22_half_circle_taper_candidate.py
```
