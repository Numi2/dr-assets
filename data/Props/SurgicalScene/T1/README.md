# DrAnmar T1 Non-Tissue Visual Package 1.2.0

This package supplies render-only OpenUSD overlays for the active T1 scene:

- `psm_visual_v1.usda` repairs unresolved visual bindings and assigns restrained
  satin steel and matte polymer without changing the referenced articulation;
- `table_visual_v1.usda` hides only the legacy table render mesh and adds an
  independently generated UV-authored rounded-edge table frame, upholstered
  pad, and woven sterile drape while retaining legacy collision;
- `legacy_needle_visual_v1.usda` deliberately subdues the compatibility
  needle instead of making its obsolete geometry visually dominant; and
- `reference_or_rig_v1.usda` provides a neutral 58 mm reference camera and
  three broad surgical-field lights for consistent asset review.

The legacy needle remains compatibility-only. Its disconnected source geometry,
mass properties, and collision contract are not repaired or promoted here.

## Authority boundary

These layers are not physics, collision, joint, transform, task, reward,
success, or clinical authority. The PSM and needle wrappers reference their
existing source assets. The table wrapper changes `visibility` only on
`/Table/Table/Table`; its collision opinion remains composed from the base
layer. New table geometry has `purpose = "render"` and no physics API.

All textures are deterministic 2048 x 2048 procedural DrAnmar work. There is no
patient imagery, scanned anatomy, web download, or third-party texture content.
Each material inherits NVIDIA's OpenPBR 1.1 MaterialX base class, vendored
unchanged from PhysicalAI-SimReady-Materials v0.2.0 under MIT-0.
`UsdPreviewSurface` remains the portable fallback.

The UV-authored table frame, pad, and drape feed their 2K base-color,
specular-roughness, and tangent-normal maps into both the OpenPBR and Preview
paths. Every normal map is lossless PNG. PSM shaft, wrist, jaws, and polymer
parts receive distinct, restrained material responses; none use a chrome-like
clear coat. The PSM and compatibility needle source meshes do not provide a
qualified UV contract, so their materials deliberately use constants rather
than pretending texture fidelity.

## Regeneration

From the asset-extension root:

```bash
uv run --no-project \
  --with numpy==2.2.6 \
  --with Pillow==11.3.0 \
  --with usd-core==25.11 \
  python tools/generate_t1_non_tissue_visuals.py
```

Native Isaac/RTX appearance, articulation spawning, and frame/contact parity
remain separate qualification steps.
This package is research-only, not clinically validated, and not approved for
patient care.
