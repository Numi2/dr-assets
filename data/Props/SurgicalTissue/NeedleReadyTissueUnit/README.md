# DrAnmar Needle-Ready Tissue Unit

This directory is the canonical geometry package for DrAnmar's first
post-handover deformable-tissue task. It contains two open wound flaps, three
conforming material layers, wound-edge-refined tetrahedra, stable semantic
coordinates, and exact nested LOD mappings. Version 2.2 adds correlated but
independently irregular wound lips, multi-scale rest shape, bounded spatial
thickness variation, depth-resolved wound-wall materials, and explicit smooth
schema normals without increasing any physics LOD count.

Use `needle_ready_tissue_training.usda` for large-batch curriculum work,
`needle_ready_tissue_contact.usda` for needle contact and intact deformation,
and `needle_ready_tissue_validation.usda` for mesh-convergence and rendered
inspection. `needle_ready_tissue_unit.usda` exposes those representations as
the `geometryLod` variant set with `contact` selected by default.

The current qualified scope is deterministic source/static geometry,
topology, semantics, nested LOD mapping, and OpenUSD composition. Because
version 2.2 changes the geometry rest state and wound-wall partition, native
Newton/Isaac intact deformation, two-way contact, fixture/retraction behavior,
visual synchronization, and performance all require a fresh run before
promotion. The asset does not create a puncture, persistent tract, cut, tear,
or thread passage. Those capabilities stay fail-closed until a
topology-capable backend and physical qualification data are present.

Material values are research seeds, not patient-specific or clinically
validated tissue properties. The nested visual package supplies OpenPBR
materials, lossless 2K maps, restrained subsurface seeds, metric semantic UVs,
depth-resolved wound appearance, and one-to-one deforming overlays. Its paired
left/right render clamp aligns with the actual outer attachment bands. This is
still a generic full-thickness layered surrogate rather than a named anatomical
site; realistic endoscopic appearance requires native RTX inspection with
calibrated lighting, exposure, camera response, and live point sync.

Regenerate from the repository root:

```bash
python tools/generate_needle_ready_tissue.py
```
