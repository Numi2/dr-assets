# DrAnmar Needle-Ready Tissue Unit

This directory is the canonical geometry package for DrAnmar's first
post-handover deformable-tissue task. It contains two open wound flaps, three
conforming material layers, wound-edge-refined tetrahedra, stable semantic
coordinates, and exact nested LOD mappings.

Use `needle_ready_tissue_training.usda` for large-batch curriculum work,
`needle_ready_tissue_contact.usda` for needle contact and intact deformation,
and `needle_ready_tissue_validation.usda` for mesh-convergence and rendered
inspection. `needle_ready_tissue_unit.usda` exposes those representations as
the `geometryLod` variant set with `contact` selected by default.

The current qualified scope is deterministic source/static geometry,
topology, semantics, nested LOD mapping, and OpenUSD composition. Because
version 2.1 changes wound-edge refinement and fixture membership, native
Newton/Isaac intact deformation, two-way contact, fixture/retraction behavior,
visual synchronization, and performance all require a fresh run before
promotion. The asset does not create a puncture, persistent tract, cut, tear,
or thread passage. Those capabilities stay fail-closed until a
topology-capable backend and physical qualification data are present.

Material values are research seeds, not patient-specific or clinically
validated tissue properties. The nested visual package supplies OpenPBR
materials, 2K maps, restrained subsurface seeds, and one-to-one deforming
overlays; realistic endoscopic appearance still requires native RTX inspection
with calibrated lighting, exposure, camera response, and live point sync.

Regenerate from the repository root:

```bash
python tools/generate_needle_ready_tissue.py
```
