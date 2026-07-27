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

The current qualified scope is geometry and intact deformable contact. The
asset does not create a puncture, persistent tract, cut, tear, or thread
passage. Those capabilities stay fail-closed until a topology-capable backend
and physical qualification data are present.

Material values are research seeds, not patient-specific or clinically
validated tissue properties. The matte preview materials are intended to avoid
the wet-plastic rendering seen in older screenshots; realistic endoscopic
appearance still requires calibrated lighting, subsurface scattering, texture,
and camera response.

Regenerate from the repository root:

```bash
python tools/generate_needle_ready_tissue.py
```
