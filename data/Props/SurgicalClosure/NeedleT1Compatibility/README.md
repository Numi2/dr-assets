# DrAnmar Needle T1 Compatibility Candidate

This package is an isolated, inactive qualification candidate for the legacy
T1 pickup/handover envelope. It is not wired into `needle_thread.py`, any task
configuration, or any runtime asset selector.

The candidate preserves the legacy analytic frame:

- circle center X: 20.040 mm;
- centerline radius: 19.154 mm;
- tip frame: `(20.040, -19.154, 0.000) mm`;
- round body diameter: approximately 1.650 mm;
- XY needle plane with +Z plane normal.

The candidate is authored at final metric size and must spawn at scale
`(1, 1, 1)` with runtime identity quaternion `(0, 0, 0, 1)`. The active legacy
needle is scaled by `(0.4, 0.4, 0.4)`. A path-only swap is therefore forbidden:
promotion must change the scale deliberately and prove composed world-space
tip, center, radius, grasp-frame, and root-orientation parity.

Unlike the inherited legacy USD, the render body is one connected watertight
taper-point solid. The apex has a finite 25 micrometre curvature seed rather
than an infinite mathematical sharpness. High-resolution render geometry has
authored UVs, smooth normals, and deterministic 2048-pixel satin-steel PBR
maps. NVIDIA's vendored OpenPBR 1.1 MaterialX graph is the primary material
context; all three maps drive that graph directly. `UsdPreviewSurface` is an
explicit portable fallback. This package makes no native RTX qualification
claim.

Runtime metadata uses the project-canonical quaternion order `(x, y, z, w)`
under fields named `orientation_xyzw`. OpenUSD-authored quaternions are
separately exposed as `orientation_usd_wxyz` because `quatd` and `quatf`
serialize `(w, x, y, z)`. The two conventions are never silently conflated.

Collision is separate from rendering: 96 overlapping capsules follow the
entire centerline and share segment endpoints. The render mesh has no collision
authority. Mass, center of mass, full inertia tensor, principal inertia, and
principal axes are integrated from the actual watertight render mesh at an
8000 kg/m3 stainless-steel engineering density seed.

Regenerate from the asset-extension root:

```bash
uv run --no-project --with numpy==2.2.6 --with Pillow==11.3.0 \
  python tools/generate_needle_t1_compatibility.py
```

## Promotion boundary

Do not replace the active needle merely because this asset is cleaner or looks
better. Promotion requires held-out parity for the qualified analytic pickup,
single-arm retention, two-arm handover, mid-air transport retention, and
native Isaac contact/CCD evidence. No such parity is claimed by this package.

This is category-level research geometry. It is not a manufacturer digital
twin, clinically validated, physics-calibrated, or approved for patient care.
