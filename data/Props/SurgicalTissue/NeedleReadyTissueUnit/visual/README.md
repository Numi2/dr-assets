# DrAnmar Needle-Ready Tissue Visual Package

This directory contains a visual-only research presentation layer for the
canonical needle-ready tissue asset. It adds deterministic 2048 px PBR maps,
an NVIDIA-derived OpenPBR 1.1 MaterialX graph, portable `UsdPreviewSurface`
fallbacks, and a render-purpose support cassette/drape. It does not replace
or modify the training, contact, or validation TetMesh assets.

The overlays reuse the existing `Visual` mesh and author face-varying UVs on
that mesh. This preserves the one-to-one particle/visual point order required
by the Isaac Lab Newton/Fabric visual sync. No detached high-resolution mesh
is introduced.

The support cassette spans from the tissue lower face near -3 mm to the draped
table plane near -20 mm, making the 17 mm scene offset visually intelligible.
It is explicitly render-only: it has no collision API, rigid-body API, mass,
contact material, or solver authority.

Regenerate from the asset-extension root:

```bash
uv run --no-project --with numpy==2.2.6 --with Pillow==11.3.0 \
  python tools/generate_needle_ready_tissue_visuals.py
```

The generator is deterministic. `visual_manifest.json` recursively hashes the
generated package and records the immutable base-physics hashes. Base color
maps use sRGB; roughness and subsurface-weight maps use raw PNG data. Normal
maps use raw tangent-space quality-99 JPEG with 4:4:4 sampling, matching the
portable texture practice in NVIDIA's source library without reducing the 2K
resolution. Moisture is sparse and bounded rather than a uniform glossy
coating.

The unchanged OpenPBR base class and a medium-skin micro-normal input come
from NVIDIA `PhysicalAI-SimReady-Materials` release `v0.2.0` under MIT-0.
Their exact archive/member hashes and license are retained under `vendor/`.
The published normal contributes bounded surface microstructure only; it does
not supply geometry, biomechanics, patient imagery, or clinical calibration.

These colors, scattering distances, geometry, and textures are research
seeds. They are not patient-specific, clinically color-calibrated, physically
calibrated, or approved for patient care. Native RTX material compilation,
lighting/camera calibration, and screenshot qualification remain separate
runtime gates.
