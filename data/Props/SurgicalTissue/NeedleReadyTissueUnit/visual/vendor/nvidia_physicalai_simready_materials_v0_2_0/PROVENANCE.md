# NVIDIA PhysicalAI SimReady Material Inputs

This directory vendors three immutable inputs from NVIDIA's
`PhysicalAI-SimReady-Materials` release `v0.2.0`:

- `open_pbr_uber_base_class.usda`
- `Skin_Medium_normal.jpg`
- `LICENSE.md`

Upstream:
`https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials`

Release archive:
`https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials/releases/download/v0.2.0/Skin.zip`

Release archive SHA-256:
`e6831eb8129179d9d05be80a582fe0f5824a7c2d9f5252db0f7f4a3102082380`

Vendored member SHA-256 values:

- `open_pbr_uber_base_class.usda`:
  `bb76ff9fa9cd74b86b6be4ed3c6ed79cdca15eff6d603ca571bdf9ce21e10c5f`
- `Skin_Medium_normal.jpg`:
  `a881565f49e80e5c486f84fbd1e87595515199d67063009e68415e791d190bc5`
- `LICENSE.md`:
  `18f74283f08ff1ed39a9c46dbe2622146d45f771023c3dbd9c631bb058e1421b`

The upstream package is MIT-0. The OpenPBR 1.1 MaterialX base class is used
unchanged. The published medium-skin tangent-space normal is used only as a
microstructure input to the deterministic DrAnmar surface-normal generator;
it does not supply geometry, biomechanics, patient data, clinical
calibration, or task semantics. DrAnmar's output textures remain research
art and must pass native RTX review before promotion.
