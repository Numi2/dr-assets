# Provenance

## DrAnmar-authored content

The overlay composition, derived material parameter overrides, UV-authored
operating-table render geometry, and all nine 2048 px texture maps were
independently and
deterministically authored by the Dr.Anmar project for package version
`1.2.0`. The generator seeds and pinned dependency versions are
recorded in `asset_manifest.json`.

No patient data, clinical photography, scanned anatomy, purchased model,
internet image, or third-party texture is included.

## Referenced immutable foundations

The overlays reference, but do not copy or rewrite, these repository assets:

- `data/Robots/dVRK/PSM/psm_col.usd`: `sha256:87330b0e46e5554d53fe1b840f2db38a9bdbf00a79ca464159009406156ca0d4`
- `data/Props/Table/table.usd`: `sha256:78f61603d0605896c9076d0a2b16bc97fd6f0109783a3b1e2dd1cfd86698b7c4`
- `data/Props/Surgical_needle/needle_sdf.usd`: `sha256:5cdcd64042695815390d77c112a932801144c6f7530d3ce87b672247f91dbf6f`

The PSM foundation is ORBIT-Surgical-derived and retains its BSD-3-Clause
repository attribution. Asset-level origin metadata for the legacy table and
needle remains incomplete; this package does not upgrade or obscure that
boundary. Their exact byte hashes are locked in the manifest.

## Vendored NVIDIA OpenPBR dependency

The package includes byte-identical copies of NVIDIA's
`open_pbr_uber_base_class.usda` and `LICENSE.md` from
PhysicalAI-SimReady-Materials v0.2.0. They are MIT-0 inputs and are not
DrAnmar-authored. Exact source and destination hashes are locked in the
manifest. No NVIDIA texture, geometry, or patient data is included.

The inherited base supplies the OpenPBR 1.1 MaterialX graph. DrAnmar authors
only material parameter overrides and retains a `UsdPreviewSurface` fallback.
This is a source interoperability contract, not evidence of native RTX visual
quality.

## Validation boundary

The checked-in gates establish deterministic source integrity, OpenUSD
composition/reference closure, resolved visual material targets, UV coverage,
and absence of overlay-authored physics opinions. They do not establish native
Isaac spawn behavior, RTX appearance, physical calibration, or clinical
validity.
