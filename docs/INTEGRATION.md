# Integration

## Dr.Anmar

Dr.Anmar consumes this repository as the extension at:

```text
source/extensions/orbit.surgical.assets
```

Clone Dr.Anmar with its assets:

```bash
git clone --recurse-submodules https://github.com/Numi2/drAnmar.git
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

This keeps one canonical asset history while preserving the import and
extension paths expected by existing Dr.Anmar tasks.

## Standalone Isaac Lab checkout

The repository root is both a Python project and an Omniverse Kit extension:

```text
config/extension.toml
data/
orbit/surgical/assets/
setup.py
```

From the Python environment used by Isaac Lab:

```bash
python -m pip install -e /absolute/path/to/dr-assets
```

The package retains the compatibility namespace:

```python
from orbit.surgical.assets import (
    ORBITSURGICAL_ASSETS_DATA_DIR,
    asset_path,
)
```

Isaac Lab, Isaac Sim, and their Python modules are runtime prerequisites. They
are intentionally not declared as ordinary PyPI dependencies.

## Catalog root override

To resolve a separately mounted or extracted data tree:

```bash
export DRANMAR_ASSET_DATA_ROOT=/datasets/dranmar-assets/data
```

`asset_path()` resolves catalog-relative entrypoints against this root. It does
not silently fall through to a remote bucket.

## Isaac for Healthcare interoperability

The category-relative layout follows the useful public conventions of the
[Isaac for Healthcare asset catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog):
assets are addressed beneath `Robots/`, `Props/`, and `Environments/`, and a
retrieval unit is a dependency-complete directory.

`DrAnmarSurgicalRobotAssets` exposes i4h-style relative names when
`i4h_asset_helper` is available. This repository is an independent Dr.Anmar
catalog; it is not an official NVIDIA catalog release.

## Versioning

- Repository releases version the extension and its integration surface.
- Each Dr.Anmar asset family also carries its own `asset_manifest.json`.
- A patient or procedure parameter set may evolve independently of mesh
  geometry.

Consumers should pin a Git commit or release tag. Dr.Anmar does this through
the submodule commit recorded by the parent repository.
