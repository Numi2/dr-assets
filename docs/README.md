# Dr.Anmar Assets extension

This Kit extension provides Dr.Anmar's OpenUSD surgical asset catalog, Isaac
Lab configurations, and contact-driven patient-effect modules.

The project keeps the `orbit.surgical.assets` namespace for compatibility with
the ORBIT-Surgical-derived task substrate used by Dr.Anmar. New development,
release metadata, procedure assets, and patient-effect architecture are owned
and maintained by Dr.Anmar.

Start with the repository [README](../README.md), [asset catalog](ASSET_CATALOG.md),
and [contact-driven effects](CONTACT_DRIVEN_EFFECTS.md).

```python
from orbit.surgical.assets import asset_path

rescue_or_usd = asset_path("autonomous_rescue_or")
```
