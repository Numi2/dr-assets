# Contributing

Contributions are welcome when they strengthen physical observability,
calibration, patient coupling, asset fidelity, or reproducible integration.

## Design rules

- Policies may request actions; only post-physics evidence may change patient
  outcomes.
- Preserve conservation relationships for mass, volume, flow, and inventory.
- Model both benefit and harm. Excessive force, pressure, speed, occlusion, or
  dwell should not be free.
- Keep asset directories dependency-complete and use relative USD references.
- Mark unmeasured parameters as provisional. Do not convert uncertainty into a
  generic “validated” claim.
- Preserve upstream copyright, license, and asset-level notice files.

## Changes to assets

Document the primary USD entrypoint, scale and units, articulation or deformable
model, collision representation, interaction frames, physics assumptions, and
known limitations in the asset's local README.

Geometry and binary assets should be committed in their final distributable
form. Do not commit caches, generated Python bytecode, simulator logs, or local
absolute paths.

## Changes to patient effects

Describe:

1. the raw scene measurements consumed;
2. why the policy cannot directly author them;
3. the state transition and conservation rule;
4. unsafe or damaging regimes;
5. calibration status and required physical data.

Small, reviewable changes are preferred. No ceremonial proof bundle is
required; the code and physical authority boundary should be understandable
directly.

