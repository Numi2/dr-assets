# DrAnmar Atraumatic Surgical Exposure Robot v0.1.0

DrAnmar-owned OpenUSD research assets for bilateral soft-tissue capture,
force-limited retraction, maintained surgical exposure, and ROI visibility
benchmarking, integrated with NVIDIA Isaac Lab and Isaac Sim.

## Catalog path

```text
Props/SurgicalExposure/AtraumaticExposureRobot/
```

## Primary assets

- `dranmar_atraumatic_exposure_tool_payload.usda`: hand-replacement payload for `panda_link8`.
- `dranmar_atraumatic_exposure_tool_standalone.usda`: standalone articulation.
- `dranmar_atraumatic_exposure_tool_rigid_proxy.usda`: perception/planning proxy.
- `dranmar_fenestrated_retraction_pad.usda`: replaceable visual fenestrated pad variant.
- `dranmar_microcup_retraction_pad.usda`: replaceable visual microcup pad variant.
- `dranmar_exposure_tissue_demo.usda`: two deformable flaps over an ROI target.

## Mechanism

Each side has an independent lateral carriage, vertical lift, pad-pitch axis,
and 6 mm compliant force-sensing axis. Each pad exposes six independent
capture cells. Tissue capture is created at runtime from overlap-prioritized,
explicitly verified deformable vertex attachments. Any evidence-driven local
or whole-pad release latches safe relief; closed-loop hold cannot resume until
a four-stage handshake completes: commanded capture pose, immediately adjacent
no-attachment preflight, exact attachment authoring before the evidence clock
advances, and consecutive post-physics confirmation intervals satisfying the
configured dwell. Every capture interval must remain within the configured
freshness ceiling. Episode, environment, source registration, and topology
lineage remain locked throughout.

The fenestrated and microcup variants are visual alternatives in this revision.
They share the same six box capture-cell colliders, material behavior, and
attachment contract, so their visual geometry does not support comparative
claims about trapping, pressure, suction, or retention.

## Evidence status

Production hold and overload release use
`dranmar.atraumatic-exposure.scene-evidence@3.0.0` and require a prim-bound, post-physics
evidence envelope for all 12 contact cells and the registered ROI. Workcell,
cell, tissue-body, attachment, visibility, ROI, and calibration registration
are locked for the capture epoch; envelope replay, per-source raw-record reuse,
and clock discontinuity fail closed.
Each cell supplies its exact contact vector and exact attachment-reaction
vector. The conservative transmitted-load gate adds their magnitudes and may
intentionally double count overlapping load paths rather than undercount a
co-directed reaction.
The native Isaac provider that must populate this envelope is not implemented, so
simulator task completion, retained capture, tissue safety, and exposure
efficacy remain unverified. The compliance-axis force estimator is only a task
proxy and is not admissible evidence.

## Research boundary

All dimensions, friction values, tissue mechanics, capture strengths, force
thresholds, vacuum behavior, and controller gains are provisional engineering
seeds. The package does not claim calibrated tissue trauma, safe surgical
force limits, clinical effectiveness, sterility, regulatory approval, or
suitability for patient care.
