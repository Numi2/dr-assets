# T1 table collider candidate

This is a deterministic, **inactive and unqualified** collision overlay for the
T1 operating table. It composes the exact hash-locked high-realism table visual
overlay without authoring render geometry, render visibility, materials, the
root transform, fixed joint, rigid-body properties, mass, or inertia.

The stronger layer disables the single 127,622-triangle legacy collider and
adds seven decomposed components: deck, chamfered patient pad, a thin supported
sterile-field top, center column, floor base, and separate feet. Only the drape
area supported by the pad is rigid; hanging fabric deliberately remains
non-rigid to avoid phantom contact.

`approximation_report.json` records the exact support envelopes and limits.
`geometry_contract.json` fixes meters, root, runtime `xyzw`, USD `wxyz`, unit
scale, authority, and budgets. `qualification_contract.json` blocks activation
until paired native Isaac/PhysX support and contact evidence passes.

No runtime configuration references this package. Static generation does not
establish contact stability, load calibration, physics calibration, instrument
calibration, or clinical validity.
