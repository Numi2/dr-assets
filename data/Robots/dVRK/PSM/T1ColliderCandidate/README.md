# T1 PSM collider candidate

This is a deterministic, **inactive and unqualified** collision overlay for the
T1 PSM. It composes the exact hash-locked high-realism PSM visual overlay and
leaves its render meshes, articulation, joints, tools, link transforms, mass,
center of mass, inertia, and material values unchanged.

The stronger layer disables exactly 11 legacy enabled colliders and adds 31
link-local components: 27 analytic primitives and four closed convex jaw
frusta totaling 48 triangles. No broad robot-wide shell is used. The
298 mm unoccupied pitch-link span remains empty, the slender roll shaft is not
inflated to its flange radius, and each jaw remains owned by its moving link.

`approximation_report.json` records measured link-local bounds and limitations.
`geometry_contract.json` fixes meters, root, runtime `xyzw`, USD `wxyz`, unit
scale, authority, and budgets. `qualification_contract.json` blocks activation
until paired native Isaac/PhysX evidence passes.

No runtime configuration references this package. Static generation does not
establish contact stability, task noninferiority, instrument calibration,
physics calibration, or clinical validity.
