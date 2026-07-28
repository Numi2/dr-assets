# T1 PSM jaw-contact candidate

This directory contains an **inactive, unqualified** OpenUSD composition that
references `../psm_col.usd` at `/psm` and changes only
`physics:staticFriction` and `physics:dynamicFriction` on
`/psm/Looks/PhysicsMaterial`. The locked base binds that material only to:

- `/psm/psm_tool_gripper1_link/collisions_xform/collisions`
- `/psm/psm_tool_gripper2_link/collisions_xform/collisions`

The selected seed (`static=0.60`,
`dynamic=0.45`) is an engineering hypothesis for a
steel/serrated-jaw contact model. It is not measured, physics-calibrated,
instrument-calibrated, or clinically validated. The layer adds no adhesion,
magnetism, suction, attachment, force injection, geometry, collision, joint, or
transform opinions.

`friction_hypothesis.json` defines the bounded, correlated sensitivity
envelope. `qualification_contract.json` blocks activation until native PhysX
evidence covers analytic pickup, first-attempt retention, mid-air transport,
two-arm handover, contact/normal-force telemetry, friction sensitivity, and
held-out seeds.

No task or robot configuration selects this asset. Use of the wrapper as a
runtime replacement requires an explicit review and a separately committed
activation change after the qualification contract passes.
