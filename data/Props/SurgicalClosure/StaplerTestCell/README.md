# Dr.Anmar Tissue Closure Bench 0.3.0

This research fixture holds the Dr.Anmar articulated skin stapler in a cradle
with a six-DOF proportional-velocity virtual fixture and exposes a bounded virtual
trigger actuator. It removes the need for a robot hand that can grasp a
gun-shaped housing. The fixture now indexes the stapler across seven guided
stations on the Dr.Anmar open-incision suturable-tissue asset.

The cell is intended to measure simulation mechanism behavior:

- commanded versus observed trigger travel;
- measured pre-fire wound-edge approximation;
- synchronized pusher travel;
- measured housing translation and rotation error;
- exactly one logical deployment on each complete threshold crossing;
- rejection of partial strokes;
- rearm behavior; and
- repeatability across cycles;
- seven rigid formed-staple visuals;
- four FEM nodes with prescribed kinematic targets per placement; and
- 6 mm closure-station spacing.

The tissue comes from the real Dr.Anmar open-incision OpenUSD asset. The authoring
tool deterministically separates its left and right connected components into
two PhysX FEM surface bodies. Their outer bands remain attached to the fixture.
At each station, two visible approximation feet drive two wound-edge nodes per
flap toward a 0.8 mm residual gap. Trigger travel does not begin until that
pre-approximation is complete.

After a measured full-threshold crossing, the room creates a kinematic formed
staple visual with collision enabled and keeps the same four tissue nodes at
prescribed kinematic targets. No `PhysxPhysicsAttachment` or other load-bearing
staple-to-tissue constraint is created. The imposed targets remain active after
the fixture advances, so the geometry stays closed even though retention has
not been modeled. Reset removes every staple visual and restores the original
tissue state.

The intended tissue target is 111.9 kPa Young's modulus and 0.461 Poisson ratio.
The interactive PhysX implementation preserves the Young's modulus but bounds
Poisson ratio to 0.40, uses measured damping 0.2307 and at least 32 solver
position iterations. Status telemetry identifies this explicitly as a
`bounded_linear_tangent_for_interactive_physx` stability proxy.

This workflow models trigger sequencing, simulated tissue motion toward a
prescribed closure pose, and visual staple placement. It does **not** model or
verify mechanical retention, staple-leg puncture, metal plasticity, pullout,
wound healing, sterility, clinical performance, or patient use. Trigger torque,
pusher force, contact friction, and all other physical values remain
provisional.

The recorded native RTX 4090 simulator run on 2026-07-24 measured an initial
3.7818 mm wound gap and verified that the approximation phase reached 0.8 mm
with the trigger still at 0 degrees. Deployment occurred only after the actual
trigger crossed 24 degrees. The run completed all seven placement stations,
created seven rigid staple visuals, applied prescribed targets to 28 FEM nodes,
preserved a 0.8 mm imposed local gap at every station, consumed exactly seven
simulated staples, and held 6 mm spacing with zero recorded spacing error. A
separate 20.0002-degree partial stroke produced no placement. Reset removed
every staple visual, restored the 3.7818 mm gap, and returned station 1 with a
full 35-staple magazine. The former retention interpretation was retracted on
source review because the implementation contains no load-bearing attachment.
See
`tissue_closure_native_simulator_evidence.json` for the recorded evidence.
