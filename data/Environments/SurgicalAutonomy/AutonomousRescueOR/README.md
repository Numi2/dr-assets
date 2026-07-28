# DrAnmar Autonomous Rescue OR v0.4.0

A simulation-only rescue source package combining DrAnmar deformable surgical
substrates with evidence-gated patient-effect models. Policy requests cannot
write outcomes. The current revision is unexecuted, has no native scene
provider, exposes a public vessel interval only, and intentionally returns zero
policy reward until evidence is bound to an admitted action and a same-step
batch covers every active rescue target.

## What the current source implements

- authored room composition for NVIDIA i4h dual dVRK PSM articulations;
- an endpoint-anchored Omni Physics volume-deformable rescue-vessel asset;
- a typed exact-identity contract for post-physics jaw-to-vessel evidence;
- bilateral force, jaw-separation, tool-speed, and target-distance measurements;
- reversible compression, force-asymmetry, overload damage, residual-flow,
  blood-loss, and distal-perfusion effects;
- complication detection and rescue-plan selection that cannot author an
  outcome;
- contract-backed action admission that rejects duplicate requests, unknown
  stations or tools, capability mismatches, and misrouted system actions;
- fail-closed resource accounting that checks conserved pump withdrawal
  against available blood, crystalloid, or vasopressor inventory before any
  simulated delivery state can change;
- diagnostic patient-state delta scoring that is not exposed as reward;
- a contact-gated compression, hold, verification, and release expert that
  cannot advance on command intent alone;
- a transition-aligned Robomimic HDF5 schema whose current-revision export has
  not been recorded or qualified.

The bowel-anastomosis, abdominal-wall, and occlusive-film update branches do
not have public exact evidence adapters in this revision and are unreachable
from `AutonomousRescueORRuntime`. Pump and ventilation evidence have their own
adapters, but there is no atomic patient-wide device/target batch or rollback.

## Historical imitation-learning workflow

Start the live room, then start its expert. Expert start resets the room and
starts recording automatically:

```bash
./dr_anmar_rescue_il.sh room
./dr_anmar_rescue_il.sh expert
```

The scripts are intended to write the normal `.npz` package plus a
`dr.anmar.autonomous-rescue-imitation.v1` HDF5 episode. The HDF5 transition
uses `state[i] -> Cartesian action[i+1] -> state[i+1]` because the workstation
samples frame state after physics applies that frame's action. Patient effects
never appear in the action contract. Current source returns zero reward because
effect evidence is not yet causally bound to an admitted action and complete
target batch.

For seeded expert generation across visual and control perturbations:

```bash
./dr_anmar_rescue_il.sh collect 2361 30
```

Merge complete episodes and train the low-dimensional rescue policy:

```bash
./dr_anmar_rescue_il.sh episodes
./dr_anmar_rescue_il.sh pack rescue_train.hdf5 episode_*.hdf5
./dr_anmar_rescue_il.sh train rescue_train.hdf5
```

Train and validation masks are assigned by complete episode, never by
individual frame. The exact feature order, validity masks, source-frame
offset, and promotion gates are defined in
`imitation_dataset_contract.json`.

The large OR USD contains four authored station and tool-change layout frames.
Those are composition metadata, not four live Franka articulations. A host may
populate them with compatible robots. The optional
`autonomous_rescue_scene_cfg()` composition does populate those frames with
four Franka/tool articulations; the current DrAnmar workstation room instead
uses the real dual-PSM task it actually instantiates.

The procedure, protocol, benchmark, resource, and tool JSON files describe
intent and scene setup. They are not a safety kernel and they do not grant
success. Scenario parameters may initialize a physical defect or fault before
an episode; policy actions cannot write clinical result fields.

## Intended patient-effect path and current boundary

Policies request interventions but cannot write bleeding, occlusion, seal,
perfusion, leak, closure, or success values. An eventual native environment
must submit monotonic
post-physics observations containing bilateral contact forces, tool speed,
measured separation, retained attachments, patch contact points, and leaked
particle counts. Hemostasis verification additionally requires measured
upstream pressure; requesting a challenge cannot synthesize one. The DrAnmar
runtime also expires active repair and ventilation effects when their evidence
is not refreshed in the current physics interval. It derives:

- temporary vessel compression from bilateral contact, symmetry, gap, speed,
  and proximity to the authored defect-control frame;
- definitive clip control from retained attachment state and contact dwell;
- patch sealing from distributed contact and dwell;
- residual bleeding and conserved blood loss;
- distal-perfusion tradeoffs, shared-patient MAP coupling, and overload damage;
- repair approximation, retention and leak;
- occlusive-film integrity from eight perimeter bonds, distributed contact,
  measured cavity pressure, leak, and a continuous pressure-hold window;
- crystalloid, blood-product, and vasopressor delivery from mutually
  consistent plunger travel, outlet flow, reservoir mass loss, vascular-line
  attachment, and line pressure;
- ventilation support from airway attachment, valve travel, net circuit flow,
  airway pressure, measured oxygen fraction, and observed chest excursion;
- oxygenation support from airway attachment, valve position, circuit flow and
  leak, airway pressure, delivered oxygen fraction, and measured chest motion;
- complication detection and rescue priority; patient-state delta is
  diagnostic only and policy reward remains zero;
- hemostasis only after a continuous measured pressure-challenge evidence
  window.

Only the rescue-vessel path is publicly wired. The repair bullets above are
model capabilities, not integrated runtime coverage. No current native
execution, provider binding, same-step batch, rollback proof, or physical
calibration is claimed.

Runtime modules:

```text
orbit.surgical.assets.autonomous_rescue_or
orbit.surgical.assets.deformable_rescue
orbit.surgical.assets.resuscitation_effects
```

## Catalog path

```text
Environments/SurgicalAutonomy/AutonomousRescueOR/
```

## Important boundary

The geometry and runtime are implemented for simulation training. Calibration
values are provisional engineering seeds.

This package is research software. It is not clinically validated,
patient-specific, a medical device, or approved for patient care.
