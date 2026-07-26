# Contact-driven patient effects

## The problem this repository addresses

In surgical reinforcement learning, a reward is not trustworthy merely because
it is wrapped in a validator. If an action handler can directly set
`bleeding_controlled`, `repair_complete`, or `patient_stable`, the policy is
being trained against authored bookkeeping rather than the simulated
intervention.

Dr.Anmar removes that write path. Policy actions describe intent. Only the
environment-owned scene adapter can publish a monotonic, post-physics evidence
frame. Effect integrators then update patient state from mutually supporting
measurements.

## Authority chain

1. **Policy intent** requests motion or an intervention such as compress, clip,
   patch, infuse, ventilate, or verify.
2. **Isaac Sim and PhysX** advance articulations, deformables, particles,
   contacts, attachments, and sensors.
3. **Scene adapters** read post-step evidence. They do not accept success or
   patient-outcome fields.
4. **Effect integrators** apply conservation, dwell, overload, retention, leak,
   and damage dynamics.
5. **Patient physiology** receives blood loss, circulating-volume support,
   perfusion, ventilation, oxygenation, and tissue damage.
6. **Reward** observes patient-state transitions and penalizes harm, wasted
   resources, stale effects, and unsafe interaction.

## Implemented evidence

The rescue pathway can consume:

- bilateral contact forces and force symmetry;
- measured tool separation, speed, and distance to the defect;
- retained clip or patch attachments and distributed contact;
- upstream pressure, leaked particle count, and pressure-hold dwell;
- pump plunger travel, outlet flow, reservoir mass loss, access attachment,
  line pressure, and extravasation;
- airway attachment, valve travel, inspiratory and leaked flow, airway
  pressure, delivered oxygen fraction, and measured chest excursion.

No single requested action is sufficient. For example, infused volume is the
mutually supported portion of plunger displacement, downstream flow, and
reservoir mass loss while vascular access and acceptable pressure are present.

## Persistence and stale evidence

Temporary compression and ventilation effects must be refreshed by current
physics evidence. Retained mechanical repairs persist only while their
attachment and integrity state persists. This prevents an agent from touching a
target once and collecting indefinite benefit.

## Learning-data boundary

Autonomous Rescue OR declares a transition-aligned imitation-learning contract:

```text
state[i] → Cartesian action[i+1] → state[i+1]
```

The offset reflects when the workstation samples post-physics state. Robot,
contact, vessel, vital-sign, and fluid-balance signals are observations.
Patient-effect fields are never policy actions. Train and validation masks are
assigned by complete episode so adjacent frames from one rescue do not leak
across the split.

The machine-readable feature order and promotion criteria are in
`data/Environments/SurgicalAutonomy/AutonomousRescueOR/imitation_dataset_contract.json`.

## Damage and tradeoffs

More force is not monotonically better. Asymmetry, excessive speed, overpressure,
poor placement, occlusion, and sustained overload can increase damage or reduce
distal perfusion. The optimal behavior is therefore constrained by the same
measured interaction that produces benefit.

## Calibration boundary

The architecture is ready for measured parameter identification; its default
numbers are not clinical truth. Calibration records should identify:

- specimen or bench model;
- instrument and contact geometry;
- synchronized force, displacement, pressure, flow, and imaging channels;
- coordinate transforms and units;
- fitted parameters and uncertainty;
- held-out protocols and failure envelopes.

That evidence belongs beside a versioned parameter set. It should not be
collapsed into a generic “validated” boolean.

## Primary runtime modules

```text
orbit/surgical/assets/deformable_rescue.py
orbit/surgical/assets/resuscitation_effects.py
orbit/surgical/assets/autonomous_rescue_or.py
orbit/surgical/assets/dynamic_abdominal_patient.py
```
