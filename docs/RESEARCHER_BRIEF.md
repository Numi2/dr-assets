# Dr.Anmar Assets: researcher brief

## Research thesis

Surgical robot learning becomes more credible when the policy controls motion
and intervention intent while the simulated patient owns the result.

Dr.Anmar Assets is built around that separation. Isaac Sim and PhysX advance
the scene; post-physics adapters publish measured evidence; effect integrators
update damage, repair, fluid balance, ventilation, perfusion, and physiology;
reward observes the resulting patient transition. There is no policy action
for writing `bleeding_controlled`, `repair_complete`, or `patient_stable`.

## What ships

| Layer | Repository surface | Research use |
| --- | --- | --- |
| OpenUSD assets | Procedure instruments, dVRK/STAR foundations, dynamic abdominal patient, Autonomous Rescue OR | Composition, articulation, collision, deformable, particle, and sensor scenes |
| Interaction contracts | Local frames, physics profiles, attachment sites, tool modes, task contracts | Reproducible scene integration and observation semantics |
| Patient-effect runtime | Rescue vessel, resuscitation, physiology, deformable rescue, dynamic patient modules | Contact-, flow-, pressure-, retention-, and damage-driven outcomes |
| Learning interface | Isaac Lab adapters and transition-aligned imitation dataset contract | RL and behavioral-cloning observations without outcome-write actions |
| Inspection media | GLB states and reproducible studio renderer | Fast visual review of geometry, mechanisms, and authored state surfaces |

## Authority boundary

```text
policy intent
    ↓
articulation + contact + deformable + particle simulation
    ↓
monotonic post-physics evidence frame
    ↓
effect integration + conservation + damage
    ↓
patient physiology
    ↓
transition reward and termination
```

Temporary effects require fresh evidence. Retained repairs require persistent
attachment and integrity. Mutually supported measurements are used where one
signal is easy to spoof: delivered infusion, for example, is bounded by
plunger displacement, outlet flow, reservoir loss, vascular access, and
acceptable pressure.

## Machine-readable research surfaces

- [`imitation_dataset_contract.json`](../data/Environments/SurgicalAutonomy/AutonomousRescueOR/imitation_dataset_contract.json)
  defines causal frame alignment, observation order, action semantics, episode
  splitting, and reference-demonstration promotion.
- [`mechanics_contract.json`](../data/Environments/SurgicalAutonomy/AutonomousRescueOR/mechanics_contract.json)
  declares scene mechanics and patient coupling.
- [`physics_profile.json`](../data/Environments/SurgicalAutonomy/AutonomousRescueOR/physics_profile.json)
  records the simulation profile for the rescue environment.
- [`safety_invariants.json`](../data/Environments/SurgicalAutonomy/AutonomousRescueOR/safety_invariants.json)
  records invariant expectations independently of policy reward.
- [`episode_schema.json`](../data/Environments/SurgicalAutonomy/AutonomousRescueOR/episode_schema.json)
  defines episode evidence and outcome data.

## High-value research work

The next scientific step is not another success validator. It is parameter
identification against synchronized bench or specimen measurements:

1. capture force, displacement, pressure, flow, imaging, and failure data;
2. identify parameters and uncertainty for a versioned geometry and material
   configuration;
3. evaluate held-out interventions and failure envelopes;
4. train policies against the identified patient-effect model;
5. measure whether behavior transfers across parameter and geometry variation.

This repository provides the asset, evidence, and learning boundaries for that
work. Its current default parameter sets are engineering seeds, not clinical
calibration.

## Suggested review sequence

1. Open the [visual showcase](SHOWCASE.md) to inspect the actual shipped
   geometry and patient-effect states.
2. Read [contact-driven effects](CONTACT_DRIVEN_EFFECTS.md) for the implemented
   authority chain and stale-evidence behavior.
3. Inspect the [Autonomous Rescue OR package](../data/Environments/SurgicalAutonomy/AutonomousRescueOR)
   for the complete environment, contracts, and complication surfaces.
4. Read [integration](INTEGRATION.md) for Isaac Lab and submodule use.
5. Read [research boundaries](RESEARCH_BOUNDARIES.md) and
   [provenance](PROVENANCE.md) before citing claims.

## Attribution boundary

Dr.Anmar owns the patient-effect architecture, procedure assets, scene
composition, interaction contracts, and research integration in this
repository. NVIDIA Isaac Sim, Isaac Lab, PhysX, and Isaac for Healthcare are
technical foundations or interoperability targets and are not bundled here.
The compatibility namespace and identified robot foundations retain their
ORBIT-Surgical-derived attribution. No NVIDIA endorsement is implied.
