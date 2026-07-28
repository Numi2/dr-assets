# DrAnmar Multimodal Perfusion Assessment Robot v0.1.1

Dr.Anmar executable simulation-training workcell for registered RGB, NIR/ICG,
laser-speckle, thermal, Doppler, ultrasound, depth, and surface-oxygenation
assessment workflows.

## Catalog path

`Props/SurgicalAssessment/PerfusionViabilityRobot`

## Primary contract

The built-in vascular graph, tracer transport, modality maps, condition
controller, and scan/intervene/rescan loop are explicitly synthetic fixtures.
Caller-supplied condition and recovery values are fixture inputs only. They are
not real scan, intervention, patency, viability, recovery, or completion
evidence.

Production-facing assessment currently abstains even when a structurally valid
envelope is supplied, because this package cannot prove that a native provider
issued it or causally bind an admitted action/target to the observed effect.
Promotion would require:

- an environment-issued provider receipt for post-physics
  `SceneEvidenceEnvelope` data from `dr.anmar.perfusion.native-scene-bridge`;
- exact source and patient target prim paths registered at scene creation;
- raw native record identities and hashes of every modality payload;
- monotonic episode, environment, physics-step, time, and topology identity;
- an immutable snapshot binding to the exact live `DynamicSurgicalPatient`
  instance; and
- separate prim-bound mechanics evidence between before and after scans for any
  simulator task-completion statement; and
- one admitted action identity, exact target, provider interval, and patient
  mutation revision joined in a causal receipt.

This package defines most of that envelope contract but deliberately keeps
modality availability/confidence at zero and completion false. It does **not**
implement the required Isaac/RTX camera, depth, optical-map, Doppler,
ultrasound, contact, attachment, and DynamicSurgicalPatient sampling bridge.
The existing native-simulator record establishes asset loading, articulation,
camera rendering, and fixture execution on the recorded stack only.

## Native bridge still required

The missing environment-owned bridge must, after each physics step:

1. resolve every registered source, patient target, probe-contact pair, and
   optional attachment against live stage prims;
2. snapshot RGB, depth, NIR/ICG, speckle, thermal, oxygenation, signed Doppler,
   and ultrasound B-mode/flow-derived relative lumen-signal buffers from real
   runtime providers, never from `perfusion_network.json`;
3. record the raw provider IDs, payload hashes, capture clocks, transform chain,
   registration residuals, and calibration-profile identity;
4. capture the exact live `DynamicSurgicalPatient` object at the same fusion
   time and bind its immutable snapshot digest to the envelope;
5. capture post-physics intervention displacement, force, contact-pair, and
   attachment identity between independently acquired before and after scans;
6. publish the immutable envelope with monotonic episode, environment,
   physics-step, simulation-time, and topology identity; and
7. bind the admitted action, exact target, patient identity/reset/mutation
   revision, and resulting state transition in a provider-issued receipt; and
8. preserve topology-transition lineage when an intervention legitimately
   changes anatomy. The current verifier intentionally rejects a changed
   topology because that lineage provider does not yet exist.

Physical calibration and external validation remain separate requirements even
after this software bridge exists.

Real-world modality physics, physical calibration, clinical viability,
clinical patency, clinical recovery, and patient-care evidence are not
established. The asset is not approved for patient care.
