# DrAnmar SafePlane Dissection Robot

A DrAnmar-owned, provider-neutral NVIDIA Isaac Sim and Isaac Lab research
asset/setup package for a proposed connectivity-aware dissection task. The
package also retains private task-proxy calculations for offline engineering
analysis; those calculations are not scene observations or patient outcomes.

## Workflow

Proposed task sequence:

`inspect → capture → counter-traction → blunt spread → hydrodissect → selectively cut or apply low energy → evacuate → inspect connectivity → release`

This sequence is not an evidence-backed completion pipeline.

## Primary assets

- `dranmar_safeplane_dissection_tool_payload.usda` — Franka payload without a nested articulation root.
- `dranmar_safeplane_dissection_tool_standalone.usda` — standalone articulated mechanism.
- `dranmar_safeplane_dissection_tool_rigid_proxy.usda` — perception/planning proxy.
- `dranmar_safeplane_tissue_demo.usda` — layered tissue, adhesion network, vessel, nerve, duct, and protected organ surface.
- `dissection_topology.json` — a static authored bridge table, category-level
  proxy thresholds, undeformed structure centerlines, and a proposed completion
  rule. It is not a live scene graph or topology record.
- `glb/` — pre-authored visual previews of tool/tissue states. These are not
  recorded simulator transitions or outcome evidence.

## Boundary

The package is not clinically validated, is not a medical device, and is not
approved for patient care. Tissue, fluid, cutting, energy, force, injury, and
safety parameters remain provisional research values.

There is currently no SafePlane `SceneEvidenceEnvelope`/registry adapter that
binds exact physics steps, source prims, raw sample IDs, attachment identity,
topology revisions, tool contact, fluid deposition, thermal state, or shared
vessel/duct/nerve and patient ledgers. Public force-driven release, bridge
division, tissue-dose, protected-structure injury, complication, and completion
entry points therefore fail closed. Runtime setup utilities may cook the two
surface meshes, create attachments, and author fluid particles; those setup
operations do not establish a dissection result.
