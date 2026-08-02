# DrAnmar Needle System 0.3.0

Research-only category-level surgical closure assets for OpenUSD, NVIDIA Isaac Sim and Isaac Lab.
They are not clinically validated, not a manufacturer digital twin and not approved for patient care.

## Assets

- `Props/SurgicalClosure/Needle/dranmar_needle.usda`
  - watertight half-circle round-body taper-point needle;
  - explicit swage recess;
  - compound capsule collision;
  - explicit mesh-derived mass, center of mass and principal inertia;
  - robot grasp, tip, swage and count frames.

- `Props/SurgicalClosure/Needle/dranmar_needle_thread_fem.usda`
  - task-compatible compound asset with exactly one rigid curved needle;
  - one 180 mm, 0.25 mm-equivalent triangular surface-FEM strand;
  - 722 simulation/collision vertices and 720 triangles at task scale;
  - hard six-vertex swage attachment to the rigid needle;
  - independent axial, shear and bending response from the DrAnmar 4-0 profile;
  - speculative CCD, self-collision and bounded depenetration;
  - no rigid segment chain or D6 joints.

- `Props/SurgicalClosure/NeedleThread/dranmar_needle_thread.usda`
  - rigid needle plus 180 mm violet braided 4-0 thread;
  - 120 independently moving rigid thread segments;
  - D6 axial, bend and twist constraints;
  - explicit swage joint;
  - coiled initial configuration.

- `Props/SurgicalClosure/NeedleThread/dranmar_needle_thread_extended.usda`
  - the same mechanism in an extended initial configuration.

- `Props/SurgicalClosure/NeedleThread/dranmar_needle_thread_rigid_proxy.usda`
  - single rigid body for perception, handover, positioning and dataset generation.

## Coordinate convention

- SI units; Z-up.
- The needle lies primarily in the XY plane.
- The tip is at the lower end of the half-circle.
- The swage is at the upper end.
- Frame quaternions are WXYZ and authored using flat USDA syntax.

## Integration

The repository overlay installs the catalog assets under the existing
`orbit.surgical.assets` data tree and provides `needle_thread.py` with:

- `make_needle_cfg`
- `make_needle_thread_rigid_proxy_cfg`
- `spawn_segmented_needle_thread`
- `frame_path`

The segmented assembly is raw USD maximal-coordinate physics. It is not exposed as
an Isaac Lab reduced-coordinate articulation because its thread uses general D6 joints.
It is retained for compatibility and authoring reference, not for the native tissue
puncture task. The puncture task uses `dranmar_needle_thread_fem.usda`; the neutral
asset root contains sibling `NeedleRigid` and `ThreadFEM` physics actors because
Omni Physics does not permit a transform-inheriting deformable below a rigid body.

Author or regenerate the FEM asset with
`scripts/author_dranmar_needle_thread_fem.py` from the parent DrAnmar repository.
Qualify it on the pinned CUDA PhysX runtime with
`scripts/validate_dranmar_needle_thread_fem.py`. The gate checks one rigid body,
one deformable, no legacy joints, finite nodal state, actual free-strand motion,
and hard swage retention before task integration.

### NVIDIA design basis

The FEM asset follows the current Omni Physics deformable model rather than
the deprecated particle-cloth or `PhysxPhysicsAttachment` paths:

- the simulation strand is a triangular `UsdGeomMesh`, because curve
  deformables are not supported;
- `OmniPhysicsVtxXformAttachment` hard-attaches the swage vertices to the one
  rigid needle body;
- `UsdShadeMaterialBindingAPI` binds the strand material with the `physics`
  purpose, while the high-friction jaw material is scoped only to
  `NeedleRigid` so cloning cannot overwrite the deformable binding;
- solver iterations, damping, velocity limits, self-collision and speculative
  CCD are authored on the deformable body.

References: [Omni Physics deformable schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/omniphysics_deformable_schema.html),
[deformable authoring](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_authoring.html),
[PhysX deformable schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/physx_deformable_schema.html), and
[deformable migration](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/deformable_migration.html).

## Provisional values

Dimensions and physics parameters are category-level engineering seeds. The needle mass
and inertia are derived from the actual generated solid mesh. Thread segment mass is based
on a 0.25 mm diameter, 180 mm long, 1300 kg/m³ braided-polymer proxy with a solver floor
of 1e-7 kg per segment. Joint stiffness, damping, friction, swage pullout and break values
remain provisional and should be calibrated by the user.

The FEM representation is simulator-engineering evidence only. Its shell
stiffnesses preserve the profile's relative stretch and bending behavior, but
they are not biomechanical or clinical validation of a particular suture.
