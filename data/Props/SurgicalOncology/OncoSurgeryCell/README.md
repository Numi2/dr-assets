# DrAnmar OncoSurgery Cell

Manufacturer-neutral research assets and non-authoritative task proxies for
multimodal tumor mapping, margin-constrained resection, protected-pedicle
management, specimen handling, cavity verification, and corrective resection
research in NVIDIA Isaac Lab and PhysX.

## Runtime entry points

- `dranmar_oncosurgery_tool.usda` is the lightweight payload-backed standalone
  interface.
- `dranmar_tumor_resection_tool_payload.usda` replaces the Franka hand at
  `panda_link8`.
- `dranmar_tumor_resection_tool_rigid_proxy.usda` is the lower-cost
  perception/contact route.
- `dranmar_oncology_liver.usda` is the registered task-state liver interface.
- `dranmar_oncosurgery_workcell.usda` composes the three-station research cell.
- `orbit.surgical.assets.oncologic_resection` provides Isaac Lab factories,
  configuration contracts, private task-proxy algorithms, and a fail-closed
  Dynamic Patient GPU volume-deformable liver route. Public sensor fusion,
  irreversible outcomes, observations, reward, and final success remain
  unavailable until a workcell-owned scene-evidence bridge exists.

The interface layers follow NVIDIA's reference-payload asset structure: public
variants and identity remain cheap to inspect, while the authored geometry and
physics can be unloaded in larger compositions. All composition arcs are
relative so the catalog subtree is relocatable.

## Safety and realism boundary

The package does not currently prove division, seal maturity, protected
structure integrity, blood or bile loss, specimen containment, or sensor
fusion from post-physics scene evidence. Public outcome mutation, phase
progression, RL observation/reward, and final success therefore fail closed.
The former thresholds remain private task-proxy parameters only.

The 3-D tumor field and discrete bonds are research task proxies. The demo
liver is a visibility-driven visual/topology substrate, not a deformable or
collision model. Native tissue mode can activate the Dynamic Abdominal Patient
liver's explicit tetrahedral PhysX GPU volume deformable for continuous nodal
motion and collision, but no source currently binds that body to the resection
graph, shared vessel/duct mechanics, or blood/bile ledgers. The specimen bag is
a rigid open/closed visual proxy without physical containment evidence.

The material values are uncalibrated research seeds. This package is not
clinically validated, is not a medical device, and is not approved for patient
care.
