# Provenance

DrAnmar independently generated the swept geometry, true swage recess,
flattened and ribbed grasp land, collision decomposition, frames, textures and
mass properties under Apache-2.0.

Selected internal dimensional input:

- `data/Props/SurgicalClosure/Needle/physics_profile.json`
- SHA-256 `746b3199e63cc42fefe42b94e883fb329f88c5f50c82844aa77c8573c39e9b6e`
- adopted: 22 mm centerline arc, 0.530 mm body, 48 collision segments

Comparison-only input, not averaged:

- `physics_next/needles/dr-anmar-needle-v1.json`
- SHA-256 `309981bb46ab14c86a6964d58dd623d9756c972f7ca83f7f0d1771af66d60d85`
- differs: 0.520 mm body and 40 capsules

Manufacturer-category source:

- J&J MedTech / Ethicon, `157677-230117-AH Suture Catalog`
- https://www.jnjmedtech.com/system/files/pdf/157677-230117-AH-Suture-Catalog-157677-230117.pdf
- used only for the category facts 22 mm needle length, one-half circle,
  taper point, longitudinal ribbing and factory swaging
- no vendor mesh, drawing, texture, body diameter, taper dimension, swage
  dimension, patient data or proprietary digital geometry is included

NVIDIA's byte-identical MIT-0 OpenPBR base-class inputs are documented under
`vendor/nvidia_physicalai_simready_materials_v0_2_0/PROVENANCE.md`.

No manufacturer digital-twin, native RTX appearance, physics calibration,
clinical validation, regulatory approval, or patient-care claim is made.
