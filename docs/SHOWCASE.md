# Visual showcase

These are studio renders of the inspection GLBs shipped in this repository.
They show the actual asset geometry and authored states; they are not generated
concept art. OpenUSD composition and the Isaac runtime add articulation,
contacts, attachments, particles, deformables, sensors, and patient effects.

## Autonomous Rescue OR

![Autonomous Rescue OR](media/hero-autonomous-rescue-or.png)

The rescue environment composes the patient, intervention stations, sterile
tool carousel, physiological monitor, resuscitation module, and shared
workspace. The visual scene is paired with post-physics scene adapters and
patient-effect integrators rather than action-authored rescue outcomes.

## Procedure systems

| Oncologic resection | SafePlane dissection |
| --- | --- |
| ![Three-arm oncologic resection cell](media/oncologic-resection.png) | ![Exploded SafePlane dissection mechanism](media/safeplane-dissection.png) |
| Multi-arm sensing, resection, margin state, and specimen handling. | Interchangeable dissection, traction, hydro, energy, and sensing components. |

| Adaptive hemostasis | Adaptive anastomosis |
| --- | --- |
| ![Adaptive hemostasis instrument](media/adaptive-hemostasis.png) | ![Adaptive anastomosis instrument](media/adaptive-anastomosis.png) |
| Compression, clip, patch, suction, irrigation, and verification. | Alignment, approximation, staple formation, reinforcement, leak test, and patency. |

| Perfusion viability | Articulated skin stapler |
| --- | --- |
| ![Multimodal perfusion instrument](media/perfusion-viability.png) | ![Articulated skin stapler](media/skin-stapler.png) |
| Multimodal optical, thermal, ultrasound, and Doppler assessment surfaces. | Trigger, pusher, magazine, placement, and deployable-staple geometry. |

## Patient and repair states

![Dynamic abdominal patient](media/dynamic-abdominal-patient.png)

The dynamic patient provides layered abdominal access, internal organs,
vasculature, nerves, scenario pathology, respiration states, and deformable
wound margins.

The following inspection assets expose state geometry used by the rescue
environment. Patient outcome is still computed from live contact, attachment,
flow, pressure, leak, dwell, and physiology evidence at runtime.

| Uncontrolled vessel | Compressed vessel | Retained repair |
| --- | --- | --- |
| ![Uncontrolled vessel](media/effect-vessel-bleeding.png) | ![Compressed vessel](media/effect-vessel-compressed.png) | ![Clipped and patched vessel](media/effect-vessel-repaired.png) |

| Leaking bowel ends | Repaired anastomosis |
| --- | --- |
| ![Leaking bowel ends](media/effect-bowel-leaking.png) | ![Repaired bowel anastomosis](media/effect-bowel-repaired.png) |

## Reproduce the views

The renderer fixes the simulation-to-GLB axis conversion at the imported scene
root and scales the studio lights by model area, avoiding the rotated and
overexposed previews that generic GLB viewers can produce.

```bash
blender --background --python tools/render_showcase.py
```

Render a single view:

```bash
DRANMAR_RENDER_ONLY=skin-stapler.png \
  blender --background --python tools/render_showcase.py
```

