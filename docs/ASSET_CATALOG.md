# Asset catalog

Every catalog family is stored as a dependency-complete directory. Keep the
directory intact when copying or publishing an asset: USD layers reference
local geometry, textures, materials, interaction frames, and physics profiles.

For rendered inspection views, see the [visual showcase](SHOWCASE.md).

## Environment and patient

| Asset | Primary entrypoint | Research surface |
| --- | --- | --- |
| Autonomous Rescue OR | `data/Environments/SurgicalAutonomy/AutonomousRescueOR/dranmar_autonomous_rescue_or.usda` | Multi-station rescue, deformable vessel, tool exchange, complications, resuscitation, ventilation |
| Dynamic abdominal patient | `data/Props/Patients/DynamicAbdominalPatient/dranmar_dynamic_abdominal_patient.usda` | Layered anatomy, scenario variants, deformable wound margins, patient physiology |

## Procedure-specific systems

| Asset | Primary entrypoint | Research surface |
| --- | --- | --- |
| Wound preparation | `data/Props/SurgicalPreparation/WoundPreparationRobot/dranmar_wound_preparation_tool_standalone.usda` | Irrigation, aspiration, brushing, debridement |
| Atraumatic exposure | `data/Props/SurgicalExposure/AtraumaticExposureRobot/dranmar_atraumatic_exposure_tool_standalone.usda` | Distributed retraction and tissue exposure |
| Adaptive hemostasis | `data/Props/SurgicalHemostasis/AdaptiveHemostasisRobot/dranmar_adaptive_hemostasis_tool_standalone.usda` | Compression, clip, patch, pressure verification |
| SafePlane dissection | `data/Props/SurgicalDissection/SafePlaneDissectionRobot/dranmar_safeplane_dissection_tool_standalone.usda` | Plane development around protected structures |
| Adaptive seal and divide | `data/Props/SurgicalDivision/AdaptiveSealDivideRobot/dranmar_adaptive_seal_divide_tool_standalone.usda` | Compression, seal-band state, division |
| Adaptive anastomosis | `data/Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/dranmar_adaptive_anastomosis_tool_standalone.usda` | Alignment, coupling, leak and patency assessment |
| Perfusion viability | `data/Props/SurgicalAssessment/PerfusionViabilityRobot/dranmar_perfusion_viability_tool_standalone.usda` | Multimodal perfusion assessment |
| OncoSurgery cell | `data/Props/SurgicalOncology/OncoSurgeryCell/dranmar_tumor_resection_tool_standalone.usda` | Margin-aware resection and specimen handling |

## Closure and count

| Asset | Directory | Notes |
| --- | --- | --- |
| Skin stapler | `data/Props/SurgicalClosure/SkinStapler` | Rigid and articulated lanes, standalone staple, placement frames |
| Skin adhesive | `data/Props/SurgicalClosure/SkinAdhesive` | Applicator state, adhesive deposition and activation surfaces |
| Closure robot | `data/Props/SurgicalClosure/ClosureRobot` | Closure tool and tissue test scene |
| Needle and thread | `data/Props/SurgicalClosure/Needle`, `NeedleThread` | Needle geometry and extended strand assets |
| Laparotomy sponge | `data/Props/SurgicalCount/LaparotomySponge` | Folded/unfolded states and wet/dry material references |

## Robot foundations

- `data/Robots/dVRK/PSM` — Patient Side Manipulator.
- `data/Robots/dVRK/ECM` — Endoscopic Camera Manipulator.
- `data/Robots/STAR` — Smart Tissue Autonomous Robot.

These foundations are ORBIT-Surgical-derived and retain their upstream
BSD-3-Clause attribution.

## Asset structure

A complete Dr.Anmar procedure asset generally contains:

```text
AssetName/
├── README.md
├── LICENSE.txt
├── asset_manifest.json
├── physics_profile.json
├── interaction_frames.json
├── *.usda
├── glb/
└── textures/
```

Not every legacy foundation uses every file. Runtime status and modeling limits
are documented in each asset's local README and manifest.
