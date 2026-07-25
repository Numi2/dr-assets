# Research boundaries

Dr.Anmar Assets is simulation research software.

## Implemented

- OpenUSD geometry, materials, variants, interaction frames, and physics
  profiles for the cataloged surgical systems.
- Isaac Lab configuration helpers and scene adapters.
- Contact-, attachment-, pressure-, flow-, inventory-, and dwell-driven
  patient-effect models.
- Patient physiology coupling for blood loss, circulating-volume support,
  perfusion, ventilation, oxygenation, and modeled damage.
- Reward derived from patient-state transitions rather than an action-authored
  success flag.

## Provisional

- Tissue constitutive parameters and failure thresholds.
- Instrument force and pressure envelopes.
- Fluid, ventilation, oxygenation, and pharmacodynamic coefficients.
- Sensor noise, domain-randomization ranges, and patient population priors.
- Correspondence between simulated outcomes and biological outcomes.

Values marked `provisional_engineering_seeds` are starting points for
identification and sensitivity analysis.

## Not claimed

This repository is not:

- clinically validated;
- patient-specific;
- a medical device;
- a diagnostic or treatment-planning system;
- evidence of clinical efficacy or safety;
- approved for patient care.

Policy performance in simulation is not clinical performance. Real-world use
requires independent calibration, validation, risk management, regulatory
review, and human oversight appropriate to the intended application.

