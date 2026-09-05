# Dashboard simulation policy

The dashboard must keep genuine frozen-detector output visibly separate from
demonstration data. A simulation is useful for explaining an interface, but it
is not evidence that the model detects real or moderate photovoltaic faults.

## Mode 1 — Frozen model / real pipeline

- Use the frozen StandardScaler, frozen Isolation Forest, exact seven-feature
  order, and frozen threshold `0.7135242760182695`.
- Score real or explicitly selected operational records through
  `src.anomaly_detection/inference.py`.
- Display the actual continuous score, signed threshold margin, and strict
  `NORMAL`/`ALERT` result. Never force or substitute an alert.
- Preserve data provenance and timezone-naive timestamps; the source timezone
  remains unspecified.
- Log an event only from real `ALERT` rows, grouped by exact two-minute
  continuity without smoothing.

## Mode 2 — Simulation demonstration

- Label every affected view, row, export, and event **Simulation** or **Demo**.
- Store demo inputs/outputs separately from baseline and final evaluation data.
- Deliberately stronger replay perturbations may illustrate possible UI states,
  but must still pass through the unchanged frozen inference pipeline.
- Do not promise that a perturbation will cross the threshold, lower the
  threshold to force a crossing, or replace actual detector output.
- Do not report demo results as model-evaluation evidence, use them to
  recalculate Day 22 metrics, overwrite evaluation datasets, or change the
  frozen detector.

No demo-input generator was added on Day 23. The inference API and zero-alert
real-pipeline feed are sufficient for integration, while postponing a generator
reduces the chance that demonstration rows are confused with evaluation
evidence. A future dashboard-specific generator may be added under this policy
when the UI actually needs it.

