# Conditional Local-History Experiment

This second-stage offline experiment tests a direction-changing historical
predictor without altering the FlowSense dashboard or production predictor.

## Method

- Uses 30 consecutive observed points and no future information in its inputs.
- Rotates ordered positions and velocities into the vehicle's current heading
  coordinate system.
- Predicts a multi-horizon, two-dimensional residual from constant velocity.
- Evaluates 5, 10, 15, and 30-frame horizons with leave-one-video-out folds.
- Bounds every residual relative to the CV displacement.
- Compares applying the correction everywhere with applying it only when the
  causal history shows at least 8 degrees of heading change or a 30% speed
  change. Stable tracks keep the unchanged CV prediction in the gated model.

Run:

```bash
python3 -m research.evaluate_conditional_local_history
```

Results are written to `research/conditional_local_history.csv` and are split
into all, stable, dynamic, turning, and speed-change samples.

## Preliminary matched LOVO result

Across the four held-out videos, the ungated local-history model improved ADE
over CV in every video: 9.65%, 13.77%, 17.76%, and 15.30%. Its macro
improvement was 14.12% overall and 11.88% on turning samples.

The gated version improved macro ADE by 3.56% overall because it deliberately
leaves stable samples unchanged. On the samples where it activates, it is
identical to the ungated model and improves macro ADE by 11.29% for all dynamic
samples and 11.88% for turning samples.

This result supports local-coordinate residual prediction, but it does not yet
show that hard gating is the best deployment policy: the ungated model also
improved stable samples and achieved the stronger overall result. Gating must
therefore remain an evaluated design choice rather than an assumed benefit.

The current thresholds and image dimensions are fixed research settings. They
must be tested through training-only sensitivity analysis before any production
integration or paper-level claim.
