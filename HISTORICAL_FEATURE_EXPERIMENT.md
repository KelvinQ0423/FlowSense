# Controlled Historical-Motion Feature Experiment

This experiment answers whether additional historical motion information helps
short-term FlowSense prediction while changing only the Ridge input features.

The current bundled FlowSense Ridge corrector already contains acceleration
summary features. Therefore acceleration is evaluated as an ablation rather
than described as a wholly new capability.

## Compared methods

- `constant_velocity`: zero learned residual.
- `constant_acceleration`: analytic kinematic comparison.
- `core_ridge`: the existing context and motion summaries without acceleration.
- `acceleration_ridge`: core plus explicit acceleration; this matches the
  current production feature design most closely.
- `history_ridge`: core plus ordered recent positions and velocities.
- `full_ridge`: the current acceleration summaries plus ordered history.

All four Ridge variants predict the same bounded scale target used by the
current FlowSense design. The learned scale is clipped to 0.25–1.75 and only
35% of its correction is applied. The variants use identical source frames,
targets, leave-one-video-out folds, normalization, and Ridge implementation.

Acceleration is the least-squares slope of the five most recent velocity
observations. This is less sensitive to one-frame detector jitter than a single
finite difference while remaining an explicit acceleration measurement.

## Run

```bash
python3 -m research.compare_historical_features
```

The default experiment uses 15 feature-history points, requires every sample to
have at least 30 consecutive history points, and evaluates 5, 10, 15, 30, and
45-frame horizons. Results are written to:

```text
research/historical_feature_ablation.csv
```

To compare history lengths on the same eligible source frames, keep
`--eligibility-history-points 30` unchanged and run separate commands with
`--history-points 5`, `10`, `15`, and `30`.

The core velocity and acceleration summaries always use the same five most
recent velocity segments from the 30-point eligibility window. Changing
`--history-points` therefore changes only the ordered-history input.

This script is offline research code. It does not replace the bundled model,
change the dashboard, or alter the production fallback behavior.

## Preliminary result

The first matched-sample LOVO run used 30-point eligibility and evaluated
5–45-frame horizons. Macro ADE was:

| Model | Macro ADE (px) | Improvement vs. CV |
| --- | ---: | ---: |
| Constant velocity | 21.8765 | 0.00% |
| Core Ridge | 17.9571 | 18.22% |
| Acceleration Ridge | 17.9536 | 18.24% |
| History Ridge, 15 points | 19.0930 | 8.24% |
| Full Ridge, 15 points | 19.0996 | 8.21% |

Explicit acceleration made almost no difference to the learned Ridge result,
while analytic constant acceleration was highly unstable. Ordered raw history
still beat constant velocity overall, but did not beat the existing summary
feature design. On the identical eligible samples, Full Ridge improvement was
14.23%, 8.89%, 8.21%, and 12.05% for 5, 10, 15, and 30 history points.

This is a valid negative preliminary result, not evidence that historical
motion can never help. It shows that simply appending correlated raw trajectory
coordinates is insufficient with the current data and camera variation.
