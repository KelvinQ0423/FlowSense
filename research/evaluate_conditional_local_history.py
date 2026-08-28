"""Evaluate gated local-coordinate historical residual prediction.

The model uses only observations available at the prediction frame.  A
multi-output Ridge model predicts a bounded two-dimensional residual from the
constant-velocity (CV) forecast.  Its correction is applied only when the
causal history indicates turning or a substantial speed change.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.compare_historical_features import (
    DEFAULT_INPUTS,
    CsvPoint,
    fit_ridge,
    load_tracks,
    percentile,
    velocities,
)


@dataclass(frozen=True, slots=True)
class LocalHistoryDataset:
    features: np.ndarray
    residual_targets: np.ndarray
    actual: np.ndarray
    cv: np.ndarray
    headings: np.ndarray
    turn_degrees: np.ndarray
    speed_change_ratios: np.ndarray
    dynamic: np.ndarray
    horizon_seconds: np.ndarray


def rotate_to_local(x: float, y: float, heading: float) -> tuple[float, float]:
    """Rotate an image-space vector so current heading becomes local +x."""
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return cosine * x + sine * y, -sine * x + cosine * y


def rotate_to_image(x: float, y: float, heading: float) -> tuple[float, float]:
    """Undo :func:`rotate_to_local`."""
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return cosine * x - sine * y, sine * x + cosine * y


def angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(second - first), math.cos(second - first))


def mean_velocity(values: list[tuple[float, float, float]]) -> tuple[float, float]:
    return (
        float(np.mean([value[1] for value in values])),
        float(np.mean([value[2] for value in values])),
    )


def causal_motion_state(
    history: list[CsvPoint],
    *,
    state_window: int,
    turn_threshold_degrees: float,
    speed_change_threshold: float,
    minimum_speed: float,
) -> tuple[float, float, bool, float, float]:
    """Return heading, turn magnitude, dynamic flag and recent mean velocity."""
    values = velocities(history)
    if len(values) < state_window * 2:
        raise ValueError("history is too short for two motion-state windows")
    earlier_x, earlier_y = mean_velocity(values[-2 * state_window : -state_window])
    recent_x, recent_y = mean_velocity(values[-state_window:])
    earlier_speed = math.hypot(earlier_x, earlier_y)
    recent_speed = math.hypot(recent_x, recent_y)
    heading = math.atan2(recent_y, recent_x) if recent_speed > 1e-9 else 0.0
    earlier_heading = (
        math.atan2(earlier_y, earlier_x) if earlier_speed > minimum_speed else heading
    )
    turn_degrees = abs(math.degrees(angle_difference(earlier_heading, heading)))
    speed_change = abs(recent_speed - earlier_speed) / max(
        earlier_speed, recent_speed, minimum_speed
    )
    dynamic = recent_speed >= minimum_speed and (
        turn_degrees >= turn_threshold_degrees
        or speed_change >= speed_change_threshold
    )
    return heading, turn_degrees, dynamic, recent_x, recent_y


def build_local_dataset(
    path: Path,
    *,
    history_points: int,
    horizons: tuple[int, ...],
    frame_width: float,
    frame_height: float,
    state_window: int,
    turn_threshold_degrees: float,
    speed_change_threshold: float,
    minimum_speed: float,
) -> LocalHistoryDataset:
    if history_points < 2 * state_window + 1:
        raise ValueError("history must contain two complete state windows")
    diagonal = math.hypot(frame_width, frame_height)
    feature_rows: list[tuple[float, ...]] = []
    target_rows: list[tuple[float, ...]] = []
    actual_rows: list[tuple[float, ...]] = []
    cv_rows: list[tuple[float, ...]] = []
    headings: list[float] = []
    turns: list[float] = []
    speed_changes: list[float] = []
    dynamics: list[bool] = []
    seconds_rows: list[tuple[float, ...]] = []

    for frames in load_tracks(path).values():
        for current_frame in sorted(frames):
            history_ids = range(current_frame - history_points + 1, current_frame + 1)
            future_ids = tuple(current_frame + horizon for horizon in horizons)
            if any(frame not in frames for frame in (*history_ids, *future_ids)):
                continue
            history = [frames[frame] for frame in history_ids]
            state = causal_motion_state(
                history,
                state_window=state_window,
                turn_threshold_degrees=turn_threshold_degrees,
                speed_change_threshold=speed_change_threshold,
                minimum_speed=minimum_speed,
            )
            heading, turn_degrees, dynamic, velocity_x, velocity_y = state
            velocity_values = velocities(history)
            earlier_x, earlier_y = mean_velocity(
                velocity_values[-2 * state_window : -state_window]
            )
            recent_speed = math.hypot(velocity_x, velocity_y)
            earlier_speed = math.hypot(earlier_x, earlier_y)
            speed_change = abs(recent_speed - earlier_speed) / max(
                recent_speed, earlier_speed, minimum_speed
            )
            current = history[-1]
            ordered: list[float] = []
            for point in history[:-1]:
                local_x, local_y = rotate_to_local(
                    point.x - current.x, point.y - current.y, heading
                )
                ordered.extend((local_x / diagonal, local_y / diagonal))
            for _, x_velocity, y_velocity in velocity_values:
                local_x, local_y = rotate_to_local(x_velocity, y_velocity, heading)
                ordered.extend((local_x / diagonal, local_y / diagonal))
            ordered.extend(
                (
                    recent_speed / diagonal,
                    math.radians(turn_degrees),
                    speed_change,
                )
            )

            residuals: list[float] = []
            actual_displacements: list[float] = []
            cv_displacements: list[float] = []
            horizon_times: list[float] = []
            for future_id in future_ids:
                future = frames[future_id]
                seconds = future.timestamp - current.timestamp
                actual_x = future.x - current.x
                actual_y = future.y - current.y
                cv_x = velocity_x * seconds
                cv_y = velocity_y * seconds
                residual_x, residual_y = rotate_to_local(
                    actual_x - cv_x, actual_y - cv_y, heading
                )
                residuals.extend((residual_x / diagonal, residual_y / diagonal))
                actual_displacements.extend((actual_x, actual_y))
                cv_displacements.extend((cv_x, cv_y))
                horizon_times.append(seconds)

            feature_rows.append(tuple(ordered))
            target_rows.append(tuple(residuals))
            actual_rows.append(tuple(actual_displacements))
            cv_rows.append(tuple(cv_displacements))
            headings.append(heading)
            turns.append(turn_degrees)
            speed_changes.append(speed_change)
            dynamics.append(dynamic)
            seconds_rows.append(tuple(horizon_times))

    if not feature_rows:
        raise ValueError(f"No eligible samples in {path}")
    return LocalHistoryDataset(
        features=np.asarray(feature_rows, dtype=float),
        residual_targets=np.asarray(target_rows, dtype=float),
        actual=np.asarray(actual_rows, dtype=float),
        cv=np.asarray(cv_rows, dtype=float),
        headings=np.asarray(headings, dtype=float),
        turn_degrees=np.asarray(turns, dtype=float),
        speed_change_ratios=np.asarray(speed_changes, dtype=float),
        dynamic=np.asarray(dynamics, dtype=bool),
        horizon_seconds=np.asarray(seconds_rows, dtype=float),
    )


def apply_local_residuals(
    dataset: LocalHistoryDataset,
    predicted_local: np.ndarray,
    *,
    diagonal: float,
    correction_strength: float,
    residual_ratio_limit: float,
    minimum_residual_limit_px: float,
    gated: bool,
) -> np.ndarray:
    cv = dataset.cv.reshape(len(dataset.features), -1, 2)
    local = predicted_local.reshape(len(dataset.features), -1, 2) * diagonal
    result = cv.copy()
    for sample_index in range(len(dataset.features)):
        if gated and not dataset.dynamic[sample_index]:
            continue
        heading = float(dataset.headings[sample_index])
        for horizon_index in range(local.shape[1]):
            residual_x, residual_y = rotate_to_image(
                float(local[sample_index, horizon_index, 0]),
                float(local[sample_index, horizon_index, 1]),
                heading,
            )
            residual = np.asarray((residual_x, residual_y), dtype=float)
            limit = max(
                minimum_residual_limit_px,
                residual_ratio_limit * float(np.linalg.norm(cv[sample_index, horizon_index])),
            )
            length = float(np.linalg.norm(residual))
            if length > limit:
                residual *= limit / length
            result[sample_index, horizon_index] += correction_strength * residual
    return result


def category_masks(dataset: LocalHistoryDataset) -> dict[str, np.ndarray]:
    turning = dataset.dynamic & (dataset.turn_degrees >= 8.0)
    speed_change = dataset.dynamic & ~turning
    return {
        "all": np.ones(len(dataset.features), dtype=bool),
        "stable": ~dataset.dynamic,
        "turning": turning,
        "speed_change": speed_change,
        "dynamic": dataset.dynamic,
    }


def metric_rows(
    errors: np.ndarray,
    cv_errors: np.ndarray,
    dataset: LocalHistoryDataset,
    *,
    split: str,
    model: str,
    horizons: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, mask in category_masks(dataset).items():
        if not np.any(mask):
            continue
        for index, horizon in enumerate(horizons):
            values = errors[mask, index]
            baseline = cv_errors[mask, index]
            baseline_mean = float(baseline.mean())
            mean_error = float(values.mean())
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "category": category,
                    "scope": "horizon",
                    "horizon_frames": horizon,
                    "samples": len(values),
                    "mean_error_px": mean_error,
                    "median_error_px": float(np.median(values)),
                    "p90_error_px": percentile(values, 0.90),
                    "improvement_vs_cv_pct": 100.0 * (baseline_mean - mean_error) / baseline_mean
                    if baseline_mean > 0
                    else 0.0,
                }
            )
        ade = errors[mask].mean(axis=1)
        cv_ade = cv_errors[mask].mean(axis=1)
        baseline_mean = float(cv_ade.mean())
        mean_error = float(ade.mean())
        rows.append(
            {
                "split": split,
                "model": model,
                "category": category,
                "scope": "ADE",
                "horizon_frames": horizons[-1],
                "samples": len(ade),
                "mean_error_px": mean_error,
                "median_error_px": float(np.median(ade)),
                "p90_error_px": percentile(ade, 0.90),
                "improvement_vs_cv_pct": 100.0 * (baseline_mean - mean_error) / baseline_mean
                if baseline_mean > 0
                else 0.0,
            }
        )
    return rows


def evaluate(
    datasets: dict[str, LocalHistoryDataset],
    *,
    horizons: tuple[int, ...],
    diagonal: float,
    alpha: float,
    correction_strength: float,
    residual_ratio_limit: float,
    minimum_residual_limit_px: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for held_out, test in datasets.items():
        training = [dataset for name, dataset in datasets.items() if name != held_out]
        train_x = np.concatenate([dataset.features for dataset in training])
        train_y = np.concatenate([dataset.residual_targets for dataset in training])
        means, scales, coefficients, intercepts = fit_ridge(train_x, train_y, alpha=alpha)
        predicted = (test.features - means) / scales @ coefficients.T + intercepts
        actual = test.actual.reshape(len(test.features), -1, 2)
        cv = test.cv.reshape(len(test.features), -1, 2)
        cv_errors = np.linalg.norm(cv - actual, axis=2)
        predictions = {
            "constant_velocity": cv,
            "local_history_ungated": apply_local_residuals(
                test,
                predicted,
                diagonal=diagonal,
                correction_strength=correction_strength,
                residual_ratio_limit=residual_ratio_limit,
                minimum_residual_limit_px=minimum_residual_limit_px,
                gated=False,
            ),
            "local_history_gated": apply_local_residuals(
                test,
                predicted,
                diagonal=diagonal,
                correction_strength=correction_strength,
                residual_ratio_limit=residual_ratio_limit,
                minimum_residual_limit_px=minimum_residual_limit_px,
                gated=True,
            ),
        }
        for model, prediction in predictions.items():
            errors = np.linalg.norm(prediction - actual, axis=2)
            rows.extend(
                metric_rows(
                    errors,
                    cv_errors,
                    test,
                    split=held_out,
                    model=model,
                    horizons=horizons,
                )
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "split", "model", "category", "scope", "horizon_frames", "samples",
        "mean_error_px", "median_error_px", "p90_error_px", "improvement_vs_cv_pct",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--history-points", type=int, default=30)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 10, 15, 30))
    parser.add_argument("--frame-width", type=float, default=1920.0)
    parser.add_argument("--frame-height", type=float, default=1080.0)
    parser.add_argument("--state-window", type=int, default=5)
    parser.add_argument("--turn-threshold-degrees", type=float, default=8.0)
    parser.add_argument("--speed-change-threshold", type=float, default=0.30)
    parser.add_argument("--minimum-speed", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--correction-strength", type=float, default=0.35)
    parser.add_argument("--residual-ratio-limit", type=float, default=0.50)
    parser.add_argument("--minimum-residual-limit-px", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("research/conditional_local_history.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.inputs:
        raise SystemExit("No canonical trajectory CSV files were found")
    horizons = tuple(sorted(set(args.horizons)))
    datasets = {
        path.stem: build_local_dataset(
            path,
            history_points=args.history_points,
            horizons=horizons,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            state_window=args.state_window,
            turn_threshold_degrees=args.turn_threshold_degrees,
            speed_change_threshold=args.speed_change_threshold,
            minimum_speed=args.minimum_speed,
        )
        for path in args.inputs
    }
    rows = evaluate(
        datasets,
        horizons=horizons,
        diagonal=math.hypot(args.frame_width, args.frame_height),
        alpha=args.alpha,
        correction_strength=args.correction_strength,
        residual_ratio_limit=args.residual_ratio_limit,
        minimum_residual_limit_px=args.minimum_residual_limit_px,
    )
    write_rows(args.output, rows)
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset.features)} samples, {dataset.dynamic.mean():.1%} dynamic")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
