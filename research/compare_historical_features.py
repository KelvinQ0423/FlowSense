"""Controlled Ridge ablation for historical vehicle-motion features.

Every learned model uses the same source frames, targets, Ridge implementation,
and leave-one-video-out folds. Only the input feature group changes.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = sorted(Path("tracking_data").glob("member2_canonical_tracks*.csv"))
MODEL_GROUPS = ("core_ridge", "acceleration_ridge", "history_ridge", "full_ridge")
CLASS_CATEGORIES = ("bus", "car", "motorcycle", "person", "truck")
SCALE_MIN = 0.25
SCALE_MAX = 1.75
CORRECTION_STRENGTH = 0.35


@dataclass(frozen=True, slots=True)
class CsvPoint:
    frame_id: int
    timestamp: float
    x: float
    y: float
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class DatasetArrays:
    core: np.ndarray
    acceleration: np.ndarray
    history: np.ndarray
    scale_targets: np.ndarray
    actual_displacements: np.ndarray
    cv_displacements: np.ndarray
    ca_displacements: np.ndarray
    horizon_seconds: np.ndarray

    def features(self, model: str) -> np.ndarray:
        if model == "core_ridge":
            return self.core
        if model == "acceleration_ridge":
            return np.concatenate((self.core, self.acceleration), axis=1)
        if model == "history_ridge":
            return np.concatenate((self.core, self.history), axis=1)
        if model == "full_ridge":
            return np.concatenate(
                (self.core, self.acceleration, self.history), axis=1
            )
        raise ValueError(f"Unknown model group: {model}")


def load_tracks(path: Path) -> dict[int, dict[int, CsvPoint]]:
    tracks: dict[int, dict[int, CsvPoint]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point = CsvPoint(
                frame_id=int(row["frame"]),
                timestamp=float(row["time_seconds"]),
                x=float(row["center_x"]),
                y=float(row["center_y"]),
                class_name=str(row["class_name"]),
                confidence=float(row["confidence"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
            )
            tracks[int(row["track_id"])][point.frame_id] = point
    return dict(tracks)


def velocities(points: list[CsvPoint]) -> list[tuple[float, float, float]]:
    values: list[tuple[float, float, float]] = []
    for previous, current in zip(points, points[1:]):
        elapsed = current.timestamp - previous.timestamp
        if elapsed <= 0:
            continue
        values.append(
            (
                (previous.timestamp + current.timestamp) / 2.0,
                (current.x - previous.x) / elapsed,
                (current.y - previous.y) / elapsed,
            )
        )
    return values


def estimate_acceleration(
    velocity_values: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Estimate acceleration as the least-squares slope of recent velocity."""
    if len(velocity_values) < 2:
        return 0.0, 0.0
    times = np.asarray([value[0] for value in velocity_values], dtype=float)
    centered_time = times - times.mean()
    denominator = float(centered_time @ centered_time)
    if denominator <= 1e-12:
        return 0.0, 0.0
    velocity_x = np.asarray([value[1] for value in velocity_values], dtype=float)
    velocity_y = np.asarray([value[2] for value in velocity_values], dtype=float)
    return (
        float(centered_time @ (velocity_x - velocity_x.mean()) / denominator),
        float(centered_time @ (velocity_y - velocity_y.mean()) / denominator),
    )


def build_dataset(
    path: Path,
    *,
    history_points: int,
    eligibility_history_points: int,
    horizons: tuple[int, ...],
    frame_width: float,
    frame_height: float,
    velocity_window: int = 5,
) -> DatasetArrays:
    if history_points < 2:
        raise ValueError("history_points must be at least two")
    if eligibility_history_points < history_points:
        raise ValueError("eligibility history cannot be shorter than feature history")
    diagonal = math.hypot(frame_width, frame_height)
    core_rows: list[tuple[float, ...]] = []
    acceleration_rows: list[tuple[float, ...]] = []
    history_rows: list[tuple[float, ...]] = []
    targets: list[tuple[float, ...]] = []
    actual_rows: list[tuple[float, ...]] = []
    cv_rows: list[tuple[float, ...]] = []
    ca_rows: list[tuple[float, ...]] = []
    seconds_rows: list[tuple[float, ...]] = []

    for frames in load_tracks(path).values():
        for current_frame in sorted(frames):
            eligibility_ids = range(
                current_frame - eligibility_history_points + 1,
                current_frame + 1,
            )
            future_ids = tuple(current_frame + horizon for horizon in horizons)
            if any(frame not in frames for frame in (*eligibility_ids, *future_ids)):
                continue
            eligibility_history = [frames[frame] for frame in eligibility_ids]
            feature_ids = range(current_frame - history_points + 1, current_frame + 1)
            history = [frames[frame] for frame in feature_ids]
            core_velocity_values = velocities(eligibility_history)[-velocity_window:]
            if len(core_velocity_values) < velocity_window:
                continue
            recent_velocities = core_velocity_values
            mean_velocity_x = float(np.mean([value[1] for value in recent_velocities]))
            mean_velocity_y = float(np.mean([value[2] for value in recent_velocities]))
            acceleration_x, acceleration_y = estimate_acceleration(recent_velocities)
            current = history[-1]
            speed = math.hypot(mean_velocity_x, mean_velocity_y)
            direction = math.atan2(mean_velocity_y, mean_velocity_x) if speed > 0 else 0.0
            segment_speeds = [math.hypot(value[1], value[2]) for value in recent_velocities]
            average_segment_speed = float(np.mean(segment_speeds))
            consistency = speed / average_segment_speed if average_segment_speed > 0 else 0.0
            std_x = float(np.std([value[1] for value in recent_velocities]))
            std_y = float(np.std([value[2] for value in recent_velocities]))
            box_width = current.x2 - current.x1
            box_height = current.y2 - current.y1
            # Context and non-acceleration motion summaries are shared by all
            # learned groups. Class one-hot values match the production model.
            class_features = tuple(
                1.0 if current.class_name == category else 0.0
                for category in CLASS_CATEGORIES
            )
            core_rows.append(
                class_features
                + (
                    current.confidence,
                    current.x / frame_width,
                    current.y / frame_height,
                    box_width / frame_width,
                    box_height / frame_height,
                    box_width / max(box_height, 1e-6),
                    mean_velocity_x / diagonal,
                    mean_velocity_y / diagonal,
                    speed / diagonal,
                    std_x / diagonal,
                    std_y / diagonal,
                    math.sin(direction),
                    math.cos(direction),
                    consistency,
                    float(min(eligibility_history_points, 30)),
                )
            )
            acceleration_rows.append(
                (
                    acceleration_x / diagonal,
                    acceleration_y / diagonal,
                    math.hypot(acceleration_x, acceleration_y) / diagonal,
                )
            )
            ordered_history: list[float] = []
            for point in history[:-1]:
                ordered_history.extend(
                    ((point.x - current.x) / diagonal, (point.y - current.y) / diagonal)
                )
            for _, velocity_x, velocity_y in velocities(history):
                ordered_history.extend((velocity_x / diagonal, velocity_y / diagonal))
            history_rows.append(tuple(ordered_history))

            scale_target: list[float] = []
            actual_displacement: list[float] = []
            cv_displacement: list[float] = []
            ca_displacement: list[float] = []
            horizon_times: list[float] = []
            for future_id in future_ids:
                future = frames[future_id]
                seconds = future.timestamp - current.timestamp
                actual_dx = future.x - current.x
                actual_dy = future.y - current.y
                cv_dx = mean_velocity_x * seconds
                cv_dy = mean_velocity_y * seconds
                ca_dx = cv_dx + 0.5 * acceleration_x * seconds * seconds
                ca_dy = cv_dy + 0.5 * acceleration_y * seconds * seconds
                baseline_squared = cv_dx * cv_dx + cv_dy * cv_dy
                optimal_scale = (
                    (actual_dx * cv_dx + actual_dy * cv_dy) / baseline_squared
                    if baseline_squared > 1e-12
                    else 1.0
                )
                scale_target.append(min(SCALE_MAX, max(SCALE_MIN, optimal_scale)))
                actual_displacement.extend((actual_dx, actual_dy))
                cv_displacement.extend((cv_dx, cv_dy))
                ca_displacement.extend((ca_dx, ca_dy))
                horizon_times.append(seconds)
            targets.append(tuple(scale_target))
            actual_rows.append(tuple(actual_displacement))
            cv_rows.append(tuple(cv_displacement))
            ca_rows.append(tuple(ca_displacement))
            seconds_rows.append(tuple(horizon_times))

    if not core_rows:
        raise ValueError(f"No eligible samples in {path}")
    return DatasetArrays(
        core=np.asarray(core_rows, dtype=float),
        acceleration=np.asarray(acceleration_rows, dtype=float),
        history=np.asarray(history_rows, dtype=float),
        scale_targets=np.asarray(targets, dtype=float),
        actual_displacements=np.asarray(actual_rows, dtype=float),
        cv_displacements=np.asarray(cv_rows, dtype=float),
        ca_displacements=np.asarray(ca_rows, dtype=float),
        horizon_seconds=np.asarray(seconds_rows, dtype=float),
    )


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    intercepts = targets.mean(axis=0)
    centered_targets = targets - intercepts
    regularized = standardized.T @ standardized + alpha * np.eye(features.shape[1])
    coefficients = np.linalg.solve(regularized, standardized.T @ centered_targets).T
    return means, scales, coefficients, intercepts


def predict_ridge(
    features: np.ndarray,
    *,
    means: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    return ((features - means) / scales) @ coefficients.T + intercepts


def percentile(values: np.ndarray, fraction: float) -> float:
    return float(np.quantile(values, fraction))


def summarize_errors(
    errors: np.ndarray,
    cv_errors: np.ndarray,
    horizon_seconds: np.ndarray,
    *,
    split: str,
    model: str,
    horizons: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, horizon in enumerate(horizons):
        values = errors[:, index]
        baseline = cv_errors[:, index]
        mean_error = float(values.mean())
        cv_mean = float(baseline.mean())
        rows.append(
            {
                "split": split,
                "model": model,
                "scope": "horizon",
                "horizon_frames": horizon,
                "horizon_seconds": float(np.median(horizon_seconds[:, index])),
                "samples": len(values),
                "mean_error_px": mean_error,
                "rmse_px": float(np.sqrt(np.mean(values**2))),
                "median_error_px": float(np.median(values)),
                "p90_error_px": percentile(values, 0.90),
                "improvement_vs_cv_pct": (
                    100.0 * (cv_mean - mean_error) / cv_mean if cv_mean > 0 else 0.0
                ),
            }
        )
    ade = errors.mean(axis=1)
    cv_ade = cv_errors.mean(axis=1)
    mean_ade = float(ade.mean())
    mean_cv_ade = float(cv_ade.mean())
    rows.append(
        {
            "split": split,
            "model": model,
            "scope": "ADE",
            "horizon_frames": horizons[-1],
            "horizon_seconds": float(np.median(horizon_seconds[:, -1])),
            "samples": len(ade),
            "mean_error_px": mean_ade,
            "rmse_px": float(np.sqrt(np.mean(ade**2))),
            "median_error_px": float(np.median(ade)),
            "p90_error_px": percentile(ade, 0.90),
            "improvement_vs_cv_pct": (
                100.0 * (mean_cv_ade - mean_ade) / mean_cv_ade
                if mean_cv_ade > 0
                else 0.0
            ),
        }
    )
    return rows


def evaluate_lovo(
    datasets: dict[str, DatasetArrays],
    *,
    horizons: tuple[int, ...],
    diagonal: float,
    alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for held_out, test in datasets.items():
        actual = test.actual_displacements.reshape(len(test.core), -1, 2)
        cv = test.cv_displacements.reshape(len(test.core), -1, 2)
        ca = test.ca_displacements.reshape(len(test.core), -1, 2)
        cv_errors = np.linalg.norm(cv - actual, axis=2)
        ca_errors = np.linalg.norm(ca - actual, axis=2)
        rows.extend(
            summarize_errors(
                cv_errors,
                cv_errors,
                test.horizon_seconds,
                split=held_out,
                model="constant_velocity",
                horizons=horizons,
            )
        )
        rows.extend(
            summarize_errors(
                ca_errors,
                cv_errors,
                test.horizon_seconds,
                split=held_out,
                model="constant_acceleration",
                horizons=horizons,
            )
        )
        training_sets = [value for name, value in datasets.items() if name != held_out]
        for model in MODEL_GROUPS:
            train_x = np.concatenate([dataset.features(model) for dataset in training_sets])
            train_y = np.concatenate([dataset.scale_targets for dataset in training_sets])
            means, scales, coefficients, intercepts = fit_ridge(
                train_x,
                train_y,
                alpha=alpha,
            )
            learned_scale = predict_ridge(
                test.features(model),
                means=means,
                scales=scales,
                coefficients=coefficients,
                intercepts=intercepts,
            )
            learned_scale = np.clip(learned_scale, SCALE_MIN, SCALE_MAX)
            applied_scale = 1.0 + CORRECTION_STRENGTH * (learned_scale - 1.0)
            learned_displacement = cv * applied_scale[:, :, np.newaxis]
            learned_errors = np.linalg.norm(learned_displacement - actual, axis=2)
            rows.extend(
                summarize_errors(
                    learned_errors,
                    cv_errors,
                    test.horizon_seconds,
                    split=held_out,
                    model=model,
                    horizons=horizons,
                )
            )
    return rows


def add_macro_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["scope"]), int(row["horizon_frames"]))].append(row)
    macro_rows: list[dict[str, object]] = []
    numeric = (
        "horizon_seconds",
        "mean_error_px",
        "rmse_px",
        "median_error_px",
        "p90_error_px",
        "improvement_vs_cv_pct",
    )
    for (model, scope, horizon), values in grouped.items():
        macro = {
            "split": "LOVO_macro",
            "model": model,
            "scope": scope,
            "horizon_frames": horizon,
            "samples": sum(int(value["samples"]) for value in values),
        }
        for name in numeric:
            macro[name] = float(np.mean([float(value[name]) for value in values]))
        macro_rows.append(macro)
    return rows + macro_rows


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "split",
        "model",
        "scope",
        "horizon_frames",
        "horizon_seconds",
        "samples",
        "mean_error_px",
        "rmse_px",
        "median_error_px",
        "p90_error_px",
        "improvement_vs_cv_pct",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--history-points", type=int, default=15)
    parser.add_argument("--eligibility-history-points", type=int, default=30)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 10, 15, 30, 45))
    parser.add_argument("--velocity-window", type=int, default=5)
    parser.add_argument("--frame-width", type=float, default=1920.0)
    parser.add_argument("--frame-height", type=float, default=1080.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/historical_feature_ablation.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.inputs:
        raise SystemExit("No canonical trajectory CSV files were found")
    horizons = tuple(sorted(set(args.horizons)))
    datasets = {
        path.stem: build_dataset(
            path,
            history_points=args.history_points,
            eligibility_history_points=args.eligibility_history_points,
            horizons=horizons,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            velocity_window=args.velocity_window,
        )
        for path in args.inputs
    }
    diagonal = math.hypot(args.frame_width, args.frame_height)
    rows = add_macro_rows(
        evaluate_lovo(
            datasets,
            horizons=horizons,
            diagonal=diagonal,
            alpha=args.alpha,
        )
    )
    write_results(args.output, rows)
    print("Samples by video:")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset.core)}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
