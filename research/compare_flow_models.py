"""Compare the current FlowSense Ridge predictor with historical-motion FlowSense.

This experiment evaluates exactly two learned predictors:

1. current_ridge:

   The existing FlowSense core Ridge predictor. It predicts a scale correction

   to the constant-velocity forecast.

2. historical_motion:

   The new local-coordinate historical-motion predictor. It uses ordered

   trajectory history and predicts a 2-D residual correction.

Both models:

- use the same canonical tracking files;

- use the same eligible prediction samples;

- use the same prediction horizons;

- use leave-one-video-out evaluation;

- are trained only on the training videos;

- are evaluated on the identical held-out samples.

The primary comparison is ADE/FDE-style displacement error in pixels.

"""

from __future__ import annotations

import argparse

import csv

import math

from pathlib import Path

import numpy as np

from research.compare_historical_features import (
    DatasetArrays,
    build_dataset,
    fit_ridge,
    predict_ridge,
)

from research.evaluate_conditional_local_history import (
    LocalHistoryDataset,
    apply_local_residuals,
    build_local_dataset,
)

DEFAULT_INPUTS = sorted(
    Path("tracking_data").glob("member2_canonical_tracks*.csv")
)

DEFAULT_HORIZONS = (5, 10, 15, 30)

# Existing FlowSense Ridge settings.

RIDGE_ALPHA = 10.0

SCALE_MIN = 0.25

SCALE_MAX = 1.75

RIDGE_CORRECTION_STRENGTH = 0.35

# Historical-motion settings used by the existing research implementation.

HISTORY_POINTS = 30

STATE_WINDOW = 5

TURN_THRESHOLD_DEGREES = 8.0

SPEED_CHANGE_THRESHOLD = 0.30

MINIMUM_SPEED = 10.0

HISTORY_CORRECTION_STRENGTH = 0.35

RESIDUAL_RATIO_LIMIT = 0.50

MINIMUM_RESIDUAL_LIMIT_PX = 5.0

FRAME_WIDTH = 1920.0

FRAME_HEIGHT = 1080.0


def prediction_errors(

    prediction: np.ndarray,

    actual: np.ndarray,

) -> np.ndarray:

    """Return Euclidean displacement error for every sample/horizon."""

    return np.linalg.norm(prediction - actual, axis=2)


def calculate_metrics(

    errors: np.ndarray,

) -> dict[str, float]:

    """Calculate summary trajectory-prediction metrics."""

    ade_per_sample = errors.mean(axis=1)

    final_errors = errors[:, -1]

    return {

        "ADE_px": float(ade_per_sample.mean()),

        "FDE_px": float(final_errors.mean()),

        "RMSE_px": float(np.sqrt(np.mean(errors**2))),

        "median_ADE_px": float(np.median(ade_per_sample)),

        "P90_ADE_px": float(np.quantile(ade_per_sample, 0.90)),

    }


def percentage_improvement(

    baseline: float,

    new_value: float,

) -> float:

    """Positive value means the new model has lower error."""

    if baseline <= 0:

        return 0.0

    return 100.0 * (baseline - new_value) / baseline


def evaluate_fold(

    datasets: dict[str, tuple[DatasetArrays, LocalHistoryDataset]],

    held_out: str,

    *,

    horizons: tuple[int, ...],

    diagonal: float,

) -> list[dict[str, object]]:

    """Train on all videos except held_out and evaluate both models."""

    train_pairs = [

        pair

        for name, pair in datasets.items()

        if name != held_out

    ]

    test_core, test_history = datasets[held_out]

    # ------------------------------------------------------------------

    # Verify that both models actually contain the same number of samples.

    # ------------------------------------------------------------------

    if len(test_core.core) != len(test_history.features):

        raise RuntimeError(

            f"Sample mismatch for {held_out}: "

            f"current Ridge has {len(test_core.core)} samples, "

            f"historical-motion has {len(test_history.features)} samples."

        )

    # ------------------------------------------------------------------

    # Train current FlowSense Ridge.

    # ------------------------------------------------------------------

    ridge_train_x = np.concatenate(

        [

            core_dataset.features("core_ridge")

            for core_dataset, _ in train_pairs

        ]

    )

    ridge_train_y = np.concatenate(

        [

            core_dataset.scale_targets

            for core_dataset, _ in train_pairs

        ]

    )

    ridge_means, ridge_scales, ridge_coefficients, ridge_intercepts = fit_ridge(

        ridge_train_x,

        ridge_train_y,

        alpha=RIDGE_ALPHA,

    )

    ridge_predicted_scale = predict_ridge(

        test_core.features("core_ridge"),

        means=ridge_means,

        scales=ridge_scales,

        coefficients=ridge_coefficients,

        intercepts=ridge_intercepts,

    )

    ridge_predicted_scale = np.clip(

        ridge_predicted_scale,

        SCALE_MIN,

        SCALE_MAX,

    )

    # ------------------------------------------------------------------

    # Train historical-motion model.

    # ------------------------------------------------------------------

    history_train_x = np.concatenate(

        [

            history_dataset.features

            for _, history_dataset in train_pairs

        ]

    )

    history_train_y = np.concatenate(

        [

            history_dataset.residual_targets

            for _, history_dataset in train_pairs

        ]

    )

    history_means, history_scales, history_coefficients, history_intercepts = (

        fit_ridge(

            history_train_x,

            history_train_y,

            alpha=50.0,

        )

    )

    history_predicted_residual = (

        (

            (test_history.features - history_means)

            / history_scales

        )

        @ history_coefficients.T

        + history_intercepts

    )

    # ------------------------------------------------------------------

    # Ground truth and common CV baseline.

    # ------------------------------------------------------------------

    actual = test_history.actual.reshape(

        len(test_history.features),

        -1,

        2,

    )

    cv = test_history.cv.reshape(

        len(test_history.features),

        -1,

        2,

    )

    # The existing Ridge model predicts a scalar scale applied to CV.

    ridge_prediction = (

        cv

        * (

            1.0

            + RIDGE_CORRECTION_STRENGTH

            * (ridge_predicted_scale - 1.0)

        )[:, :, np.newaxis]

    )

    # The historical-motion model predicts a 2-D residual.

    history_prediction = apply_local_residuals(

        test_history,

        history_predicted_residual,

        diagonal=diagonal,

        correction_strength=HISTORY_CORRECTION_STRENGTH,

        residual_ratio_limit=RESIDUAL_RATIO_LIMIT,

        minimum_residual_limit_px=MINIMUM_RESIDUAL_LIMIT_PX,

        gated=False,

    )

    # ------------------------------------------------------------------

    # Calculate errors.

    # ------------------------------------------------------------------

    ridge_errors = prediction_errors(

        ridge_prediction,

        actual,

    )

    history_errors = prediction_errors(

        history_prediction,

        actual,

    )

    cv_errors = prediction_errors(

        cv,

        actual,

    )

    rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------

    # Horizon-by-horizon results.

    # ------------------------------------------------------------------

    for horizon_index, horizon in enumerate(horizons):

        ridge_error = ridge_errors[:, horizon_index]

        history_error = history_errors[:, horizon_index]

        cv_error = cv_errors[:, horizon_index]

        ridge_mean = float(ridge_error.mean())

        history_mean = float(history_error.mean())

        cv_mean = float(cv_error.mean())

        rows.append(

            {

                "split": held_out,

                "model": "current_ridge",

                "scope": "horizon",

                "horizon_frames": horizon,

                "samples": len(ridge_error),

                "mean_error_px": ridge_mean,

                "median_error_px": float(np.median(ridge_error)),

                "p90_error_px": float(np.quantile(ridge_error, 0.90)),

                "rmse_px": float(np.sqrt(np.mean(ridge_error**2))),

                "improvement_vs_current_ridge_pct": 0.0,

                "improvement_vs_cv_pct": percentage_improvement(

                    cv_mean,

                    ridge_mean,

                ),

            }

        )

        rows.append(

            {

                "split": held_out,

                "model": "historical_motion",

                "scope": "horizon",

                "horizon_frames": horizon,

                "samples": len(history_error),

                "mean_error_px": history_mean,

                "median_error_px": float(np.median(history_error)),

                "p90_error_px": float(np.quantile(history_error, 0.90)),

                "rmse_px": float(np.sqrt(np.mean(history_error**2))),

                "improvement_vs_current_ridge_pct": percentage_improvement(

                    ridge_mean,

                    history_mean,

                ),

                "improvement_vs_cv_pct": percentage_improvement(

                    cv_mean,

                    history_mean,

                ),

            }

        )

    # ------------------------------------------------------------------

    # Overall ADE/FDE-style summary.

    # ------------------------------------------------------------------

    ridge_metrics = calculate_metrics(ridge_errors)

    history_metrics = calculate_metrics(history_errors)

    rows.append(

        {

            "split": held_out,

            "model": "current_ridge",

            "scope": "overall",

            "horizon_frames": horizons[-1],

            "samples": len(ridge_errors),

            "mean_error_px": ridge_metrics["ADE_px"],

            "median_error_px": ridge_metrics["median_ADE_px"],

            "p90_error_px": ridge_metrics["P90_ADE_px"],

            "rmse_px": ridge_metrics["RMSE_px"],

            "FDE_px": ridge_metrics["FDE_px"],

            "improvement_vs_current_ridge_pct": 0.0,

            "improvement_vs_cv_pct": percentage_improvement(

                calculate_metrics(cv_errors)["ADE_px"],

                ridge_metrics["ADE_px"],

            ),

        }

    )

    rows.append(

        {

            "split": held_out,

            "model": "historical_motion",

            "scope": "overall",

            "horizon_frames": horizons[-1],

            "samples": len(history_errors),

            "mean_error_px": history_metrics["ADE_px"],

            "median_error_px": history_metrics["median_ADE_px"],

            "p90_error_px": history_metrics["P90_ADE_px"],

            "rmse_px": history_metrics["RMSE_px"],

            "FDE_px": history_metrics["FDE_px"],

            "improvement_vs_current_ridge_pct": percentage_improvement(

                ridge_metrics["ADE_px"],

                history_metrics["ADE_px"],

            ),

            "improvement_vs_cv_pct": percentage_improvement(

                calculate_metrics(cv_errors)["ADE_px"],

                history_metrics["ADE_px"],

            ),

        }

    )

    return rows


def add_overall_rows(

    rows: list[dict[str, object]],

) -> list[dict[str, object]]:

    """Calculate pooled results across all held-out videos."""

    result = list(rows)

    for model in ("current_ridge", "historical_motion"):

        horizon_rows = [

            row

            for row in rows

            if row["model"] == model

            and row["scope"] == "horizon"

        ]

        for horizon in sorted(

            {

                int(row["horizon_frames"])

                for row in horizon_rows

            }

        ):

            matching = [

                row

                for row in horizon_rows

                if int(row["horizon_frames"]) == horizon

            ]

            total_samples = sum(

                int(row["samples"])

                for row in matching

            )

            if total_samples == 0:

                continue

            weighted_mean = sum(

                float(row["mean_error_px"]) * int(row["samples"])

                for row in matching

            ) / total_samples

            result.append(

                {

                    "split": "ALL_VIDEOS",

                    "model": model,

                    "scope": "horizon",

                    "horizon_frames": horizon,

                    "samples": total_samples,

                    "mean_error_px": weighted_mean,

                    "median_error_px": float(

                        np.mean(

                            [

                                float(row["median_error_px"])

                                for row in matching

                            ]

                        )

                    ),

                    "p90_error_px": float(

                        np.mean(

                            [

                                float(row["p90_error_px"])

                                for row in matching

                            ]

                        )

                    ),

                    "rmse_px": float(

                        np.mean(

                            [

                                float(row["rmse_px"])

                                for row in matching

                            ]

                        )

                    ),

                    "improvement_vs_current_ridge_pct": (

                        0.0

                        if model == "current_ridge"

                        else None

                    ),

                }

            )

    return result


def write_results(

    path: Path,

    rows: list[dict[str, object]],

) -> None:

    """Write results to CSV."""

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    fields = [

        "split",

        "model",

        "scope",

        "horizon_frames",

        "samples",

        "mean_error_px",

        "median_error_px",

        "p90_error_px",

        "rmse_px",

        "FDE_px",

        "improvement_vs_current_ridge_pct",

        "improvement_vs_cv_pct",

    ]

    with path.open(

        "w",

        newline="",

        encoding="utf-8",

    ) as handle:

        writer = csv.DictWriter(

            handle,

            fieldnames=fields,

            extrasaction="ignore",

        )

        writer.writeheader()

        for row in rows:

            cleaned = {}

            for field in fields:

                value = row.get(field, "")

                if isinstance(value, float):

                    value = round(value, 4)

                cleaned[field] = value

            writer.writerow(cleaned)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description=__doc__,

    )

    parser.add_argument(

        "inputs",

        nargs="*",

        type=Path,

        default=DEFAULT_INPUTS,

    )

    parser.add_argument(

        "--horizons",

        type=int,

        nargs="+",

        default=DEFAULT_HORIZONS,

    )

    parser.add_argument(

        "--frame-width",

        type=float,

        default=FRAME_WIDTH,

    )

    parser.add_argument(

        "--frame-height",

        type=float,

        default=FRAME_HEIGHT,

    )

    parser.add_argument(

        "--output",

        type=Path,

        default=Path(

            "research/flow_model_comparison.csv"

        ),

    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if not args.inputs:

        raise SystemExit(

            "No canonical trajectory CSV files were found."

        )

    horizons = tuple(

        sorted(

            set(args.horizons)

        )

    )

    if any(horizon <= 0 for horizon in horizons):

        raise SystemExit(

            "Prediction horizons must be positive."

        )

    diagonal = math.hypot(

        args.frame_width,

        args.frame_height,

    )

    datasets: dict[

        str,

        tuple[DatasetArrays, LocalHistoryDataset],

    ] = {}

    print("Building identical evaluation samples...")

    for path in args.inputs:

        print(f"  Loading {path}")

        # Current FlowSense:

        #

        # Use 30 points for eligibility so that its sample set matches

        # the historical-motion model, while retaining the existing

        # core feature construction.

        current_dataset = build_dataset(

            path,

            history_points=15,

            eligibility_history_points=HISTORY_POINTS,

            horizons=horizons,

            frame_width=args.frame_width,

            frame_height=args.frame_height,

            velocity_window=5,

        )

        # Historical-motion:

        #

        # Use the actual research model's 30-point ordered history.

        history_dataset = build_local_dataset(

            path,

            history_points=HISTORY_POINTS,

            horizons=horizons,

            frame_width=args.frame_width,

            frame_height=args.frame_height,

            state_window=STATE_WINDOW,

            turn_threshold_degrees=TURN_THRESHOLD_DEGREES,

            speed_change_threshold=SPEED_CHANGE_THRESHOLD,

            minimum_speed=MINIMUM_SPEED,

        )

        if len(current_dataset.core) != len(history_dataset.features):

            raise RuntimeError(

                f"Sample mismatch in {path.name}: "

                f"current Ridge = {len(current_dataset.core)}, "

                f"historical motion = {len(history_dataset.features)}"

            )

        datasets[path.stem] = (

            current_dataset,

            history_dataset,

        )

        print(

            f"    {len(current_dataset.core)} identical samples"

        )

    print()

    print("Running leave-one-video-out comparison...")

    rows: list[dict[str, object]] = []

    for held_out in datasets:

        print(f"  Held out: {held_out}")

        fold_rows = evaluate_fold(

            datasets,

            held_out,

            horizons=horizons,

            diagonal=diagonal,

        )

        rows.extend(fold_rows)

        overall_rows = [

            row

            for row in fold_rows

            if row["scope"] == "overall"

        ]

        for row in overall_rows:

            print(

                f"    {row['model']}: "

                f"ADE={float(row['mean_error_px']):.2f}px"

            )

    rows = add_overall_rows(rows)

        print()
    print("Pooled horizon results:")
    print()
    print(f"{'Horizon':<12}{'Current Ridge':>18}{'Historical Motion':>22}")
    print("-" * 52)

    for horizon in horizons:
        ridge_row = next(
            row for row in rows
            if row["split"] == "ALL_VIDEOS"
            and row["scope"] == "horizon"
            and row["model"] == "current_ridge"
            and int(row["horizon_frames"]) == horizon
        )

        history_row = next(
            row for row in rows
            if row["split"] == "ALL_VIDEOS"
            and row["scope"] == "horizon"
            and row["model"] == "historical_motion"
            and int(row["horizon_frames"]) == horizon
        )

        ridge_error = float(ridge_row["mean_error_px"])
        history_error = float(history_row["mean_error_px"])

        improvement = percentage_improvement(
            ridge_error,
            history_error,
        )

        print(
            f"{horizon:<12}"
            f"{ridge_error:>18.4f}"
            f"{history_error:>22.4f}"
        )

        print(
            f"    Historical Motion vs Current Ridge: "
            f"{improvement:+.2f}%"
        )

    write_results(

        args.output,

        rows,

    )

    print()

    print(

        f"Results written to: "

        f"{args.output.resolve()}"

    )

    print()

    print("IMPORTANT:")

    print(

        "The primary comparison is historical_motion "

        "vs current_ridge."

    )

    print(

        "Positive improvement_vs_current_ridge_pct "

        "means historical motion has lower error."

    )


if __name__ == "__main__":

    main()
