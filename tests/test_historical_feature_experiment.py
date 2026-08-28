"""Tests for the controlled historical-motion feature experiment."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.compare_historical_features import (
    build_dataset,
    estimate_acceleration,
    fit_ridge,
    predict_ridge,
)


class HistoricalFeatureExperimentTests(unittest.TestCase):
    def test_acceleration_estimate_uses_velocity_slope(self) -> None:
        values = [(0.0, 1.0, -1.0), (1.0, 3.0, 2.0), (2.0, 5.0, 5.0)]
        acceleration_x, acceleration_y = estimate_acceleration(values)
        self.assertAlmostEqual(acceleration_x, 2.0)
        self.assertAlmostEqual(acceleration_y, 3.0)

    def test_feature_groups_share_identical_samples_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "frame",
                        "time_seconds",
                        "track_id",
                        "class_name",
                        "confidence",
                        "center_x",
                        "center_y",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ),
                )
                writer.writeheader()
                for frame in range(90):
                    writer.writerow(
                        {
                            "frame": frame,
                            "time_seconds": frame / 30.0,
                            "track_id": 1,
                            "class_name": "car",
                            "confidence": 0.9,
                            "center_x": frame + 0.02 * frame * frame,
                            "center_y": frame * 0.5,
                            "x1": frame - 2,
                            "y1": frame * 0.5 - 1,
                            "x2": frame + 2,
                            "y2": frame * 0.5 + 1,
                        }
                    )
            dataset = build_dataset(
                path,
                history_points=15,
                eligibility_history_points=30,
                horizons=(5, 30),
                frame_width=1920,
                frame_height=1080,
            )
            lengths = {
                len(dataset.features(model))
                for model in (
                    "core_ridge",
                    "acceleration_ridge",
                    "history_ridge",
                    "full_ridge",
                )
            }
            self.assertEqual(lengths, {31})
            self.assertEqual(dataset.scale_targets.shape, (31, 2))
            self.assertEqual(dataset.features("core_ridge").shape[1], 20)
            self.assertEqual(dataset.features("acceleration_ridge").shape[1], 23)
            self.assertEqual(dataset.features("history_ridge").shape[1], 76)
            self.assertEqual(dataset.features("full_ridge").shape[1], 79)
            self.assertTrue(np.all(dataset.scale_targets >= 0.25))
            self.assertTrue(np.all(dataset.scale_targets <= 1.75))

    def test_multioutput_ridge_reproduces_linear_targets(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        targets = np.concatenate((2.0 * features + 1.0, -features + 4.0), axis=1)
        means, scales, coefficients, intercepts = fit_ridge(
            features,
            targets,
            alpha=0.0,
        )
        predicted = predict_ridge(
            features,
            means=means,
            scales=scales,
            coefficients=coefficients,
            intercepts=intercepts,
        )
        np.testing.assert_allclose(predicted, targets, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
