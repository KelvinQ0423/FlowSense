"""Tests for conditional local-coordinate history prediction."""

from __future__ import annotations

import math
import unittest

import numpy as np

from research.compare_historical_features import CsvPoint
from research.evaluate_conditional_local_history import (
    LocalHistoryDataset,
    apply_local_residuals,
    causal_motion_state,
    rotate_to_image,
    rotate_to_local,
)


def point(frame: int, x: float, y: float) -> CsvPoint:
    return CsvPoint(frame, frame / 30.0, x, y, "car", 0.9, x - 1, y - 1, x + 1, y + 1)


class ConditionalLocalHistoryTests(unittest.TestCase):
    def test_local_rotation_round_trip(self) -> None:
        for heading in (0.0, math.pi / 3.0, -math.pi / 2.0):
            local = rotate_to_local(12.0, -7.0, heading)
            image = rotate_to_image(*local, heading)
            np.testing.assert_allclose(image, (12.0, -7.0), atol=1e-12)

    def test_causal_state_detects_a_turn(self) -> None:
        history = [point(frame, float(frame), 0.0) for frame in range(7)]
        history.extend(point(frame, 6.0, float(frame - 6)) for frame in range(7, 13))
        _, turn, dynamic, _, _ = causal_motion_state(
            history,
            state_window=5,
            turn_threshold_degrees=8.0,
            speed_change_threshold=0.30,
            minimum_speed=1.0,
        )
        self.assertGreater(turn, 70.0)
        self.assertTrue(dynamic)

    def test_gate_preserves_cv_for_stable_sample(self) -> None:
        dataset = LocalHistoryDataset(
            features=np.zeros((1, 1)),
            residual_targets=np.zeros((1, 2)),
            actual=np.asarray([[10.0, 0.0]]),
            cv=np.asarray([[8.0, 0.0]]),
            headings=np.asarray([0.0]),
            turn_degrees=np.asarray([0.0]),
            speed_change_ratios=np.asarray([0.0]),
            dynamic=np.asarray([False]),
            horizon_seconds=np.asarray([[1.0]]),
        )
        prediction = apply_local_residuals(
            dataset,
            np.asarray([[100.0, 100.0]]),
            diagonal=1.0,
            correction_strength=1.0,
            residual_ratio_limit=0.5,
            minimum_residual_limit_px=5.0,
            gated=True,
        )
        np.testing.assert_array_equal(prediction, np.asarray([[[8.0, 0.0]]]))

    def test_residual_is_bounded(self) -> None:
        dataset = LocalHistoryDataset(
            features=np.zeros((1, 1)),
            residual_targets=np.zeros((1, 2)),
            actual=np.asarray([[10.0, 0.0]]),
            cv=np.asarray([[10.0, 0.0]]),
            headings=np.asarray([0.0]),
            turn_degrees=np.asarray([20.0]),
            speed_change_ratios=np.asarray([0.0]),
            dynamic=np.asarray([True]),
            horizon_seconds=np.asarray([[1.0]]),
        )
        prediction = apply_local_residuals(
            dataset,
            np.asarray([[100.0, 0.0]]),
            diagonal=1.0,
            correction_strength=1.0,
            residual_ratio_limit=0.5,
            minimum_residual_limit_px=0.0,
            gated=True,
        )
        np.testing.assert_allclose(prediction, np.asarray([[[15.0, 0.0]]]))


if __name__ == "__main__":
    unittest.main()
