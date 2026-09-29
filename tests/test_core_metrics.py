"""Regression tests for the paper's core temporal-metric definitions."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


def load_script(module_name: str, relative_path: str):
    """Load a numbered pipeline script without renaming the source file."""
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPOSITORY_ROOT / relative_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


temporal = load_script(
    "corrected_temporal_metrics_for_tests",
    "temporal_analysis/45_recalculate_corrected_temporal_metrics.py",
)
clustered = load_script(
    "sequence_clustered_statistics_for_tests",
    "temporal_analysis/48_sequence_clustered_temporal_statistics.py",
)
# The fog transform under test does not use OpenCV.  A minimal import stub lets
# this unit test run in CPU-only review environments before optional OpenCV is
# installed; the full pipeline still requires opencv-python.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

degradation = load_script(
    "degraded_test_sets_for_tests",
    "data_preparation/33_build_degraded_test_sets.py",
)


class CorrectedTemporalMetricTests(unittest.TestCase):
    def test_valid_window_rejects_gap_above_two_seconds(self):
        observations = [
            {"timestamp": 0.0},
            {"timestamp": 1.0},
            {"timestamp": 3.1},
            {"timestamp": 4.1},
        ]

        window = temporal.corrected_valid_window(
            observations,
            start_index=0,
            window_seconds=2.0,
        )

        self.assertIsNone(window)

    def test_non_red_state_terminates_red_miss_episode(self):
        observations = [
            {
                "timestamp": 0.0,
                "ground_truth_state": "red",
                "complete_red_miss": True,
            },
            {
                "timestamp": 1.0,
                "ground_truth_state": "red",
                "complete_red_miss": True,
            },
            {
                "timestamp": 2.0,
                "ground_truth_state": "green",
                "complete_red_miss": False,
            },
            {
                "timestamp": 3.0,
                "ground_truth_state": "red",
                "complete_red_miss": True,
            },
            {
                "timestamp": 4.0,
                "ground_truth_state": "red",
                "complete_red_miss": True,
            },
            {
                "timestamp": 5.0,
                "ground_truth_state": "red",
                "complete_red_miss": True,
            },
        ]

        longest_frames, longest_seconds = temporal.longest_complete_red_miss(
            observations
        )

        self.assertEqual(longest_frames, 3)
        self.assertEqual(longest_seconds, 2.0)

    def test_auc5_matches_hand_computed_case(self):
        # Contributions over 0--5 s are 1.0, 0.5, 0.0, 0.0, and 0.0.
        starts = [0.0, 2.5, 5.0, None, 6.0]

        self.assertAlmostEqual(temporal.auc5_from_starts(starts), 30.0)

    def test_sequence_key_returns_three_level_sequence_identifier(self):
        self.assertEqual(
            clustered.sequence_key(
                "Essen/Essen2/2015-04-22_11-54-56/Trafficlight_1"
            ),
            "Essen/Essen2/2015-04-22_11-54-56",
        )
        self.assertEqual(
            clustered.sequence_key(
                r"Essen\Essen2\2015-04-22_11-54-56\Trafficlight_1"
            ),
            "Essen/Essen2/2015-04-22_11-54-56",
        )

    def test_fog_transform_uses_linear_white_blending(self):
        image = np.asarray([[[0, 100, 255]]], dtype=np.uint8)

        transformed = degradation.apply_fog(image, alpha=0.25)

        expected = np.asarray([[[63, 138, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(transformed, expected)
        self.assertEqual(transformed.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
