"""CPU-only trajectory diagnostics; skip cleanly without optional NumPy."""

import unittest
import warnings

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from src.utils.trajectory_controls import summarize_trajectory_controls


@unittest.skipIf(np is None, "NumPy required; use TimeSeries/bin/python")
class TrajectoryControlTests(unittest.TestCase):
    def test_monotonic_logistic_like_and_manual_peak_settling(self):
        time = np.linspace(-8, 12, 80)
        logistic = 1 / (1 + np.exp(-time))
        settling = np.r_[np.linspace(.1, 3, 20), np.linspace(3, 1, 50), np.ones(10)]
        result = summarize_trajectory_controls(np.array([logistic, settling])[:, None, :])
        self.assertEqual(set(result), {
            "overshoot_ratio", "peak_excess_ratio", "starts_below_terminal",
            "terminal_level", "tail_cv", "tail_drift", "valid", "stationary",
        })
        self.assertTrue(result["valid"].all())
        self.assertTrue(result["stationary"].all())
        self.assertAlmostEqual(result["overshoot_ratio"][0], 0., places=4)
        self.assertAlmostEqual(result["overshoot_ratio"][1], 2.)
        self.assertAlmostEqual(result["terminal_level"][1], 1.)
        for value in result.values():
            self.assertEqual(value.shape, (2,))

    def test_multiplication_preserves_ratios_but_scales_terminal(self):
        values = np.array([[[1, 4, 2, 2, 2, 2], [2, 8, 4, 4, 4, 4]]], dtype=float)
        original = summarize_trajectory_controls(values, tail_points=4)
        scaled = summarize_trajectory_controls(values * 7, tail_points=4)
        for name in ("overshoot_ratio", "tail_cv", "tail_drift", "valid", "stationary"):
            np.testing.assert_allclose(original[name], scaled[name])
        np.testing.assert_allclose(scaled["terminal_level"], 7 * original["terminal_level"])
        np.testing.assert_allclose(original["terminal_level"], [3])
        np.testing.assert_allclose(original["overshoot_ratio"], [1])

    def test_initially_high_decay_is_excess_not_rise_from_below_overshoot(self):
        values = np.array([[[4., 3., 2., 1., 1., 1., 1.]]])
        result = summarize_trajectory_controls(values, tail_points=4)
        self.assertEqual(result["overshoot_ratio"][0], 0.)
        self.assertEqual(result["peak_excess_ratio"][0], 3.)
        self.assertFalse(result["starts_below_terminal"][0])

    def test_species_swap_invariance_and_no_mutation(self):
        values = np.array([[[1, 6, 2, 2, 2, 2], [2, 2, 4, 4, 4, 4]]], dtype=float)
        before = values.copy()
        values.flags.writeable = False
        result = summarize_trajectory_controls(values, tail_points=4)
        swapped = summarize_trajectory_controls(values[:, ::-1], tail_points=4)
        for name in result:
            np.testing.assert_allclose(result[name], swapped[name])
        np.testing.assert_array_equal(values, before)

    def test_invalid_rows_are_retained_without_warnings_or_zero_replacement(self):
        values = np.array([
            [[1, 1, 1, 1, 1, 1]],
            [[1, 1, .001, .001, .001, .001]],
            [[0, 0, 0, 0, 0, 0]],
            [[np.nan, 1, 1, 1, 1, 1]],
            [[1, 1, 1, -1, 1, 1]],
            [[np.inf, 1, 1, 1, 1, 1]],
            [[1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9]],
        ])
        before = values.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = summarize_trajectory_controls(values, tail_points=4)
        np.testing.assert_array_equal(result["valid"], [True, False, False, False, False, False, False])
        self.assertEqual(np.count_nonzero(~result["valid"]), 6)
        self.assertFalse(result["stationary"][1:].any())
        for name in ("overshoot_ratio", "tail_cv", "tail_drift"):
            self.assertTrue(np.isnan(result[name][1:]).all())
        np.testing.assert_allclose(result["terminal_level"][:3], [1, .001, 0])
        self.assertTrue(np.isnan(result["terminal_level"][3:6]).all())
        np.testing.assert_array_equal(values, before)

    def test_eligibility_boundaries_are_strict(self):
        values = np.array([[[1, .02, .02, .02, .02]], [[1e-8] * 5]])
        result = summarize_trajectory_controls(values, tail_points=4)
        self.assertFalse(result["valid"].any())

    def test_stationarity_requires_both_cv_and_drift(self):
        values = np.array([[[1, 1, 2, 2]], [[1, 2, 1, 2]], [[1, 1, 1, 1]]])
        result = summarize_trajectory_controls(values, tail_points=4)
        np.testing.assert_allclose(result["tail_cv"], [1 / 3, 1 / 3, 0])
        np.testing.assert_allclose(result["tail_drift"], [2 / 3, 0, 0])
        np.testing.assert_array_equal(result["stationary"], [False, False, True])
        cv_only = summarize_trajectory_controls(
            values, tail_points=4, stationary_cv_tolerance=.4,
        )
        np.testing.assert_array_equal(cv_only["stationary"], [False, True, True])
        drift_only = summarize_trajectory_controls(
            values, tail_points=4, stationary_drift_tolerance=.7,
        )
        np.testing.assert_array_equal(drift_only["stationary"], [False, False, True])

    def test_odd_tail_uses_all_points_and_population_std(self):
        values = np.array([[[7, 1, 1, 2, 2, 2]]])
        result = summarize_trajectory_controls(values, tail_points=5)
        self.assertAlmostEqual(result["terminal_level"][0], 1.6)
        self.assertAlmostEqual(result["tail_cv"][0], np.std([1, 1, 2, 2, 2]) / 1.6)
        self.assertAlmostEqual(result["tail_drift"][0], 1 / 1.6)

    def test_large_finite_values_do_not_overflow(self):
        values = np.full((1, 2, 10), np.finfo(float).max)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = summarize_trajectory_controls(values)
        self.assertTrue(result["valid"][0])
        self.assertTrue(result["stationary"][0])
        self.assertEqual(result["terminal_level"][0], np.finfo(float).max)

    def test_all_invalid_batch(self):
        result = summarize_trajectory_controls(np.full((2, 1, 4), np.nan), tail_points=4)
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["terminal_level"]).all())

    def test_invalid_dimensions_and_configuration(self):
        for shape in [(4,), (2, 4), (0, 1, 10), (1, 0, 10), (1, 1, 3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                summarize_trajectory_controls(np.ones(shape), tail_points=4)
        values = np.ones((1, 1, 10))
        configs = [{"tail_points": value} for value in (0, 3, 4., True, 11, np.nan)]
        for name in ("min_terminal_fraction", "stationary_cv_tolerance", "stationary_drift_tolerance"):
            configs.extend({name: value} for value in (0, -1, np.nan, np.inf, True, "0.05", [0.05]))
        configs.extend({"min_terminal_fraction": value} for value in (1, 2))
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                summarize_trajectory_controls(values, **config)
        with self.assertRaises(ValueError):
            summarize_trajectory_controls(values.astype(complex))


if __name__ == "__main__":
    unittest.main()
