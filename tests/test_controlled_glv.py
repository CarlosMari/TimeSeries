"""Tiny CPU fixtures only; no production parameter search or artifact writes."""

from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from data_generation import controlled_glv as glv
except ImportError:
    np = glv = None


@unittest.skipIf(glv is None, "NumPy/SciPy required; use TimeSeries/bin/python")
class ControlledGLVTests(unittest.TestCase):
    def setUp(self):
        self.config = glv.GLVConfig(species=1, tmax=1.0, n_points=9,
                                    solver_max_step=0.1, solver_rtol=1e-9,
                                    solver_atol=1e-11, selection="none",
                                    max_parameter_attempts=2)
        self.parameters = glv.GLVParameters(np.array([1.0]), np.array([[-1.0]]),
                                            np.array([0.25]))

    def generate(self, **kwargs):
        return glv.generate_one(self.config, parameter_seed=12,
                                noise_seed=kwargs.pop("noise_seed", 34),
                                parameters=self.parameters, **kwargs)

    def test_analytic_logistic_and_physical_grid(self):
        record = self.generate()
        self.assertIsInstance(record, glv.GLVRecord)
        np.testing.assert_array_equal(record.times, np.linspace(0, 1, 9))
        exact = 1 / (1 + 3 * np.exp(-record.times))
        np.testing.assert_allclose(record.clean[0], exact, rtol=1e-8, atol=1e-10)
        np.testing.assert_array_equal(record.clean[:, 0], self.parameters.x0)
        np.testing.assert_array_equal(record.equilibrium, [1.0])
        self.assertEqual(record.clean.shape, (1, 9))
        self.assertEqual(record.parameter_source, "provided")
        self.assertEqual(record.config, self.config)
        self.assertEqual((record.parameter_seed, record.noise_seed), (12, 34))
        self.assertEqual(record.parameter_attempts, 1)
        self.assertEqual(record.solver_method, "RK45")
        self.assertGreater(record.rhs_evaluations, 0)
        for name in ("r", "A", "x0"):
            np.testing.assert_array_equal(getattr(record.parameters, name),
                                          getattr(self.parameters, name))
            self.assertIsNot(getattr(record.parameters, name), getattr(self.parameters, name))

    def test_proposed_study_configuration_matches_simulator_contract(self):
        path = Path(__file__).resolve().parent.parent / "configs/controlled_study_v2.proposed.json"
        proposal = json.loads(path.read_text())
        configured = glv.GLVConfig(**proposal["simulator"])
        fractions = glv.SplitFractions(**proposal["splits"]["fractions"])
        self.assertEqual(configured.species, 7)
        self.assertEqual(configured.selection, "clean")
        self.assertEqual(fractions.train, .8)
        self.assertFalse(proposal["execution_policy"]["gpu_allowed"])

    def test_replay_and_noise_pairing(self):
        first, replay = self.generate(), self.generate()
        other = self.generate(noise_seed=35)
        np.testing.assert_array_equal(first.clean, replay.clean)
        np.testing.assert_array_equal(first.observed, replay.observed)
        np.testing.assert_array_equal(first.clean, other.clean)
        np.testing.assert_array_equal(first.times, other.times)
        self.assertFalse(np.array_equal(first.observed, other.observed))
        expected = glv.observe(first.clean, self.config.observation_lognormal_sigma, 34)
        np.testing.assert_array_equal(first.observed, expected)

    def test_sampled_replay_uses_seed_without_global_rng(self):
        # One species is always feasible/stable for positive exponential draws.
        before = np.random.get_state()
        first = glv.generate_one(self.config, parameter_seed=8, noise_seed=9)
        second = glv.generate_one(self.config, parameter_seed=8, noise_seed=9)
        self.assertIsInstance(first, glv.GLVRecord)
        np.testing.assert_array_equal(first.parameters.A, second.parameters.A)
        np.testing.assert_array_equal(first.parameters.r, second.parameters.r)
        np.testing.assert_array_equal(first.parameters.x0, second.parameters.x0)
        np.testing.assert_array_equal(first.clean, second.clean)
        np.testing.assert_array_equal(first.observed, second.observed)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_mean_one_noise_formula_and_seed(self):
        clean = np.ones((1, 20000))
        sigma = 0.3
        noisy = glv.observe(clean, sigma, 19)
        expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([19, glv.NOISE_DOMAIN]))).lognormal(
            -sigma**2 / 2, sigma, clean.shape)
        np.testing.assert_array_equal(noisy, expected)
        self.assertAlmostEqual(noisy.mean(), 1.0, delta=0.01)
        self.assertTrue((noisy > 0).all())
        np.testing.assert_array_equal(glv.observe(clean, 0, 19), clean)

    def test_equal_numeric_seeds_have_distinct_parameter_and_noise_streams(self):
        parameter_rng = np.random.default_rng(np.random.SeedSequence([42, glv.PARAMETER_DOMAIN]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([42, glv.NOISE_DOMAIN]))
        self.assertFalse(np.array_equal(parameter_rng.random(16), noise_rng.random(16)))
        record = glv.generate_one(self.config, parameter_seed=42, noise_seed=42,
                                  parameters=self.parameters)
        self.assertEqual(record.rng_scheme, glv.RNG_SCHEME)

    def test_r_scale_intervention_preserves_other_parameter_draws(self):
        id_parameters = glv.draw_parameters(self.config, np.random.default_rng(7))
        ood_config = replace(self.config, r_scale=2 * self.config.r_scale)
        ood_parameters = glv.draw_parameters(ood_config, np.random.default_rng(7))
        np.testing.assert_array_equal(ood_parameters.r, 2 * id_parameters.r)
        np.testing.assert_array_equal(ood_parameters.A, id_parameters.A)
        np.testing.assert_array_equal(ood_parameters.x0, id_parameters.x0)

    def test_frozen_config_and_only_declared_ood_intervention(self):
        with self.assertRaises(FrozenInstanceError):
            self.config.r_scale = 3.0
        ood = replace(self.config, r_scale=3.0)
        self.assertEqual([f.name for f in fields(self.config)
                          if getattr(self.config, f.name) != getattr(ood, f.name)], ["r_scale"])
        for changes in ({"tmax": 0}, {"n_points": 1}, {"species": True},
                        {"solver_rtol": np.nan}, {"max_rhs_evaluations": 0},
                        {"max_parameter_attempts": 0}, {"selection": "noisy"},
                        {"observation_lognormal_sigma": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_parameter_attempt_cap_and_retained_reasons(self):
        singular = replace(self.parameters, A=np.zeros((1, 1)))
        with patch.object(glv, "draw_parameters", return_value=singular) as draw:
            result = glv.generate_one(self.config, parameter_seed=1, noise_seed=2)
        self.assertIsInstance(result, glv.GenerationFailure)
        self.assertEqual(result.reason, "parameter_attempts_exhausted")
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(result.parameter_attempts, 2)
        self.assertEqual(result.parameter_rejections, ("equilibrium_singular",) * 2)
        with patch.object(glv, "draw_parameters", side_effect=[singular, self.parameters]):
            result = glv.generate_one(self.config, parameter_seed=1, noise_seed=2)
        self.assertIsInstance(result, glv.GLVRecord)
        self.assertEqual(result.parameter_attempts, 2)
        self.assertEqual(result.parameter_rejections, ("equilibrium_singular",))

    def test_parameter_failures(self):
        unstable = glv.GLVParameters(np.array([-1.0]), np.array([[1.0]]), np.array([0.25]))
        fixtures = [
            (replace(self.parameters, A=np.zeros((1, 1))), "equilibrium_singular"),
            (replace(self.parameters, r=np.array([-1.0])), "equilibrium_infeasible"),
            (unstable, "equilibrium_unstable"),
            (replace(self.parameters, x0=np.array([0.])), "initial_condition_nonpositive"),
            (replace(self.parameters, r=np.array([np.nan])), "parameter_nonfinite"),
            (replace(self.parameters, r=np.ones(2)), "parameter_shape"),
        ]
        for parameters, reason in fixtures:
            with self.subTest(reason=reason):
                result = glv.generate_one(self.config, parameter_seed=1, noise_seed=2,
                                          parameters=parameters)
                self.assertEqual(result.reason, "provided_parameters_rejected")
                self.assertEqual(result.parameter_rejections, (reason,))

    def test_deterministic_solver_work_limit(self):
        config = replace(self.config, max_rhs_evaluations=1)
        result = glv.generate_one(config, parameter_seed=1, noise_seed=2,
                                  parameters=self.parameters)
        self.assertEqual(result.reason, "rhs_evaluation_limit")

    def test_solver_failures_are_explicit(self):
        for solver_result, reason in [
            (SimpleNamespace(success=False, message="fixture failure"), "solver_failed"),
            (SimpleNamespace(success=True, t=np.array([0.])), "solver_incomplete_grid"),
        ]:
            with self.subTest(reason=reason), patch.object(glv, "solve_ivp", return_value=solver_result):
                self.assertEqual(self.generate().reason, reason)
        with patch.object(glv, "solve_ivp", side_effect=ValueError("fixture error")):
            result = self.generate()
        self.assertEqual(result.reason, "solver_exception")
        self.assertEqual(result.detail, "fixture error")

    def test_clean_selection_thresholds_and_invalid_trajectories(self):
        config = replace(self.config, species=3, n_points=12, selection="clean")
        good = np.full((3, 12), 0.2)
        good[:, 0] = 0.5
        self.assertIsNone(glv.clean_selection_reason(good, config))
        cases = [(np.full((3, 12), 4.), "selection_max_abundance"),
                 (np.full((3, 12), 0.05), "selection_min_species_peak"),
                 (np.full((3, 12), 0.5), "selection_overshoot"),
                 (np.full((3, 12), np.nan), "trajectory_nonfinite"),
                 (np.full((3, 12), -1.), "trajectory_negative"),
                 (np.ones((1, 2)), "trajectory_shape")]
        for clean, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(glv.clean_selection_reason(clean, config), reason)
        boundary = np.full((3, 12), 0.1)
        boundary[:, 0] = 3.0
        self.assertIsNone(glv.clean_selection_reason(boundary, config))

    def test_no_selection_on_noisy_observation(self):
        # Clean logistic curve is allowed with explicitly zero required overshoots.
        config = replace(self.config, selection="clean", min_overshoot_species=0)
        with patch.object(glv, "observe", return_value=np.full((1, 9), 99.0)):
            result = glv.generate_one(config, parameter_seed=1, noise_seed=2,
                                      parameters=self.parameters)
        self.assertIsInstance(result, glv.GLVRecord)
        self.assertGreater(result.observed.max(), config.max_abundance)
        with patch.object(glv, "observe", return_value=np.full((1, 9), np.inf)):
            self.assertEqual(self.generate().reason, "observation_nonfinite")
        config = replace(config, min_species_peak=2.0)
        with patch.object(glv, "observe") as observe:
            result = glv.generate_one(config, parameter_seed=1, noise_seed=2,
                                      parameters=self.parameters)
            observe.assert_not_called()
        self.assertEqual(result.reason, "selection_min_species_peak")

    def test_explicit_noise_seed_does_not_change_clean_acceptance(self):
        config = replace(self.config, selection="clean", min_overshoot_species=0,
                         observation_lognormal_sigma=0.5)
        first = glv.generate_one(config, parameter_seed=7, noise_seed=8,
                                 parameters=self.parameters)
        second = glv.generate_one(config, parameter_seed=7, noise_seed=9,
                                  parameters=self.parameters)
        self.assertIsInstance(first, glv.GLVRecord)
        self.assertIsInstance(second, glv.GLVRecord)
        np.testing.assert_array_equal(first.clean, second.clean)
        self.assertEqual(glv.assign_split(first.parameter_seed),
                         glv.assign_split(second.parameter_seed))

    def test_grouped_splits_and_fraction_report(self):
        seeds = [1, 2, 3, 4, 5, 1, 1, 2]
        assignments = [glv.assign_split(seed) for seed in seeds]
        self.assertEqual(assignments[0], assignments[-2])
        self.assertEqual(assignments[1], assignments[-1])
        self.assertEqual(assignments, list(reversed([glv.assign_split(s) for s in reversed(seeds)])))
        groups = {name: {s for s in seeds if glv.assign_split(s) == name}
                  for name in ("train", "validation", "test")}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        report = glv.split_report(seeds)
        self.assertEqual(sum(report["system_counts"].values()), 5)
        self.assertEqual(sum(report["observation_counts"].values()), 8)
        self.assertAlmostEqual(sum(report["system_fractions"].values()), 1)
        self.assertAlmostEqual(sum(report["observation_fractions"].values()), 1)
        self.assertEqual(report["target_fractions"], {"train": .8, "validation": .1, "test": .1})
        self.assertEqual(glv.split_report([])["system_fractions"], dict.fromkeys(groups, 0.0))
        for fractions, expected in [(glv.SplitFractions(1, 0, 0), "train"),
                                    (glv.SplitFractions(0, 1, 0), "validation"),
                                    (glv.SplitFractions(0, 0, 1), "test")]:
            self.assertEqual(glv.assign_split(12, fractions), expected)
        for values in ((.8, .2, .1), (-.1, .5, .6), (np.nan, 0, 1)):
            with self.assertRaises(ValueError):
                glv.SplitFractions(*values)
        for seed in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                glv.assign_split(seed)


if __name__ == "__main__":
    unittest.main()
