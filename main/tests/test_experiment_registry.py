import unittest

from experiments.registry import (
    UnknownExperimentError,
    get_experiment,
    list_experiments,
    register_experiment,
)


class TestExperimentPackageImports(unittest.TestCase):
    def test_only_current_experiments_are_registered(self):
        import experiments  # noqa: F401

        registered = list_experiments()
        self.assertIn("exp1", registered)
        self.assertIn("exp2_arm", registered)
        self.assertNotIn("angle_1", registered)
        self.assertNotIn("angle_2_a", registered)
        self.assertNotIn("angle_2_b", registered)
        self.assertNotIn("angle_2_c", registered)


class TestExperimentRegistry(unittest.TestCase):
    def test_known_experiments_are_registered(self):
        # experiments/__init__.py registers these as an import side effect;
        # importing experiments.registry directly (as this test does) does
        # NOT trigger that, so we register a local dummy to test the API
        # in isolation instead of depending on import order.
        @register_experiment("dummy_for_registry_test")
        def dummy(args):
            return "ran"

        self.assertIn("dummy_for_registry_test", list_experiments())
        self.assertEqual(get_experiment("dummy_for_registry_test")({}), "ran")

    def test_unknown_experiment_raises_with_available_list(self):
        @register_experiment("dummy_a")
        def a(args):
            pass

        @register_experiment("dummy_b")
        def b(args):
            pass

        with self.assertRaises(UnknownExperimentError) as ctx:
            get_experiment("totally_not_registered")

        message = str(ctx.exception)
        self.assertIn("totally_not_registered", message)
        self.assertIn("dummy_a", message)
        self.assertIn("dummy_b", message)

    def test_registering_same_name_twice_with_different_fn_raises(self):
        @register_experiment("dummy_conflict")
        def first(args):
            pass

        def second(args):
            pass

        with self.assertRaises(ValueError):
            register_experiment("dummy_conflict")(second)

    def test_re_registering_same_function_is_idempotent(self):
        def fn(args):
            pass

        register_experiment("dummy_idempotent")(fn)
        # re-registering the exact same function object must not raise
        register_experiment("dummy_idempotent")(fn)
        self.assertIs(get_experiment("dummy_idempotent"), fn)

    def test_future_experiment_is_trivially_addable(self):
        # This is the whole point of the registry: adding a new experiment
        # must not require touching run.py or any existing experiment module.
        @register_experiment("angle_3_hypothetical")
        def future_experiment(args):
            return args.get("value")

        self.assertIn("angle_3_hypothetical", list_experiments())
        self.assertEqual(get_experiment("angle_3_hypothetical")({"value": 42}), 42)


if __name__ == "__main__":
    unittest.main()
