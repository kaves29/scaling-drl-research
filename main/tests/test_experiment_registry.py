import unittest

from experiments.registry import (
    UnknownExperimentError,
    get_experiment,
    list_experiments,
    register_experiment,
)


class TestExperimentPackageImports(unittest.TestCase):
    """Regression guard: `experiments/angle_2_a.py` (the entry-point module,
    WITH the underscore between "2" and "a") must never be renamed to
    `angle_2a.py` - that collides with the `experiments/angle_2a/` package
    directory, Python silently resolves `import experiments.angle_2a` to the
    PACKAGE instead of the module, and `@register_experiment("angle_2_a")`
    inside the module never runs. This previously broke `import experiments`
    entirely (and therefore `run.py` for Angle 1 too), confirmed by direct
    execution during the post-audit correction pass."""

    def test_experiments_package_imports_without_error(self):
        import experiments  # noqa: F401 - the import succeeding is the test

    def test_both_angle_1_and_angle_2_a_are_registered_after_import(self):
        import experiments  # noqa: F401

        registered = list_experiments()
        self.assertIn("angle_1", registered)
        self.assertIn("angle_2_a", registered)

    def test_angle_2_a_entry_point_module_is_not_named_angle_2a(self):
        import pathlib

        entry_point = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "angle_2_a.py"
        colliding_name = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "angle_2a.py"
        self.assertTrue(entry_point.exists(), "entry-point module experiments/angle_2_a.py must exist")
        self.assertFalse(
            colliding_name.exists(),
            "experiments/angle_2a.py must not exist alongside the experiments/angle_2a/ "
            "package - it would shadow the real entry point and never register",
        )


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
