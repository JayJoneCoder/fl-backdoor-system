"""Client-count propagation from config to Flower federation and strategy."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.experiment_worker import simulation_overrides
from fl_backdoor.config import ExperimentConfig
from fl_backdoor.server.server import build_strategy
from fl_backdoor.utils.ray_runtime import HOOK, configure_worker_environment, runtime_env_hook


def flower_config(**updates):
    config = {"num-clients": 10, "num-server-rounds": 2}
    config.update(updates)
    return {"tool": {"flwr": {"app": {"config": config}}}}


class SimulationOverridesTests(unittest.TestCase):
    def test_propagates_client_count_as_supernodes(self):
        for clients in (1, 3, 10, 100):
            with self.subTest(clients=clients):
                self.assertEqual(simulation_overrides(flower_config(**{"num-clients": clients})),
                                 f"num-supernodes={clients}")

    def test_defaults_to_ten_without_explicit_field(self):
        self.assertEqual(simulation_overrides({"tool": {"flwr": {"app": {"config": {}}}}}),
                         "num-supernodes=10")

    def test_rejects_non_positive_or_non_integer_clients(self):
        for clients in (0, -1, 1.5, True, "3", None):
            with self.subTest(clients=clients):
                with self.assertRaises(ValueError):
                    simulation_overrides(flower_config(**{"num-clients": clients}))


class StrategyClientCountTests(unittest.TestCase):
    def test_strategy_waits_for_configured_client_count(self):
        for clients in (2, 3, 7):
            with self.subTest(clients=clients):
                cfg = ExperimentConfig(num_clients=clients, fraction_evaluate=0.5)
                strategy = build_strategy(cfg)
                self.assertEqual(strategy.min_train_nodes, clients)
                self.assertEqual(strategy.min_available_nodes, clients)
                self.assertEqual(strategy.min_evaluate_nodes, 1)

    def test_config_parses_client_count_from_run_config(self):
        cfg = ExperimentConfig.from_run_config({"num-clients": 5, "num-server-rounds": 3})
        self.assertEqual(cfg.num_clients, 5)
        self.assertEqual(cfg.num_server_rounds, 3)


class RayRuntimeTests(unittest.TestCase):
    def test_snapshot_path_uses_native_separator_and_preserves_options(self):
        snapshot = str(Path("snapshot").absolute())
        environment = {"env_vars": {"PYTHONPATH": "broken:path", "KEEP": "value"}, "num_workers": 2}
        with patch.dict(os.environ, {"FL_BACKDOOR_RUNTIME_APP": snapshot}, clear=True), \
                patch.object(sys, "path", [snapshot, "libraries", "libraries"]):
            actual = runtime_env_hook(environment)
        self.assertEqual(actual["env_vars"]["PYTHONPATH"].split(os.pathsep), [snapshot, "libraries"])
        self.assertEqual(actual["env_vars"]["KEEP"], "value")
        self.assertEqual(actual["num_workers"], 2)
        self.assertEqual(environment["env_vars"]["PYTHONPATH"], "broken:path")

    def test_hook_is_only_configured_in_child_environment(self):
        before = dict(os.environ)
        env = {"RAY_RUNTIME_ENV_HOOK": "other_module.hook", "KEEP": "value"}
        configure_worker_environment(env, Path("snapshot"))
        self.assertEqual(env["RAY_RUNTIME_ENV_HOOK"], HOOK)
        self.assertEqual(env["FL_BACKDOOR_UPSTREAM_RAY_HOOK"], "other_module.hook")
        self.assertEqual(env["FL_BACKDOOR_RUNTIME_APP"], "snapshot")
        self.assertEqual(dict(os.environ), before)

    def test_existing_hook_is_chained_and_its_options_are_retained(self):
        def hook(environment):
            return {**environment, "env_vars": {"UPSTREAM": "kept"}, "working_dir": "work"}

        with patch.dict(os.environ, {"FL_BACKDOOR_UPSTREAM_RAY_HOOK": "other_module.hook"}, clear=True), \
                patch.dict(sys.modules, {"other_module": SimpleNamespace(hook=hook)}):
            result = runtime_env_hook({"other": True})
        self.assertEqual(result["env_vars"]["UPSTREAM"], "kept")
        self.assertEqual(result["working_dir"], "work")
        self.assertTrue(result["other"])


if __name__ == "__main__":
    unittest.main()
