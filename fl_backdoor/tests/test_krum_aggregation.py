"""Tests for Krum parameter feasibility after detection filtering."""

import unittest

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

from fl_backdoor.defenses.pipeline import DefensePipelineConfig, DefensePipelineFedAvg
from fl_backdoor.defenses.server.aggregation.krum import _krum_select, resolve_krum_params


class ResolveKrumParamsTest(unittest.TestCase):
    def test_feasible_configuration_is_unchanged(self):
        self.assertEqual(resolve_krum_params(10, 3, 3), (3, 3))
        self.assertEqual(resolve_krum_params(12, 3, 1), (3, 1))

    def test_f_is_lowered_when_detection_leaves_too_few_updates(self):
        # 7 updates: n > 2*f+2 requires f <= 2 (regression: test_experiment).
        self.assertEqual(resolve_krum_params(7, 3, 3), (2, 3))
        self.assertEqual(resolve_krum_params(8, 3, 3), (2, 3))

    def test_k_is_lowered_to_the_feasible_range(self):
        self.assertEqual(resolve_krum_params(5, 3, 3), (1, 2))
        self.assertEqual(resolve_krum_params(4, 0, 5), (0, 2))
        self.assertEqual(resolve_krum_params(3, 0, 3), (0, 1))

    def test_negative_f_is_clamped_to_zero(self):
        self.assertEqual(resolve_krum_params(10, -1, 3), (0, 3))

    def test_too_few_updates_raise_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "at least 3 client updates"):
            resolve_krum_params(2, 0, 1)


class KrumSelectWithAdjustedParamsTest(unittest.TestCase):
    def test_adjusted_parameters_produce_selection(self):
        rng = np.random.default_rng(42)
        updates = [rng.normal(size=8) for _ in range(7)]
        f, k = resolve_krum_params(len(updates), 3, 3)
        selected = _krum_select(updates, f=f, k=k)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertTrue(all(0 <= index < len(updates) for index in selected))

    def test_infeasible_parameters_still_raise(self):
        updates = [np.zeros(4) for _ in range(7)]
        with self.assertRaisesRegex(ValueError, r"n > 2\*f \+ 2"):
            _krum_select(updates, f=3, k=3)


def _reply(index: int, *, malicious: bool = False) -> Message:
    content = RecordDict({
        "arrays": ArrayRecord({"w": Array(np.arange(4, dtype=np.float32) + index)}),
        "metrics": MetricRecord({
            "client_id": index,
            "is_malicious": int(malicious),
            "num-examples": 10,
        }),
    })
    return Message(dst_node_id=0, message_type="train", content=content)


class PipelineKrumAdaptationTest(unittest.TestCase):
    def test_detection_reduced_updates_are_still_aggregated(self):
        strategy = DefensePipelineFedAvg(
            pipeline_config=DefensePipelineConfig(
                aggregation_type="krum",
                aggregation_kwargs={"num_malicious": 3, "krum_k": 3},
            )
        )
        strategy.current_arrays = ArrayRecord({"w": Array(np.zeros(4, dtype=np.float32))})
        replies = [_reply(index, malicious=index < 3) for index in range(7)]

        arrays, metrics = strategy._apply_aggregation(replies)

        self.assertIsNotNone(arrays)
        self.assertEqual(metrics["agg_total_clients"], 7)
        self.assertEqual(metrics["agg_selected_clients"], 3)
        self.assertEqual(metrics["agg_krum_f"], 2)
        self.assertEqual(metrics["agg_krum_k"], 3)
        self.assertEqual(metrics["agg_krum_f_configured"], 3)
        self.assertEqual(metrics["agg_krum_k_configured"], 3)
        self.assertEqual(metrics["agg_krum_adapted"], 1)


if __name__ == "__main__":
    unittest.main()
