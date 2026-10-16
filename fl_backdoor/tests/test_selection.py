"""Tests for malicious client selection modes (random / random_once / fixed)."""

import unittest

from fl_backdoor.attacks.base import AttackConfig
from fl_backdoor.attacks.selection import select_malicious_clients


class SelectMaliciousClientsTest(unittest.TestCase):
    def test_random_reelects_across_rounds(self):
        round_zero = select_malicious_clients(
            num_clients=10, malicious_ratio=0.3, seed=42,
            malicious_mode="random", server_round=0,
        )
        round_one = select_malicious_clients(
            num_clients=10, malicious_ratio=0.3, seed=42,
            malicious_mode="random", server_round=1,
        )
        self.assertEqual(round_zero, {0, 5, 6})
        self.assertEqual(round_one, {4, 6, 8})
        self.assertNotEqual(round_zero, round_one)

    def test_random_once_is_stable_across_rounds(self):
        first = select_malicious_clients(
            num_clients=10, malicious_ratio=0.3, seed=42,
            malicious_mode="random_once", server_round=0,
        )
        later = select_malicious_clients(
            num_clients=10, malicious_ratio=0.3, seed=42,
            malicious_mode="random_once", server_round=5,
        )
        self.assertEqual(first, later)
        self.assertEqual(first, {0, 5, 6})

    def test_random_once_matches_random_round_zero(self):
        once = select_malicious_clients(
            num_clients=10, malicious_ratio=0.3, seed=42,
            malicious_mode="random_once", server_round=3,
        )
        round_zero = select_malicious_clients(
            num_clients=10, malicious_ratio=0.3, seed=42,
            malicious_mode="random", server_round=0,
        )
        self.assertEqual(once, round_zero)

    def test_fixed_uses_explicit_ids(self):
        selected = select_malicious_clients(
            num_clients=10, malicious_ratio=0.0, seed=42,
            malicious_mode="fixed", fixed_malicious_clients=[0, 2, 5],
        )
        self.assertEqual(selected, {0, 2, 5})

    def test_fixed_filters_out_of_range_ids(self):
        selected = select_malicious_clients(
            num_clients=5, malicious_ratio=0.0, seed=42,
            malicious_mode="fixed", fixed_malicious_clients=[0, 9],
        )
        self.assertEqual(selected, {0})

    def test_fixed_accepts_comma_separated_string(self):
        selected = select_malicious_clients(
            num_clients=10, malicious_ratio=0.0, seed=42,
            malicious_mode="fixed", fixed_malicious_clients="0, 2, 5",
        )
        self.assertEqual(selected, {0, 2, 5})

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            select_malicious_clients(
                num_clients=10, malicious_ratio=0.3, seed=42,
                malicious_mode="bogus",
            )


class AttackConfigValidationTest(unittest.TestCase):
    def test_random_once_is_accepted(self):
        AttackConfig(extra={"malicious_mode": "random_once"}).validate()

    def test_unknown_mode_still_rejected(self):
        with self.assertRaises(ValueError):
            AttackConfig(extra={"malicious_mode": "bogus"}).validate()


if __name__ == "__main__":
    unittest.main()
