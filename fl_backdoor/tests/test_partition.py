"""Partition and derived-seed helpers must be deterministic and RNG-isolated."""

import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fl_backdoor.dataset.partition import partition_loaders
from fl_backdoor.utils.randomness import derived_seed, seed_everything


class DummyDataset(torch.utils.data.Dataset):
    """Tiny in-memory dataset shaped like the loader wrappers."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {"img": torch.tensor([float(idx)]), "label": idx % 3}


def leaf_indices(subset):
    """Unwrap nested Subset layers back to original dataset indices."""
    if isinstance(subset, Subset):
        inner = leaf_indices(subset.dataset)
        return [inner[i] for i in subset.indices]
    return list(range(len(subset)))


class DerivedSeedTests(unittest.TestCase):
    def test_deterministic_and_stable(self):
        self.assertEqual(derived_seed(42, "split", 0), derived_seed(42, "split", 0))
        self.assertIsInstance(derived_seed(42), int)
        self.assertGreaterEqual(derived_seed(42), 0)

    def test_distinct_parts_and_seeds_diverge(self):
        seeds = {
            derived_seed(42, "split", 0),
            derived_seed(42, "split", 1),
            derived_seed(42, "shuffle", 0),
            derived_seed(43, "split", 0),
        }
        self.assertEqual(len(seeds), 4)

    def test_seed_everything_is_reproducible(self):
        seed_everything(123)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        seed_everything(123)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertAlmostEqual(first[2], second[2], places=6)


class PartitionTests(unittest.TestCase):
    def test_partitions_cover_every_sample_exactly_once(self):
        total = 101
        num_partitions = 7
        dataset = DummyDataset(total)
        seen: set[int] = set()
        counts = []
        for pid in range(num_partitions):
            train, val = partition_loaders(dataset, pid, num_partitions, batch_size=4, seed=42)
            indices = leaf_indices(train.dataset) + leaf_indices(val.dataset)
            self.assertEqual(len(indices), len(set(indices)), f"partition {pid} overlaps itself")
            seen.update(indices)
            counts.append(len(indices))
        self.assertEqual(seen, set(range(total)))
        self.assertEqual(sum(counts), total)

    def test_deterministic_and_per_partition_distinct(self):
        total = 90
        dataset = DummyDataset(total)
        first = partition_loaders(dataset, 0, 6, batch_size=2, seed=7)
        second = partition_loaders(dataset, 0, 6, batch_size=2, seed=7)
        a = leaf_indices(first[0].dataset)
        b = leaf_indices(second[0].dataset)
        self.assertEqual(a, b)

        other = partition_loaders(dataset, 1, 6, batch_size=2, seed=7)
        c = leaf_indices(other[0].dataset)
        self.assertNotEqual(a, c)

    def test_partition_does_not_pollute_global_rng(self):
        dataset = DummyDataset(80)
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        expected = (torch.rand(3).tolist(), np.random.rand(3).tolist(), [random.random() for _ in range(3)])
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        partition_loaders(dataset, 0, 4, batch_size=2, seed=123)
        actual = (torch.rand(3).tolist(), np.random.rand(3).tolist(), [random.random() for _ in range(3)])
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertEqual(expected[2], actual[2])

    def test_rejects_invalid_partition_shape(self):
        dataset = DummyDataset(10)
        with self.assertRaises(ValueError):
            partition_loaders(dataset, 0, 0, batch_size=2, seed=1)
        with self.assertRaises(ValueError):
            partition_loaders(dataset, 3, 3, batch_size=2, seed=1)
        with self.assertRaises(ValueError):
            partition_loaders(dataset, 0, 11, batch_size=2, seed=1)

    def test_small_partitions_keep_nonempty_validation_split(self):
        # 2 samples per client would floor 0.2*2 to zero; validation must stay usable.
        dataset = DummyDataset(8)
        train, val = partition_loaders(dataset, 0, 4, batch_size=1, seed=1)
        self.assertGreaterEqual(len(train.dataset), 1)
        self.assertGreaterEqual(len(val.dataset), 1)

    def test_single_sample_partitions_are_rejected_before_building_loaders(self):
        # Reject the whole partition plan, even if the final partition is larger.
        for samples, partitions in ((1, 1), (4, 4), (5, 3)):
            for pid in range(partitions):
                with self.subTest(samples=samples, partitions=partitions, pid=pid):
                    with self.assertRaisesRegex(ValueError, "至少需要 2 个样本"):
                        partition_loaders(DummyDataset(samples), pid, partitions, batch_size=1)


if __name__ == "__main__":
    unittest.main()
