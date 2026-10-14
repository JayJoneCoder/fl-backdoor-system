"""DBA/FCBA dataset regressions using CPU-only synthetic images."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from backend import config_manager
from fl_backdoor.attacks import build_attack
from fl_backdoor.attacks.dba import build_dba_attack
from fl_backdoor.attacks.fcba import (
    FCBAPoisonedDataset,
    FCBATriggeredDataset,
    build_fcba_attack,
)
from fl_backdoor.config import ExperimentConfig
from fl_backdoor.dataset.cifar10 import CIFAR10Dataset
from fl_backdoor.dataset.mnist import MNISTDataset


DATASET_METAS = (MNISTDataset.meta, CIFAR10Dataset.meta)


def synthetic_sample(meta, raw_value=0.2, label=1, dtype=torch.float32):
    raw = torch.full(meta.input_shape, raw_value, dtype=dtype)
    mean = raw.new_tensor(meta.mean).view(-1, 1, 1)
    std = raw.new_tensor(meta.std).view(-1, 1, 1)
    return {"img": (raw - mean) / std, "img_raw": raw, "label": label}


class DistributedAttackTests(unittest.TestCase):
    def make_attack(self, kind, meta, **kwargs):
        return build_attack(kind, dataset_meta=meta, poison_rate=1.0, **kwargs)

    def trigger_mask(self, attack, kind, image, local=False, client_id=0):
        mask = torch.zeros_like(image, dtype=torch.bool)
        if kind == "dba":
            config = (attack.get_sub_trigger_config(client_id) if local
                      else attack.get_global_trigger_config())
            row, col = config["location"]
            size = config["size"]
            mask[:, row:row + size, col:col + size] = True
        else:
            config = (attack.get_local_trigger_config(client_id) if local
                      else attack.get_global_trigger_config())
            row, col = config["location"]
            small = config["mask"].bool()
            height, width = small.shape[-2:]
            mask[:, row:row + height, col:col + width] = small
        return mask

    def assert_only_trigger_changed(self, clean, changed, mask):
        self.assertEqual(changed.shape, clean.shape)
        self.assertEqual(changed.dtype, clean.dtype)
        self.assertEqual(changed.device, clean.device)
        self.assertTrue(torch.equal(changed[~mask], clean[~mask]))
        self.assertTrue(torch.all(changed[mask] != clean[mask]))

    def test_default_trigger_fits_mnist_and_cifar10(self):
        for meta in DATASET_METAS:
            for kind in ("dba", "fcba"):
                with self.subTest(dataset=meta.name, attack=kind):
                    attack = self.make_attack(kind, meta)
                    self.assertEqual(attack.global_trigger_location,
                                     (meta.input_shape[1] - 4, meta.input_shape[2] - 4))
                    sample = synthetic_sample(meta)
                    original = sample["img"].clone()
                    loader = DataLoader([sample], batch_size=1)
                    triggered = attack.get_triggered_loader(loader).dataset[0]
                    mask = self.trigger_mask(attack, kind, original)
                    self.assert_only_trigger_changed(original, triggered["img"], mask)
                    self.assertEqual(int(mask.sum()), meta.input_shape[0] * 16)
                    self.assertEqual(int(triggered["label"]), sample["label"])
                    self.assertTrue(torch.equal(sample["img"], original))

    def test_poisoned_training_uses_only_client_local_pattern(self):
        for meta in DATASET_METAS:
            for kind in ("dba", "fcba"):
                with self.subTest(dataset=meta.name, attack=kind):
                    attack = self.make_attack(kind, meta)
                    attack._current_client_id = 0
                    sample = synthetic_sample(meta)
                    loader = DataLoader([sample], batch_size=1, drop_last=True)
                    poisoned_loader = attack.get_poisoned_loader(loader)
                    poisoned = poisoned_loader.dataset[0]
                    mask = self.trigger_mask(attack, kind, sample["img"], local=True)
                    self.assert_only_trigger_changed(sample["img"], poisoned["img"], mask)
                    self.assertEqual(int(mask.sum()), meta.input_shape[0] * 4)
                    self.assertEqual(int(poisoned["label"]), 0)
                    self.assertEqual(poisoned_loader.batch_size, loader.batch_size)
                    self.assertTrue(poisoned_loader.drop_last)
                    batch = next(iter(poisoned_loader))
                    self.assertEqual(tuple(batch["img"].shape), (1, *meta.input_shape))

    def test_poison_rate_zero_leaves_samples_unchanged(self):
        for meta in DATASET_METAS:
            for kind in ("dba", "fcba"):
                with self.subTest(dataset=meta.name, attack=kind):
                    attack = build_attack(kind, dataset_meta=meta, poison_rate=0.0)
                    attack._current_client_id = 0
                    sample = synthetic_sample(meta)
                    result = attack.get_poisoned_loader(DataLoader([sample])).dataset[0]
                    self.assertTrue(torch.equal(result["img"], sample["img"]))
                    self.assertEqual(int(result["label"]), 1)

    def test_target_label_samples_are_not_poisoned(self):
        for meta in DATASET_METAS:
            for kind in ("dba", "fcba"):
                with self.subTest(dataset=meta.name, attack=kind):
                    attack = self.make_attack(kind, meta)
                    attack._current_client_id = 0
                    sample = synthetic_sample(meta, label=0)
                    result = attack.get_poisoned_loader(DataLoader([sample])).dataset[0]
                    self.assertTrue(torch.equal(result["img"], sample["img"]))
                    self.assertEqual(int(result["label"]), 0)

    def test_dba_value_is_raw_pixel_value_after_normalization(self):
        for meta in DATASET_METAS:
            with self.subTest(dataset=meta.name):
                attack = self.make_attack("dba", meta, dba_global_trigger_value=0.75)
                sample = synthetic_sample(meta)
                result = attack.get_triggered_loader(DataLoader([sample])).dataset[0]["img"]
                row, col = attack.global_trigger_location
                expected = (result.new_tensor(0.75) - result.new_tensor(meta.mean)) / result.new_tensor(meta.std)
                torch.testing.assert_close(result[:, row, col], expected)

    def test_fcba_adds_in_raw_space_and_does_not_clip_background(self):
        for meta in DATASET_METAS:
            with self.subTest(dataset=meta.name):
                attack = self.make_attack("fcba", meta, fcba_global_trigger_value=0.25)
                sample = synthetic_sample(meta, raw_value=0.05)
                loader = DataLoader([sample])
                attack._current_client_id = 0
                for local, output in (
                    (False, attack.get_triggered_loader(loader).dataset[0]),
                    (True, attack.get_poisoned_loader(loader).dataset[0]),
                ):
                    mask = self.trigger_mask(attack, "fcba", sample["img"], local=local)
                    self.assert_only_trigger_changed(sample["img"], output["img"], mask)
                    mean = output["img"].new_tensor(meta.mean).view(-1, 1, 1)
                    std = output["img"].new_tensor(meta.std).view(-1, 1, 1)
                    recovered = output["img"] * std + mean
                    torch.testing.assert_close(recovered[mask], torch.full_like(recovered[mask], 0.30))
                    self.assertTrue(torch.all(output["img"][~mask] < 0))

    def test_fcba_clamps_only_trigger_pixels_in_raw_space(self):
        for meta in DATASET_METAS:
            with self.subTest(dataset=meta.name):
                attack = self.make_attack("fcba", meta, fcba_global_trigger_value=0.6)
                sample = synthetic_sample(meta, raw_value=0.8)
                sample["img"][:, 0, 0] = -10.0
                result = attack.get_triggered_loader(DataLoader([sample])).dataset[0]["img"]
                mask = self.trigger_mask(attack, "fcba", sample["img"])
                self.assertTrue(torch.equal(result[~mask], sample["img"][~mask]))
                mean = result.new_tensor(meta.mean).view(-1, 1, 1)
                std = result.new_tensor(meta.std).view(-1, 1, 1)
                torch.testing.assert_close((result * std + mean)[mask], torch.ones_like(result[mask]))

    def test_metadata_controls_channels_and_rectangular_shape(self):
        meta = replace(CIFAR10Dataset.meta, input_shape=(2, 19, 23),
                       mean=(0.1, 0.6), std=(0.2, 0.8))
        for kind in ("dba", "fcba"):
            with self.subTest(attack=kind):
                attack = self.make_attack(kind, meta)
                self.assertEqual(attack.global_trigger_location, (15, 19))
                sample = synthetic_sample(meta)
                result = attack.get_triggered_loader(DataLoader([sample])).dataset[0]["img"]
                mask = self.trigger_mask(attack, kind, sample["img"])
                self.assert_only_trigger_changed(sample["img"], result, mask)
                expected = (result.new_tensor(1.0) - result.new_tensor(meta.mean)) / result.new_tensor(meta.std)
                torch.testing.assert_close(result[:, -1, -1], expected)

    def test_metadata_overrides_legacy_fcba_image_size(self):
        attack = build_fcba_attack(dataset_meta=MNISTDataset.meta, image_size=64)
        self.assertEqual(attack.global_trigger_location, (24, 24))
        self.assertEqual(tuple(attack.global_mask_small.shape), (1, 4, 4))

    def test_dtype_is_preserved(self):
        for meta in DATASET_METAS:
            for kind in ("dba", "fcba"):
                with self.subTest(dataset=meta.name, attack=kind):
                    sample = synthetic_sample(meta, dtype=torch.float64)
                    attack = self.make_attack(kind, meta)
                    attack._current_client_id = 0
                    for loader in (attack.get_poisoned_loader(DataLoader([sample])),
                                   attack.get_triggered_loader(DataLoader([sample]))):
                        self.assertEqual(loader.dataset[0]["img"].dtype, torch.float64)

    def test_legacy_factory_without_metadata_accepts_raw_cifar_shape(self):
        for factory in (build_dba_attack, build_fcba_attack):
            with self.subTest(factory=factory.__name__):
                sample = {"img": torch.full((3, 32, 32), 0.2), "label": 1}
                attack = factory()
                result = attack.get_triggered_loader(DataLoader([sample])).dataset[0]["img"]
                self.assertEqual(result.shape, sample["img"].shape)
                self.assertTrue(torch.equal(result[:, :28, :], sample["img"][:, :28, :]))
                self.assertTrue(torch.all(result[:, 28:, 28:] == 1.0))

    def test_fcba_dataset_wrappers_use_actual_grayscale_shape(self):
        sample = {"img": torch.full((1, 28, 28), 0.2), "label": 1}
        mask = torch.ones(1, 4, 4)
        config = {"mask": mask, "value": 1.0, "location": (24, 24)}
        for dataset in (
            FCBAPoisonedDataset([sample], 1.0, 0, config),
            FCBATriggeredDataset([sample], mask, 1.0, (24, 24)),
        ):
            with self.subTest(wrapper=type(dataset).__name__):
                result = dataset[0]["img"]
                self.assertEqual(tuple(result.shape), (1, 28, 28))
                self.assertTrue(torch.equal(result[:, :24, :], sample["img"][:, :24, :]))

    def test_default_dba_parts_tile_the_global_trigger(self):
        attack = self.make_attack("dba", MNISTDataset.meta)
        image = synthetic_sample(MNISTDataset.meta)["img"]
        parts = [self.trigger_mask(attack, "dba", image, local=True, client_id=cid)
                 for cid in range(4)]
        combined = torch.stack(parts).sum(dim=0)
        self.assertTrue(torch.equal(combined.bool(), self.trigger_mask(attack, "dba", image)))
        self.assertEqual(int(combined.max()), 1)

    def test_default_fcba_combinations_are_distinct_and_match_channels(self):
        for meta in DATASET_METAS:
            with self.subTest(dataset=meta.name):
                attack = self.make_attack("fcba", meta)
                self.assertEqual(attack.num_local_triggers, 14)
                masks = attack.local_trigger_masks
                for mask in masks:
                    self.assertEqual(tuple(mask.shape), (meta.input_shape[0], 4, 4))
                    self.assertGreater(int(mask.sum()), 0)
                    self.assertLess(int(mask.sum()), int(attack.global_mask_small.sum()))
                self.assertEqual(len({tuple(mask.flatten().tolist()) for mask in masks}), 14)

    def test_explicit_valid_location_is_honored(self):
        for factory in (build_dba_attack, build_fcba_attack):
            with self.subTest(factory=factory.__name__):
                attack = factory(dataset_meta=MNISTDataset.meta, global_trigger_location=(0, 2))
                self.assertEqual(attack.global_trigger_location, (0, 2))
                sample = synthetic_sample(MNISTDataset.meta)
                result = attack.get_triggered_loader(DataLoader([sample])).dataset[0]["img"]
                self.assertTrue(torch.all(result[:, :4, 2:6] != sample["img"][:, :4, 2:6]))

    def test_out_of_bounds_locations_fail_instead_of_silent_clipping(self):
        for factory in (build_dba_attack, build_fcba_attack):
            for location in ((28, 28), (-1, 0), (0, -1), (25, 24), (24, 25)):
                with self.subTest(factory=factory.__name__, location=location):
                    with self.assertRaisesRegex(ValueError, "location|fit|bounds"):
                        factory(dataset_meta=MNISTDataset.meta, global_trigger_location=location)

    def test_malformed_locations_are_rejected(self):
        for factory in (build_dba_attack, build_fcba_attack):
            for location in ((1,), (1, 2, 3), (1.5, 2), "[1,2]"):
                with self.subTest(factory=factory.__name__, location=location):
                    with self.assertRaises(ValueError):
                        factory(dataset_meta=MNISTDataset.meta, global_trigger_location=location)

    def test_oversized_global_trigger_is_rejected(self):
        for factory in (build_dba_attack, build_fcba_attack):
            with self.subTest(factory=factory.__name__):
                with self.assertRaisesRegex(ValueError, "fit|size"):
                    factory(dataset_meta=MNISTDataset.meta, trigger_size=32)

    def test_invalid_raw_trigger_values_are_rejected(self):
        for factory in (build_dba_attack, build_fcba_attack):
            for value in (-0.1, 1.1, float("nan"), float("inf")):
                with self.subTest(factory=factory.__name__, value=value):
                    with self.assertRaisesRegex(ValueError, "value"):
                        factory(dataset_meta=MNISTDataset.meta, global_trigger_value=value)

    def test_invalid_normalization_metadata_is_rejected(self):
        for factory in (build_dba_attack, build_fcba_attack):
            for meta in (
                replace(MNISTDataset.meta, mean=(0.1, 0.2)),
                replace(MNISTDataset.meta, std=(0.0,)),
                replace(MNISTDataset.meta, std=(-1.0,)),
                replace(MNISTDataset.meta, mean=(float("nan"),)),
            ):
                with self.subTest(factory=factory.__name__, meta=meta):
                    with self.assertRaisesRegex(ValueError, "mean|std|normalization"):
                        factory(dataset_meta=meta)

    def test_wrong_sample_shape_is_rejected(self):
        for kind in ("dba", "fcba"):
            with self.subTest(attack=kind):
                attack = self.make_attack(kind, MNISTDataset.meta)
                sample = {"img": torch.zeros(3, 32, 32), "label": 1}
                loader = attack.get_triggered_loader(DataLoader([sample]))
                with self.assertRaisesRegex(ValueError, "shape"):
                    loader.dataset[0]

    def test_random_patterns_are_deterministic_and_within_bounds(self):
        for factory in (build_dba_attack, build_fcba_attack):
            with self.subTest(factory=factory.__name__):
                attacks = [factory(dataset_meta=MNISTDataset.meta, split_strategy="random", poison_rate=1.0)
                           for _ in range(2)]
                sample = synthetic_sample(MNISTDataset.meta)
                outputs = []
                for attack in attacks:
                    attack._current_client_id = 0
                    outputs.append(attack.get_poisoned_loader(DataLoader([sample])).dataset[0]["img"])
                self.assertTrue(torch.equal(outputs[0], outputs[1]))
                self.assertTrue(torch.any(outputs[0] != sample["img"]))

    def test_missing_client_id_is_still_rejected(self):
        for kind in ("dba", "fcba"):
            with self.subTest(attack=kind):
                attack = self.make_attack(kind, MNISTDataset.meta)
                with self.assertRaises(RuntimeError):
                    attack.get_poisoned_loader(DataLoader([synthetic_sample(MNISTDataset.meta)]))


class DistributedAttackConfigTests(unittest.TestCase):
    def test_ui_default_location_is_automatic_for_both_attacks(self):
        for kind in ("dba", "fcba"):
            with self.subTest(attack=kind):
                key = f"{kind}-global-trigger-location"
                self.assertEqual(config_manager.CONFIG_SCHEMA[key]["default"], "")
                with patch.object(config_manager, "_read_toml", return_value={}):
                    ui_config = config_manager.get_config()
                cfg = ExperimentConfig.from_run_config(ui_config)
                attack = build_attack(dataset_meta=MNISTDataset.meta,
                                      **{**cfg.to_attack_kwargs(), "attack_type": kind})
                self.assertEqual(attack.global_trigger_location, (24, 24))

    def test_explicit_location_reaches_both_attack_factories(self):
        for kind in ("dba", "fcba"):
            for raw in ("[2, 3]", "(2,3)", [2, 3], (2, 3)):
                with self.subTest(attack=kind, raw=raw):
                    cfg = ExperimentConfig.from_run_config(
                        {"attack": kind, f"{kind}-global-trigger-location": raw})
                    attack = build_attack(dataset_meta=MNISTDataset.meta, **cfg.to_attack_kwargs())
                    self.assertEqual(attack.global_trigger_location, (2, 3))

    def test_empty_location_means_automatic(self):
        for kind in ("dba", "fcba"):
            for raw in (None, "", "   "):
                with self.subTest(attack=kind, raw=raw):
                    cfg = ExperimentConfig.from_run_config(
                        {"attack": kind, f"{kind}-global-trigger-location": raw})
                    attack = build_attack(dataset_meta=MNISTDataset.meta, **cfg.to_attack_kwargs())
                    self.assertEqual(attack.global_trigger_location, (24, 24))

    def test_invalid_location_text_does_not_silently_become_default(self):
        for kind in ("dba", "fcba"):
            for raw in ("garbage", "[1]", "[1,2,3]", "[1.5,2]", [1.5, 2]):
                with self.subTest(attack=kind, raw=raw):
                    with self.assertRaisesRegex(ValueError, "location|tuple|coordinates"):
                        ExperimentConfig.from_run_config(
                            {"attack": kind, f"{kind}-global-trigger-location": raw})


if __name__ == "__main__":
    unittest.main()