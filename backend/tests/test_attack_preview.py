"""Read-only, forced trigger previews; no downloads or training required."""

import base64
import io
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from backend.tests.auth_support import authenticated_client
from PIL import Image

from backend import attack_preview as preview
from backend.main import app
from fl_backdoor.attacks import build_attack
from fl_backdoor.config import ExperimentConfig
from fl_backdoor.dataset import get_dataset
from torch.utils.data import DataLoader


def sample_dataset(name):
    size = 28 if name == "mnist" else 32
    pixels = np.arange(size * size, dtype=np.int64).reshape(size, size) % 180
    if name == "cifar10":
        pixels = np.stack([pixels, pixels // 2, 180 - pixels], axis=-1)
    image = Image.fromarray(pixels.astype(np.uint8))
    return [(image, 0), (image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), 2)]


def decode(url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))


class AttackPreviewTests(unittest.TestCase):
    def setUp(self):
        self.real_loader = preview._load_training_dataset
        self.loader = patch.object(preview, "_load_training_dataset", side_effect=lambda name, root: sample_dataset(name))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def request(self, dataset="mnist", attack="badnets", **updates):
        config = {"dataset": dataset, "attack": attack, "seed": 73, "target-label": 0, "poison-rate": 0,
                  "num-clients": 10, "trigger-size": 4, **updates}
        return {"config": config}

    def test_default_first_image_matches_original_pixels_and_label(self):
        for name in ("mnist", "cifar10"):
            with self.subTest(dataset=name):
                result = preview.preview_attack(self.request(dataset=name))
                self.assertEqual(result["index"], 0)
                self.assertEqual(result["total"], 2)
                self.assertEqual(result["label"], 0)
                np.testing.assert_array_equal(decode(result["clean_image"]), np.asarray(sample_dataset(name)[0][0]))

    def test_every_attack_forces_trigger_even_with_zero_rate_and_target_label(self):
        for name in ("mnist", "cifar10"):
            for attack in ("badnets", "wanet", "frequency", "frequency_fft", "dba", "fcba"):
                with self.subTest(dataset=name, attack=attack):
                    result = preview.preview_attack(self.request(name, attack))
                    self.assertGreater(result["changed_pixels"], 0)
                    self.assertFalse(np.array_equal(decode(result["clean_image"]), decode(result["poisoned_image"])))
                    self.assertIn("不代表", result["notice"])

    def test_none_keeps_image_unchanged(self):
        result = preview.preview_attack(self.request(attack="none"))
        self.assertEqual(result["changed_pixels"], 0)
        self.assertEqual(result["clean_image"], result["poisoned_image"])

    def test_can_select_another_dataset_image(self):
        request = self.request()
        first = preview.preview_attack(request)
        request["index"] = 1
        second = preview.preview_attack(request)
        self.assertEqual(second["index"], 1)
        self.assertEqual(second["label"], 2)
        self.assertNotEqual(first["clean_image"], second["clean_image"])

    def test_distributed_local_patterns_always_applied_and_change_by_client(self):
        for attack in ("dba", "fcba"):
            with self.subTest(attack=attack):
                request = {**self.request(attack=attack), "scope": "local", "client_id": 0}
                first = preview.preview_attack(request)
                second = preview.preview_attack({**request, "client_id": 1})
                full = preview.preview_attack({**request, "scope": "global"})
                self.assertGreater(first["changed_pixels"], 0)
                self.assertNotEqual(first["poisoned_image"], second["poisoned_image"])
                self.assertGreater(full["changed_pixels"], first["changed_pixels"])

    def test_full_preview_matches_actual_triggered_loader(self):
        for attack_name in ("badnets", "wanet", "frequency", "dba", "fcba"):
            with self.subTest(attack=attack_name):
                request = self.request(attack=attack_name)
                cfg = ExperimentConfig.from_run_config(request["config"])
                dataset = get_dataset(cfg.dataset)
                image, label = sample_dataset(cfg.dataset)[0]
                sample = dataset._apply_transforms((image, label))
                attack = build_attack(dataset_meta=dataset.meta, **cfg.to_attack_kwargs())
                expected = attack.get_triggered_loader(DataLoader([sample], batch_size=1)).dataset[0]["img"]
                self.assertEqual(preview.preview_attack(request)["poisoned_image"], preview._image_url(expected, dataset.meta))

    def test_distributed_returns_three_images_in_one_request_for_both_scopes(self):
        for name in ("mnist", "cifar10"):
            for attack in ("dba", "fcba"):
                with self.subTest(dataset=name, attack=attack):
                    request = self.request(name, attack)
                    full = preview.preview_attack(request)
                    local = preview.preview_attack({**request, "scope": "local"})
                    self.assertEqual(full["clean_image"], local["clean_image"])
                    self.assertEqual(full["global_image"], local["global_image"])
                    self.assertEqual(full["local_image"], local["local_image"])
                    self.assertEqual(full["poisoned_image"], full["global_image"])
                    self.assertEqual(local["poisoned_image"], local["local_image"])
                    self.assertGreater(local["local_changed_pixels"], 0)
                    self.assertGreater(local["global_changed_pixels"], local["local_changed_pixels"])
                    self.assertNotEqual(local["local_image"], local["global_image"])

    def test_local_image_matches_actual_training_poison_wrapper(self):
        for name in ("mnist", "cifar10"):
            for attack_name in ("dba", "fcba"):
                for strategy in ("grid", "random"):
                    for client_id in (0, 1, 4, 7):
                        with self.subTest(dataset=name, attack=attack_name, strategy=strategy, client=client_id):
                            request = self.request(name, attack_name, **{f"{attack_name}-split-strategy": strategy})
                            request["client_id"] = client_id
                            result = preview.preview_attack(request)
                            cfg = ExperimentConfig.from_run_config(request["config"])
                            dataset = get_dataset(name)
                            image, label = sample_dataset(name)[0]
                            sample = dataset._apply_transforms((image, label))
                            attack = build_attack(dataset_meta=dataset.meta, **cfg.to_attack_kwargs())
                            attack._current_client_id = client_id
                            training = attack.get_poisoned_loader(DataLoader([sample], batch_size=1)).dataset
                            # Force only sample selection, leaving the training trigger untouched.
                            training.poison_flags[:] = True
                            training.exclude_target_label = False
                            expected = training[0]
                            self.assertEqual(int(expected["label"]), cfg.target_label)
                            self.assertEqual(result["local_image"], preview._image_url(expected["img"], dataset.meta))

    def test_fcba_local_preview_includes_multi_block_combinations_and_wraps_client_id(self):
        request = self.request(attack="fcba", **{"num-clients": 15})
        single = preview.preview_attack(request)
        combination = preview.preview_attack({**request, "client_id": 4})
        wrapped = preview.preview_attack({**request, "client_id": 14})
        self.assertEqual(single["local_sub_blocks"], 1)
        self.assertEqual(combination["local_sub_blocks"], 2)
        self.assertEqual(combination["local_trigger_index"], 4)
        self.assertEqual(combination["local_trigger_count"], 14)
        self.assertEqual(combination["local_changed_pixels"], 8)
        self.assertEqual(combination["global_changed_pixels"], 16)
        self.assertNotEqual(single["local_image"], combination["local_image"])
        self.assertEqual(single["global_image"], combination["global_image"])
        self.assertEqual(single["local_image"], wrapped["local_image"])
        self.assertEqual(wrapped["local_trigger_index"], 0)

    def test_other_attacks_keep_two_image_response(self):
        for attack in ("none", "badnets", "wanet", "frequency"):
            with self.subTest(attack=attack):
                result = preview.preview_attack(self.request(attack=attack))
                self.assertIn("poisoned_image", result)
                self.assertNotIn("local_image", result)
                self.assertNotIn("global_image", result)

    def test_unrelated_invalid_training_config_does_not_block_preview(self):
        request = self.request(**{"defense": "krum", "defense-num-malicious": 999, "results-dir": "bad\x00path"})
        self.assertGreater(preview.preview_attack(request)["changed_pixels"], 0)

    def test_preview_does_not_mutate_sample_or_global_random_state(self):
        source = sample_dataset("mnist")
        before = np.array(source[0][0])
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()
        with patch.object(preview, "_load_training_dataset", return_value=source):
            for attack in ("badnets", "wanet", "frequency", "dba", "fcba"):
                preview.preview_attack(self.request(attack=attack))
        np.testing.assert_array_equal(before, np.asarray(source[0][0]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])

    def test_invalid_requests_are_rejected(self):
        for update in ({"index": -1}, {"index": 2}, {"index": 1.5}, {"client_id": 10},
                       {"scope": "invalid"}, {"config": []}, {"config": {"dataset": "cifar100"}},
                       {"config": {"trigger-size": 99}}, {"config": {"attack": "fcba", "fcba-num-sub-blocks": 20}}):
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    preview.preview_attack({**self.request(), **update})

    def test_api_returns_images_without_writing_configuration_or_starting_experiment(self):
        with patch("backend.main.runner.start") as start, patch("backend.config_manager.update_config") as save:
            response = authenticated_client(self, app).post("/api/attack/preview", json=self.request())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["poisoned_image"].startswith("data:image/png;base64,"))
        start.assert_not_called()
        save.assert_not_called()

    def test_api_localizes_bad_input_and_missing_dataset(self):
        client = authenticated_client(self, app)
        response = client.post("/api/attack/preview", json={**self.request(), "index": -1})
        self.assertEqual(response.status_code, 400)
        self.assertIn("非负整数", response.json()["detail"])
        with patch.object(preview, "_load_training_dataset", side_effect=FileNotFoundError("本地未找到完整数据集")):
            response = client.post("/api/attack/preview", json=self.request())
        self.assertEqual(response.status_code, 404)
        self.assertIn("本地未找到", response.json()["detail"])

    def test_loader_never_downloads(self):
        with patch.object(preview.datasets, "MNIST", side_effect=RuntimeError("not found")) as loader:
            with self.assertRaises(FileNotFoundError):
                self.real_loader.__wrapped__("mnist", "missing-data")
        self.assertFalse(loader.call_args.kwargs["download"])