"""Frequency attack regressions with synthetic CPU images, no data downloads."""

import io
import math
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from fl_backdoor.attacks import build_attack
from fl_backdoor.attacks.frequency import (
    _apply_frequency_trigger,
    _dct2,
    _idct2,
    build_frequency_attack,
    get_poisoned_loader,
    get_triggered_loader,
)
from fl_backdoor.dataset.cifar10 import CIFAR10Dataset
from fl_backdoor.dataset.mnist import MNISTDataset
from fl_backdoor.server.server import evaluate_asr
from fl_backdoor.task import Net, test as evaluate_model, train


METAS = (MNISTDataset.meta, CIFAR10Dataset.meta)
MODES = ("fft", "dct")


def sample_for(meta, *, include_raw=True, label=1, dtype=torch.float32):
    raw = torch.linspace(0, 1, math.prod(meta.input_shape), dtype=dtype).reshape(meta.input_shape)
    mean = raw.new_tensor(meta.mean).view(-1, 1, 1)
    std = raw.new_tensor(meta.std).view(-1, 1, 1)
    sample = {"img": (raw - mean) / std, "label": label}
    if include_raw:
        sample["img_raw"] = raw
    return sample


def basis(size):
    """Independent orthonormal DCT-II definition, not production matrices."""
    return torch.tensor([
        [math.sqrt((1 if k == 0 else 2) / size)
         * math.cos(math.pi * (i + 0.5) * k / size) for i in range(size)]
        for k in range(size)
    ], dtype=torch.float64)


class FrequencyAttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)

    def setUp(self):
        state = torch.random.fork_rng(devices=[])
        state.__enter__()
        self.addCleanup(state.__exit__, None, None, None)
        torch.manual_seed(42)

    def attack(self, meta, mode="fft", **kwargs):
        options = dict(dataset_meta=meta, frequency_mode=mode, frequency_band="high",
                       frequency_window_size=6, frequency_intensity=0.35,
                       poison_rate=1.0, target_label=0)
        options.update(kwargs)
        with redirect_stdout(io.StringIO()):
            return build_attack("frequency", **options)

    def apply(self, image, mode="fft", **kwargs):
        options = dict(mode=mode, band="high", window=6, intensity=0.35,
                       seed=42, mix_alpha=1.0, trigger_style="structured", input_domain="raw")
        options.update(kwargs)
        return _apply_frequency_trigger(image, **options)

    def test_zero_poison_rate_preserves_every_sample_exactly(self):
        for meta in METAS:
            for mode in MODES:
                for raw in (True, False):
                    with self.subTest(dataset=meta.name, mode=mode, raw=raw):
                        sample = sample_for(meta, include_raw=raw)
                        attack = self.attack(meta, mode, poison_rate=0)
                        result = attack.get_poisoned_loader(DataLoader([sample])).dataset[0]
                        self.assertTrue(torch.equal(result["img"], sample["img"]))
                        self.assertEqual(int(result["label"]), 1)

    def test_excluded_target_samples_remain_unchanged(self):
        for meta in METAS:
            for mode in MODES:
                with self.subTest(dataset=meta.name, mode=mode):
                    sample = sample_for(meta, label=0)
                    result = self.attack(meta, mode).get_poisoned_loader(DataLoader([sample])).dataset[0]
                    self.assertTrue(torch.equal(result["img"], sample["img"]))
                    self.assertEqual(int(result["label"]), 0)

    def test_partial_poisoning_changes_only_selected_non_target_samples(self):
        for meta in METAS:
            with self.subTest(dataset=meta.name):
                samples = [sample_for(meta) for _ in range(16)]
                dataset = self.attack(meta, poison_rate=0.5).get_poisoned_loader(DataLoader(samples)).dataset
                self.assertTrue(dataset.poison_flags.any())
                self.assertTrue((~dataset.poison_flags).any())
                for i, selected in enumerate(dataset.poison_flags):
                    result = dataset[i]
                    self.assertEqual(int(result["label"]), 0 if selected else 1)
                    self.assertEqual(torch.equal(result["img"], samples[i]["img"]), not selected)

    def test_zero_strength_preserves_images_for_training_and_asr(self):
        for meta in METAS:
            for mode in MODES:
                for raw in (True, False):
                    with self.subTest(dataset=meta.name, mode=mode, raw=raw):
                        sample = sample_for(meta, include_raw=raw)
                        attack = self.attack(meta, mode, frequency_intensity=0)
                        loader = DataLoader([sample])
                        poisoned = attack.get_poisoned_loader(loader).dataset[0]
                        triggered = attack.get_triggered_loader(loader).dataset[0]
                        self.assertTrue(torch.equal(poisoned["img"], sample["img"]))
                        self.assertTrue(torch.equal(triggered["img"], sample["img"]))
                        # Label poisoning is separate from trigger strength.
                        self.assertEqual(int(poisoned["label"]), 0)
                        self.assertEqual(int(triggered["label"]), 1)

    def test_raw_trigger_is_normalized_with_the_actual_dataset_stats(self):
        for meta in METAS:
            for mode in MODES:
                for band in ("low", "high"):
                    with self.subTest(dataset=meta.name, mode=mode, band=band):
                        sample = sample_for(meta)
                        raw = self.apply(sample["img_raw"], mode, band=band)
                        mean = raw.new_tensor(meta.mean).view(-1, 1, 1)
                        std = raw.new_tensor(meta.std).view(-1, 1, 1)
                        result = self.attack(meta, mode, frequency_band=band).get_triggered_loader(DataLoader([sample])).dataset[0]
                        torch.testing.assert_close(result["img"], (raw - mean) / std)

    def test_normalized_only_samples_match_raw_domain_samples(self):
        for meta in METAS:
            for mode in MODES:
                for band in ("low", "high"):
                    with self.subTest(dataset=meta.name, mode=mode, band=band):
                        sample = sample_for(meta)
                        normalized_only = {key: value for key, value in sample.items() if key != "img_raw"}
                        attack = self.attack(meta, mode, frequency_band=band)
                        for method in (attack.get_poisoned_loader, attack.get_triggered_loader):
                            with_raw = method(DataLoader([sample])).dataset[0]
                            without_raw = method(DataLoader([normalized_only])).dataset[0]
                            torch.testing.assert_close(with_raw["img"], without_raw["img"], atol=2e-6, rtol=1e-5)
                            self.assertEqual(int(with_raw["label"]), int(without_raw["label"]))

    def test_training_and_asr_share_the_same_trigger_but_not_labels(self):
        for meta in METAS:
            for mode in MODES:
                for style in ("structured", "random"):
                    with self.subTest(dataset=meta.name, mode=mode, style=style):
                        with redirect_stdout(io.StringIO()):
                            attack = build_frequency_attack(dataset_meta=meta, frequency_mode=mode,
                                                            trigger_style=style, poison_rate=1)
                        loader = DataLoader([sample_for(meta)])
                        poisoned = attack.get_poisoned_loader(loader).dataset[0]
                        triggered = attack.get_triggered_loader(loader).dataset[0]
                        self.assertTrue(torch.equal(poisoned["img"], triggered["img"]))
                        self.assertEqual(int(poisoned["label"]), 0)
                        self.assertEqual(int(triggered["label"]), 1)

    def test_wrappers_do_not_mutate_base_images_or_tensor_labels(self):
        for meta in METAS:
            for mode in MODES:
                with self.subTest(dataset=meta.name, mode=mode):
                    sample = sample_for(meta, label=torch.tensor(1))
                    original = {key: value.clone() for key, value in sample.items()}
                    attack = self.attack(meta, mode)
                    for method in (attack.get_poisoned_loader, attack.get_triggered_loader):
                        result = method(DataLoader([sample])).dataset[0]
                        result["img"].zero_()
                        result["label"].fill_(9)
                        for key, value in sample.items():
                            self.assertTrue(torch.equal(value, original[key]), msg=key)

    def test_mnist_model_values_above_one_are_not_clipped(self):
        meta = MNISTDataset.meta
        sample = sample_for(meta)
        for mode in MODES:
            for raw in (True, False):
                with self.subTest(mode=mode, raw=raw):
                    base = sample if raw else {k: v for k, v in sample.items() if k != "img_raw"}
                    result = self.attack(meta, mode).get_triggered_loader(DataLoader([base])).dataset[0]["img"]
                    self.assertGreater(float(result.max()), 2)
                    reconstructed = result * meta.std[0] + meta.mean[0]
                    self.assertGreaterEqual(float(reconstructed.min()), -1e-6)
                    self.assertLessEqual(float(reconstructed.max()), 1 + 1e-6)

    def test_per_channel_stats_and_float64_are_respected(self):
        meta = replace(CIFAR10Dataset.meta, mean=(0.1, 0.2, 0.3), std=(0.25, 0.4, 0.6))
        for mode in MODES:
            with self.subTest(mode=mode):
                sample = sample_for(meta, dtype=torch.float64)
                raw = self.apply(sample["img_raw"], mode)
                expected = (raw - raw.new_tensor(meta.mean).view(-1, 1, 1)) / raw.new_tensor(meta.std).view(-1, 1, 1)
                result = self.attack(meta, mode).get_triggered_loader(DataLoader([sample])).dataset[0]["img"]
                self.assertEqual(result.dtype, torch.float64)
                torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)

    def test_direct_loader_helpers_accept_dataset_metadata(self):
        meta = MNISTDataset.meta
        sample = sample_for(meta)
        options = dict(dataset_meta=meta, frequency_mode="fft", frequency_band="high",
                       frequency_window_size=6, frequency_intensity=0)
        loader = DataLoader([sample])
        poisoned = get_poisoned_loader(loader, poison_rate=0, target_label=0, **options)
        triggered = get_triggered_loader(loader, **options)
        self.assertTrue(torch.equal(poisoned.dataset[0]["img"], sample["img"]))
        self.assertTrue(torch.equal(triggered.dataset[0]["img"], sample["img"]))

    def test_loaders_preserve_batch_shape_and_settings(self):
        for meta in METAS:
            loader = DataLoader([sample_for(meta)] * 5, batch_size=2, drop_last=True)
            for method in (self.attack(meta).get_poisoned_loader, self.attack(meta).get_triggered_loader):
                result = method(loader)
                self.assertEqual(result.batch_size, 2)
                self.assertTrue(result.drop_last)
                self.assertEqual(len(result.dataset), 5)
                self.assertEqual(tuple(next(iter(result))["img"].shape), (2, *meta.input_shape))

    def test_legacy_callers_without_metadata_keep_cifar_normalization(self):
        sample = sample_for(CIFAR10Dataset.meta)
        attack = self.attack(None)
        raw = self.apply(sample["img_raw"])
        result = attack.get_triggered_loader(DataLoader([sample])).dataset[0]["img"]
        torch.testing.assert_close(result, (raw - 0.5) / 0.5)

    def test_dct_roundtrip_matches_original_for_square_rectangular_and_2d_images(self):
        for shape in ((28, 28), (1, 28, 28), (3, 32, 32), (2, 19, 28)):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(shape=shape, dtype=dtype):
                    image = torch.rand(shape, dtype=dtype)
                    original = image.clone()
                    restored = _idct2(_dct2(image))
                    self.assertEqual(restored.dtype, image.dtype)
                    torch.testing.assert_close(restored, image, atol=2e-6, rtol=2e-6)
                    self.assertTrue(torch.equal(image, original))

    def test_dct_coefficients_match_independent_definition(self):
        image = torch.rand((2, 7, 11), dtype=torch.float64)
        height, width = basis(7), basis(11)
        expected = height @ image @ width.T
        torch.testing.assert_close(_dct2(image), expected, atol=1e-12, rtol=1e-12)

    def test_idct_matches_independent_inverse_definition(self):
        coefficients = torch.rand((2, 7, 11), dtype=torch.float64)
        expected = basis(7).T @ coefficients @ basis(11)
        torch.testing.assert_close(_idct2(coefficients), expected, atol=1e-12, rtol=1e-12)

    def test_constant_image_has_only_dc_coefficient(self):
        coefficients = _dct2(torch.ones((1, 8, 12)))
        self.assertAlmostEqual(float(coefficients[0, 0, 0]), math.sqrt(96), places=5)
        coefficients[0, 0, 0] = 0
        self.assertLess(float(coefficients.abs().max()), 2e-6)

    def test_zero_strength_core_trigger_is_exact_identity(self):
        for meta in METAS:
            sample = sample_for(meta)
            for mode in MODES:
                for domain, key in (("raw", "img_raw"), ("normalized", "img")):
                    with self.subTest(dataset=meta.name, mode=mode, domain=domain):
                        output = self.apply(sample[key], mode, input_domain=domain, dataset_meta=meta, intensity=0)
                        self.assertTrue(torch.equal(output, sample[key]))
                        self.assertNotEqual(output.data_ptr(), sample[key].data_ptr())

    def test_core_trigger_preserves_shape_dtype_and_device(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for shape in ((17, 21), (1, 28, 28), (3, 32, 32)):
                for mode in MODES:
                    with self.subTest(dtype=dtype, shape=shape, mode=mode):
                        image = torch.full(shape, 0.5, dtype=dtype)
                        result = self.apply(image, mode)
                        self.assertEqual(result.shape, image.shape)
                        self.assertEqual(result.dtype, image.dtype)
                        self.assertEqual(result.device, image.device)
                        self.assertTrue(torch.isfinite(result).all())

    def test_trigger_is_repeatable_and_not_in_place(self):
        image = torch.full((1, 28, 28), 0.5)
        for mode in MODES:
            for style in ("structured", "random"):
                with self.subTest(mode=mode, style=style):
                    first = self.apply(image, mode, trigger_style=style)
                    second = self.apply(image, mode, trigger_style=style)
                    self.assertTrue(torch.equal(first, second))
                    self.assertGreater(float((first - image).abs().max()), 1e-4)
                    self.assertTrue(torch.all(image == 0.5))

    def test_invalid_normalization_metadata_fails_clearly(self):
        for meta in (replace(MNISTDataset.meta, std=(0,)),
                     replace(MNISTDataset.meta, mean=(0.1, 0.2)),
                     replace(MNISTDataset.meta, std=(float("nan"),))):
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                self.attack(meta).get_triggered_loader(DataLoader([sample_for(MNISTDataset.meta)])).dataset[0]

    def test_nonfinite_intensity_is_rejected(self):
        for intensity in (float("nan"), float("inf"), -0.1):
            with self.subTest(intensity=intensity), self.assertRaises(ValueError):
                self.attack(MNISTDataset.meta, frequency_intensity=intensity)

    def test_invalid_input_domain_is_rejected(self):
        with self.assertRaises(ValueError):
            self.apply(torch.zeros(1, 28, 28), input_domain="unknown")

    def test_cnn_can_train_and_evaluate_with_each_dataset_and_mode(self):
        for meta in METAS:
            for mode in MODES:
                with self.subTest(dataset=meta.name, mode=mode):
                    model = Net(input_shape=meta.input_shape, num_classes=meta.num_classes)
                    loader = DataLoader([sample_for(meta, label=1), sample_for(meta, label=2)], batch_size=2)
                    attack = self.attack(meta, mode)
                    loss = train(model, attack.get_poisoned_loader(loader), epochs=1, lr=0.001, device="cpu")
                    clean_loss, accuracy = evaluate_model(model, loader, "cpu")
                    with redirect_stdout(io.StringIO()):
                        asr = evaluate_asr(model, attack.get_triggered_loader(loader), "cpu", target_label=0)
                    self.assertTrue(math.isfinite(loss))
                    self.assertTrue(math.isfinite(clean_loss))
                    self.assertTrue(0 <= accuracy <= 1)
                    self.assertTrue(0 <= asr <= 1)


if __name__ == "__main__":
    unittest.main()