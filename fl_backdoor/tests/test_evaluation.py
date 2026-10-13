"""Regression tests for evaluation mode using CPU-only synthetic data."""

import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_backdoor.server.server import evaluate_asr
from fl_backdoor.task import Net, test as evaluate_model, train


class ProbeClassifier(nn.Module):
    """Record the mode of every layer and gradient tracking during inference."""

    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(3, 4))
        self.forward_states = []
        self.fail_on_forward = None

    def forward(self, images):
        self.forward_states.append(
            (tuple(module.training for module in self.modules()), torch.is_grad_enabled())
        )
        if len(self.forward_states) == self.fail_on_forward:
            raise RuntimeError("synthetic forward failure")
        return self.classifier(self.dropout(self.batch_norm(images)))


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def setUp(self):
        random_state = torch.random.fork_rng(devices=[])
        random_state.__enter__()
        self.addCleanup(random_state.__exit__, None, None, None)
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.model = ProbeClassifier()
        images = torch.tensor(
            [[6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0],
             [6.0, 0.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0],
             [6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0]]
        )
        labels = torch.tensor([0, 1, 2, 2, 1, 0])
        self.loader = DataLoader(
            [{"img": image, "label": label} for image, label in zip(images, labels)],
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

    def snapshot(self, model):
        return {
            "state": {name: value.detach().clone() for name, value in model.state_dict().items()},
            "modes": tuple(module.training for module in model.modules()),
            "gradients": {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            },
            "requires_grad": tuple(parameter.requires_grad for parameter in model.parameters()),
        }

    def assert_model_unchanged(self, model, before):
        self.assertEqual(tuple(module.training for module in model.modules()), before["modes"])
        self.assertEqual(
            tuple(parameter.requires_grad for parameter in model.parameters()),
            before["requires_grad"],
        )
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before["state"][name]), msg=name)
        for name, parameter in model.named_parameters():
            expected = before["gradients"][name]
            if expected is None:
                self.assertIsNone(parameter.grad, msg=name)
            else:
                self.assertIsNotNone(parameter.grad, msg=name)
                self.assertTrue(torch.equal(parameter.grad, expected), msg=name)

    def assert_evaluation_forwards(self, model):
        self.assertTrue(model.forward_states)
        for modes, grad_enabled in model.forward_states:
            self.assertFalse(any(modes), msg="a layer was still in training mode")
            self.assertFalse(grad_enabled, msg="evaluation was tracking gradients")

    def reference_metrics(self):
        reference = copy.deepcopy(self.model).eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for batch in self.loader:
                outputs = reference(batch["img"])
                loss += nn.functional.cross_entropy(outputs, batch["label"]).item()
                correct += (outputs.argmax(dim=1) == batch["label"]).sum().item()
        return loss / len(self.loader), correct / len(self.loader.dataset)

    def test_clean_evaluation_uses_eval_without_changing_model_state(self):
        expected_loss, expected_accuracy = self.reference_metrics()
        before = self.snapshot(self.model)

        loss, accuracy = evaluate_model(self.model, self.loader, self.device)

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)
        self.assertAlmostEqual(loss, expected_loss)
        self.assertEqual(accuracy, expected_accuracy)

    def test_clean_evaluation_preserves_existing_eval_mode(self):
        self.model.eval()
        before = self.snapshot(self.model)

        evaluate_model(self.model, self.loader, self.device)

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_clean_evaluation_restores_individual_layer_modes(self):
        self.model.eval()
        self.model.dropout.train()
        before = self.snapshot(self.model)

        evaluate_model(self.model, self.loader, self.device)

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_repeated_clean_evaluations_are_identical(self):
        before = self.snapshot(self.model)

        first = evaluate_model(self.model, self.loader, self.device)
        second = evaluate_model(self.model, self.loader, self.device)

        self.assertEqual(first, second)
        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_clean_evaluation_preserves_existing_gradients(self):
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        self.model.classifier.weight.requires_grad_(False)
        before = self.snapshot(self.model)

        evaluate_model(self.model, self.loader, self.device)

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_clean_forward_failure_restores_modes_and_gradient_tracking(self):
        self.model.batch_norm.eval()
        self.model.fail_on_forward = 2
        before = self.snapshot(self.model)

        with torch.enable_grad():
            with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
                evaluate_model(self.model, self.loader, self.device)
            self.assertTrue(torch.is_grad_enabled())

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_clean_loader_failure_restores_modes(self):
        before = self.snapshot(self.model)

        def broken_loader():
            yield next(iter(self.loader))
            raise RuntimeError("synthetic loader failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
            evaluate_model(self.model, broken_loader(), self.device)

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_asr_preserves_state_and_excludes_original_target_labels(self):
        before = self.snapshot(self.model)

        asr = evaluate_asr(self.model, self.loader, self.device, target_label=0)

        # Two of the four non-target samples predict the target label.
        self.assertEqual(asr, 0.5)
        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_asr_preserves_existing_eval_mode(self):
        self.model.eval()
        before = self.snapshot(self.model)

        evaluate_asr(self.model, self.loader, self.device, target_label=0)

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_asr_restores_individual_layer_modes(self):
        self.model.batch_norm.eval()
        before = self.snapshot(self.model)

        evaluate_asr(self.model, self.loader, self.device, target_label=0)

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_asr_forward_failure_restores_modes_and_gradient_tracking(self):
        self.model.batch_norm.eval()
        self.model.fail_on_forward = 2
        before = self.snapshot(self.model)

        with patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            with torch.enable_grad():
                with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
                    evaluate_asr(self.model, self.loader, self.device, target_label=0)
                self.assertTrue(torch.is_grad_enabled())

        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_asr_without_non_target_samples_returns_zero_and_restores_modes(self):
        target_batch = [{"img": torch.ones(2, 4), "label": torch.zeros(2, dtype=torch.long)}]
        before = self.snapshot(self.model)

        for batches in ([], target_batch):
            with self.subTest(batches=len(batches)):
                self.assertEqual(evaluate_asr(self.model, batches, self.device, target_label=0), 0.0)
                self.assert_model_unchanged(self.model, before)
        self.assertEqual(self.model.forward_states, [])

    def test_clean_evaluation_does_not_contaminate_following_asr(self):
        before = self.snapshot(self.model)
        original_asr = evaluate_asr(self.model, self.loader, self.device, target_label=0)

        evaluate_model(self.model, self.loader, self.device)
        following_asr = evaluate_asr(self.model, self.loader, self.device, target_label=0)

        self.assertEqual(original_asr, following_asr)
        self.assert_evaluation_forwards(self.model)
        self.assert_model_unchanged(self.model, before)

    def test_actual_cnn_is_unchanged_for_mnist_and_cifar10_evaluations(self):
        for input_shape in ((1, 28, 28), (3, 32, 32)):
            with self.subTest(input_shape=input_shape):
                model = Net(input_shape=input_shape, num_classes=10)
                images = torch.randn(4, *input_shape)
                labels = torch.tensor([0, 1, 2, 3])
                loader = DataLoader(
                    [{"img": image, "label": label} for image, label in zip(images, labels)],
                    batch_size=2,
                    shuffle=False,
                    num_workers=0,
                )
                before = self.snapshot(model)
                first_metrics = evaluate_model(model, loader, self.device)
                first_asr = evaluate_asr(model, loader, self.device, target_label=0)

                self.assertEqual(evaluate_model(model, loader, self.device), first_metrics)
                self.assertEqual(evaluate_asr(model, loader, self.device, target_label=0), first_asr)
                self.assert_model_unchanged(model, before)

    def test_training_still_updates_model_after_evaluation(self):
        self.model.eval()
        evaluate_model(self.model, self.loader, self.device)
        before = self.snapshot(self.model)
        self.model.forward_states.clear()

        loss = train(self.model, self.loader, epochs=1, lr=0.01, device=self.device)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(self.model.training)
        self.assertTrue(all(all(modes) and grad for modes, grad in self.model.forward_states))
        self.assertEqual(self.model.batch_norm.num_batches_tracked.item(), len(self.loader))
        self.assertTrue(any(
            not torch.equal(value, before["state"][name])
            for name, value in self.model.named_parameters()
        ))


if __name__ == "__main__":
    unittest.main()