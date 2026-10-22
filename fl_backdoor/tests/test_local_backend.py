"""Local (Ray-free) simulation backend semantics."""

import random
import unittest

import torch
from flwr.clientapp.client_app import ClientAppException, LoadClientAppError

from fl_backdoor.simulation.local_backend import LocalBackend, register


class FakeApp:
    def __init__(self, action=None):
        self.action = action

    def __call__(self, message=None, context=None):
        if self.action:
            self.action()
        return f"reply:{message}"


class LocalBackendTests(unittest.TestCase):
    def make(self, app):
        backend = LocalBackend({"client_resources": {}, "init_args": {}})
        backend.build(lambda: app)
        return backend

    def test_process_message_calls_client_app_and_returns_context(self):
        backend = self.make(FakeApp())
        out, context = backend.process_message("m1", {"node": 0})
        self.assertEqual(out, "reply:m1")
        self.assertEqual(context, {"node": 0})
        self.assertEqual(backend.num_workers, 1)
        self.assertTrue(backend.is_worker_idle())
        backend.terminate()
        with self.assertRaises(ValueError):
            backend.process_message("m2", {})

    def test_exceptions_are_wrapped_like_ray_actors(self):
        def boom():
            raise ValueError("bad client")

        with self.assertRaises(ClientAppException) as caught:
            self.make(FakeApp(boom)).process_message("m", {})
        self.assertIn("bad client", str(caught.exception))

    def test_load_errors_propagate_unchanged(self):
        def boom():
            raise LoadClientAppError("cannot load")

        with self.assertRaises(LoadClientAppError):
            self.make(FakeApp(boom)).process_message("m", {})

    def test_global_random_state_is_restored_after_a_client_run(self):
        random.seed(1234)
        expected = random.random()
        random.seed(1234)

        def reseed():
            random.seed(999)
            random.random()

        self.make(FakeApp(reseed)).process_message("m", {})
        self.assertEqual(random.random(), expected)

    def test_torch_rng_state_is_restored_after_a_client_run(self):
        torch.manual_seed(7)
        expected = torch.rand(1).item()
        torch.manual_seed(7)

        def reseed():
            torch.manual_seed(999)
            torch.rand(1)

        self.make(FakeApp(reseed)).process_message("m", {})
        self.assertEqual(torch.rand(1).item(), expected)

    def test_register_exposes_the_backend_to_flower(self):
        from flwr.server.superlink.fleet.vce.backend import supported_backends

        register()
        self.assertIs(supported_backends["local"], LocalBackend)


if __name__ == "__main__":
    unittest.main()
