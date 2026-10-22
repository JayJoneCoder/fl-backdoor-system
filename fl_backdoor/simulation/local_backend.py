"""Ray-free local backend for Flower's virtual client engine.

Flower 1.29's simulation engine ships only a Ray backend and Ray is officially
experimental on Windows. The desktop build registers this backend instead so the
platform never starts a GCS/raylet process.

Semantics intentionally mirror ``ClientAppActor.run``:

* ``LoadClientAppError`` is propagated unchanged;
* every other exception becomes ``ClientAppException`` so the VCE replies with
  an Error message instead of crashing;
* the caller's global RNG state is restored afterwards. Clients call
  ``seed_everything`` per run, and without this the server's own random choices
  (client sampling, evaluation sampling) would change because both now live in
  one process.
"""

from collections.abc import Callable
import random

from flwr.clientapp.client_app import ClientApp, ClientAppException, LoadClientAppError
from flwr.common import Context, Message
from flwr.server.superlink.fleet.vce.backend.backend import Backend, BackendConfig


BACKEND_NAME = "local"


def _capture_rng() -> dict:
    state: dict = {"random": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:  # pragma: no cover - numpy is always bundled
        pass
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["random"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch" in state:
        import torch

        torch.set_rng_state(state["torch"])
        if "torch_cuda" in state:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            except Exception:
                pass


class LocalBackend(Backend):
    """Execute ClientApps sequentially inside the simulation process."""

    def __init__(self, backend_config: BackendConfig) -> None:
        self._app_fn: Callable[[], ClientApp] | None = None
        self._processed = 0

    def build(self, app_fn: Callable[[], ClientApp]) -> None:
        self._app_fn = app_fn
        print("[Simulation] local backend active: workers=1, Ray disabled", flush=True)

    @property
    def num_workers(self) -> int:
        return 1

    def is_worker_idle(self) -> bool:
        return True

    def terminate(self) -> None:
        self._app_fn = None

    def process_message(
        self,
        message: Message,
        context: Context,
    ) -> tuple[Message, Context]:
        if self._app_fn is None:
            raise ValueError("LocalBackend.build() must run before processing messages.")
        rng = _capture_rng()
        try:
            app = self._app_fn()
            out_message = app(message=message, context=context)
        except LoadClientAppError:
            raise
        except Exception as exc:
            raise ClientAppException(str(exc)) from exc
        finally:
            _restore_rng(rng)
        self._processed += 1
        return out_message, context


def register() -> None:
    """Register ``local`` with Flower's backend registry (idempotent)."""
    from flwr.server.superlink.fleet.vce.backend import supported_backends

    supported_backends[BACKEND_NAME] = LocalBackend
    print(f"[Simulation] registered backend '{BACKEND_NAME}'", flush=True)
