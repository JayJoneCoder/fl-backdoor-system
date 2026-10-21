"""Authenticate legacy API regression fixtures without changing their file semantics.

Use only with tests that already replace config/result roots. New identity tests
exercise the real boundary and private configuration paths without this adapter.
"""

from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.identity import DEFAULT_ADMIN_PASSWORD, IdentityStore


def authenticated_session(case):
    """Create a disposable identity store and return the (token, csrf) pair."""
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    store = IdentityStore(Path(directory.name) / "identity.sqlite3")
    store.initialize()
    token, csrf, _, _ = store.login("admin", DEFAULT_ADMIN_PASSWORD)
    replacement = patch("backend.auth.get_identity", return_value=store)
    replacement.start()
    case.addCleanup(replacement.stop)
    # Retain existing regression fixture expectations about the active TOML and
    # backup locations; private per-user paths are covered in test_identity.py.
    from backend import config_manager as cm
    for name, value in (("config_path", lambda: cm.TOML_PATH),
                        ("_backup_project_root", lambda: cm.PROJECT_ROOT),
                        ("CURRENT_USER", type("NoUserContext", (), {"get": staticmethod(lambda: None)})())):
        replacement = patch.object(cm, name, value)
        replacement.start()
        case.addCleanup(replacement.stop)
    return token, csrf


def authenticated_client(case, app, **kwargs):
    token, csrf = authenticated_session(case)
    return TestClient(app, headers={"X-CSRF-Token": csrf, "X-Lianan-Request": "1"},
                      cookies={"lianan_session": token}, **kwargs)