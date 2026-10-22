"""Admin-only platform shutdown requires the current password."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main
from backend.identity import IdentityStore


PASSWORD = "Admin-Secure-2026"


class AdminShutdownTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = IdentityStore(Path(directory.name) / "identity.sqlite3")
        self.store.initialize()
        self.store.force_reset_password(self.store.admin_uid, PASSWORD)
        self.store.create_user("alice", PASSWORD)
        replacement = patch("backend.auth.get_identity", return_value=self.store)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.client = TestClient(main.app, headers={"X-Lianan-Request": "1"}, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def login(self, username="admin"):
        response = self.client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
        self.assertEqual(response.status_code, 200, response.text)
        self.client.headers["X-CSRF-Token"] = response.json()["csrf_token"]

    def test_wrong_password_does_not_schedule_shutdown(self):
        self.login()
        with patch("backend.main.schedule_shutdown") as schedule:
            response = self.client.post("/api/admin/shutdown", json={"password": "wrong-password"})
        self.assertEqual(response.status_code, 403)
        self.assertIn("不正确", response.json()["detail"])
        schedule.assert_not_called()

    def test_missing_password_is_rejected(self):
        self.login()
        with patch("backend.main.schedule_shutdown") as schedule:
            response = self.client.post("/api/admin/shutdown", json={})
        self.assertEqual(response.status_code, 422)
        schedule.assert_not_called()

    def test_correct_password_schedules_shutdown(self):
        self.login()
        with patch("backend.main.schedule_shutdown") as schedule:
            response = self.client.post("/api/admin/shutdown", json={"password": PASSWORD})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "stopping")
        schedule.assert_called_once_with()

    def test_regular_user_cannot_stop_the_platform(self):
        self.login("alice")
        with patch("backend.main.schedule_shutdown") as schedule:
            response = self.client.post("/api/admin/shutdown", json={"password": PASSWORD})
        self.assertEqual(response.status_code, 403)
        schedule.assert_not_called()

    def test_verify_user_password_does_not_create_a_session(self):
        self.assertTrue(self.store.verify_user_password(self.store.admin_uid, PASSWORD))
        self.assertFalse(self.store.verify_user_password(self.store.admin_uid, "wrong-password"))


if __name__ == "__main__":
    unittest.main()
