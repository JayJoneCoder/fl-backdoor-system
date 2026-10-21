"""Real HTTP/WS identity boundaries; every database/artifact lives in a temp dir."""

import asyncio
import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import auth, config_manager as cm, main
from backend.batch_manager import BatchManager
from backend.catalog import Catalog
from backend.identity import CURRENT_STORE, CURRENT_USER, DEFAULT_ADMIN_PASSWORD, IdentityStore
from backend.tests.test_result_protection import configuration
from fl_backdoor.utils.run_artifacts import reserve_experiment

PASSWORD = "Research-Secure-2026"


class IdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "results"
        self.root.mkdir()
        self.config = self.base / "pyproject.toml"
        self.config.write_bytes(configuration(**{"results-dir": str(self.root)}))
        self.original = self.config.read_bytes()
        self.store = IdentityStore(self.base / "state" / "identity.sqlite3")
        self.store.initialize()
        self.manager = BatchManager(self.root / "batches")
        self.catalog = Catalog(self.base / "catalog.sqlite3")
        replacements = [patch.object(auth, "get_identity", return_value=self.store),
                        patch.object(main, "get_identity", return_value=self.store),
                        patch.object(main, "get_catalog", return_value=self.catalog),
                        patch.object(main, "batch_manager", self.manager),
                        patch.object(main, "RESULTS_DIR", self.root),
                        patch.object(cm, "PROJECT_ROOT", self.base),
                        patch.object(cm, "TOML_PATH", self.config),
                        patch.object(cm, "BACKUP_PATH", self.base / "pyproject.toml.bak")]
        for replacement in replacements:
            replacement.start(); self.addCleanup(replacement.stop)
        self.guest = self.client()

    def client(self):
        client = TestClient(main.app, headers={"X-Lianan-Request": "1"}, raise_server_exceptions=False)
        self.addCleanup(client.close)
        return client

    def initialize(self):
        # Activate the auto-created admin by resetting its default password to a known one.
        self.store.force_reset_password(self.store.admin_uid, PASSWORD)

    def login(self, username="admin", password=PASSWORD):
        client = self.client()
        response = client.post("/api/auth/login", json={"username": username, "password": password})
        self.assertEqual(response.status_code, 200, response.text)
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
        return client, response.json()["user"]

    def accounts(self):
        self.initialize()
        admin, owner = self.login()
        for name in ("alice", "bob"):
            response = self.guest.post("/api/auth/register", json={"username": name, "password": PASSWORD})
            self.assertEqual(response.status_code, 200, response.text)
        alice, a = self.login("alice")
        bob, b = self.login("bob")
        return admin, alice, bob, owner, a, b

    def experiment(self, name, uid=None):
        directory, _ = reserve_experiment(self.root, name)
        (directory / f"{name}.csv").write_text("round,accuracy,asr,loss\n1,0.8,0.1,0.3\n")
        (directory / "run.log").write_text("private training log")
        if uid:
            self.store.claim("experiment", directory, uid, replace=True)
        return directory

    def test_first_deployment_creates_admin_with_default_password_and_forces_reset(self):
        status = self.guest.get("/api/auth/status").json()
        self.assertTrue(status["admin_must_change_password"])
        self.assertEqual(status["default_admin_password"], DEFAULT_ADMIN_PASSWORD)
        self.assertEqual(self.guest.get("/api/experiments").status_code, 401)
        self.assertEqual(self.guest.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code, 401)
        response = self.guest.post("/api/auth/login", json={"username": "admin", "password": DEFAULT_ADMIN_PASSWORD})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["user"]["must_change_password"], 1)
        self.assertFalse((self.store.path.parent / "admin-setup-token.txt").exists())
        self.initialize()
        self.assertFalse(self.guest.get("/api/auth/status").json()["admin_must_change_password"])
        admin, user = self.login()
        self.assertEqual(user["role"], "admin")
        self.assertEqual(user["avatar"], "")
        self.assertEqual(len(user["uid"]), 36)
        self.assertEqual(user["must_change_password"], 0)
        self.assertEqual(len(self.store.users()), 1)
        cookie = admin.get("/api/auth/me")
        self.assertNotIn("password_hash", cookie.text)

    def test_duplicate_unicode_case_names_and_role_injection_are_rejected(self):
        self.initialize()
        for name in ("Alice", "Ａｌｉｃｅ", "alice"):
            response = self.guest.post("/api/auth/register", json={"username": name, "password": PASSWORD, "role": "admin"})
            self.assertEqual(response.status_code, 200 if name == "Alice" else 409)
            if response.status_code == 200:
                self.assertEqual(response.json()["user"]["role"], "user")
        self.assertEqual(self.guest.post("/api/auth/register", json={"username": "ADMIN", "password": PASSWORD}).status_code, 409)

    def test_list_pagination_detail_download_and_static_files_are_owner_scoped(self):
        admin, alice, bob, _, a, b = self.accounts()
        self.experiment("legacy")
        self.experiment("alice-run", a["uid"])
        self.experiment("bob-run", b["uid"])
        self.assertEqual(len(admin.get("/api/experiments").json()), 3)
        items = {item["name"]: item["owner"] for item in admin.get("/api/experiments").json()}
        self.assertEqual(items["legacy"], {"uid": self.store.admin_uid, "username": "admin"})
        self.assertEqual(items["alice-run"], {"uid": a["uid"], "username": "alice"})
        self.assertEqual(items["bob-run"], {"uid": b["uid"], "username": "bob"})
        page = alice.get("/api/experiments?paginated=true&limit=1").json()
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["name"], "alice-run")
        self.assertEqual(page["items"][0]["owner"], {"uid": a["uid"], "username": "alice"})
        self.assertEqual(admin.get("/api/experiments/alice-run").json()["owner"],
                         {"uid": a["uid"], "username": "alice"})
        for path in ("/api/experiments/bob-run", "/api/experiments/legacy", "/api/experiments/bob-run/csv/latest",
                     "/api/experiments/bob-run/all/download", "/results/bob-run/run.log", "/results/batches/task.json"):
            with self.subTest(path=path):
                self.assertEqual(alice.get(path).status_code, 404)
        self.assertEqual(alice.post("/api/experiments/bob-run/plot").status_code, 404)
        self.assertEqual(alice.delete("/api/experiments/bob-run").status_code, 404)
        response = alice.get("/results/alice-run/run.log")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.store.owner("experiment", self.root / "legacy"), self.store.admin_uid)
        self.assertEqual(self.guest.get("/results/alice-run/run.log").status_code, 401)

    def test_summary_sources_lists_and_assets_are_scoped(self):
        admin, alice, bob, _, a, b = self.accounts()
        self.experiment("bob-run", b["uid"])
        for name, uid in (("alice-summary", a["uid"]), ("bob-summary", b["uid"])):
            path = self.root / "summaries" / name
            path.mkdir(parents=True)
            (path / "summary.csv").write_text("experiment,accuracy\nbob-run,0.8\n")
            (path / "chart.png").write_bytes(b"image")
            self.store.claim("summary", path, uid, replace=True)
        self.assertEqual([s["name"] for s in alice.get("/api/summaries").json()], ["alice-summary"])
        self.assertEqual(alice.get("/api/summaries/bob-summary").status_code, 404)
        self.assertEqual(alice.delete("/api/summaries/bob-summary").status_code, 404)
        self.assertTrue((self.root / "summaries" / "bob-summary").is_dir())
        self.assertEqual(alice.get("/results/summaries/bob-summary/chart.png").status_code, 404)
        self.assertEqual(admin.get("/results/summaries/bob-summary/chart.png").status_code, 200)
        with patch.object(main, "_run_report", new_callable=AsyncMock) as report:
            self.assertEqual(alice.post("/api/summarize", json={"experiments": ["bob-run"]}).status_code, 404)
            report.assert_not_called()

    def test_batch_controls_status_and_socket_are_owner_scoped(self):
        admin, alice, bob, _, a, b = self.accounts()
        task = self.manager.create_task([{"name": "pending-run"}])
        self.store.claim("batch", self.manager._save_dir / task.batch_id, b["uid"], replace=True)
        self.assertEqual(alice.get("/api/batch/status").json(), {"status": "idle"})
        for operation in ("cancel", "pause", "resume", "stop-current"):
            self.assertEqual(alice.post(f"/api/batch/{task.batch_id}/{operation}", json={"run_id": "x"}).status_code, 404)
        self.assertEqual(alice.get(f"/api/batch/{task.batch_id}/status").status_code, 404)
        self.assertEqual(bob.get(f"/api/batch/{task.batch_id}/status").status_code, 200)
        self.assertEqual(alice.post("/api/experiment/stop").status_code, 404)
        with self.assertRaises(WebSocketDisconnect):
            with alice.websocket_connect(f"/ws/batch/{task.batch_id}", headers={"Origin": "http://testserver"}):
                pass

    def test_websocket_requires_session_origin_and_ownership(self):
        _, alice, bob, _, a, b = self.accounts()
        self.experiment("bob-run", b["uid"])
        for client, origin in ((self.guest, "http://testserver"), (alice, "http://testserver"), (bob, "https://evil.test")):
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect("/ws/logs/bob-run", headers={"Origin": origin}):
                    pass
        with bob.websocket_connect("/ws/logs/bob-run", headers={"Origin": "http://testserver"}) as ws:
            self.assertIsNotNone(ws.receive())

    def test_csrf_untrusted_origins_and_http_only_cookie(self):
        self.initialize()
        admin, _ = self.login()
        response = self.client().post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("SameSite=lax", response.headers["set-cookie"])
        self.assertEqual(admin.post("/api/auth/logout", headers={"X-CSRF-Token": "wrong"}).status_code, 403)
        self.assertEqual(admin.post("/api/auth/logout", headers={"Origin": "https://evil.test"}).status_code, 403)
        self.assertEqual(self.client().post("/api/auth/login", headers={"X-Lianan-Request": ""}, json={"username": "admin", "password": PASSWORD}).status_code, 403)

    def test_logout_password_change_disable_and_admin_only_overview(self):
        admin, alice, bob, _, a, b = self.accounts()
        self.assertEqual(alice.get("/api/admin/overview").status_code, 403)
        self.assertEqual(alice.get("/api/catalog/status").status_code, 403)
        self.assertEqual(len(admin.get("/api/admin/overview").json()["users"]), 3)
        self.assertEqual(admin.put(f'/api/admin/users/{b["uid"]}/disabled', json={"disabled": True}).status_code, 200)
        self.assertEqual(bob.get("/api/auth/me").status_code, 401)
        self.assertEqual(alice.post("/api/auth/password", json={"current_password": "wrong", "new_password": PASSWORD + "new"}).status_code, 400)
        second, _ = self.login("alice")
        self.assertEqual(alice.post("/api/auth/password", json={"current_password": PASSWORD, "new_password": PASSWORD + "new"}).status_code, 200)
        self.assertEqual(second.get("/api/auth/me").status_code, 401)
        third, _ = self.login("alice", PASSWORD + "new")
        self.assertEqual(third.post("/api/auth/logout").status_code, 200)
        self.assertEqual(third.get("/api/auth/me").status_code, 401)

    def test_private_config_backups_and_results_root_are_isolated(self):
        admin, alice, bob, _, a, b = self.accounts()
        self.assertEqual(alice.post("/api/config", json={"batch-size": 8}).status_code, 200)
        self.assertEqual(alice.get("/api/config").json()["batch-size"], 8)
        self.assertEqual(bob.get("/api/config").json()["batch-size"], 32)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(alice.post("/api/config/backups", json={"name": "alice-only"}).status_code, 200)
        self.assertFalse(any(item["filename"] == "alice-only.toml" for item in bob.get("/api/config/backups").json()["backups"]))
        self.assertEqual(bob.get("/api/config/backups/alice-only.toml").status_code, 404)
        self.assertEqual(alice.post("/api/config", json={"results-dir": str(self.base)}).status_code, 400)
        self.assertEqual(alice.get("/api/config/raw").json()["content"].count('batch-size = 8'), 1)

    def test_avatar_validation_persistence_and_default_reset(self):
        self.initialize()
        admin, _ = self.login()
        self.assertEqual(admin.put("/api/auth/avatar", json={"avatar": "https://tracking.test/a.svg"}).status_code, 422)
        self.assertEqual(admin.put("/api/auth/avatar", json={"avatar": "data:image/svg+xml;base64,PHN2Zz4="}).status_code, 422)
        value = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"test").decode()
        self.assertEqual(admin.put("/api/auth/avatar", json={"avatar": value}).status_code, 200)
        self.assertEqual(admin.get("/api/auth/me").json()["user"]["avatar"], value)
        self.assertEqual(admin.put("/api/auth/avatar", json={"avatar": ""}).status_code, 200)

    def test_sessions_persist_restart_and_passwords_are_never_plaintext(self):
        self.initialize()
        token, _, _, _ = self.store.login("admin", PASSWORD)
        other = IdentityStore(self.store.path)
        self.assertEqual(other.authenticate(token)["username"], "admin")
        with other.connection() as con:
            self.assertTrue(con.execute("SELECT password_hash FROM users").fetchone()[0].startswith("scrypt$"))
            self.assertNotEqual(con.execute("SELECT token_hash FROM sessions LIMIT 1").fetchone()[0], token)
        self.assertNotIn(PASSWORD.encode(), self.store.path.read_bytes())

    def test_authentication_attempts_are_rate_limited(self):
        self.initialize()
        for i in range(13):
            response = self.guest.post("/api/auth/login", json={"username": "unknown", "password": PASSWORD})
        self.assertEqual(response.status_code, 429)

    def test_starting_an_experiment_claims_it_to_the_creator(self):
        _, alice, _, _, a, _ = self.accounts()
        directory = self.experiment("alice-created")
        with patch.object(main.runner, "start", new=AsyncMock()), \
                patch.object(main.runner, "_experiment_dir", directory), \
                patch.object(main.runner, "run_id", "run-123"):
            response = alice.post("/api/experiment/start", json={"name": "alice-created"})
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.store.owner("experiment", directory), a["uid"])

    def test_forgot_password_and_force_reset_flow(self):
        self.initialize()
        result = self.guest.post("/api/auth/forgot", json={"username": "admin"}).json()
        self.assertTrue(result["admin_reset"])
        client, user = self.login("admin", DEFAULT_ADMIN_PASSWORD)
        self.assertEqual(user["must_change_password"], 1)
        reset = client.post("/api/auth/force-reset", json={"new_password": PASSWORD})
        self.assertEqual(reset.status_code, 200, reset.text)
        self.assertEqual(reset.json()["user"]["must_change_password"], 0)

    def test_forgot_deferred_for_users_and_admin_resets_password(self):
        admin, alice, _, _, a, _ = self.accounts()
        result = self.guest.post("/api/auth/forgot", json={"username": "alice"}).json()
        self.assertFalse(result["admin_reset"])
        self.assertEqual(alice.put(f"/api/admin/users/{a['uid']}/password", json={"new_password": PASSWORD}).status_code, 403)
        response = admin.put(f"/api/admin/users/{a['uid']}/password", json={"new_password": "Reset-Now-2026"})
        self.assertEqual(response.status_code, 200, response.text)
        _, user = self.login("alice", "Reset-Now-2026")
        self.assertEqual(user["must_change_password"], 1)

    def test_login_remember_extends_session_cookie(self):
        self.initialize()
        response = self.client().post("/api/auth/login", json={"username": "admin", "password": PASSWORD, "remember": True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Max-Age=2592000", response.headers["set-cookie"])


if __name__ == "__main__":
    unittest.main()