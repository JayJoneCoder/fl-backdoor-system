"""Local identities, revocable sessions and artifact ownership; no training imports."""

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
import unicodedata
import uuid

from fastapi import HTTPException

CURRENT_USER = ContextVar("platform_user", default=None)
CURRENT_STORE = ContextVar("identity_store", default=None)
SESSION_COOKIE = "lianan_session"
SESSION_SECONDS = 12 * 60 * 60
SESSION_SECONDS_LONG = 30 * 24 * 60 * 60
DEFAULT_ADMIN_PASSWORD = "admin123"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def password_hash(password: str, *, validate=True) -> str:
    if not isinstance(password, str) or len(password) > 128:
        raise HTTPException(422, "密码须为不超过 128 个字符。")
    if validate:
        if len(password) < 12:
            raise HTTPException(422, "密码须为 12–128 个字符。")
        if sum((bool(re.search(r"[A-Za-z]", password)), bool(re.search(r"\d", password)),
                bool(re.search(r"[^A-Za-z\d]", password)))) < 2:
            raise HTTPException(422, "密码须至少包含字母、数字、符号中的两类。")
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=32768, r=8, p=3, maxmem=64 * 1024 * 1024)
    return "scrypt$" + salt.hex() + "$" + key.hex()


def verify_password(password: str, encoded: str | None) -> bool:
    if not isinstance(password, str) or len(password) > 128:
        return False
    # The same KDF is executed for unknown users to avoid a fast enumeration path.
    parts = (encoded or ("scrypt$" + "00" * 16 + "$" + "00" * 64)).split("$")
    key = hashlib.scrypt(password.encode(), salt=bytes.fromhex(parts[1]), n=32768, r=8, p=3,
                         maxmem=64 * 1024 * 1024)
    return hmac.compare_digest(key.hex(), parts[2]) and encoded is not None


def username_value(value: str) -> str:
    if not isinstance(value, str):
        raise HTTPException(422, "请输入用户名。")
    value = unicodedata.normalize("NFKC", value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-\u3400-\u9fff]{3,32}", value):
        raise HTTPException(422, "用户名须为 3–32 位中文、字母、数字、下划线或短横线。")
    return value


def public_user(row) -> dict:
    return {key: row[key] for key in ("uid", "username", "role", "avatar", "created_at", "last_seen", "disabled", "must_change_password")}


class IdentityStore:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self._lock = threading.RLock()
        self._ready = False

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
        finally:
            con.close()

    def initialize(self):
        with self._lock:
            if self._ready:
                return
            with self.connection() as con:
                con.executescript("""
                    CREATE TABLE IF NOT EXISTS users (
                        uid TEXT PRIMARY KEY, username TEXT NOT NULL, username_key TEXT NOT NULL UNIQUE,
                        password_hash TEXT, role TEXT NOT NULL CHECK(role IN ('admin','user')),
                        avatar TEXT NOT NULL DEFAULT '', disabled INTEGER NOT NULL DEFAULT 0,
                        must_change_password INTEGER NOT NULL DEFAULT 0,
                        created_at INTEGER NOT NULL, last_seen INTEGER);
                    CREATE TABLE IF NOT EXISTS sessions (
                        token_hash TEXT PRIMARY KEY, uid TEXT NOT NULL REFERENCES users(uid),
                        csrf TEXT NOT NULL, expires INTEGER NOT NULL);
                    CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS ownership (
                        kind TEXT NOT NULL, location TEXT NOT NULL, uid TEXT NOT NULL REFERENCES users(uid),
                        PRIMARY KEY(kind, location));
                    CREATE TABLE IF NOT EXISTS attempts (
                        key TEXT PRIMARY KEY, count INTEGER NOT NULL, started INTEGER NOT NULL);
                """)
                columns = {row["name"] for row in con.execute("PRAGMA table_info(users)")}
                if "must_change_password" not in columns:
                    con.execute("ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0")
                con.execute("INSERT OR IGNORE INTO users(uid,username,username_key,role,created_at) VALUES (?,?,?,'admin',?)",
                            (str(uuid.uuid4()), "admin", "admin", int(time.time())))
                admin = con.execute("SELECT * FROM users WHERE username_key='admin'").fetchone()
                if admin["password_hash"] is None:
                    # First run (or a legacy token-based install that was never set up):
                    # assign the documented default password and force a reset on first login.
                    con.execute("UPDATE users SET password_hash=?, must_change_password=1 WHERE uid=?",
                                (password_hash(DEFAULT_ADMIN_PASSWORD, validate=False), admin["uid"]))
                    (self.path.parent / "admin-setup-token.txt").unlink(missing_ok=True)
            self._ready = True

    def admin_must_change_password(self) -> bool:
        self.initialize()
        with self.connection() as con:
            row = con.execute("SELECT must_change_password FROM users WHERE username_key='admin'").fetchone()
            return bool(row and row[0])

    def throttle(self, key: str, *, maximum=12):
        self.initialize()
        now = int(time.time())
        with self.connection() as con:
            con.execute("DELETE FROM attempts WHERE started<?", (now - 900,))
            con.execute("INSERT INTO attempts VALUES (?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1", (digest(key), now))
            if con.execute("SELECT count FROM attempts WHERE key=?", (digest(key),)).fetchone()[0] > maximum:
                raise HTTPException(429, "尝试次数过多，请在 15 分钟后重试。")

    def create_user(self, username: str, password: str):
        self.initialize()
        username = username_value(username)
        encoded = password_hash(password)
        try:
            with self.connection() as con:
                uid = str(uuid.uuid4())
                con.execute("INSERT INTO users(uid,username,username_key,password_hash,role,created_at) VALUES (?,?,?,?,'user',?)",
                            (uid, username, username.casefold(), encoded, int(time.time())))
                return public_user(con.execute("SELECT * FROM users WHERE uid=?", (uid,)).fetchone())
        except sqlite3.IntegrityError as exc:
            raise HTTPException(409, "用户名已存在，请使用其他名称。") from exc

    def login(self, username: str, password: str, remember: bool = False):
        self.initialize()
        key = unicodedata.normalize("NFKC", str(username)).strip().casefold()
        with self.connection() as con:
            row = con.execute("SELECT * FROM users WHERE username_key=?", (key,)).fetchone()
            valid = verify_password(password, row["password_hash"] if row else None)
            if not row or not valid or row["disabled"]:
                raise HTTPException(401, "用户名或密码不正确，或账户已停用。")
            token, csrf = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
            now = int(time.time())
            lifetime = SESSION_SECONDS_LONG if remember else SESSION_SECONDS
            con.execute("DELETE FROM sessions WHERE expires<?", (now,))
            con.execute("INSERT INTO sessions VALUES (?,?,?,?)", (digest(token), row["uid"], csrf, now + lifetime))
            con.execute("UPDATE users SET last_seen=? WHERE uid=?", (now, row["uid"]))
            return token, csrf, public_user(row), now + lifetime

    def authenticate(self, token: str):
        self.initialize()
        with self.connection() as con:
            row = con.execute("SELECT users.*, sessions.csrf FROM sessions JOIN users USING(uid) WHERE token_hash=? AND expires>? AND disabled=0",
                              (digest(token), int(time.time()))).fetchone()
            if not row:
                raise HTTPException(401, "登录已过期，请重新登录。")
            con.execute("UPDATE users SET last_seen=? WHERE uid=? AND (last_seen IS NULL OR last_seen<?)",
                        (int(time.time()), row["uid"], int(time.time()) - 30))
            return {**public_user(row), "csrf": row["csrf"]}

    def logout(self, token: str):
        with self.connection() as con:
            con.execute("DELETE FROM sessions WHERE token_hash=?", (digest(token),))

    def change_password(self, uid: str, old: str, new: str):
        encoded = password_hash(new)
        with self.connection() as con:
            row = con.execute("SELECT * FROM users WHERE uid=?", (uid,)).fetchone()
            if not verify_password(old, row["password_hash"]):
                raise HTTPException(400, "当前密码不正确。")
            con.execute("BEGIN IMMEDIATE")
            con.execute("UPDATE users SET password_hash=?, must_change_password=0 WHERE uid=?", (encoded, uid))
            con.execute("DELETE FROM sessions WHERE uid=?", (uid,))
            con.commit()

    def force_reset_password(self, uid: str, new_password: str):
        """Complete the forced first-login reset; only allowed while flagged."""
        encoded = password_hash(new_password)
        with self.connection() as con:
            row = con.execute("SELECT * FROM users WHERE uid=?", (uid,)).fetchone()
            if not row or not row["must_change_password"]:
                raise HTTPException(403, "当前无需强制修改密码。")
            con.execute("UPDATE users SET password_hash=?, must_change_password=0 WHERE uid=?", (encoded, uid))
            return public_user(con.execute("SELECT * FROM users WHERE uid=?", (uid,)).fetchone())

    def reset_user_password(self, uid: str, new_password: str):
        """Admin resets a regular user's password; they must change it next login."""
        encoded = password_hash(new_password)
        with self.connection() as con:
            row = con.execute("SELECT * FROM users WHERE uid=?", (uid,)).fetchone()
            if not row:
                raise HTTPException(404, "用户不存在。")
            if row["role"] == "admin":
                raise HTTPException(403, "不能重置管理员密码，请使用忘记密码流程。")
            con.execute("UPDATE users SET password_hash=?, must_change_password=1 WHERE uid=?", (encoded, uid))
            con.execute("DELETE FROM sessions WHERE uid=?", (uid,))

    def forgot_password(self, username: str) -> dict:
        """Self-service recovery for the local admin; regular users go through admin."""
        key = unicodedata.normalize("NFKC", str(username)).strip().casefold()
        with self.connection() as con:
            row = con.execute("SELECT * FROM users WHERE username_key=?", (key,)).fetchone()
            if not row:
                return {"admin_reset": False, "message": "若该账户存在，请联系管理员重置密码。"}
            if row["role"] != "admin":
                return {"admin_reset": False, "message": "普通用户无法自助找回，请联系管理员在用户总览中重置密码。"}
            encoded = password_hash(DEFAULT_ADMIN_PASSWORD, validate=False)
            con.execute("UPDATE users SET password_hash=?, must_change_password=1 WHERE uid=?", (encoded, row["uid"]))
            con.execute("DELETE FROM sessions WHERE uid=?", (row["uid"],))
            return {"admin_reset": True, "message": "管理员密码已重置为默认密码 admin123，请使用默认密码登录并立即修改。"}

    def update_avatar(self, uid: str, avatar: str):
        # Raster-only, bounded inline images; no external URLs, SVG or executable uploads.
        if not isinstance(avatar, str) or len(avatar) > 370000:
            raise HTTPException(422, "头像须为不超过 256 KB 的 PNG、JPEG 或 WebP 图片。")
        if avatar:
            match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", avatar)
            try:
                raw = base64.b64decode(match[2], validate=True) if match else b""
            except ValueError:
                raw = b""
            valid = match and ((match[1] == "png" and raw.startswith(b"\x89PNG\r\n\x1a\n"))
                              or (match[1] == "jpeg" and raw.startswith(b"\xff\xd8\xff"))
                              or (match[1] == "webp" and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"))
            if not valid or len(raw) > 256 * 1024:
                raise HTTPException(422, "头像格式无效，仅接受 PNG、JPEG 或 WebP 图片。")
        with self.connection() as con:
            con.execute("UPDATE users SET avatar=? WHERE uid=?", (avatar, uid))

    @property
    def admin_uid(self):
        self.initialize()
        with self.connection() as con:
            return con.execute("SELECT uid FROM users WHERE username_key='admin'").fetchone()[0]

    @staticmethod
    def location(path: Path):
        return os.path.normcase(str(Path(path).resolve())).casefold()

    def claim(self, kind: str, path: Path, uid: str | None = None, *, replace=False):
        self.initialize()
        uid = uid or self.admin_uid
        with self.connection() as con:
            clause = " ON CONFLICT(kind,location) DO UPDATE SET uid=excluded.uid" if replace else " ON CONFLICT DO NOTHING"
            con.execute("INSERT INTO ownership VALUES (?,?,?)" + clause, (kind, self.location(path), uid))

    def owner(self, kind: str, path: Path):
        # Unassigned legacy/CLI artifacts belong to admin, never to the requesting user.
        self.claim(kind, path)
        with self.connection() as con:
            return con.execute("SELECT uid FROM ownership WHERE kind=? AND location=?", (kind, self.location(path))).fetchone()[0]

    def owner_info(self, kind: str, path: Path) -> dict:
        """Return the owning username and UID; unassigned legacy runs belong to admin."""
        uid = self.owner(kind, path)
        with self.connection() as con:
            row = con.execute("SELECT username FROM users WHERE uid=?", (uid,)).fetchone()
        return {"uid": uid, "username": row["username"] if row else "已删除用户"}

    def owner_infos(self, kind: str, paths: list[Path]) -> list[dict]:
        """Resolve many owners with one read; only truly unassigned paths are claimed."""
        self.initialize()
        locations = [self.location(path) for path in paths]
        if not locations:
            return []
        found = {}
        with self.connection() as con:
            for start in range(0, len(locations), 500):
                chunk = locations[start:start + 500]
                rows = con.execute(
                    "SELECT o.location, o.uid, u.username FROM ownership o LEFT JOIN users u ON u.uid=o.uid "
                    f"WHERE o.kind=? AND o.location IN ({','.join('?' * len(chunk))})", (kind, *chunk))
                found.update({row["location"]: {"uid": row["uid"], "username": row["username"] or "已删除用户"}
                              for row in rows})
        return [found.get(location) or self.owner_info(kind, path) for location, path in zip(locations, paths)]

    def allowed(self, user, kind: str, path: Path):
        return user is None or user["role"] == "admin" or self.owner(kind, path) == user["uid"]

    def require(self, user, kind: str, path: Path):
        if not self.allowed(user, kind, path):
            raise HTTPException(404, "资源不存在或无权访问。")

    def users(self):
        with self.connection() as con:
            return [public_user(row) for row in con.execute("SELECT * FROM users ORDER BY created_at,username")]

    def set_disabled(self, uid: str, disabled: bool):
        with self.connection() as con:
            row = con.execute("SELECT * FROM users WHERE uid=?", (uid,)).fetchone()
            if not row:
                raise HTTPException(404, "用户不存在。")
            if row["role"] == "admin":
                raise HTTPException(403, "不能停用管理员。")
            con.execute("UPDATE users SET disabled=? WHERE uid=?", (int(disabled), uid))
            if disabled:
                con.execute("DELETE FROM sessions WHERE uid=?", (uid,))


def require_resource(kind: str, path: Path):
    store = CURRENT_STORE.get()
    if store:
        store.require(CURRENT_USER.get(), kind, path)


def claim_resource(kind: str, path: Path):
    store, user = CURRENT_STORE.get(), CURRENT_USER.get()
    if store and user:
        store.claim(kind, path, user["uid"], replace=True)