"""Authentication routes and fail-closed ASGI access boundary (including files/WS)."""

import asyncio
import hmac
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.requests import HTTPConnection
from fl_backdoor.utils.app_paths import state_root

from backend.identity import (CURRENT_STORE, CURRENT_USER, DEFAULT_ADMIN_PASSWORD, IdentityStore,
                              SESSION_COOKIE, SESSION_SECONDS, SESSION_SECONDS_LONG)

router = APIRouter(prefix="/api/auth", tags=["身份认证"])
_stores = {}


def get_identity() -> IdentityStore:
    default = state_root() / "identity.sqlite3"
    path = Path(os.environ.get("FL_IDENTITY_DATABASE", str(default))).resolve()
    if path not in _stores:
        _stores[path] = IdentityStore(path)
    _stores[path].initialize()
    return _stores[path]


def trusted_origin(connection) -> bool:
    origin = connection.headers.get("origin")
    if not origin:
        return connection.scope["type"] != "websocket"
    allowed = {"http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"}
    allowed.update(x.strip().rstrip("/") for x in os.environ.get("FL_ALLOWED_ORIGINS", "").split(",") if x.strip())
    scheme = "https" if connection.scope["scheme"] in {"https", "wss"} else "http"
    allowed.add(f"{scheme}://{connection.headers.get('host', '')}")
    return origin.rstrip("/") in allowed


def admin_required():
    user = CURRENT_USER.get()
    if not user or user["role"] != "admin":
        raise HTTPException(403, "此操作仅限管理员。")
    return user


class AccessBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        path = scope["path"]
        if scope["type"] == "http" and scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        connection = HTTPConnection(scope)
        store = get_identity()
        user = None
        try:
            public = path in {"/api/auth/status", "/api/auth/login", "/api/auth/register", "/api/auth/forgot", "/api/health"}
            protected = path.startswith(("/api/", "/results", "/ws/")) or path in {"/docs", "/openapi.json", "/redoc"}
            if protected and not public:
                user = await asyncio.to_thread(store.authenticate, connection.cookies.get(SESSION_COOKIE, ""))
            if not trusted_origin(connection):
                raise HTTPException(403, "请求来源不受信任。")
            if scope["type"] == "http" and scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
                if public:
                    if connection.headers.get("x-lianan-request") != "1":
                        raise HTTPException(403, "请求缺少安全标记。")
                elif user and not hmac.compare_digest(connection.headers.get("x-csrf-token", ""), user["csrf"]):
                    raise HTTPException(403, "安全令牌无效，请刷新页面。")
            scope["platform_user"] = user
            user_token, store_token = CURRENT_USER.set(user), CURRENT_STORE.set(store)
            try:
                if path.startswith("/results/"):
                    from backend.main import _experiment_directory, _summary_directory
                    parts = path.removeprefix("/results/").split("/")
                    if parts[0] == "summaries" and len(parts) == 3:
                        _summary_directory(parts[1])
                    elif len(parts) == 2 and parts[0] not in {"batches", "summaries"}:
                        _experiment_directory(parts[0])
                    else:
                        raise HTTPException(404, "文件不存在或无权访问。")
                if path.startswith("/api/catalog/") or path.startswith("/api/admin/"):
                    admin_required()
                last_check = 0.0

                async def guarded_send(message):
                    nonlocal last_check
                    if scope["type"] == "websocket" and message["type"] == "websocket.send":
                        now = asyncio.get_running_loop().time()
                        if now - last_check > 5:
                            await asyncio.to_thread(store.authenticate, connection.cookies.get(SESSION_COOKIE, ""))
                            last_check = now
                    if message["type"] == "http.response.start" and protected:
                        headers = list(message.get("headers", []))
                        headers = [(k, v) for k, v in headers if k.lower() != b"cache-control"]
                        headers.extend([(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")])
                        message = {**message, "headers": headers}
                    await send(message)

                await self.app(scope, receive, guarded_send)
            finally:
                CURRENT_USER.reset(user_token)
                CURRENT_STORE.reset(store_token)
        except HTTPException as exc:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401 if exc.status_code == 401 else 4403})
            else:
                response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                                        headers={"Cache-Control": "no-store"})
                await response(scope, receive, send)


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128)
    remember: bool = False


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


class ForceReset(BaseModel):
    new_password: str = Field(min_length=12, max_length=128)


class ForgotPassword(BaseModel):
    username: str = Field(min_length=1, max_length=128)


class AvatarChange(BaseModel):
    avatar: str = Field(max_length=370000)


def no_store(data):
    return JSONResponse(data, headers={"Cache-Control": "no-store"})


@router.get("/status")
async def status():
    store = get_identity()
    return no_store({"admin_must_change_password": await asyncio.to_thread(store.admin_must_change_password),
                     "default_admin_password": DEFAULT_ADMIN_PASSWORD})


@router.post("/register")
async def register(data: Credentials, request: Request):
    store = get_identity()
    await asyncio.to_thread(store.throttle, "register:" + request.client.host, maximum=5)
    user = await asyncio.to_thread(store.create_user, data.username, data.password)
    return no_store({"user": user})


@router.post("/login")
async def login(data: Credentials, request: Request):
    store = get_identity()
    await asyncio.to_thread(store.throttle, "login-ip:" + request.client.host, maximum=30)
    await asyncio.to_thread(store.throttle, "login-user:" + data.username.casefold(), maximum=12)
    token, csrf, user, _ = await asyncio.to_thread(store.login, data.username, data.password, data.remember)
    response = no_store({"user": user, "csrf_token": csrf})
    response.set_cookie(SESSION_COOKIE, token,
                        max_age=SESSION_SECONDS_LONG if data.remember else SESSION_SECONDS,
                        httponly=True, samesite="lax",
                        secure=request.url.scheme == "https" or os.environ.get("FL_COOKIE_SECURE") == "1", path="/")
    return response


@router.get("/me")
async def me():
    user = dict(CURRENT_USER.get())
    csrf = user.pop("csrf")
    return no_store({"user": user, "csrf_token": csrf})


@router.post("/logout")
async def logout(request: Request):
    await asyncio.to_thread(get_identity().logout, request.cookies.get(SESSION_COOKIE, ""))
    response = no_store({"message": "已安全退出。"})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.post("/password")
async def change_password(data: PasswordChange):
    user = CURRENT_USER.get()
    await asyncio.to_thread(get_identity().throttle, "password:" + user["uid"])
    await asyncio.to_thread(get_identity().change_password, user["uid"], data.current_password, data.new_password)
    response = no_store({"message": "密码已修改，请重新登录。"})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.post("/force-reset")
async def force_reset(data: ForceReset):
    user = CURRENT_USER.get()
    if not user:
        raise HTTPException(401, "请先登录。")
    updated = await asyncio.to_thread(get_identity().force_reset_password, user["uid"], data.new_password)
    return no_store({"user": updated, "csrf_token": user["csrf"]})


@router.post("/forgot")
async def forgot(data: ForgotPassword, request: Request):
    store = get_identity()
    await asyncio.to_thread(store.throttle, "forgot:" + request.client.host, maximum=10)
    return no_store(await asyncio.to_thread(store.forgot_password, data.username))


@router.put("/avatar")
async def avatar(data: AvatarChange):
    await asyncio.to_thread(get_identity().update_avatar, CURRENT_USER.get()["uid"], data.avatar)
    return no_store({"avatar": data.avatar})