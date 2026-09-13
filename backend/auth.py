"""Simple local auth for Sign Bridge (student/demo accounts)."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
USERS_PATH = ROOT / "data" / "users.json"
SESSIONS_PATH = ROOT / "data" / "sessions.json"

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5, max_length=120)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    ).hex()


def _public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "created_at": user.get("created_at"),
    }


def _get_session(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    sessions = _load_json(SESSIONS_PATH, {})
    sess = sessions.get(token)
    if not sess:
        return None
    if sess.get("expires_at", 0) < time.time():
        sessions.pop(token, None)
        _save_json(SESSIONS_PATH, sessions)
        return None
    return sess


def _user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    users = _load_json(USERS_PATH, {"users": []})
    for u in users.get("users", []):
        if u.get("id") == user_id:
            return u
    return None


@router.post("/register")
def register(body: RegisterRequest) -> Dict[str, Any]:
    email = body.email.strip().lower()
    name = body.name.strip()
    if "@" not in email:
        raise HTTPException(400, "Enter a valid email")

    store = _load_json(USERS_PATH, {"users": []})
    if any(u.get("email") == email for u in store["users"]):
        raise HTTPException(409, "An account with this email already exists")

    salt = secrets.token_hex(16)
    user = {
        "id": secrets.token_hex(8),
        "name": name,
        "email": email,
        "salt": salt,
        "password_hash": _hash_password(body.password, salt),
        "created_at": int(time.time()),
    }
    store["users"].append(user)
    _save_json(USERS_PATH, store)

    token = secrets.token_urlsafe(24)
    sessions = _load_json(SESSIONS_PATH, {})
    sessions[token] = {
        "user_id": user["id"],
        "expires_at": time.time() + 60 * 60 * 24 * 14,
    }
    _save_json(SESSIONS_PATH, sessions)
    return {"token": token, "user": _public_user(user)}


@router.post("/login")
def login(body: LoginRequest) -> Dict[str, Any]:
    email = body.email.strip().lower()
    store = _load_json(USERS_PATH, {"users": []})
    user = next((u for u in store["users"] if u.get("email") == email), None)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    hashed = _hash_password(body.password, user["salt"])
    if not hmac.compare_digest(hashed, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    token = secrets.token_urlsafe(24)
    sessions = _load_json(SESSIONS_PATH, {})
    sessions[token] = {
        "user_id": user["id"],
        "expires_at": time.time() + 60 * 60 * 24 * 14,
    }
    _save_json(SESSIONS_PATH, sessions)
    return {"token": token, "user": _public_user(user)}


@router.get("/me")
def me(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    sess = _get_session(token)
    if not sess:
        raise HTTPException(401, "Not signed in")
    user = _user_by_id(sess["user_id"])
    if not user:
        raise HTTPException(401, "User missing")
    return {"user": _public_user(user)}


@router.post("/logout")
def logout(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if token:
        sessions = _load_json(SESSIONS_PATH, {})
        sessions.pop(token, None)
        _save_json(SESSIONS_PATH, sessions)
    return {"ok": True}
