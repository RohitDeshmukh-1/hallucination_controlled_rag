from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from configs.settings import settings

logger = logging.getLogger(__name__)

_TOKEN_SPLIT = "."
_PASSWORD_SPLIT = "$"
_PBKDF2_ITERATIONS = 200_000
_bearer_scheme = HTTPBearer(auto_error=False)


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    username: str
    display_name: str

    def to_dict(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
        }


class AuthError(Exception):
    pass


class UserStore:
    def __init__(self):
        self.path = settings.USERS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"users": []})

    def _read(self) -> dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    @staticmethod
    def _normalize_username(username: str) -> str:
        normalized = username.strip().lower()
        if len(normalized) < 3:
            raise AuthError("Username must be at least 3 characters long")
        return normalized

    def find_by_username(self, username: str) -> Optional[dict[str, Any]]:
        normalized = self._normalize_username(username)
        data = self._read()
        return next((user for user in data["users"] if user["username"] == normalized), None)

    def find_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        data = self._read()
        return next((user for user in data["users"] if user["user_id"] == user_id), None)

    def create_user(self, username: str, password: str) -> AuthenticatedUser:
        normalized = self._normalize_username(username)
        password = password.strip()
        if len(password) < 6:
            raise AuthError("Password must be at least 6 characters long")

        data = self._read()
        if any(user["username"] == normalized for user in data["users"]):
            raise AuthError("That username is already registered")

        record = {
            "user_id": uuid.uuid4().hex[:16],
            "username": normalized,
            "display_name": username.strip() or normalized,
            "password_hash": hash_password(password),
            "created_at": int(time.time()),
        }
        data["users"].append(record)
        self._write(data)
        logger.info("Created user %s", normalized)
        return self._to_user(record)

    def authenticate(self, username: str, password: str) -> AuthenticatedUser:
        normalized = self._normalize_username(username)
        record = self.find_by_username(normalized)
        if not record or not verify_password(password, record["password_hash"]):
            raise AuthError("Invalid username or password")
        return self._to_user(record)

    def get_user(self, user_id: str) -> AuthenticatedUser:
        record = self.find_by_id(user_id)
        if not record:
            raise AuthError("User account was not found")
        return self._to_user(record)

    @staticmethod
    def _to_user(record: dict[str, Any]) -> AuthenticatedUser:
        return AuthenticatedUser(
            user_id=record["user_id"],
            username=record["username"],
            display_name=record.get("display_name") or record["username"],
        )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return f"{_b64encode(salt)}{_PASSWORD_SPLIT}{_b64encode(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt_b64, digest_b64 = password_hash.split(_PASSWORD_SPLIT, maxsplit=1)
    except ValueError:
        return False

    expected = _b64decode(digest_b64)
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        _b64decode(salt_b64),
        _PBKDF2_ITERATIONS,
    )
    return hmac.compare_digest(candidate, expected)


def create_access_token(user: AuthenticatedUser) -> str:
    payload = {
        "sub": user.user_id,
        "usr": user.username,
        "name": user.display_name,
        "exp": int(time.time()) + settings.AUTH_TOKEN_TTL_HOURS * 3600,
    }
    payload_b64 = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = _b64encode(
        hmac.new(
            settings.AUTH_SECRET_KEY.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    )
    return f"{payload_b64}{_TOKEN_SPLIT}{signature}"


def decode_access_token(token: str) -> AuthenticatedUser:
    try:
        payload_b64, signature = token.split(_TOKEN_SPLIT, maxsplit=1)
    except ValueError as exc:
        raise AuthError("Invalid authentication token") from exc

    expected_signature = _b64encode(
        hmac.new(
            settings.AUTH_SECRET_KEY.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    )
    if not hmac.compare_digest(signature, expected_signature):
        raise AuthError("Invalid authentication token")

    try:
        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise AuthError("Invalid authentication token") from exc

    if payload.get("exp", 0) < int(time.time()):
        raise AuthError("Session expired, please log in again")

    return user_store.get_user(payload["sub"])


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    try:
        return decode_access_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


user_store = UserStore()
