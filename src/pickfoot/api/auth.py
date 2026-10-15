"""Single-admin password verification, signed sessions, and write rate limits."""

from __future__ import annotations

import base64
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import hmac
import json
import secrets
import threading
import time
from fastapi import HTTPException, Request, Response, status

from pickfoot.api.config import get_settings

SESSION_COOKIE = "pickfoot_admin"
_rate_lock = threading.Lock()
_request_times: dict[str, deque[float]] = defaultdict(deque)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str, *, salt: bytes | None = None, rounds: int = 310_000) -> str:
    actual_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), actual_salt, rounds)
    return f"pbkdf2_sha256${rounds}${_b64(actual_salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds_text, salt_text, digest_text = encoded.split("$", maxsplit=3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), _unb64(salt_text), int(rounds_text)
        )
        return hmac.compare_digest(actual, _unb64(digest_text))
    except (ValueError, TypeError):
        return False


def _sign(value: str, secret: str) -> str:
    return _b64(hmac.new(secret.encode("utf-8"), value.encode("ascii"), hashlib.sha256).digest())


def create_session(username: str, secret: str, ttl_seconds: int) -> str:
    payload = json.dumps({
        "sub": username,
        "exp": int((datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).timestamp()),
    }, separators=(",", ":"), sort_keys=True).encode("utf-8")
    body = _b64(payload)
    return f"{body}.{_sign(body, secret)}"


def verify_session(token: str, secret: str) -> str | None:
    try:
        body, signature = token.split(".", maxsplit=1)
        if not hmac.compare_digest(signature, _sign(body, secret)):
            return None
        payload = json.loads(_unb64(body))
        if int(payload["exp"]) <= int(datetime.now(timezone.utc).timestamp()):
            return None
        username = payload["sub"]
        return username if isinstance(username, str) else None
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def set_session_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="none" if settings.cookie_secure else "lax",
        path="/",
    )


def require_same_origin(request: Request) -> None:
    settings = get_settings()
    origin = request.headers.get("origin", "").rstrip("/")
    allowed = {settings.web_origin.rstrip("/"), *(item.rstrip("/") for item in settings.cors_origins)}
    if not origin or origin not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request origin is not allowed.",
        )


def enforce_rate_limit(
    request: Request,
    *,
    scope: str,
    maximum: int,
    window_seconds: int,
) -> None:
    now = time.monotonic()
    client = request.client.host if request.client else "unknown"
    key = f"{scope}:{client}"
    with _rate_lock:
        events = _request_times[key]
        while events and now - events[0] >= window_seconds:
            events.popleft()
        if len(events) >= maximum:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Try again later.",
                headers={"Retry-After": str(window_seconds)},
            )
        events.append(now)
        if len(_request_times) > 10_000:
            expired = [
                item for item, values in _request_times.items()
                if not values or now - values[-1] >= 3600
            ]
            for item in expired:
                _request_times.pop(item, None)


def admin_user(request: Request) -> str:
    settings = get_settings()
    if not settings.session_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication is not configured.",
        )
    token = request.cookies.get(SESSION_COOKIE, "")
    username = verify_session(token, settings.session_secret)
    if username != settings.admin_username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin login required.")
    return username


def password_hash_main() -> None:
    first = getpass.getpass("New PickFoot admin password: ")
    second = getpass.getpass("Repeat password: ")
    if len(first) < 14 or first != second:
        raise SystemExit("Passwords must match and contain at least 14 characters.")
    print(hash_password(first))


if __name__ == "__main__":
    password_hash_main()
