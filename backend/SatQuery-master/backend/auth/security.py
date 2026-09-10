import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import bcrypt
import jwt

from core.config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)


def hash_password(password: str) -> str:
    """Hash password securely using bcrypt."""
    pw_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def hash_token(token_str: str) -> str:
    """Compute SHA-256 hash of a token for secure database lookup."""
    return hashlib.sha256(token_str.encode("utf-8")).hexdigest()


import uuid


def create_access_token(user_id: str, email: str, expires_delta: Optional[timedelta] = None) -> Tuple[str, int]:
    """
    Create signed JWT access token.
    Returns (token_string, expires_in_seconds).
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    expires_in = int((expire - now).total_seconds())
    return token, expires_in



def create_refresh_token() -> Tuple[str, str, datetime]:
    """
    Generate cryptographically secure refresh token.
    Returns (raw_token, token_hash, expires_at).
    """
    raw_token = secrets.token_urlsafe(64)
    token_hash = hash_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return raw_token, token_hash, expires_at


def decode_access_token(token: str) -> dict:
    """
    Decode and validate signed JWT access token.
    Raises jwt.PyJWTError on failure or expired token.
    """
    payload = jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        options={"require": ["sub", "exp", "type"]},
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("Invalid token type")
    return payload
