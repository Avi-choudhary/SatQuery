import os
import socket
from pathlib import Path
from typing import List

# Auto-load .env if present in backend directory or workspace root
_env_candidates = [
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parents[3] / ".env",
]
for _env_path in _env_candidates:
    if _env_path.exists():
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

BASE_DIR = Path(__file__).resolve().parents[1]

# Database settings
DEFAULT_DB_PATH = BASE_DIR / "satquery.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Fix postgresql:// vs postgresql+psycopg2:// if needed
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# JWT settings
_FALLBACK_SECRET = "satquery-v2-production-secret-sih-2026-key-9f8a7b6c5d4e3f2a1"
JWT_SECRET_KEY = os.getenv("SATQUERY_JWT_SECRET", os.getenv("JWT_SECRET_KEY", _FALLBACK_SECRET))
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Cookie settings for refresh token
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1", "yes")
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")

# CORS settings
_RAW_CORS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
)
CORS_ORIGINS: List[str] = [origin.strip() for origin in _RAW_CORS.split(",") if origin.strip()]

# Model & server network settings
MODEL_PATH = os.getenv("MODEL_PATH", "/models")
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "")  # Remote GPU server e.g. "https://xxxx.trycloudflare.com"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
TEMP_DIR = os.getenv("TEMP_DIR", "temp_uploads")


def get_lan_ip() -> str:
    """Detects the primary local area network (LAN) IP of this machine for dual-device access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
