# Environment variables & network configuration
import os
import socket
from pathlib import Path

# Auto-load .env if present in backend directory or workspace root
_env_candidates = [
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parents[3] / ".env"
]
for _env_path in _env_candidates:
    if _env_path.exists():
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

MODEL_PATH = os.getenv("MODEL_PATH", "/models")
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "")  # Standalone remote GPU server e.g. "https://xxxx.trycloudflare.com"
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

