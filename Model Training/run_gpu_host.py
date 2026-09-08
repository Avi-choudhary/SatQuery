"""
SatQuery Remote GPU Host Launcher
=================================
Runs the SatQuery Vision-Language Model on your desktop GPU (RTX 5060 Ti)
and exposes it to the entire internet via a zero-config, secure HTTPS tunnel.

This allows any laptop on any Wi-Fi network (college, cafe, hotspot, hackathon)
to query the model as an on-demand AI inference API.
"""

import os
import sys
import time
import json
import re
import shutil
import urllib.request
import subprocess
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED_EXE = BASE_DIR / "cloudflared.exe"
CLOUDFLARED_DOWNLOAD_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
PORT = 8008


def get_python_exe() -> str:
    """Finds the Python executable with GPU / PyTorch CUDA support."""
    venv_py = BASE_DIR / "venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def ensure_cloudflared() -> Optional[Path]:
    """Ensures cloudflared.exe is available; downloads it if missing."""
    # Check if in PATH
    which_path = shutil.which("cloudflared")
    if which_path:
        return Path(which_path)

    # Check local file
    if CLOUDFLARED_EXE.exists():
        return CLOUDFLARED_EXE

    print("\n[Host Setup] Cloudflare Tunnel binary not found locally.")
    print(f"[Host Setup] Downloading official zero-config cloudflared binary (~35MB)...")
    try:
        urllib.request.urlretrieve(CLOUDFLARED_DOWNLOAD_URL, str(CLOUDFLARED_EXE))
        print("[Host Setup] Download complete!\n")
        return CLOUDFLARED_EXE
    except Exception as e:
        print(f"[Host Setup] Warning: Could not auto-download cloudflared ({e}).")
        return None


def wait_for_server(port: int = PORT, timeout_secs: int = 120) -> bool:
    """Waits until the FastAPI model server is healthy and model is loaded."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    print("[Host] Waiting for VLM model to load into GPU memory...")
    last_print = 0

    while time.time() - start < timeout_secs:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SatQueryHostLauncher/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "ready":
                    gpu_device = data.get("device_name", "GPU")
                    vram = data.get("vram_allocated_gb", 0)
                    print(f"[Host] Model successfully loaded on {gpu_device}! ({vram} GB VRAM allocated)")
                    return True
        except Exception:
            pass

        time.sleep(2)
        elapsed = int(time.time() - start)
        if elapsed - last_print >= 10:
            print(f"[Host] Still loading weights... ({elapsed}s elapsed)")
            last_print = elapsed

    return False


def start_tunnel(port: int = PORT):
    """Starts a public HTTPS tunnel and returns (process, public_url)."""
    cf_path = ensure_cloudflared()

    if cf_path:
        print("[Tunnel] Starting Cloudflare Quick Tunnel (Free, encrypted HTTPS)...")
        cmd = [str(cf_path), "tunnel", "--url", f"http://127.0.0.1:{port}"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        url = None
        # Parse output for trycloudflare.com link
        for line in proc.stdout:
            # print(f"[Cloudflare Log] {line.strip()}")
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                url = match.group(0)
                break
        return proc, url, "Cloudflare Quick Tunnel"

    # Fallback to localtunnel via npx if node is installed
    npx_path = shutil.which("npx")
    if npx_path:
        print("[Tunnel] Using LocalTunnel via npx fallback...")
        cmd = ["cmd.exe", "/c", "npx", "-y", "localtunnel", "--port", str(port)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        url = None
        for line in proc.stdout:
            match = re.search(r"https://[a-zA-Z0-9-]+\.loca\.lt", line)
            if match:
                url = match.group(0)
                break
        return proc, url, "LocalTunnel"

    return None, None, "None"


def main():
    py_exe = get_python_exe()
    server_script = BASE_DIR / "serve_model.py"

    print("=" * 72)
    print("      🛰️  SATQUERY REMOTE GPU INFERENCE SERVER LAUNCHER  🛰️")
    print("=" * 72)
    print(f"Python Runtime: {py_exe}")
    print(f"Model Script:   {server_script}")
    print(f"Local Port:     {PORT}")
    print("=" * 72)

    # 1. Start Model Server
    print("\n[Host] Starting Model API Server...")
    server_cmd = [py_exe, str(server_script), "--host", "127.0.0.1", "--port", str(PORT)]
    server_proc = subprocess.Popen(server_cmd)

    tunnel_proc = None
    try:
        # 2. Wait until model is resident in VRAM
        if not wait_for_server(PORT):
            print("[Host] Error: Model server timed out or failed to initialize.")
            server_proc.terminate()
            return

        # 3. Launch HTTPS Tunnel
        tunnel_proc, public_url, provider = start_tunnel(PORT)

        print("\n" + "=" * 72)
        print("  🎉 SATQUERY GPU SERVER IS ONLINE AND ACCESSIBLE WORLDWIDE! 🎉")
        print("=" * 72)
        print(f"  • Local Endpoint:  http://127.0.0.1:{PORT}")
        if public_url:
            print(f"  • Public HTTPS:    {public_url}")
            print(f"  • Provider:        {provider}")
        else:
            print("  • Public HTTPS:    Tunnel URL could not be auto-parsed.")
        print("=" * 72)
        print("  👉 ON YOUR LAPTOP (Any Wi-Fi / Anywhere):")
        print("     1. Open 'backend/.env' (or 'core/config.py') and set:")
        if public_url:
            print(f"        MODEL_SERVICE_URL={public_url}")
        else:
            print(f"        MODEL_SERVICE_URL=http://<YOUR_PC_PUBLIC_IP>:{PORT}")
        print("     2. Start laptop backend:  uvicorn main:app --port 8000")
        print("     3. Start laptop frontend: npm run dev")
        print("=" * 72)
        print("  Press Ctrl+C to stop the GPU host and tunnel.\n")

        # Keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Host] Shutting down GPU host and tunnel...")
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
        if server_proc:
            try:
                server_proc.terminate()
            except Exception:
                pass
        print("[Host] Clean shutdown complete. Goodbye!")


if __name__ == "__main__":
    main()
