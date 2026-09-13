"""
Sign Bridge native desktop window (WebView2 via pywebview).
Used when the Tauri .exe is not built yet, and as a reliable launcher.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000/"


def python_exe() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def healthy() -> bool:
    try:
        with urllib.request.urlopen(URL + "api/health", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server() -> None:
    if healthy():
        return
    subprocess.Popen(
        [
            python_exe(),
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for _ in range(60):
        time.sleep(0.4)
        if healthy():
            return
    raise RuntimeError("Sign Bridge API failed to start on :8000")


def main() -> None:
    ensure_server()
    try:
        import webview
    except ImportError:
        subprocess.check_call([python_exe(), "-m", "pip", "install", "pywebview"])
        import webview

    webview.create_window(
        "Sign Bridge",
        URL,
        width=1360,
        height=860,
        min_size=(960, 640),
    )
    webview.start()


if __name__ == "__main__":
    main()
