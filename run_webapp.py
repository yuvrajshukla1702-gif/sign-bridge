"""Launch the Sign Bridge web app (FastAPI + frontend)."""

from __future__ import annotations

import socket
import sys

import uvicorn


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


if __name__ == "__main__":
    port = 8000
    if _port_busy(port):
        print(f"Port {port} is already in use.")
        print("If Sign Bridge is already running, open: http://127.0.0.1:8000")
        print("Or run: powershell -ExecutionPolicy Bypass -File .\\scripts\\start-app.ps1")
        sys.exit(1)

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=port,
        reload=True,
    )
