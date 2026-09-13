"""
Dev live-reload: cheap fingerprint of UI/API files + SSE notify.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter(prefix="/api/dev", tags=["dev"])

# Explicit roots only (avoid scanning .venv / src-tauri / OneDrive noise)
WATCH_ROOTS = [
    ROOT / "web",
    ROOT / "backend",
    ROOT / "src",
    ROOT / "data" / "references",
    ROOT / "assets" / "signs_dict",
]


def fingerprint() -> str:
    latest = 0.0
    count = 0
    for root in WATCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".html",
                ".css",
                ".js",
                ".py",
                ".json",
                ".png",
                ".gif",
                ".jpg",
                ".jpeg",
                ".webp",
            }:
                continue
            try:
                latest = max(latest, path.stat().st_mtime)
                count += 1
            except OSError:
                continue
    return f"{count}:{latest:.3f}"


@router.get("/version")
def version() -> dict:
    return {"fingerprint": fingerprint(), "ts": time.time()}


@router.get("/reload-stream")
async def reload_stream() -> StreamingResponse:
    async def event_gen():
        last = fingerprint()
        yield f"data: connected {last}\n\n"
        while True:
            await asyncio.sleep(0.8)
            try:
                cur = await asyncio.to_thread(fingerprint)
            except Exception:
                continue
            if cur != last:
                last = cur
                yield "event: reload\ndata: changed\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
