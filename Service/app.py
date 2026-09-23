"""FastAPI service and browser frontend for realtime Muse emotion detection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Service.detector import EmotionDetectionService
from Service.muse_client import DEFAULT_ADDRESS, MUSE_SAMPLE_RATE, MuseReceiver


STATIC_DIR = Path(__file__).resolve().parent / "static"

receiver = MuseReceiver(DEFAULT_ADDRESS)
detector = EmotionDetectionService(receiver)


class ConnectRequest(BaseModel):
    """Optional Muse address supplied by the browser dashboard."""

    address: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    detector.stop()
    receiver.stop()


app = FastAPI(
    title="Muse EEG Emotion Recognition API",
    description="Connect to a Muse headset and expose realtime emotion predictions.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "receiver": receiver.snapshot(),
        "detector": detector.status(),
    }


@app.get("/api/eeg-waves")
def eeg_waves(
    seconds: float = Query(default=4.0, ge=1.0, le=10.0),
    points: int = Query(default=420, ge=64, le=1024),
) -> dict[str, Any]:
    """Return recent Muse samples for the live four-channel waveform panel."""
    samples_needed = int(seconds * MUSE_SAMPLE_RATE)
    return receiver.waveform_snapshot(samples_needed, points)


@app.post("/api/connect")
def connect(payload: ConnectRequest | None = None) -> dict[str, Any]:
    requested_address = payload.address if payload else None
    address = (requested_address or DEFAULT_ADDRESS).strip() or DEFAULT_ADDRESS
    receiver.start(address)
    return {"ok": True, "message": "Connection started.", "status": receiver.snapshot()}


@app.post("/api/disconnect")
def disconnect() -> dict[str, Any]:
    detector.stop()
    receiver.stop()
    return {"ok": True, "message": "Disconnected.", "status": receiver.snapshot()}


@app.post("/api/start-detection")
def start_detection() -> Any:
    if not receiver.running:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": "Connect the Muse device first."},
        )
    detector.start()
    return {"ok": True, "message": "Detection started.", "status": detector.status()}


@app.post("/api/stop-detection")
def stop_detection() -> dict[str, Any]:
    detector.stop()
    return {"ok": True, "message": "Detection stopped.", "status": detector.status()}
