"""FastAPI + WebSocket telemetry backend for live training metrics."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import psutil
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


@dataclass
class TelemetryRecord:
    """Serializable snapshot of a single train/validation update."""

    epoch: int
    step: int
    phase: str
    loss: float | None = None
    accuracy: float | None = None
    val_auc: float | None = None
    learning_rate: float | None = None
    attention_mean: float | None = None
    attention_peak: float | None = None
    vram_allocated_mb: float | None = None
    vram_reserved_mb: float | None = None
    cpu_percent: float | None = None
    ram_percent: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TelemetryHub:
    """Thread-safe fan-out for broadcasting metrics to connected WebSockets."""

    def __init__(self) -> None:
        self._latest: dict[str, Any] | None = None
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def snapshot(self) -> dict[str, Any] | None:
        return self._latest

    def register(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)
        with self._lock:
            self._subscribers.add(queue)
        if self._latest is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, self._latest)
        return queue

    def unregister(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def publish(self, record: TelemetryRecord | dict[str, Any]) -> None:
        payload = asdict(record) if isinstance(record, TelemetryRecord) else dict(record)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload["cpu_percent"] = _to_float(payload.get("cpu_percent"))
        payload["ram_percent"] = _to_float(payload.get("ram_percent"))
        payload["loss"] = _to_float(payload.get("loss"))
        payload["accuracy"] = _to_float(payload.get("accuracy"))
        payload["val_auc"] = _to_float(payload.get("val_auc"))
        payload["learning_rate"] = _to_float(payload.get("learning_rate"))
        payload["attention_mean"] = _to_float(payload.get("attention_mean"))
        payload["attention_peak"] = _to_float(payload.get("attention_peak"))
        if payload.get("vram_allocated_mb") is None and torch.cuda.is_available():
            payload["vram_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        if payload.get("vram_reserved_mb") is None and torch.cuda.is_available():
            payload["vram_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)

        self._latest = payload
        if self._loop is None:
            return

        with self._lock:
            subscribers = list(self._subscribers)

        for queue in subscribers:
            self._loop.call_soon_threadsafe(self._safe_enqueue, queue, payload)

    def _safe_enqueue(self, queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


def create_app(hub: TelemetryHub | None = None) -> FastAPI:
    """Create the FastAPI application used for the cyber HUD."""

    hub = hub or TelemetryHub()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.attach_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="TDA-GATv2 Telemetry", version="1.0.0", lifespan=lifespan)
    app.state.hub = hub

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics/latest")
    async def latest_metrics() -> dict[str, Any] | None:
        return hub.snapshot()

    @app.websocket("/ws/telemetry")
    async def telemetry_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = hub.register()
        try:
            if hub.snapshot() is not None:
                await websocket.send_json(hub.snapshot())
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            hub.unregister(queue)

    return app


class TelemetryServer:
    """Background Uvicorn server that streams live training telemetry."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.hub = TelemetryHub()
        self.app = create_app(self.hub)
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the API server in a daemon thread."""

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        server = uvicorn.Server(config)

        def _run() -> None:
            server.run()

        self._thread = threading.Thread(target=_run, name="telemetry-server", daemon=True)
        self._thread.start()

    def publish(self, record: TelemetryRecord | dict[str, Any]) -> None:
        self.hub.publish(record)


def build_runtime_snapshot() -> dict[str, float]:
    """Collect CPU and RAM usage from psutil for telemetry payloads."""

    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(interval=None),
        "ram_percent": process.memory_percent(),
    }


__all__ = ["TelemetryHub", "TelemetryRecord", "TelemetryServer", "build_runtime_snapshot", "create_app"]
