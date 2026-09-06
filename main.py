"""vLLM telemetry dashboard: /metrics + nvidia-smi + docker logs in one page."""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from .sources import DockerLogSource, GpuSource, MetricsSource

STATIC = Path(__file__).parent / "static"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


CONFIG = {
    "metrics_url": env("VLLM_METRICS_URL", "http://host.docker.internal:8000/metrics"),
    "container": env("VLLM_CONTAINER", "vllm-vllm-1"),
    "docker_socket": env("DOCKER_SOCKET", "/var/run/docker.sock"),
    "poll_interval": float(env("POLL_INTERVAL", "2")),
    "history_minutes": float(env("HISTORY_MINUTES", "30")),
    "log_tail": int(env("LOG_TAIL", "200")),
    "show_health_logs": env("SHOW_HEALTH_LOGS", "0") not in ("0", "false", "no", ""),
    "read_docker_logs": env("READ_DOCKER_LOGS", "1") not in ("0", "false", "no"),
}


class Store:
    def __init__(self) -> None:
        points = max(60, int(CONFIG["history_minutes"] * 60 / CONFIG["poll_interval"]))
        self.history: deque[dict[str, Any]] = deque(maxlen=points)
        self.events: deque[dict[str, Any]] = deque(maxlen=300)
        self.event_seq = 0
        self.vllm: dict[str, Any] = {"ok": False, "error": "not polled yet"}
        self.gpu: dict[str, Any] = {"ok": False, "gpus": [], "procs": [], "error": None}
        self.started = time.time()
        self.session_start_totals: dict[str, float] | None = None

    def add_event(self, kind: str, text: str) -> None:
        self.event_seq += 1
        self.events.append({"id": self.event_seq, "ts": time.time(), "kind": kind, "text": text})


store = Store()
metrics_source = MetricsSource(CONFIG["metrics_url"])
gpu_source = GpuSource()
log_source: DockerLogSource | None = None


async def poll_loop() -> None:
    interval = CONFIG["poll_interval"]
    while True:
        cycle_start = time.time()
        vllm, gpu = await asyncio.gather(
            metrics_source.poll(cycle_start), gpu_source.poll()
        )

        if vllm.get("ok"):
            totals = vllm["totals"]
            if store.session_start_totals is None:
                store.session_start_totals = dict(totals)
            previous = store.vllm.get("engine_state") if store.vllm.get("ok") else None
            if previous and previous != vllm["engine_state"]:
                store.add_event("state", f"Engine {previous} → {vllm['engine_state']}")
            vllm["session"] = {
                k: totals.get(k, 0) - store.session_start_totals.get(k, 0) for k in totals
            }
        store.vllm, store.gpu = vllm, gpu

        gpus = gpu.get("gpus", [])
        store.history.append(
            {
                "t": round(cycle_start, 1),
                "ptps": round(vllm.get("rates", {}).get("prompt_tps", 0.0), 1) if vllm.get("ok") else None,
                "gtps": round(vllm.get("rates", {}).get("gen_tps", 0.0), 1) if vllm.get("ok") else None,
                "run": vllm.get("queue", {}).get("running") if vllm.get("ok") else None,
                "wait": vllm.get("queue", {}).get("waiting") if vllm.get("ok") else None,
                "kv": _round(vllm.get("cache", {}).get("kv_usage_pct")) if vllm.get("ok") else None,
                "gu": [g.get("util_pct") for g in gpus],
                "gm": [_pct(g.get("mem_used_mb"), g.get("mem_total_mb")) for g in gpus],
                "gt": [g.get("temp_c") for g in gpus],
                "gp": [g.get("power_w") for g in gpus],
            }
        )
        await asyncio.sleep(max(0.25, interval - (time.time() - cycle_start)))


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


def _pct(used: float | None, total: float | None) -> float | None:
    if not used or not total:
        return None
    return round(used / total * 100.0, 1)


async def api_state(request) -> JSONResponse:
    try:
        since = float(request.query_params.get("since", 0) or 0)
    except ValueError:
        since = 0.0
    try:
        after = int(request.query_params.get("after", 0) or 0)
    except ValueError:
        after = 0

    history = [p for p in store.history if p["t"] > since] if since else list(store.history)
    events = [e for e in store.events if e["id"] > after] if after else list(store.events)[-80:]

    return JSONResponse(
        {
            "now": time.time(),
            "config": {
                "container": CONFIG["container"],
                "metrics_url": CONFIG["metrics_url"],
                "poll_interval": CONFIG["poll_interval"],
                "history_minutes": CONFIG["history_minutes"],
                "dashboard_uptime_s": time.time() - store.started,
            },
            "vllm": store.vllm,
            "gpu": store.gpu,
            "logs": {
                "enabled": CONFIG["read_docker_logs"],
                "connected": bool(log_source and log_source.connected),
                "error": log_source.error if log_source else "docker log reader disabled",
                "container_state": log_source.container_state if log_source else None,
                "engine": log_source.engine if log_source else {},
                "engine_age_s": (
                    time.time() - log_source.engine_at
                    if log_source and log_source.engine_at
                    else None
                ),
                "access_counts": (
                    sorted(log_source.access_counts.items(), key=lambda kv: -kv[1])[:10]
                    if log_source
                    else []
                ),
            },
            "history": history,
            "events": events,
            "event_cursor": store.event_seq,
        }
    )


async def index(request) -> FileResponse:
    return FileResponse(STATIC / "index.html")


async def healthz(request) -> JSONResponse:
    return JSONResponse({"ok": True, "metrics": store.vllm.get("ok", False)})


@contextlib.asynccontextmanager
async def lifespan(app):
    global log_source
    tasks = [asyncio.create_task(poll_loop())]
    if CONFIG["read_docker_logs"]:
        log_source = DockerLogSource(
            CONFIG["docker_socket"],
            CONFIG["container"],
            store.add_event,
            tail=CONFIG["log_tail"],
            show_noisy=CONFIG["show_health_logs"],
        )
        tasks.append(asyncio.create_task(log_source.run()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await metrics_source.close()
        if log_source:
            await log_source.close()


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/state", api_state),
        Route("/healthz", healthz),
    ],
    lifespan=lifespan,
)
