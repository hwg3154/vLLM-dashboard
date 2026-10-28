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

from .sources import (DockerLogSource, EngineControl, GpuSource, MetricsSource,
                      base_url_of, discover_metrics_urls, token_activity)

STATIC = Path(__file__).parent / "static"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


CONFIG = {
    "metrics_url": env("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics"),
    "container": env("VLLM_CONTAINER", "vllm-vllm-1"),
    "docker_socket": env("DOCKER_SOCKET", "/var/run/docker.sock"),
    "poll_interval": float(env("POLL_INTERVAL", "2")),
    "history_minutes": float(env("HISTORY_MINUTES", "30")),
    "log_tail": int(env("LOG_TAIL", "200")),
    "show_health_logs": env("SHOW_HEALTH_LOGS", "0") not in ("0", "false", "no", ""),
    "read_docker_logs": env("READ_DOCKER_LOGS", "1") not in ("0", "false", "no"),
    # Sleep/wake buttons. These change the served model's state, so anyone who
    # can open the dashboard can stop it serving. Set CONTROL_TOKEN if the
    # dashboard is reachable by anyone you would not trust to do that.
    "enable_controls": env("ENABLE_CONTROLS", "1") not in ("0", "false", "no"),
    "control_token": env("CONTROL_TOKEN", ""),
    "sleep_level": int(env("SLEEP_LEVEL", "2")),
    # Auto-sleep: put the engine to sleep after this many minutes without
    # token generation (0 disables it).
    "auto_sleep_minutes": float(env("AUTO_SLEEP_MINUTES", "10")),
    # "manual": only the dashboard button wakes the engine. "responsive":
    # the dashboard wakes it itself when a /v1/*completions request lands in
    # the logs. Manual is the safe default on every startup.
    "wake_mode": env("WAKE_MODE", "manual").lower(),
}
if CONFIG["wake_mode"] not in ("manual", "responsive"):
    CONFIG["wake_mode"] = "manual"


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


class SleepPolicy:
    """Session-scoped auto sleep/wake decisions.

    The env vars are startup defaults; the dashboard's input box and toggle
    override them for the life of this process and nothing else.
    """

    def __init__(self) -> None:
        self.idle_minutes: float = CONFIG["auto_sleep_minutes"]
        self.wake_mode: str = CONFIG["wake_mode"]
        # When the engine last produced a token (reset here, not at process
        # start, so the first minute of uptime never counts as idle).
        self.last_activity: float | None = None
        self.auto_sleep_armed = False
        self._waking = False
        self._waking_since: float | None = None
        self.note: str | None = None

    def on_activity(self, now: float) -> None:
        self.last_activity = now
        self.auto_sleep_armed = False
        self.note = None

    def tick(self, now: float, vllm: dict[str, Any]) -> None:
        """Runs each poll cycle.

        First it reconciles a wake that is in flight (responsive mode), then
        it walks the idle clock toward auto-sleep.
        """
        state = vllm.get("engine_state") if vllm.get("ok") else None

        if self._waking:
            if state == "awake":
                # The wake landed. The request that triggered it is about to
                # do work, so give it a fresh idle window.
                self._waking = False
                self._waking_since = None
                self.last_activity = now
                self.auto_sleep_armed = False
                if self.note and self.note.startswith("responsive"):
                    self.note = None
            elif now - (self._waking_since or now) > 120:
                # Wake was issued but never confirmed: release the latch.
                self._waking = False
                self._waking_since = None
                self.note = "responsive wake unconfirmed after 120s"
                store.add_event("error", self.note)

        if not self.idle_minutes:
            return
        if state == "awake":
            act = token_activity(vllm)
            if act["active"]:
                self.on_activity(now)
            elif self.last_activity is None:
                # First poll ever: the idle clock starts now.
                self.last_activity = now
            elif not self.auto_sleep_armed:
                idle = now - self.last_activity
                if idle >= self.idle_minutes * 60.0:
                    self.auto_sleep_armed = True
                    self.note = f"auto-sleep: {self.idle_minutes:g} idle min reached, sleeping…"
        elif state in ("asleep_l1", "asleep_l2"):
            self.auto_sleep_armed = False

    def on_wake_request(self, *args: Any) -> None:
        """A /v1/chat/completions (or /completions) line landed in the logs."""
        if self.wake_mode != "responsive":
            return
        state = store.vllm.get("engine_state")
        # The latch is set synchronously so a burst of request lines (or a
        # client retrying while the wake is in flight) cannot double-fire.
        if state not in ("asleep_l1", "asleep_l2") or self._waking:
            return
        self._waking = True
        self._waking_since = time.time()
        self.note = "responsive wake in progress…"
        # Fire and forget; tick() clears the latch once the engine is awake.
        asyncio.ensure_future(self._do_wake())

    async def _do_wake(self) -> None:
        try:
            result = await engine_control.wake()
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if result["ok"]:
            store.add_event("wake", "Responsive mode: woke the engine on a client request")
            # Latch stays set until tick() sees the engine awake, or the
            # 120s unconfirmed timeout releases it.
        else:
            self._waking = False
            self._waking_since = None
            self.note = f"responsive wake failed: {result.get('error', '')}"[:300]
            store.add_event("error", self.note)

    async def do_sleep(self) -> dict[str, Any]:
        """Called by the poll loop when the idle threshold is crossed."""
        if not self.auto_sleep_armed:
            return {"ok": True, "skipped": True}
        self.auto_sleep_armed = False
        self.note = None
        try:
            result = await engine_control.sleep(CONFIG["sleep_level"])
        except Exception as exc:  # noqa: BLE001
            self.note = f"auto-sleep failed: {exc}"[:300]
            store.add_event("error", self.note)
            return {"ok": False, "error": self.note}
        if result["ok"]:
            store.add_event("sleep", f"Auto-sleep: no tokens for {self.idle_minutes:g} min → level {CONFIG['sleep_level']}")
            self.last_activity = time.time()
        else:
            self.note = f"auto-sleep failed: {result.get('error', '')}"[:300]
            store.add_event("error", self.note)
        return result

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        idle_s = (now - self.last_activity) if (self.last_activity and self.idle_minutes) else None
        if idle_s is None:
            would_sleep_in_s = None
        elif idle_s >= self.idle_minutes * 60.0:
            would_sleep_in_s = 0
        else:
            would_sleep_in_s = round(self.idle_minutes * 60.0 - idle_s, 1)
        return {
            "enabled": bool(self.idle_minutes),
            "idle_minutes": self.idle_minutes,
            "wake_mode": self.wake_mode,
            "idle_s": round(idle_s, 1) if idle_s is not None else None,
            "would_sleep_in_s": would_sleep_in_s,
            "note": self.note,
        }


sleep_policy = SleepPolicy()
metrics_source = MetricsSource(
    CONFIG["metrics_url"],
    # Discovery reads the Docker socket, so it is only available to installs
    # that mounted it. It runs only when the configured URL is failing.
    discover=(
        (lambda: discover_metrics_urls(CONFIG["docker_socket"], CONFIG["container"]))
        if CONFIG["read_docker_logs"]
        else None
    ),
)
gpu_source = GpuSource()
# Follows metrics_source.url, so discovery fixes the controls too.
engine_control = EngineControl(lambda: base_url_of(metrics_source.url))
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

        # Auto-sleep: tick the idle clock, then act on it. Sleeping is the
        # one slow path here (weights offload), so it runs before the chart
        # point is written, not in parallel with the next poll.
        sleep_policy.tick(cycle_start, vllm)
        if sleep_policy.auto_sleep_armed:
            await sleep_policy.do_sleep()

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
                "gpwr": (gpu.get("totals") or {}).get("power_w"),
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
                # The live URL, which discovery may have moved off the configured one.
                "metrics_url": metrics_source.url,
                "metrics_url_configured": metrics_source.configured_url,
                "poll_interval": CONFIG["poll_interval"],
                "history_minutes": CONFIG["history_minutes"],
                "dashboard_uptime_s": time.time() - store.started,
                "controls_enabled": CONFIG["enable_controls"],
                "controls_need_token": bool(CONFIG["control_token"]),
                "sleep_level": CONFIG["sleep_level"],
            },
            "sleep": sleep_policy.snapshot(),
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


def _control_refused(request) -> JSONResponse | None:
    """Controls change what the server does, so they are gated separately."""
    if not CONFIG["enable_controls"]:
        return JSONResponse(
            {"ok": False, "error": "controls are disabled (ENABLE_CONTROLS=0)"}, status_code=403
        )
    token = CONFIG["control_token"]
    if token and request.headers.get("x-control-token") != token:
        return JSONResponse(
            {"ok": False, "error": "missing or wrong control token"}, status_code=403
        )
    return None


async def _run_control(request, what: str, action) -> JSONResponse:
    refused = _control_refused(request)
    if refused is not None:
        return refused
    result = await action()
    if result["ok"]:
        store.add_event("state", f"{what} requested from the dashboard")
    else:
        store.add_event("error", f"{what} failed: {result.get('error', '')}"[:300])
    return JSONResponse(result, status_code=200 if result["ok"] else 502)


async def api_sleep(request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - an empty or junk body just means "default"
        body = {}
    try:
        level = int(body.get("level", CONFIG["sleep_level"]))
    except (TypeError, ValueError):
        level = CONFIG["sleep_level"]
    if level not in (1, 2):
        return JSONResponse({"ok": False, "error": "level must be 1 or 2"}, status_code=400)
    return await _run_control(request, f"Sleep level {level}",
                              lambda: engine_control.sleep(level))


async def api_wake(request) -> JSONResponse:
    return await _run_control(request, "Wake-up", engine_control.wake)


async def api_config_get(request) -> JSONResponse:
    """Current session settings: auto-sleep minutes and wake mode."""
    return JSONResponse(
        {
            "auto_sleep_minutes": sleep_policy.idle_minutes,
            "wake_mode": sleep_policy.wake_mode,
            "sleep_level": CONFIG["sleep_level"],
        }
    )


async def api_config_set(request) -> JSONResponse:
    """Session-scoped overrides. They outlive this browser tab but not a
    restart of the dashboard process, which is the "session" the UI means."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict) or not body:
        return JSONResponse({"ok": False, "error": "expected {auto_sleep_minutes?, wake_mode?}"}, status_code=400)

    minutes = body.get("auto_sleep_minutes")
    if minutes is not None:
        try:
            minutes = float(minutes)
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "auto_sleep_minutes must be a number"}, status_code=400)
        if minutes < 0 or minutes > 24 * 60:
            return JSONResponse({"ok": False, "error": "auto_sleep_minutes must be 0–1440"}, status_code=400)
        if not minutes:
            minutes = 0.0  # 0 = auto-sleep off
        # Changing the window restarts the idle clock: the previous one is
        # no longer what the user asked for.
        sleep_policy.idle_minutes = minutes
        sleep_policy.last_activity = time.time()
        sleep_policy.auto_sleep_armed = False
        sleep_policy.note = None

    mode = body.get("wake_mode")
    if mode is not None:
        mode = str(mode).lower()
        if mode not in ("manual", "responsive"):
            return JSONResponse({"ok": False, "error": "wake_mode must be manual or responsive"}, status_code=400)
        sleep_policy.wake_mode = mode

    return JSONResponse(
        {
            "ok": True,
            "auto_sleep_minutes": sleep_policy.idle_minutes,
            "wake_mode": sleep_policy.wake_mode,
        }
    )


async def index(request) -> FileResponse:
    return FileResponse(STATIC / "index.html")


async def favicon_svg(request) -> FileResponse:
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")


async def favicon_ico(request) -> FileResponse:
    return FileResponse(STATIC / "favicon.ico", media_type="image/x-icon")


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
            on_wake_request=(
                sleep_policy.on_wake_request if CONFIG["enable_controls"] else None
            ),
        )
        tasks.append(asyncio.create_task(log_source.run()))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await metrics_source.close()
        await engine_control.close()
        if log_source:
            await log_source.close()


app = Starlette(
    routes=[
        Route("/", index),
        Route("/favicon.svg", favicon_svg),
        Route("/favicon.ico", favicon_ico),
        Route("/api/state", api_state),
        Route("/api/control/sleep", api_sleep, methods=["POST"]),
        Route("/api/control/wake", api_wake, methods=["POST"]),
        Route("/api/config", api_config_get),
        Route("/api/config", api_config_set, methods=["POST"]),
        Route("/healthz", healthz),
    ],
    lifespan=lifespan,
)
