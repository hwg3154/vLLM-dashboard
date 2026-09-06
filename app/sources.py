"""The three things this dashboard reads: /metrics, nvidia-smi, docker logs."""
from __future__ import annotations

import asyncio
import re
import shutil
import time
from typing import Any, Callable

import httpx

from . import prom

# --- 1. vLLM /metrics ---------------------------------------------------

KV_USAGE = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
PREFIX_QUERIES = ("vllm:prefix_cache_queries_total",)
PREFIX_HITS = ("vllm:prefix_cache_hits_total",)

LATENCY_HISTOGRAMS = {
    "ttft": "vllm:time_to_first_token_seconds",
    "itl": "vllm:inter_token_latency_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
}
SIZE_HISTOGRAMS = {
    "prompt_tokens": "vllm:request_prompt_tokens",
    "generation_tokens": "vllm:request_generation_tokens",
}


def sleep_state(prom_data) -> str:
    states = prom.by_label(prom_data, "vllm:engine_sleep_state", "sleep_state")
    if not states:
        return "unknown"
    if states.get("discard_all", 0) >= 1:
        return "asleep_l2"
    if states.get("weights_offloaded", 0) >= 1:
        return "asleep_l1"
    if states.get("awake", 0) >= 1:
        return "awake"
    return "unknown"


def _explain_scrape(url: str, exc: Exception) -> str:
    """ConnectTimeout with an empty str() tells the user nothing. Say more."""
    name = type(exc).__name__
    detail = str(exc).strip()
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return (
            f"can't reach {url} ({name}). The dashboard has no route to vLLM. "
            f"With network_mode: host, vLLM is on 127.0.0.1:8000; across a "
            f"bridge, put both containers on one network and scrape by name."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{url} returned HTTP {exc.response.status_code}"
    return f"{name}: {detail}" if detail else f"{name} while scraping {url}"


class MetricsSource:
    """Scrapes the Prometheus endpoint and turns counters into rates."""

    def __init__(self, url: str, timeout: float = 5.0,
                 discover: Callable[[], Any] | None = None):
        self.url = url
        self.configured_url = url
        self.timeout = timeout
        self.discover = discover
        self.ok = False
        self.error: str | None = None
        self.tried: list[str] = []
        self._prev: tuple[float, dict[str, float]] | None = None
        self._baseline: dict[str, Any] = {}
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def _fetch(self, url: str) -> str | None:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.error = _explain_scrape(url, exc)
            return None
        return resp.text

    async def _fallback(self) -> str | None:
        """Configured URL is dead. Ask Docker where vLLM actually answers."""
        configured_error = self.error
        try:
            candidates = [u for u in await self.discover() if u != self.url]
        except Exception:  # noqa: BLE001 - discovery is best-effort
            return None
        for url in candidates:
            text = await self._fetch(url)
            if text is not None:
                self.url = url  # stick with whatever works
                return text
        self.tried = candidates
        self.error = configured_error
        return None

    async def poll(self, now: float) -> dict[str, Any]:
        text = await self._fetch(self.url)
        if text is None and self.discover is not None:
            text = await self._fallback()
        if text is None:
            self.ok = False
            return {"ok": False, "error": self.error, "tried": self.tried}
        self.ok, self.error, self.tried = True, None, []
        return self._derive(prom.parse(text), now)

    def _derive(self, data, now: float) -> dict[str, Any]:
        counters = {
            "prompt": prom.total(data, "vllm:prompt_tokens_total") or 0.0,
            "generation": prom.total(data, "vllm:generation_tokens_total") or 0.0,
            "cached": prom.total(data, "vllm:prompt_tokens_cached_total") or 0.0,
            "requests": sum(prom.by_label(data, "vllm:request_success_total", "finished_reason").values()),
            "preemptions": prom.total(data, "vllm:num_preemptions_total") or 0.0,
        }
        by_source = prom.by_label(data, "vllm:prompt_tokens_by_source_total", "source")
        counters["computed"] = by_source.get("local_compute", 0.0)

        rates = {k: 0.0 for k in counters}
        if self._prev:
            prev_t, prev = self._prev
            dt = now - prev_t
            if dt > 0.05:
                for key, value in counters.items():
                    before = prev.get(key, value)
                    # A counter that went backwards means vLLM restarted.
                    rates[key] = max(0.0, value - before) / dt if value >= before else 0.0
        self._prev = (now, counters)

        latency, sizes = {}, {}
        for key, base in {**LATENCY_HISTOGRAMS, **SIZE_HISTOGRAMS}.items():
            snap = prom.snapshot_histogram(data, base)
            if not snap:
                continue
            self._baseline.setdefault(key, snap)
            target = latency if key in LATENCY_HISTOGRAMS else sizes
            target[key] = {
                "session": prom.summarize(prom.diff_histogram(snap, self._baseline[key])),
                "all": prom.summarize(snap),
            }

        queries = prom.total(data, *PREFIX_QUERIES) or 0.0
        hits = prom.total(data, *PREFIX_HITS) or 0.0
        kv = prom.first(data, *KV_USAGE)
        waiting_by_reason = prom.by_label(data, "vllm:num_requests_waiting_by_reason", "reason")
        cache_cfg = prom.info_labels(data, "vllm:cache_config_info")
        started = prom.first(data, "process_start_time_seconds")

        endpoints = []
        for labels, value in data.get("http_requests_total", ()):
            endpoints.append(
                {
                    "handler": labels.get("handler", "?"),
                    "method": labels.get("method", ""),
                    "status": labels.get("status", ""),
                    "count": value,
                }
            )
        endpoints.sort(key=lambda e: -e["count"])

        return {
            "ok": True,
            "model": prom.label_of(data, "vllm:generation_tokens_total", "model_name"),
            "engine_state": sleep_state(data),
            "uptime_s": _positive(now - started) if started else None,
            "totals": {
                "prompt_tokens": counters["prompt"],
                "generation_tokens": counters["generation"],
                "cached_tokens": counters["cached"],
                "computed_tokens": counters["computed"],
                "requests": counters["requests"],
                "preemptions": counters["preemptions"],
            },
            "rates": {
                "prompt_tps": rates["prompt"],
                "gen_tps": rates["generation"],
                "computed_tps": rates["computed"],
                "req_per_min": rates["requests"] * 60.0,
            },
            "queue": {
                "running": prom.total(data, "vllm:num_requests_running") or 0.0,
                "waiting": prom.total(data, "vllm:num_requests_waiting") or 0.0,
                "waiting_capacity": waiting_by_reason.get("capacity", 0.0),
                "waiting_deferred": waiting_by_reason.get("deferred", 0.0),
            },
            "cache": {
                "kv_usage_pct": (kv * 100.0) if kv is not None else None,
                "prefix_hit_pct": (hits / queries * 100.0) if queries else None,
                "kv_cache_size_tokens": _as_int(cache_cfg.get("kv_cache_size_tokens")),
                "gpu_memory_utilization": _as_float(cache_cfg.get("gpu_memory_utilization")),
                "block_size": _as_int(cache_cfg.get("block_size")),
            },
            "finish_reasons": prom.by_label(data, "vllm:request_success_total", "finished_reason"),
            "endpoints": endpoints[:12],
            "latency": latency,
            "sizes": sizes,
            "server": {
                "rss_bytes": prom.first(data, "process_resident_memory_bytes"),
                "cpu_seconds": prom.first(data, "process_cpu_seconds_total"),
                "open_fds": prom.first(data, "process_open_fds"),
            },
        }


def _positive(v: float) -> float | None:
    return v if v and v > 0 else None


def _as_int(v: str | None) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(v: str | None) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _container_ports(info: dict[str, Any], fallback: int) -> list[int]:
    ports: list[int] = []
    for source in (
        (info.get("NetworkSettings") or {}).get("Ports") or {},
        (info.get("Config") or {}).get("ExposedPorts") or {},
    ):
        for spec in source:
            try:
                port = int(str(spec).split("/")[0])
            except ValueError:
                continue
            if port not in ports:
                ports.append(port)
    return ports or [fallback]


async def discover_metrics_urls(socket_path: str, container: str,
                                path: str = "/metrics", fallback_port: int = 8000) -> list[str]:
    """Every address the vLLM container might answer on, per the Docker API.

    Only runs when the configured URL is already failing, so the cost of
    opening a short-lived socket client here does not matter.
    """
    transport = httpx.AsyncHTTPTransport(uds=socket_path)
    async with httpx.AsyncClient(transport=transport, base_url="http://docker") as client:
        resp = await client.get(f"/containers/{container}/json", timeout=10)
        resp.raise_for_status()
        info = resp.json()

    urls: list[str] = []

    def add(host: str, port: int) -> None:
        url = f"http://{host}:{port}{path}"
        if url not in urls:
            urls.append(url)

    networks = ((info.get("NetworkSettings") or {}).get("Networks") or {})
    name = (info.get("Name") or "").lstrip("/")
    for port in _container_ports(info, fallback_port):
        # DNS name first: it survives a container restart, an IP does not.
        if name:
            add(name, port)
        for net in networks.values():
            if net.get("IPAddress"):
                add(net["IPAddress"], port)
        add("127.0.0.1", port)
    return urls


# --- 2. nvidia-smi ------------------------------------------------------

GPU_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "fan.speed",
    "clocks.sm",
    "clocks.mem",
]
GPU_KEYS = [
    "index",
    "name",
    "util_pct",
    "mem_io_pct",
    "mem_used_mb",
    "mem_total_mb",
    "temp_c",
    "power_w",
    "power_limit_w",
    "fan_pct",
    "sm_mhz",
    "mem_mhz",
]


class GpuSource:
    """Shells out to nvidia-smi. Absent driver is a normal state, not an error."""

    def __init__(self, binary: str = "nvidia-smi"):
        self.binary = binary
        self.available = shutil.which(binary) is not None
        self.error: str | None = None if self.available else f"{binary} not found in PATH"

    async def _run(self, *args: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.binary, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=10)
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        if proc.returncode != 0:
            self.error = (err.decode(errors="replace").strip() or "nvidia-smi failed")[:200]
            return None
        self.error = None
        return out.decode(errors="replace")

    async def poll(self) -> dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": self.error, "gpus": [], "totals": {}, "procs": []}
        raw = await self._run(
            "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"
        )
        if raw is None:
            return {"ok": False, "error": self.error, "gpus": [], "totals": {}, "procs": []}

        gpus = []
        for line in raw.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) != len(GPU_KEYS):
                continue
            gpu: dict[str, Any] = {}
            for key, cell in zip(GPU_KEYS, cells):
                if key == "name":
                    gpu[key] = cell
                elif key == "index":
                    gpu[key] = _as_int(cell)
                else:
                    gpu[key] = None if cell.startswith("[") else _as_float(cell)
            gpus.append(gpu)

        procs = []
        raw_procs = await self._run(
            "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"
        )
        for line in (raw_procs or "").strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) == 3:
                procs.append(
                    {"pid": _as_int(cells[0]), "name": cells[1].split("/")[-1], "mem_mb": _as_float(cells[2])}
                )
        procs.sort(key=lambda p: -(p["mem_mb"] or 0))
        return {"ok": True, "error": None, "gpus": gpus,
                "totals": gpu_totals(gpus), "procs": procs[:8]}


def gpu_totals(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Whole-box roll-up: what every card together is drawing and holding."""
    def summed(key: str) -> float | None:
        values = [g[key] for g in gpus if g.get(key) is not None]
        return round(sum(values), 2) if values else None

    utils = [g["util_pct"] for g in gpus if g.get("util_pct") is not None]
    temps = [g["temp_c"] for g in gpus if g.get("temp_c") is not None]
    power, limit = summed("power_w"), summed("power_limit_w")
    return {
        "count": len(gpus),
        "power_w": power,
        "power_limit_w": limit,
        "power_pct": round(power / limit * 100.0, 1) if power is not None and limit else None,
        "mem_used_mb": summed("mem_used_mb"),
        "mem_total_mb": summed("mem_total_mb"),
        "util_pct": round(sum(utils) / len(utils), 1) if utils else None,
        "temp_max_c": max(temps) if temps else None,
    }


# --- 3. docker logs -----------------------------------------------------

ENGINE_LINE = "Avg prompt throughput"
ENGINE_FIELDS = {
    "prompt_tps": r"Avg prompt throughput:\s*([\d.]+)",
    "gen_tps": r"Avg generation throughput:\s*([\d.]+)",
    "running": r"Running:\s*(\d+)",
    "waiting": r"Waiting:\s*(\d+)",
    "kv_usage_pct": r"KV cache usage:\s*([\d.]+)",
    "prefix_hit_pct": r"Prefix cache hit rate:\s*([\d.]+)",
}
ENGINE_RE = {k: re.compile(v) for k, v in ENGINE_FIELDS.items()}

ACCESS_RE = re.compile(
    r'([\d.]+|\[[0-9a-fA-F:]+\]):(\d+)\s+-\s+"(\w+)\s+(\S+)\s+HTTP/[\d.]+"\s+(\d{3})'
)
WAKE_RE = re.compile(r"wake up the engine")
SLEEP_RE = re.compile(r"sleep the engine|Sleep mode freed")
WOKE_RE = re.compile(r"It took ([\d.]+) seconds to wake up")
TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT[\d:.]+Z?)\s+")
NOISY_PATHS = {"/health", "/metrics", "/ping", "/version"}


class Demuxer:
    """Docker's log stream is 8-byte-framed unless the container has a TTY."""

    def __init__(self, tty: bool):
        self.tty = tty
        self.raw = bytearray()
        self.text = bytearray()

    def feed(self, chunk: bytes) -> list[str]:
        if self.tty:
            self.text.extend(chunk)
        else:
            self.raw.extend(chunk)
            while len(self.raw) >= 8:
                length = int.from_bytes(self.raw[4:8], "big")
                if len(self.raw) < 8 + length:
                    break
                self.text.extend(self.raw[8 : 8 + length])
                del self.raw[: 8 + length]
        lines = []
        while True:
            idx = self.text.find(b"\n")
            if idx < 0:
                break
            lines.append(bytes(self.text[:idx]).decode("utf-8", "replace").rstrip("\r"))
            del self.text[: idx + 1]
        return lines


class DockerLogSource:
    """Tails one container over the Docker socket and parses vLLM's chatter."""

    def __init__(self, socket_path: str, container: str, on_event: Callable[[str, str], None],
                 tail: int = 200, show_noisy: bool = False):
        self.socket_path = socket_path
        self.container = container
        self.on_event = on_event
        self.tail = tail
        self.show_noisy = show_noisy
        self.connected = False
        self.error: str | None = None
        self.engine: dict[str, Any] = {}
        self.engine_at: float | None = None
        self.access_counts: dict[str, float] = {}
        self.container_state: str | None = None
        self._client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=socket_path), base_url="http://docker"
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _inspect(self) -> dict[str, Any]:
        resp = await self._client.get(f"/containers/{self.container}/json", timeout=10)
        resp.raise_for_status()
        return resp.json()

    async def run(self) -> None:
        """Reconnect forever: container restarts should not kill the dashboard."""
        while True:
            try:
                info = await self._inspect()
                tty = bool(info.get("Config", {}).get("Tty"))
                self.container_state = info.get("State", {}).get("Status")
                demux = Demuxer(tty)
                url = (
                    f"/containers/{self.container}/logs"
                    f"?stdout=1&stderr=1&follow=1&timestamps=1&tail={self.tail}"
                )
                async with self._client.stream("GET", url, timeout=None) as resp:
                    resp.raise_for_status()
                    self.connected, self.error = True, None
                    async for chunk in resp.aiter_raw():
                        for line in demux.feed(chunk):
                            self._handle(line)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.connected = False
                self.error = self._explain(exc)
            await asyncio.sleep(3)

    def _explain(self, exc: Exception) -> str:
        detail = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, httpx.ConnectError):
            return f"no Docker socket at {self.socket_path}. Check the volume mount. ({detail})"
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            return (f"no container named '{self.container}'. Run docker ps to check the name, "
                    f"then set VLLM_CONTAINER.")
        return detail

    def _handle(self, line: str) -> None:
        ts_match = TS_RE.match(line)
        if ts_match:
            line = line[ts_match.end():]
        if not line.strip():
            return

        if ENGINE_LINE in line:
            parsed = {}
            for key, pattern in ENGINE_RE.items():
                found = pattern.search(line)
                if found:
                    parsed[key] = float(found.group(1))
            if parsed:
                self.engine = parsed
                self.engine_at = time.time()
            return

        access = ACCESS_RE.search(line)
        if access:
            _, _, method, path, status = access.groups()
            key = f"{method} {path}"
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            base_path = path.split("?")[0]
            if base_path in NOISY_PATHS and not self.show_noisy:
                return
            kind = "request" if status.startswith("2") else "bad-request"
            self.on_event(kind, f"{method} {path} → {status}")
            return

        woke = WOKE_RE.search(line)
        if woke:
            self.on_event("wake", f"Engine awake in {float(woke.group(1)):.2f}s")
            return
        if WAKE_RE.search(line):
            self.on_event("wake", "Wake-up requested")
            return
        if SLEEP_RE.search(line):
            self.on_event("sleep", line.split("] ", 1)[-1][:200])
            return
        if " ERROR " in line or "Traceback" in line:
            self.on_event("error", line.split("] ", 1)[-1][:300])
            return
        if " WARNING " in line:
            self.on_event("warn", line.split("] ", 1)[-1][:300])
