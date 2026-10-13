import asyncio, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import prom
import httpx
from app.sources import (MetricsSource, Demuxer, DockerLogSource, sleep_state,
                         gpu_totals, _container_ports, EngineControl, base_url_of)

text = open(os.path.join(ROOT, "tests", "fixture_metrics.txt")).read()
data = prom.parse(text)
fails = []

def check(name, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-6
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f"  (want {want!r})"))
    if not ok: fails.append(name)

# --- parser --------------------------------------------------------
check("prompt_tokens", prom.total(data, "vllm:prompt_tokens_total"), 281863.0)
check("gen_tokens", prom.total(data, "vllm:generation_tokens_total"), 133323.0)
check("model label", prom.label_of(data, "vllm:generation_tokens_total", "model_name"), "Qwen3.8-Flash-Next-vLLM-W4A16")
check("sleep state", sleep_state(data), "asleep_l1")
check("by_source local_compute", prom.by_label(data, "vllm:prompt_tokens_by_source_total", "source")["local_compute"], 124663.0)
check("finish reasons", prom.by_label(data, "vllm:request_success_total", "finished_reason"), {"stop":32.0,"length":0.0,"abort":1.0})
# label value containing "[]" and commas inside the info metric must not break parsing
cfg = prom.info_labels(data, "vllm:cache_config_info")
check("cache cfg kv tokens", cfg.get("kv_cache_size_tokens"), "126445")
check("cache cfg skip layers", cfg.get("kv_cache_dtype_skip_layers"), "[]")
check("bare metric (no labels)", prom.first(data, "process_open_fds"), 71.0)
check("scientific notation", prom.first(data, "process_start_time_seconds"), 1788040645.01)

# --- histogram quantiles ------------------------------------------
ttft = prom.snapshot_histogram(data, "vllm:time_to_first_token_seconds")
summary = prom.summarize(ttft)
print("\nTTFT:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
check("ttft n", summary["n"], 34.0)
# 34 obs, p50 target = 17 -> falls in the (0.75, 1.0] bucket (13 -> 15)... verify monotonic sanity
ok = summary["p50"] < summary["p90"] < summary["p99"]
print(f"{'ok  ' if ok else 'FAIL'} quantiles monotonic: p50={summary['p50']:.2f} p90={summary['p90']:.2f} p99={summary['p99']:.2f}")
if not ok: fails.append("monotonic")
check("p50 well under the skewed mean", summary["p50"] < 5.0 and summary["mean"] > 90, True)
check("p99 catches the sleep outlier", summary["p99"] > 100, True)

# baseline diff => empty session
diffed = prom.diff_histogram(ttft, ttft)
check("session diff empties", prom.summarize(diffed), None)

# --- rates ---------------------------------------------------------
src = MetricsSource("http://unused")
NOW = time.time()
d1 = src._derive(data, NOW)
check("first poll has no rate", d1["rates"]["gen_tps"], 0.0)
check("kv usage scaled to pct", round(d1["cache"]["kv_usage_pct"], 1), 32.4)
check("prefix hit pct", round(d1["cache"]["prefix_hit_pct"], 1), 55.8)
check("kv size from cfg", d1["cache"]["kv_cache_size_tokens"], 126445)
check("endpoints sorted", d1["endpoints"][0]["handler"], "/v1/chat/completions")
check("uptime positive", d1["uptime_s"] > 0, True)

bumped = prom.parse(text.replace("vllm:generation_tokens_total{engine=\"0\",model_name=\"Qwen3.8-Flash-Next-vLLM-W4A16\"} 133323.0",
                                 "vllm:generation_tokens_total{engine=\"0\",model_name=\"Qwen3.8-Flash-Next-vLLM-W4A16\"} 133523.0"))
d2 = src._derive(bumped, NOW + 10)
check("gen rate over 10s", d2["rates"]["gen_tps"], 20.0)
# counter reset (vLLM restarted) must not produce a negative or huge spike
d3 = src._derive(prom.parse(text.replace("} 133323.0", "} 5.0")), NOW + 20)
check("counter reset -> 0 not negative", d3["rates"]["gen_tps"], 0.0)

# --- docker log demuxer -------------------------------------------
def frame(payload, stream=1):
    b = payload.encode()
    return bytes([stream,0,0,0]) + len(b).to_bytes(4,"big") + b

d = Demuxer(tty=False)
check("frame split across chunks", d.feed(frame("hello\nwor")), ["hello"])
check("continuation", d.feed(frame("ld\n")), ["world"])
# a single frame carrying several lines
check("multi-line frame", d.feed(frame("a\nb\nc\n")), ["a","b","c"])
# byte-by-byte delivery must still reassemble
d2m = Demuxer(tty=False)
out = []
for byte in frame("dripfed line\n"):
    out += d2m.feed(bytes([byte]))
check("byte-at-a-time", out, ["dripfed line"])
dt = Demuxer(tty=True)
check("tty mode raw", dt.feed(b"raw line\r\n"), ["raw line"])

print("\n--- log parsing ---")
events = []
ls = DockerLogSource.__new__(DockerLogSource)
ls.on_event = lambda k,t: events.append((k,t))
ls.engine = {}; ls.engine_at = None; ls.access_counts = {}; ls.show_noisy = False

sample = """2026-09-06T04:12:58.1Z (APIServer pid=1) INFO: 172.18.0.1:57734 - "GET /metrics HTTP/1.1" 200 OK
2026-09-06T04:12:59.0Z (APIServer pid=1) INFO 09-06 04:12:59 [api_router.py:38] wake up the engine with tags: None
2026-09-06T04:12:59.1Z (EngineCore pid=811) INFO 09-06 04:12:59 [abstract.py:356] It took 0.066009 seconds to wake up tags {'kv_cache', 'weights'}.
2026-09-06T04:13:12.0Z (APIServer pid=1) INFO: 10.210.37.227:46484 - "POST /v1/chat/completions HTTP/1.1" 200 OK
2026-09-06T04:13:13.0Z (APIServer pid=1) INFO 09-06 04:13:13 [loggers.py:310] Engine 000: Avg prompt throughput: 2282.3 tokens/s, Avg generation throughput: 117.6 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 32.4%, Prefix cache hit rate: 9.5%
2026-09-06T04:13:34.0Z (EngineCore pid=811) WARNING 09-06 04:13:34 [abstract.py:344] Executor is not sleeping.
2026-09-06T04:13:35.0Z (APIServer pid=1) INFO: 10.210.37.227:41764 - "GET /v1/models HTTP/1.1" 200 OK
2026-09-06T04:13:40.0Z (APIServer pid=1) INFO: 10.210.37.227:41764 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error"""
for line in sample.splitlines():
    ls._handle(line)

check("engine line parsed", ls.engine, {"prompt_tps":2282.3,"gen_tps":117.6,"running":2.0,"waiting":0.0,"kv_usage_pct":32.4,"prefix_hit_pct":9.5})
kinds = [k for k,_ in events]
check("noisy /metrics suppressed", "GET /metrics" in str(events), False)
check("/metrics still counted", ls.access_counts.get("GET /metrics"), 1)
check("wake events", kinds.count("wake"), 2)
check("warn captured", "warn" in kinds, True)
check("500 flagged", ("bad-request" in kinds), True)
check("request events", kinds.count("request"), 2)
print("events:", *[f"\n   {k:<12} {t}" for k,t in events])

# --- gpu roll-up ---------------------------------------------------
# Readings taken from the live 4x RTX A5000 box while idle.
four = [
    {"index":0,"util_pct":0.0,"mem_used_mb":22644.0,"mem_total_mb":24564.0,"temp_c":37.0,"power_w":19.43,"power_limit_w":230.0},
    {"index":1,"util_pct":0.0,"mem_used_mb":22999.0,"mem_total_mb":24564.0,"temp_c":54.0,"power_w":25.11,"power_limit_w":230.0},
    {"index":2,"util_pct":0.0,"mem_used_mb":22607.0,"mem_total_mb":24564.0,"temp_c":54.0,"power_w":23.07,"power_limit_w":230.0},
    {"index":3,"util_pct":0.0,"mem_used_mb":22607.0,"mem_total_mb":24564.0,"temp_c":52.0,"power_w":22.87,"power_limit_w":230.0},
]
t = gpu_totals(four)
check("gpu count", t["count"], 4)
check("system power sum", t["power_w"], 90.48)
check("system power limit", t["power_limit_w"], 920.0)
check("system power pct", t["power_pct"], 9.8)
check("system mem used", t["mem_used_mb"], 90857.0)
check("hottest card", t["temp_max_c"], 54.0)
check("mean utilisation", t["util_pct"], 0.0)

# A card that reports "[N/A]" for a field must not zero out the total.
partial = gpu_totals([{"power_w": 10.0, "temp_c": 60.0}, {"power_w": None, "temp_c": None}])
check("partial power kept", partial["power_w"], 10.0)
check("partial limit absent", partial["power_limit_w"], None)
check("no limit means no pct", partial["power_pct"], None)
check("empty roll-up", gpu_totals([])["power_w"], None)

# --- metrics url discovery ------------------------------------------
inspect_fixture = {
    "Name": "/vllm-vllm-1",
    "Config": {"ExposedPorts": {"8000/tcp": {}}},
    "NetworkSettings": {"Ports": {"8000/tcp": None}, "Networks": {"vllm_default": {"IPAddress": "172.18.0.2"}}},
}
check("ports parsed", _container_ports(inspect_fixture, 9999), [8000])
check("port fallback", _container_ports({}, 8000), [8000])

# --- engine control -------------------------------------------------
for url, want in [
    ("http://127.0.0.1:8000/metrics", "http://127.0.0.1:8000"),
    ("http://vllm-vllm-1:8000/metrics", "http://vllm-vllm-1:8000"),
    ("https://vllm.internal:443/metrics", "https://vllm.internal:443"),
    ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
]:
    check(f"base_url_of {url}", base_url_of(url), want)

seen = {}

def _record(request, status=200, text=""):
    seen.clear()
    seen.update(method=request.method, url=str(request.url),
                params=dict(request.url.params), body=request.content.decode())
    return httpx.Response(status, text=text)

async def _exercise(handler):
    ctl = EngineControl(lambda: "http://vllm:8000")
    ctl._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out = {}
    out["sleep"] = await ctl.sleep(2)
    out["sleep_seen"] = dict(seen)
    out["wake"] = await ctl.wake()
    out["wake_seen"] = dict(seen)
    await ctl.close()
    return out

r = asyncio.run(_exercise(_record))
check("sleep succeeds", r["sleep"]["ok"], True)
check("sleep is a POST", r["sleep_seen"]["method"], "POST")
check("sleep path", r["sleep_seen"]["url"].split("?")[0], "http://vllm:8000/sleep")
# vLLM reads level from the query string; the body is belt and braces.
check("sleep level in query", r["sleep_seen"]["params"].get("level"), "2")
check("sleep level in body", json.loads(r["sleep_seen"]["body"] or "{}").get("level"), 2)
check("wake path", r["wake_seen"]["url"], "http://vllm:8000/wake_up")
check("wake sends no body", r["wake_seen"]["body"], "")
check("wake succeeds", r["wake"]["ok"], True)

r404 = asyncio.run(_exercise(lambda req: _record(req, 404, "Not Found")))
check("404 surfaces failure", r404["sleep"]["ok"], False)
check("404 explains dev mode", "VLLM_SERVER_DEV_MODE" in r404["sleep"]["error"], True)

def _boom(request):
    raise httpx.ConnectError("nope", request=request)

rerr = asyncio.run(_exercise(_boom))
check("connect error handled", rerr["sleep"]["ok"], False)
check("connect error names url", "http://vllm:8000/sleep" in rerr["sleep"]["error"], True)

# --- token activity --------------------------------------------------
from app.sources import token_activity, WAKE_REQUEST_RE

check("unreachable -> inactive", token_activity(None),
      {"tokens": 0, "requests": 0, "gen_tps": 0.0, "active": False})
check("unreachable dict -> inactive",
      token_activity({"ok": False, "error": "x"})["active"], False)
idle_vllm = {"ok": True, "session": {"generation_tokens": 5000, "requests": 40},
             "rates": {"gen_tps": 0.0, "prompt_tps": 0.0, "req_per_min": 0.0},
             "queue": {"running": 0, "waiting": 0}}
# A big session total but nothing happening now is still idle.
check("session totals do not count as activity", token_activity(idle_vllm)["active"], False)
check("session totals still reported", token_activity(idle_vllm)["tokens"], 5000)
busy = dict(idle_vllm, rates=dict(idle_vllm["rates"], gen_tps=42.0))
check("gen_tps marks active", token_activity(busy)["active"], True)
check("running request marks active",
      token_activity(dict(idle_vllm, queue={"running": 2, "waiting": 0}))["active"], True)
check("waiting request marks active",
      token_activity(dict(idle_vllm, queue={"running": 0, "waiting": 1}))["active"], True)

# --- wake-request detection -------------------------------------------
lines = {
    "POST /v1/chat/completions": True,
    "POST /v1/completions": True,
    "GET /v1/chat/completions": True,
    "POST /v1/models": False,
    "GET /v1/models": False,
    "POST /health": False,
    "POST /v1/embeddings": False,
    "GET /metrics": False,
    "POST /v1/chat/completions?stream=true": True,
}
for line, want in lines.items():
    check(f"wake-request pattern: {line}", bool(WAKE_REQUEST_RE.match(line)), want)

# End-to-end through _handle: access lines fire the callback, others don't.
wakes = []
ls2 = DockerLogSource.__new__(DockerLogSource)
ls2.on_event = lambda k, t: None
ls2.on_wake_request = lambda m, p: wakes.append((m, p))
ls2.engine = {}; ls2.engine_at = None; ls2.access_counts = {}; ls2.show_noisy = False
access_lines = [
    '2026-10-13T12:25:36.0Z (APIServer pid=1) INFO: 10.1.2.3:5001 - "POST /v1/chat/completions HTTP/1.1" 200 OK',
    '2026-10-13T12:25:36.1Z (APIServer pid=1) INFO: 10.1.2.3:5001 - "GET /v1/models HTTP/1.1" 200 OK',
    '2026-10-13T12:25:37.0Z (APIServer pid=1) INFO: 10.1.2.3:5002 - "POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable',
    '2026-10-13T12:25:38.0Z (APIServer pid=1) INFO: 10.1.2.3:5003 - "GET /health HTTP/1.1" 200 OK',
]
for line in access_lines:
    ls2._handle(line)
check("wake callback fires per completions line", wakes,
      [("POST", "/v1/chat/completions"), ("POST", "/v1/chat/completions")])
check("/v1/models is not a wake request",
      all(p == "/v1/chat/completions" for _, p in wakes), True)
# A failed request (503) still wakes: the client wanted the model.
check("failed request still counts", len(wakes) == 2, True)
# No callback installed: nothing breaks.
ls3 = DockerLogSource.__new__(DockerLogSource)
ls3.on_event = lambda k, t: None
ls3.engine = {}; ls3.engine_at = None; ls3.access_counts = {}; ls3.show_noisy = False
for line in access_lines:
    ls3._handle(line)  # must not raise

# --- sleep policy ------------------------------------------------------
# Import the app module: it reads env at import time.
os.environ["AUTO_SLEEP_MINUTES"] = "0"  # keep the real app's policy quiet below
from app import main as appmod

policy = appmod.SleepPolicy()
policy.idle_minutes = 10.0
NOW = 1_000_000.0
awake_idle = {"ok": True, "engine_state": "awake",
              "session": {"generation_tokens": 0, "requests": 0},
              "rates": {"gen_tps": 0.0, "prompt_tps": 0.0},
              "queue": {"running": 0, "waiting": 0}}
awake_busy = dict(awake_idle, rates={"gen_tps": 10.0, "prompt_tps": 0.0})

policy.tick(NOW, awake_idle)                     # first poll: clock starts
check("first poll arms nothing", policy.auto_sleep_armed, False)
check("first poll records baseline", policy.last_activity == NOW, True)
policy.tick(NOW + 599, awake_idle)               # 9 min 59 s idle
check("9m59s idle does not arm", policy.auto_sleep_armed, False)
policy.tick(NOW + 600, awake_idle)               # 10 min idle
check("10m idle arms auto-sleep", policy.auto_sleep_armed, True)
policy.tick(NOW + 602, awake_busy)               # tokens resume
check("activity disarms", policy.auto_sleep_armed, False)
check("activity refreshes the clock", policy.last_activity == NOW + 602, True)
policy.tick(NOW + 603, awake_idle)
check("idle clock restarted, not armed", policy.auto_sleep_armed, False)
asleep = {"ok": True, "engine_state": "asleep_l2", "session": {}, "rates": {}, "queue": {}}
policy.auto_sleep_armed = True
policy.tick(NOW + 700, asleep)
check("asleep state clears the arm", policy.auto_sleep_armed, False)

# Disabled (0 minutes) never acts, whatever the state.
off = appmod.SleepPolicy()
off.idle_minutes = 0.0
off.tick(NOW, awake_idle)
check("disabled policy stays quiet", off.last_activity is None and not off.auto_sleep_armed, True)

# Responsive wake: only in responsive mode, only while asleep, once at a time.
woken = {"ok": 0}
async def _wake_flow():
    appmod.store.vllm = asleep
    appmod.engine_control.wake = fake_wake
    p = appmod.SleepPolicy()
    p.wake_mode = "manual"
    p.on_wake_request("POST", "/v1/chat/completions")
    await asyncio.sleep(0.05)
    manual_fired = woken["ok"]
    p.wake_mode = "responsive"
    p.on_wake_request("POST", "/v1/chat/completions")   # asleep -> fires
    p.on_wake_request("POST", "/v1/chat/completions")   # mid-wake -> ignored
    await asyncio.sleep(0.05)
    responsive_fired = woken["ok"]
    appmod.store.vllm = awake_idle                      # awake -> ignored
    p.on_wake_request("POST", "/v1/chat/completions")
    await asyncio.sleep(0.05)
    awake_fired = woken["ok"]
    return manual_fired, responsive_fired, awake_fired

async def fake_wake():
    woken["ok"] += 1
    return {"ok": True, "status": 200}

m, r, a = asyncio.run(_wake_flow())
check("manual mode never auto-wakes", m, 0)
check("responsive mode wakes once per request burst", r, 1)
check("already-awake state ignored", a, 1)

# The latch releases once tick() confirms the engine awake, and a later
# request can then trigger a fresh wake.
async def _refire_flow():
    appmod.store.vllm = asleep
    p = appmod.SleepPolicy()
    p.idle_minutes = 10.0
    p.wake_mode = "responsive"
    woken["ok"] = 0
    T = 2_000_000.0
    p.on_wake_request("POST", "/v1/chat/completions")
    p.on_wake_request("POST", "/v1/chat/completions")
    await asyncio.sleep(0.02)
    n1 = woken["ok"]
    p.tick(T, awake_idle)             # wake confirmed: latch released
    check("tick confirms the wake", p._waking, False)
    check("confirmed wake refreshes idle clock", p.last_activity == T, True)
    appmod.store.vllm = asleep
    p.on_wake_request("POST", "/v1/chat/completions")
    await asyncio.sleep(0.02)
    return n1, woken["ok"]

n1, n2 = asyncio.run(_refire_flow())
check("first burst fires once", n1, 1)
check("later request fires a fresh wake", n2, 2)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)