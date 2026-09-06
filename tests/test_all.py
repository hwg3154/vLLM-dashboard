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

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
