# vLLM telemetry dashboard

A single-page dashboard for one vLLM container: token throughput, cache behaviour,
latency percentiles, GPU stats, and a live activity feed. Roughly 800 lines of
Python, one HTML file, three dependencies, no database, no Prometheus, no Grafana.

## Quick start

```bash
cd telemetry_dashboard
docker compose up -d --build
```

Open <http://localhost:8501>. Defaults assume your container is `vllm-vllm-1`
with port 8000 published on the host — no configuration needed if that holds.

## Where the numbers come from

| Source | Used for | Why |
| --- | --- | --- |
| `GET /metrics` | token counters, queue depth, cache stats, latency histograms, sleep state | Exact cumulative values |
| `docker logs` (via socket) | activity feed, wake/sleep events, warnings and errors, request lines | Events that aren't in the metrics |
| `nvidia-smi` | utilisation, memory, temperature, power, clocks, compute processes | Not exposed by vLLM at all |

Token rates are computed by differencing `vllm:prompt_tokens_total` and
`vllm:generation_tokens_total` between scrapes. The `loggers.py` lines in your
container logs report the same thing as a 10-second average, but the counters
give exact totals and a rate at whatever interval you choose.

## Reading the latency table

Percentiles come from the histogram buckets, not from `_sum / _count`. This
matters on your setup: at the time of writing your server had a mean TTFT of
93 seconds and a **median of 1.3 seconds**. The mean is dominated by two requests
that arrived while the engine was asleep and had to wait through a wake-up.

"This session" covers requests completed since the dashboard started; "All time"
covers the life of the vLLM process. If nothing has finished since the dashboard
started, the session column falls back to all-time values, greyed out.

The engine state pill reads `vllm:engine_sleep_state` and distinguishes awake,
sleep level 1 (weights offloaded) and level 2 (cache discarded). State changes
are recorded in the activity feed, so a slow request can be matched against a
sleep window.

## Configuration

Copy `.env.example` to `.env` and edit, or leave it out entirely for the defaults.

| Variable | Default | Notes |
| --- | --- | --- |
| `VLLM_CONTAINER` | `vllm-vllm-1` | From `docker ps --format '{{.Names}}'` |
| `VLLM_METRICS_URL` | `http://127.0.0.1:8000/metrics` | Host networking puts vLLM on loopback; see below |
| `BIND_HOST` | `127.0.0.1` | Under host networking there is no port mapping. Do not use `0.0.0.0` on a shared machine |
| `BIND_PORT` | `8501` | |
| `POLL_INTERVAL` | `2` | Seconds between scrapes; also the chart resolution |
| `HISTORY_MINUTES` | `30` | In-memory ring buffer, cleared on restart |
| `SHOW_HEALTH_LOGS` | `0` | Set to `1` to include `/health` and `/metrics` polls in the feed |
| `LOG_TAIL` | `200` | Backlog lines read on attach |
| `READ_DOCKER_LOGS` | `1` | Set to `0` to run without the Docker socket |
| `ENABLE_CONTROLS` | `1` | Sleep/wake buttons. `0` removes them and refuses the endpoints |
| `CONTROL_TOKEN` | *(empty)* | If set, controls require it. See below |
| `SLEEP_LEVEL` | `2` | Level the Sleep button asks for: 1 offloads weights, 2 also discards the KV cache |

### Sleep and wake

The header has **Wake** and **Sleep** buttons, posting to vLLM's `/wake_up` and
`/sleep`. Sleeping is what frees the VRAM; level 2 discards the KV cache as well
as offloading weights. The buttons target whatever host the metrics scrape
settled on, so discovery fixes them too.

Two things to know:

- vLLM reads the sleep level from the **query string**, not the request body.
  `curl -X POST .../sleep -d '{"level": 2}'` sends a body the handler ignores and
  sleeps at level 1. The dashboard sends the level both ways.
- `/sleep` and `/wake_up` exist only when vLLM runs with `VLLM_SERVER_DEV_MODE=1`.
  A 404 from these buttons means that flag is missing, and the banner says so.

**These endpoints are as exposed as the dashboard is.** Anyone who can load the
page can stop your server from serving. If it is reachable beyond your own
machine — behind a tunnel, on a LAN — either put an authenticating proxy in
front, set `ENABLE_CONTROLS=0`, or set a token:

```bash
CONTROL_TOKEN=$(openssl rand -hex 16)
```

Then open the dashboard once as `https://…/?token=YOUR_TOKEN`. It is kept in
`localStorage` and stripped from the address bar, so it stays out of your
history and out of any link you paste. The token is sent as a header over
whatever transport you are using: it keeps honest strangers out, it is not a
substitute for TLS and real auth.

### Reaching vLLM

The dashboard runs with `network_mode: host`, so vLLM is reachable at
`127.0.0.1:8000` — the same address anything else on the host uses.

The bridge alternative, `host.docker.internal` plus `extra_hosts:
host-gateway`, does resolve on Linux, but the connection still dies with a
`ConnectTimeout` in two common cases: a host firewall filtering the docker
bridge, or vLLM publishing its port bound to `127.0.0.1` only, so the gateway
address has nothing listening. Host networking sidesteps both.

If host networking is unavailable, join vLLM's own network and address it by
service name instead (there is a commented block in `docker-compose.yml`):

```yaml
    networks: [default, vllm]
    environment:
      VLLM_METRICS_URL: http://vllm:8000/metrics
      BIND_HOST: 0.0.0.0               # a port mapping protects it again
    ports: ["127.0.0.1:8501:8501"]
networks:
  vllm:
    external: true
    name: vllm_default                 # docker network ls for the real name
```

Whichever you choose, a failed scrape is self-diagnosing: the dashboard asks
the Docker API where the vLLM container actually answers, tries those
addresses, and adopts the first that works. The banner lists what it tried.

## About the Docker socket

The socket is mounted `:ro`, which prevents writing to the socket *file* but does
not restrict the Docker API — any container with access can start privileged
containers and is effectively root on the host. The dashboard only ever calls
`GET /containers/{name}/json` and `GET /containers/{name}/logs`, but you are
trusting the image, not the mount flag.

Two ways to avoid that trust:

- Set `READ_DOCKER_LOGS=0` and drop the volume. You lose the activity feed and
  keep everything else, since all the numbers come from `/metrics`.
- Put a filtering socket proxy in front of it (e.g. `tecnativa/docker-socket-proxy`
  with `CONTAINERS=1`) and set `DOCKER_SOCKET` to the proxied socket.

## Troubleshooting

Failures show as a red banner naming the exact cause; the rest of the dashboard
keeps working. Each source degrades independently.

**"no container named …"** — check `docker ps --format '{{.Names}}'` and set
`VLLM_CONTAINER`. Compose names are usually `<project>-<service>-<n>`.

**"can't reach … (ConnectTimeout)"** — the dashboard has no route to vLLM. Under
host networking, check vLLM is actually on loopback: `curl -s -o /dev/null -w
'%{http_code}\n' http://127.0.0.1:8000/health` from the host itself. A
`ConnectTimeout` rather than a refusal points at a firewall dropping the
traffic; a refusal means nothing is listening on that address. The banner also
lists the addresses discovery found via the Docker API — if one of those works,
set it as `VLLM_METRICS_URL`.

**"nvidia-smi unavailable"** — the container needs the NVIDIA runtime. If the
`deploy.resources` block doesn't work on your Docker version, replace it with
`runtime: nvidia` at the service level. Verify with
`docker exec vllm-dashboard nvidia-smi`.

**Feed is empty but the container is running** — vLLM logs to stderr and both
streams are captured, so an empty feed usually means only `/health` and
`/metrics` polls are arriving. Set `SHOW_HEALTH_LOGS=1` to confirm lines are
flowing.

## Layout

```
app/
  main.py      state store, poll loop, JSON API
  prom.py      Prometheus text parser, histogram quantiles
  sources.py   metrics scraper, nvidia-smi, docker log tailer
  static/
    index.html the whole UI: no build step, no CDN, works offline
tests/
  test_all.py  parser, quantiles, rate maths, log framing, regex coverage
```

Run the tests with `python3 tests/test_all.py` from the project root
(needs `pip install -r requirements.txt`).
