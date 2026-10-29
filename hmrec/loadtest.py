"""Latency load test: starts the API in a subprocess and measures p50/p95/p99.

``python -m hmrec.loadtest --requests 400 --concurrency 8`` runs a *cold* pass (distinct
customers, empty cache) and a *warm* pass (repeat requests, cache hits), and writes
``results/loadtest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np


def percentiles(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "n": len(a),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "p99_ms": round(float(np.percentile(a, 99)), 1),
        "mean_ms": round(float(a.mean()), 1),
    }


_local = threading.local()


def _client(base: str) -> httpx.Client:
    """One keep-alive client per worker thread (a fresh client per call measures TLS setup)."""
    if not hasattr(_local, "client"):
        _local.client = httpx.Client(base_url=base, timeout=60)
    return _local.client


def run_pass(base: str, ids: list[str], concurrency: int, lam: float) -> dict:
    def call(cid: str) -> tuple[float, float, bool]:
        t0 = time.perf_counter()
        r = _client(base).get("/recommend", params={"customer_id": cid, "lam": lam})
        r.raise_for_status()
        body = r.json()
        return (time.perf_counter() - t0) * 1000, float(body["latency_ms"]), bool(body["cached"])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        out = list(ex.map(call, ids))
    wall = time.perf_counter() - t0
    res = percentiles([o[0] for o in out])
    server = percentiles([o[1] for o in out])
    res.update(
        {
            "server_p50_ms": server["p50_ms"],
            "server_p95_ms": server["p95_ms"],
            "throughput_rps": round(len(ids) / wall, 1),
            "cache_hit_share": round(float(np.mean([o[2] for o in out])), 3),
        }
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sample-frac", default="0.1")
    ap.add_argument("--out", default="results/loadtest.json")
    args = ap.parse_args()
    env = {**os.environ, "HM_SAMPLE_FRAC": str(args.sample_frac)}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "hmrec.api:app",
            "--port",
            str(args.port),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(900):
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            raise RuntimeError("API did not start")
        health = httpx.get(f"{base}/health").json()
        ids = httpx.get(f"{base}/customers", params={"limit": min(args.requests, 100)}).json()[
            "customer_ids"
        ]
        ids = (ids * (args.requests // len(ids) + 1))[: args.requests]
        distinct = list(dict.fromkeys(ids))
        # single-request latency without contention, then cold and warm passes under load
        single = run_pass(base, distinct[:30], 1, 0.9)
        cold = run_pass(base, distinct[30:] or distinct, args.concurrency, 1.0)
        warm = run_pass(base, ids, args.concurrency, 1.0)
        # a pass where every request is a cache hit isolates HTTP + serialisation overhead
        cached = run_pass(base, distinct[30:] or distinct, args.concurrency, 1.0)
        result = {
            "health": health,
            "concurrency": args.concurrency,
            "single_request_cold": single,
            "cold_cache_under_load": cold,
            "mixed_cache_under_load": warm,
            "all_cached_under_load": cached,
        }
    finally:
        proc.terminate()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
