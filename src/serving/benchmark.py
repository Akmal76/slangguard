"""Measure how long POST /predict takes for one short sentence.

    python -m src.serving.benchmark --url http://localhost:8000 --requests 100

It uses only the standard library, so it runs anywhere. Requests are sent one after another,
which is the latency one user sees on an idle server.
"""

import argparse
import json
import statistics
import time
import urllib.request


def summarize(times_ms: list[float]) -> dict[str, float]:
    # "inclusive" keeps the percentiles inside the measured range, which suits a small sample.
    percentiles = statistics.quantiles(times_ms, n=100, method="inclusive")
    return {"p50": percentiles[49], "p95": percentiles[94], "max": max(times_ms)}


def post_once(url: str, text: str) -> float:
    """Send one request and return how long it took in milliseconds."""
    request = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"texts": [text]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        response.read()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5, help="requests that are not counted")
    parser.add_argument("--text", default="makanannya enak banget, pelayanannya ramah")
    args = parser.parse_args()

    for _ in range(args.warmup):  # the first calls are slower while caches fill
        post_once(args.url, args.text)
    times = [post_once(args.url, args.text) for _ in range(args.requests)]

    stats = summarize(times)
    print(
        f"{args.requests} requests: p50 {stats['p50']:.0f} ms, p95 {stats['p95']:.0f} ms, "
        f"max {stats['max']:.0f} ms"
    )


if __name__ == "__main__":
    main()
