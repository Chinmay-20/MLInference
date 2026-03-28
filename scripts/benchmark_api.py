from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {"input": [1,2,3,4,5,6,7,8,9,10]}

NUM_WARMUP = 5
NUM_REQUESTS = 30
NUM_CONCURRENT = 1


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def do_request():
    start = time.perf_counter()
    response = requests.post(URL, json=PAYLOAD, timeout=30)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.raise_for_status()
    return elapsed_ms, response.json()


def main():
    print("Warming up...")
    for _ in range(NUM_WARMUP):
        latency_ms, body = do_request()
        print(f"warmup latency_ms={latency_ms:.3f} prediction={body['predictions']}")

    print("\nBenchmarking...")
    latencies = []
    start_total = time.perf_counter()

    if NUM_CONCURRENT == 1:
        for _ in range(NUM_REQUESTS):
            latency_ms, _ = do_request()
            latencies.append(latency_ms)
    else:
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as executor:
            futures = [executor.submit(do_request) for _ in range(NUM_REQUESTS)]
            for future in as_completed(futures):
                latency_ms, _ = future.result()
                latencies.append(latency_ms)

    total_time = time.perf_counter() - start_total
    throughput = NUM_REQUESTS / total_time if total_time > 0 else 0.0

    print("\nResults")
    print(f"requests      : {NUM_REQUESTS}")
    print(f"concurrency   : {NUM_CONCURRENT}")
    print(f"avg latency   : {statistics.mean(latencies):.3f} ms")
    print(f"min latency   : {min(latencies):.3f} ms")
    print(f"max latency   : {max(latencies):.3f} ms")
    print(f"p50 latency   : {percentile(latencies, 50):.3f} ms")
    print(f"p95 latency   : {percentile(latencies, 95):.3f} ms")
    print(f"p99 latency   : {percentile(latencies, 99):.3f} ms")
    print(f"throughput    : {throughput:.3f} req/s")


if __name__ == "__main__":
    main()
