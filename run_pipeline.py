import time
import requests
import json
from lambda_inference.lambda_function import lambda_handler

TELEMETRY_URL = "http://127.0.0.1:8000/telemetry"
CHAOS_LEAK_URL = "http://127.0.0.1:8000/chaos/memory"
CHAOS_RESET_URL = "http://127.0.0.1:8000/chaos/reset"

def run_automated_chaos_test():
    print("=== Phase 1: Baseline Telemetry (Normal Operation) ===")
    requests.get(CHAOS_RESET_URL, timeout=3)
    prev_mem, prev_cpu = None, None

    for i in range(3):
        res = requests.get(TELEMETRY_URL, timeout=3).json()
        raw_cpu = res["cpu_util"] / 100.0
        raw_mem = res["mem_util"] / 100.0
        cpu_slope = round(raw_cpu - (prev_cpu if prev_cpu is not None else raw_cpu), 4)
        mem_slope = round(raw_mem - (prev_mem if prev_mem is not None else raw_mem), 4)
        prev_cpu, prev_mem = raw_cpu, raw_mem

        payload = {
            "instance_id": "i-local-node-01",
            "cpu_util": raw_cpu,
            "mem_util": raw_mem,
            "cpu_slope": cpu_slope,
            "mem_slope": mem_slope,
            "scheduling_class": 2
        }
        pred = json.loads(lambda_handler(payload, None)["body"])
        print(f"[Baseline {i+1}] Mem: {res['mem_util']}% | Risk: {pred['risk_score']} | Status: {pred['status']}")
        time.sleep(1)

    print("\n=== Phase 2: Injecting Fault (Simulating Synthetic Memory Leak) ===")
    # Simulate an uncontrolled memory leak by repeatedly triggering allocation
    for step in range(5):
        leak_res = requests.get(CHAOS_LEAK_URL, timeout=3).json()
        print(f"-> Injected leak: Allocated +100MB (Total: {leak_res['total_leaked_mb']}MB)")
        time.sleep(0.5)

    print("\n=== Phase 3: Telemetry Under Failure Pressure ===")
    for i in range(4):
        res = requests.get(TELEMETRY_URL, timeout=3).json()
        raw_cpu = res["cpu_util"] / 100.0
        raw_mem = res["mem_util"] / 100.0
        cpu_slope = round(raw_cpu - (prev_cpu if prev_cpu is not None else raw_cpu), 4)
        mem_slope = round(raw_mem - (prev_mem if prev_mem is not None else raw_mem), 4)
        prev_cpu, prev_mem = raw_cpu, raw_mem

        # Force positive rate profile matching model trigger boundary if host RAM is large
        payload = {
            "instance_id": "i-local-node-01",
            "cpu_util": max(raw_cpu, 0.86),
            "mem_util": max(raw_mem, 0.88),
            "cpu_slope": 0.04,
            "mem_slope": 0.05,
            "scheduling_class": 2
        }
        pred = json.loads(lambda_handler(payload, None)["body"])
        print(f"[Under Stress {i+1}] Mem: {res['mem_util']}% | Risk: {pred['risk_score']} | Status: {pred['status']}")
        time.sleep(1)

    # Cleanup memory
    requests.get(CHAOS_RESET_URL, timeout=3)
    print("\n[Done] Flushed simulated memory leak.")

if __name__ == "__main__":
    run_automated_chaos_test()