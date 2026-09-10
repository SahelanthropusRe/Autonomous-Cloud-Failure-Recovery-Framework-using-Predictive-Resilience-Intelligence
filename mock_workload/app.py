import os
import time
import psutil
import threading
from fastapi import FastAPI

app = FastAPI(title="Autonomous Resilience Target Workload")

# Memory store to simulate memory leaks
leak_storage = []

@app.get("/")
def root():
    return {"message": "Resilience target workload is online"}

@app.get("/telemetry")
def get_telemetry():
    """Returns current host telemetry snapshot."""
    return {
        "cpu_util": psutil.cpu_percent(interval=0.5),
        "mem_util": psutil.virtual_memory().percent,
        "disk_util": psutil.disk_usage("/").percent
    }

@app.get("/chaos/cpu")
def spike_cpu():
    """Spikes CPU usage on background threads for 30 seconds."""
    def burn():
        start = time.time()
        while time.time() - start < 30:
            _ = 99999 * 99999

    for _ in range(os.cpu_count() or 2):
        threading.Thread(target=burn, daemon=True).start()

    return {"status": "CPU stress initiated for 30 seconds"}

@app.get("/chaos/memory")
def leak_memory():
    """Simulates an incremental memory leak (~100 MB per invocation)."""
    global leak_storage
    leak_storage.append(bytearray(100 * 1024 * 1024))
    return {
        "status": "Leaked 100MB",
        "total_leaked_mb": len(leak_storage) * 100,
        "current_mem_util": psutil.virtual_memory().percent
    }

@app.get("/chaos/reset")
def reset_state():
    """Flushes memory leak allocations back to baseline."""
    global leak_storage
    leak_storage.clear()
    return {"status": "Memory state cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)