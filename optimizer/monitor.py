"""System-wide sampled resources. Missing GPU readings stay None, never zero."""

import statistics
import subprocess
import threading
import time

import psutil


def gpu_snapshot(index=0):
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={index}",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, total, used, utilization = [x.strip() for x in result.stdout.strip().split(",")]
        return {"gpu_name": name, "gpu_total_mib": float(total),
                "gpu_used_mib": float(used), "gpu_util_percent": float(utilization)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"gpu_name": None, "gpu_total_mib": None,
                "gpu_used_mib": None, "gpu_util_percent": None}


class ResourceMonitor:
    def __init__(self, gpu_index=0, interval=0.5):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self, measure_cpu=True):
        memory = psutil.virtual_memory()
        self.samples.append({
            "elapsed_s": time.perf_counter() - self.start,
            "cpu_percent": psutil.cpu_percent(interval=None) if measure_cpu else None,
            "ram_used_mib": (memory.total - memory.available) / 2**20,
            **gpu_snapshot(self.gpu_index),
        })

    def _loop(self):
        psutil.cpu_percent(interval=None)  # Prime the thread-local CPU counter.
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.start = time.perf_counter()
        self._sample(measure_cpu=False)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self._sample(measure_cpu=False)

    def summary(self):
        gpu_values = [s["gpu_used_mib"] for s in self.samples if s["gpu_used_mib"] is not None]
        cpu_values = [s["cpu_percent"] for s in self.samples if s["cpu_percent"] is not None]
        return {
            "peak_vram_mib": max(gpu_values) if gpu_values else None,
            "baseline_vram_mib": self.samples[0]["gpu_used_mib"],
            "peak_ram_mib": max(s["ram_used_mib"] for s in self.samples),
            "mean_cpu_percent": statistics.mean(cpu_values) if cpu_values else None,
            "gpu_sample_count": len(gpu_values),
            "resource_samples": self.samples,
        }
