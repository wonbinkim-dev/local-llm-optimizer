"""Sequential warm-run grid search and conservative per-model recommendations."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
import statistics

from optimizer.monitor import ResourceMonitor


@dataclass(frozen=True)
class Config:
    model: str
    num_ctx: int = 2048
    num_gpu: int = -1
    num_batch: int = 128
    think: bool = False
    num_predict: int = 128
    seed: int = 42

    def __post_init__(self):
        if not self.model.strip() or self.num_ctx < 512 or self.num_gpu < -1:
            raise ValueError("모델, 문맥 크기 또는 GPU 레이어 설정이 잘못되었습니다.")
        if self.num_batch <= 0 or self.num_predict <= 0 or self.num_predict >= self.num_ctx:
            raise ValueError("배치/출력 길이는 양수이고 출력 길이는 문맥보다 작아야 합니다.")

    def payload(self, prompt):
        return {
            "model": self.model, "prompt": prompt, "think": self.think,
            "keep_alive": "5m",
            "options": {
                "num_ctx": self.num_ctx, "num_gpu": self.num_gpu,
                "num_batch": self.num_batch, "num_predict": self.num_predict,
                "seed": self.seed, "temperature": 0,
            },
        }


def make_grid(models, contexts, gpu_layers, batches, think=False, num_predict=128):
    return [Config(m, c, g, b, think, num_predict)
            for m, c, g, b in product(dict.fromkeys(models), dict.fromkeys(contexts),
                                      dict.fromkeys(gpu_layers), dict.fromkeys(batches))]


def run_config(client, config, prompt, repeats=3, gpu_index=0, profile="미확인"):
    """Unload only the tested model; warm it before sampling each measured request."""
    if repeats < 1 or not prompt.strip():
        raise ValueError("반복 횟수와 질문을 확인하세요.")
    common = {**asdict(config), "profile": profile, "source": "live"}
    cleanup_warning = None
    rows = []
    try:
        client.unload(config.model)
        warmup = config.payload("Say ready.")
        warmup["options"] = {**warmup["options"], "num_predict": 1}
        client.generate(warmup)
        for trial in range(1, repeats + 1):
            row = {**common, "trial": trial,
                   "timestamp": datetime.now(timezone.utc).isoformat()}
            monitor = ResourceMonitor(gpu_index)
            try:
                with monitor:
                    metrics = client.generate(config.payload(prompt))
                running = client.get("/api/ps").get("models", [])
                resident = next((m for m in running if config.model in
                                 {m.get("name"), m.get("model")}), None)
                row.update(metrics)
                row.update(monitor.summary())
                row.update(status="ok", error=None, residency=resident)
            except Exception as exc:
                row.update(status="error", error=str(exc))
                if monitor.samples:
                    row.update(monitor.summary())
            rows.append(row)
    except Exception as exc:
        rows.append({**common, "trial": 0, "status": "error", "error": str(exc),
                     "timestamp": datetime.now(timezone.utc).isoformat()})
    finally:
        try:
            client.unload(config.model)
        except Exception as exc:
            cleanup_warning = str(exc)
    for row in rows:
        row["cleanup_warning"] = cleanup_warning
        yield row


KEYS = ("source", "profile", "model", "num_ctx", "num_gpu", "num_batch", "think", "num_predict", "seed")


def summarize(rows, expected_repeats=3):
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in KEYS)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, trials in groups.items():
        good = [r for r in trials if r["status"] == "ok"]
        summary = dict(zip(KEYS, key))
        summary.update(successes=len(good), failures=len(trials) - len(good),
                       complete=len(good) == expected_repeats and len(trials) == expected_repeats)
        for field in ("tokens_per_s", "ttft_s", "wall_s", "output_tokens", "mean_cpu_percent"):
            values = [r[field] for r in good if r.get(field) is not None]
            summary[field] = statistics.median(values) if values else None
        for field in ("peak_vram_mib", "peak_ram_mib"):
            values = [r[field] for r in good if r.get(field) is not None]
            summary[field] = max(values) if len(values) == len(good) and values else None
        summaries.append(summary)
    return summaries


def recommend(summaries, vram_budget_mib, ram_budget_mib, minimum_context):
    """One fastest eligible setting per model. Does not compare model quality."""
    best = {}
    for row in summaries:
        if (not row["complete"] or row["num_ctx"] < minimum_context or
                row.get("tokens_per_s") is None or
                row.get("peak_vram_mib") is None or row.get("peak_ram_mib") is None or
                row["peak_vram_mib"] > vram_budget_mib or
                row["peak_ram_mib"] > ram_budget_mib):
            continue
        key = (row["source"], row["profile"], row["model"])
        if key not in best or row["tokens_per_s"] > best[key]["tokens_per_s"]:
            best[key] = row
    return list(best.values())
