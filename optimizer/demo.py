"""Deterministic, explicitly synthetic data for UI exploration without Ollama."""

from dataclasses import asdict


def demo_rows(config, repeats, profile):
    for trial in range(1, repeats + 1):
        large = "4b" in config.model
        gpu = config.num_gpu != 0
        vram = (2550 if large else 1450) + config.num_ctx * 0.12 if gpu else 350
        speed = (17 if large else 32) * (1 if gpu else 0.25) * (2048 / config.num_ctx)**0.08
        speed *= 1 + (trial % 3 - 1) * 0.03
        yield {**asdict(config), "profile": profile, "source": "demo",
               "trial": trial, "status": "ok", "error": None,
               "tokens_per_s": speed, "ttft_s": 0.4 if gpu else 1.4,
               "wall_s": config.num_predict / speed,
               "output_tokens": config.num_predict, "prompt_tokens": 40,
               "load_s": 0, "peak_vram_mib": vram, "peak_ram_mib": 7200,
               "mean_cpu_percent": 25 if gpu else 85,
               "response": "[가상 응답] 실제 모델의 답변이나 성능이 아닙니다.",
               "thinking": "", "done_reason": "length", "resource_samples": []}
