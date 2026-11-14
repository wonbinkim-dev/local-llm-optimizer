import unittest
from unittest.mock import patch

from optimizer.benchmark import Config, make_grid, recommend, run_config, summarize
from optimizer.demo import demo_rows


class FakeMonitor:
    def __init__(self, *_):
        self.samples = [1]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def summary(self):
        return {"peak_vram_mib": 3000, "peak_ram_mib": 8000}


class FakeClient:
    def __init__(self, fail=False):
        self.payloads = []
        self.unloaded = []
        self.fail = fail

    def unload(self, model):
        self.unloaded.append(model)

    def generate(self, payload):
        self.payloads.append(payload)
        if self.fail:
            raise RuntimeError("out of memory")
        return {"tokens_per_s": 20, "ttft_s": 0.2, "wall_s": 1,
                "output_tokens": 20, "response": "answer"}

    def get(self, _):
        return {"models": [{"name": "qwen3:4b", "size_vram": 2500000000}]}


class BenchmarkTests(unittest.TestCase):
    def test_grid_and_payload(self):
        grid = make_grid(["qwen3:4b", "qwen3:4b"], [1024, 2048], [-1, 0], [128])
        self.assertEqual(len(grid), 4)
        payload = grid[0].payload("hello")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_gpu"], -1)
        with self.assertRaises(ValueError):
            Config("model", num_ctx=512, num_predict=512)

    @patch("optimizer.benchmark.ResourceMonitor", FakeMonitor)
    def test_warmup_is_excluded_and_model_unloaded(self):
        client = FakeClient()
        rows = list(run_config(client, Config("qwen3:4b"), "hello", repeats=3))
        self.assertEqual(len(client.payloads), 4)
        self.assertEqual(client.payloads[0]["options"]["num_predict"], 1)
        self.assertEqual(len(rows), 3)
        self.assertEqual(client.unloaded, ["qwen3:4b"] * 2)
        self.assertEqual(rows[0]["residency"]["size_vram"], 2500000000)

    def test_warmup_failure_is_recorded_and_cleanup_runs(self):
        client = FakeClient(fail=True)
        rows = list(run_config(client, Config("qwen3:4b"), "hello"))
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(client.unloaded, ["qwen3:4b"] * 2)
        self.assertFalse(summarize(rows)[0]["complete"])

    def test_recommend_excludes_unknown_memory_failed_and_incomplete(self):
        rows = list(demo_rows(Config("qwen3:4b"), 3, "test"))
        rows[0]["peak_vram_mib"] = None
        self.assertEqual(recommend(summarize(rows), 4000, 14000, 2048), [])
        rows[0]["peak_vram_mib"] = 3000
        self.assertEqual(len(recommend(summarize(rows), 4000, 14000, 2048)), 1)
        self.assertEqual(recommend(summarize(rows[:2]), 4000, 14000, 2048), [])
        rows[0]["status"] = "error"
        self.assertEqual(recommend(summarize(rows), 4000, 14000, 2048), [])

    def test_models_are_ranked_separately_and_context_budget_respected(self):
        rows = []
        for config in make_grid(["qwen3:1.7b", "qwen3:4b"], [1024, 2048], [-1], [128]):
            rows.extend(demo_rows(config, 3, "test"))
        best = recommend(summarize(rows), 4000, 14000, 2048)
        self.assertEqual(len(best), 2)
        self.assertTrue(all(r["num_ctx"] == 2048 for r in best))
        self.assertEqual(recommend(summarize(rows), 100, 14000, 2048), [])


if __name__ == "__main__":
    unittest.main()
