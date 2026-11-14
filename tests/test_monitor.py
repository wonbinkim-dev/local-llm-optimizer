import subprocess
import unittest
from unittest.mock import Mock, patch

from optimizer.monitor import ResourceMonitor, gpu_snapshot


class MonitorTests(unittest.TestCase):
    @patch("optimizer.monitor.subprocess.run", side_effect=FileNotFoundError)
    def test_unavailable_gpu_stays_unknown(self, _):
        self.assertIsNone(gpu_snapshot()["gpu_used_mib"])

    @patch("optimizer.monitor.subprocess.run")
    def test_gpu_csv_and_timeout(self, run):
        run.return_value = Mock(stdout="NVIDIA GeForce GTX 1650, 4096, 2710, 85\n")
        self.assertEqual(gpu_snapshot()["gpu_used_mib"], 2710)
        run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 2)
        self.assertIsNone(gpu_snapshot()["gpu_used_mib"])

    def test_missing_cpu_and_vram_are_not_zero(self):
        monitor = ResourceMonitor()
        monitor.samples = [{"gpu_used_mib": None, "cpu_percent": None, "ram_used_mib": 5000}]
        summary = monitor.summary()
        self.assertIsNone(summary["peak_vram_mib"])
        self.assertIsNone(summary["mean_cpu_percent"])
        self.assertEqual(summary["peak_ram_mib"], 5000)


if __name__ == "__main__":
    unittest.main()
