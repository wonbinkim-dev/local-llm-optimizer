from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
NO_GPU = {"gpu_name": None, "gpu_total_mib": None,
          "gpu_used_mib": None, "gpu_util_percent": None}


class AppTests(unittest.TestCase):
    @patch("optimizer.monitor.gpu_snapshot", return_value=NO_GPU)
    def test_demo_from_form_to_results(self, _):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        submit = next(b for b in app.button if b.label == "실험 실행")
        submit.click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["rows"]), 6)
        self.assertTrue(all(r["source"] == "demo" for r in app.session_state["rows"]))
        self.assertTrue(any("가상 데이터" in w.value for w in app.warning))
        self.assertGreaterEqual(len(app.dataframe), 2)

    @patch("optimizer.monitor.gpu_snapshot", return_value=NO_GPU)
    def test_missing_model_shows_validation_error(self, _):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        next(t for t in app.text_input if t.label.startswith("모델 이름")).set_value("")
        next(b for b in app.button if b.label == "실험 실행").click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("모델, 설정 후보" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
