import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import Mock, patch

from optimizer.client import OllamaClient, OllamaError


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def tearDown(self):
        self.client.close()

    def response(self, items):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_lines.return_value = [json.dumps(i).encode() for i in items]
        return response

    def test_stream_metrics_include_thinking_and_use_nanoseconds(self):
        response = self.response([
            {"thinking": "consider"}, {"response": "answer"},
            {"done": True, "eval_count": 20, "eval_duration": 2000000000},
        ])
        with patch.object(self.client, "_request", return_value=response):
            result = self.client.generate({"model": "test"})
        self.assertEqual(result["tokens_per_s"], 10)
        self.assertEqual(result["thinking"], "consider")
        self.assertEqual(result["response"], "answer")
        self.assertIsNotNone(result["ttft_s"])

    def test_truncated_error_and_zero_duration_streams_fail(self):
        for items in ([{"response": "partial"}], [{"error": "out of memory"}],
                      [{"done": True, "eval_count": 1, "eval_duration": 0}]):
            with self.subTest(items=items):
                with patch.object(self.client, "_request", return_value=self.response(items)):
                    with self.assertRaises(OllamaError):
                        self.client.generate({"model": "test"})

    def test_remote_urls_rejected(self):
        for url in ("http://example.com:11434", "http://localhost/other", "http://user@localhost"):
            with self.assertRaises(ValueError):
                OllamaClient(url)

    def test_real_http_stream_with_local_stub(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = b'\n'.join(json.dumps(item).encode() for item in [
                    {"response": "hello"},
                    {"done": True, "eval_count": 12, "eval_duration": 3000000000},
                ]) + b'\n'
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = OllamaClient(f"http://127.0.0.1:{server.server_port}")
        try:
            result = client.generate({"model": "stub", "think": False, "options": {"num_ctx": 2048}})
            self.assertEqual(result["tokens_per_s"], 4)
            self.assertEqual(result["response"], "hello")
            self.assertTrue(captured[0]["stream"])
            self.assertFalse(captured[0]["think"])
            self.assertEqual(captured[0]["options"]["num_ctx"], 2048)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
