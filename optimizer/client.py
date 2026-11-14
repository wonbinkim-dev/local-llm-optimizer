"""Small Ollama HTTP client; importing it never starts or changes a server."""

import json
import time
from urllib.parse import urlparse

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", timeout=600):
        parsed = urlparse(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in
                {"localhost", "127.0.0.1", "::1"} or parsed.username or
                parsed.password or parsed.path not in {"", "/"} or
                parsed.query or parsed.fragment):
            raise ValueError("시스템 측정을 위해 로컬 HTTP Ollama 주소만 사용하세요.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False  # Local traffic must not use an HTTP proxy.

    def close(self):
        self.session.close()

    def _request(self, method, path, **kwargs):
        try:
            response = self.session.request(
                method, self.base_url + path, timeout=(5, min(120, self.timeout)),
                **kwargs,
            )
            if not response.ok:
                message = response.text[:1000]
                response.close()
                raise OllamaError(f"HTTP {response.status_code}: {message}")
            return response
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama 연결 실패: {exc}") from exc

    def get(self, path):
        with self._request("GET", path) as response:
            try:
                return response.json()
            except ValueError as exc:
                raise OllamaError("Ollama가 JSON이 아닌 응답을 반환했습니다.") from exc

    def unload(self, model):
        with self._request("POST", "/api/generate", json={
            "model": model, "keep_alive": 0, "stream": False,
        }) as response:
            data = response.json()
            if data.get("error"):
                raise OllamaError(str(data["error"]))

    def generate(self, payload):
        start = time.perf_counter()
        first_token = None
        answer, thinking = [], []
        final = None
        try:
            with self._request("POST", "/api/generate", json={
                **payload, "stream": True,
            }, stream=True) as response:
                for line in response.iter_lines(chunk_size=1):
                    if time.perf_counter() - start > self.timeout:
                        raise OllamaError("요청 전체 시간 제한을 초과했습니다.")
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("error"):
                        raise OllamaError(str(item["error"]))
                    if item.get("response") or item.get("thinking"):
                        if first_token is None:
                            first_token = time.perf_counter() - start
                    answer.append(item.get("response", ""))
                    thinking.append(item.get("thinking", ""))
                    if item.get("done"):
                        final = item
                        break
        except (requests.RequestException, ValueError) as exc:
            raise OllamaError(f"생성 스트림 오류: {exc}") from exc
        if final is None:
            raise OllamaError("완료 메시지 없이 스트림이 종료되었습니다.")
        count, duration = final.get("eval_count"), final.get("eval_duration")
        if not isinstance(count, int) or count <= 0 or not duration or duration <= 0:
            raise OllamaError("출력 토큰 수/생성 시간이 없어 속도를 계산할 수 없습니다.")
        return {
            "ttft_s": first_token,
            "wall_s": time.perf_counter() - start,
            "tokens_per_s": count / (duration / 1e9),
            "output_tokens": count,
            "prompt_tokens": final.get("prompt_eval_count"),
            "prompt_cached_tokens": final.get("prompt_eval_cached_count"),
            "load_s": final.get("load_duration", 0) / 1e9,
            "response": "".join(answer),
            "thinking": "".join(thinking),
            "done_reason": final.get("done_reason"),
            "ollama_metrics": final,
        }
