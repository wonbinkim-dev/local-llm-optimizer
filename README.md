# Local LLM Optimizer

**내 PC에서 Ollama 모델을 직접 비교하고, 메모리 한도에 맞는 설정을 찾는 로컬 벤치마크 도구입니다.**

Python · Ollama HTTP API · Streamlit · psutil · NVIDIA `nvidia-smi`

GTX 1650 4GB VRAM / RAM 16GB에서 Qwen3 1.7B와 4B를 비교하는 상황을 출발점으로 만들었습니다. 다른 로컬 Ollama 텍스트 모델도 정확한 태그를 입력해 사용할 수 있습니다.

## 1. 무엇을 하나요?

- 모델 × 문맥 크기 × GPU 레이어 수 × 입력 배치의 설정 조합을 순차 실행합니다.
- 별도 워밍업 후 반복 측정하여 생성 속도, 첫 토큰 지연, 응답 시간, 로딩 시간을 기록합니다.
- CPU·RAM 및 선택 NVIDIA GPU의 전체 메모리를 샘플링합니다.
- 실패한 실험도 오류와 함께 남기고 다음 설정으로 진행합니다.
- 메모리 한도와 최소 문맥 조건을 통과한 설정 중 **모델별로** 생성 속도가 가장 높은 설정을 제안합니다.
- 실제 답변과 원시 Ollama 지표를 확인하고 CSV·JSON으로 저장합니다.
- Ollama가 없어도 명확히 표시된 가상 데이터로 데모 화면을 볼 수 있습니다.

이 도구는 모델을 재학습하거나 압축하지 않습니다. 내 하드웨어에서 어떤 실행 설정이 적합한지 측정합니다. 정답률과 답변 품질을 자동 평가하지 않으므로, 속도만으로 1.7B가 4B보다 좋다고 결론 내리지 않습니다.

## 2. 이론부터 배우기

[이론 안내](docs/THEORY.md)를 먼저 읽으세요. 메모리 구성 → 설정의 의미 → 실험 → 추천 순서입니다.

코드는 다음 순서로 읽으면 됩니다.

1. `optimizer/client.py`: Ollama에 요청을 보내고 응답 스트림에서 시간을 측정합니다.
2. `optimizer/monitor.py`: 별도 스레드에서 CPU·RAM·VRAM을 관측합니다.
3. `optimizer/benchmark.py`: 설정 조합, 워밍업, 반복 실험, 요약과 추천을 담당합니다.
4. `app.py`: 설정 입력, 진행 상황, 표·그래프와 파일 다운로드 화면입니다.

## 3. 설치 및 실행 — Windows PowerShell

Python 3.11 이상을 사용하세요. 아래 명령은 `D:\dev\LocalAi` 기준입니다.

```powershell
cd D:\dev\LocalAi
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

브라우저에서 터미널에 표시되는 주소(기본 `http://127.0.0.1:8501`)를 여세요. 가상환경 활성화 없이 실행하므로 `Activate.ps1` 실행 정책 변경이 필요 없습니다.

macOS/Linux에서는 다음을 사용합니다. NVIDIA GPU가 없으면 GPU 메모리는 미측정으로 표시됩니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

### 실제 모델 준비

[Ollama](https://ollama.com/download)를 설치한 후 새 터미널에서 실행합니다.

```powershell
ollama pull qwen3:1.7b
ollama pull qwen3:4b
ollama list
```

Ollama가 서버를 실행 중이어야 합니다. 앱에서 **Ollama 연결 확인 / 모델 목록**을 누르세요. 모델명은 설치된 정확한 태그를 사용합니다. 이 앱은 모델을 자동 다운로드하지 않습니다.

## 4. 첫 번째 실험

1. 먼저 데모 모드로 실행해 표와 그래프의 의미를 확인합니다. 가상 수치는 실제 성능이 아닙니다.
2. 데모를 끄고 `qwen3:4b`, 문맥 `2048`, GPU `-1`, 배치 `128`, 반복 `3`, 출력 `128`, Thinking 끄기로 시작합니다.
3. 다른 AI 요청을 멈추고 `ollama ps`에 실행 중인 모델이 있으면 `ollama stop 모델명`으로 해제합니다. 앱이 기존 작업을 임의 종료하지는 않습니다.
4. 기본 실험이 끝나면 `qwen3:1.7b`도 넣어 같은 질문으로 비교합니다.
5. 문맥 `1024, 2048, 4096`과 GPU `-1, 0, 16`으로 확장합니다. 후보 수를 한꺼번에 많이 늘리지 마세요.
6. 모델별 추천 설정의 실제 답변을 읽고, 잘림이나 품질 저하가 없는지 확인합니다.

`num_gpu=-1`은 자동 배치, `0`은 CPU 실행, 양수는 GPU에 배치할 레이어 수 요청입니다. GPU 개수가 아닙니다. Ollama 버전·백엔드에 따라 지원/반영 방식이 다를 수 있으므로 원시 결과의 `residency`와 `ollama ps`도 확인하세요. 화면에는 **요청한 값**을 표시하며 실제 적용을 보장하지 않습니다.

짧은 질문만으로 긴 문서 처리 성능을 평가할 수는 없습니다. 처음에는 모든 문맥 후보에 충분히 들어가는 짧은 질문으로 시작하고, 실제 사용에 맞는 긴 입력은 별도 실험하세요. 모델이 입력을 자를 수 있으므로 `prompt_tokens`와 원문을 확인하세요.

## 5. 두 번째 실험 — KV 캐시 압축 비교

KV 캐시와 Flash Attention은 Ollama **서버 설정**입니다. Streamlit 프로세스의 환경변수만 바꾸면 이미 실행 중인 Ollama에는 적용되지 않습니다.

1. 트레이에서 Ollama를 완전히 종료합니다.
2. 다음 명령으로 전용 터미널에서 서버를 실행합니다.

```powershell
.\scripts\start-ollama.ps1 -CacheType q8_0
```

3. Streamlit 앱에서 서버 프로필 메모를 `flash=1, cache=q8_0 요청`으로 입력하고 실험합니다.
4. 서버를 `Ctrl+C`로 종료한 후 `f16`, 필요하면 `q4_0`으로 바꿔 같은 실험을 반복합니다. 각 결과를 JSON으로 보관하세요. 현재 버전의 비교표는 한 실험 안에서 비교합니다.

스크립트 실행이 정책으로 차단된다면 아래 명령을 PowerShell에 직접 입력할 수 있습니다. 시스템 실행 정책을 변경할 필요는 없습니다.

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_CONTEXT_LENGTH = "2048"
ollama serve
```

서버 로그에서 실제 지원·적용 여부를 확인하세요. 프로필 메모는 사용자가 적은 기록이며 API로 확인된 사실이 아닙니다. Flash Attention과 KV 캐시 압축의 지원 여부는 GPU, 모델, Ollama 버전에 따라 다릅니다.

## 6. 결과 해석과 한계

| 지표 | 의미 |
|---|---|
| `tokens_per_s` | 출력 토큰 수 / Ollama 생성 시간. 높을수록 빠릅니다. |
| `ttft_s` | 요청부터 첫 응답 또는 Thinking 토큰 수신까지의 클라이언트 관측 시간 |
| `wall_s` | HTTP 요청 시작부터 완료까지. 샘플러 준비/정리 시간 제외 |
| `load_s` | 측정 요청 안에서 Ollama가 보고한 로딩 시간. 별도 워밍업의 로딩 시간 제외 |
| `peak_vram_mib` | 선택 GPU의 **시스템 전체** VRAM 관측 최대값 |
| `peak_ram_mib` | 시스템 전체 RAM의 `total - available` 관측 최대값 |
| `mean_cpu_percent` | 측정 중 샘플 CPU 사용률의 산술평균. 전체 논리 CPU 기준 0~100% |
| `complete` | 모든 요청한 반복이 성공했는지 여부 |

- 속도·지연·출력 길이·CPU는 반복별 값의 중앙값, 메모리 값은 반복 전체의 최대값으로 요약합니다.
- 자원은 약 0.5초 대기 간격에 GPU 조회 시간을 더한 간격으로 샘플링합니다. 순간 피크는 놓칠 수 있습니다. 측정 자체에도 약간의 부하가 있습니다.
- 워밍업을 제외한 **warm inference** 벤치마크입니다. 초기 모델 로딩 중의 RAM/VRAM 최대값과 cold-start 시간은 측정하지 않습니다.
- 같은 질문을 반복하므로 입력 캐시가 재사용될 수 있습니다. TTFT는 캐시가 없는 첫 질문의 지연을 뜻하지 않습니다.
- NVIDIA GPU가 없거나 `nvidia-smi`가 실패하면 GPU 값은 `null`입니다. 0으로 대체하지 않으며 메모리 한도를 검증할 수 없어 추천에서 제외합니다. CPU·RAM과 속도 결과는 볼 수 있습니다.
- 다중 GPU에서는 선택한 1장의 수치만 관측합니다. 전체 GPU 사용량을 합산하지 않습니다. 첫 버전은 단일 GPU 실험에 맞춥니다.
- CPU·RAM·VRAM은 Ollama 프로세스 전용 수치가 아닙니다. 다른 프로그램의 사용량도 포함됩니다.
- 모델별 토큰화가 다르면 tok/s만으로 사용자 체감 속도를 완전히 비교할 수 없습니다. 응답 시간과 답변을 함께 보세요.
- `done_reason=length`인 답변은 토큰 한도로 잘렸을 수 있습니다. Thinking을 켜면 추론 토큰도 한도를 소비할 수 있습니다.
- 전체 요청 시간 제한은 기본 600초, 연결 제한은 5초, 연속 무응답 읽기 제한은 최대 120초입니다. 전체 시간 제한은 수신 이벤트에서 검사하므로 블로킹 읽기 중에는 읽기 제한만큼 늦게 종료될 수 있습니다.
- 앱을 닫거나 실행을 중단하면 서버 작업과 모델이 남을 수 있습니다. `ollama ps`로 확인하세요. UI의 협력적 중단 버튼은 아직 없습니다.
- 동시 사용자/동시 벤치마크를 지원하지 않습니다. 로컬에서 한 번에 하나의 실험을 실행하세요.
- 결과 JSON은 질문과 답변을 포함하며 `results/`는 Git에서 제외됩니다. 공개용 예시를 만들 때는 내용을 검토하세요.

## 7. 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

실제 GPU와 모델 없이도 스트림 처리, 오류 처리, 워밍업 제외, 모델 해제, 추천 제외 조건을 검증합니다. GitHub Actions는 Windows/Linux와 Python 3.11/3.13에서 같은 테스트를 실행하도록 구성했습니다. 실제 Ollama 성능 검증은 별도로 필요합니다.

## 8. GitHub에 올리기

[GitHub 업로드 안내](docs/GITHUB.md)에 새 저장소 생성, 첫 커밋, 업로드와 포트폴리오 설명 예시를 정리했습니다.

## 구조

```text
LocalAi/
├── app.py
├── optimizer/
│   ├── client.py
│   ├── monitor.py
│   ├── benchmark.py
│   └── demo.py
├── scripts/start-ollama.ps1
├── docs/
│   ├── THEORY.md
│   └── GITHUB.md
├── tests/
├── .github/workflows/tests.yml
└── requirements.txt
```

## 근거 문서

- [Ollama Generate API: 지표와 나노초 단위](https://docs.ollama.com/api/generate)
- [Ollama 실행 모델과 VRAM 배치](https://docs.ollama.com/api/ps)
- [Ollama FAQ: 문맥·서버 설정·캐시 양자화](https://docs.ollama.com/faq)
- [Ollama runtime options 소스](https://github.com/ollama/ollama/blob/main/api/types.go)
- [Ollama Thinking](https://docs.ollama.com/capabilities/thinking)
- [Streamlit Forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form)

## 다음 확장 후보

태스크별 품질 평가 세트, 여러 실험 JSON 합치기, 전체 요청 취소 버튼, 시간/전력 계측, 독립 프로세스에서 서버 프로필별 자동 재시작, 긴 문서 입력 평가를 추가할 수 있습니다. 현재 구현된 기능과는 구분합니다.
