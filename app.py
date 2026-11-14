"""Run with: python -m streamlit run app.py"""

import csv
from datetime import datetime, timezone
import io
import json
import platform
import random
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

from optimizer.benchmark import make_grid, recommend, run_config, summarize
from optimizer.client import OllamaClient
from optimizer.demo import demo_rows
from optimizer.monitor import gpu_snapshot


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = (
    "Python에서 리스트와 딕셔너리의 차이를 설명하고, 각각을 사용하기 좋은 상황을 "
    "예제 코드와 함께 5문장 이상으로 설명해 주세요."
)


def csv_export(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    # Spreadsheet applications may interpret model names / profile labels as formulas.
    for row in rows:
        writer.writerow({k: "'" + v if isinstance(v, str) and
                         v.startswith(("=", "+", "-", "@")) else v for k, v in row.items()})
    return buffer.getvalue().encode("utf-8-sig")


def render_results():
    experiment = st.session_state.get("experiment")
    if not experiment:
        st.info("설정을 선택하고 실험을 실행하세요. 처음에는 데모 모드로 화면을 확인할 수 있습니다.")
        return
    rows = st.session_state.get("rows", [])
    if not rows:
        st.info("아직 완료된 측정이 없습니다.")
        return
    demo = experiment["mode"] == "demo"
    if demo:
        st.warning("데모 결과: 모든 성능·자원 수치는 가상 데이터입니다. 하드웨어 선택 근거로 사용하지 마세요.")
    summaries = summarize(rows, experiment["repeats"])
    results = pd.DataFrame(summaries)
    ok = [r for r in rows if r["status"] == "ok"]
    a, b, c = st.columns(3)
    a.metric("성공 / 전체 측정", f"{len(ok)} / {len(rows)}")
    b.metric("실험 설정 수", len(summaries))
    c.metric("최대 생성 속도", f"{max((r['tokens_per_s'] for r in ok), default=0):.1f} tok/s")
    st.subheader("설정별 비교")
    st.caption("속도·지연·CPU는 반복 측정 중앙값, VRAM·RAM은 관측 최대값입니다. 메모리 단위: MiB.")
    st.dataframe(results, hide_index=True, width="stretch")
    if ok:
        chart = results[results.tokens_per_s.notna()].copy()
        chart["설정"] = chart.apply(
            lambda r: f"{r.model} / ctx {r.num_ctx} / GPU {r.num_gpu} / batch {r.num_batch}", axis=1)
        left, right = st.columns(2)
        with left:
            st.caption("생성 속도: 높을수록 빠름")
            st.bar_chart(chart.set_index("설정")[["tokens_per_s"]])
        with right:
            st.caption("선택 GPU의 시스템 전체 VRAM 관측 최대값")
            if chart.peak_vram_mib.notna().any():
                st.bar_chart(chart.set_index("설정")[["peak_vram_mib"]])
            else:
                st.info("NVIDIA GPU 측정값이 없습니다.")
    st.subheader("메모리 한도 내 모델별 추천" + (" · 가상 결과" if demo else ""))
    st.caption("답변 품질 평가는 포함하지 않습니다. 실패·미완료·메모리 미측정 설정은 제외합니다.")
    a, b, c = st.columns(3)
    vram = a.number_input("전체 GPU 메모리 한도 (MiB)", 0, 1000000, 3700, step=100)
    ram = b.number_input("전체 시스템 RAM 한도 (MiB)", 0, 1000000, 14000, step=100)
    context = c.selectbox("필요한 최소 문맥", [512, 1024, 2048, 4096, 8192], index=2)
    best = recommend(summaries, vram, ram, context)
    if best:
        st.dataframe(pd.DataFrame(best), hide_index=True, width="stretch")
    else:
        st.info("조건을 충족하는 설정이 없습니다. 실패 내용·측정 누락·메모리 한도를 확인하세요.")
    with st.expander("실제 응답·오류·자원 시계열 확인"):
        index = st.selectbox("측정 선택", range(len(rows)), format_func=lambda i:
                             f"{i + 1}. {rows[i]['model']} / ctx {rows[i]['num_ctx']} / "
                             f"GPU {rows[i]['num_gpu']} / 반복 {rows[i]['trial']}")
        row = rows[index]
        if row["status"] == "error":
            st.error(row["error"])
        else:
            st.text(row.get("response", ""))
            if row.get("thinking"):
                st.text(row["thinking"])
            if row.get("done_reason") == "length":
                st.warning("출력 토큰 제한에 도달했습니다. 잘린 답변인지 확인하세요.")
            if row.get("resource_samples"):
                sample_df = pd.DataFrame(row["resource_samples"]).set_index("elapsed_s")
                st.line_chart(sample_df[["ram_used_mib", "gpu_used_mib"]])
            st.json({k: v for k, v in row.items() if k not in
                     {"response", "thinking", "resource_samples", "ollama_metrics"}})
        if row.get("cleanup_warning"):
            st.warning("실험 모델 해제 실패: " + row["cleanup_warning"])
    report = {"schema_version": 1, "experiment": experiment,
              "summary": summaries, "trials": rows}
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    a, b, c = st.columns(3)
    a.download_button("전체 JSON 다운로드", report_json, "benchmark.json", "application/json")
    b.download_button("비교표 CSV 다운로드", csv_export(summaries), "summary.csv", "text/csv")
    if c.button("results 폴더에 저장"):
        directory = ROOT / "results"
        directory.mkdir(exist_ok=True)
        filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
        (directory / filename).write_text(report_json, encoding="utf-8")
        st.success(f"저장 완료: results/{filename}")


def main():
    st.set_page_config(page_title="Local LLM Optimizer", page_icon="⚡", layout="wide")
    st.title("Local LLM Optimizer")
    st.write("내 PC에서 직접 측정하고, 메모리 한도에 맞는 Ollama 설정을 찾습니다.")
    st.caption("Qwen3 1.7B · 4B 비교 | 순차 실험 | 생성 속도 · 첫 토큰 지연 · 시스템 자원")
    with st.sidebar:
        st.header("연결 및 시스템")
        demo = st.toggle("데모 모드 · 가상 데이터", value=True)
        base_url = st.text_input("로컬 Ollama 주소", "http://localhost:11434")
        gpu_index = st.number_input("측정할 NVIDIA GPU 번호", 0, 32, 0)
        gpu = gpu_snapshot(gpu_index)
        st.write(gpu["gpu_name"] or "NVIDIA GPU 측정 불가")
        if gpu["gpu_total_mib"] is not None:
            st.caption(f"VRAM {gpu['gpu_used_mib']:.0f} / {gpu['gpu_total_mib']:.0f} MiB")
        st.caption(f"시스템 RAM {psutil.virtual_memory().total / 2**30:.1f} GiB")
        if st.button("Ollama 연결 확인 / 모델 목록"):
            client = None
            try:
                client = OllamaClient(base_url)
                names = [m["name"] for m in client.get("/api/tags").get("models", [])]
                st.success(f"연결 성공 · {client.get('/api/version').get('version', '?')}")
                st.write(names or "설치된 모델이 없습니다.")
            except Exception as exc:
                st.error(str(exc))
            finally:
                if client:
                    client.close()
        st.divider()
        st.caption("실제 측정 시 다른 AI 요청과 GPU 작업을 멈추세요. 측정치는 이 PC 전체 사용량입니다.")
    with st.expander("처음 사용하는 경우: 측정 원리와 순서"):
        st.markdown(
            "1. 데모 모드로 화면을 확인합니다.\n"
            "2. Ollama 설치 후 비교할 모델을 다운로드합니다.\n"
            "3. 데모를 끄고 모델 1개·문맥 2048·GPU 자동부터 시작합니다.\n"
            "4. 반복 측정한 뒤 문맥과 GPU 레이어 후보를 늘립니다.\n"
            "5. 속도와 메모리뿐 아니라 실제 답변이 충분한지도 확인합니다.\n\n"
            "**생성 속도** = 출력 토큰 수 ÷ 생성 시간. **TTFT**는 요청부터 첫 응답/추론 토큰까지입니다. "
            "모델을 해제한 후 별도 워밍업 1회를 실행합니다. 측정은 따뜻한 상태이며 반복 질문의 "
            "프롬프트 캐시가 재사용될 수 있습니다. 자세한 이론은 `docs/THEORY.md`에 있습니다."
        )
    with st.form("benchmark"):
        models_text = st.text_input("모델 이름 · 쉼표로 구분", "qwen3:1.7b, qwen3:4b")
        a, b, c = st.columns(3)
        contexts = a.multiselect("문맥 크기 후보", [512, 1024, 2048, 4096, 8192], default=[2048])
        layers = b.multiselect("GPU 레이어 수 후보", [-1, 0, 8, 16, 24, 32], default=[-1],
                               help="-1: 자동, 0: CPU, 양수: GPU로 보낼 레이어 수 요청. GPU 개수가 아닙니다.")
        batches = c.multiselect("입력 처리 배치 후보", [32, 64, 128, 256, 512], default=[128])
        a, b, c = st.columns(3)
        repeats = a.number_input("설정별 반복 횟수", 1, 10, 3)
        output_limit = b.number_input("최대 출력 토큰", 16, 1024, 128, step=16)
        think = c.checkbox("Thinking 켜기", value=False)
        prompt = st.text_area("모든 설정에서 사용할 동일 질문", DEFAULT_PROMPT, height=100)
        profile = st.text_input("서버 프로필 메모 · 실제 적용 여부는 서버 로그로 확인", "기본 설정 / KV 캐시 미확인")
        st.caption("KV 캐시·Flash Attention은 이 화면에서 바뀌지 않습니다. 서버 재시작 후 별도 실험하세요.")
        submitted = st.form_submit_button("실험 실행", type="primary")
    if submitted:
        client = None
        try:
            models = [m.strip() for m in models_text.split(",") if m.strip()]
            if not models or not contexts or not layers or not batches or not prompt.strip():
                raise ValueError("모델, 설정 후보와 질문을 모두 입력하세요.")
            grid = make_grid(models, contexts, layers, batches, think, output_limit)
            if len(grid) > 48:
                raise ValueError("한 번에 최대 48개 설정을 지원합니다. 후보를 줄여서 나누어 실행하세요.")
            random.Random(42).shuffle(grid)
            metadata = {"version": "demo", "models": []}
            if not demo:
                client = OllamaClient(base_url)
                metadata = {"version": client.get("/api/version").get("version"),
                            "models": client.get("/api/tags").get("models", [])}
                installed = {m["name"] for m in metadata["models"]}
                missing = set(models) - installed
                if missing:
                    raise ValueError("먼저 다운로드하거나 정확한 태그를 입력하세요: " + ", ".join(sorted(missing)))
                active = client.get("/api/ps").get("models", [])
                if active:
                    raise ValueError("공정한 실험을 위해 실행 중인 모델을 먼저 종료하세요: " +
                                     ", ".join(m.get("name", "?") for m in active) +
                                     " (예: ollama stop qwen3:4b)")
            st.session_state["rows"] = []
            st.session_state["experiment"] = {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "mode": "demo" if demo else "live", "prompt": prompt,
                "repeats": repeats, "profile_note_unverified": profile,
                "gpu_index": gpu_index, "gpu": gpu, "ollama": metadata,
                "platform": platform.platform(), "python": platform.python_version(),
                "ram_total_mib": psutil.virtual_memory().total / 2**20,
                "order_seed": 42, "measurement": "warm; repeated prompts may reuse cache",
            }
            progress = st.progress(0, text="실험 준비")
            for index, config in enumerate(grid):
                progress.progress(index / len(grid), text=
                                  f"{index + 1}/{len(grid)} · {config.model} · ctx {config.num_ctx} · GPU {config.num_gpu}")
                measurements = (demo_rows(config, repeats, profile) if demo else
                                run_config(client, config, prompt, repeats, gpu_index, profile))
                for row in measurements:
                    st.session_state["rows"].append(row)
            progress.progress(1.0, text="실험 완료")
        except Exception as exc:
            st.error(str(exc))
        finally:
            if client:
                client.close()
    render_results()


if __name__ == "__main__":
    main()
