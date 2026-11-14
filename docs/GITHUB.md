# GitHub 업로드와 포트폴리오 정리

## 1. 로컬에서 먼저 실행

README의 데모 실행과 테스트를 통과시킨 뒤, 가능하면 실제 모델로 작은 실험도 수행하세요.

```powershell
cd D:\dev\LocalAi
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

처음 공개할 때에는 다음 정도가 있으면 좋습니다.

- 프로젝트 목적과 설치 명령
- 실제 화면 캡처
- 하드웨어·모델 태그·문맥·반복 횟수를 명시한 결과
- 데모 수치와 실제 측정치 구분
- 메모리 측정의 범위와 추천의 한계

모델 가중치는 올리지 않습니다. `results/`에는 질문·답변이 포함될 수 있어 기본적으로 제외했습니다. 공개용 예시가 필요하면 검토한 파일만 `examples/` 등에 따로 넣으세요. 현재 프로젝트에는 실제 벤치마크 결과나 화면 캡처가 포함되어 있지 않습니다.

## 2. 로컬 Git 저장소 만들기

현재 폴더가 이미 다른 Git 저장소에 속해 있는지 확인하세요. 독립 프로젝트로 공개하려면 프로젝트 폴더에서 다음 명령을 실행합니다.

```powershell
cd D:\dev\LocalAi
git init -b main
git status --short
git add .
git diff --cached --stat
git commit -m "Build local Ollama benchmark and optimization dashboard"
```

커밋할 때 작성자 설정이 없다는 오류가 나면 이 저장소에 사용할 본인 정보를 설정합니다.

```powershell
git config user.name "본인 이름"
git config user.email "본인 GitHub 이메일 또는 noreply 이메일"
```

그 후 `git commit`을 다시 실행하세요.

## 3. GitHub 웹에서 빈 저장소 생성

GitHub에서 새 저장소 이름을 `local-llm-optimizer`로 만듭니다. Public/Private는 본인이 선택하세요. 로컬에 README가 있으므로 GitHub에서 README·gitignore·라이선스를 자동 생성하는 옵션은 선택하지 않는 편이 처음 업로드하기 쉽습니다.

아래 `YOUR_GITHUB_ID`를 본인 계정으로 바꿉니다.

```powershell
git remote add origin https://github.com/YOUR_GITHUB_ID/local-llm-optimizer.git
git push -u origin main
```

Git Credential Manager 등의 로그인 안내를 따르세요. 토큰이나 비밀번호를 코드나 remote URL에 넣지 마세요.

## 4. GitHub CLI를 쓰는 경우

GitHub CLI가 설치되어 있다면 웹에서 빈 저장소를 만드는 단계 대신 다음을 사용할 수 있습니다. `--public`은 공개 저장소, `--private`은 비공개 저장소입니다.

```powershell
gh auth login
gh repo create local-llm-optimizer --public --source=. --remote=origin --push
```

위 명령은 GitHub 저장소 생성과 업로드를 실제로 수행합니다. 웹 방식과 CLI 방식 중 하나만 선택합니다. [GitHub 공식 안내](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github), [CLI 문서](https://cli.github.com/manual/gh_repo_create)

## 5. 라이선스 선택

공개 저장소와 오픈소스 라이선스 부여는 별개입니다. 재사용을 허용하는 범위를 정한 뒤 GitHub의 Add file → Create new file → `LICENSE`에서 템플릿을 선택할 수 있습니다. 이 프로젝트에는 사용자의 선택을 위해 라이선스를 아직 추가하지 않았습니다.

## 6. 포트폴리오 설명 예시

> VRAM 4GB 환경에서 로컬 LLM 실행 설정을 비교하기 위해 Ollama API와 Streamlit 기반 벤치마크 도구를 개발했다. 모델·문맥·GPU 레이어·입력 배치 조합을 순차 탐색하고, 별도 워밍업 후 반복 측정한 생성 속도와 시스템 자원을 기록한다. 측정 실패와 GPU 정보 누락을 추천에서 제외하고, 모델별 메모리 제약 안에서 실행 설정을 제안한다.

속도 향상률은 실제 결과를 얻은 후에 적으세요. 실험 조건, 반복 횟수, baseline 설정과 비교 설정을 함께 공개해야 의미가 있습니다.

README에 화면 캡처를 넣을 때는 다음처럼 저장합니다.

```text
docs/images/dashboard.png
```

```markdown
![Local LLM Optimizer 화면](docs/images/dashboard.png)
```

## 7. 첫 공개 후 개선하기

먼저 한 모델의 기준 실험 결과를 남기고, 한 번에 한 기능씩 이슈와 커밋으로 추가하세요. 품질 평가 세트, 실험 기록 합치기, cold-start 측정 등은 각각 독립적인 개선 주제가 됩니다.
