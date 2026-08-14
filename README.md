# Community Shortform Generator

커뮤니티 신규 글과 댓글을 수집하고, 반응·자극도·대중성을 기준으로 선별해 한국어로 요약하는 Stage 1–2 파이프라인입니다. 대본·제목 생성과 게시 자동화(Stage 3)는 포함하지 않습니다.

## 지원 소스

`sources.yaml`에는 GeekNews, Hacker News, Reddit의 `hacking`, `AI_Agents`, `OpenAI`, `GeminiAI`, `claude`, DCInside 특이점이 온다 갤러리가 등록되어 있습니다. 기본값은 GeekNews만 활성화됩니다. Reddit을 활성화하려면 공식 OAuth 액세스 토큰을 `REDDIT_ACCESS_TOKEN` 환경 변수로 제공해야 합니다.

## 설치

Python 3.11 이상이 필요합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

비밀이 아닌 실행 설정은 `config.yaml`에서 관리합니다. `.env`에는 `config.yaml`이 지정한 환경변수 이름에 대응하는 비밀값만 넣으며, CLI 시작 시 자동으로 로드됩니다. 이미 설정된 운영체제 환경변수는 `.env`가 덮어쓰지 않습니다. `.env`, 키, `credentials.json`, 가상환경, JSON 산출물과 SQLite 파일은 Git에서 제외됩니다.

## 실행

기본 개발 설정은 외부 LLM 없이 전체 경계를 확인하는 fixture 모드입니다.

```powershell
python -m community_shorts run --since-hours 168
```

선택적으로 Ollama를 설치한 경우 `config.yaml`의 `llm.mode`를 `openai`로 변경하면 같은 OpenAI 호환 클라이언트를 사용합니다.

```powershell
python -m community_shorts run
```

임시 실행값은 설정 파일을 수정하지 않고 CLI에서 덮어쓸 수 있습니다.

```powershell
python -m community_shorts curate --llm-mode openai --base-url http://127.0.0.1:11434/v1 --model qwen3:8b
```

각 단계는 독립 실행할 수 있습니다.

```powershell
python -m community_shorts ingest --since-hours 24
python -m community_shorts curate --llm-mode fixture
```

출력은 `data/items.json`, `data/curated.json`, `data/state.sqlite`입니다. `curated.json`에는 원문 본문과 댓글 전문이 들어가지 않습니다.

## 운영 기본값

- 수집: 30분마다
- 선별: 09:00, 15:00, 21:00 Asia/Seoul
- 프리필터: 회차당 20개, 소스당 5개
- GeekNews 단독 시험: 최대 10개
- 결과: 회차당 2개, 목표 6개/일, 최대 8개/일
- 선별 점수: 반응 40% + 자극도 25% + 대중성 35%
- 통과 기준: 선별 점수 0.62 이상, 원문 충실도 0.75 이상, 안전성 통과

Linux cron 예시:

```cron
*/30 * * * * cd /opt/community-shortform-generator && .venv/bin/python -m community_shorts ingest
0 9,15,21 * * * cd /opt/community-shortform-generator && .venv/bin/python -m community_shorts curate
```

## 테스트

```powershell
python -m pytest -m "not live" -q
python -m pytest tests/live/test_geeknews_live.py -m live -q -s
```

실사이트 테스트는 네트워크를 사용하며 GeekNews 한 건만 수집합니다.
