# Stage 1–2 Community Shorts Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that collects configured community posts into `items.json`, selects and summarizes them in Korean through an OpenAI-compatible local LLM, and writes `curated.json` without implementing Stage 3.

**Architecture:** A typed domain model and YAML source registry sit between source-specific adapters and two isolated services. Stage 1 owns HTTP ingestion and SQLite seen-state; Stage 2 owns deterministic scoring, the LLM boundary, selection gates, and Korean output. Atomic JSON files are the only boundary between stages, while injected transports and LLM clients keep tests offline and deterministic.

**Tech Stack:** Python 3.11+, Pydantic 2, httpx, selectolax, PyYAML, OpenAI Python SDK, SQLite, pytest, pytest-cov

## Global Constraints

- Implement Stage 1 and Stage 2 only; do not create script generation, title generation, title scoring, or Markdown output.
- Register all eight requested sources and enable only `geeknews` by default.
- Require Korean Stage 2 output for both Korean and English sources.
- Collect every 30 minutes; curate at 09:00, 15:00, and 21:00 Asia/Seoul.
- Prefilter at most 20 items per cycle and five per source; GeekNews-only test mode may use ten from GeekNews.
- Keep two passing items per cycle, target six per day, and enforce a hard cap of eight per day.
- Compute `curation_score = reaction_score * 0.40 + provocation_score * 0.25 + mass_appeal_score * 0.35`.
- Require `curation_score >= 0.62` and `fidelity_score >= 0.75`.
- Do not persist credentials, full original text in `curated.json`, or Stage 3 artifacts.
- Respect robots.txt, configured request delays, explicit User-Agent, timeouts, retry, and per-source failure isolation.
- Do not copy GPL or AGPL code; all implementation is original and uses dependency APIs only.

---

### Task 1: Project Skeleton, Configuration, and Domain Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `sources.yaml`
- Create: `src/community_shorts/__init__.py`
- Create: `src/community_shorts/config.py`
- Create: `src/community_shorts/models.py`
- Test: `tests/test_config.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `AppConfig`, `SourceConfig`, `load_sources(path: Path) -> list[SourceConfig]`
- Produces: `RawItem`, `Comment`, `Metrics`, `LlmAssessment`, `CuratedItem`

- [ ] **Step 1: Write failing configuration and model tests**

```python
def test_load_sources_registers_eight_sources_and_only_geeknews_is_enabled(tmp_path: Path):
    path = tmp_path / "sources.yaml"
    path.write_text(SOURCES_YAML, encoding="utf-8")
    sources = load_sources(path)
    assert len(sources) == 8
    assert [source.source_id for source in sources if source.enabled] == ["geeknews"]

def test_raw_item_rejects_naive_fetched_at():
    with pytest.raises(ValidationError):
        RawItem(
            item_id="geeknews:1", source_id="geeknews", url="https://news.hada.io/topic?id=1",
            title="제목", body="본문", source_language="ko", fetched_at=datetime(2026, 8, 14),
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_config.py tests/test_models.py -q`
Expected: FAIL because `community_shorts.config` and `community_shorts.models` do not exist.

- [ ] **Step 3: Add packaging and dependencies**

Define a `src` layout, the `community-shorts` console script, Python `>=3.11`, runtime dependencies (`httpx`, `openai`, `pydantic`, `PyYAML`, `selectolax`) and test extras (`pytest`, `pytest-cov`). Ignore `.env`, `*.key`, `credentials.json`, virtual environments, caches, `data/*.json`, `data/*.sqlite*`, and logs.

- [ ] **Step 4: Implement validated configuration and models**

Use Pydantic models with bounded scores, non-negative metrics, `HttpUrl` URLs serialized as strings, timezone-aware datetimes, and explicit `source_language`/`output_language` literals. `load_sources` must reject duplicate IDs and unsupported adapter names.

- [ ] **Step 5: Populate all source entries**

Use adapter values `geeknews`, `hackernews`, `reddit`, and `dcinside`. Set GeekNews to `enabled: true`, `language: ko`, `fetch_limit: 20`, `rate_limit_seconds: 1.0`; set all other entries disabled. Configure HN and all five Reddit entries as English, and DCInside as Korean.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_config.py tests/test_models.py -q`
Expected: PASS.

- [ ] **Step 7: Commit the contracts**

```powershell
git add pyproject.toml .gitignore sources.yaml src/community_shorts tests/test_config.py tests/test_models.py
git commit -m "feat: add pipeline configuration and data contracts"
```

### Task 2: Source Adapters and Parsing Fixtures

**Files:**
- Create: `src/community_shorts/adapters/__init__.py`
- Create: `src/community_shorts/adapters/base.py`
- Create: `src/community_shorts/adapters/geeknews.py`
- Create: `src/community_shorts/adapters/hackernews.py`
- Create: `src/community_shorts/adapters/reddit.py`
- Create: `src/community_shorts/adapters/dcinside.py`
- Create: `src/community_shorts/http.py`
- Create: `tests/fixtures/geeknews_new.html`
- Create: `tests/fixtures/geeknews_topic.html`
- Create: `tests/fixtures/hackernews_items.json`
- Create: `tests/fixtures/reddit_listing.json`
- Create: `tests/fixtures/dcinside_list.html`
- Test: `tests/adapters/test_geeknews.py`
- Test: `tests/adapters/test_hackernews.py`
- Test: `tests/adapters/test_reddit.py`
- Test: `tests/adapters/test_dcinside.py`
- Test: `tests/test_http.py`

**Interfaces:**
- Consumes: `SourceConfig`, `RawItem`, `Comment`, `Metrics`
- Produces: `SourceAdapter.fetch_new(since: datetime) -> list[RawItem]`
- Produces: `build_adapter(config: SourceConfig, client: HttpClient) -> SourceAdapter`
- Produces: `HttpClient.get_text(url: str) -> str`, `HttpClient.get_json(url: str) -> object`

- [ ] **Step 1: Save minimal synthetic fixtures and write parser tests**

```python
def test_geeknews_parser_maps_points_comments_and_topic_id(fixture_text):
    items = parse_listing(fixture_text("geeknews_new.html"), fetched_at=NOW)
    assert items[0].item_id == "geeknews:12345"
    assert items[0].metrics.likes == 42
    assert items[0].metrics.comments == 7

def test_reddit_parser_marks_english_and_ignores_stickied_posts(fixture_json):
    items = parse_listing(fixture_json("reddit_listing.json"), SOURCE, NOW)
    assert [item.source_language for item in items] == ["en"]
    assert all("stickied" not in item.item_id for item in items)
```

Fixtures must be original reduced test data, not copied page bodies. They include only the structural elements and values needed by tests.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest tests/adapters tests/test_http.py -q`
Expected: FAIL because adapters and HTTP wrapper do not exist.

- [ ] **Step 3: Implement the adapter protocol and registry**

Make adapters async to avoid blocking across independent source requests:

```python
class SourceAdapter(Protocol):
    source_id: str
    async def fetch_new(self, since: datetime) -> list[RawItem]: ...
```

`build_adapter` validates Reddit credentials only when a Reddit source is enabled.

- [ ] **Step 4: Implement source-specific parsers**

GeekNews parses `/new` and up to three discussion comments; HN fetches `newstories.json` then item JSON; Reddit consumes OAuth listing JSON and comment JSON; DCInside parses list rows and view pages. Each adapter rejects announcements/stickies, items older than `since`, malformed IDs, and empty titles.

- [ ] **Step 5: Implement polite HTTP behavior**

`HttpClient` checks `robots.txt` before HTML access, sends the configured User-Agent, sleeps for rate limit plus bounded jitter, uses a 15-second timeout, and retries 429/5xx/network errors twice with exponential backoff. It raises a typed `RobotsDeniedError` without requesting the disallowed page.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `python -m pytest tests/adapters tests/test_http.py -q`
Expected: PASS with no network access.

- [ ] **Step 7: Commit adapters**

```powershell
git add src/community_shorts/adapters src/community_shorts/http.py tests/adapters tests/fixtures tests/test_http.py
git commit -m "feat: add community source adapters"
```

### Task 3: Atomic Artifact Storage, SQLite State, and Ingestion Service

**Files:**
- Create: `src/community_shorts/storage.py`
- Create: `src/community_shorts/state.py`
- Create: `src/community_shorts/ingest.py`
- Test: `tests/test_storage.py`
- Test: `tests/test_state.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `list[SourceAdapter]`, `list[RawItem]`
- Produces: `ArtifactStore(root: Path)`, `ArtifactStore.read_items() -> list[RawItem]`, `ArtifactStore.write_items(items: Sequence[RawItem]) -> None`, `ArtifactStore.write_curated(items: Sequence[CuratedItem]) -> None`
- Produces: `StateStore.seen_ids(source_id: str) -> set[str]`, `StateStore.mark_ingested(items: Sequence[RawItem]) -> None`
- Produces: `IngestService.run(since: datetime) -> IngestReport`

- [ ] **Step 1: Write failing atomic-write and state-ordering tests**

```python
async def test_ingest_marks_seen_only_after_artifact_write(tmp_path: Path):
    storage = FailingStorage()
    state = StateStore(tmp_path / "state.sqlite")
    service = IngestService([FakeAdapter([RAW_ITEM])], storage, state)
    with pytest.raises(OSError):
        await service.run(SINCE)
    assert state.seen_ids("geeknews") == set()

async def test_one_source_failure_keeps_other_source_results(tmp_path: Path):
    report = await service_with_one_good_and_one_bad_source(tmp_path).run(SINCE)
    assert report.collected == 1
    assert report.failed_sources == ["broken"]
```

- [ ] **Step 2: Run storage/ingest tests and verify RED**

Run: `python -m pytest tests/test_storage.py tests/test_state.py tests/test_ingest.py -q`
Expected: FAIL because storage, state, and ingestion modules do not exist.

- [ ] **Step 3: Implement atomic JSON storage**

Write UTF-8 JSON with Korean preserved, flush and close the temporary sibling file, then replace the target with `Path.replace`. Merge existing items by `item_id` without destroying records from a prior run.

- [ ] **Step 4: Implement SQLite state transitions**

Create `items(item_id PRIMARY KEY, source_id, ingested_at, curated_at, status, error)` and `runs(run_id PRIMARY KEY, stage, started_at, finished_at, status, details_json)`. Use transactions and parameterized SQL only.

- [ ] **Step 5: Implement isolated ingestion orchestration**

Run enabled adapters independently, filter seen IDs and same-run duplicates, atomically write the merged artifact, then mark the new IDs ingested. Log source failures and continue; raise only when all enabled sources fail.

- [ ] **Step 6: Run tests and verify GREEN**

Run: `python -m pytest tests/test_storage.py tests/test_state.py tests/test_ingest.py -q`
Expected: PASS.

- [ ] **Step 7: Commit Stage 1 orchestration**

```powershell
git add src/community_shorts/storage.py src/community_shorts/state.py src/community_shorts/ingest.py tests/test_storage.py tests/test_state.py tests/test_ingest.py
git commit -m "feat: persist ingested items safely"
```

### Task 4: Deterministic Prefilter and Scoring Gates

**Files:**
- Create: `src/community_shorts/scoring.py`
- Create: `src/community_shorts/prefilter.py`
- Test: `tests/test_scoring.py`
- Test: `tests/test_prefilter.py`

**Interfaces:**
- Consumes: `Sequence[RawItem]`, `LlmAssessment`
- Produces: `reaction_scores(items: Sequence[RawItem]) -> dict[str, float]`
- Produces: `curation_score(reaction: float, provocation: float, mass_appeal: float) -> float`
- Produces: `prefilter(items, global_limit: int, per_source_limit: int) -> list[ScoredRawItem]`
- Produces: `passes_gates(score: float, fidelity: float, safe: bool) -> bool`

- [ ] **Step 1: Write failing literal-expectation scoring tests**

```python
def test_curation_score_uses_40_25_35_weights():
    assert curation_score(1.0, 0.4, 0.2) == pytest.approx(0.57)

def test_fidelity_below_point_75_fails_even_with_high_scores():
    assert passes_gates(score=0.99, fidelity=0.74, safe=True) is False

def test_prefilter_enforces_source_diversity():
    selected = prefilter(MIXED_ITEMS, global_limit=4, per_source_limit=2)
    assert Counter(item.raw.source_id for item in selected) == {"a": 2, "b": 2}
```

- [ ] **Step 2: Run scoring tests and verify RED**

Run: `python -m pytest tests/test_scoring.py tests/test_prefilter.py -q`
Expected: FAIL because scoring functions do not exist.

- [ ] **Step 3: Implement independent score calculation**

Compute source-local average-rank percentiles over `log1p(max(metric, 0))`, using `0.5` for a one-item source group. Round serialized scores to six decimals but compare unrounded values at gates.

- [ ] **Step 4: Implement duplicate/spam filtering and quotas**

Canonicalize URLs by removing fragments and known tracking parameters. Drop duplicate URLs, empty content, and configurable spam patterns. Rank by reaction score, then fetched time, then item ID for deterministic ties. Enforce source and global limits.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_scoring.py tests/test_prefilter.py -q`
Expected: PASS.

- [ ] **Step 6: Commit deterministic selection**

```powershell
git add src/community_shorts/scoring.py src/community_shorts/prefilter.py tests/test_scoring.py tests/test_prefilter.py
git commit -m "feat: add deterministic curation scoring"
```

### Task 5: OpenAI-Compatible LLM Client and Korean Curation Service

**Files:**
- Create: `src/community_shorts/llm.py`
- Create: `src/community_shorts/prompts.py`
- Create: `src/community_shorts/curate.py`
- Test: `tests/test_llm.py`
- Test: `tests/test_curate.py`

**Interfaces:**
- Consumes: `ScoredRawItem`, `AppConfig`, `StateStore`
- Produces: `LlmClient.assess(item: ScoredRawItem) -> LlmAssessment`
- Produces: `OpenAiLlmClient`, `FixtureLlmClient`
- Produces: `CurateService.run(now: datetime) -> CurateReport`

- [ ] **Step 1: Write failing retry, Korean-output, and daily-cap tests**

```python
async def test_client_retries_invalid_json_once():
    transport = SequencedChat(["not-json", VALID_ASSESSMENT_JSON])
    result = await OpenAiLlmClient(transport=transport, model="test").assess(SCORED_ITEM)
    assert result.output_language == "ko"
    assert transport.calls == 2

async def test_english_item_prompt_requires_korean_output():
    client = CapturingFixtureLlmClient()
    await client.assess(ENGLISH_ITEM)
    assert client.last_payload["source_language"] == "en"
    assert client.last_payload["output_language"] == "ko"

async def test_curate_keeps_two_per_cycle_and_never_exceeds_eight_per_day(tmp_path: Path):
    report = await make_curator(tmp_path, candidates=12, already_curated_today=7).run(NOW)
    assert report.passed == 1
```

- [ ] **Step 2: Run LLM and curation tests and verify RED**

Run: `python -m pytest tests/test_llm.py tests/test_curate.py -q`
Expected: FAIL because the LLM boundary and curation service do not exist.

- [ ] **Step 3: Implement the Korean schema prompt**

The system prompt requires a single JSON object, Korean narrative fields, preserved proper nouns, grounded claims only, bounded scores, and empty summary fields when unsafe or spam. The user payload contains source language, title, body, metrics, up to three comments, and the deterministic reaction score.

- [ ] **Step 4: Implement production and fixture LLM clients**

Use `AsyncOpenAI` with configurable `base_url`, API key, model, and timeout. Request JSON schema output, validate it with Pydantic, and retry once with a correction message. `FixtureLlmClient` returns deterministic Korean content based on input title and is selected only by explicit `--llm-mode fixture`.

- [ ] **Step 5: Implement curation orchestration and independent gates**

Prefilter candidates, assess each independently, replace any model-provided reaction or combined score with locally calculated values, sort eligible results, apply two-per-cycle and eight-per-day limits, write `curated.json` atomically, then mark selected IDs curated. Do not copy `RawItem.body` or full comment text into `CuratedItem`.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_llm.py tests/test_curate.py -q`
Expected: PASS.

- [ ] **Step 7: Commit Stage 2**

```powershell
git add src/community_shorts/llm.py src/community_shorts/prompts.py src/community_shorts/curate.py tests/test_llm.py tests/test_curate.py
git commit -m "feat: curate and summarize items in Korean"
```

### Task 6: CLI, End-to-End Verification, and Operating Documentation

**Files:**
- Create: `src/community_shorts/cli.py`
- Create: `src/community_shorts/__main__.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_e2e.py`
- Create: `tests/live/test_geeknews_live.py`
- Create: `README.md`

**Interfaces:**
- Consumes: all prior public interfaces
- Produces: CLI commands `ingest`, `curate`, and `run`
- Produces: `main(argv: Sequence[str] | None = None) -> int`
- Produces: process exit codes `0` success, `1` stage failure, `2` configuration/usage error

- [ ] **Step 1: Write failing CLI and end-to-end tests**

```python
async def test_fixture_run_writes_only_stage_one_and_two_artifacts(tmp_path: Path):
    ingest = IngestService([FakeAdapter([RAW_ITEM])], ArtifactStore(tmp_path), StateStore(tmp_path / "state.sqlite"))
    await ingest.run(SINCE)
    curate = CurateService(ArtifactStore(tmp_path), StateStore(tmp_path / "state.sqlite"), FixtureLlmClient())
    await curate.run(NOW)
    assert (tmp_path / "items.json").exists()
    assert (tmp_path / "curated.json").exists()
    assert not (tmp_path / "scripts").exists()

def test_help_exits_successfully(capsys):
    assert main(["--help"]) == 0
    assert "ingest" in capsys.readouterr().out

@pytest.mark.live
async def test_geeknews_live_returns_at_least_one_well_formed_item():
    items = await live_geeknews_adapter().fetch_new(datetime.now(UTC) - timedelta(days=7))
    assert items and all(item.source_id == "geeknews" for item in items)
```

- [ ] **Step 2: Run CLI and E2E tests and verify RED**

Run: `python -m pytest tests/test_cli.py tests/test_e2e.py -q`
Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement CLI composition and structured logging**

Use standard-library `argparse` and `logging` to avoid another dependency. Support `--sources`, `--data-dir`, `--state-db`, `--since-hours`, `--llm-mode`, `--base-url`, `--model`, and `--log-level`. Never print API keys or original full bodies.

- [ ] **Step 4: Document installation and operations**

Document Python 3.11 virtual environment setup, editable install, fixture run, real Furiosa run, individual stages, environment variables, output contracts, cron examples for 30-minute ingestion and three daily curation runs, Reddit credential requirements, and the explicit Stage 3 exclusion.

- [ ] **Step 5: Run offline suite and verify GREEN**

Run: `python -m pytest -m "not live" -q`
Expected: PASS with no network or LLM server.

- [ ] **Step 6: Run GeekNews live smoke test**

Run: `python -m pytest tests/live/test_geeknews_live.py -m live -q -s`
Expected: PASS with at least one well-formed live item. If it fails, print the HTTP status, URL, timeout/connection class, configured delay, and parser item count before changing code.

- [ ] **Step 7: Run the acceptance pipeline**

Run: `python -m community_shorts run --llm-mode fixture --since-hours 168 --log-level INFO`
Expected: exit `0`, non-empty `data/items.json`, schema-valid `data/curated.json`, and no `scripts` directory.

- [ ] **Step 8: Inspect changes and repository safety**

Run: `git diff --check`
Run: `git status --short`
Run: `git diff -- . ':(exclude)PRD_community_shorts_pipeline.md'`
Expected: no whitespace errors, no secrets, no `.env`, keys, credentials, dependency directories, build outputs, or Stage 3 artifacts.

- [ ] **Step 9: Commit verified pipeline**

```powershell
git add README.md src/community_shorts tests
git commit -m "feat: expose stage 1 and 2 pipeline CLI"
```

### Task 7: Final Regression and Requirement Audit

**Files:**
- Modify only files implicated by a failing test or audit finding.

**Interfaces:**
- Consumes: complete Stage 1–2 application
- Produces: verified repository state and acceptance report

- [ ] **Step 1: Run the complete offline regression suite**

Run: `python -m pytest -m "not live" --cov=community_shorts --cov-report=term-missing -q`
Expected: all tests pass and every new non-trivial function has behavioral coverage.

- [ ] **Step 2: Verify CLI help and configuration errors**

Run: `python -m community_shorts --help`
Run: `python -m community_shorts curate --llm-mode openai --base-url http://127.0.0.1:1/v1`
Expected: help exits `0`; unreachable endpoint exits `1` with host, port, connection class, and remediation, without exposing secrets.

- [ ] **Step 3: Audit the implementation against the design**

Confirm eight source entries, only GeekNews enabled, English sources marked `en`, Korean output enforced, score weights and gates exact, per-cycle/daily caps exact, atomic JSON writes, seen-state ordering, one LLM retry, Stage 3 absent, and original body absent from `curated.json`.

- [ ] **Step 4: Run final diff checks**

Run: `git diff --check`
Run: `git status --short`
Run: `git diff HEAD`
Expected: only intentional project files remain and the user-owned PRD stays unchanged.

- [ ] **Step 5: Commit audit fixes if any**

```powershell
git add README.md pyproject.toml sources.yaml src tests docs/superpowers
git commit -m "test: verify stage 1 and 2 pipeline"
```
