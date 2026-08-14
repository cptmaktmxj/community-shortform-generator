# Stage 2 Recalibration and Stage 3 Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrate Stage 2 for a non-technical software/AI/technology-market audience, atomically rebuild `curated.json`, and generate resumable Korean analyses, 30–60 second narration scripts, and titles with `gpt-5.4-mini`.

**Architecture:** Stage 2 keeps deterministic reaction scoring and K-EXAONE safety evaluation while replacing vague content scores with schema-constrained anchor bands and independent gates. Stage 3 is a separate `generate` service with focused analysis, script, revision, title, duration-estimation, state, and artifact boundaries; the existing `run` command remains Stage 1–2 to avoid an unexpected paid API call.

**Tech Stack:** Python 3.11+, Pydantic 2, OpenAI Python SDK Responses API, K-EXAONE through an OpenAI-compatible Chat Completions endpoint, SQLite, JSON atomic replacement, PyYAML, pytest, pytest-asyncio

## Global Constraints

- Keep Stage 1 behavior and all eight registered sources unchanged; GeekNews remains the only default-enabled source.
- Keep Stage 2 safety evaluation on K-EXAONE and allow only `safety_ok=true` records into Stage 3.
- Compute `curation_score = reaction * 0.20 + provocation * 0.40 + mass_appeal * 0.40`.
- Require `curation_score >= 0.62`, `provocation >= 0.35`, `mass_appeal >= 0.45`, `fidelity >= 0.75`, and `safety_ok=true`.
- Target non-technical viewers interested in common software, AI, workplace impact, and technology stocks.
- Use `gpt-5.4-mini` through the official OpenAI Responses API for all production Stage 3 calls.
- Generate in this exact order: analysis, script, duration validation/revision, then title.
- Estimate 45 seconds at 1.2x; accept 30–60 seconds, classify 40–50 seconds as ideal, and permit at most two duration revisions.
- Allow provocative wording such as “충격”, “무조건”, and “드디어 밝혀졌다”, but never invent events, figures, causality, outcomes, profit guarantees, or direct investment instructions.
- Never persist `.env`, API keys, original full bodies, or original full comments in Stage 2 or Stage 3 output.
- Keep `run` as Stage 1–2; expose paid Stage 3 only through explicit `generate` commands.
- Do not synthesize TTS audio, render video, or publish content in this delivery.
- Every implementation change follows red-green-refactor and every task ends with focused tests and an intentional commit.

## File Responsibility Map

- `src/community_shorts/models.py`: Stage 1–2 contracts, anchored score bands, and audited curated fields.
- `src/community_shorts/scoring.py`: deterministic score maps, 20/40/40 combination, and independent Stage 2 gates.
- `src/community_shorts/prompts.py`: K-EXAONE audience definition, anchored score rubric, Korean output, and safety rules.
- `src/community_shorts/curate.py`: incremental and atomic rebuild orchestration.
- `src/community_shorts/generation_models.py`: Stage 3 Pydantic input/output, job, and final artifact contracts.
- `src/community_shorts/duration.py`: replaceable character/punctuation duration estimator.
- `src/community_shorts/generation_prompts.py`: separate analysis, draft, revision, and title prompts.
- `src/community_shorts/generation_llm.py`: official OpenAI Responses API and deterministic fixture clients.
- `src/community_shorts/generate.py`: resumable Stage 3 orchestration and title validation.
- `src/community_shorts/storage.py`: atomic items, curated, and scripts artifacts plus replace-versus-merge operations.
- `src/community_shorts/state.py`: Stage 2 rebuild transactions and Stage 3 intermediate job persistence.
- `src/community_shorts/config.py`, `config.yaml`, `.env.example`: non-secret generation model/timing settings and environment lookup names.
- `src/community_shorts/cli.py`: `curate --rebuild`, `generate`, and `generate --rebuild` composition.
- `README.md`: new scoring, configuration, commands, artifacts, and acceptance workflow.

---

### Task 1: Anchor Stage 2 Content Scores and Apply New Gates

**Files:**
- Modify: `src/community_shorts/models.py`
- Modify: `src/community_shorts/scoring.py`
- Modify: `src/community_shorts/prompts.py`
- Modify: `src/community_shorts/llm.py`
- Modify: `src/community_shorts/curate.py`
- Modify: `tests/test_models.py`
- Modify: `tests/test_scoring.py`
- Modify: `tests/test_llm.py`
- Modify: `tests/test_curate.py`

**Interfaces:**
- Produces: `MassAppealBand`, `ProvocationBand`, `MASS_APPEAL_SCORES`, `PROVOCATION_SCORES`
- Produces: `passes_gates(*, score: float, provocation: float, mass_appeal: float, fidelity: float, safety_ok: bool) -> bool`
- Extends: `LlmAssessment` with `mass_appeal_band`, `mass_appeal_reason`, `provocation_band`, `provocation_reason`
- Extends: `CuratedItem` with the same four audit fields

- [ ] **Step 1: Write failing anchored-score model tests**

Add literal tests to `tests/test_models.py`:

```python
def valid_assessment_payload() -> dict[str, object]:
    return {
        "provocation_band": "challenges_expectation",
        "provocation_score": 0.5,
        "provocation_reason": "대중이 가진 일반적인 예상을 뒤집습니다.",
        "mass_appeal_band": "general_interest",
        "mass_appeal_score": 0.5,
        "mass_appeal_reason": "프로그램과 AI 관심층이 이해할 수 있습니다.",
        "fidelity_score": 0.9,
        "safety_ok": True,
        "safety_reason": "일반적인 기술 뉴스로 안전하게 생성할 수 있습니다.",
        "safety_categories": [],
        "reason": "타깃 대중에게 설명할 수 있는 변화입니다.",
        "summary": "핵심 내용을 한국어로 요약했습니다.",
        "key_claim": "중요한 변화가 발생했습니다.",
        "hook_points": ["예상 밖의 결과"],
        "tone": "정보형",
        "output_language": "ko",
    }


def test_assessment_requires_scores_to_match_anchor_bands() -> None:
    payload = valid_assessment_payload() | {
        "mass_appeal_band": "specialist_only",
        "mass_appeal_score": 0.7,
        "mass_appeal_reason": "특정 프레임워크 사용자만 이해하는 구현 세부사항입니다.",
        "provocation_band": "routine",
        "provocation_score": 0.5,
        "provocation_reason": "일반적인 유지보수 릴리스입니다.",
    }

    with pytest.raises(ValidationError, match="anchor score"):
        LlmAssessment.model_validate(payload)


def test_assessment_accepts_exact_anchor_scores() -> None:
    assessment = LlmAssessment.model_validate(
        valid_assessment_payload()
        | {
            "mass_appeal_band": "general_interest",
            "mass_appeal_score": 0.5,
            "mass_appeal_reason": "일반적인 프로그램과 AI 관심층이 이해할 수 있습니다.",
            "provocation_band": "challenges_expectation",
            "provocation_score": 0.5,
            "provocation_reason": "대중이 가진 일반적인 예상을 뒤집습니다.",
        }
    )

    assert assessment.mass_appeal_score == 0.5
    assert assessment.provocation_score == 0.5
```

- [ ] **Step 2: Write failing scoring and gate tests**

Replace old weight expectations and add every independent gate to `tests/test_scoring.py`:

```python
def test_curation_score_uses_20_40_40_weights() -> None:
    assert curation_score(1.0, 0.5, 0.7) == pytest.approx(0.68)


@pytest.mark.parametrize(
    ("score", "provocation", "mass_appeal", "fidelity", "safety_ok"),
    [
        (0.619, 0.5, 0.5, 0.9, True),
        (0.9, 0.34, 0.7, 0.9, True),
        (0.9, 0.7, 0.44, 0.9, True),
        (0.9, 0.7, 0.7, 0.74, True),
        (0.9, 0.7, 0.7, 0.9, False),
    ],
)
def test_each_independent_gate_can_reject(
    score: float,
    provocation: float,
    mass_appeal: float,
    fidelity: float,
    safety_ok: bool,
) -> None:
    assert passes_gates(
        score=score,
        provocation=provocation,
        mass_appeal=mass_appeal,
        fidelity=fidelity,
        safety_ok=safety_ok,
    ) is False
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_models.py tests/test_scoring.py -q
```

Expected: failures for missing band fields, missing anchor validation, old weights, and the old `passes_gates` signature.

- [ ] **Step 4: Add exact anchor enums and validators**

Add to `models.py`:

```python
MassAppealBand = Literal[
    "specialist_only",
    "tech_enthusiast",
    "general_interest",
    "direct_impact",
    "broad_impact",
]
ProvocationBand = Literal[
    "routine",
    "specialist_novelty",
    "challenges_expectation",
    "clear_disruption",
    "broad_shock",
]

MASS_APPEAL_SCORES = {
    "specialist_only": 0.1,
    "tech_enthusiast": 0.3,
    "general_interest": 0.5,
    "direct_impact": 0.7,
    "broad_impact": 0.9,
}
PROVOCATION_SCORES = {
    "routine": 0.1,
    "specialist_novelty": 0.3,
    "challenges_expectation": 0.5,
    "clear_disruption": 0.7,
    "broad_shock": 0.9,
}
```

Require both reasons to contain Korean, require exact mapped scores in the existing `LlmAssessment` model validator, and copy all four audit fields into `CuratedItem`.

- [ ] **Step 5: Implement the new score weights and gates**

Change `scoring.py` to:

```python
def curation_score(reaction: float, provocation: float, mass_appeal: float) -> float:
    """Weight audience-fit content signals above source-local reactions."""

    return reaction * 0.20 + provocation * 0.40 + mass_appeal * 0.40


def passes_gates(
    *,
    score: float,
    provocation: float,
    mass_appeal: float,
    fidelity: float,
    safety_ok: bool,
) -> bool:
    """Require combined quality and every independent Stage 2 threshold."""

    return (
        safety_ok
        and score >= 0.62
        and provocation >= 0.35
        and mass_appeal >= 0.45
        and fidelity >= 0.75
    )
```

Pass the new arguments from `CurateService` and persist the audit fields in `_to_curated`.

- [ ] **Step 6: Replace the vague prompt with target-audience anchors**

Update `SYSTEM_PROMPT` to name the target audience and require the exact five anchor values. Include these Korean rules verbatim:

```text
대상은 기술 종사자가 아니지만 일상·업무용 프로그램, AI, 테크주에 관심이 많은 일반 대중입니다.
개발 프레임워크·라이브러리·언어·인프라의 구현 세부사항은 원문에 일반 사용자의 시간·돈·업무·고용·투자 판단에 미치는 구체적 영향이 없으면 mass_appeal_band를 specialist_only 또는 tech_enthusiast로 제한하세요.
단순 출시·업데이트는 provocation_band=routine, 업계 내부의 특이한 선택은 specialist_novelty를 넘지 않습니다.
각 band에 대응하는 점수는 각각 0.1, 0.3, 0.5, 0.7, 0.9 중 정확한 값만 사용하세요.
```

Update `FixtureLlmClient`, shared test payloads, and curation fixtures with exact band values and Korean reasons.

- [ ] **Step 7: Run Stage 2 focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_models.py tests/test_scoring.py tests/test_llm.py tests/test_curate.py -q
```

Expected: all focused tests pass with no legacy 40/25/35 assertion.

- [ ] **Step 8: Commit the calibrated scoring contract**

```powershell
git add src/community_shorts/models.py src/community_shorts/scoring.py src/community_shorts/prompts.py src/community_shorts/llm.py src/community_shorts/curate.py tests/test_models.py tests/test_scoring.py tests/test_llm.py tests/test_curate.py
git commit -m "Recalibrate Stage 2 audience scoring"
```

### Task 2: Add Atomic Stage 2 Rebuild

**Files:**
- Modify: `src/community_shorts/storage.py`
- Modify: `src/community_shorts/state.py`
- Modify: `src/community_shorts/curate.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_state.py`
- Modify: `tests/test_curate.py`

**Interfaces:**
- Produces: `ArtifactStore.replace_curated(items: Sequence[CuratedItem]) -> None`
- Produces: `StateStore.replace_stage2_results(*, curated_ids: Sequence[str], safety_rejections: Mapping[str, str], at: datetime) -> None`
- Changes: `CurateService.run(now: datetime, *, rebuild: bool = False) -> CurateReport`
- Produces: `RebuildIncompleteError(RuntimeError)`

- [ ] **Step 1: Write failing replacement-storage tests**

Add to `tests/test_storage.py`:

```python
def test_replace_curated_does_not_merge_old_results(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.write_curated([make_curated("geeknews:old")])

    store.replace_curated([make_curated("geeknews:new")])

    assert [item.item_id for item in store.read_curated()] == ["geeknews:new"]
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Write failing Stage 2 state replacement tests**

Add to `tests/test_state.py`:

```python
def test_replace_stage2_results_resets_old_terminal_states(tmp_path: Path) -> None:
    state = populated_state(tmp_path, ["geeknews:old", "geeknews:new", "geeknews:unsafe"])
    state.mark_curated(["geeknews:old"], at=NOW)
    state.mark_safety_rejected("geeknews:unsafe", at=NOW, reason="old")

    state.replace_stage2_results(
        curated_ids=["geeknews:new"],
        safety_rejections={"geeknews:unsafe": "actionable_cyber_abuse: 새 판정"},
        at=NOW,
    )

    assert state.curated_ids() == {"geeknews:new"}
    assert state.stage2_terminal_ids() == {"geeknews:new", "geeknews:unsafe"}
```

- [ ] **Step 3: Write failing curation rebuild tests**

Add two tests to `tests/test_curate.py`:

```python
@pytest.mark.asyncio
async def test_rebuild_reassesses_terminal_items_and_replaces_artifact(tmp_path: Path) -> None:
    store, state = prepare_existing_curated_state(tmp_path, item_id="geeknews:old")
    service = CurateService(store, state, PassingAnchoredLlm(), cycle_limit=2)

    report = await service.run(NOW, rebuild=True)

    assert report.rebuilt is True
    assert service.llm.seen_ids == {"geeknews:old"}
    assert all(item.mass_appeal_score >= 0.45 for item in store.read_curated())


@pytest.mark.asyncio
async def test_failed_rebuild_preserves_old_artifact_and_states(tmp_path: Path) -> None:
    store, state = prepare_existing_curated_state(tmp_path, item_id="geeknews:old")
    before = (tmp_path / "curated.json").read_bytes()

    with pytest.raises(RebuildIncompleteError):
        await CurateService(store, state, AlwaysFailLlm()).run(NOW, rebuild=True)

    assert (tmp_path / "curated.json").read_bytes() == before
    assert state.curated_ids() == {"geeknews:old"}
```

- [ ] **Step 4: Run rebuild tests and verify RED**

Run:

```powershell
python -m pytest tests/test_storage.py tests/test_state.py tests/test_curate.py -q
```

Expected: failures for missing replacement methods, `rebuild`, and `CurateReport.rebuilt`.

- [ ] **Step 5: Implement artifact and state replacement**

Implement `replace_curated` as a direct call to the existing atomic `_write_models` without reading or merging the prior file.

Implement `replace_stage2_results` in one SQLite transaction:

```python
def replace_stage2_results(
    self,
    *,
    curated_ids: Sequence[str],
    safety_rejections: Mapping[str, str],
    at: datetime,
) -> None:
    """Replace terminal Stage 2 states after a successful rebuild artifact write."""

    with self._connect() as connection:
        connection.execute(
            """
            UPDATE items
            SET curated_at = NULL, status = 'ingested', error = NULL
            WHERE curated_at IS NOT NULL OR status = 'safety_rejected'
            """
        )
        connection.executemany(
            "UPDATE items SET status = 'safety_rejected', error = ? WHERE item_id = ?",
            [(reason, item_id) for item_id, reason in safety_rejections.items()],
        )
        connection.executemany(
            "UPDATE items SET curated_at = ?, status = 'curated', error = NULL WHERE item_id = ?",
            [(at.isoformat(), item_id) for item_id in curated_ids],
        )
```

- [ ] **Step 6: Implement all-or-nothing rebuild orchestration**

When `rebuild=True`, include all raw items regardless of terminal state, defer safety state writes until evaluation completes, and raise `RebuildIncompleteError` if any candidate assessment fails. Bypass the old curated daily count because the artifact is being replaced, but preserve the configured cycle limit. Write the replacement artifact first, then replace Stage 2 states. Add `rebuilt: bool` to `CurateReport`.

Normal `rebuild=False` behavior stays incremental and keeps per-item failure isolation.

- [ ] **Step 7: Run focused rebuild tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_storage.py tests/test_state.py tests/test_curate.py -q
```

Expected: all tests pass, including preservation of the old artifact on failed evaluation.

- [ ] **Step 8: Commit rebuild support**

```powershell
git add src/community_shorts/storage.py src/community_shorts/state.py src/community_shorts/curate.py tests/test_storage.py tests/test_state.py tests/test_curate.py
git commit -m "Add atomic Stage 2 rebuild"
```

### Task 3: Add Stage 3 Configuration and Domain Contracts

**Files:**
- Create: `src/community_shorts/generation_models.py`
- Create: `tests/test_generation_models.py`
- Modify: `src/community_shorts/config.py`
- Modify: `config.yaml`
- Modify: `.env.example`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `GenerationLlmConfig`, `ScriptTimingConfig`
- Extends: `ResolvedAppConfig` with `generation_llm_*` and `script_timing`
- Produces: `ContentAnalysis`, `ScriptDraft`, `TitleCandidate`, `TitlePackage`, `GeneratedScript`, `GenerationJob`, `GenerationStatus`, `DurationClass`

- [ ] **Step 1: Write failing configuration tests**

Add to `tests/test_config.py`:

```python
def test_generation_config_resolves_openai_key_without_hardcoding(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(APP_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("TEST_GENERATION_KEY", "test-secret")

    config = load_app_config(path)
    resolved = config.resolve()

    assert resolved.generation_llm_model == "gpt-5.4-mini"
    assert resolved.generation_llm_api_key == "test-secret"
    assert resolved.script_timing.playback_speed == 1.2
    assert "test-secret" not in config.model_dump_json()


def test_generation_config_rejects_invalid_duration_bounds(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        APP_CONFIG_YAML.replace("min_seconds: 30", "min_seconds: 61"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duration bounds"):
        load_app_config(path)
```

Extend the existing `APP_CONFIG_YAML` fixture with the approved `generation_llm` and `script` sections, using `api_key_env: TEST_GENERATION_KEY` so these tests never depend on the developer's real environment.

- [ ] **Step 2: Write failing Stage 3 model tests**

Create `tests/test_generation_models.py` with:

```python
def test_analysis_requires_korean_and_core_facts() -> None:
    with pytest.raises(ValidationError):
        ContentAnalysis(
            topic_category="ai_impact",
            audience_relevance="work impact",
            core_facts=[],
            angle="angle",
            hook_strategy="hook",
            claims_to_avoid=[],
            recommended_tone="fast",
        )


def test_title_package_requires_three_distinct_styles() -> None:
    with pytest.raises(ValidationError, match="three title styles"):
        TitlePackage(
            candidates=[title_candidate("direct_impact")] * 3,
            selected_title="AI가 업무를 바꾸는 결정적인 이유",
        )


def test_generated_script_never_accepts_original_body() -> None:
    payload = generated_script_payload() | {"body": "원문 전문"}

    with pytest.raises(ValidationError):
        GeneratedScript.model_validate(payload)
```

- [ ] **Step 3: Run configuration and model tests and verify RED**

Run:

```powershell
python -m pytest tests/test_config.py tests/test_generation_models.py -q
```

Expected: import failures for `generation_models` and missing configuration sections.

- [ ] **Step 4: Implement generation and timing configuration**

Add strict Pydantic models:

```python
class GenerationLlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["openai", "fixture"]
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    timeout_seconds: float = Field(gt=0)


class ScriptTimingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_seconds: float = Field(gt=0)
    min_seconds: float = Field(gt=0)
    max_seconds: float = Field(gt=0)
    playback_speed: float = Field(gt=0)
    base_spoken_units_per_second: float = Field(gt=0)
    max_revisions: int = Field(ge=0, le=5)
```

Use a model validator to require `min_seconds < target_seconds < max_seconds`. Resolve `OPENAI_API_KEY` to `str | None`; do not fail config loading for `ingest` or `curate` when the key is absent.

Add the exact approved values to `config.yaml` and a commented `OPENAI_API_KEY=replace-with-openai-key` example to `.env.example`.

- [ ] **Step 5: Implement strict Stage 3 models**

Use `StrictModel` or the same `ConfigDict(extra="forbid")` behavior. Define:

```python
TopicCategory = Literal[
    "ai_impact",
    "consumer_software",
    "work_productivity",
    "tech_market",
    "technology_trend",
]
TitleStyle = Literal["direct_impact", "question", "conventional_wisdom_reversal"]
DurationClass = Literal["ideal", "acceptable"]
GenerationStatus = Literal[
    "pending",
    "analyzed",
    "scripted",
    "duration_failed",
    "title_failed",
    "completed",
    "failed",
]
```

Require Korean in every non-empty narrative field, at least one `core_fact`, a non-empty script with no URL, exactly three distinct title styles, and only `completed` records in `GeneratedScript`.

`TitleCandidate` contains `style`, `title`, and `supporting_script_excerpt`. `GeneratedScript` serializes candidate title strings only, while the richer package remains in SQLite for validation and audit.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_config.py tests/test_generation_models.py -q
```

Expected: all configuration and model tests pass.

- [ ] **Step 7: Commit Stage 3 contracts**

```powershell
git add src/community_shorts/config.py src/community_shorts/generation_models.py config.yaml .env.example tests/test_config.py tests/test_generation_models.py
git commit -m "Add Stage 3 configuration and contracts"
```

### Task 4: Implement the Replaceable Duration Estimator

**Files:**
- Create: `src/community_shorts/duration.py`
- Create: `tests/test_duration.py`

**Interfaces:**
- Produces: `DurationEstimate(seconds: float, classification: Literal["ideal", "acceptable", "short", "long"])`
- Produces: `estimate_duration(text: str, config: ScriptTimingConfig) -> DurationEstimate`
- Produces: `spoken_units(text: str) -> float`
- Produces: `pause_seconds(text: str) -> float`

- [ ] **Step 1: Write failing estimator tests with literal arithmetic**

Create `tests/test_duration.py`:

```python
def timing(**overrides) -> ScriptTimingConfig:
    values = {
        "target_seconds": 45,
        "min_seconds": 30,
        "max_seconds": 60,
        "playback_speed": 1.2,
        "base_spoken_units_per_second": 4.0,
        "max_revisions": 2,
    }
    return ScriptTimingConfig(**(values | overrides))


def test_hangul_and_sentence_pause_are_scaled_by_playback_speed() -> None:
    estimate = estimate_duration("가나다라.", timing(min_seconds=0.5, target_seconds=1, max_seconds=2))

    assert estimate.seconds == pytest.approx((4 / 4.0 + 0.35) / 1.2)


def test_paragraph_pause_replaces_adjacent_sentence_pause() -> None:
    assert pause_seconds("첫 문장.\n\n둘째 문장") == pytest.approx(0.45)


def test_english_abbreviation_and_number_cost_more_than_raw_character_count() -> None:
    assert spoken_units("GPT-5") > spoken_units("가나다")


@pytest.mark.parametrize(
    ("seconds", "classification"),
    [(29.9, "short"), (30.0, "acceptable"), (40.0, "ideal"), (50.0, "ideal"), (60.0, "acceptable"), (60.1, "long")],
)
def test_duration_boundaries(seconds: float, classification: str) -> None:
    assert classify_duration(seconds, timing()) == classification
```

- [ ] **Step 2: Run estimator tests and verify RED**

Run:

```powershell
python -m pytest tests/test_duration.py -q
```

Expected: import failure because `duration.py` does not exist.

- [ ] **Step 3: Implement token and pause accounting**

Implement original regex-based code without copying external source:

- Hangul syllables and CJK characters: one unit each.
- Lowercase/mixed-case English words: `max(2.0, len(token) * 0.65)` units.
- All-capital abbreviations: `max(2.0, len(token) * 1.25)` units.
- Number runs: `max(1.5, digit_count * 1.5)` units; an immediately attached Korean unit remains counted normally.
- Comma, colon, semicolon: 0.15 seconds.
- Period, question mark, exclamation mark: 0.35 seconds.
- Paragraph boundary: 0.45 seconds, replacing an adjacent sentence pause.
- Markdown punctuation and whitespace: zero units.

Round the public estimate to three decimal places only after classification comparisons use the unrounded value.

- [ ] **Step 4: Run estimator tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_duration.py -q
```

Expected: all duration arithmetic and boundary tests pass.

- [ ] **Step 5: Commit the estimator**

```powershell
git add src/community_shorts/duration.py tests/test_duration.py
git commit -m "Add narration duration estimator"
```

### Task 5: Add Structured Stage 3 Prompts and GPT Client

**Files:**
- Create: `src/community_shorts/generation_prompts.py`
- Create: `src/community_shorts/generation_llm.py`
- Create: `tests/test_generation_llm.py`

**Interfaces:**
- Produces: `GenerationLlmClient` protocol with `analyze`, `draft`, `revise`, and `title`
- Produces: `OpenAiGenerationLlmClient`, `FixtureGenerationLlmClient`
- Produces: `ResponsesTransport.parse(*, model: str, input: Sequence[dict[str, str]], output_type: type[ModelT], max_output_tokens: int) -> ModelT`
- Produces: `build_analysis_input`, `build_draft_input`, `build_revision_input`, `build_title_input`

- [ ] **Step 1: Write failing payload-boundary tests**

Create `tests/test_generation_llm.py` with a representative `CuratedItem` and capturing transport:

```python
@pytest.mark.asyncio
async def test_analysis_payload_uses_only_transformed_stage2_fields() -> None:
    transport = CapturingResponsesTransport(ContentAnalysis, valid_analysis())
    client = OpenAiGenerationLlmClient(transport=transport, model="gpt-5.4-mini")

    await client.analyze(curated_item())

    payload = json.dumps(transport.calls[0].input, ensure_ascii=False)
    assert curated_item().summary in payload
    assert "body" not in payload
    assert "top_comments" not in payload


@pytest.mark.asyncio
async def test_title_payload_contains_final_script_and_not_original_summary() -> None:
    transport = CapturingResponsesTransport(TitlePackage, valid_titles())
    client = OpenAiGenerationLlmClient(transport=transport, model="gpt-5.4-mini")

    await client.title(curated_item(), valid_analysis(), "최종 검증된 한국어 대본입니다.")

    payload = json.dumps(transport.calls[0].input, ensure_ascii=False)
    assert "최종 검증된 한국어 대본입니다." in payload
    assert curated_item().summary not in payload
```

- [ ] **Step 2: Write failing Responses API and retry tests**

```python
@pytest.mark.asyncio
async def test_openai_transport_uses_responses_parse_and_pydantic_schema(monkeypatch) -> None:
    captured = {}

    class FakeResponses:
        async def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=valid_analysis())

    monkeypatch.setattr(generation_llm_module, "AsyncOpenAI", fake_client(FakeResponses()))
    transport = OpenAiResponsesTransport(
        base_url="https://api.openai.com/v1",
        api_key="test",
        timeout_seconds=120,
    )

    await transport.parse(
        model="gpt-5.4-mini",
        input=[],
        output_type=ContentAnalysis,
        max_output_tokens=800,
    )

    assert captured["model"] == "gpt-5.4-mini"
    assert captured["text_format"] is ContentAnalysis
    assert captured["max_output_tokens"] == 800


@pytest.mark.asyncio
async def test_schema_failure_retries_once_with_korean_correction() -> None:
    transport = SequencedResponsesTransport([ValidationErrorResponse(), valid_analysis()])

    result = await OpenAiGenerationLlmClient(transport, "gpt-5.4-mini").analyze(curated_item())

    assert result.core_facts
    assert len(transport.calls) == 2
    assert "한국어" in transport.calls[1].input[-1]["content"]
```

- [ ] **Step 3: Run Stage 3 LLM tests and verify RED**

Run:

```powershell
python -m pytest tests/test_generation_llm.py -q
```

Expected: import failures for the prompt and client modules.

- [ ] **Step 4: Implement four isolated prompt builders**

Each builder returns a fresh list of Responses API input messages. Keep shared factual and style policies in one constant. The analysis prompt requires audience relevance and source-bounded facts. The draft prompt requires the five-part narration structure and 45-second target. The revision prompt includes `direction: Literal["expand", "shorten"]`, current duration, target duration, validated analysis, and current script. The title prompt includes only analysis and the final validated script.

Include this fact-versus-style rule exactly once in the shared policy:

```text
“충격”, “무조건”, “드디어 밝혀졌다” 같은 강한 표현은 허용하지만 입력에 없는 사건·수치·인과관계·성과·확실성을 만들어내면 안 됩니다.
```

- [ ] **Step 5: Implement the official Responses API transport**

Use the documented Pydantic parse helper:

```python
response = await self._client.responses.parse(
    model=model,
    input=list(input),
    text_format=output_type,
    max_output_tokens=max_output_tokens,
)
parsed = response.output_parsed
if parsed is None:
    raise GenerationResponseError("OpenAI response did not contain parsed output")
return output_type.model_validate(parsed)
```

Configure `AsyncOpenAI` with the resolved base URL, key, bounded connect timeout, and `max_retries=0`. The higher client retries a schema/refusal/empty-output failure once with a Korean correction message; a second failure raises `GenerationResponseError`.

Use explicit output caps of 800 tokens for analysis, 1,200 for draft and revision, and 600 for titles. Do not set a nonzero temperature; preserve the model's deterministic default behavior for schema-bound production calls.

- [ ] **Step 6: Implement production and fixture generation clients**

The production client delegates each method to the matching prompt and output model. The fixture client returns deterministic Korean analysis, an approximately 45-second configurable script, deterministic expand/shorten revisions, and three styled title candidates whose evidence excerpts are exact script substrings.

- [ ] **Step 7: Run Stage 3 client tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_generation_llm.py -q
```

Expected: all payload, Responses API, retry, Korean, and fixture tests pass.

- [ ] **Step 8: Commit the Stage 3 model boundary**

```powershell
git add src/community_shorts/generation_prompts.py src/community_shorts/generation_llm.py tests/test_generation_llm.py
git commit -m "Add GPT Stage 3 generation client"
```

### Task 6: Persist Stage 3 Jobs and Final Scripts

**Files:**
- Modify: `src/community_shorts/storage.py`
- Modify: `src/community_shorts/state.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_state.py`

**Interfaces:**
- Produces: `ArtifactStore.read_scripts() -> list[GeneratedScript]`
- Produces: `ArtifactStore.write_scripts(items: Sequence[GeneratedScript]) -> None`
- Produces: `ArtifactStore.replace_scripts(items: Sequence[GeneratedScript]) -> None`
- Produces: `StateStore.load_generation_job(item_id: str) -> GenerationJob | None`
- Produces: `save_generation_analysis`, `save_generation_script`, `save_generation_failure`, `mark_generation_completed`, `reset_generation_jobs`

- [ ] **Step 1: Write failing scripts-artifact tests**

Add to `tests/test_storage.py`:

```python
def test_scripts_artifact_merges_normally_and_replaces_explicitly(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.write_scripts([make_generated("geeknews:1")])
    store.write_scripts([make_generated("geeknews:2")])
    assert {item.item_id for item in store.read_scripts()} == {"geeknews:1", "geeknews:2"}

    store.replace_scripts([make_generated("geeknews:3")])
    assert [item.item_id for item in store.read_scripts()] == ["geeknews:3"]
    assert "body" not in (tmp_path / "scripts.json").read_text(encoding="utf-8")
```

- [ ] **Step 2: Write failing job-resumption state tests**

Add to `tests/test_state.py`:

```python
def test_generation_job_round_trips_analysis_and_script(tmp_path: Path) -> None:
    state = StateStore(tmp_path / "state.sqlite")
    state.save_generation_analysis("geeknews:1", valid_analysis(), model="gpt-5.4-mini", at=NOW)
    state.save_generation_script(
        "geeknews:1",
        script="검증할 한국어 대본입니다.",
        estimated_duration=44.5,
        revision_count=1,
        at=NOW,
    )

    job = state.load_generation_job("geeknews:1")

    assert job.status == "scripted"
    assert job.analysis == valid_analysis()
    assert job.revision_count == 1


def test_reset_generation_jobs_only_targets_current_curated_ids(tmp_path: Path) -> None:
    state = state_with_completed_jobs(tmp_path, ["geeknews:1", "geeknews:2"])

    state.reset_generation_jobs(["geeknews:2"])

    assert state.load_generation_job("geeknews:1").status == "completed"
    assert state.load_generation_job("geeknews:2") is None
```

- [ ] **Step 3: Run persistence tests and verify RED**

Run:

```powershell
python -m pytest tests/test_storage.py tests/test_state.py -q
```

Expected: missing scripts paths, storage methods, `generation_jobs` table, and state methods.

- [ ] **Step 4: Extend atomic storage for completed scripts**

Add `scripts_path = root / "scripts.json"`. Reuse `_read_models` and `_write_models`; normal writes merge by item ID and explicit replacements never read the old artifact.

- [ ] **Step 5: Add and migrate the generation jobs table**

Create idempotently in `_initialize`:

```sql
CREATE TABLE IF NOT EXISTS generation_jobs (
    item_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    analysis_json TEXT,
    script_text TEXT,
    estimated_duration REAL,
    revision_count INTEGER NOT NULL DEFAULT 0,
    title_json TEXT,
    model TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT
);
```

Serialize Pydantic objects with `model_dump_json()` and validate them when reading. Every save method uses a transaction and updates only fields owned by that substage. `mark_generation_completed` stores the validated title package and clears errors. `save_generation_failure` accepts only `duration_failed`, `title_failed`, or `failed`.

- [ ] **Step 6: Run persistence tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_storage.py tests/test_state.py -q
```

Expected: all legacy and new storage/state tests pass against both fresh and existing SQLite schemas.

- [ ] **Step 7: Commit Stage 3 persistence**

```powershell
git add src/community_shorts/storage.py src/community_shorts/state.py tests/test_storage.py tests/test_state.py
git commit -m "Persist resumable Stage 3 jobs"
```

### Task 7: Orchestrate Analysis, Script Revision, and Titles

**Files:**
- Create: `src/community_shorts/generate.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Produces: `GenerateService(storage, state, llm, timing)`
- Produces: `GenerateService.run(now: datetime, *, rebuild: bool = False) -> GenerateReport`
- Produces: `GenerateReport(attempted, completed, duration_failed, title_failed, failed_item_ids, rebuilt)`
- Produces: `validate_title_package(package: TitlePackage, script: str) -> TitlePackage`

- [ ] **Step 1: Write failing happy-path call-order test**

Create `tests/test_generate.py`:

```python
@pytest.mark.asyncio
async def test_generation_orders_analysis_script_duration_then_title(tmp_path: Path) -> None:
    store = store_with_curated(tmp_path, [curated_item()])
    llm = RecordingGenerationLlm(script=ideal_length_script())
    service = GenerateService(store, StateStore(tmp_path / "state.sqlite"), llm, timing())

    report = await service.run(NOW)

    assert llm.calls == ["analyze", "draft", "title"]
    assert report.completed == 1
    assert store.read_scripts()[0].estimated_duration_seconds >= 40
```

- [ ] **Step 2: Write failing duration-revision tests**

```python
@pytest.mark.asyncio
async def test_short_script_expands_at_most_twice_before_title(tmp_path: Path) -> None:
    llm = RecordingGenerationLlm(
        script="짧은 대본.",
        revisions=["여전히 짧은 대본.", ideal_length_script()],
    )
    service = prepared_service(tmp_path, llm)

    report = await service.run(NOW)

    assert llm.calls == ["analyze", "draft", "revise:expand", "revise:expand", "title"]
    assert report.completed == 1


@pytest.mark.asyncio
async def test_duration_failure_never_calls_title(tmp_path: Path) -> None:
    llm = RecordingGenerationLlm(script="짧음.", revisions=["짧음.", "짧음."])
    service = prepared_service(tmp_path, llm)

    report = await service.run(NOW)

    assert "title" not in llm.calls
    assert report.duration_failed == 1
    assert service.state.load_generation_job(curated_item().item_id).status == "duration_failed"
```

- [ ] **Step 3: Write failing resumption and title validation tests**

```python
@pytest.mark.asyncio
async def test_title_retry_reuses_saved_analysis_and_script(tmp_path: Path) -> None:
    service, llm = service_with_scripted_job(tmp_path)

    await service.run(NOW)

    assert llm.calls == ["title"]


def test_title_validation_allows_strong_words_with_exact_script_evidence() -> None:
    script = "이 변화는 직장인에게 충격적인 업무 변화를 만들 수 있습니다."
    package = title_package(
        title="직장인에게 충격, AI 업무 변화",
        excerpt="충격적인 업무 변화를 만들 수 있습니다",
    )

    assert validate_title_package(package, script).candidates


def test_title_validation_rejects_missing_evidence_and_investment_instruction() -> None:
    package = title_package(title="무조건 지금 이 주식을 매수하세요", excerpt="없는 근거")

    with pytest.raises(TitleValidationError):
        validate_title_package(package, "AI 시장 변화에 관한 설명입니다.")
```

- [ ] **Step 4: Run orchestration tests and verify RED**

Run:

```powershell
python -m pytest tests/test_generate.py -q
```

Expected: import failure because `generate.py` does not exist.

- [ ] **Step 5: Implement local title validation**

For each candidate, require:

- `18 <= len(title.strip()) <= 34`;
- the exact stripped `supporting_script_excerpt` occurs in the final script;
- no URL;
- no direct investment instruction markers such as `매수하세요`, `매도하세요`, `전액 투자`, or `수익 보장`;
- three distinct styles and titles.

Do not reject “충격”, “무조건”, or “드디어 밝혀졌다” by keyword alone.

- [ ] **Step 6: Implement resumable per-item generation**

For each curated safe item:

1. Skip a completed job during normal execution.
2. Reuse saved analysis when status is `analyzed` or later.
3. Reuse a saved in-range script when status is `scripted` or `title_failed`.
4. Draft a new script otherwise and estimate it.
5. Revise with `expand` or `shorten` until accepted or the configured maximum is reached.
6. Save `duration_failed` and continue without a title call when still outside 30–60 seconds.
7. Generate and locally validate titles.
8. Save `title_failed` and continue when no package passes.
9. Build `GeneratedScript`, atomically merge it into `scripts.json`, then mark the job completed.

Catch failures per item, log sanitized item/substage/exception details, persist `failed`, and continue. If every attempted item fails, raise `RuntimeError("All Stage 3 generations failed")` after states are recorded.

When `rebuild=True`, reset jobs only for current curated IDs before processing and replace `scripts.json` with only newly completed current results after the whole run. If every item fails, preserve the prior scripts artifact.

- [ ] **Step 7: Run orchestration tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_generate.py tests/test_duration.py tests/test_state.py tests/test_storage.py -q
```

Expected: all happy path, revision, no-title, resumption, rebuild, and failure-isolation tests pass.

- [ ] **Step 8: Commit Stage 3 orchestration**

```powershell
git add src/community_shorts/generate.py tests/test_generate.py
git commit -m "Generate resumable shortform scripts"
```

### Task 8: Expose CLI Commands and Document Operations

**Files:**
- Modify: `src/community_shorts/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_e2e.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `community-shorts curate --rebuild`
- Produces: `community-shorts generate [--rebuild] [--generation-base-url URL] [--generation-model MODEL] [--generation-llm-mode openai|fixture]`
- Preserves: `community-shorts run` as Stage 1–2

- [ ] **Step 1: Write failing CLI parser tests**

Add to `tests/test_cli.py`:

```python
def test_curate_accepts_rebuild_flag(monkeypatch) -> None:
    captured = capture_execute_args(monkeypatch)

    assert main(["curate", "--rebuild", "--llm-mode", "fixture"]) == 0
    assert captured.rebuild is True


def test_generate_uses_separate_model_overrides(monkeypatch) -> None:
    captured = capture_execute_args(monkeypatch)

    assert main([
        "generate",
        "--generation-llm-mode", "fixture",
        "--generation-model", "gpt-5.4-mini",
    ]) == 0
    assert captured.generation_model == "gpt-5.4-mini"


def test_run_does_not_construct_generation_client(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "OpenAiGenerationLlmClient", forbidden_constructor)

    assert main(["run", "--llm-mode", "fixture"]) == 0
```

- [ ] **Step 2: Write failing fixture end-to-end test**

Add to `tests/test_e2e.py`:

```python
@pytest.mark.asyncio
async def test_fixture_stage3_writes_completed_script_without_original_text(tmp_path: Path) -> None:
    store, state = await fixture_stage1_and_stage2(tmp_path)
    service = GenerateService(store, state, FixtureGenerationLlmClient(), timing())

    report = await service.run(NOW)

    assert report.completed >= 1
    payload = json.loads((tmp_path / "scripts.json").read_text(encoding="utf-8"))
    assert 30 <= payload[0]["estimated_duration_seconds"] <= 60
    assert len(payload[0]["title_candidates"]) == 3
    assert "body" not in payload[0]
    assert "top_comments" not in payload[0]
```

- [ ] **Step 3: Run CLI and E2E tests and verify RED**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_e2e.py -q
```

Expected: missing `generate` parser, flags, and service composition.

- [ ] **Step 4: Split parser arguments by command and compose Stage 3**

Add `--rebuild` only to `curate` and `generate`. Add generation-specific override flags only to `generate`. Resolve the generation key but raise `ConfigError("OPENAI_API_KEY is required for generation_llm.mode=openai")` only when executing paid generation.

Construct `OpenAiResponsesTransport` and `OpenAiGenerationLlmClient` for production, or `FixtureGenerationLlmClient` only after explicit fixture selection. Pass the resolved `ScriptTimingConfig` to `GenerateService`.

Add generation diagnostics using the configured generation endpoint host, port, and scheme without credentials. Keep `run` limited to ingestion and curation.

- [ ] **Step 5: Update operating documentation**

Document:

- the target audience and 20/40/40 Stage 2 formula;
- all independent gates and exact anchor values;
- `curate --rebuild` replacement semantics;
- `OPENAI_API_KEY` without a real key value;
- `gpt-5.4-mini` analysis/script/title use;
- the 1.2x, 45-second target, 30–60 acceptance, and estimator caveat;
- `generate` and `generate --rebuild`;
- `scripts.json` and SQLite intermediate state;
- actual TTS, video, and publishing exclusions.

- [ ] **Step 6: Run CLI, E2E, and complete offline tests**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_e2e.py -q
python -m pytest -m "not live" -q
python -m pip check
```

Expected: all offline tests pass and dependencies are consistent.

- [ ] **Step 7: Commit CLI and documentation**

```powershell
git add src/community_shorts/cli.py tests/test_cli.py tests/test_e2e.py README.md
git commit -m "Expose Stage 3 generation commands"
```

### Task 9: Live Rebuild, GPT Generation, and Final Audit

**Files:**
- Modify only files implicated by a failed acceptance check.
- Generate ignored local artifacts: `data/curated.json`, `data/scripts.json`, `data/state.sqlite`

**Interfaces:**
- Consumes: Furiosa K-EXAONE at `http://127.0.0.1:8000/v1`
- Consumes: official OpenAI endpoint and `OPENAI_API_KEY`
- Produces: schema-valid replacement `curated.json` and completed `scripts.json`

- [ ] **Step 1: Verify services and secret presence without exposing values**

Run:

```powershell
curl.exe -sS -o NUL -w "furiosa_version_http=%{http_code}`n" --max-time 5 http://127.0.0.1:8000/version
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENAI_API_KEY_present=' + str(bool(os.getenv('OPENAI_API_KEY'))))"
```

Expected: Furiosa returns 200 and key presence prints `True`; no key value appears.

- [ ] **Step 2: Run the complete offline regression before paid calls**

Run:

```powershell
python -m pytest -m "not live" -q
python -m pip check
```

Expected: all tests pass and no broken requirements are reported.

- [ ] **Step 3: Rebuild Stage 2 with K-EXAONE**

Run:

```powershell
python -m community_shorts curate --rebuild --llm-mode openai --base-url http://127.0.0.1:8000/v1 --model furiosa-ai/K-EXAONE-236B-A23B-NVFP4A16 --log-level INFO
```

Expected: up to ten GeekNews candidates are reassessed, no failed item remains, and the old `curated.json` is atomically replaced.

- [ ] **Step 4: Validate the replacement selection contract**

Run a read-only validator using project models:

```powershell
python -c "from pathlib import Path; from community_shorts.storage import ArtifactStore; xs=ArtifactStore(Path('data')).read_curated(); assert all(x.safety_ok and x.provocation_score >= .35 and x.mass_appeal_score >= .45 and x.curation_score >= .62 for x in xs); print(f'curated={len(xs)} STAGE2_REBUILD=PASS')"
```

Expected: `STAGE2_REBUILD=PASS`; print each selected title and component score separately without source bodies.

- [ ] **Step 5: Generate real Stage 3 scripts with GPT-5.4 mini**

Run:

```powershell
python -m community_shorts generate --rebuild --generation-llm-mode openai --generation-model gpt-5.4-mini --generation-base-url https://api.openai.com/v1 --log-level INFO
```

Expected: every successfully selected item is processed in analysis→script→duration→title order and completed records are written to `data/scripts.json`.

- [ ] **Step 6: Validate generated answers and print safe acceptance output**

Validate through `ArtifactStore.read_scripts()` and assert:

```python
assert scripts
assert all(item.model == "gpt-5.4-mini" for item in scripts)
assert all(30 <= item.estimated_duration_seconds <= 60 for item in scripts)
assert all(item.playback_speed == 1.2 for item in scripts)
assert all(len(item.title_candidates) == 3 for item in scripts)
assert all(item.selected_title in item.title_candidates for item in scripts)
assert all(any("가" <= character <= "힣" for character in item.script) for item in scripts)
```

Print only item ID, audience relevance, script, estimated seconds, candidates, and selected title. Do not print credentials, source bodies, comments, or raw API responses containing request headers.

- [ ] **Step 7: Run live GeekNews smoke and final regression**

Run:

```powershell
python -m pytest tests/live/test_geeknews_live.py -m live -q
python -m pytest -m "not live" -q
python -m pip check
```

Expected: the live source test and complete offline suite pass.

- [ ] **Step 8: Perform repository and server safety audit**

Run:

```powershell
/diff
git diff --check
git status --short
git diff HEAD
git check-ignore .env data/curated.json data/scripts.json data/state.sqlite
curl.exe -sS --max-time 5 http://127.0.0.1:8000/metrics
```

On Windows, `/diff` is expected to be unavailable; record that error and use the Git diff commands as the required fallback. Confirm no API key, `.env`, JSON/SQLite generated data, dependency directory, build output, or raw content is staged. Confirm Furiosa reports zero running and waiting requests after acceptance.

- [ ] **Step 9: Commit acceptance fixes only when needed**

If acceptance required code or test changes, stage only those explicit project files and commit:

```powershell
git commit -m "Verify Stage 3 live generation"
```

Do not commit ignored generated data or secrets. If acceptance required no fixes, leave the already verified implementation commits unchanged.
