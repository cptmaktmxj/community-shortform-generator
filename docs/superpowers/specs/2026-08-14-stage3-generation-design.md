# Community Shorts Stage 2 Recalibration and Stage 3 Generation Design

## Scope

This delivery recalibrates Stage 2 selection and implements Stage 3 through Korean analysis, narration-script generation, estimated-duration validation, and title generation. It does not synthesize audio, render video, publish content, or provide investment recommendations.

The pipeline continues to use K-EXAONE for Stage 2 scoring and safety evaluation. Stage 3 uses `gpt-5.4-mini` through the official OpenAI API. The model supports Structured Outputs and is intended for efficient high-volume work according to the [official model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

## Target Audience

The target is not a technical practitioner. The intended viewer is a general audience member interested in software used in everyday life or work, AI, and technology stocks. A topic scores highly when it has a clear connection to time, money, work, common software use, AI adoption, employment, a recognizable technology company, or an investment-relevant market event.

Developer-only implementation details do not become broadly appealing merely because they are novel. Their score can rise only when the source establishes a concrete effect on the target audience.

## Stage 2 Score Recalibration

Stage 2 keeps reaction scoring deterministic and K-EXAONE responsible for `provocation_score`, `mass_appeal_score`, `fidelity_score`, and `safety_ok`. The final score changes to:

```text
curation_score =
  reaction_score * 0.20
  + provocation_score * 0.40
  + mass_appeal_score * 0.40
```

An item passes only when every gate holds:

```text
curation_score >= 0.62
provocation_score >= 0.35
mass_appeal_score >= 0.45
fidelity_score >= 0.75
safety_ok = true
```

### Mass-Appeal Anchors

| Score | Meaning |
|---|---|
| `0.1` | Only a specialist in the specific technical field is likely to understand or care. |
| `0.3` | Interesting mainly to technology enthusiasts, without a demonstrated daily or work impact. |
| `0.5` | Understandable and relevant to general viewers interested in software, AI, or technology companies. |
| `0.7` | Has a direct, source-supported effect on time, money, work, employment, or investment judgment. |
| `0.9` | Has an immediate and broad effect on many people's daily lives or jobs. |

A framework, library, programming-language, infrastructure, or developer-tool story is capped below `0.45` unless the source explicitly establishes a concrete impact on the target audience.

### Provocation Anchors

| Score | Meaning |
|---|---|
| `0.1` | Routine launch, update, maintenance release, or ordinary announcement. |
| `0.3` | An unusual choice or disagreement mainly within a specialist community. |
| `0.5` | Credibly challenges a common expectation held by the target audience. |
| `0.7` | Presents clear conflict, stakes, or disruption affecting work, costs, jobs, or a market. |
| `0.9` | A highly unusual, well-supported event with broad and immediate consequences. |

Technical novelty alone does not justify a score above `0.3`. Every score response includes a Korean explanation tied to one of these anchors.

### Reaction Score

Reaction remains a source-local 55/45 combination of like and comment percentiles. Its reduced 20% weight prevents a weak absolute response from dominating merely because the item ranks first in a small collection window. Historical or cross-source absolute normalization remains outside this delivery because only GeekNews is enabled for the current acceptance run.

## Explicit Stage 2 Rebuild

The CLI adds:

```powershell
python -m community_shorts curate --rebuild
```

Normal `curate` behavior remains incremental. `--rebuild` evaluates the full current `items.json` candidate set without excluding earlier Stage 2 terminal states. It builds a complete replacement result before changing the existing artifact.

On success, the service atomically replaces `curated.json` rather than merging it and updates SQLite Stage 2 states to match the replacement. On evaluation or artifact-write failure, the previous `curated.json` and Stage 2 terminal states remain usable. A rebuild never modifies `items.json` and never puts original bodies or comments into `curated.json`.

## Stage 3 Data Flow

Stage 3 runs only for `CuratedItem` records with `pass=true` and `safety_ok=true`:

```text
curated.json
  -> structured analysis
  -> narration script
  -> local duration estimate
  -> zero to two script revisions
  -> title candidates and selection
  -> scripts.json
```

The required order is analysis, script, duration validation, and title. A title request is never made before the script passes duration validation.

### Analysis Contract

The analysis call receives only the Stage 2 transformed fields and scoring metadata. It does not receive the original full body or comments. The Structured Output contains:

```json
{
  "topic_category": "ai_impact",
  "audience_relevance": "일반 직장인의 업무 방식에 미치는 영향",
  "core_facts": ["스크립트에 반드시 포함할 검증된 사실"],
  "angle": "이 변화가 내 업무에 어떤 의미인지",
  "hook_strategy": "기존 통념과 달라진 점",
  "claims_to_avoid": ["입력으로 확인할 수 없는 확대 해석"],
  "recommended_tone": "빠르고 명료한 정보형"
}
```

`topic_category` is one of `ai_impact`, `consumer_software`, `work_productivity`, `tech_market`, or `technology_trend`.

### Script Contract

The script call receives the validated analysis and Stage 2 transformed fields. Its Korean narration follows this sequence:

1. Open with the central impact, reversal, or conflict without a greeting.
2. Explain what happened using the minimum necessary context.
3. Cover the analysis `core_facts` without introducing new facts.
4. Connect the event to daily life, work, AI use, or technology-market interest.
5. Close with one concise implication or question.

Channel promotion, subscription requests, URLs, Markdown, unsupported numbers, invented causality, profit guarantees, and direct buy-or-sell instructions are prohibited.

Attention-grabbing expressions such as “충격”, “무조건”, and “드디어 밝혀졌다” are allowed. They are style choices only: they cannot change the factual meaning, introduce an unsupported event, imply an unsupported certainty, or fabricate a number or outcome.

### Title Contract

The title call receives the validated final script and analysis. It returns three candidates:

- a direct-impact title;
- a question title;
- a conventional-wisdom reversal title.

Each candidate is 18–34 Korean characters excluding surrounding whitespace, is entailed by the script, and follows the same fact-versus-style distinction as the script. Strong or provocative wording is allowed, while unsupported facts, fabricated urgency, profit guarantees, and direct investment instructions are not.

The model recommends one candidate. Local code rejects candidates that violate length, forbidden-content, or required-field rules. If all candidates fail, the item stops at `title_failed` without repeating analysis or script generation.

## Duration Estimation

Stage 3 does not call a TTS provider in this delivery. It uses a replaceable estimator configured as:

```yaml
script:
  target_seconds: 45
  min_seconds: 30
  max_seconds: 60
  playback_speed: 1.2
  base_spoken_units_per_second: 4.3
  max_revisions: 2
```

The estimate is:

```text
seconds_at_1x = spoken_units / base_spoken_units_per_second + pause_seconds
estimated_seconds = seconds_at_1x / playback_speed
```

Hangul syllables contribute one spoken unit. English tokens contribute a length-based Korean-pronunciation estimate, with an additional abbreviation cost for all-capital tokens. Number runs contribute a digit- and unit-aware estimate. Whitespace and Markdown syntax contribute no spoken units. URLs are invalid script content.

Default pauses are 0.15 seconds for a comma, colon, or semicolon; 0.35 seconds for a period, question mark, or exclamation mark; and 0.45 seconds for a paragraph boundary. A paragraph boundary adjacent to sentence punctuation uses the larger pause rather than adding both.

The generation target is 45 seconds at 1.2x. An estimate from 40 through 50 seconds is `ideal`; 30 through 60 seconds is `acceptable`; anything else requires revision. A short script is expanded using missing context or audience impact from the analysis. A long script removes repetition and secondary detail while preserving every `core_fact`.

After no more than two revisions, a remaining out-of-range script becomes `duration_failed`. No title call occurs for that item. Estimator constants remain configuration values so later voice calibration or an actual TTS-duration provider can replace the implementation without changing the generation service.

## Models and Configuration

Stage 2 and Stage 3 use separate configuration sections and credentials:

```yaml
llm:
  mode: fixture
  base_url: http://127.0.0.1:11434/v1
  model: qwen3:8b
  api_key_env: CURATION_LLM_API_KEY
  api_key_default: ollama
  timeout_seconds: 120

generation_llm:
  mode: openai
  base_url: https://api.openai.com/v1
  model: gpt-5.4-mini
  api_key_env: OPENAI_API_KEY
  timeout_seconds: 120
```

Production Stage 3 uses the OpenAI Responses API with Structured Outputs and `gpt-5.4-mini`. Tests use a deterministic fixture implementation selected explicitly; production never silently falls back to fixture output.

## Persistence and Resumption

The CLI adds:

```powershell
python -m community_shorts generate
python -m community_shorts generate --rebuild
```

`generate` processes curated items that do not already have a completed Stage 3 result. `generate --rebuild` explicitly regenerates all currently curated items. The existing `run` command remains Stage 1–2 for backward compatibility and does not unexpectedly invoke a paid OpenAI endpoint.

SQLite stores one Stage 3 job per item, including status, validated analysis JSON, current validated script, estimated duration, revision count, title data, model name, timestamps, and a sanitized error. Intermediate state enables these resumptions:

- analysis complete and script failed: reuse analysis;
- script complete and title failed: reuse the validated script;
- completed: make no API call during a normal rerun;
- explicit rebuild: create a fresh Stage 3 attempt for the current curated set.

Only completed records are written to `data/scripts.json`. The final artifact contains:

```json
{
  "item_id": "geeknews:123",
  "source_id": "geeknews",
  "source_url": "https://example.com/item",
  "analysis": {},
  "script": "한국어 낭독 대본",
  "estimated_duration_seconds": 44.8,
  "duration_class": "ideal",
  "playback_speed": 1.2,
  "title_candidates": ["후보 하나", "후보 둘", "후보 셋"],
  "selected_title": "최종 제목",
  "model": "gpt-5.4-mini"
}
```

Atomic temporary-file replacement protects `scripts.json`. It never stores the original full body or comments.

## Failure Handling

One item failure never stops other items. Structured-output or validation failure is retried once within the current substage. Duration revisions are separate from schema retries and are capped at two. Errors are logged with item ID, substage, exception class, and attempt count without credentials or full source bodies.

Stage 3 statuses are `pending`, `analyzed`, `scripted`, `duration_failed`, `title_failed`, `completed`, and `failed`. Safety remains a Stage 2 hard gate; Stage 3 cannot override it.

## Testing and Acceptance

Tests follow red-green-refactor and cover:

- the exact 20/40/40 score weights and all independent gates;
- anchored mass-appeal and provocation prompt language;
- the developer-only mass-appeal cap;
- rebuild replacement instead of merge and preservation on failure;
- analysis, script, duration, and title call ordering;
- Korean, English, number, punctuation, and paragraph duration estimates at 1.2x;
- short and long revision prompts and the two-revision limit;
- no title call after `duration_failed`;
- title length and forbidden-content validation;
- intermediate-state resumption without repeated successful calls;
- atomic `scripts.json` writes without original body or comments;
- fixture end-to-end Stage 3 generation;
- complete Stage 1–3 offline regression.

Live acceptance first runs K-EXAONE Stage 2 with `curate --rebuild`, validates the replacement `curated.json`, then runs `generate` against `gpt-5.4-mini`. Acceptance requires at least one completed script with Korean analysis, a 30–60 second 1.2x estimate, three title candidates, one selected title, schema-valid persisted output, and no leaked credential or original full body.

## Out of Scope

- TTS audio synthesis or actual audio-duration measurement
- voice selection and acoustic calibration
- video rendering, captions, thumbnails, or publishing
- automatic trading or personalized investment advice
- analytics-based title optimization after publication
