# Community Shorts Stage 1–2 Pipeline Design

## Scope

This delivery implements only Stage 1 and Stage 2 of the PRD:

- Stage 1 collects new posts, engagement metrics, and selected comments.
- Stage 2 applies a deterministic prefilter and uses an OpenAI-compatible local LLM to select and summarize candidates in Korean.
- The pipeline writes `data/items.json` and `data/curated.json`.
- Stage 3 script generation, title generation, title scoring, and Markdown output are excluded.

Eight sources are registered. Only GeekNews is enabled for today's live test:

| Source ID | URL | Adapter | Default |
|---|---|---|---|
| `geeknews` | `https://news.hada.io/new` | GeekNews HTML | enabled |
| `hackernews` | `https://news.ycombinator.com/newest` | Hacker News API | disabled |
| `reddit_hacking` | `https://www.reddit.com/r/hacking/` | Reddit OAuth | disabled |
| `reddit_ai_agents` | `https://www.reddit.com/r/AI_Agents/` | Reddit OAuth | disabled |
| `reddit_openai` | `https://www.reddit.com/r/OpenAI/` | Reddit OAuth | disabled |
| `reddit_geminiai` | `https://www.reddit.com/r/GeminiAI/` | Reddit OAuth | disabled |
| `reddit_claude` | `https://www.reddit.com/r/claude/` | Reddit OAuth | disabled |
| `dcinside_singularity` | `https://gall.dcinside.com/mgallery/board/lists/?id=thesingularity` | DCInside HTML | disabled |

Reddit communities share one adapter implementation and differ only by configuration. Reddit requests require OAuth credentials and a descriptive User-Agent before those sources can be enabled.

## Runtime and Configuration

The project targets Python 3.11 or later and exposes a CLI:

```powershell
python -m community_shorts ingest
python -m community_shorts curate
python -m community_shorts run
```

`sources.yaml` contains source URLs, languages, rate limits, fetch limits, and enable flags. Application settings come from environment variables or CLI arguments. Secrets are never stored in `sources.yaml` and `.env` is ignored by Git.

`run` executes ingest followed by curate. Either stage can be rerun independently against its JSON boundary.

## Stage 1: Ingest

All source adapters implement this interface:

```python
class SourceAdapter(Protocol):
    source_id: str

    def fetch_new(self, since: datetime) -> list[RawItem]: ...
```

The adapter registry maps configuration types to `GeekNewsAdapter`, `HackerNewsAdapter`, `RedditAdapter`, or `DCInsideAdapter`. This delivery implements each adapter contract and parser, but only GeekNews participates in the live acceptance test.

GeekNews collection parses the `/new` listing for item identifiers, titles, summaries, points, comment counts, and discussion URLs. It fetches discussion pages for new items to collect up to three top comments. Requests use an explicit User-Agent, timeout, per-source delay, jitter, retry with exponential backoff, and `robots.txt` enforcement.

The other adapters use these access methods:

- Hacker News uses its official Firebase API.
- Reddit uses OAuth API responses and one configurable subreddit per source entry.
- DCInside uses a source-specific HTML parser with a conservative request rate.

SQLite records item IDs and stage states. An item is marked ingested only after `items.json` is atomically replaced, preventing a write failure from losing an item. `items.json` stores complete `RawItem` objects for internal processing. Original bodies never appear in the final Stage 2 output.

## Stage 2: Prefilter, Select, and Summarize

Stage 2 reads uncurated records from `items.json`. Spam and exact URL duplicates are removed before scoring. Engagement is normalized within each source over the rolling 24-hour candidate window so a small community is not dominated by a large one.

The reaction score is:

```text
reaction_score =
  percentile(log1p(likes_or_points)) * 0.55
  + percentile(log1p(comment_count)) * 0.45
```

Each selection cycle sends at most 20 candidates to the LLM, with at most five from one source. For today's GeekNews-only run, the source and global prefilter limits are both 10.

The local OpenAI-compatible LLM returns one schema-validated object per candidate with:

- `pass`
- `reaction_score`
- `provocation_score`
- `mass_appeal_score`
- `curation_score`
- `fidelity_score`
- `curation_reason`
- `summary`
- `key_claim`
- `hook_points`
- `tone`
- `source_language`
- `output_language`
- `model`

The final selection score is:

```text
curation_score =
  reaction_score * 0.40
  + provocation_score * 0.25
  + mass_appeal_score * 0.35
```

After the LLM returns scores, the pipeline independently recomputes `curation_score` and ranks candidates. The item receives `pass=true` only when all gates hold:

- `curation_score >= 0.62`
- `fidelity_score >= 0.75`
- the safety and spam checks pass
- the item falls within the cycle and daily caps

The normal operating schedule is collection every 30 minutes and curation at 09:00, 15:00, and 21:00 Asia/Seoul. Each curation cycle evaluates up to 20 prefiltered candidates and keeps the top two passing items. An unused quota can be filled by a later cycle. Since Stage 3 is excluded, Stage 2 targets six curated results per day and enforces a hard cap of eight.

## English Input and Korean Output

Each source declares `source_language` as `ko` or `en`. All Stage 2 prompts require `output_language: ko`. For English sources, the model must:

- write the summary, claim, hook points, tone, and reason in Korean;
- preserve product names, people, organizations, and technical terms when translation would reduce precision;
- avoid adding facts that are absent from the source;
- score semantic fidelity between the Korean output and the source text.

The prompt handles selection and Korean summarization in one call. A separate translation model or translation stage is not introduced.

## LLM Boundary and Offline Verification

The production client targets the OpenAI-compatible `/v1/chat/completions` endpoint exposed by Furiosa-LLM. Its base URL, model ID, API key placeholder, and timeout are configurable. JSON output is validated and retried once after a schema or parse failure. A second failure is logged and skipped without aborting other items.

A deterministic fixture LLM implements the same internal interface. It exists only for automated tests and local end-to-end verification when the NPU server is unavailable; production runs do not silently fall back to fixture output.

## Files and Failure Handling

JSON artifacts are written through a temporary file and atomic rename. Logs include the run ID, source ID, item ID, stage, duration, retry count, and error class without including secrets or full original content.

Failure behavior is isolated:

- One source failure does not discard successful source results.
- One malformed item does not stop its batch.
- LLM schema failure is retried once, then recorded as failed.
- Existing `items.json` remains available if Stage 2 fails.
- SQLite and JSON updates are ordered to avoid falsely marking unwritten items complete.

## Testing and Acceptance

Tests follow red-green-refactor and cover observable behavior:

- adapter parser fixtures for GeekNews, Hacker News, Reddit, and DCInside;
- source registration and enable flags;
- duplicate and seen-ID handling;
- source-local reaction normalization and weighted scoring;
- score, fidelity, daily-cap, and language gates;
- atomic JSON boundaries and SQLite state transitions;
- LLM JSON retry and per-item failure isolation;
- English input producing schema-valid Korean fields through the fixture client;
- end-to-end fixture run producing `items.json` and `curated.json`;
- an opt-in GeekNews live smoke test.

Today's acceptance run enables only GeekNews and must collect live records into `items.json`. Stage 2 is verified with the real Furiosa endpoint when it is available; otherwise the fixture client verifies pipeline behavior and the unavailable endpoint is reported explicitly. No Stage 3 files are produced.
