"""Command-line entry point for Stage 1 and Stage 2."""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from community_shorts.adapters import build_adapter
from community_shorts.config import ConfigError, load_app_config, load_sources
from community_shorts.curate import CurateService
from community_shorts.generate import GenerateService
from community_shorts.generation_llm import (
    FixtureGenerationLlmClient,
    OpenAiGenerationLlmClient,
    OpenAiResponsesTransport,
)
from community_shorts.http import HttpClient
from community_shorts.ingest import IngestService
from community_shorts.llm import FixtureLlmClient, OpenAiChatTransport, OpenAiLlmClient
from community_shorts.progress import ConsoleProgress
from community_shorts.state import StateStore
from community_shorts.storage import ArtifactStore


LOGGER = logging.getLogger(__name__)
SEOUL = ZoneInfo("Asia/Seoul")


def build_parser() -> argparse.ArgumentParser:
    """Create Stage 1 through Stage 3 pipeline command parsers."""

    parser = argparse.ArgumentParser(
        prog="community-shorts",
        description="커뮤니티 수집 및 한국어 선별·요약 파이프라인",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("ingest", "Stage 1 신규 글과 댓글 수집"),
        ("curate", "Stage 2 선별 및 한국어 요약"),
        ("run", "Stage 1과 Stage 2 순차 실행"),
        ("generate", "Stage 3 분석·대본·제목 생성"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        _add_common_arguments(subparser)
        if command == "generate":
            subparser.add_argument("--rebuild", action="store_true")
            subparser.add_argument(
                "--title-mode",
                choices=("gpt-ranked", "legacy"),
                default="gpt-ranked",
            )
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add paths, LLM settings, and logging controls shared by commands."""

    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--sources", type=Path, default=Path("sources.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--state-db", type=Path)
    parser.add_argument("--since-hours", type=float, default=24.0)
    parser.add_argument(
        "--llm-mode",
        choices=("openai", "fixture"),
        default=None,
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute the requested stages, and return a stable exit code."""

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        runtime = load_app_config(args.config).resolve()
        if args.command == "generate":
            args.llm_mode = args.llm_mode or runtime.generation_llm_mode
            args.base_url = args.base_url or runtime.generation_llm_base_url
            args.model = args.model or runtime.generation_llm_model
            args.llm_api_key = runtime.generation_llm_api_key
            args.llm_timeout_seconds = runtime.generation_llm_timeout_seconds
        else:
            args.llm_mode = args.llm_mode or runtime.llm_mode
            args.base_url = args.base_url or runtime.llm_base_url
            args.model = args.model or runtime.llm_model
            args.llm_api_key = runtime.llm_api_key
            args.llm_timeout_seconds = runtime.llm_timeout_seconds
        args.user_agent = runtime.user_agent
        args.reddit_access_token = runtime.reddit_access_token
        args.script_timing = runtime.script_timing
        return asyncio.run(_execute(args))
    except (ConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        LOGGER.exception("pipeline command failed")
        print(f"Pipeline failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        if getattr(args, "llm_mode", None) == "openai" and args.command in {
            "curate",
            "run",
            "generate",
        }:
            print(_endpoint_diagnostic(args.base_url), file=sys.stderr)
        return 1


async def _execute(args: argparse.Namespace) -> int:
    """Compose runtime dependencies and execute selected pipeline stages."""

    data_dir: Path = args.data_dir
    state_path: Path = args.state_db or data_dir / "state.sqlite"
    store = ArtifactStore(data_dir)
    state = StateStore(state_path)
    now = datetime.now(SEOUL)
    progress = ConsoleProgress()

    if args.command == "generate":
        if args.llm_mode == "fixture":
            generation_llm = FixtureGenerationLlmClient()
        else:
            if not args.llm_api_key:
                raise ConfigError("OPENAI_API_KEY is required for Stage 3 generation")
            generation_llm = OpenAiGenerationLlmClient(
                transport=OpenAiResponsesTransport(
                    base_url=args.base_url,
                    api_key=args.llm_api_key,
                    timeout_seconds=args.llm_timeout_seconds,
                ),
                model=args.model,
            )
        report = await GenerateService(
            store,
            state,
            generation_llm,
            args.script_timing,
            gpt_title_ranking=args.title_mode == "gpt-ranked",
            progress=progress,
        ).run(now, rebuild=args.rebuild)
        LOGGER.info(
            "generate complete attempted=%d completed=%d duration_failed=%d title_failed=%d",
            report.attempted,
            report.completed,
            report.duration_failed,
            report.title_failed,
        )
        return 0

    if args.command in {"ingest", "run"}:
        sources = load_sources(args.sources)
        enabled = [source for source in sources if source.enabled]
        if not enabled:
            raise ConfigError("At least one source must be enabled")
        async with httpx.AsyncClient(follow_redirects=True) as raw_client:
            adapters = [
                build_adapter(
                    source,
                    HttpClient(
                        raw_client,
                        user_agent=args.user_agent,
                        rate_limit_seconds=source.rate_limit_seconds,
                    ),
                    reddit_access_token=args.reddit_access_token,
                )
                for source in enabled
            ]
            report = await IngestService(
                adapters,
                store,
                state,
                progress=progress,
                source_names={
                    source.source_id: urlsplit(str(source.url)).netloc
                    for source in enabled
                },
            ).run(
                now - timedelta(hours=args.since_hours)
            )
            LOGGER.info(
                "ingest complete collected=%d skipped_seen=%d failed_sources=%s",
                report.collected,
                report.skipped_seen,
                report.failed_sources,
            )

    if args.command in {"curate", "run"}:
        if args.llm_mode == "fixture":
            llm = FixtureLlmClient()
        else:
            transport = OpenAiChatTransport(
                base_url=args.base_url,
                api_key=args.llm_api_key,
                timeout_seconds=args.llm_timeout_seconds,
            )
            llm = OpenAiLlmClient(transport=transport, model=args.model)
        report = await CurateService(
            store, state, llm, progress=progress
        ).run(now)
        LOGGER.info(
            "curate complete evaluated=%d passed=%d safety_rejected=%d failed_items=%s",
            report.evaluated,
            report.passed,
            report.safety_rejected,
            report.failed_item_ids,
        )
    return 0


def _endpoint_diagnostic(base_url: str) -> str:
    """Describe localhost endpoint details without exposing credentials."""

    parsed = urlsplit(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return (
        f"LLM endpoint diagnostic: host={parsed.hostname}, port={port}, scheme={parsed.scheme}. "
        "Check that furiosa-llm serve is running and test both the configured URL and "
        f"https://127.0.0.1:{port}/v1 when TLS is enabled."
    )
