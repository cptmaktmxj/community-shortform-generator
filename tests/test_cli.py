import os
from pathlib import Path

from community_shorts.cli import build_parser
from community_shorts.cli import main
from community_shorts.storage import ArtifactStore
from tests.test_storage import make_curated


def test_help_exits_successfully_and_exposes_pipeline_commands(capsys) -> None:
    assert main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "ingest" in output
    assert "curate" in output
    assert "run" in output
    assert "generate" in output
    assert "clickbait" not in output


def test_missing_sources_file_is_configuration_error(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "ingest",
            "--sources",
            str(tmp_path / "missing.yaml"),
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )

    assert exit_code == 2
    assert "Invalid source configuration" in capsys.readouterr().err


def test_main_loads_dotenv_before_parsing(monkeypatch, tmp_path, capsys) -> None:
    """Catch a CLI startup that silently ignores the project's .env file."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COMMUNITY_USER_AGENT", raising=False)
    (tmp_path / ".env").write_text(
        "COMMUNITY_USER_AGENT=community-shorts-dotenv-test\n",
        encoding="utf-8",
    )

    assert main(["--help"]) == 0
    capsys.readouterr()
    assert os.environ["COMMUNITY_USER_AGENT"] == "community-shorts-dotenv-test"


def test_parser_defers_runtime_defaults_to_config_file() -> None:
    """Catch CLI defaults bypassing config.yaml with embedded environment values."""

    args = build_parser().parse_args(["curate"])

    assert args.config == Path("config.yaml")
    assert args.llm_mode is None
    assert args.base_url is None
    assert args.model is None


def test_generate_parser_defaults_to_gpt_ranked_titles() -> None:
    """Catch Stage 3 silently reverting to the legacy unjudged title path."""

    args = build_parser().parse_args(["generate"])

    assert args.title_mode == "gpt-ranked"


def test_generate_command_runs_stage_three_with_fixture_model(
    tmp_path, capsys
) -> None:
    """Catch the Stage 3 CLI failing to compose timing and generation dependencies."""

    data_dir = tmp_path / "data"
    ArtifactStore(data_dir).write_curated([make_curated("geeknews:1")])

    exit_code = main(
        [
            "generate",
            "--config",
            str(Path("config.yaml").resolve()),
            "--data-dir",
            str(data_dir),
            "--llm-mode",
            "fixture",
        ]
    )

    assert exit_code == 0
    assert len(ArtifactStore(data_dir).read_scripts()) == 1
    output = capsys.readouterr().out
    assert "분석 중" in output
    assert "스크립트 생성 중" in output
    assert "제목 후보 생성 중" in output
    assert "제목 생성 완료" in output
