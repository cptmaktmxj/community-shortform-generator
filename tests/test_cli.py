import os
from pathlib import Path

from community_shorts.cli import build_parser
from community_shorts.cli import main


def test_help_exits_successfully_and_exposes_only_stage_one_and_two(capsys) -> None:
    assert main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "ingest" in output
    assert "curate" in output
    assert "run" in output
    assert "generate" not in output


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
