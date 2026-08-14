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
