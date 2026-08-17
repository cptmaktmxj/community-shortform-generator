from datetime import datetime
from io import StringIO

from community_shorts.progress import ConsoleProgress, ProgressEvent


class TtyBuffer(StringIO):
    """String buffer that behaves like an interactive terminal."""

    def isatty(self) -> bool:
        """Report interactive output so ANSI rendering can be verified."""

        return True


def test_console_progress_formats_stage_position_message_and_detail() -> None:
    """Catch progress output losing the state or item position users need."""

    stream = StringIO()
    progress = ConsoleProgress(
        stream=stream,
        clock=lambda: datetime(2026, 8, 17, 21, 5, 9),
    )

    progress.emit(
        ProgressEvent(
            stage="ingest",
            status="running",
            message="news.hada.io 글 수집 중",
            current=1,
            total=2,
            detail="최신 글 목록 확인",
        )
    )

    assert stream.getvalue() == (
        "[21:05:09] [수집] [1/2] news.hada.io 글 수집 중 · 최신 글 목록 확인\n"
    )


def test_console_progress_uses_no_ansi_codes_for_redirected_output() -> None:
    """Catch terminal color escapes polluting redirected logs."""

    stream = StringIO()
    ConsoleProgress(stream=stream).emit(
        ProgressEvent(stage="curate", status="completed", message="선별 완료")
    )

    assert "\x1b[" not in stream.getvalue()


def test_console_progress_colors_status_on_an_interactive_terminal() -> None:
    """Catch running, completed, and failed states becoming visually identical."""

    stream = TtyBuffer()
    ConsoleProgress(stream=stream).emit(
        ProgressEvent(stage="generate", status="failed", message="제목 생성 실패")
    )

    assert stream.getvalue().startswith("\x1b[31m")
    assert stream.getvalue().endswith("\x1b[0m\n")
