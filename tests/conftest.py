import json
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_text():
    """Return a loader for UTF-8 HTML fixture text."""

    return lambda name: (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def fixture_json():
    """Return a loader for JSON fixture values."""

    return lambda name: json.loads((FIXTURES / name).read_text(encoding="utf-8"))
