import httpx
import pytest

from community_shorts.http import HttpClient, RobotsDeniedError


@pytest.mark.asyncio
async def test_http_client_does_not_request_page_denied_by_robots() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private")
        return httpx.Response(200, text="secret")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as raw_client:
        client = HttpClient(raw_client, user_agent="test-agent", rate_limit_seconds=0)
        with pytest.raises(RobotsDeniedError):
            await client.get_text("https://example.com/private/story")

    assert requested == ["/robots.txt"]


@pytest.mark.asyncio
async def test_http_client_retries_server_error_twice() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        attempts += 1
        return httpx.Response(503 if attempts < 3 else 200, text="ok")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as raw_client:
        client = HttpClient(
            raw_client,
            user_agent="test-agent",
            rate_limit_seconds=0,
            backoff_seconds=0,
        )
        assert await client.get_text("https://example.com/story") == "ok"

    assert attempts == 3
