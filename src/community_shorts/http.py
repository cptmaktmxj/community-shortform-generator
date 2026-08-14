"""Polite asynchronous HTTP access with robots and retry handling."""

import asyncio
import random
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx


class RobotsDeniedError(PermissionError):
    """Raised before requesting a URL disallowed by robots.txt."""


class HttpClient:
    """Apply common request policy to an injected httpx client."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        user_agent: str,
        rate_limit_seconds: float,
        jitter_seconds: float = 0.1,
        backoff_seconds: float = 0.5,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._delay = rate_limit_seconds
        self._jitter = jitter_seconds
        self._backoff = backoff_seconds
        self._max_retries = max_retries
        self._robots: dict[str, RobotFileParser] = {}

    async def get_text(self, url: str) -> str:
        """Fetch an HTML page after checking its origin robots policy."""

        if not await self._robots_allowed(url):
            raise RobotsDeniedError(f"robots.txt disallows {url}")
        response = await self._request(url)
        return response.text

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> object:
        """Fetch and decode a JSON API response."""

        response = await self._request(url, headers=headers)
        return response.json()

    async def _robots_allowed(self, url: str) -> bool:
        """Load robots.txt once per origin and evaluate the requested URL."""

        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        parser = self._robots.get(origin)
        if parser is None:
            robots_url = f"{origin}/robots.txt"
            try:
                response = await self._request(robots_url, apply_delay=False)
                lines = response.text.splitlines() if response.status_code < 400 else []
            except httpx.HTTPError:
                lines = []
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(lines)
            self._robots[origin] = parser
        return parser.can_fetch(self._user_agent, url)

    async def _request(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        apply_delay: bool = True,
    ) -> httpx.Response:
        """Request a URL and retry transient failures with exponential backoff."""

        request_headers = {"User-Agent": self._user_agent, **(headers or {})}
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if apply_delay and (self._delay or self._jitter):
                await asyncio.sleep(self._delay + random.uniform(0, self._jitter))
            try:
                response = await self._client.get(url, headers=request_headers, timeout=15.0)
                if response.status_code not in {429} and response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_error = httpx.HTTPStatusError(
                    f"transient HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
            if attempt < self._max_retries and self._backoff:
                await asyncio.sleep(self._backoff * (2**attempt))
        assert last_error is not None
        raise last_error
