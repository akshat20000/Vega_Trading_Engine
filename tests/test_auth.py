"""
Tests for Phase 10C: Authentication and Token Refresh.

Verifies:
    1. Valid access token is attached to headers on every request.
    2. Successful request proceeds without token refresh.
    3. HTTP 401 triggers token refresh and retries once.
    4. Refreshed token is used on the retry attempt.
    5. Persistent 401 (second 401) fails immediately without infinite retry loops.
    6. Token refresh failure propagates immediately to the caller.
    7. Token refresh is not repeated on subsequent requests once refreshed.
    8. Non-401 HTTP errors (400, 403, 500) do NOT trigger token refresh.
    9. Integration with RateLimiter and RetryPolicy.
    10. Async request_async executes the 401 refresh flow identically.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from vega.broker.auth import (
    APIClient,
    APIResponse,
    APIResponseError,
    AuthenticationError,
    InMemoryTokenProvider,
    TokenRefreshError,
)
from vega.broker.rate_limiter import RateLimiter
from vega.broker.retry import RetryPolicy


def test_valid_token_attached() -> None:
    """Verify Authorization Bearer header is automatically attached with current token."""
    captured_headers: List[Dict[str, str]] = []

    def mock_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
        captured_headers.append(headers)
        return APIResponse(status_code=200, data={"ok": True})

    provider = InMemoryTokenProvider(initial_token="test_token_abc")
    client = APIClient(token_provider=provider, transport=mock_transport)

    resp = client.request("/orders")

    assert resp.status_code == 200
    assert len(captured_headers) == 1
    assert captured_headers[0]["Authorization"] == "Bearer test_token_abc"
    assert provider.refresh_count == 0


def test_successful_request_does_not_refresh() -> None:
    """Verify successful request proceeds without triggering refresh."""
    provider = InMemoryTokenProvider(initial_token="valid_token")
    client = APIClient(token_provider=provider)

    resp = client.request("/positions")

    assert resp.status_code == 200
    assert provider.refresh_count == 0
    assert provider.get_access_token() == "valid_token"


def test_401_triggers_refresh_and_succeeds_on_retry() -> None:
    """
    Verify 401 triggers token refresh and the retried request succeeds:
        401 -> refresh -> retry once -> success
    """
    seen_tokens: List[str] = []

    def mock_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
        token = headers["Authorization"].replace("Bearer ", "")
        seen_tokens.append(token)
        if token == "expired_token":
            return APIResponse(status_code=401, data={"error": "Token expired"})
        return APIResponse(status_code=200, data={"profile": "trader"})

    provider = InMemoryTokenProvider(initial_token="expired_token")
    client = APIClient(token_provider=provider, transport=mock_transport)

    resp = client.request("/profile")

    assert resp.status_code == 200
    assert resp.data == {"profile": "trader"}
    assert provider.refresh_count == 1
    # Attempt 1 used expired_token, Attempt 2 used refreshed token
    assert seen_tokens == ["expired_token", "token_refreshed_1"]
    assert provider.get_access_token() == "token_refreshed_1"


def test_second_401_fails_immediately_no_infinite_loop() -> None:
    """
    Critical invariant: A second 401 on the retried request aborts immediately.
    Never endlessly retry a 401.
        401 -> refresh -> retry once -> 401 -> FAIL
    """
    attempts = 0

    def mock_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
        nonlocal attempts
        attempts += 1
        return APIResponse(status_code=401, data={"error": "Invalid credentials permanently"})

    provider = InMemoryTokenProvider(initial_token="bad_token_1")
    client = APIClient(token_provider=provider, transport=mock_transport)

    with pytest.raises(AuthenticationError) as exc_info:
        client.request("/sensitive-data")

    assert "unauthorized (401) after token refresh" in str(exc_info.value)
    # Exact attempt count: 1 initial attempt + 1 retry attempt
    assert attempts == 2
    # Only 1 refresh was attempted
    assert provider.refresh_count == 1


def test_refresh_failure_propagates_immediately() -> None:
    """Verify failure during refresh_token() aborts immediately without attempting retry."""
    def failing_refresh() -> str:
        raise TokenRefreshError("Refresh token expired on auth server")

    provider = InMemoryTokenProvider(initial_token="expired_token", refresh_fn=failing_refresh)

    def mock_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
        return APIResponse(status_code=401, data={"error": "Expired"})

    client = APIClient(token_provider=provider, transport=mock_transport)

    with pytest.raises(TokenRefreshError) as exc_info:
        client.request("/orders")

    assert "Refresh token expired on auth server" in str(exc_info.value)
    assert provider.refresh_count == 1


def test_token_refresh_not_performed_repeatedly() -> None:
    """Verify that once refreshed, subsequent requests use the new token directly."""
    seen_tokens: List[str] = []

    def mock_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
        token = headers["Authorization"].replace("Bearer ", "")
        seen_tokens.append(token)
        if token == "old_token":
            return APIResponse(status_code=401, data={"error": "Expired"})
        return APIResponse(status_code=200, data={"data": "ok"})

    provider = InMemoryTokenProvider(initial_token="old_token")
    client = APIClient(token_provider=provider, transport=mock_transport)

    # First request: 401 -> refresh -> retry success
    resp1 = client.request("/first")
    assert resp1.status_code == 200
    assert provider.refresh_count == 1

    # Second request: must directly use refreshed token, 0 additional refreshes
    resp2 = client.request("/second")
    assert resp2.status_code == 200
    assert provider.refresh_count == 1
    assert seen_tokens == ["old_token", "token_refreshed_1", "token_refreshed_1"]


def test_no_refresh_for_non_401_errors() -> None:
    """Verify non-401 HTTP errors (400, 403, 404, 500) do NOT trigger token refresh."""
    for error_code in [400, 403, 404, 500]:
        provider = InMemoryTokenProvider(initial_token="valid_token")

        def mock_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
            return APIResponse(status_code=error_code, data={"error": f"Error {error_code}"})

        client = APIClient(token_provider=provider, transport=mock_transport)

        with pytest.raises(APIResponseError) as exc_info:
            client.request("/test-error")

        assert exc_info.value.status_code == error_code
        assert provider.refresh_count == 0


def test_integration_with_rate_limiter_and_retry_policy() -> None:
    """Verify APIClient coordinates with RateLimiter and RetryPolicy seamlessly."""
    clock_time = 100.0
    sleeps: List[float] = []

    def mock_clock() -> float:
        return clock_time

    def mock_sleep(d: float) -> None:
        nonlocal clock_time
        sleeps.append(d)
        clock_time += d

    limiter = RateLimiter(
        max_requests=2,
        window_seconds=1.0,
        clock_fn=mock_clock,
        sleep_fn=mock_sleep,
    )

    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.1,
        sleep_fn=mock_sleep,
    )

    attempts = 0

    def mock_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise APIResponseError(503, "Service Unavailable")
        return APIResponse(status_code=200, data={"status": "recovered"})

    provider = InMemoryTokenProvider()
    client = APIClient(
        token_provider=provider,
        transport=mock_transport,
        rate_limiter=limiter,
        retry_policy=retry_policy,
    )

    resp = client.request("/health")

    assert resp.status_code == 200
    assert attempts == 2
    assert limiter.total_requests == 1
    # Slept for retry
    assert len(sleeps) == 1
    assert sleeps[0] == 0.1


def test_async_request_401_refresh_flow() -> None:
    """Verify async request_async executes 401 refresh flow identically."""
    async def _test():
        seen_tokens: List[str] = []

        async def async_transport(endpoint: str, method: str, headers: Dict[str, str], params: Any) -> APIResponse:
            token = headers["Authorization"].replace("Bearer ", "")
            seen_tokens.append(token)
            if token == "async_expired":
                return APIResponse(status_code=401, data={"error": "Expired"})
            return APIResponse(status_code=200, data={"async": "ok"})

        provider = InMemoryTokenProvider(initial_token="async_expired")
        client = APIClient(token_provider=provider, transport=async_transport)

        resp = await client.request_async("/async-data")

        assert resp.status_code == 200
        assert provider.refresh_count == 1
        assert seen_tokens == ["async_expired", "token_refreshed_1"]

    asyncio.run(_test())
