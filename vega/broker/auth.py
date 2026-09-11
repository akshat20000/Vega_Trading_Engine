"""
Authentication, token refresh, and API client reliability abstractions.

Provides:
    - TokenProvider (Protocol) & InMemoryTokenProvider: Token lifecycle & refresh management.
    - APIClient: Resilient HTTP client coordinating TokenProvider, RateLimiter, and RetryPolicy.
    - AuthenticationError, TokenRefreshError, APIResponseError: Structured error models.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Protocol, TypeVar

from vega.broker.rate_limiter import RateLimiter
from vega.broker.retry import RetryPolicy


# ─── Exceptions ───────────────────────────────────────────────────────────────


class AuthenticationError(Exception):
    """Raised when authentication fails (e.g. 401 persisted after token refresh)."""
    pass


class TokenRefreshError(AuthenticationError):
    """Raised when the token refresh operation itself fails."""
    pass


class APIResponseError(Exception):
    """Raised when an API endpoint returns an HTTP error status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


# ─── Data Models ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class APIResponse:
    """Standardized representation of a simulated or actual API response."""

    status_code: int
    data: Any = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """True if HTTP status code is in 2xx range."""
        return 200 <= self.status_code < 300


# ─── Token Provider Protocol & Implementation ─────────────────────────────────


class TokenProvider(Protocol):
    """Abstract interface for managing access tokens and lifecycle refreshes."""

    def get_access_token(self) -> str:
        """Retrieve the current active access token."""
        ...

    def refresh_token(self) -> str:
        """Acquire a new access token via refresh flow and return it."""
        ...


class InMemoryTokenProvider:
    """
    In-memory TokenProvider maintaining an access token and refresh counter.

    Args:
        initial_token: Initial valid access token.
        refresh_fn: Optional callable producing a new token string upon refresh.
    """

    def __init__(
        self,
        initial_token: str = "token_initial",
        refresh_fn: Callable[[], str] | None = None,
    ) -> None:
        self._current_token = str(initial_token)
        self._refresh_fn = refresh_fn
        self.refresh_count = 0

    def get_access_token(self) -> str:
        """Return the current cached access token."""
        return self._current_token

    def refresh_token(self) -> str:
        """Execute refresh flow to produce and store a new access token."""
        self.refresh_count += 1
        if self._refresh_fn is not None:
            self._current_token = str(self._refresh_fn())
        else:
            self._current_token = f"token_refreshed_{self.refresh_count}"
        return self._current_token


# ─── API Client ───────────────────────────────────────────────────────────────


class APIClient:
    """
    Broker API client integrating TokenProvider, RateLimiter, and RetryPolicy.

    Core Invariants:
        On HTTP 401 Unauthorized:
            1. Calls `token_provider.refresh_token()` once.
            2. Retries the request ONCE with the refreshed token.
            3. If the retry also returns 401, raises AuthenticationError immediately
               (never endlessly retrying 401).
        Non-401 errors (400, 403, 404, 500) do NOT trigger token refresh.
        5xx server errors / connection failures are handled by RetryPolicy if configured.
    """

    def __init__(
        self,
        token_provider: TokenProvider,
        transport: Callable[..., Any] | None = None,
        rate_limiter: RateLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """
        Initialize APIClient.

        Args:
            token_provider: Manages access token acquisition and refresh.
            transport: Underlying HTTP transport function(endpoint, method, headers, params).
            rate_limiter: Optional RateLimiter to throttle outgoing requests.
            retry_policy: Optional RetryPolicy for transient network / 5xx retries.
        """
        self.token_provider = token_provider
        self._transport = transport or self._default_transport
        self.rate_limiter = rate_limiter
        self.retry_policy = retry_policy

    @staticmethod
    def _default_transport(
        endpoint: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """Default stub transport returning HTTP 200."""
        return APIResponse(status_code=200, data={"endpoint": endpoint, "status": "ok"})

    def _execute_send(
        self,
        endpoint: str,
        token: str,
        params: dict[str, Any] | None,
        method: str,
    ) -> APIResponse:
        """Execute transport with Bearer token header, raising on 5xx to enable retries."""
        headers = {"Authorization": f"Bearer {token}"}
        try:
            res = self._transport(endpoint=endpoint, method=method, headers=headers, params=params)
        except APIResponseError as err:
            if err.status_code == 401:
                return APIResponse(status_code=401, data={"error": err.message}, headers=headers)
            raise

        if isinstance(res, APIResponse):
            resp = res
        else:
            resp = APIResponse(status_code=200, data=res, headers=headers)

        if resp.status_code >= 500:
            raise APIResponseError(resp.status_code, f"Server error {resp.status_code}")

        return resp

    def request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        method: str = "GET",
    ) -> APIResponse:
        """
        Execute a synchronous API request with rate limiting and 401 token refresh.

        Returns:
            Successful APIResponse.

        Raises:
            AuthenticationError: If 401 persists after token refresh.
            TokenRefreshError: If token refresh operation fails.
            APIResponseError: If non-401 HTTP error occurs.
        """
        # 1. Enforce rate limit
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        def _attempt(t: str) -> APIResponse:
            if self.retry_policy is not None:
                return self.retry_policy.execute(self._execute_send, endpoint, t, params, method)
            return self._execute_send(endpoint, t, params, method)

        # 2. First attempt with active token
        current_token = self.token_provider.get_access_token()
        resp = _attempt(current_token)

        # 3. Handle 401 Unauthorized: refresh and retry ONCE
        if resp.status_code == 401:
            new_token = self.token_provider.refresh_token()
            resp2 = _attempt(new_token)
            if resp2.status_code == 401:
                raise AuthenticationError(
                    f"Request to {endpoint} unauthorized (401) after token refresh"
                )
            if not resp2.is_success:
                raise APIResponseError(resp2.status_code, f"API error {resp2.status_code} for {endpoint}")
            return resp2

        # 4. Handle other HTTP errors
        if not resp.is_success:
            raise APIResponseError(resp.status_code, f"API error {resp.status_code} for {endpoint}")

        return resp

    async def _execute_send_async(
        self,
        endpoint: str,
        token: str,
        params: dict[str, Any] | None,
        method: str,
    ) -> APIResponse:
        """Async transport executor raising on 5xx to enable retries."""
        headers = {"Authorization": f"Bearer {token}"}
        try:
            res = self._transport(endpoint=endpoint, method=method, headers=headers, params=params)
            if inspect.isawaitable(res):
                res = await res
        except APIResponseError as err:
            if err.status_code == 401:
                return APIResponse(status_code=401, data={"error": err.message}, headers=headers)
            raise

        if isinstance(res, APIResponse):
            resp = res
        else:
            resp = APIResponse(status_code=200, data=res, headers=headers)

        if resp.status_code >= 500:
            raise APIResponseError(resp.status_code, f"Server error {resp.status_code}")

        return resp

    async def request_async(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        method: str = "GET",
    ) -> APIResponse:
        """
        Execute an asynchronous API request with rate limiting and 401 token refresh.
        """
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire_async()

        async def _attempt_async(t: str) -> APIResponse:
            if self.retry_policy is not None:
                return await self.retry_policy.execute_async(
                    self._execute_send_async, endpoint, t, params, method
                )
            return await self._execute_send_async(endpoint, t, params, method)

        current_token = self.token_provider.get_access_token()
        resp = await _attempt_async(current_token)

        if resp.status_code == 401:
            new_token = self.token_provider.refresh_token()
            resp2 = await _attempt_async(new_token)
            if resp2.status_code == 401:
                raise AuthenticationError(
                    f"Request to {endpoint} unauthorized (401) after token refresh"
                )
            if not resp2.is_success:
                raise APIResponseError(resp2.status_code, f"API error {resp2.status_code} for {endpoint}")
            return resp2

        if not resp.is_success:
            raise APIResponseError(resp.status_code, f"API error {resp.status_code} for {endpoint}")

        return resp
