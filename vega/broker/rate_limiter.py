"""
Rate limiter utility for broker API interactions.

Provides:
    - RateLimiter: Deterministic, sliding-window rate limiter with delay-on-exceed policy.
"""

from __future__ import annotations

import asyncio
from collections import deque
import inspect
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RateLimiter:
    """
    In-memory, deterministic rate limiter that delays requests exceeding capacity.

    Policy:
        Requests within capacity (`max_requests` per `window_seconds`) proceed immediately.
        Requests beyond capacity are delayed until the oldest request falls out of the window.

    Attributes:
        max_requests: Maximum allowed requests within the time window.
        window_seconds: Duration of the sliding window in seconds (default 1.0s).
        clock_fn: Time provider (injectable for instant, deterministic testing).
        sleep_fn: Synchronous sleep provider (injectable).
        async_sleep_fn: Asynchronous sleep provider for coroutines.
    """

    def __init__(
        self,
        max_requests: int = 5,
        window_seconds: float = 1.0,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        async_sleep_fn: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if max_requests < 1:
            raise ValueError(f"max_requests must be >= 1, got {max_requests}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")

        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self.clock_fn = clock_fn
        self.sleep_fn = sleep_fn
        self.async_sleep_fn = async_sleep_fn

        self._timestamps: deque[float] = deque()
        self.total_requests = 0
        self.total_delayed_requests = 0
        self.total_wait_time = 0.0

    def _evict_expired(self, now: float) -> None:
        """Remove timestamps outside the current window."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    @property
    def current_count(self) -> int:
        """Number of requests recorded in the current active window."""
        now = self.clock_fn()
        self._evict_expired(now)
        return len(self._timestamps)

    @property
    def available_capacity(self) -> int:
        """Remaining requests permitted in the current window."""
        return max(0, self.max_requests - self.current_count)

    def try_acquire(self) -> bool:
        """
        Attempt to acquire a request token without waiting.

        Returns:
            True if capacity was available and consumed, False otherwise.
        """
        now = self.clock_fn()
        self._evict_expired(now)
        if len(self._timestamps) < self.max_requests:
            self._timestamps.append(now)
            self.total_requests += 1
            return True
        return False

    def acquire(self) -> float:
        """
        Acquire permission to proceed with a request, delaying if at capacity.

        Returns:
            Wait time in seconds (0.0 if capacity was immediately available).
        """
        wait_time = 0.0
        now = self.clock_fn()
        self._evict_expired(now)

        if len(self._timestamps) >= self.max_requests:
            oldest = self._timestamps[0]
            # Time when the oldest request will expire from window
            needed_wait = (oldest + self.window_seconds) - now
            if needed_wait > 0:
                wait_time = needed_wait
                self.sleep_fn(wait_time)
                # Re-evaluate current time and evict after sleep
                now = self.clock_fn()
                self._evict_expired(now)

        self._timestamps.append(now)
        self.total_requests += 1
        if wait_time > 0:
            self.total_delayed_requests += 1
            self.total_wait_time += wait_time

        return wait_time

    async def acquire_async(self) -> float:
        """
        Asynchronously acquire permission to proceed, awaiting if at capacity.

        Returns:
            Wait time in seconds (0.0 if capacity was immediately available).
        """
        wait_time = 0.0
        now = self.clock_fn()
        self._evict_expired(now)

        if len(self._timestamps) >= self.max_requests:
            oldest = self._timestamps[0]
            needed_wait = (oldest + self.window_seconds) - now
            if needed_wait > 0:
                wait_time = needed_wait
                res = self.async_sleep_fn(wait_time)
                if inspect.isawaitable(res):
                    await res
                now = self.clock_fn()
                self._evict_expired(now)

        self._timestamps.append(now)
        self.total_requests += 1
        if wait_time > 0:
            self.total_delayed_requests += 1
            self.total_wait_time += wait_time

        return wait_time

    def __enter__(self) -> RateLimiter:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def __aenter__(self) -> RateLimiter:
        await self.acquire_async()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def __call__(self, target_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Support decorator usage on sync and async functions."""
        if inspect.iscoroutinefunction(target_fn):
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await self.acquire_async()
                return await target_fn(*args, **kwargs)
            return async_wrapper

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            self.acquire()
            return target_fn(*args, **kwargs)

        return sync_wrapper
