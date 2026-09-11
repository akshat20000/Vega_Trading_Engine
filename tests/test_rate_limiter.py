"""
Tests for Phase 10B: Rate Limiter.

Verifies:
    1. Requests within capacity proceed immediately without delay.
    2. Requests exceeding capacity are delayed until the window opens.
    3. Window expiration resets available capacity.
    4. Non-blocking try_acquire behaves correctly.
    5. Injected clock and sleep ensure zero wall-clock sleep overhead.
    6. Context manager and decorator usage (sync and async).
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from vega.broker.rate_limiter import RateLimiter


class MockClock:
    """Deterministic simulated clock for rate limiter testing."""

    def __init__(self, initial_time: float = 1000.0) -> None:
        self.current_time = float(initial_time)
        self.sleep_calls: List[float] = []

    def now(self) -> float:
        return self.current_time

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current_time += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep(seconds)


def test_requests_within_capacity_proceed_immediately() -> None:
    """Verify first 5 requests within 1-second window have zero delay."""
    clock = MockClock()
    limiter = RateLimiter(
        max_requests=5,
        window_seconds=1.0,
        clock_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    delays = [limiter.acquire() for _ in range(5)]

    assert delays == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert clock.sleep_calls == []
    assert limiter.current_count == 5
    assert limiter.available_capacity == 0
    assert limiter.total_requests == 5
    assert limiter.total_delayed_requests == 0


def test_request_beyond_capacity_is_delayed_and_allowed_after_window() -> None:
    """
    Verify 6th request is delayed until the 1st request rolls out of the window:
        request 1 -> allowed
        ...
        request 5 -> allowed
        request 6 -> waits
        request 6 -> allowed after window
    """
    clock = MockClock(initial_time=100.0)
    limiter = RateLimiter(
        max_requests=5,
        window_seconds=1.0,
        clock_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    # First 5 requests at t=100.0
    for _ in range(5):
        assert limiter.acquire() == 0.0

    assert clock.now() == 100.0
    assert clock.sleep_calls == []

    # 6th request at t=100.0 must wait until oldest (t=100.0) expires (t=101.0)
    wait_time = limiter.acquire()

    assert wait_time == 1.0
    assert clock.sleep_calls == [1.0]
    # Clock advanced to 101.0
    assert clock.now() == 101.0
    assert limiter.total_requests == 6
    assert limiter.total_delayed_requests == 1
    assert limiter.total_wait_time == 1.0


def test_sliding_window_partial_expiry_delays_correctly() -> None:
    """Verify request is delayed by only the delta needed for the oldest request to expire."""
    clock = MockClock(initial_time=100.0)
    limiter = RateLimiter(
        max_requests=3,
        window_seconds=1.0,
        clock_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    # Req 1 at 100.0
    limiter.acquire()
    # Req 2 at 100.2
    clock.current_time = 100.2
    limiter.acquire()
    # Req 3 at 100.4
    clock.current_time = 100.4
    limiter.acquire()

    # Req 4 at 100.6 (capacity 3 full, oldest was at 100.0, expires at 101.0)
    # Needed wait is (100.0 + 1.0) - 100.6 = 0.4s
    clock.current_time = 100.6
    wait = limiter.acquire()

    assert pytest.approx(wait, 1e-6) == 0.4
    assert pytest.approx(clock.now(), 1e-6) == 101.0
    assert len(clock.sleep_calls) == 1
    assert pytest.approx(clock.sleep_calls[0], 1e-6) == 0.4


def test_window_expiration_resets_capacity() -> None:
    """Verify that advancing the clock past window_seconds fully restores capacity."""
    clock = MockClock(initial_time=100.0)
    limiter = RateLimiter(
        max_requests=5,
        window_seconds=1.0,
        clock_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    for _ in range(5):
        limiter.acquire()

    assert limiter.available_capacity == 0

    # Advance clock beyond window
    clock.current_time = 102.5
    assert limiter.available_capacity == 5
    assert limiter.current_count == 0

    # Can make 5 immediate requests again
    delays = [limiter.acquire() for _ in range(5)]
    assert delays == [0.0] * 5
    assert clock.sleep_calls == []


def test_try_acquire_non_blocking() -> None:
    """Verify try_acquire returns False without sleeping when at capacity."""
    clock = MockClock()
    limiter = RateLimiter(
        max_requests=2,
        window_seconds=1.0,
        clock_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    # 3rd request should fail immediately without waiting
    assert limiter.try_acquire() is False
    assert clock.sleep_calls == []
    assert limiter.total_requests == 2


def test_rate_limiter_context_manager_and_decorator() -> None:
    """Verify context manager and decorator wrappers."""
    clock = MockClock()
    limiter = RateLimiter(
        max_requests=2,
        window_seconds=1.0,
        clock_fn=clock.now,
        sleep_fn=clock.sleep,
    )

    with limiter:
        val1 = "executed 1"
    assert val1 == "executed 1"

    @limiter
    def decorated_call() -> str:
        return "executed 2"

    assert decorated_call() == "executed 2"
    assert limiter.total_requests == 2


def test_rate_limiter_async() -> None:
    """Verify async acquire and async context manager."""
    clock = MockClock()
    limiter = RateLimiter(
        max_requests=2,
        window_seconds=1.0,
        clock_fn=clock.now,
        async_sleep_fn=clock.async_sleep,
    )

    async def _run():
        d1 = await limiter.acquire_async()
        assert d1 == 0.0
        d2 = await limiter.acquire_async()
        assert d2 == 0.0

        # 3rd request must wait 1.0s
        d3 = await limiter.acquire_async()
        assert d3 == 1.0

        # Async context manager
        async with limiter:
            pass

    asyncio.run(_run())
    assert clock.sleep_calls == [1.0]
    assert limiter.total_requests == 4
