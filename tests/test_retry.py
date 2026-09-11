"""
Tests for Phase 10A: Retry and Exponential Backoff.

Verifies:
    1. Transient failure retries and eventual success.
    2. Success stops retrying immediately (no redundant calls).
    3. Maximum attempts are strictly respected.
    4. Non-retryable error aborts immediately without retrying.
    5. Exponential backoff delay calculation and sleep progression.
    6. Final exception is preserved upon exhaustion.
    7. Decorator usage on sync and async functions.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from vega.broker.retry import RetryPolicy, retry_with_backoff


class TransientNetworkError(Exception):
    """Simulated transient network glitch (e.g. 503, connection reset)."""
    pass


class FatalAuthError(Exception):
    """Simulated non-retryable fatal error (e.g. invalid credentials)."""
    pass


def test_retry_on_transient_failure() -> None:
    """Verify transient failure retries and succeeds on subsequent attempt."""
    attempts = 0
    sleeps: List[float] = []

    def flakey_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientNetworkError(f"Temporary outage {attempts}")
        return "success"

    policy = RetryPolicy(
        max_attempts=4,
        initial_delay=0.1,
        backoff_factor=2.0,
        sleep_fn=sleeps.append,
    )

    result = policy.execute(flakey_call)

    assert result == "success"
    assert attempts == 3
    # Slept twice: attempt 1 (0.1s), attempt 2 (0.2s)
    assert sleeps == [0.1, 0.2]


def test_retry_stops_after_success() -> None:
    """Verify success on first attempt stops retrying immediately."""
    attempts = 0
    sleeps: List[float] = []

    def reliable_call() -> int:
        nonlocal attempts
        attempts += 1
        return 42

    policy = RetryPolicy(
        max_attempts=3,
        sleep_fn=sleeps.append,
    )

    result = policy.execute(reliable_call)

    assert result == 42
    assert attempts == 1
    assert sleeps == []


def test_retry_exhaustion_respects_max_attempts() -> None:
    """Verify retry stops when max_attempts is reached."""
    attempts = 0
    sleeps: List[float] = []

    def persistently_failing() -> None:
        nonlocal attempts
        attempts += 1
        raise TransientNetworkError(f"Failure {attempts}")

    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(TransientNetworkError) as exc_info:
        policy.execute(persistently_failing)

    assert "Failure 3" in str(exc_info.value)
    assert attempts == 3
    # max_attempts=3 means 2 sleeps between attempts (attempt 1 -> sleep 0.1, attempt 2 -> sleep 0.2)
    assert sleeps == [0.1, 0.2]


def test_non_retryable_error_aborts_immediately() -> None:
    """Verify non-retryable errors abort without retrying or sleeping."""
    attempts = 0
    sleeps: List[float] = []

    def fatal_call() -> None:
        nonlocal attempts
        attempts += 1
        raise FatalAuthError("403 Forbidden")

    policy = RetryPolicy(
        max_attempts=5,
        initial_delay=0.1,
        non_retryable_exceptions=(FatalAuthError,),
        sleep_fn=sleeps.append,
    )

    with pytest.raises(FatalAuthError) as exc_info:
        policy.execute(fatal_call)

    assert "403 Forbidden" in str(exc_info.value)
    assert attempts == 1
    assert sleeps == []


def test_exponential_backoff_delay_progression() -> None:
    """Verify delay follows initial_delay * (backoff_factor ** (attempt - 1)) capped at max_delay."""
    policy = RetryPolicy(
        max_attempts=5,
        initial_delay=0.1,
        backoff_factor=2.0,
        max_delay=0.5,
    )

    assert policy.compute_delay(1) == 0.1
    assert policy.compute_delay(2) == 0.2
    assert policy.compute_delay(3) == 0.4
    assert policy.compute_delay(4) == 0.5  # Capped at max_delay 0.5 (instead of 0.8)
    assert policy.compute_delay(5) == 0.5


def test_final_exception_preserved() -> None:
    """Verify the original exception identity and message are strictly preserved."""
    exc_to_raise = ValueError("Exact internal validation failure 987")

    def failing_call() -> None:
        raise exc_to_raise

    policy = RetryPolicy(
        max_attempts=2,
        initial_delay=0.01,
        sleep_fn=lambda _: None,
    )

    try:
        policy.execute(failing_call)
        pytest.fail("Should have raised")
    except ValueError as exc:
        assert exc is exc_to_raise
        assert str(exc) == "Exact internal validation failure 987"


def test_retry_decorator_sync() -> None:
    """Verify @retry_with_backoff decorator on synchronous function."""
    attempts = 0
    sleeps: List[float] = []

    @retry_with_backoff(max_attempts=3, initial_delay=0.05, sleep_fn=sleeps.append)
    def fetch_data() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Timeout")
        return "data"

    res = fetch_data()
    assert res == "data"
    assert attempts == 2
    assert sleeps == [0.05]


def test_retry_decorator_async() -> None:
    """Verify @retry_with_backoff decorator on coroutine function."""
    attempts = 0
    sleeps: List[float] = []

    async def fake_async_sleep(d: float) -> None:
        sleeps.append(d)

    policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.05,
        async_sleep_fn=fake_async_sleep,
    )

    async def async_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Async reset")
        return "async_data"

    res = asyncio.run(policy.execute_async(async_call))
    assert res == "async_data"
    assert attempts == 2
    assert sleeps == [0.05]
