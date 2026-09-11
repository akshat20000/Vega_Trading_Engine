"""
Retry and exponential backoff utility for broker REST / API operations.

Provides:
    - RetryPolicy: Configuration and executor for retry logic with exponential backoff.
    - retry_with_backoff: Decorator / callable wrapper for resilient operation execution.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """
    Configuration policy for retry execution with exponential backoff.

    Attributes:
        max_attempts: Maximum number of execution attempts (must be >= 1).
        initial_delay: Delay in seconds before the first retry (default 0.1s).
        backoff_factor: Multiplier applied to delay on consecutive failures (default 2.0).
        max_delay: Upper cap on retry delay in seconds (default 60.0s).
        retryable_exceptions: Exception classes eligible for retry.
        non_retryable_exceptions: Exception classes that immediately abort retry.
        sleep_fn: Synchronous sleep function (injectable for instant tests).
        async_sleep_fn: Async sleep function for coroutines.
    """

    max_attempts: int = 3
    initial_delay: float = 0.1
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
    non_retryable_exceptions: tuple[type[Exception], ...] = ()
    sleep_fn: Callable[[float], None] = time.sleep
    async_sleep_fn: Callable[[float], Any] = asyncio.sleep

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {self.initial_delay}")
        if self.backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be >= 1.0, got {self.backoff_factor}")

    def compute_delay(self, attempt: int) -> float:
        """
        Calculate backoff delay for the given failed attempt.

        Formula:
            attempt 1 failure -> initial_delay * (backoff_factor ** 0)
            attempt 2 failure -> initial_delay * (backoff_factor ** 1)
            capped at max_delay.
        """
        if attempt < 1:
            return 0.0
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def is_retryable(self, exc: Exception) -> bool:
        """Return True if the exception should trigger a retry attempt."""
        if self.non_retryable_exceptions and isinstance(exc, self.non_retryable_exceptions):
            return False
        return isinstance(exc, self.retryable_exceptions)

    def execute(
        self,
        fn: Callable[..., T],
        *args: Any,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute a synchronous callable with retry and exponential backoff.

        Args:
            fn: Target callable to execute.
            *args: Positional arguments for fn.
            on_retry: Optional callback(attempt, exception, delay) before sleeping.
            **kwargs: Keyword arguments for fn.

        Returns:
            Return value of successful fn call.

        Raises:
            The original exception if non-retryable or max attempts exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exception = exc

                # Check if exception is retryable and more attempts remain
                if not self.is_retryable(exc) or attempt >= self.max_attempts:
                    raise exc

                delay = self.compute_delay(attempt)
                if on_retry is not None:
                    on_retry(attempt, exc, delay)

                self.sleep_fn(delay)

        assert last_exception is not None
        raise last_exception

    async def execute_async(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an asynchronous callable or coroutine function with retry.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                res = fn(*args, **kwargs)
                if inspect.isawaitable(res):
                    res = await res
                return res
            except Exception as exc:
                last_exception = exc

                if not self.is_retryable(exc) or attempt >= self.max_attempts:
                    raise exc

                delay = self.compute_delay(attempt)
                if on_retry is not None:
                    on_retry(attempt, exc, delay)

                res_sleep = self.async_sleep_fn(delay)
                if inspect.isawaitable(res_sleep):
                    await res_sleep

        assert last_exception is not None
        raise last_exception


def retry_with_backoff(
    fn: Callable[..., T] | None = None,
    *,
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    non_retryable_exceptions: tuple[type[Exception], ...] = (),
    sleep_fn: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Any:
    """
    Decorator / wrapper to execute a function with exponential backoff retry.

    Usage as decorator:
        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def place_order():
            ...

    Usage as direct wrapper:
        result = retry_with_backoff(
            max_attempts=3, initial_delay=0.1
        )(place_order, arg1, arg2)
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
        sleep_fn=sleep_fn,
    )

    def decorator(target_fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(target_fn):
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await policy.execute_async(target_fn, *args, on_retry=on_retry, **kwargs)
            return async_wrapper

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return policy.execute(target_fn, *args, on_retry=on_retry, **kwargs)

        return sync_wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
