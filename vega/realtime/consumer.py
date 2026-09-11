"""
Real-time tick consumer and bar accumulator.

Components:
    - RealtimeBarAccumulator: Pure-Python tick-to-OHLCV aggregator without pandas.
    - AsyncTickConsumer: Async consumer processing events from an asyncio.Queue,
      with fine-grained asyncio.Lock protection over shared mutable accumulator state.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta
from typing import Any, Callable

from vega.data.models import Bar, Tick
from vega.realtime.events import BarEvent, TickEvent


def get_bucket_start(dt: datetime, bucket_seconds: int = 60) -> datetime:
    """
    Calculate the start timestamp of the aggregation bucket for a given datetime.

    Pure-Python implementation without pandas dependency.

    Args:
        dt: The tick timestamp.
        bucket_seconds: Width of each bar bucket in seconds (default 60 for 1-minute bars).

    Returns:
        The bucket start datetime (seconds/microseconds truncated to boundary).
    """
    if bucket_seconds == 60:
        return dt.replace(second=0, microsecond=0)
    if bucket_seconds < 60:
        sec = (dt.second // bucket_seconds) * bucket_seconds
        return dt.replace(second=sec, microsecond=0)
    if bucket_seconds % 60 == 0 and bucket_seconds < 3600:
        mins = bucket_seconds // 60
        m = (dt.minute // mins) * mins
        return dt.replace(minute=m, second=0, microsecond=0)
    if bucket_seconds % 3600 == 0 and bucket_seconds < 86400:
        hours = bucket_seconds // 3600
        h = (dt.hour // hours) * hours
        return dt.replace(hour=h, minute=0, second=0, microsecond=0)
    if bucket_seconds == 86400:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # General arbitrary seconds from midnight
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_midnight = int((dt - midnight).total_seconds())
    bucket_sec = (seconds_since_midnight // bucket_seconds) * bucket_seconds
    return midnight + timedelta(seconds=bucket_sec)


class RealtimeBarAccumulator:
    """
    Aggregates incoming ticks into OHLCV Bar objects in real-time.

    Pure-Python, zero external dependencies.
    When a tick arrives whose timestamp falls outside the current bucket:
      1. The current bucket is finalized and emitted as a completed Bar.
      2. A new bucket is initialized with the current tick.

    Calling `flush()` emits the final in-progress bar upon shutdown.
    """

    def __init__(self, bucket_seconds: int = 60, symbol: str = "NIFTY50") -> None:
        """
        Initialize the real-time accumulator.

        Args:
            bucket_seconds: Bar bucket duration in seconds (default 60 for 1-min).
            symbol: Default symbol for emitted bars.
        """
        if bucket_seconds <= 0:
            raise ValueError(f"bucket_seconds must be > 0, got {bucket_seconds}")

        self.bucket_seconds = bucket_seconds
        self.symbol = symbol

        # Internal aggregation state
        self._current_bucket_start: datetime | None = None
        self._open: float = 0.0
        self._high: float = 0.0
        self._low: float = 0.0
        self._close: float = 0.0
        self._volume: float = 0.0
        self._first_tick_timestamp: datetime | None = None
        self._last_tick_timestamp: datetime | None = None
        self._ticks_in_bucket: int = 0

        # Accumulated history and operational metrics
        self._bars: list[Bar] = []
        self._bars_emitted: int = 0
        self._total_ticks_processed: int = 0
        self._total_volume_processed: float = 0.0
        self._late_ticks_dropped: int = 0

    @property
    def bars(self) -> list[Bar]:
        """Return a copy of all completed Bar objects emitted so far."""
        return list(self._bars)

    @property
    def bars_emitted(self) -> int:
        """Total number of completed bars emitted."""
        return self._bars_emitted

    @property
    def total_ticks_processed(self) -> int:
        """Total number of ticks processed into the accumulator."""
        return self._total_ticks_processed

    @property
    def total_volume_processed(self) -> float:
        """Total cumulative volume processed."""
        return self._total_volume_processed

    @property
    def late_ticks_dropped(self) -> int:
        """Number of late ticks dropped for already-finalized buckets."""
        return self._late_ticks_dropped

    @property
    def current_bucket_start(self) -> datetime | None:
        """Timestamp of the current active bucket, or None if idle."""
        return self._current_bucket_start

    @property
    def current_bar(self) -> Bar | None:
        """Return a snapshot of the currently active in-progress bar, if any."""
        if self._current_bucket_start is None:
            return None
        return Bar(
            timestamp=self._current_bucket_start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
        )

    def add_tick(self, tick: Tick) -> Bar | None:
        """
        Add a single tick to the accumulator.

        If the tick belongs to a new time bucket, the previous bucket is completed,
        recorded, and returned as a Bar. Otherwise, the current bucket is updated
        and None is returned.

        Args:
            tick: The incoming Tick object.

        Returns:
            Completed Bar if a bucket boundary was crossed, otherwise None.
        """
        bucket_start = get_bucket_start(tick.timestamp, self.bucket_seconds)
        emitted_bar: Bar | None = None

        if self._current_bucket_start is None:
            # First tick: initialize active bucket
            self._current_bucket_start = bucket_start
            self._open = tick.price
            self._high = tick.price
            self._low = tick.price
            self._close = tick.price
            self._volume = tick.volume
            self._first_tick_timestamp = tick.timestamp
            self._last_tick_timestamp = tick.timestamp
            self._ticks_in_bucket = 1
        elif bucket_start == self._current_bucket_start:
            # Same bucket: update OHLCV
            if tick.price > self._high:
                self._high = tick.price
            if tick.price < self._low:
                self._low = tick.price
            if self._first_tick_timestamp is None or tick.timestamp <= self._first_tick_timestamp:
                self._open = tick.price
                self._first_tick_timestamp = tick.timestamp
            if self._last_tick_timestamp is None or tick.timestamp >= self._last_tick_timestamp:
                self._close = tick.price
                self._last_tick_timestamp = tick.timestamp
            self._volume += tick.volume
            self._ticks_in_bucket += 1
        elif bucket_start > self._current_bucket_start:
            # New bucket: close existing bucket
            emitted_bar = Bar(
                timestamp=self._current_bucket_start,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                volume=self._volume,
            )
            self._bars.append(emitted_bar)
            self._bars_emitted += 1

            # Start new bucket with current tick
            self._current_bucket_start = bucket_start
            self._open = tick.price
            self._high = tick.price
            self._low = tick.price
            self._close = tick.price
            self._volume = tick.volume
            self._first_tick_timestamp = tick.timestamp
            self._last_tick_timestamp = tick.timestamp
            self._ticks_in_bucket = 1
        else:
            # Out-of-order tick for already closed bucket:
            # In streaming ingestion, ignore late ticks to preserve time monotonicity.
            self._late_ticks_dropped += 1

        self._total_ticks_processed += 1
        self._total_volume_processed += tick.volume
        return emitted_bar

    def flush(self) -> Bar | None:
        """
        Flush and finalize the currently active in-progress bar.

        Called during shutdown to avoid dropping ticks belonging to the final bucket.

        Returns:
            Finalized Bar, or None if no bucket was in progress.
        """
        if self._current_bucket_start is None:
            return None

        bar = Bar(
            timestamp=self._current_bucket_start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
        )
        self._bars.append(bar)
        self._bars_emitted += 1

        # Reset active bucket
        self._current_bucket_start = None
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0.0
        self._first_tick_timestamp = None
        self._last_tick_timestamp = None
        self._ticks_in_bucket = 0

        return bar


class AsyncTickConsumer:
    """
    Asynchronous consumer for market tick events from an asyncio.Queue.

    Concurrency Design:
        - Only the small critical section updating shared state (`RealtimeBarAccumulator`)
          is guarded by `asyncio.Lock`.
        - Queue `get()`, deserialization, notification callbacks, and `task_done()`
          run outside the lock for high throughput.
        - Shutdown semantics: Even if `stop()` is called, the consumer drains any remaining
          ticks in the queue before terminating (`not stop_requested or not queue.empty()`).
    """

    def __init__(
        self,
        queue: asyncio.Queue[TickEvent],
        accumulator: RealtimeBarAccumulator | None = None,
        lock: asyncio.Lock | None = None,
        consumer_id: str = "consumer-1",
        on_tick: Callable[[TickEvent], Any] | None = None,
        on_bar: Callable[[BarEvent], Any] | None = None,
    ) -> None:
        """
        Initialize the async consumer.

        Args:
            queue: Source bounded queue.
            accumulator: Shared or dedicated RealtimeBarAccumulator.
            lock: Optional asyncio.Lock for protecting shared accumulator mutations.
            consumer_id: Human-readable identifier.
            on_tick: Optional async or sync callback invoked for every tick.
            on_bar: Optional async or sync callback invoked when a bar completes.
        """
        self.queue = queue
        self.accumulator = accumulator
        self._lock = lock
        self.consumer_id = consumer_id
        self.on_tick = on_tick
        self.on_bar = on_bar

        self._stop_requested = False
        self._is_running = False
        self.ticks_consumed = 0
        self.bars_emitted = 0

    @property
    def is_running(self) -> bool:
        """Return True if the consumer loop is actively running."""
        return self._is_running

    def stop(self) -> None:
        """
        Signal the consumer to stop cooperatively.

        Note: The consumer will still process all remaining ticks in the queue
        before terminating to ensure zero work is dropped.
        """
        self._stop_requested = True

    async def _handle_callback(self, cb: Callable[..., Any], *args: Any) -> None:
        """Execute a callback, awaiting if it returns an awaitable."""
        res = cb(*args)
        if inspect.isawaitable(res):
            await res

    async def _process_tick(self, event: TickEvent) -> Bar | None:
        """
        Process a single tick event.

        Only the mutation of the shared accumulator is protected by `self._lock`.
        """
        # 1. Pre-processing / tick callback (outside lock)
        if self.on_tick is not None:
            await self._handle_callback(self.on_tick, event)

        # 2. Critical section: mutate shared accumulator under lock
        bar: Bar | None = None
        if self.accumulator is not None:
            if self._lock is not None:
                async with self._lock:
                    bar = self.accumulator.add_tick(event.tick)
            else:
                bar = self.accumulator.add_tick(event.tick)

        # 3. Post-processing / bar emission callback (outside lock)
        if bar is not None:
            self.bars_emitted += 1
            if self.on_bar is not None:
                bar_event = BarEvent(symbol=event.symbol, bar=bar)
                await self._handle_callback(self.on_bar, bar_event)

        self.ticks_consumed += 1
        return bar

    async def consume(self) -> int:
        """
        Run the consumer loop.

        Pulls items from `queue`, executes `_process_tick()`, and ensures
        `queue.task_done()` is always invoked in a `finally` block.

        Terminates ONLY when `stop()` was called AND `queue.empty()` is True.

        Returns:
            Total ticks consumed by this worker.
        """
        self._is_running = True
        try:
            while True:
                # Cooperative shutdown check:
                # Stop ONLY if shutdown was requested AND all queued ticks are drained.
                if self._stop_requested and self.queue.empty():
                    break

                try:
                    # Short timeout allows checking the shutdown condition periodically
                    # even if no new ticks arrive on an empty queue.
                    event = await asyncio.wait_for(self.queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if self._stop_requested and self.queue.empty():
                        break
                    continue
                except asyncio.CancelledError:
                    # Task cancelled externally
                    break

                try:
                    await self._process_tick(event)
                finally:
                    self.queue.task_done()

            return self.ticks_consumed
        except asyncio.CancelledError:
            return self.ticks_consumed
        finally:
            self._is_running = False
