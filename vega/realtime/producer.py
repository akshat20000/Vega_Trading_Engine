"""
Asynchronous tick producer for real-time streaming pipeline.

Streams deterministic or pre-configured ticks into an asyncio.Queue with:
- Configurable inter-tick delay (in seconds, e.g. 0.0 for instant testing).
- Bounded queue backpressure handling (awaits queue.put).
- Graceful cooperative stop via stop().
- Deterministic operational metric tracking: ticks_produced.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

from vega.data.models import Tick
from vega.realtime.events import TickEvent


def generate_deterministic_ticks(
    symbol: str = "NIFTY50",
    start_time: datetime | None = None,
    prices: Sequence[float] | None = None,
    interval_seconds: int = 10,
    base_volume: float = 10.0,
) -> list[TickEvent]:
    """
    Generate a reproducible, deterministic sequence of ticks for testing.

    Args:
        symbol: Ticker symbol (default 'NIFTY50').
        start_time: Initial tick timestamp (default 2023-01-02 09:30:00).
        prices: Sequence of price levels. If None, uses standard reproducible sequence.
        interval_seconds: Seconds between consecutive ticks.
        base_volume: Volume for each generated tick.

    Returns:
        List of TickEvent objects.
    """
    if start_time is None:
        start_time = datetime(2023, 1, 2, 9, 30, 0)

    if prices is None:
        prices = [100.0, 101.0, 99.0, 102.0, 101.5, 103.0, 102.0, 104.0, 103.5, 105.0]

    events: list[TickEvent] = []
    current_time = start_time
    for price in prices:
        tick = Tick(timestamp=current_time, price=float(price), volume=float(base_volume))
        events.append(TickEvent(symbol=symbol, tick=tick, received_at=current_time))
        current_time += timedelta(seconds=interval_seconds)

    return events


class AsyncTickProducer:
    """
    Asynchronously streams market ticks into an asyncio.Queue.

    Demonstrates:
        - Non-blocking async event emission.
        - Backpressure compliance: uses `await queue.put()` which suspends
          execution when the queue capacity reaches maxsize.
        - Clean cooperative shutdown without dropping state.
        - Immediate termination when stopped while blocked on a full queue.
    """

    def __init__(
        self,
        queue: asyncio.Queue[TickEvent],
        ticks: Iterable[Tick | TickEvent] | None = None,
        delay: float = 0.0,
        symbol: str = "NIFTY50",
    ) -> None:
        """
        Initialize the async producer.

        Args:
            queue: Destination bounded asyncio.Queue.
            ticks: Optional sequence of Tick or TickEvent items.
                   If None, generates 10 deterministic ticks.
            delay: Delay in seconds between tick emissions (0.0 for max speed).
            symbol: Default symbol when converting raw Ticks.
        """
        self.queue = queue
        self.delay = max(0.0, float(delay))
        self.symbol = symbol

        # Normalize ticks into TickEvent objects
        if ticks is None:
            self._ticks = generate_deterministic_ticks(symbol=symbol)
        else:
            self._ticks = []
            for item in ticks:
                if isinstance(item, TickEvent):
                    self._ticks.append(item)
                elif isinstance(item, Tick):
                    self._ticks.append(
                        TickEvent(symbol=symbol, tick=item, received_at=item.timestamp)
                    )
                else:
                    raise TypeError(
                        f"Expected Tick or TickEvent, got {type(item).__name__}"
                    )

        self._stop_requested = False
        self._is_running = False
        self._task: asyncio.Task[Any] | None = None
        self.ticks_produced = 0

    @property
    def is_running(self) -> bool:
        """Return True if the producer loop is actively running."""
        return self._is_running

    def stop(self) -> None:
        """
        Signal the producer to stop producing ticks cooperatively.

        If the producer is currently blocked in `await queue.put()` due to
        full queue backpressure, cancels the active task to unblock immediately.
        """
        self._stop_requested = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def produce(self) -> int:
        """
        Run the production loop.

        Iterates through configured ticks, awaiting `queue.put(event)`
        and applying configurable inter-tick delay.

        Returns:
            Total number of ticks successfully placed onto the queue.
        """
        self._task = asyncio.current_task()
        self._is_running = True
        try:
            for event in self._ticks:
                if self._stop_requested:
                    break

                if self.delay > 0:
                    await asyncio.sleep(self.delay)

                if self._stop_requested:
                    break

                # Backpressure point: if queue is bounded and full,
                # this suspends until a consumer frees a slot.
                await self.queue.put(event)
                self.ticks_produced += 1

            return self.ticks_produced
        except asyncio.CancelledError:
            return self.ticks_produced
        finally:
            self._is_running = False
