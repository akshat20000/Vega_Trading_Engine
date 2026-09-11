"""
Real-time pipeline coordinator.

Coordinates:
    Simulated Feed / Ticks
             │
             ▼
      AsyncTickProducer
             │  await put()
             ▼
    asyncio.Queue(maxsize=100)
             │  await get()
             ▼
    AsyncTickConsumer(s)  [guarded by fine-grained asyncio.Lock on accumulator]
             │
             ▼
   RealtimeBarAccumulator
             │
             ▼
        OHLCV Bars

Enforces:
    - Bounded queue backpressure.
    - Deterministic metrics: ticks_produced == ticks_consumed.
    - Zero orphaned tasks and queue.unfinished_tasks == 0 on shutdown.
    - Proper queue draining during shutdown before consumer termination.
    - Non-deadlocking shutdown even when producer is blocked on a full queue.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

from vega.data.models import Bar, Tick
from vega.realtime.consumer import AsyncTickConsumer, RealtimeBarAccumulator
from vega.realtime.events import BarEvent, TickEvent
from vega.realtime.producer import AsyncTickProducer

T = TypeVar("T")

# Ensure asyncio.Queue exposes standard unfinished_tasks property
if not hasattr(asyncio.Queue, "unfinished_tasks"):
    asyncio.Queue.unfinished_tasks = property(lambda self: self._unfinished_tasks)  # type: ignore[attr-defined]


class VegaQueue(asyncio.Queue[T], Generic[T]):
    """
    Asyncio queue subclass explicitly exposing `unfinished_tasks` property.
    """

    @property
    def unfinished_tasks(self) -> int:
        """Number of unfinished tasks currently in the queue."""
        return self._unfinished_tasks


@dataclass(frozen=True)
class PipelineMetrics:
    """
    Deterministic operational metrics for the real-time pipeline.

    Invariants at completion:
        - ticks_produced == ticks_consumed
        - queue_unfinished_tasks == 0
    """

    ticks_produced: int
    ticks_consumed: int
    bars_emitted: int
    queue_unfinished_tasks: int


class RealtimePipeline:
    """
    Coordinates producers, consumers, and state accumulators in an async pipeline.
    """

    def __init__(
        self,
        producer: AsyncTickProducer,
        consumers: Sequence[AsyncTickConsumer],
        queue: asyncio.Queue[TickEvent],
        accumulator: RealtimeBarAccumulator | None = None,
        lock: asyncio.Lock | None = None,
    ) -> None:
        """
        Initialize the pipeline coordinator.

        Args:
            producer: The tick producer worker.
            consumers: One or more consumer workers.
            queue: The bounded communication queue.
            accumulator: Optional shared RealtimeBarAccumulator.
            lock: Optional asyncio.Lock for shared state protection across consumers.
        """
        self.producer = producer
        self.consumers = list(consumers)
        self.queue = queue
        self.accumulator = accumulator
        self.lock = lock

        self._producer_task: asyncio.Task[int] | None = None
        self._consumer_tasks: list[asyncio.Task[int]] = []
        self._is_running = False

    @classmethod
    def create(
        cls,
        ticks: Sequence[Tick | TickEvent] | None = None,
        queue_size: int = 100,
        num_consumers: int = 1,
        bucket_seconds: int = 60,
        symbol: str = "NIFTY50",
        producer_delay: float = 0.0,
        on_tick: Callable[[TickEvent], Any] | None = None,
        on_bar: Callable[[BarEvent], Any] | None = None,
    ) -> RealtimePipeline:
        """
        Convenience factory to construct a configured pipeline.

        Automatically wires a shared asyncio.Lock if num_consumers > 1.
        """
        queue: VegaQueue[TickEvent] = VegaQueue(maxsize=queue_size)
        shared_lock = asyncio.Lock() if num_consumers > 1 else None
        accumulator = RealtimeBarAccumulator(bucket_seconds=bucket_seconds, symbol=symbol)
        producer = AsyncTickProducer(
            queue=queue,
            ticks=ticks,
            delay=producer_delay,
            symbol=symbol,
        )

        consumers = [
            AsyncTickConsumer(
                queue=queue,
                accumulator=accumulator,
                lock=shared_lock,
                consumer_id=f"consumer-{i+1}",
                on_tick=on_tick,
                on_bar=on_bar,
            )
            for i in range(num_consumers)
        ]

        return cls(
            producer=producer,
            consumers=consumers,
            queue=queue,
            accumulator=accumulator,
            lock=shared_lock,
        )

    @property
    def is_running(self) -> bool:
        """True if the pipeline has active running tasks."""
        return self._is_running

    @property
    def ticks_produced(self) -> int:
        """Total ticks produced by the producer."""
        return self.producer.ticks_produced

    @property
    def ticks_consumed(self) -> int:
        """Total ticks consumed across all consumers."""
        return sum(c.ticks_consumed for c in self.consumers)

    @property
    def bars_emitted(self) -> int:
        """Total bars emitted by the accumulator."""
        return self.accumulator.bars_emitted if self.accumulator else 0

    @property
    def queue_unfinished_tasks(self) -> int:
        """Current number of unfinished tasks in the queue."""
        if hasattr(self.queue, "unfinished_tasks"):
            return self.queue.unfinished_tasks
        return getattr(self.queue, "_unfinished_tasks", 0)

    @property
    def bars(self) -> list[Bar]:
        """All completed bars emitted by the accumulator."""
        return self.accumulator.bars if self.accumulator else []

    def get_metrics(self) -> PipelineMetrics:
        """Return snapshot of pipeline operational metrics."""
        return PipelineMetrics(
            ticks_produced=self.ticks_produced,
            ticks_consumed=self.ticks_consumed,
            bars_emitted=self.bars_emitted,
            queue_unfinished_tasks=self.queue_unfinished_tasks,
        )

    async def start(self) -> None:
        """
        Start producer and consumer background tasks.
        """
        if self._is_running:
            return

        self._consumer_tasks = [
            asyncio.create_task(c.consume(), name=f"vega-{c.consumer_id}")
            for c in self.consumers
        ]
        self._producer_task = asyncio.create_task(
            self.producer.produce(), name="vega-producer"
        )
        self._is_running = True

    async def stop(self, timeout: float = 5.0, flush_accumulator: bool = True) -> None:
        """
        Execute cooperative graceful shutdown.

        Shutdown Sequence:
            1. Signal producer to stop & cancel producer task if it is blocked on
               a full bounded queue (`await queue.put`).
            2. Await producer task termination (guarantee producer has stopped).
            3. Await queue.join() — guarantee all ticks that were successfully
               enqueued are drained and processed by consumers.
            4. Signal consumers to stop (consumers exit when queue is empty).
            5. Await consumer tasks termination.
            6. Optionally flush the final active bar from the accumulator.
        """
        async def _do_stop() -> None:
            # Step 1 & 2: Stop producer and terminate producer task even if blocked on full queue
            self.producer.stop()
            if self._producer_task is not None and not self._producer_task.done():
                self._producer_task.cancel()
                try:
                    await self._producer_task
                except asyncio.CancelledError:
                    pass

            # Step 3: Wait for all successfully enqueued items to be processed
            await self.queue.join()

            # Step 4: Work is drained; now instruct consumers to terminate
            for consumer in self.consumers:
                consumer.stop()

            # Step 5: Wait for consumer tasks to finish
            if self._consumer_tasks:
                await asyncio.gather(*self._consumer_tasks, return_exceptions=True)

            # Step 6: Flush final bar from accumulator
            if flush_accumulator and self.accumulator is not None:
                if self.lock is not None:
                    async with self.lock:
                        self.accumulator.flush()
                else:
                    self.accumulator.flush()

        try:
            await asyncio.wait_for(_do_stop(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await self._cancel_all()
            raise
        finally:
            self._is_running = False

    async def run_until_complete(
        self, timeout: float = 10.0, flush_accumulator: bool = True
    ) -> PipelineMetrics:
        """
        Run the pipeline to completion for finite tick feeds.

        Starts workers, awaits producer completion, drains queue, stops consumers,
        flushes accumulator, and returns verified metrics.
        """
        async def _run() -> PipelineMetrics:
            await self.start()

            # Wait for producer to place all items
            if self._producer_task is not None:
                try:
                    await self._producer_task
                except asyncio.CancelledError:
                    pass

            # Wait for all queued items to be processed
            await self.queue.join()

            # Instruct consumers to stop
            for consumer in self.consumers:
                consumer.stop()

            # Wait for consumer tasks to exit cleanly
            if self._consumer_tasks:
                await asyncio.gather(*self._consumer_tasks, return_exceptions=True)

            # Flush final active bar
            if flush_accumulator and self.accumulator is not None:
                if self.lock is not None:
                    async with self.lock:
                        self.accumulator.flush()
                else:
                    self.accumulator.flush()

            self._is_running = False
            return self.get_metrics()

        try:
            return await asyncio.wait_for(_run(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await self._cancel_all()
            raise
        finally:
            self._is_running = False

    async def _cancel_all(self) -> None:
        """
        Cancel all running tasks in emergency situations and await their termination.

        Guarantees zero orphaned or pending tasks remain on the event loop.
        """
        tasks_to_cancel: list[asyncio.Task[Any]] = []
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
            tasks_to_cancel.append(self._producer_task)
        for t in self._consumer_tasks:
            if not t.done():
                t.cancel()
                tasks_to_cancel.append(t)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
