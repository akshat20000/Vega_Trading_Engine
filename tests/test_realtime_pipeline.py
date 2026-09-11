"""
Tests for Phase 9: Real-time Concurrency Pipeline.

Verifies:
    1. RealtimeBarAccumulator pure-Python OHLCV generation without pandas.
    2. Exact user-specified accumulator test fixture (09:30:10 -> 09:31:05).
    3. Deterministic tick generator reproducibility.
    4. AsyncTickProducer backpressure on bounded asyncio.Queue.
    5. AsyncTickConsumer and FIFO ordering.
    6. Consumer task_done() contract and queue.join() unblocking.
    7. Multi-consumer concurrency with fine-grained asyncio.Lock.
    8. Shutdown edge case: queue draining completely when shutdown requested.
    9. RealtimePipeline lifecycle, graceful shutdown, and operational invariants:
       - ticks_produced == ticks_consumed
       - queue.unfinished_tasks == 0
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List

import pytest

from vega.data.models import Bar, Tick
from vega.realtime.consumer import (
    AsyncTickConsumer,
    RealtimeBarAccumulator,
    get_bucket_start,
)
from vega.realtime.events import BarEvent, TickEvent
from vega.realtime.pipeline import PipelineMetrics, RealtimePipeline
from vega.realtime.producer import (
    AsyncTickProducer,
    generate_deterministic_ticks,
)


# ─── 1. Accumulator Unit Tests ────────────────────────────────────────────────


def test_realtime_bar_accumulator_user_fixture() -> None:
    """
    Verify exact test fixture requested by user:
        09:30:10 100
        09:30:20 103
        09:30:40  99
        09:31:05 101

    Expected first bar:
        open   = 100
        high   = 103
        low    = 99
        close  = 99

    Then second bucket begins at 101.
    Demonstrates pure-Python tick -> OHLCV conversion without pandas.
    """
    accum = RealtimeBarAccumulator(bucket_seconds=60, symbol="NIFTY50")

    # First three ticks in bucket 09:30:00
    b1 = accum.add_tick(Tick(datetime(2023, 1, 2, 9, 30, 10), 100.0, 10.0))
    assert b1 is None  # Bucket still open

    b2 = accum.add_tick(Tick(datetime(2023, 1, 2, 9, 30, 20), 103.0, 15.0))
    assert b2 is None

    b3 = accum.add_tick(Tick(datetime(2023, 1, 2, 9, 30, 40), 99.0, 5.0))
    assert b3 is None

    # Fourth tick crosses boundary into bucket 09:31:00
    first_bar = accum.add_tick(Tick(datetime(2023, 1, 2, 9, 31, 5), 101.0, 20.0))

    # Assert first bar is emitted and exact
    assert first_bar is not None
    assert first_bar.timestamp == datetime(2023, 1, 2, 9, 30, 0)
    assert first_bar.open == 100.0
    assert first_bar.high == 103.0
    assert first_bar.low == 99.0
    assert first_bar.close == 99.0
    assert first_bar.volume == 30.0  # 10 + 15 + 5

    # Verify second bucket state in accumulator
    assert accum.current_bucket_start == datetime(2023, 1, 2, 9, 31, 0)
    cur = accum.current_bar
    assert cur is not None
    assert cur.timestamp == datetime(2023, 1, 2, 9, 31, 0)
    assert cur.open == 101.0
    assert cur.high == 101.0
    assert cur.low == 101.0
    assert cur.close == 101.0
    assert cur.volume == 20.0

    # Flush final bar
    second_bar = accum.flush()
    assert second_bar is not None
    assert second_bar.timestamp == datetime(2023, 1, 2, 9, 31, 0)
    assert second_bar.open == 101.0
    assert second_bar.close == 101.0
    assert second_bar.volume == 20.0

    # Operational metrics on accumulator
    assert accum.bars_emitted == 2
    assert accum.total_ticks_processed == 4
    assert accum.total_volume_processed == 50.0
    assert len(accum.bars) == 2


def test_accumulator_empty_and_flush() -> None:
    """Verify flush on empty accumulator returns None."""
    accum = RealtimeBarAccumulator(bucket_seconds=60)
    assert accum.flush() is None
    assert accum.bars_emitted == 0
    assert len(accum.bars) == 0


def test_get_bucket_start_boundaries() -> None:
    """Verify bucket boundary calculations across 1m, 5m, and 15s."""
    dt = datetime(2023, 1, 2, 9, 37, 45)
    # 60s
    assert get_bucket_start(dt, 60) == datetime(2023, 1, 2, 9, 37, 0)
    # 300s (5min) -> 35
    assert get_bucket_start(dt, 300) == datetime(2023, 1, 2, 9, 35, 0)
    # 15s -> 45
    assert get_bucket_start(dt, 15) == datetime(2023, 1, 2, 9, 37, 45)


# ─── 2. Deterministic Generator Tests ─────────────────────────────────────────


def test_generate_deterministic_ticks() -> None:
    """Verify deterministic tick generator produces reproducible data."""
    ticks1 = generate_deterministic_ticks(
        symbol="NIFTY50",
        start_time=datetime(2023, 1, 2, 9, 30, 0),
        prices=[100.0, 101.0, 99.0],
        interval_seconds=10,
        base_volume=10.0,
    )
    ticks2 = generate_deterministic_ticks(
        symbol="NIFTY50",
        start_time=datetime(2023, 1, 2, 9, 30, 0),
        prices=[100.0, 101.0, 99.0],
        interval_seconds=10,
        base_volume=10.0,
    )

    assert len(ticks1) == 3
    assert [t.tick.price for t in ticks1] == [100.0, 101.0, 99.0]
    assert [t.tick.timestamp for t in ticks1] == [
        datetime(2023, 1, 2, 9, 30, 0),
        datetime(2023, 1, 2, 9, 30, 10),
        datetime(2023, 1, 2, 9, 30, 20),
    ]
    # Identical deterministic output
    assert [t.tick.price for t in ticks1] == [t.tick.price for t in ticks2]


# ─── 3. Producer & Consumer Concurrency Tests ─────────────────────────────────


def test_producer_generates_expected_ticks() -> None:
    """Verify AsyncTickProducer puts all items onto the queue."""
    async def _test():
        queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=10)
        ticks = generate_deterministic_ticks(prices=[100.0, 101.0, 102.0])
        producer = AsyncTickProducer(queue=queue, ticks=ticks, delay=0.0)

        count = await producer.produce()
        assert count == 3
        assert producer.ticks_produced == 3
        assert queue.qsize() == 3

    asyncio.run(_test())


def test_consumer_receives_all_ticks_and_fifo() -> None:
    """Verify AsyncTickConsumer preserves exact FIFO ordering."""
    async def _test():
        queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=10)
        ticks = generate_deterministic_ticks(prices=[100.0, 105.0, 95.0, 102.0])
        for t in ticks:
            await queue.put(t)

        received_prices: list[float] = []

        def on_tick(event: TickEvent):
            received_prices.append(event.tick.price)

        consumer = AsyncTickConsumer(queue=queue, on_tick=on_tick)
        consumer.stop()  # Stop requested, but queue has 4 items! Should drain all 4.

        consumed = await consumer.consume()

        assert consumed == 4
        assert consumer.ticks_consumed == 4
        assert received_prices == [100.0, 105.0, 95.0, 102.0]
        assert queue.empty()
        assert queue.unfinished_tasks == 0

    asyncio.run(_test())


def test_bounded_queue_backpressure() -> None:
    """
    Verify backpressure: bounded queue with maxsize=2 pauses producer
    until consumer frees space.
    """
    async def _test():
        queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=2)
        ticks = generate_deterministic_ticks(prices=[100.0, 101.0, 102.0, 103.0, 104.0])
        producer = AsyncTickProducer(queue=queue, ticks=ticks, delay=0.0)

        # Producer starts in background
        prod_task = asyncio.create_task(producer.produce())

        # Yield to allow producer to fill queue to maxsize
        await asyncio.sleep(0.01)

        # Queue must be full at maxsize 2
        assert queue.full()
        assert queue.qsize() == 2
        # Producer has placed 2 items and is currently blocked on the 3rd
        assert producer.ticks_produced == 2
        assert not prod_task.done()

        # Consumer drains 1 item
        item = await queue.get()
        queue.task_done()
        assert item.tick.price == 100.0

        # Yield to let producer resume and fill the freed slot
        await asyncio.sleep(0.01)
        assert queue.full()
        assert producer.ticks_produced == 3

        # Drain the rest
        while not queue.empty():
            await queue.get()
            queue.task_done()

        # Producer finishes the remaining items
        await asyncio.wait_for(prod_task, timeout=1.0)
        assert producer.ticks_produced == 5

        # Drain leftover from last put
        while not queue.empty():
            await queue.get()
            queue.task_done()

        assert queue.unfinished_tasks == 0

    asyncio.run(_test())


def test_consumer_calls_task_done_and_queue_join() -> None:
    """Verify queue.join() unblocks as consumer calls task_done()."""
    async def _test():
        queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=10)
        ticks = generate_deterministic_ticks(prices=[100.0, 101.0, 102.0])
        for t in ticks:
            await queue.put(t)

        consumer = AsyncTickConsumer(queue=queue)
        consumer_task = asyncio.create_task(consumer.consume())

        # queue.join() must resolve without deadlock
        await asyncio.wait_for(queue.join(), timeout=1.0)
        assert queue.unfinished_tasks == 0

        # Now signal consumer to exit
        consumer.stop()
        await asyncio.wait_for(consumer_task, timeout=1.0)
        assert consumer.ticks_consumed == 3

    asyncio.run(_test())


def test_shutdown_edge_case_queue_drains_completely() -> None:
    """
    Interview-level shutdown edge case explicitly requested by user:
        producer finishes
            +
        queue still contains ticks
            +
        shutdown requested
            ↓
        all queued ticks are consumed

    Guarantees consumer does NOT terminate prematurely on shutdown_requested
    if the queue still holds work.
    """
    async def _test():
        queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=20)
        ticks = generate_deterministic_ticks(
            prices=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        )

        # Producer finishes putting all 6 ticks
        producer = AsyncTickProducer(queue=queue, ticks=ticks)
        await producer.produce()
        assert producer.ticks_produced == 6
        assert queue.qsize() == 6
        assert queue.unfinished_tasks == 6

        # Consumer is instantiated
        consumer = AsyncTickConsumer(queue=queue)

        # SHUTDOWN REQUESTED BEFORE CONSUMER EVEN RUNS
        consumer.stop()
        assert consumer._stop_requested is True

        # Run consumer. Even though stop was requested, it MUST drain the queue!
        consumed = await asyncio.wait_for(consumer.consume(), timeout=1.0)

        # Operational invariants:
        assert consumed == 6
        assert consumer.ticks_consumed == 6
        assert queue.qsize() == 0
        assert queue.unfinished_tasks == 0
        assert not consumer.is_running

    asyncio.run(_test())


# ─── 4. Multi-Consumer & Lock Verification ───────────────────────────────────


def test_two_consumers_with_lock_no_corruption() -> None:
    """
    Verify two concurrent consumers updating a shared RealtimeBarAccumulator
    under an asyncio.Lock do not corrupt state or drop ticks.
    """
    async def _test():
        queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=50)
        lock = asyncio.Lock()
        accum = RealtimeBarAccumulator(bucket_seconds=60)

        # 30 deterministic ticks across 2 minutes
        prices = [100.0 + (i % 5) for i in range(30)]
        ticks = generate_deterministic_ticks(
            start_time=datetime(2023, 1, 2, 9, 30, 0),
            prices=prices,
            interval_seconds=4,  # 30 * 4s = 120s (2 full 1-min bars)
            base_volume=10.0,
        )

        for t in ticks:
            await queue.put(t)

        c1 = AsyncTickConsumer(queue=queue, accumulator=accum, lock=lock, consumer_id="c1")
        c2 = AsyncTickConsumer(queue=queue, accumulator=accum, lock=lock, consumer_id="c2")

        # Start both consumers concurrently
        t1 = asyncio.create_task(c1.consume())
        t2 = asyncio.create_task(c2.consume())

        # Wait for all 30 ticks to be processed
        await asyncio.wait_for(queue.join(), timeout=2.0)

        # Stop consumers
        c1.stop()
        c2.stop()
        await asyncio.gather(t1, t2)

        # Invariants:
        assert queue.unfinished_tasks == 0
        assert c1.ticks_consumed + c2.ticks_consumed == 30
        assert accum.total_ticks_processed == 30
        assert accum.total_volume_processed == 300.0

        # Flush final bar
        accum.flush()
        assert accum.bars_emitted >= 2

    asyncio.run(_test())


# ─── 5. End-to-End RealtimePipeline Tests ─────────────────────────────────────


def test_pipeline_run_until_complete_metrics_invariant() -> None:
    """
    End-to-end pipeline execution verifying key operational invariants:
        ticks_produced == ticks_consumed
        queue.unfinished_tasks == 0
    """
    async def _test():
        ticks = [
            Tick(datetime(2023, 1, 2, 9, 30, 10), 100.0, 10.0),
            Tick(datetime(2023, 1, 2, 9, 30, 20), 103.0, 10.0),
            Tick(datetime(2023, 1, 2, 9, 30, 40), 99.0, 10.0),
            Tick(datetime(2023, 1, 2, 9, 31, 5), 101.0, 10.0),
        ]

        pipeline = RealtimePipeline.create(
            ticks=ticks,
            queue_size=10,
            num_consumers=2,
            bucket_seconds=60,
            producer_delay=0.001,
        )

        metrics: PipelineMetrics = await pipeline.run_until_complete(timeout=2.0)

        # Operational invariants:
        assert metrics.ticks_produced == 4
        assert metrics.ticks_consumed == 4
        assert metrics.ticks_produced == metrics.ticks_consumed
        assert metrics.queue_unfinished_tasks == 0

        # Bar outputs:
        bars = pipeline.bars
        assert len(bars) == 2
        # First bar: 09:30:00 [100, 103, 99, 99]
        assert bars[0].timestamp == datetime(2023, 1, 2, 9, 30, 0)
        assert bars[0].open == 100.0
        assert bars[0].high == 103.0
        assert bars[0].low == 99.0
        assert bars[0].close == 99.0
        assert bars[0].volume == 30.0

        # Second bar: 09:31:00 [101, 101, 101, 101]
        assert bars[1].timestamp == datetime(2023, 1, 2, 9, 31, 0)
        assert bars[1].open == 101.0
        assert bars[1].close == 101.0
        assert bars[1].volume == 10.0

    asyncio.run(_test())


def test_pipeline_graceful_stop_during_flight() -> None:
    """
    Verify pipeline stop() gracefully drains queued work and leaves no orphaned tasks
    when stopped while producer is still active.
    """
    async def _test():
        # 50 ticks with 10ms delay between ticks
        prices = [100.0 + i for i in range(50)]
        ticks = generate_deterministic_ticks(prices=prices)

        pipeline = RealtimePipeline.create(
            ticks=ticks,
            queue_size=20,
            num_consumers=2,
            producer_delay=0.01,
        )

        await pipeline.start()

        # Let it produce and consume a few ticks
        await asyncio.sleep(0.05)

        # Trigger graceful stop mid-flight
        await pipeline.stop(timeout=2.0)

        # Invariants:
        assert not pipeline.is_running
        assert pipeline.queue_unfinished_tasks == 0
        # All produced ticks were consumed (no dropped ticks)
        assert pipeline.ticks_produced == pipeline.ticks_consumed
        assert pipeline.ticks_produced > 0

    asyncio.run(_test())


def test_pipeline_callbacks() -> None:
    """Verify on_tick and on_bar callbacks are invoked during execution."""
    async def _test():
        tick_events: List[TickEvent] = []
        bar_events: List[BarEvent] = []

        async def async_on_tick(te: TickEvent) -> None:
            tick_events.append(te)

        def sync_on_bar(be: BarEvent) -> None:
            bar_events.append(be)

        ticks = [
            Tick(datetime(2023, 1, 2, 9, 30, 0), 100.0, 10.0),
            Tick(datetime(2023, 1, 2, 9, 31, 0), 105.0, 10.0),
        ]

        pipeline = RealtimePipeline.create(
            ticks=ticks,
            num_consumers=1,
            bucket_seconds=60,
            on_tick=async_on_tick,
            on_bar=sync_on_bar,
        )

        await pipeline.run_until_complete(timeout=1.0)

        assert len(tick_events) == 2
        # Bar emitted when 9:31:00 tick arrived, plus flush at completion
        assert len(bar_events) >= 1
        assert bar_events[0].bar.open == 100.0

    asyncio.run(_test())


def test_backpressure_heavy_stress() -> None:
    """
    Stress test backpressure: 100 ticks pushed through tiny queue of maxsize=5
    with 2 concurrent consumers.
    """
    async def _test():
        prices = [100.0 + (i % 7) for i in range(100)]
        ticks = generate_deterministic_ticks(
            prices=prices, interval_seconds=1, base_volume=5.0
        )

        pipeline = RealtimePipeline.create(
            ticks=ticks,
            queue_size=5,
            num_consumers=2,
            bucket_seconds=10,
            producer_delay=0.0,
        )

        metrics = await pipeline.run_until_complete(timeout=5.0)

        assert metrics.ticks_produced == 100
        assert metrics.ticks_consumed == 100
        assert metrics.ticks_produced == metrics.ticks_consumed
        assert metrics.queue_unfinished_tasks == 0
        assert pipeline.queue.unfinished_tasks == 0
        assert len(pipeline.bars) >= 9

    asyncio.run(_test())


def test_stop_while_producer_blocked_on_full_queue() -> None:
    """
    Verify producer can be cancelled cleanly while blocked on a full bounded queue
    without deadlocking or corrupting task state.
    """
    async def _test():
        queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=2)
        ticks = generate_deterministic_ticks(prices=[100.0, 101.0, 102.0, 103.0, 104.0])
        producer = AsyncTickProducer(queue=queue, ticks=ticks, delay=0.0)

        # Start producer task without consumers
        prod_task = asyncio.create_task(producer.produce())

        # Give producer time to fill queue (capacity 2) and block on 3rd put
        await asyncio.sleep(0.01)
        assert queue.full()
        assert producer.ticks_produced == 2
        assert not prod_task.done()

        # Cancel producer while blocked on full queue
        prod_task.cancel()
        await asyncio.gather(prod_task, return_exceptions=True)
        assert not producer.is_running
        assert producer.ticks_produced == 2

        # Now drain the 2 items that were queued
        consumer = AsyncTickConsumer(queue=queue)
        consumer.stop()
        consumed = await consumer.consume()
        assert consumed == 2
        assert queue.unfinished_tasks == 0

    asyncio.run(_test())


def test_cancellation_leaves_no_orphaned_tasks() -> None:
    """Verify emergency cancellation cancels and awaits all tasks without leaving running orphans."""
    async def _test():
        pipeline = RealtimePipeline.create(
            ticks=generate_deterministic_ticks(prices=[100.0] * 20),
            queue_size=5,
            num_consumers=2,
            producer_delay=0.1,  # slow producer
        )

        await pipeline.start()
        await asyncio.sleep(0.02)
        assert pipeline.is_running

        # Emergency cancel all and await
        await pipeline._cancel_all()

        assert pipeline._producer_task.done()
        for t in pipeline._consumer_tasks:
            assert t.done()

        # Verify no pending tasks in event loop
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert len(pending) == 0

    asyncio.run(_test())


def test_pipeline_shutdown_while_producer_blocked_on_full_queue() -> None:
    """
    Test shutdown when producer is blocked in `await queue.put()` on a full queue:

        queue size = 1
               ↓
        producer has more ticks
               ↓
        consumer is deliberately blocked/slow
               ↓
        producer becomes blocked on queue.put()
               ↓
        pipeline.stop()
               ↓
        producer terminates
               ↓
        queued work drains
               ↓
        consumer terminates
               ↓
        no pending tasks
    """
    async def _test():
        # Producer has 5 ticks, queue capacity = 1
        ticks = generate_deterministic_ticks(
            prices=[100.0, 101.0, 102.0, 103.0, 104.0],
            interval_seconds=1,
            base_volume=10.0,
        )

        unblock_consumer = asyncio.Event()

        # Consumer will block on the first tick until unblock_consumer is set
        async def slow_on_tick(te: TickEvent) -> None:
            await unblock_consumer.wait()

        pipeline = RealtimePipeline.create(
            ticks=ticks,
            queue_size=1,
            num_consumers=1,
            producer_delay=0.0,
            on_tick=slow_on_tick,
        )

        # 1. Start pipeline
        await pipeline.start()

        # Give tasks time to execute:
        # Consumer pulls tick 0 and blocks on unblock_consumer.
        # Producer puts tick 1 into queue (qsize=1, full).
        # Producer attempts to put tick 2 -> producer becomes blocked in queue.put()!
        await asyncio.sleep(0.05)

        # Verify producer state: placed 2 ticks, blocked on 3rd
        assert pipeline.ticks_produced == 2
        assert pipeline.queue.full()
        assert pipeline.queue.qsize() == 1
        assert not pipeline._producer_task.done()

        # 2. Call pipeline.stop() in background task
        stop_task = asyncio.create_task(pipeline.stop(timeout=2.0))

        # Give stop() time to execute Step 1: signal & cancel producer
        await asyncio.sleep(0.02)

        # Producer must terminate immediately!
        assert pipeline._producer_task.done()
        assert not pipeline.producer.is_running
        # Ticks placed on queue remains 2 (tick 2 was not enqueued)
        assert pipeline.ticks_produced == 2

        # 3. Consumer is still working/waiting; now unblock consumer so queued work can drain
        unblock_consumer.set()

        # 4. Wait for stop() to finish draining and stopping consumers
        await asyncio.wait_for(stop_task, timeout=2.0)

        # 5. Verify all queued work drained and consumers terminated
        assert not pipeline.is_running
        assert all(t.done() for t in pipeline._consumer_tasks)
        assert pipeline.queue_unfinished_tasks == 0
        assert pipeline.queue.unfinished_tasks == 0
        assert pipeline.ticks_consumed == 2
        assert pipeline.ticks_produced == pipeline.ticks_consumed

        # 6. Verify no pending tasks remain
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert len(pending) == 0

    asyncio.run(_test())


def test_pipeline_stop_with_slow_consumer_drains_queued_work() -> None:
    """
    Verify pipeline.stop() with naturally slow consumer:
    producer gets backpressured on queue_size=1, stop() unblocks producer,
    drains queued work, and leaves zero orphaned tasks.
    """
    async def _test():
        ticks = generate_deterministic_ticks(
            prices=[100.0, 101.0, 102.0, 103.0],
            interval_seconds=1,
            base_volume=10.0,
        )

        async def slow_on_tick(te: TickEvent) -> None:
            await asyncio.sleep(0.02)

        pipeline = RealtimePipeline.create(
            ticks=ticks,
            queue_size=1,
            num_consumers=1,
            producer_delay=0.0,
            on_tick=slow_on_tick,
        )

        await pipeline.start()
        await asyncio.sleep(0.01)

        # Producer is blocked on queue.put(); call stop()
        await pipeline.stop(timeout=2.0)

        assert pipeline._producer_task.done()
        assert all(t.done() for t in pipeline._consumer_tasks)
        assert pipeline.queue_unfinished_tasks == 0
        assert pipeline.ticks_produced == pipeline.ticks_consumed
        assert pipeline.ticks_produced >= 1

        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert len(pending) == 0

    asyncio.run(_test())


def test_accumulator_out_of_order_late_tick() -> None:
    """Verify late ticks arriving after bucket finalization are counted and dropped."""
    accum = RealtimeBarAccumulator(bucket_seconds=60)

    # First bucket: 09:30
    accum.add_tick(Tick(datetime(2023, 1, 2, 9, 30, 10), 100.0, 10.0))

    # Second bucket: 09:31 (closes 09:30)
    bar1 = accum.add_tick(Tick(datetime(2023, 1, 2, 9, 31, 5), 105.0, 10.0))
    assert bar1 is not None
    assert bar1.timestamp == datetime(2023, 1, 2, 9, 30, 0)

    # Late tick belonging to already closed bucket 09:30 arrives
    late_bar = accum.add_tick(Tick(datetime(2023, 1, 2, 9, 30, 45), 98.0, 5.0))
    assert late_bar is None
    assert accum.late_ticks_dropped == 1

    # Current bucket still intact
    assert accum.current_bucket_start == datetime(2023, 1, 2, 9, 31, 0)
    cur = accum.current_bar
    assert cur is not None
    assert cur.open == 105.0


