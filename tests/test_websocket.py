"""
Tests for Phase 10D: WebSocket Connection Lifecycle and Market Data Reliability.

Verifies:
    Lifecycle:
        - Initial state is DISCONNECTED.
        - Successful connection transitions to CONNECTED.
        - Runtime transport disconnect transitions to RECONNECTING -> CONNECTED.
        - CLOSED state is strictly terminal (no auto-reconnection).
    Reconnection:
        - Reconnect flow uses RetryPolicy exponential backoff without wall-clock sleeps.
        - Reconnect succeeds after transient connection failures.
        - Exhausting maximum reconnect attempts raises ConnectionClosedError.
    Subscriptions:
        - Subscriptions persist as client state and are restored upon reconnect.
        - Unsubscribed symbols are not restored upon reconnect.
    Sequence:
        - In-order sequential ticks are processed normally.
        - Duplicate ticks (seq == last_seen) are dropped (idempotency invariant).
        - Stale/old ticks (seq < last_seen) are dropped.
        - Sequence gaps trigger resync(expected, seq - 1).
        - Resynced gap ticks are applied in order before newly arrived tick.
        - Resync failure is not silently ignored; aborts into CLOSED with ResyncError.
    Heartbeat:
        - Heartbeat timeout on inactivity triggers transport disconnect and reconnect.
    Shutdown:
        - Explicit close() does not trigger reconnection.
        - close() is idempotent.
        - Adversarial race: disconnect followed by immediate close() blocks reconnection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from vega.broker.retry import RetryPolicy
from vega.broker.websocket import (
    ConnectionClosedError,
    ResyncError,
    SequencedTick,
    SimulatedWebSocketTransport,
    WebSocketClient,
    WebSocketState,
)


def _make_tick(seq: int, price: float = 100.0, symbol: str = "NIFTY50") -> SequencedTick:
    """Helper to create a SequencedTick."""
    return SequencedTick(
        sequence=seq,
        symbol=symbol,
        price=price,
        volume=10.0,
        timestamp=datetime(2023, 1, 2, 9, 30, 0, tzinfo=timezone.utc),
    )


# ─── 1. Lifecycle Tests ───────────────────────────────────────────────────────


def test_initial_state_is_disconnected() -> None:
    """Verify fresh client starts in DISCONNECTED state with empty subscriptions."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)

    assert client.state == WebSocketState.DISCONNECTED
    assert client.subscriptions == set()
    assert client.last_sequence is None
    assert client.processed_ticks == []


def test_successful_connection() -> None:
    """Verify connect() transitions to CONNECTED state."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)

    client.connect()

    assert client.state == WebSocketState.CONNECTED
    assert transport.is_connected is True


def test_reconnect_after_disconnect() -> None:
    """Verify transport drop causes automatic transition through RECONNECTING to CONNECTED."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()
    assert client.state == WebSocketState.CONNECTED

    # Simulate dropped socket
    transport.disconnect()

    # Client must automatically reconnect back to CONNECTED
    assert client.state == WebSocketState.CONNECTED
    assert transport.is_connected is True


def test_closed_state_is_terminal() -> None:
    """Verify CLOSED state is terminal; connect() on closed client raises ConnectionClosedError."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    client.close()
    assert client.state == WebSocketState.CLOSED
    assert transport.is_connected is False

    with pytest.raises(ConnectionClosedError) as exc_info:
        client.connect()

    assert "Cannot connect on a CLOSED WebSocket client" in str(exc_info.value)
    assert client.state == WebSocketState.CLOSED


# ─── 2. Reconnection Tests ───────────────────────────────────────────────────


def test_reconnect_uses_retry_policy() -> None:
    """Verify reconnect uses injected RetryPolicy and sleep delays without real wall-clock sleeps."""
    sleeps: List[float] = []
    # Fail 2 connection attempts before succeeding on 3rd attempt
    transport = SimulatedWebSocketTransport(failure_count_before_success=2)
    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.1,
        backoff_factor=2.0,
        sleep_fn=sleeps.append,
    )

    client = WebSocketClient(transport=transport, retry_policy=retry_policy)
    client.connect()

    assert client.state == WebSocketState.CONNECTED
    # Retried twice: attempt 1 (0.1s), attempt 2 (0.2s)
    assert sleeps == [0.1, 0.2]
    assert transport.connection_attempts == 3


def test_reconnect_succeeds_after_transient_failures() -> None:
    """Verify runtime reconnect succeeds after intermediate connection dropouts."""
    sleeps: List[float] = []
    transport = SimulatedWebSocketTransport()
    retry_policy = RetryPolicy(
        max_attempts=4,
        initial_delay=0.05,
        backoff_factor=2.0,
        sleep_fn=sleeps.append,
    )
    client = WebSocketClient(transport=transport, retry_policy=retry_policy)
    client.connect()

    # Now make subsequent 2 reconnect attempts fail before succeeding
    transport.failure_count_before_success = transport.connection_attempts + 2

    transport.disconnect()

    assert client.state == WebSocketState.CONNECTED
    assert len(sleeps) == 2
    assert sleeps == [0.05, 0.1]


def test_max_reconnect_attempts_raise() -> None:
    """Verify exceeding max reconnect attempts transitions to CLOSED and raises ConnectionClosedError."""
    sleeps: List[float] = []
    # Always fail
    transport = SimulatedWebSocketTransport(failure_count_before_success=999)
    retry_policy = RetryPolicy(
        max_attempts=3,
        initial_delay=0.1,
        sleep_fn=sleeps.append,
    )
    client = WebSocketClient(transport=transport, retry_policy=retry_policy)

    with pytest.raises(ConnectionClosedError) as exc_info:
        client.connect()

    assert "Failed to connect after 3 attempts" in str(exc_info.value)
    assert client.state == WebSocketState.CLOSED
    assert len(sleeps) == 2


# ─── 3. Subscription Restoration Tests ────────────────────────────────────────


def test_subscriptions_are_restored_after_reconnect() -> None:
    """
    Verify subscriptions are client state and are automatically re-sent
    to the new transport instance upon reconnect.
    """
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)

    # Subscriptions added before and after connect
    client.subscribe("NIFTY50")
    client.connect()
    client.subscribe("BANKNIFTY")

    assert transport.subscribed_symbols == {"NIFTY50", "BANKNIFTY"}
    assert client.subscriptions == {"NIFTY50", "BANKNIFTY"}

    # Simulate transport wipe/disconnect
    transport.subscribed_symbols.clear()
    transport.disconnect()

    # After reconnect, subscriptions must be fully restored on transport
    assert client.state == WebSocketState.CONNECTED
    assert transport.subscribed_symbols == {"NIFTY50", "BANKNIFTY"}


def test_unsubscribed_symbol_is_not_restored() -> None:
    """Verify unsubscribing removes the symbol from client state so it is not restored."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    client.subscribe("NIFTY50")
    client.subscribe("FINNIFTY")
    client.unsubscribe("FINNIFTY")

    assert client.subscriptions == {"NIFTY50"}
    assert transport.subscribed_symbols == {"NIFTY50"}

    transport.subscribed_symbols.clear()
    transport.disconnect()

    assert transport.subscribed_symbols == {"NIFTY50"}


# ─── 4. Sequence & Gap Resync Tests ───────────────────────────────────────────


def test_sequential_ticks_are_processed() -> None:
    """Verify contiguous in-order ticks are processed sequentially."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    t1 = _make_tick(1050, 100.0)
    t2 = _make_tick(1051, 101.0)
    t3 = _make_tick(1052, 102.0)

    client.handle_tick(t1)
    client.handle_tick(t2)
    client.handle_tick(t3)

    assert [t.sequence for t in client.processed_ticks] == [1050, 1051, 1052]
    assert client.last_sequence == 1052
    assert client.dropped_ticks_count == 0
    assert client.resync_events_count == 0


def test_duplicate_tick_is_dropped() -> None:
    """Verify duplicate tick (seq == last_seen) is dropped without processing."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    client.handle_tick(_make_tick(1050))
    client.handle_tick(_make_tick(1051))
    # Duplicate 1051
    client.handle_tick(_make_tick(1051))

    assert [t.sequence for t in client.processed_ticks] == [1050, 1051]
    assert client.last_sequence == 1051
    assert client.dropped_ticks_count == 1


def test_old_tick_is_dropped() -> None:
    """Verify stale out-of-order tick (seq < last_seen) is dropped."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    client.handle_tick(_make_tick(1050))
    client.handle_tick(_make_tick(1051))
    client.handle_tick(_make_tick(1052))
    # Stale tick 1050 arriving late
    client.handle_tick(_make_tick(1050))

    assert [t.sequence for t in client.processed_ticks] == [1050, 1051, 1052]
    assert client.last_sequence == 1052
    assert client.dropped_ticks_count == 1


def test_sequence_gap_triggers_resync() -> None:
    """
    Verify sequence gap detection triggers transport.resync:
        1050, 1051, 1052
        receive 1055 -> gap [1053..1054]
        resync returns 1053, 1054
        final processed: 1050, 1051, 1052, 1053, 1054, 1055
    """
    transport = SimulatedWebSocketTransport()
    # Store ticks on transport for resync lookup
    transport.stored_ticks[1053] = _make_tick(1053, 103.0)
    transport.stored_ticks[1054] = _make_tick(1054, 104.0)

    client = WebSocketClient(transport=transport)
    client.connect()

    client.handle_tick(_make_tick(1050))
    client.handle_tick(_make_tick(1051))
    client.handle_tick(_make_tick(1052))

    # Inbound tick jumps from 1052 directly to 1055
    client.handle_tick(_make_tick(1055, 105.0))

    assert [t.sequence for t in client.processed_ticks] == [
        1050, 1051, 1052, 1053, 1054, 1055
    ]
    assert client.last_sequence == 1055
    assert client.resync_events_count == 1
    assert client.dropped_ticks_count == 0


def test_resync_ticks_are_applied_before_new_tick() -> None:
    """Verify missing ticks are emitted in chronological order prior to the newly arrived tick."""
    transport = SimulatedWebSocketTransport()
    transport.stored_ticks[101] = _make_tick(101, 11.0)
    transport.stored_ticks[102] = _make_tick(102, 12.0)

    emitted_order: List[int] = []
    client = WebSocketClient(transport=transport, on_tick=lambda t: emitted_order.append(t.sequence))
    client.connect()

    client.handle_tick(_make_tick(100, 10.0))
    # Gap: 101, 102 missing. Incoming tick: 103
    client.handle_tick(_make_tick(103, 13.0))

    assert emitted_order == [100, 101, 102, 103]
    assert client.last_sequence == 103


def test_resync_failure_is_not_silently_ignored() -> None:
    """Verify resync failure raises ResyncError and sets client state to CLOSED."""
    transport = SimulatedWebSocketTransport()
    transport.fail_resync = True  # Resync will fail

    client = WebSocketClient(transport=transport)
    client.connect()

    client.handle_tick(_make_tick(1050))
    client.handle_tick(_make_tick(1051))

    # Gap: 1052 missing, incoming is 1053
    with pytest.raises(ResyncError) as exc_info:
        client.handle_tick(_make_tick(1053))

    assert "Resync failed for gap 1052..1052" in str(exc_info.value)
    assert client.state == WebSocketState.CLOSED


# ─── 5. Heartbeat Tests ───────────────────────────────────────────────────────


def test_heartbeat_timeout_causes_reconnect() -> None:
    """Verify inactivity exceeding heartbeat_timeout triggers transport disconnect and reconnect."""
    clock_time = 1000.0

    def mock_clock() -> float:
        return clock_time

    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(
        transport=transport,
        clock_fn=mock_clock,
        heartbeat_timeout=15.0,
    )
    client.connect()
    assert client.state == WebSocketState.CONNECTED

    # Check heartbeat immediately (0s elapsed) -> healthy
    assert client.check_heartbeat() is False
    assert transport.connection_attempts == 1

    # Advance clock by 10s (within 15s timeout) -> healthy
    clock_time = 1010.0
    assert client.check_heartbeat() is False
    assert transport.connection_attempts == 1

    # Advance clock past timeout (20s since connect) -> heartbeat timeout triggered
    clock_time = 1020.0
    timeout_triggered = client.check_heartbeat()

    assert timeout_triggered is True
    # Disconnect & reconnect occurred
    assert client.state == WebSocketState.CONNECTED
    assert transport.connection_attempts == 2


# ─── 6. Shutdown & Adversarial Tests ──────────────────────────────────────────


def test_shutdown_does_not_reconnect() -> None:
    """Verify calling close() cleanly disconnects and does not auto-reconnect."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    client.close()

    assert client.state == WebSocketState.CLOSED
    assert transport.is_connected is False


def test_shutdown_is_idempotent() -> None:
    """Verify repeated close() calls are safe and keep state CLOSED."""
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    client.close()
    client.close()
    client.close()

    assert client.state == WebSocketState.CLOSED


def test_adversarial_disconnect_then_close_prevents_reconnect() -> None:
    """
    Adversarial race condition test:
    Transport disconnect is triggered, but before reconnection proceeds,
    client.close() is called. Client must remain terminal CLOSED without reconnecting.
    """
    transport = SimulatedWebSocketTransport()
    client = WebSocketClient(transport=transport)
    client.connect()

    # Override disconnect callback to simulate close() called immediately upon disconnect
    def on_disconnect_race():
        client.close()

    transport._on_disconnect = on_disconnect_race

    # Trigger disconnect
    transport.disconnect()

    # Invariant: State must remain CLOSED, no reconnect attempts allowed
    assert client.state == WebSocketState.CLOSED
    assert transport.is_connected is False
