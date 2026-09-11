"""
WebSocket connection management and market-data reliability abstractions.

Provides:
    - WebSocketState: Explicit 5-state connection lifecycle.
    - SequencedTick: Market tick carrying an integer sequence number.
    - SimulatedWebSocketTransport: Deterministic transport test double.
    - WebSocketClient: Resilient market data client with reconnect, resync,
      subscription restoration, and heartbeat monitoring.
    - WebSocketError, ConnectionClosedError, ResyncError: Domain exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Any, Callable, Sequence

from vega.broker.retry import RetryPolicy
from vega.data.models import Tick


# ─── Exceptions ───────────────────────────────────────────────────────────────


class WebSocketError(Exception):
    """Base exception for WebSocket client operations."""
    pass


class ConnectionClosedError(WebSocketError):
    """Raised when WebSocket connection is closed or reconnect attempts exhausted."""
    pass


class ResyncError(WebSocketError):
    """Raised when sequence resynchronization fails."""
    pass


# ─── Connection States ────────────────────────────────────────────────────────


class WebSocketState(str, Enum):
    """
    Explicit connection lifecycle states.
    CLOSED is terminal — once CLOSED, no reconnection is attempted.
    """

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    CLOSED = "CLOSED"


# ─── Sequenced Market Data Event ──────────────────────────────────────────────


@dataclass(frozen=True)
class SequencedTick:
    """
    Market tick event carrying a deterministic sequence number.

    Attributes:
        sequence: Monotonically increasing sequence identifier.
        symbol: Ticker symbol (e.g. 'NIFTY50').
        price: Traded price.
        volume: Traded volume.
        timestamp: Trade timestamp.
    """

    sequence: int
    symbol: str
    price: float
    volume: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_tick(self) -> Tick:
        """Convert to internal Vega domain Tick model."""
        return Tick(timestamp=self.timestamp, price=self.price, volume=self.volume)


# ─── Simulated Transport ──────────────────────────────────────────────────────


class SimulatedWebSocketTransport:
    """
    Deterministic simulated WebSocket transport for unit testing.

    Permits testing connection failures, disconnects, resync queries, and tick streams
    without live network sockets.
    """

    def __init__(self, failure_count_before_success: int = 0) -> None:
        self.is_connected = False
        self.subscribed_symbols: set[str] = set()
        self.connection_attempts = 0
        self.failure_count_before_success = failure_count_before_success
        self.stored_ticks: dict[int, SequencedTick] = {}
        self.fail_resync = False

        self._on_message: Callable[[SequencedTick], None] | None = None
        self._on_disconnect: Callable[[], None] | None = None

    def connect(self) -> None:
        """Simulate connection attempt."""
        self.connection_attempts += 1
        if self.connection_attempts <= self.failure_count_before_success:
            raise ConnectionError(
                f"Simulated connection failure #{self.connection_attempts}"
            )
        self.is_connected = True

    def disconnect(self) -> None:
        """Simulate transport disconnect."""
        if not self.is_connected:
            return
        self.is_connected = False
        if self._on_disconnect is not None:
            self._on_disconnect()

    def subscribe(self, symbol: str) -> None:
        """Record subscription on transport."""
        self.subscribed_symbols.add(symbol)

    def unsubscribe(self, symbol: str) -> None:
        """Remove subscription on transport."""
        self.subscribed_symbols.discard(symbol)

    def resync(self, start_seq: int, end_seq: int) -> list[SequencedTick]:
        """Fetch missing sequence gap ticks."""
        if self.fail_resync:
            raise ResyncError(f"Failed to fetch missing sequences {start_seq}..{end_seq}")
        return [
            self.stored_ticks[seq]
            for seq in range(start_seq, end_seq + 1)
            if seq in self.stored_ticks
        ]

    def emit_tick(self, tick: SequencedTick) -> None:
        """Simulate inbound tick from the transport."""
        self.stored_ticks[tick.sequence] = tick
        if self._on_message is not None:
            self._on_message(tick)


# ─── WebSocket Client ─────────────────────────────────────────────────────────


class WebSocketClient:
    """
    Market-data WebSocket client with state-machine connection management.

    Key Reliability Guarantees:
        - Strict 5-state lifecycle (CLOSED is terminal).
        - Reconnect with RetryPolicy exponential backoff.
        - Subscriptions survive transport drops and are automatically restored upon reconnect.
        - Monotonic sequence tracking:
            - seq <= last_seen: duplicate/old tick -> DROPPED (idempotent).
            - seq == last_seen + 1: normal progression.
            - seq > last_seen + 1: gap detected -> triggers resync(expected, seq - 1),
              applies missing ticks first, then applies incoming tick.
            - resync failure aborts into CLOSED state with ResyncError.
        - Heartbeat timeout triggers automated reconnect.
    """

    def __init__(
        self,
        transport: SimulatedWebSocketTransport,
        retry_policy: RetryPolicy | None = None,
        on_tick: Callable[[SequencedTick], None] | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        heartbeat_timeout: float = 30.0,
    ) -> None:
        self.transport = transport
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=3, initial_delay=0.1, backoff_factor=2.0
        )
        self.on_tick = on_tick
        self.clock_fn = clock_fn
        self.heartbeat_timeout = float(heartbeat_timeout)

        self.state: WebSocketState = WebSocketState.DISCONNECTED
        self.subscriptions: set[str] = set()
        self.last_sequence: int | None = None
        self.last_message_time: float = self.clock_fn()

        # Operational metrics
        self.processed_ticks: list[SequencedTick] = []
        self.dropped_ticks_count: int = 0
        self.resync_events_count: int = 0

        # Wire transport callbacks
        self.transport._on_message = self.handle_tick
        self.transport._on_disconnect = self._handle_transport_disconnect

    def _do_connect(self) -> None:
        """Single connection attempt to transport."""
        self.transport.connect()

    def connect(self) -> None:
        """
        Initiate connection using configured RetryPolicy.

        Raises:
            ConnectionClosedError: If client is CLOSED or max attempts exhausted.
        """
        if self.state == WebSocketState.CLOSED:
            raise ConnectionClosedError("Cannot connect on a CLOSED WebSocket client")
        if self.state == WebSocketState.CONNECTED:
            return

        self.state = WebSocketState.CONNECTING
        try:
            self.retry_policy.execute(self._do_connect)
            self.state = WebSocketState.CONNECTED
            self.last_message_time = self.clock_fn()
            self._restore_subscriptions()
        except Exception as exc:
            self.state = WebSocketState.CLOSED
            raise ConnectionClosedError(
                f"Failed to connect after {self.retry_policy.max_attempts} attempts: {exc}"
            ) from exc

    def _handle_transport_disconnect(self) -> None:
        """Callback invoked when transport drops connection."""
        if self.state == WebSocketState.CLOSED:
            return
        self.reconnect()

    def reconnect(self) -> None:
        """
        Execute reconnect flow with exponential backoff and subscription restoration.

        Raises:
            ConnectionClosedError: If client is CLOSED or max attempts exhausted.
        """
        if self.state == WebSocketState.CLOSED:
            return

        self.state = WebSocketState.RECONNECTING
        try:
            self.state = WebSocketState.CONNECTING
            self.retry_policy.execute(self._do_connect)
            self.state = WebSocketState.CONNECTED
            self.last_message_time = self.clock_fn()
            self._restore_subscriptions()
        except Exception as exc:
            self.state = WebSocketState.CLOSED
            raise ConnectionClosedError(
                f"Reconnect failed after {self.retry_policy.max_attempts} attempts: {exc}"
            ) from exc

    def close(self) -> None:
        """
        Terminally close the client connection.
        Once closed, no automatic reconnect will be attempted.
        """
        if self.state == WebSocketState.CLOSED:
            return
        self.state = WebSocketState.CLOSED
        self.transport.disconnect()

    def subscribe(self, symbol: str) -> None:
        """
        Subscribe to market ticks for a symbol.
        Subscription is client state and persists across disconnects.
        """
        self.subscriptions.add(symbol)
        if self.state == WebSocketState.CONNECTED:
            self.transport.subscribe(symbol)

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from market ticks for a symbol."""
        self.subscriptions.discard(symbol)
        if self.state == WebSocketState.CONNECTED:
            self.transport.unsubscribe(symbol)

    def _restore_subscriptions(self) -> None:
        """Re-subscribe all tracked symbols on the active transport."""
        for symbol in self.subscriptions:
            self.transport.subscribe(symbol)

    def _emit_tick(self, tick: SequencedTick) -> None:
        """Dispatch a validated, in-sequence tick to consumer."""
        self.processed_ticks.append(tick)
        if self.on_tick is not None:
            self.on_tick(tick)

    def handle_tick(self, tick: SequencedTick) -> None:
        """
        Process an inbound tick with sequence verification, deduplication, and gap resync.

        Policy:
            - seq <= last_seen: Duplicate or stale tick -> DROPPED (idempotency invariant).
            - seq == last_seen + 1: Sequential tick -> PROCESSED.
            - seq > last_seen + 1: Sequence gap -> resync(expected, seq - 1) called.
              Missing ticks applied in order first, then new tick applied.
            - Resync failure aborts into CLOSED state with ResyncError.
        """
        self.last_message_time = self.clock_fn()

        if self.last_sequence is None:
            self.last_sequence = tick.sequence
            self._emit_tick(tick)
            return

        expected = self.last_sequence + 1

        # 1. Duplicate or old tick: drop
        if tick.sequence < expected:
            self.dropped_ticks_count += 1
            return

        # 2. Sequential in-order tick
        if tick.sequence == expected:
            self.last_sequence = tick.sequence
            self._emit_tick(tick)
            return

        # 3. Sequence gap detected (tick.sequence > expected)
        start_gap = expected
        end_gap = tick.sequence - 1
        self.resync_events_count += 1

        try:
            missing_ticks = self.transport.resync(start_gap, end_gap)
        except Exception as exc:
            self.state = WebSocketState.CLOSED
            raise ResyncError(
                f"Resync failed for gap {start_gap}..{end_gap}: {exc}"
            ) from exc

        # Apply missing ticks in chronological sequence order
        for missing in sorted(missing_ticks, key=lambda t: t.sequence):
            if missing.sequence == self.last_sequence + 1:
                self.last_sequence = missing.sequence
                self._emit_tick(missing)
            elif missing.sequence <= self.last_sequence:
                pass
            else:
                self.state = WebSocketState.CLOSED
                raise ResyncError(
                    f"Resync gap not fully resolved: expected {self.last_sequence + 1}, got {missing.sequence}"
                )

        # After missing ticks are applied, apply the incoming tick
        if tick.sequence == self.last_sequence + 1:
            self.last_sequence = tick.sequence
            self._emit_tick(tick)
        elif tick.sequence <= self.last_sequence:
            self.dropped_ticks_count += 1
        else:
            self.state = WebSocketState.CLOSED
            raise ResyncError(
                f"Resync left gap before incoming tick: expected {self.last_sequence + 1}, got {tick.sequence}"
            )

    def check_heartbeat(self) -> bool:
        """
        Inspect heartbeat elapsed time against configured timeout.

        Returns:
            True if heartbeat timed out and reconnect was triggered, False otherwise.
        """
        if self.state != WebSocketState.CONNECTED:
            return False

        now = self.clock_fn()
        if now - self.last_message_time > self.heartbeat_timeout:
            # Heartbeat loss detected: trigger transport disconnect
            self.transport.disconnect()
            return True

        return False
