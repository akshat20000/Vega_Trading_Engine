"""
Structured event logger and alerting system for the Vega Quant Trading Engine.

Provides:
    - LogEventType: Standardized domain event types.
    - LogEvent: Immutable structured event record with JSON serialization.
    - StructuredLogger: Observer logging machine-readable JSON events and dispatching
      critical alert callbacks (Kill switch, critical errors) without noise on routine fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from typing import Any, Callable


class LogEventType(str, Enum):
    """Domain event categories for structured logging."""

    ORDER = "ORDER"
    FILL = "FILL"
    POSITION_CHANGE = "POSITION_CHANGE"
    PNL_UPDATE = "PNL_UPDATE"
    KILL_SWITCH = "KILL_SWITCH"
    ERROR = "ERROR"
    ALERT = "ALERT"


@dataclass(frozen=True)
class LogEvent:
    """
    Structured machine-readable log event with a fixed schema.

    Schema:
        timestamp: ISO-8601 UTC timestamp string.
        event_type: One of LogEventType.
        level: Severity ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        message: Human-readable explanation.
        data: Arbitrary structured metadata dictionary.
    """

    timestamp: str
    event_type: str
    level: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "level": self.level,
            "message": self.message,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Serialize event to a valid JSON string."""
        return json.dumps(self.to_dict(), default=str)


class StructuredLogger:
    """
    Structured event logger layered on top of Python's standard logging infrastructure.

    Alert Semantics:
        - Routine events (ORDER, FILL, POSITION_CHANGE, PNL_UPDATE) are logged for audit/analytics.
        - High-severity events (KILL_SWITCH, CRITICAL errors/alerts) trigger the `on_alert` callback.
        - Prevents notification fatigue by strictly separating logging from alerting.
    """

    def __init__(
        self,
        name: str = "vega.engine",
        level: int = logging.INFO,
        on_alert: Callable[[LogEvent], None] | None = None,
        clock_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.on_alert = on_alert
        self.clock_fn = clock_fn or (lambda: datetime.now(timezone.utc))

        self.events: list[LogEvent] = []
        self.alert_events: list[LogEvent] = []

    def _create_event(
        self,
        event_type: LogEventType,
        level: str,
        message: str,
        data: dict[str, Any],
    ) -> LogEvent:
        """Construct, record, and dispatch a structured event."""
        now = self.clock_fn()
        ts_str = now.isoformat() if isinstance(now, datetime) else str(now)

        event = LogEvent(
            timestamp=ts_str,
            event_type=event_type.value,
            level=level.upper(),
            message=message,
            data=data,
        )

        self.events.append(event)

        # Log through Python standard logging infrastructure
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, event.to_json())

        # Alert dispatch rule: only genuinely critical events trigger alert callback
        is_alert_worthy = (
            event_type == LogEventType.KILL_SWITCH
            or level.upper() == "CRITICAL"
            or (event_type == LogEventType.ALERT and level.upper() in ("CRITICAL", "ERROR"))
        )

        if is_alert_worthy:
            self.alert_events.append(event)
            if self.on_alert is not None:
                self.on_alert(event)

        return event

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        status: str,
        strategy: str = "",
        reason: str = "",
    ) -> LogEvent:
        """Log order placement, rejection, or cancellation (Log only)."""
        data = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": status,
            "strategy": strategy,
            "reason": reason,
        }
        return self._create_event(
            event_type=LogEventType.ORDER,
            level="INFO",
            message=f"Order {order_id} {status}: {side} {quantity} {symbol} @ {price}",
            data=data,
        )

    def log_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        brokerage: float,
        slippage: float,
        realized_pnl: float,
        strategy: str = "",
    ) -> LogEvent:
        """Log trade execution fill (Log only)."""
        data = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "brokerage": brokerage,
            "slippage": slippage,
            "realized_pnl": realized_pnl,
            "strategy": strategy,
        }
        return self._create_event(
            event_type=LogEventType.FILL,
            level="INFO",
            message=f"Fill executed for {order_id}: {side} {quantity} {symbol} @ {price}",
            data=data,
        )

    def log_position_change(
        self,
        symbol: str,
        old_quantity: int,
        new_quantity: int,
        avg_entry_price: float,
    ) -> LogEvent:
        """Log portfolio position change (Log only)."""
        data = {
            "symbol": symbol,
            "old_quantity": old_quantity,
            "new_quantity": new_quantity,
            "avg_entry_price": avg_entry_price,
        }
        return self._create_event(
            event_type=LogEventType.POSITION_CHANGE,
            level="INFO",
            message=f"Position updated for {symbol}: {old_quantity} -> {new_quantity}",
            data=data,
        )

    def log_pnl_update(
        self,
        equity: float,
        cash: float,
        realized_pnl: float,
        unrealized_pnl: float,
        drawdown: float,
        drawdown_pct: float,
    ) -> LogEvent:
        """Log end-of-bar / mark-to-market P&L metrics (Log only)."""
        data = {
            "equity": equity,
            "cash": cash,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
        }
        return self._create_event(
            event_type=LogEventType.PNL_UPDATE,
            level="INFO",
            message=f"P&L update: Equity ₹{equity:,.2f} | Drawdown {drawdown_pct:.1f}%",
            data=data,
        )

    def log_kill_switch(
        self,
        reason: str,
        triggered_by: str = "risk_manager",
        details: dict[str, Any] | None = None,
    ) -> LogEvent:
        """
        Log emergency kill switch activation.
        ALWAYS dispatches to alert callback.
        """
        data = {
            "reason": reason,
            "triggered_by": triggered_by,
            **(details or {}),
        }
        return self._create_event(
            event_type=LogEventType.KILL_SWITCH,
            level="CRITICAL",
            message=f"KILL SWITCH ACTIVATED by {triggered_by}: {reason}",
            data=data,
        )

    def log_error(
        self,
        message: str,
        error_type: str = "Error",
        is_critical: bool = False,
        details: dict[str, Any] | None = None,
    ) -> LogEvent:
        """
        Log an error. Critical errors dispatch to alert callback.
        """
        level = "CRITICAL" if is_critical else "ERROR"
        data = {
            "error_type": error_type,
            **(details or {}),
        }
        return self._create_event(
            event_type=LogEventType.ERROR,
            level=level,
            message=message,
            data=data,
        )

    def log_alert(
        self,
        message: str,
        level: str = "WARNING",
        details: dict[str, Any] | None = None,
    ) -> LogEvent:
        """
        Log an operational alert. Levels 'CRITICAL' or 'ERROR' dispatch to alert callback.
        """
        return self._create_event(
            event_type=LogEventType.ALERT,
            level=level,
            message=message,
            data=details or {},
        )
