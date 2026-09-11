"""
Deterministic Paper Broker for the Vega Quant Trading Engine.

Simulates order intake, cancellation, and execution fills with deterministic
slippage and brokerage models. Supports idempotent order submission and
safe public query methods without exposing internal mutable collections.

Numerical Precision Policy:
    - Indicators and raw prices are processed as standard 64-bit floats (float64).
    - Fill execution price calculation:
        BUY:  market_price * (1.0 + slippage_pct)
        SELL: market_price * (1.0 - slippage_pct)
    - Intermediate calculations retain full float64 precision without premature rounding.
    - Final Fill prices and brokerage charges are rounded to 2 decimal places
      (representing Indian Rupee paise precision) using Python's built-in round()
      upon Fill instantiation.
    - Python round() semantics note: Built-in round(val, 2) uses IEEE 754
      round-half-to-even (banker's rounding), preventing directional bias across fills.
      The complete monetary accounting policy will be formalized during the Portfolio phase.
"""

from __future__ import annotations

from datetime import datetime

from vega.broker.base import AbstractBroker
from vega.config import VegaConfig, load_config
from vega.orders.models import Fill, Order, OrderSide, OrderStatus


class BrokerError(Exception):
    """Base exception for broker execution and communication errors."""

    pass


class IdempotencyConflictError(BrokerError):
    """
    Raised when an order is submitted with an existing client_order_id
    but with different order parameters (symbol, side, quantity, price).
    """

    pass


class PaperBroker(AbstractBroker):
    """
    In-memory, deterministic paper broker.

    Key Responsibilities:
        1. Order Submission: Validates and enqueues orders as PENDING.
        2. Idempotency: Duplicate `client_order_id` with identical parameters returns existing order;
           conflicting parameters raise IdempotencyConflictError without altering the existing order.
        3. Cancellation: Transitions PENDING orders to CANCELLED.
        4. Execution: Separate execution trigger simulating fills at given market prices.
        5. Fills & Accounting: Applies deterministic slippage and brokerage models.
    """

    def __init__(
        self,
        config: VegaConfig | None = None,
        slippage_pct: float | None = None,
        brokerage_pct: float | None = None,
    ) -> None:
        """
        Initialize PaperBroker.

        Args:
            config: Optional VegaConfig instance. If not provided, default loaded.
            slippage_pct: Optional override for slippage percentage (e.g. 0.001 = 0.1%).
            brokerage_pct: Optional override for brokerage percentage (e.g. 0.0003 = 0.03%).
        """
        cfg = config or load_config()
        self.slippage_pct: float = float(slippage_pct if slippage_pct is not None else cfg.slippage_pct)
        self.brokerage_pct: float = float(brokerage_pct if brokerage_pct is not None else cfg.brokerage_pct)

        # Private internal state - never exposed directly to callers
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []

        # Failure simulation controls for testing
        self.simulate_broker_failure: bool = False
        self.simulate_order_rejection: bool = False

    # ── AbstractBroker Interface Methods ────────────────────────────────────────

    def place_order(self, order: Order) -> Order:
        """
        Submit an order to the paper broker.

        Enforces idempotency:
            - Case A (Identical parameters): If an order with the same client_order_id
              and identical symbol, side, quantity, and price was already submitted,
              returns the existing order without creating a duplicate.
            - Case B (Conflicting parameters): If the client_order_id exists but any
              core parameter differs, raises IdempotencyConflictError and leaves the
              original order unmodified.

        Args:
            order: The Order to place. Must have status PENDING.

        Returns:
            The newly placed or existing Order.

        Raises:
            BrokerError: If simulated failure is active.
            IdempotencyConflictError: If client_order_id exists with different parameters.
            ValueError: If order is not in PENDING status on initial submission.
        """
        if self.simulate_broker_failure:
            raise BrokerError("Simulated broker network/system failure during place_order.")

        # Idempotency check
        if order.client_order_id in self._orders:
            existing = self._orders[order.client_order_id]
            # Verify if order parameters match identically
            if (
                existing.symbol == order.symbol
                and existing.side == order.side
                and existing.quantity == order.quantity
                and existing.price == order.price
            ):
                return existing
            raise IdempotencyConflictError(
                f"Idempotency conflict for client_order_id '{order.client_order_id}': "
                f"Existing order has (symbol={existing.symbol}, side={existing.side.value}, "
                f"quantity={existing.quantity}, price={existing.price}), but received conflicting "
                f"(symbol={order.symbol}, side={order.side.value}, quantity={order.quantity}, price={order.price}). "
                f"Original order remains unchanged."
            )

        if order.status != OrderStatus.PENDING:
            raise ValueError(
                f"New orders must be submitted in PENDING status, got {order.status.value}"
            )

        if self.simulate_order_rejection:
            order.mark_rejected(reason="Simulated broker rejection")
            self._orders[order.client_order_id] = order
            return order

        self._orders[order.client_order_id] = order
        return order

    def cancel_order(self, client_order_id: str) -> Order:
        """
        Cancel a pending order by client_order_id.

        Args:
            client_order_id: The client_order_id of the order to cancel.

        Returns:
            The cancelled Order.

        Raises:
            BrokerError: If simulated failure is active.
            KeyError: If order is not found.
            InvalidOrderStateTransitionError: If order cannot be cancelled (e.g. already FILLED).
        """
        if self.simulate_broker_failure:
            raise BrokerError("Simulated broker network/system failure during cancel_order.")

        if client_order_id not in self._orders:
            raise KeyError(f"Order '{client_order_id}' not found.")

        order = self._orders[client_order_id]
        order.mark_cancelled(reason="Cancelled by client")
        return order

    def get_order(self, client_order_id: str) -> Order | None:
        """
        Look up an order by client_order_id.

        Returns:
            The Order if found, or None.
        """
        return self._orders.get(client_order_id)

    def get_all_orders(self) -> list[Order]:
        """
        Return a list of all orders tracked by the broker.

        Returns:
            A new list containing all Order objects (shallow copy).
        """
        return list(self._orders.values())

    def get_fills(self, order_id: str | None = None) -> list[Fill]:
        """
        Return recorded execution fills.

        Args:
            order_id: If provided, filter fills by this client_order_id.

        Returns:
            List of Fill objects.
        """
        if order_id is not None:
            return [f for f in self._fills if f.order_id == order_id]
        return list(self._fills)

    # ── Execution Mechanism (Separation of Submission & Execution) ─────────────

    def execute_order(
        self,
        client_order_id: str,
        market_price: float,
        timestamp: datetime,
    ) -> Fill:
        """
        Execute a single pending order at the given market price.

        Applies deterministic slippage and brokerage models:
            BUY:  fill_price = round(market_price * (1.0 + slippage_pct), 2)
            SELL: fill_price = round(market_price * (1.0 - slippage_pct), 2)
            brokerage = round(fill_price * quantity * brokerage_pct, 2)

        Args:
            client_order_id: Identifier of the pending order.
            market_price: Current market price at execution time.
            timestamp: Execution timestamp.

        Returns:
            The resulting Fill object.

        Raises:
            KeyError: If order not found.
            ValueError: If order is not in PENDING status.
        """
        if client_order_id not in self._orders:
            raise KeyError(f"Order '{client_order_id}' not found.")

        order = self._orders[client_order_id]
        if order.status != OrderStatus.PENDING:
            raise ValueError(
                f"Cannot execute order '{client_order_id}' with status {order.status.value}. "
                f"Only PENDING orders can be executed."
            )

        if market_price <= 0.0:
            raise ValueError(f"market_price must be positive, got {market_price}")

        # Deterministic slippage model
        if order.side == OrderSide.BUY:
            raw_fill_price = market_price * (1.0 + self.slippage_pct)
        else:
            raw_fill_price = market_price * (1.0 - self.slippage_pct)

        fill_price = round(raw_fill_price, 2)
        slippage_amt = round(abs(fill_price - market_price), 2)
        brokerage = round(fill_price * order.quantity * self.brokerage_pct, 2)

        # Transition order state machine to FILLED
        order.mark_filled()

        fill = Fill(
            order_id=order.client_order_id,
            filled_price=fill_price,
            filled_qty=order.quantity,
            timestamp=timestamp,
            slippage=slippage_amt,
            brokerage=brokerage,
        )
        self._fills.append(fill)
        return fill

    def execute_pending_orders(
        self,
        market_price: float,
        timestamp: datetime,
    ) -> list[Fill]:
        """
        Execute all currently PENDING orders at the specified market price.

        This decoupled method allows backtesting and paper trading engines
        to simulate fills at bar open/close prices independently from the
        strategy order generation cycle.

        Args:
            market_price: Current reference price (e.g. next bar's Open).
            timestamp: Execution timestamp.

        Returns:
            List of newly created Fill objects.
        """
        pending_ids = [
            order.client_order_id
            for order in self._orders.values()
            if order.status == OrderStatus.PENDING
        ]

        fills: list[Fill] = []
        for oid in pending_ids:
            fill = self.execute_order(
                client_order_id=oid,
                market_price=market_price,
                timestamp=timestamp,
            )
            fills.append(fill)

        return fills
