"""
Risk Manager for the Vega Quant Trading Engine.

Acts as a strict pre-order validation gate answering:
    "Is this order allowed?"

Enforces 6 non-negotiable risk rules:
    1. Kill Switch: Blocks all new orders when engaged.
    2. Macro Circuit Breaker: Rejects orders when macro volatility circuit breaker fires.
    3. Daily Loss Limit: Rejects orders when daily realized loss breaches threshold.
    4. Max Order Quantity: Blocks single orders exceeding maximum allowed quantity.
    5. Position Cap: Blocks orders resulting in absolute net position exceeding maximum.
    6. Pyramiding Limit: Prevents adding more same-direction entry legs than configured.

The RiskManager does NOT:
    - Place or execute orders
    - Mutate broker or order state
    - Calculate strategy signals or indicators
"""

from __future__ import annotations

from dataclasses import dataclass

from enum import Enum

from vega.config import VegaConfig, load_config
from vega.orders.models import Order, OrderSide


class OrderEffect(Enum):
    """
    Explicit classification of how an order interacts with the current position.

    Values:
        NEW_ENTRY: Opening a fresh position from flat (current_position == 0).
        SAME_DIRECTION_ENTRY: Adding a leg to an existing open position (pyramiding limit applies).
        PARTIAL_REDUCTION: Decreasing the size of an open position without flattening it.
        COMPLETE_REDUCTION: Flattening an open position exactly to zero.
        REVERSAL: Completely closing an open position and establishing an opposite position.
    """

    NEW_ENTRY = "NEW_ENTRY"
    SAME_DIRECTION_ENTRY = "SAME_DIRECTION_ENTRY"
    PARTIAL_REDUCTION = "PARTIAL_REDUCTION"
    COMPLETE_REDUCTION = "COMPLETE_REDUCTION"
    REVERSAL = "REVERSAL"


@dataclass(frozen=True)
class RiskDecision:
    """
    Result of a risk validation check on a proposed order.

    Attributes:
        is_allowed: True if order passes all risk controls, False otherwise.
        reason: Explanation if rejected, or empty string if approved.
        order_effect: Explicitly classified effect on position (e.g. SAME_DIRECTION_ENTRY, REVERSAL).
    """

    is_allowed: bool
    reason: str = ""
    order_effect: OrderEffect | None = None

    def __bool__(self) -> bool:
        """Allow evaluating `if decision:` directly."""
        return self.is_allowed


class RiskManager:
    """
    Validates proposed orders against configurable risk constraints.

    All limits are parameterized via VegaConfig with sensible defaults.
    """

    def __init__(
        self,
        config: VegaConfig | None = None,
        max_position_size: int | None = None,
        max_order_quantity: int | None = None,
        max_pyramids: int | None = None,
        daily_loss_limit: float | None = None,
    ) -> None:
        """
        Initialize RiskManager with risk parameters.

        Args:
            config: Optional VegaConfig instance.
            max_position_size: Maximum absolute net position (in lots) allowed.
            max_order_quantity: Maximum single order quantity (in lots). Defaults to max_position_size.
            max_pyramids: Maximum open same-direction entry legs.
            daily_loss_limit: Maximum permissible daily realized loss in INR (positive value).
        """
        cfg = config or load_config()

        self.max_position_size: int = int(
            max_position_size if max_position_size is not None else cfg.max_position_size
        )
        self.max_order_quantity: int = int(
            max_order_quantity if max_order_quantity is not None else self.max_position_size
        )
        self.max_pyramids: int = int(
            max_pyramids if max_pyramids is not None else cfg.max_pyramids
        )
        self.daily_loss_limit: float = float(
            daily_loss_limit if daily_loss_limit is not None else cfg.daily_loss_limit
        )

        # Kill switch state
        self._kill_switch_active: bool = False
        self._kill_switch_reason: str = ""

    # ── Kill Switch Controls ───────────────────────────────────────────────────

    def enable_kill_switch(self, reason: str = "Manual kill switch activated") -> None:
        """
        Engage the emergency kill switch to block all new orders.

        Does not itself flatten open positions (that belongs to an execution layer).
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason

    def disable_kill_switch(self) -> None:
        """Disengage the kill switch and resume normal order validation."""
        self._kill_switch_active = False
        self._kill_switch_reason = ""

    def is_kill_switch_active(self) -> bool:
        """Return True if the kill switch is currently engaged."""
        return self._kill_switch_active

    # ── Order Effect Classification ───────────────────────────────────────────

    @staticmethod
    def classify_order_effect(order: Order, current_position: int) -> OrderEffect:
        """
        Classify how a proposed order will interact with the current position.

        Explicitly distinguishes:
            - NEW_ENTRY: Opening from flat (current_position == 0).
            - SAME_DIRECTION_ENTRY: Adding to existing long (BUY) or short (SELL).
            - PARTIAL_REDUCTION: Opposite-side order smaller than current position.
            - COMPLETE_REDUCTION: Opposite-side order exactly matching current position.
            - REVERSAL: Opposite-side order exceeding current position, flipping net exposure.
        """
        if current_position == 0:
            return OrderEffect.NEW_ENTRY

        if current_position > 0:
            # Currently LONG
            if order.side == OrderSide.BUY:
                return OrderEffect.SAME_DIRECTION_ENTRY
            # order.side == OrderSide.SELL
            if order.quantity < current_position:
                return OrderEffect.PARTIAL_REDUCTION
            if order.quantity == current_position:
                return OrderEffect.COMPLETE_REDUCTION
            return OrderEffect.REVERSAL

        # Currently SHORT (current_position < 0)
        if order.side == OrderSide.SELL:
            return OrderEffect.SAME_DIRECTION_ENTRY
        # order.side == OrderSide.BUY
        short_units = abs(current_position)
        if order.quantity < short_units:
            return OrderEffect.PARTIAL_REDUCTION
        if order.quantity == short_units:
            return OrderEffect.COMPLETE_REDUCTION
        return OrderEffect.REVERSAL

    # ── Pre-Order Risk Validation Gate ────────────────────────────────────────

    def validate_order(
        self,
        order: Order,
        current_position: int = 0,
        current_pyramids: int = 0,
        daily_realized_pnl: float = 0.0,
        macro_circuit_breaker: bool = False,
        position_cap_override: int | None = None,
    ) -> RiskDecision:
        """
        Evaluate whether a proposed order is permissible under risk rules.

        Args:
            order: Proposed Order to validate.
            current_position: Current net position in units/lots (+ for long, - for short, 0 for flat).
            current_pyramids: Current count of same-direction entry legs in open position.
            daily_realized_pnl: Realized P&L for the day in INR (negative for loss).
            macro_circuit_breaker: True if the Macro Regime circuit breaker is active.
            position_cap_override: Optional lower position cap (e.g. from BEARISH macro regime).

        Returns:
            RiskDecision: Object indicating whether order is allowed, rejection reason, and OrderEffect.
        """
        # Determine explicit order effect (entry, reduction, or reversal)
        effect = self.classify_order_effect(order, current_position)

        # 1. Kill Switch Check
        if self._kill_switch_active:
            return RiskDecision(
                is_allowed=False,
                reason=f"Kill switch is ACTIVE: {self._kill_switch_reason}",
                order_effect=effect,
            )

        # 2. Macro Circuit Breaker Check
        if macro_circuit_breaker:
            return RiskDecision(
                is_allowed=False,
                reason="Macro circuit breaker is active (market halted / high macro risk).",
                order_effect=effect,
            )

        # 3. Daily Loss Limit Check
        # Boundary is inclusive: if daily_realized_pnl <= -daily_loss_limit, reject.
        if daily_realized_pnl <= -self.daily_loss_limit:
            return RiskDecision(
                is_allowed=False,
                reason=(
                    f"Daily loss limit breached: realized P&L ({daily_realized_pnl:,.2f} INR) "
                    f"<= -{self.daily_loss_limit:,.2f} INR limit."
                ),
                order_effect=effect,
            )

        # 4. Maximum Order Quantity Check (single order size)
        if order.quantity > self.max_order_quantity:
            return RiskDecision(
                is_allowed=False,
                reason=(
                    f"Order quantity {order.quantity} exceeds max order quantity "
                    f"limit of {self.max_order_quantity}."
                ),
                order_effect=effect,
            )

        # 5. Position Cap Check (resulting net position)
        effective_cap = (
            position_cap_override
            if position_cap_override is not None and position_cap_override < self.max_position_size
            else self.max_position_size
        )

        proposed_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
        proposed_position = current_position + proposed_delta

        if abs(proposed_position) > effective_cap:
            return RiskDecision(
                is_allowed=False,
                reason=(
                    f"Proposed position ({proposed_position}) exceeds position cap "
                    f"({effective_cap}). Current position: {current_position}, order: {order.side.value} {order.quantity}."
                ),
                order_effect=effect,
            )

        # 6. Pyramiding Limit Check
        # Explicitly evaluates order effect:
        # - SAME_DIRECTION_ENTRY: subject to current_pyramids < max_pyramids
        # - PARTIAL_REDUCTION & COMPLETE_REDUCTION: never blocked by pyramiding controls
        # - REVERSAL: closes old position, establishes leg 1 of new opposite position
        # - NEW_ENTRY: establishes leg 1 from flat
        if effect == OrderEffect.SAME_DIRECTION_ENTRY and current_pyramids >= self.max_pyramids:
            return RiskDecision(
                is_allowed=False,
                reason=(
                    f"Pyramiding limit reached: current legs ({current_pyramids}) "
                    f">= max permitted ({self.max_pyramids}) for same-direction entry."
                ),
                order_effect=effect,
            )

        if effect in (OrderEffect.NEW_ENTRY, OrderEffect.REVERSAL) and self.max_pyramids < 1:
            return RiskDecision(
                is_allowed=False,
                reason=f"Pyramiding limit ({self.max_pyramids}) does not permit establishing new positions.",
                order_effect=effect,
            )

        return RiskDecision(is_allowed=True, reason="", order_effect=effect)

    def is_order_allowed(
        self,
        order: Order,
        current_position: int = 0,
        current_pyramids: int = 0,
        daily_realized_pnl: float = 0.0,
        macro_circuit_breaker: bool = False,
        position_cap_override: int | None = None,
    ) -> bool:
        """Convenience boolean check for order validation."""
        decision = self.validate_order(
            order=order,
            current_position=current_position,
            current_pyramids=current_pyramids,
            daily_realized_pnl=daily_realized_pnl,
            macro_circuit_breaker=macro_circuit_breaker,
            position_cap_override=position_cap_override,
        )
        return decision.is_allowed
