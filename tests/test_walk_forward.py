"""
Unit, regression, and state-isolation tests for WalkForwardEvaluator.

Covers:
1. Correct windows (700 bars -> exact 7 windows for train=252, test=63, step=63)
2. No overlapping test windows (test 1: 252->315, test 2: 315->378, test 3: 378->441)
3. Insufficient data handled cleanly (len(bars) < train_size + test_size)
4. Fresh state isolation (assert object identity: portfolio, broker, strategy, risk)
5. No lookahead (test window only accesses bars within its test range)
6. Fresh strategy instance per window
7. Results contain complete performance metrics
8. Support for both attribute (w.result) and dictionary (w["result"]) access
9. Walk-forward execution with real strategy (StopAndReverseStrategy)
10. Parameter validation (train_size <= 0, test_size <= 0, step <= 0 raise ValueError)
11. Multi-window summary formatting
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from vega.data.models import Bar
from vega.engine.backtest import BacktestResult
from vega.engine.walk_forward import (
    WalkForwardEvaluator,
    WalkForwardResult,
    WalkForwardWindow,
    WalkForwardWindowResult,
)
from vega.orders.models import Fill, Order, OrderSide
from vega.strategy.base import BaseStrategy
from vega.strategy.stop_and_reverse import StopAndReverseStrategy


def make_bar(
    price: float,
    dt: datetime | None = None,
    volume: float = 1000.0,
) -> Bar:
    """Helper to construct a valid Bar object."""
    timestamp = dt or datetime(2026, 1, 1, 9, 15, 0, tzinfo=timezone.utc)
    p = float(price)
    return Bar(
        timestamp=timestamp,
        open=p,
        high=p + 1.0,
        low=p - 1.0,
        close=p,
        volume=float(volume),
    )


def generate_bars(count: int, base_price: float = 100.0) -> list[Bar]:
    """Generate chronological bars for walk-forward testing."""
    base_dt = datetime(2024, 1, 1, 9, 15, 0, tzinfo=timezone.utc)
    return [
        make_bar(price=base_price + (i % 20), dt=base_dt + timedelta(days=i))
        for i in range(count)
    ]


class MockPositionStrategy(BaseStrategy):
    """Controllable test strategy that always enters a position on the first bar."""

    def __init__(self, symbol: str = "NIFTY50") -> None:
        super().__init__(symbol=symbol)
        self.bar_count: int = 0
        self.received_fills: list[Fill] = []
        self.net_position: int = 0

    def on_bar(self, bar: Bar, regime=None) -> list[Order]:
        orders = []
        if self.bar_count == 0:
            # Emit BUY order to establish a position in this window
            orders.append(
                Order(
                    client_order_id=f"TEST-BUY-{bar.timestamp.strftime('%Y%m%d')}",
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=5,
                    price=bar.close,
                )
            )
        self.bar_count += 1
        return orders

    def on_fill(self, fill: Fill) -> None:
        self.received_fills.append(fill)
        if fill.side == OrderSide.BUY:
            self.net_position += fill.filled_qty
        else:
            self.net_position -= fill.filled_qty


# ── Test 1: Correct Windows ──────────────────────────────────────────────────


def test_correct_windows() -> None:
    """
    Test 1: For 700 bars with train=252, test=63, step=63:
    Verify exactly 7 windows are produced with exact start/end indices.
    """
    bars = generate_bars(700)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        train_size=252,
        test_size=63,
        step=63,
    )
    windows = evaluator.get_windows()

    assert len(windows) == 7

    # Verify first 3 windows match specification exactly
    assert windows[0].train_start_idx == 0
    assert windows[0].train_end_idx == 252
    assert windows[0].test_start_idx == 252
    assert windows[0].test_end_idx == 315

    assert windows[1].train_start_idx == 63
    assert windows[1].train_end_idx == 315
    assert windows[1].test_start_idx == 315
    assert windows[1].test_end_idx == 378

    assert windows[2].train_start_idx == 126
    assert windows[2].train_end_idx == 378
    assert windows[2].test_start_idx == 378
    assert windows[2].test_end_idx == 441

    # Verify final window 6: train 378..630, test 630..693
    assert windows[6].train_start_idx == 378
    assert windows[6].train_end_idx == 630
    assert windows[6].test_start_idx == 630
    assert windows[6].test_end_idx == 693


# ── Test 2: No Overlapping Test Windows ───────────────────────────────────────


def test_no_overlapping_test_windows() -> None:
    """
    Test 2: Test windows are contiguous and do not overlap.
    test 1: 252 -> 315
    test 2: 315 -> 378
    test 3: 378 -> 441
    """
    bars = generate_bars(700)
    evaluator = WalkForwardEvaluator(bars=bars, train_size=252, test_size=63, step=63)
    windows = evaluator.get_windows()

    for i in range(len(windows) - 1):
        # Current test end must strictly equal next test start
        assert windows[i].test_end_idx == windows[i + 1].test_start_idx
        # Test length must always equal test_size
        assert windows[i].test_end_idx - windows[i].test_start_idx == 63


# ── Test 3: Insufficient Data ────────────────────────────────────────────────


def test_insufficient_data() -> None:
    """
    Test 3: If len(bars) < train_size + test_size,
    get_windows() and run() handle it cleanly returning empty results.
    """
    # 300 bars is less than 252 + 63 = 315 bars
    bars = generate_bars(300)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy=MockPositionStrategy(),
        train_size=252,
        test_size=63,
    )
    assert evaluator.get_windows() == []

    result = evaluator.run()
    assert len(result) == 0
    assert result.windows == []


# ── Test 4: Fresh State Isolation (Assert Object Identity) ───────────────────


def test_fresh_state_isolation() -> None:
    """
    Test 4 (CRITICAL): Assert object identity and state isolation across windows.
    No Portfolio, Broker, Strategy, or RiskManager may be reused across windows.
    A position opened in Window 0 must NEVER leak into Window 1.
    """
    # 380 bars provides 2 windows (315 + 63 = 378 <= 380)
    bars = generate_bars(380)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy_factory=lambda: MockPositionStrategy(),
        train_size=252,
        test_size=63,
        step=63,
        initial_cash=1_000_000.0,
    )
    result = evaluator.run()

    assert len(result) == 2
    w0 = result.windows[0]
    w1 = result.windows[1]

    # 1. Assert strict object identity separation (different memory instances)
    assert w0.portfolio is not w1.portfolio
    assert w0.broker is not w1.broker
    assert w0.strategy is not w1.strategy
    assert w0.risk_manager is not w1.risk_manager

    # 2. Window 0 opened a position of 5 units
    assert w0.strategy.net_position == 5
    assert w0.portfolio.get_position("NIFTY50").quantity == 5

    # 3. Window 1 starts completely clean with initial_cash and 0 position
    assert w1.portfolio.get_initial_cash() == 1_000_000.0
    # Before fills on window 1's subsequent bars, starting position was 0
    assert len(w1.broker.get_fills()) == 1  # Window 1's own fill only
    assert len(w0.broker.get_fills()) == 1  # Window 0's own fill only


# ── Test 5: No Lookahead ─────────────────────────────────────────────────────


def test_no_lookahead() -> None:
    """
    Test 5: A test window must not use bars after its test end.
    The backtest for window 0 only receives bars up to test_end_idx.
    """
    bars = generate_bars(500)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy_factory=lambda: MockPositionStrategy(),
        train_size=252,
        test_size=63,
        step=63,
    )
    windows = evaluator.get_windows()
    result = evaluator.run()

    for w_meta, w_res in zip(windows, result.windows):
        # The number of equity points in the backtest result must equal test_size
        assert len(w_res.result.equity_curve) == 63
        # The final equity point timestamp must match the last bar in the test slice
        last_bar_ts = bars[w_meta.test_end_idx - 1].timestamp
        assert w_res.result.equity_curve[-1].timestamp == last_bar_ts


# ── Test 6: Fresh Strategy ───────────────────────────────────────────────────


def test_fresh_strategy() -> None:
    """Test 6: Each backtest receives a new strategy instance."""
    bars = generate_bars(400)
    strat = MockPositionStrategy()
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy=strat,
        train_size=252,
        test_size=63,
        step=63,
    )
    result = evaluator.run()

    # Even when passed an instance, evaluator clones it so every window gets a fresh strategy
    assert len(result) >= 2
    assert result.windows[0].strategy is not result.windows[1].strategy
    assert result.windows[0].strategy is not strat


# ── Test 7: Results Contain Complete Performance Information ─────────────────


def test_results_contain_complete_performance_info() -> None:
    """Test 7: Each window result contains enough metrics to compare performance."""
    bars = generate_bars(400)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy_factory=lambda: MockPositionStrategy(),
        train_size=252,
        test_size=63,
        step=63,
    )
    result = evaluator.run()

    for w in result:
        assert isinstance(w.result, BacktestResult)
        assert hasattr(w.result, "total_return")
        assert hasattr(w.result, "final_equity")
        assert hasattr(w.result, "max_drawdown")
        assert hasattr(w.result, "number_of_fills")
        assert w.test_start is not None
        assert w.test_end is not None


# ── Test 8: Dict and Attribute Access ─────────────────────────────────────────


def test_walk_forward_dict_and_attribute_access() -> None:
    """Test 8: Window results support both attribute and dictionary access."""
    bars = generate_bars(400)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy_factory=lambda: MockPositionStrategy(),
        train_size=252,
        test_size=63,
        step=63,
    )
    result = evaluator.run()
    w = result[0]

    # Attribute access
    assert w.result is not None
    assert w.test_start is not None

    # Dictionary access
    assert w["result"] is w.result
    assert w["test_start"] == w.test_start
    assert w["test_end"] == w.test_end
    assert w["train_start"] == w.train_start

    with pytest.raises(KeyError, match="Invalid window result key"):
        _ = w["nonexistent_key"]


# ── Test 9: Walk-Forward with Real Strategy ───────────────────────────────────


def test_walk_forward_with_real_strategy() -> None:
    """Test 9: End-to-end walk-forward evaluation using real StopAndReverseStrategy."""
    bars = generate_bars(400, base_price=150.0)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy_factory=lambda: StopAndReverseStrategy("NIFTY50", fast_period=5, slow_period=10),
        train_size=252,
        test_size=63,
        step=63,
    )
    result = evaluator.run()

    assert len(result) == 2
    for w in result:
        assert isinstance(w.result, BacktestResult)
        assert w.result.initial_cash == 1_000_000.0


# ── Test 10: Parameter Validation ────────────────────────────────────────────


def test_invalid_parameters() -> None:
    """Test 10: train_size <= 0, test_size <= 0, step <= 0, initial_cash <= 0 raise ValueError."""
    bars = generate_bars(100)

    with pytest.raises(ValueError, match="train_size must be a positive integer"):
        WalkForwardEvaluator(bars=bars, train_size=0)

    with pytest.raises(ValueError, match="test_size must be a positive integer"):
        WalkForwardEvaluator(bars=bars, test_size=-5)

    with pytest.raises(ValueError, match="step must be a positive integer"):
        WalkForwardEvaluator(bars=bars, step=0)

    with pytest.raises(ValueError, match="initial_cash must be positive"):
        WalkForwardEvaluator(bars=bars, initial_cash=-1000.0)


# ── Test 11: Summary Output ──────────────────────────────────────────────────


def test_summary_output() -> None:
    """Test 11: WalkForwardResult.summary() outputs clean multi-window performance text."""
    bars = generate_bars(400)
    evaluator = WalkForwardEvaluator(
        bars=bars,
        strategy_factory=lambda: MockPositionStrategy(),
        train_size=252,
        test_size=63,
        step=63,
    )
    result = evaluator.run()

    summary_text = result.summary()
    assert "Walk-Forward Evaluation Summary" in summary_text
    assert "Total Windows:" in summary_text
    assert "Train Size:     252 bars" in summary_text
    assert "Test Size:      63 bars" in summary_text
