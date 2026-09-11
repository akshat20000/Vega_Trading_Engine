"""
Rolling Walk-Forward Evaluation for the Vega Quant Trading Engine.

Implements sequential out-of-sample performance evaluation across rolling time horizons:

Historical Data
      │
      ▼
┌───────────────┐
│ Train Window  │ 252 bars
└───────────────┘
      │
      ▼
┌───────────────┐
│ Test Window   │ 63 bars (Out-of-sample backtest)
└───────────────┘
      │
      ▼
window moves step (63 bars)
      │
      ▼
┌───────────────┐
│ Train Window  │
└───────────────┘
      │
      ▼
┌───────────────┐
│ Test Window   │
└───────────────┘

Core Invariants:
    1. Rolling Evaluation Only: Evaluates fixed strategy parameters out-of-sample.
       Parameter optimization is intentionally not performed in this phase.
    2. Pure get_windows(): Window boundary calculation does NOT instantiate components,
       run backtests, or mutate any state.
    3. Zero State Leakage: Every evaluation window instantiates completely fresh,
       unconnected Strategy, PaperBroker, RiskManager, and Portfolio instances.
    4. No Lookahead: Out-of-sample backtest sees strictly bars within its test slice.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

from vega.broker.paper import PaperBroker
from vega.config import VegaConfig
from vega.data.models import Bar
from vega.engine.backtest import BacktestEngine, BacktestResult
from vega.macro.models import MacroSnapshot
from vega.portfolio.portfolio import Portfolio
from vega.risk.manager import RiskManager
from vega.strategy.atr_grid import ATRGridStrategy
from vega.strategy.base import BaseStrategy
from vega.strategy.stop_and_reverse import StopAndReverseStrategy


@dataclass(frozen=True)
class WalkForwardWindow:
    """
    Metadata defining the index and timestamp boundaries of a walk-forward window.

    Attributes:
        window_index: 0-indexed sequence counter of this window.
        train_start_idx: Start index of the training period (inclusive).
        train_end_idx: End index of the training period (exclusive).
        test_start_idx: Start index of the testing period (inclusive).
        test_end_idx: End index of the testing period (exclusive).
        train_start: Timestamp of the first bar in train period.
        train_end: Timestamp of the last bar in train period.
        test_start: Timestamp of the first bar in test period.
        test_end: Timestamp of the last bar in test period.
    """

    window_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start: datetime | None
    train_end: datetime | None
    test_start: datetime | None
    test_end: datetime | None


@dataclass
class WalkForwardWindowResult:
    """
    Evaluation results for a single walk-forward window.

    Attributes:
        window_index: Index of the evaluated window.
        train_start_idx: Index of train start.
        train_end_idx: Index of train end.
        test_start_idx: Index of test start.
        test_end_idx: Index of test end.
        train_start: Timestamp of train start.
        train_end: Timestamp of train end.
        test_start: Timestamp of test start.
        test_end: Timestamp of test end.
        result: Out-of-sample BacktestResult on the test window.
        train_result: Optional in-sample BacktestResult on the train window.
        strategy: Strategy instance used in this window (for state isolation verification).
        portfolio: Portfolio instance used in this window (for state isolation verification).
        broker: Broker instance used in this window (for state isolation verification).
        risk_manager: RiskManager instance used in this window.
    """

    window_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start: datetime | None
    train_end: datetime | None
    test_start: datetime | None
    test_end: datetime | None
    result: BacktestResult
    train_result: BacktestResult | None = None
    strategy: BaseStrategy | None = None
    portfolio: Portfolio | None = None
    broker: PaperBroker | None = None
    risk_manager: RiskManager | None = None

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access (e.g. window['result'], window['test_start'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Invalid window result key: '{key}'")


@dataclass
class WalkForwardResult:
    """
    Aggregated container for all walk-forward evaluation windows.

    Attributes:
        windows: List of WalkForwardWindowResult objects.
        train_size: Training window length in bars.
        test_size: Testing window length in bars.
        step: Step size between windows in bars.
        total_bars: Total input bars provided.
    """

    windows: list[WalkForwardWindowResult] = field(default_factory=list)
    train_size: int = 252
    test_size: int = 63
    step: int = 63
    total_bars: int = 0

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self):
        return iter(self.windows)

    def __getitem__(self, idx: int) -> WalkForwardWindowResult:
        return self.windows[idx]

    def summary(self) -> str:
        """Generate a formatted multi-window evaluation summary."""
        lines = [
            f"=== Walk-Forward Evaluation Summary ===",
            f"Total Windows:  {len(self.windows)}",
            f"Train Size:     {self.train_size} bars",
            f"Test Size:      {self.test_size} bars",
            f"Step:           {self.step} bars",
            f"Total Bars:     {self.total_bars}",
            "-" * 60,
            f"{'Win':<4} {'Test Start':<20} {'Test End':<20} {'Return':<10} {'MaxDD%':<8} {'Fills':<6}",
            "-" * 60,
        ]
        for w in self.windows:
            start_str = str(w.test_start)[:19] if w.test_start else f"idx {w.test_start_idx}"
            end_str = str(w.test_end)[:19] if w.test_end else f"idx {w.test_end_idx}"
            ret_str = f"{w.result.total_return:+.2f}%"
            dd_str = f"{w.result.max_drawdown_pct:.2f}%"
            fills_str = str(w.result.number_of_fills)
            lines.append(
                f"{w.window_index:<4} {start_str:<20} {end_str:<20} {ret_str:<10} {dd_str:<8} {fills_str:<6}"
            )
        lines.append("-" * 60)
        return "\n".join(lines)


def _clone_strategy(strategy: BaseStrategy) -> BaseStrategy:
    """
    Produce a completely fresh, unlinked instance of a strategy with identical parameters.

    Ensures zero state leakage (position=0, active_entries=[], bar_index=0).
    """
    if isinstance(strategy, ATRGridStrategy):
        return ATRGridStrategy(
            symbol=strategy.symbol,
            atr_period=strategy.atr_period,
            grid_multiplier=strategy.grid_multiplier,
            levels=strategy.levels,
            position_cap=strategy.position_cap,
            quantity_per_level=strategy.quantity_per_level,
        )
    if isinstance(strategy, StopAndReverseStrategy):
        return StopAndReverseStrategy(
            symbol=strategy.symbol,
            fast_period=strategy.fast_period,
            slow_period=strategy.slow_period,
            quantity=strategy.quantity,
        )
    # Generic clone fallback
    fresh = copy.deepcopy(strategy)
    # Reset known state attributes if custom
    if hasattr(fresh, "net_position"):
        fresh.net_position = 0
    if hasattr(fresh, "position"):
        fresh.position = 0
    if hasattr(fresh, "active_entries"):
        fresh.active_entries = []
    if hasattr(fresh, "history"):
        fresh.history = []
    if hasattr(fresh, "bar_index"):
        fresh.bar_index = 0
    return fresh


class WalkForwardEvaluator:
    """
    Deterministic Rolling Walk-Forward Evaluator.

    Splits historical bars into sequential (train, test) slices and coordinates
    independent BacktestEngine runs for each window, strictly guaranteeing zero
    state leakage.
    """

    def __init__(
        self,
        bars: Sequence[Bar],
        strategy: BaseStrategy | Callable[[], BaseStrategy] | None = None,
        train_size: int = 252,
        test_size: int = 63,
        step: int = 63,
        initial_cash: float = 1_000_000.0,
        config: VegaConfig | None = None,
        strategy_factory: Callable[[], BaseStrategy] | None = None,
        macro_snapshots: Sequence[MacroSnapshot] | None = None,
        evaluate_train: bool = False,
    ) -> None:
        """
        Initialize the WalkForwardEvaluator with parameters.

        Args:
            bars: Complete sequence of chronological Bar objects.
            strategy: Strategy instance or factory function producing fresh strategy instances.
            train_size: Lookback length in bars for historical context (default: 252).
            test_size: Length in bars for out-of-sample backtest window (default: 63).
            step: Offset in bars to shift window forward on each step (default: 63).
            initial_cash: Starting cash balance for each independent window (default: 10,00,000 INR).
            config: Optional VegaConfig instance.
            strategy_factory: Optional explicit factory callable returning BaseStrategy.
            macro_snapshots: Optional macro proxy snapshots for regime awareness.
            evaluate_train: If True, also run an in-sample backtest on train_bars.

        Raises:
            ValueError: If train_size, test_size, or step are <= 0, or initial_cash <= 0.
        """
        if train_size <= 0:
            raise ValueError(f"train_size must be a positive integer, got {train_size}")
        if test_size <= 0:
            raise ValueError(f"test_size must be a positive integer, got {test_size}")
        if step <= 0:
            raise ValueError(f"step must be a positive integer, got {step}")
        if initial_cash <= 0.0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        self.bars: Sequence[Bar] = bars
        self.strategy: BaseStrategy | Callable[[], BaseStrategy] | None = strategy
        self.train_size: int = int(train_size)
        self.test_size: int = int(test_size)
        self.step: int = int(step)
        self.initial_cash: float = float(initial_cash)
        self.config: VegaConfig | None = config
        self.strategy_factory: Callable[[], BaseStrategy] | None = strategy_factory
        self.macro_snapshots: Sequence[MacroSnapshot] | None = macro_snapshots
        self.evaluate_train: bool = evaluate_train

    def get_windows(self) -> list[WalkForwardWindow]:
        """
        Pure calculation of rolling window index and timestamp boundaries.

        Does NOT instantiate strategies, run backtests, or mutate any state.

        Returns:
            List of WalkForwardWindow objects with exact start and end index bounds.
        """
        n = len(self.bars)
        if n < self.train_size + self.test_size:
            return []

        windows: list[WalkForwardWindow] = []
        w_idx = 0
        current_offset = 0

        while current_offset + self.train_size + self.test_size <= n:
            train_start = current_offset
            train_end = current_offset + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            ts_train_start = self.bars[train_start].timestamp if train_start < n else None
            ts_train_end = self.bars[train_end - 1].timestamp if train_end > 0 else None
            ts_test_start = self.bars[test_start].timestamp if test_start < n else None
            ts_test_end = self.bars[test_end - 1].timestamp if test_end > 0 else None

            windows.append(
                WalkForwardWindow(
                    window_index=w_idx,
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                    train_start=ts_train_start,
                    train_end=ts_train_end,
                    test_start=ts_test_start,
                    test_end=ts_test_end,
                )
            )

            current_offset += self.step
            w_idx += 1

        return windows

    def _create_strategy(
        self,
        strategy_override: BaseStrategy | Callable[[], BaseStrategy] | None = None,
    ) -> BaseStrategy:
        """Instantiate a completely fresh strategy instance for a window."""
        target = strategy_override or self.strategy_factory or self.strategy
        if target is None:
            raise ValueError(
                "No strategy or strategy_factory provided to WalkForwardEvaluator."
            )
        if callable(target):
            return target()
        return _clone_strategy(target)

    def run(
        self,
        strategy: BaseStrategy | Callable[[], BaseStrategy] | None = None,
    ) -> WalkForwardResult:
        """
        Execute walk-forward evaluation across all rolling windows.

        For each window:
            - Instantiates fresh Strategy, PaperBroker, RiskManager, Portfolio.
            - Runs BacktestEngine on the out-of-sample test window slice.
            - Guarantees zero state leakage between windows.

        Args:
            strategy: Optional strategy or factory override.

        Returns:
            WalkForwardResult containing evaluation metrics for each window.
        """
        windows = self.get_windows()
        if not windows:
            return WalkForwardResult(
                windows=[],
                train_size=self.train_size,
                test_size=self.test_size,
                step=self.step,
                total_bars=len(self.bars),
            )

        window_results: list[WalkForwardWindowResult] = []

        for w in windows:
            # Out-of-sample test slice
            test_slice = self.bars[w.test_start_idx : w.test_end_idx]

            # Mandatory Rule: Fresh isolated state for EVERY window
            fresh_strategy = self._create_strategy(strategy)
            fresh_broker = PaperBroker(config=self.config)
            fresh_risk = RiskManager(config=self.config)
            fresh_portfolio = Portfolio(initial_cash=self.initial_cash)

            engine = BacktestEngine(
                strategy=fresh_strategy,
                broker=fresh_broker,
                risk_manager=fresh_risk,
                portfolio=fresh_portfolio,
                config=self.config,
                macro_snapshots=self.macro_snapshots,
            )

            test_result = engine.run(test_slice)

            train_result = None
            if self.evaluate_train:
                # Independent fresh run on train slice if requested
                train_slice = self.bars[w.train_start_idx : w.train_end_idx]
                train_strat = self._create_strategy(strategy)
                train_broker = PaperBroker(config=self.config)
                train_risk = RiskManager(config=self.config)
                train_port = Portfolio(initial_cash=self.initial_cash)
                train_engine = BacktestEngine(
                    strategy=train_strat,
                    broker=train_broker,
                    risk_manager=train_risk,
                    portfolio=train_port,
                    config=self.config,
                    macro_snapshots=self.macro_snapshots,
                )
                train_result = train_engine.run(train_slice)

            w_res = WalkForwardWindowResult(
                window_index=w.window_index,
                train_start_idx=w.train_start_idx,
                train_end_idx=w.train_end_idx,
                test_start_idx=w.test_start_idx,
                test_end_idx=w.test_end_idx,
                train_start=w.train_start,
                train_end=w.train_end,
                test_start=w.test_start,
                test_end=w.test_end,
                result=test_result,
                train_result=train_result,
                strategy=fresh_strategy,
                portfolio=fresh_portfolio,
                broker=fresh_broker,
                risk_manager=fresh_risk,
            )
            window_results.append(w_res)

        return WalkForwardResult(
            windows=window_results,
            train_size=self.train_size,
            test_size=self.test_size,
            step=self.step,
            total_bars=len(self.bars),
        )
