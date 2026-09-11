"""
Redis State Cache for the Vega Quant Trading Engine.

Architectural Invariant:
    Redis is NEVER the source of truth.
    The domain engine (Portfolio, RiskManager, PaperBroker) maintains full accounting
    and position authority. Redis acts purely as a transient, low-latency latest-state
    cache for external telemetry and visualization (FastAPI, Streamlit).
    If Redis fails, flushes, or disconnects, the trading engine continues operations
    unaffected and can rehydrate the cache projection on demand.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class InMemoryRedisClient:
    """
    In-memory mock client implementing core Redis string operations.

    Used for deterministic local testing and zero-dependency environments.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        """Get string value by key."""
        return self._store.get(name)

    def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Set string value for key."""
        self._store[name] = str(value)
        return True

    def delete(self, *names: str) -> int:
        """Delete one or more keys."""
        deleted = 0
        for name in names:
            if name in self._store:
                del self._store[name]
                deleted += 1
        return deleted

    def keys(self, pattern: str = "*") -> list[str]:
        """Return list of keys matching prefix or all keys."""
        if pattern == "*":
            return list(self._store.keys())
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._store if k.startswith(prefix)]
        return [k for k in self._store if k == pattern]

    def flushdb(self) -> bool:
        """Clear all stored keys."""
        self._store.clear()
        return True

    def ping(self) -> bool:
        """Health-check ping."""
        return True


class RedisStateCache:
    """
    Transient state cache client for publishing latest trading engine telemetry.

    Explicit Client Requirement:
        - Production: RedisStateCache(redis_client=real_redis_connection)
        - Testing / Local: RedisStateCache(redis_client=InMemoryRedisClient())
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str = "vega:",
        fail_silent: bool = True,
    ) -> None:
        """
        Initialize the cache adapter.

        Args:
            redis_client: Redis client instance (e.g. redis.Redis or InMemoryRedisClient).
            prefix: Key prefix namespace (defaults to 'vega:').
            fail_silent: If True, catches client connection/write errors and logs warnings
                         instead of bubbling up to crash the trading domain engine.
        """
        self.client = redis_client
        self.prefix = prefix
        self.fail_silent = fail_silent

    @property
    def is_in_memory(self) -> bool:
        """Return True if using the in-memory fallback client."""
        return isinstance(self.client, InMemoryRedisClient)

    @classmethod
    def create_in_memory(
        cls,
        prefix: str = "vega:",
        fail_silent: bool = True,
    ) -> RedisStateCache:
        """Factory creating an explicitly in-memory cache instance for tests."""
        return cls(redis_client=InMemoryRedisClient(), prefix=prefix, fail_silent=fail_silent)

    @classmethod
    def from_url(
        cls,
        url: str = "redis://localhost:6379/0",
        prefix: str = "vega:",
        fail_silent: bool = True,
    ) -> RedisStateCache:
        """
        Factory connecting to an external Redis server.

        Raises:
            ImportError: If the 'redis' package is not installed.
            ConnectionError: If connection cannot be established and fail_silent is False.
        """
        try:
            import redis
            client = redis.from_url(url, decode_responses=True)
            return cls(redis_client=client, prefix=prefix, fail_silent=fail_silent)
        except ImportError as e:
            if not fail_silent:
                raise ImportError("redis package required to connect to external Redis server.") from e
            logger.warning("redis package not installed; falling back to InMemoryRedisClient.")
            return cls.create_in_memory(prefix=prefix, fail_silent=fail_silent)

    # ─── Low-Level Safe Storage Operations ────────────────────────────────────

    def _key(self, key_name: str) -> str:
        """Build namespaced key."""
        return f"{self.prefix}{key_name}"

    def _safe_set(self, key_name: str, payload: dict[str, Any]) -> bool:
        """Serialize payload to JSON and write to cache with failure isolation."""
        full_key = self._key(key_name)
        try:
            val = json.dumps(payload, default=str)
            self.client.set(full_key, val)
            return True
        except Exception as exc:
            if self.fail_silent:
                logger.warning("Redis cache write failed for key '%s': %s", full_key, exc)
                return False
            raise

    def _safe_get(self, key_name: str) -> dict[str, Any] | None:
        """Retrieve and parse JSON payload from cache with failure isolation."""
        full_key = self._key(key_name)
        try:
            raw = self.client.get(full_key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            if self.fail_silent:
                logger.warning("Redis cache read failed for key '%s': %s", full_key, exc)
                return None
            raise

    # ─── Domain-Specific Cache Accessors ──────────────────────────────────────

    def set_latest_price(
        self,
        symbol: str,
        price: float,
        timestamp: datetime | str | None = None,
    ) -> bool:
        """Cache latest market price for an instrument."""
        ts_str = (
            timestamp.isoformat()
            if isinstance(timestamp, datetime)
            else str(timestamp or datetime.now(timezone.utc).isoformat())
        )
        payload = {
            "symbol": symbol,
            "price": float(price),
            "timestamp": ts_str,
        }
        return self._safe_set(f"latest:{symbol.upper()}", payload)

    def get_latest_price(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve cached latest price for an instrument."""
        return self._safe_get(f"latest:{symbol.upper()}")

    def set_position(
        self,
        symbol: str,
        quantity: int,
        avg_entry_price: float,
    ) -> bool:
        """Cache currently held position size and cost basis."""
        payload = {
            "symbol": symbol.upper(),
            "quantity": int(quantity),
            "avg_entry_price": float(avg_entry_price),
        }
        return self._safe_set(f"position:{symbol.upper()}", payload)

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve cached position for a symbol."""
        return self._safe_get(f"position:{symbol.upper()}")

    def set_portfolio(
        self,
        equity: float,
        cash: float,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
    ) -> bool:
        """Cache top-level portfolio valuation metrics."""
        payload = {
            "equity": float(equity),
            "cash": float(cash),
            "unrealized_pnl": float(unrealized_pnl),
            "realized_pnl": float(realized_pnl),
            "drawdown_pct": float(drawdown_pct),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return self._safe_set("portfolio:equity", payload)

    def get_portfolio(self) -> dict[str, Any] | None:
        """Retrieve cached top-level portfolio metrics."""
        return self._safe_get("portfolio:equity")

    def set_system_status(
        self,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Cache engine health and kill-switch status."""
        payload = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(details or {}),
        }
        return self._safe_set("system:status", payload)

    def get_system_status(self) -> dict[str, Any] | None:
        """Retrieve cached system status."""
        return self._safe_get("system:status")

    # ─── Rehydration & Cleanup ────────────────────────────────────────────────

    def rehydrate(
        self,
        portfolio: Any,
        latest_prices: dict[str, float] | None = None,
        system_status: str = "RUNNING",
    ) -> int:
        """
        Project current in-memory domain state into Redis cache.

        Flow (strictly one-way):
            Domain (Portfolio / Engine) ──> Redis Cache

        Returns:
            Number of cache keys successfully refreshed.
        """
        count = 0

        # 1. Rehydrate all positions
        if hasattr(portfolio, "get_all_positions"):
            positions = portfolio.get_all_positions()
            for sym, pos in positions.items():
                if self.set_position(sym, pos.quantity, float(pos.average_entry_price)):
                    count += 1

        # 2. Rehydrate top-level portfolio equity metrics
        if hasattr(portfolio, "get_equity") and hasattr(portfolio, "get_cash"):
            equity = float(portfolio.get_equity())
            cash = float(portfolio.get_cash())
            drawdown_pct = (
                float(portfolio.get_drawdown_pct())
                if hasattr(portfolio, "get_drawdown_pct")
                else 0.0
            )
            realized_pnl = (
                float(portfolio.get_realized_pnl())
                if hasattr(portfolio, "get_realized_pnl")
                else 0.0
            )
            unrealized_pnl = (
                float(portfolio.get_unrealized_pnl())
                if hasattr(portfolio, "get_unrealized_pnl")
                else 0.0
            )
            if self.set_portfolio(
                equity=equity,
                cash=cash,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                drawdown_pct=drawdown_pct,
            ):
                count += 1

        # 3. Rehydrate latest prices if provided
        if latest_prices:
            for sym, px in latest_prices.items():
                if self.set_latest_price(sym, px):
                    count += 1

        # 4. Rehydrate system status
        if self.set_system_status(system_status):
            count += 1

        return count

    def clear(self) -> bool:
        """Clear all cache keys matching this adapter's namespace prefix."""
        try:
            if hasattr(self.client, "flushdb") and self.prefix == "vega:":
                self.client.flushdb()
                return True
            pattern = f"{self.prefix}*"
            keys_to_del = self.client.keys(pattern)
            if keys_to_del:
                self.client.delete(*keys_to_del)
            return True
        except Exception as exc:
            if self.fail_silent:
                logger.warning("Failed to clear cache: %s", exc)
                return False
            raise
