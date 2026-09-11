"""
Cache package for the Vega Quant Trading Engine.

Exports:
    - RedisStateCache: State cache adapter for external telemetry.
    - InMemoryRedisClient: Lightweight in-memory mock client for tests.
"""

from vega.cache.redis_cache import InMemoryRedisClient, RedisStateCache

__all__ = [
    "RedisStateCache",
    "InMemoryRedisClient",
]
