"""
Real-time streaming pipeline module for the Vega Quant Trading Engine.

Exports:
    - Events: TickEvent, BarEvent
    - Producer: AsyncTickProducer, generate_deterministic_ticks
    - Consumer: RealtimeBarAccumulator, AsyncTickConsumer, get_bucket_start
    - Pipeline: RealtimePipeline, PipelineMetrics, VegaQueue
"""

from vega.realtime.consumer import (
    AsyncTickConsumer,
    RealtimeBarAccumulator,
    get_bucket_start,
)
from vega.realtime.events import BarEvent, TickEvent
from vega.realtime.pipeline import PipelineMetrics, RealtimePipeline, VegaQueue
from vega.realtime.producer import (
    AsyncTickProducer,
    generate_deterministic_ticks,
)

__all__ = [
    "TickEvent",
    "BarEvent",
    "AsyncTickProducer",
    "generate_deterministic_ticks",
    "RealtimeBarAccumulator",
    "AsyncTickConsumer",
    "get_bucket_start",
    "RealtimePipeline",
    "PipelineMetrics",
    "VegaQueue",
]
