"""Benchmark abstractions for medical AI evaluation."""

from .base import Benchmark, BenchmarkDataPoint
from .rexvqa_benchmark import ReXVQABenchmark

__all__ = [
    "Benchmark",
    "BenchmarkDataPoint", 
    "ReXVQABenchmark",
] 