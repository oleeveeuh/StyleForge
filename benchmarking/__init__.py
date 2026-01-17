"""
StyleForge Benchmarking Framework

A comprehensive framework for benchmarking CUDA kernels with automated
validation, performance metrics, and report generation.

Usage:
    >>> from benchmarking import BenchmarkFramework, BenchmarkConfig
    >>> from benchmarking.reports import BenchmarkReport
    >>>
    >>> framework = BenchmarkFramework()
    >>> result = framework.compare(...)
    >>> BenchmarkReport.generate_markdown_report(results, 'report.md')
"""

from .benchmark_framework import (
    BenchmarkFramework,
    BenchmarkConfig,
    BenchmarkResult,
    ComparisonResult,
)

from .metrics import PerformanceMetrics, GPUSpecs
from .reports import BenchmarkReport, ReportFormat
from .visualize import BenchmarkVisualizer, HAS_MATPLOTLIB

__all__ = [
    "BenchmarkFramework",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ComparisonResult",
    "PerformanceMetrics",
    "GPUSpecs",
    "BenchmarkReport",
    "ReportFormat",
    "BenchmarkVisualizer",
    "HAS_MATPLOTLIB",
]

__version__ = "0.1.0"
