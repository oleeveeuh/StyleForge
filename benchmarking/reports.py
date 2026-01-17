"""
StyleForge - Report Generation Module

Generate professional reports from benchmark results in multiple formats:
- Markdown (human-readable)
- JSON (machine-readable)
- HTML (web-ready)
- CSV (spreadsheet-compatible)
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
from dataclasses import asdict


class ReportFormat(Enum):
    """Report format options."""
    MARKDOWN = "md"
    JSON = "json"
    HTML = "html"
    CSV = "csv"


class BenchmarkReport:
    """Generate reports from benchmark results."""

    def __init__(
        self,
        title: str = "CUDA Kernel Benchmark Report",
        subtitle: str = "",
        author: str = "StyleForge Benchmarking Framework"
    ):
        """
        Args:
            title: Report title
            subtitle: Report subtitle
            author: Report author
        """
        self.title = title
        self.subtitle = subtitle
        self.author = author

    def generate_markdown(
        self,
        results: List[Dict],
        output_path: str,
        include_detailed: bool = True,
        include_charts: bool = False
    ):
        """
        Generate comprehensive markdown report.

        Args:
            results: List of benchmark comparison results
            output_path: Path to save markdown file
            include_detailed: Include per-benchmark detailed stats
            include_charts: Include placeholder for charts
        """
        lines = []

        # Header
        lines.append(f"# {self.title}\n")
        if self.subtitle:
            lines.append(f"## {self.subtitle}\n")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Author:** {self.author}\n")
        lines.append("\n---\n")

        # System info
        gpu_info = self._get_gpu_info()
        if gpu_info:
            lines.append("## System Configuration\n")
            lines.append("| Property | Value |\n")
            lines.append("|----------|-------|\n")
            for key, value in gpu_info.items():
                lines.append(f"| {key} | {value} |\n")
            lines.append("\n---\n")

        # Summary table
        lines.append("## Performance Summary\n")
        lines.append("| Configuration | PyTorch (ms) | CUDA (ms) | Speedup | Status |\n")
        lines.append("|---------------|--------------|-----------|---------|--------|\n")

        for result in results:
            if result is None:
                continue

            config = result.get('config', 'N/A')
            baseline = result.get('baseline', {})
            optimized = result.get('optimized', {})
            speedup = result.get('speedup', 0)

            baseline_ms = baseline.get('mean_ms', 0)
            optimized_ms = optimized.get('mean_ms', 0)

            # Status emoji
            status = self._get_speedup_status(speedup)

            lines.append(
                f"| {config} | {baseline_ms:.4f} | {optimized_ms:.4f} | "
                f"{speedup:.2f}x | {status} |\n"
            )

        lines.append("\n")

        # Overall statistics
        if results:
            speedups = [r.get('speedup', 0) for r in results if r is not None]
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                max_speedup = max(speedups)
                min_speedup = min(speedups)

                lines.append("### Overall Statistics\n")
                lines.append(f"- **Average Speedup:** {avg_speedup:.2f}x\n")
                lines.append(f"- **Max Speedup:** {max_speedup:.2f}x\n")
                lines.append(f"- **Min Speedup:** {min_speedup:.2f}x\n")
                lines.append(f"- **Benchmarks Run:** {len(results)}\n")
                lines.append("\n")

        lines.append("---\n")

        # Detailed results
        if include_detailed:
            lines.append("## Detailed Results\n")

            for result in results:
                if result is None:
                    continue

                config = result.get('config', 'Unknown')
                baseline = result.get('baseline', {})
                optimized = result.get('optimized', {})

                lines.append(f"### {config}\n")

                # Baseline stats
                lines.append("**PyTorch Baseline:**\n")
                lines.append("```\n")
                lines.append(f"Mean:       {baseline.get('mean_ms', 0):.4f} ms\n")
                lines.append(f"Median:     {baseline.get('median_ms', 0):.4f} ms\n")
                lines.append(f"Std Dev:    {baseline.get('std_ms', 0):.4f} ms\n")
                lines.append(f"Min:        {baseline.get('min_ms', 0):.4f} ms\n")
                lines.append(f"Max:        {baseline.get('max_ms', 0):.4f} ms\n")
                lines.append(f"P95:        {baseline.get('p95_ms', 0):.4f} ms\n")
                lines.append(f"P99:        {baseline.get('p99_ms', 0):.4f} ms\n")
                lines.append(f"Throughput: {baseline.get('throughput_fps', 0):.1f} FPS\n")
                lines.append("```\n")

                # Optimized stats
                lines.append("\n**CUDA Optimized:**\n")
                lines.append("```\n")
                lines.append(f"Mean:       {optimized.get('mean_ms', 0):.4f} ms\n")
                lines.append(f"Median:     {optimized.get('median_ms', 0):.4f} ms\n")
                lines.append(f"Std Dev:    {optimized.get('std_ms', 0):.4f} ms\n")
                lines.append(f"Min:        {optimized.get('min_ms', 0):.4f} ms\n")
                lines.append(f"Max:        {optimized.get('max_ms', 0):.4f} ms\n")
                lines.append(f"P95:        {optimized.get('p95_ms', 0):.4f} ms\n")
                lines.append(f"P99:        {optimized.get('p99_ms', 0):.4f} ms\n")
                lines.append(f"Throughput: {optimized.get('throughput_fps', 0):.1f} FPS\n")
                lines.append("```\n")

                # Improvement
                lines.append("\n**Improvement:**\n")
                lines.append(f"- Speedup: {result.get('speedup', 0):.2f}x\n")
                lines.append(
                    f"- Memory reduction: {result.get('memory_transfer_reduction', 'N/A')}\n"
                )
                lines.append(f"- Max error: {result.get('max_error', 0):.2e}\n")
                lines.append(f"- Validation: {'✓ PASSED' if result.get('passed_validation') else '✗ FAILED'}\n")
                lines.append("\n")

        # Chart placeholders
        if include_charts:
            lines.append("---\n")
            lines.append("## Visualizations\n")
            lines.append("\n![Speedup Comparison](charts/speedup_comparison.png)\n")
            lines.append("\n![Latency Distribution](charts/latency_distribution.png)\n")
            lines.append("\n")

        # Footer
        lines.append("---\n")
        lines.append(f"\n*Report generated by {self.author}*\n")

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"✅ Markdown report saved to: {output_path}")

    def generate_json(
        self,
        results: List[Dict],
        output_path: str,
        pretty: bool = True
    ):
        """
        Generate JSON report.

        Args:
            results: List of benchmark results
            output_path: Path to save JSON file
            pretty: Use pretty formatting
        """
        data = {
            'metadata': {
                'title': self.title,
                'subtitle': self.subtitle,
                'author': self.author,
                'timestamp': datetime.now().isoformat(),
                'gpu_info': self._get_gpu_info(),
            },
            'results': results,
            'summary': self._calculate_summary(results),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)

        print(f"✅ JSON report saved to: {output_path}")

    def generate_csv(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        Generate CSV report (spreadsheet-compatible).

        Args:
            results: List of benchmark results
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Configuration',
                'Baseline Mean (ms)',
                'Baseline Std (ms)',
                'Baseline P95 (ms)',
                'Optimized Mean (ms)',
                'Optimized Std (ms)',
                'Optimized P95 (ms)',
                'Speedup',
                'Max Error',
                'Validation Status'
            ])

            # Data rows
            for result in results:
                if result is None:
                    continue

                baseline = result.get('baseline', {})
                optimized = result.get('optimized', {})

                writer.writerow([
                    result.get('config', ''),
                    baseline.get('mean_ms', ''),
                    baseline.get('std_ms', ''),
                    baseline.get('p95_ms', ''),
                    optimized.get('mean_ms', ''),
                    optimized.get('std_ms', ''),
                    optimized.get('p95_ms', ''),
                    result.get('speedup', ''),
                    result.get('max_error', ''),
                    'PASS' if result.get('passed_validation') else 'FAIL'
                ])

        print(f"✅ CSV report saved to: {output_path}")

    def generate_html(
        self,
        results: List[Dict],
        output_path: str,
        include_styles: bool = True
    ):
        """
        Generate HTML report.

        Args:
            results: List of benchmark results
            output_path: Path to save HTML file
            include_styles: Include CSS styling
        """
        lines = []

        if include_styles:
            lines.append("<!DOCTYPE html>\n")
            lines.append("<html>\n")
            lines.append("<head>\n")
            lines.append("<meta charset='utf-8'>\n")
            lines.append("<title>" + self.title + "</title>\n")
            lines.append("<style>\n")
            lines.append(self._get_html_styles())
            lines.append("</style>\n")
            lines.append("</head>\n")

        lines.append("<body>\n")
        lines.append(f"<h1>{self.title}</h1>\n")

        if self.subtitle:
            lines.append(f"<h2>{self.subtitle}</h2>\n")

        lines.append(f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")

        # System info
        gpu_info = self._get_gpu_info()
        if gpu_info:
            lines.append("<div class='card'>\n")
            lines.append("<h3>System Configuration</h3>\n")
            lines.append("<table>\n")
            for key, value in gpu_info.items():
                lines.append(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
            lines.append("</table>\n")
            lines.append("</div>\n")

        # Summary table
        lines.append("<div class='card'>\n")
        lines.append("<h3>Performance Summary</h3>\n")
        lines.append("<table class='results'>\n")
        lines.append("<tr><th>Configuration</th><th>PyTorch (ms)</th><th>CUDA (ms)</th><th>Speedup</th><th>Status</th></tr>\n")

        for result in results:
            if result is None:
                continue

            config = result.get('config', 'N/A')
            baseline = result.get('baseline', {})
            optimized = result.get('optimized', {})
            speedup = result.get('speedup', 0)

            baseline_ms = baseline.get('mean_ms', 0)
            optimized_ms = optimized.get('mean_ms', 0)

            status_class = self._get_speedup_class(speedup)
            status = self._get_speedup_status(speedup)

            lines.append(
                f"<tr><td>{config}</td><td>{baseline_ms:.4f}</td><td>{optimized_ms:.4f}</td>"
                f"<td class='{status_class}'>{speedup:.2f}x</td><td>{status}</td></tr>\n"
            )

        lines.append("</table>\n")
        lines.append("</div>\n")

        # Overall stats
        if results:
            speedups = [r.get('speedup', 0) for r in results if r is not None]
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)

                lines.append("<div class='card'>\n")
                lines.append("<h3>Overall Statistics</h3>\n")
                lines.append(f"<p><strong>Average Speedup:</strong> {avg_speedup:.2f}x</p>\n")
                lines.append(f"<p><strong>Benchmarks Run:</strong> {len(results)}</p>\n")
                lines.append("</div>\n")

        lines.append("</body>\n")
        lines.append("</html>\n")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"✅ HTML report saved to: {output_path}")

    def generate_all_formats(
        self,
        results: List[Dict],
        output_dir: str
    ):
        """
        Generate reports in all formats.

        Args:
            results: List of benchmark results
            output_dir: Directory to save reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.generate_markdown(results, output_dir / "report.md")
        self.generate_json(results, output_dir / "report.json")
        self.generate_csv(results, output_dir / "report.csv")
        self.generate_html(results, output_dir / "report.html")

    @staticmethod
    def _get_gpu_info() -> Optional[Dict[str, str]]:
        """Get current GPU information."""
        import torch

        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(0)

        return {
            'GPU': props.name,
            'CUDA Version': torch.version.cuda,
            'PyTorch Version': torch.__version__,
            'Compute Capability': f"{props.major}.{props.minor}",
            'Total Memory': f"{props.total_memory / 1e9:.2f} GB",
        }

    @staticmethod
    def _get_speedup_status(speedup: float) -> str:
        """Get status message for speedup value."""
        if speedup >= 5.0:
            return "✓✓✓ Excellent"
        elif speedup >= 3.0:
            return "✓✓ Good"
        elif speedup >= 2.0:
            return "✓ Modest"
        elif speedup >= 1.0:
            return "⚠ Minimal"
        else:
            return "✗ Slower"

    @staticmethod
    def _get_speedup_class(speedup: float) -> str:
        """Get CSS class for speedup value."""
        if speedup >= 5.0:
            return "excellent"
        elif speedup >= 3.0:
            return "good"
        elif speedup >= 2.0:
            return "modest"
        else:
            return "poor"

    @staticmethod
    def _calculate_summary(results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        speedups = [r.get('speedup', 0) for r in results if r is not None]

        if not speedups:
            return {}

        return {
            'avg_speedup': sum(speedups) / len(speedups),
            'max_speedup': max(speedups),
            'min_speedup': min(speedups),
            'count': len(speedups),
        }

    @staticmethod
    def _get_html_styles() -> str:
        """Get CSS styles for HTML report."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; }
        .card {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .timestamp { color: #888; font-style: italic; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        table.results th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
        table.results td { padding: 10px; border-bottom: 1px solid #ddd; }
        table.results tr:hover { background: #f9f9f9; }
        .excellent { color: #4CAF50; font-weight: bold; }
        .good { color: #8BC34A; font-weight: bold; }
        .modest { color: #FFC107; font-weight: bold; }
        .poor { color: #F44336; font-weight: bold; }
        """
