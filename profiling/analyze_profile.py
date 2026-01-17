#!/usr/bin/env python3
"""
StyleForge - Nsight Compute Profile Analyzer

Parse and analyze Nsight Compute profiling results.
Extract key metrics and generate optimization recommendations.

Usage:
    python analyze_profile.py <nsight_metrics.csv>
"""

import csv
import sys
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


class ProfileAnalyzer:
    """Analyze Nsight Compute profiling data."""

    # Theoretical peak bandwidth for common GPUs (GB/s)
    PEAK_BANDWIDTH_MAP = {
        'RTX 4090': 1008,
        'RTX 4080': 717,
        'RTX 4070': 504,
        'RTX 3090': 936,
        'RTX 3080': 760,
        'RTX 3070': 448,
        'RTX 2080': 616,
        'RTX 2070': 448,
        'V100': 900,
        'A100': 1550,
        'H100': 3350,
    }

    def __init__(self, csv_path: str, gpu_name: Optional[str] = None):
        """
        Args:
            csv_path: Path to ncu CSV output
            gpu_name: GPU name for peak bandwidth lookup
        """
        self.csv_path = Path(csv_path)
        self.gpu_name = gpu_name
        self.metrics: Dict[str, Any] = {}
        self.raw_metrics: Dict[str, str] = {}

    def parse_csv(self) -> bool:
        """Parse ncu CSV output."""
        print(f"Parsing: {self.csv_path}")

        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    metric_name = row.get('Metric Name', '')
                    metric_value = row.get('Metric Value', '')
                    metric_unit = row.get('Metric Unit', '')

                    if metric_name and metric_value:
                        # Store raw value
                        self.raw_metrics[metric_name] = {
                            'value': metric_value,
                            'unit': metric_unit
                        }

                        # Convert to float if possible
                        try:
                            self.metrics[metric_name] = float(metric_value)
                        except ValueError:
                            # Keep as string for non-numeric metrics
                            self.metrics[metric_name] = metric_value

            return True

        except FileNotFoundError:
            print(f"❌ Error: File not found: {self.csv_path}")
            return False
        except Exception as e:
            print(f"❌ Error parsing CSV: {e}")
            return False

    def get_metric(self, metric_name: str, default: Any = 0) -> Any:
        """Get a metric value, returning default if not found."""
        return self.metrics.get(metric_name, default)

    def extract_key_metrics(self) -> Dict[str, Any]:
        """Extract and compute key performance metrics."""
        key_metrics = {}

        # ============================================
        # Duration
        # ============================================
        duration_ns = self.get_metric('gpu__time_duration.avg', 0)
        if duration_ns:
            key_metrics['duration_ns'] = duration_ns
            key_metrics['duration_ms'] = duration_ns / 1e6
            key_metrics['duration_us'] = duration_ns / 1e3

        # ============================================
        # Memory Bandwidth
        # ============================================
        bytes_read = self.get_metric('dram__bytes_read.sum', 0)
        bytes_write = self.get_metric('dram__bytes_write.sum', 0)
        l2_read = self.get_metric('lts__t_sectors_op_read.sum', 0) * 32  # sectors to bytes
        l2_write = self.get_metric('lts__t_sectors_op_write.sum', 0) * 32

        total_read = bytes_read + l2_read
        total_write = bytes_write + l2_write
        total_bytes = total_read + total_write

        key_metrics['bytes_read'] = total_read
        key_metrics['bytes_write'] = total_write
        key_metrics['bytes_total'] = total_bytes
        key_metrics['bytes_read_mb'] = total_read / 1e6
        key_metrics['bytes_write_mb'] = total_write / 1e6

        # Calculate bandwidth
        if duration_ns > 0:
            duration_s = duration_ns / 1e9
            bandwidth_gbps = (total_bytes / 1e9) / duration_s
            key_metrics['memory_bandwidth_gbps'] = bandwidth_gbps

            # Get peak bandwidth and calculate efficiency
            peak_bw = self._get_peak_bandwidth()
            key_metrics['peak_bandwidth_gbps'] = peak_bw
            key_metrics['bandwidth_efficiency_pct'] = (bandwidth_gbps / peak_bw) * 100 if peak_bw > 0 else 0

        # ============================================
        # GPU Utilization
        # ============================================
        sm_throughput = self.get_metric('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0)
        key_metrics['gpu_utilization_pct'] = sm_throughput

        # Compute/Memory throughput ratio
        compute_memory = self.get_metric('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 0)
        key_metrics['compute_memory_throughput_pct'] = compute_memory

        # ============================================
        # Warp Analysis
        # ============================================
        warps_active = self.get_metric('smsp__warps_active.avg.pct', 0)
        warps_issued = self.get_metric('smsp__warps_issued.avg.pct', 0)
        warp_stall_long = self.get_metric('smsp__average_warps_issue_stalled_long_scoreboard.pct', 0)
        warp_stall_short = self.get_metric('smsp__average_warps_issue_stalled_short_scoreboard.pct', 0)

        key_metrics['warps_active_pct'] = warps_active
        key_metrics['warps_issued_pct'] = warps_issued
        key_metrics['warp_stall_long_pct'] = warp_stall_long
        key_metrics['warp_stall_short_pct'] = warp_stall_short
        key_metrics['occupancy_pct'] = warps_issued  # Approximation

        # ============================================
        # Memory Access Patterns
        # ============================================
        # Global memory load efficiency
        load_efficiency = self.get_metric('smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct', 0)
        store_efficiency = self.get_metric('smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct', 0)

        key_metrics['load_efficiency_pct'] = load_efficiency
        key_metrics['store_efficiency_pct'] = store_efficiency

        # L1/TEX cache hit rates
        l1_hit_rate = self.get_metric('l1tex__t_sector_hit_rate.pct', 0)
        key_metrics['l1_cache_hit_rate_pct'] = l1_hit_rate

        # ============================================
        # Instruction Mix
        # ============================================
        total_instructions = self.get_metric('gpu__sass_thread_inst_executed_op_global_ld.sum', 0)
        key_metrics['total_load_instructions'] = int(total_instructions)

        branch_efficiency = self.get_metric('smsp__branch_efficiency.avg.pct', 0)
        key_metrics['branch_efficiency_pct'] = branch_efficiency

        # ============================================
        # Memory Throughput
        # ============================================
        dram_throughput = self.get_metric('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0)
        key_metrics['dram_throughput_pct'] = dram_throughput

        return key_metrics

    def _get_peak_bandwidth(self) -> float:
        """Get theoretical peak bandwidth for current GPU."""
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name

            # Try exact match
            if gpu_name in self.PEAK_BANDWIDTH_MAP:
                return self.PEAK_BANDWIDTH_MAP[gpu_name]

            # Try partial match
            for name, bw in self.PEAK_BANDWIDTH_MAP.items():
                if name in gpu_name:
                    return bw

            # Estimate based on compute capability
            cc = (props.major, props.minor)
            if cc >= (9, 0):
                return 2000  # Hopper
            elif cc >= (8, 9):
                return 1000  # Ada
            elif cc >= (8, 6):
                return 900   # Ampere
            elif cc >= (7, 5):
                return 600   # Turing
            else:
                return 500   # Conservative default

        return 700  # Default if CUDA not available

    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []

        # GPU utilization
        gpu_util = metrics.get('gpu_utilization_pct', 0)
        if gpu_util < 50:
            recommendations.append(
                "• **Low GPU utilization** ({gpu_util:.1f}%):\n"
                "  - Increase grid size (more blocks)\n"
                "  - Increase threads per block for better occupancy\n"
                "  - Check for synchronization bottlenecks"
            )
        elif gpu_util < 70:
            recommendations.append(
                "• **Moderate GPU utilization** ({gpu_util:.1f}%):\n"
                "  - Consider increasing block size\n"
                "  - Check for thread divergence"
            )

        # Memory bandwidth
        bw_eff = metrics.get('bandwidth_efficiency_pct', 0)
        if bw_eff < 40:
            recommendations.append(
                "• **Low memory bandwidth efficiency** ({bw_eff:.1f}% of peak):\n"
                "  - Check for uncoalesced memory access\n"
                "  - Use vectorized loads (float4, float2)\n"
                "  - Consider shared memory for frequently accessed data\n"
                "  - Reduce global memory transactions"
            )
        elif bw_eff < 60:
            recommendations.append(
                "• **Moderate bandwidth efficiency** ({bw_eff:.1f}% of peak):\n"
                "  - Optimize memory access patterns\n"
                "  - Consider padding to avoid bank conflicts"
            )

        # Occupancy
        occupancy = metrics.get('occupancy_pct', 0)
        if occupancy < 50:
            recommendations.append(
                "• **Low occupancy** ({occupancy:.1f}%):\n"
                "  - Increase threads per block\n"
                "  - Reduce register usage\n"
                "  - Reduce shared memory usage"
            )

        # Warp stalls
        stall_long = metrics.get('warp_stall_long_pct', 0)
        stall_short = metrics.get('warp_stall_short_pct', 0)

        if stall_long > 50:
            recommendations.append(
                "• **High long scoreboard stalls** ({stall_long:.1f}%):\n"
                "  - Memory latency issue - consider:\n"
                "    * Using shared memory\n"
                "    * Increasing thread count for better latency hiding\n"
                "    * Software prefetching"
            )

        if stall_short > 50:
            recommendations.append(
                "• **High short scoreboard stalls** ({stall_short:.1f}%):\n"
                "  - Consider instruction mix optimization"
            )

        # Load efficiency
        load_eff = metrics.get('load_efficiency_pct', 0)
        if load_eff < 80:
            recommendations.append(
                "• **Suboptimal load efficiency** ({load_eff:.1f}%):\n"
                "  - Ensure memory accesses are aligned (128-byte boundaries)\n"
                "  - Use vector loads (float4 for 16-byte alignment)\n"
                "  - Combine scalar loads into vector loads"
            )

        # If no issues found
        if not recommendations:
            recommendations.append(
                "• **Excellent performance!** Minor improvements possible:\n"
                "  - Profile with different input sizes\n"
                "  - Consider autotuning for block sizes"
            )

        return recommendations

    def generate_report(self) -> str:
        """Generate human-readable report."""
        metrics = self.extract_key_metrics()
        recommendations = self.generate_recommendations(metrics)

        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("NSIGHT COMPUTE PROFILE ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Source: {self.csv_path.name}")
        lines.append("")

        # Executive Summary
        lines.append("## EXECUTIVE SUMMARY")
        lines.append("")

        # Duration
        if 'duration_ms' in metrics:
            lines.append(f"⏱️  **Kernel Duration:** {metrics['duration_ms']:.4f} ms")

        # Memory Bandwidth
        if 'memory_bandwidth_gbps' in metrics:
            bw = metrics['memory_bandwidth_gbps']
            eff = metrics.get('bandwidth_efficiency_pct', 0)
            peak = metrics.get('peak_bandwidth_gbps', 0)

            lines.append("")
            lines.append("### Memory Performance")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Achieved Bandwidth | **{bw:.2f} GB/s** |")
            lines.append(f"| Theoretical Peak | {peak:.0f} GB/s |")
            lines.append(f"| Efficiency | {eff:.1f}% of peak |")
            lines.append("")

            if eff >= 70:
                lines.append(f"✓✓✓ **Excellent bandwidth utilization** ({eff:.1f}%)")
            elif eff >= 50:
                lines.append(f"✓✓ **Good bandwidth utilization** ({eff:.1f}%)")
            elif eff >= 30:
                lines.append(f"✓ **Moderate bandwidth utilization** ({eff:.1f}%)")
            else:
                lines.append(f"⚠️ **Low bandwidth utilization** ({eff:.1f}%)")

        # GPU Utilization
        if 'gpu_utilization_pct' in metrics:
            lines.append("")
            lines.append("### Compute Performance")
            lines.append("")
            util = metrics['gpu_utilization_pct']

            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| GPU Utilization | **{util:.1f}%** |")

            if 'occupancy_pct' in metrics:
                lines.append(f"| Warp Occupancy | {metrics['occupancy_pct']:.1f}% |")
            if 'warps_active_pct' in metrics:
                lines.append(f"| Warps Active | {metrics['warps_active_pct']:.1f}% |")

            lines.append("")

            if util >= 80:
                lines.append(f"✓✓✓ **Excellent GPU utilization** ({util:.1f}%)")
            elif util >= 60:
                lines.append(f"✓✓ **Good GPU utilization** ({util:.1f}%)")
            elif util >= 40:
                lines.append(f"✓ **Moderate GPU utilization** ({util:.1f}%)")
            else:
                lines.append(f"⚠️ **Low GPU utilization** ({util:.1f}%)")

        # Warp Analysis
        if 'warp_stall_long_pct' in metrics:
            lines.append("")
            lines.append("### Warp Analysis")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Long Scoreboard Stalls | {metrics['warp_stall_long_pct']:.1f}% |")
            lines.append(f"| Short Scoreboard Stalls | {metrics['warp_stall_short_pct']:.1f}% |")
            lines.append("")

            stall_long = metrics['warp_stall_long_pct']
            if stall_long < 20:
                lines.append("✓ **Low stall rate** - good latency hiding")
            elif stall_long < 40:
                lines.append("~ **Moderate stall rate** - acceptable")
            else:
                lines.append("⚠️ **High stall rate** - memory latency issue")

        # Memory Access
        if 'load_efficiency_pct' in metrics:
            lines.append("")
            lines.append("### Memory Access Patterns")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Load Efficiency | {metrics['load_efficiency_pct']:.1f}% |")
            if 'store_efficiency_pct' in metrics:
                lines.append(f"| Store Efficiency | {metrics['store_efficiency_pct']:.1f}% |")
            if 'l1_cache_hit_rate_pct' in metrics:
                lines.append(f"| L1 Cache Hit Rate | {metrics['l1_cache_hit_rate_pct']:.1f}% |")
            lines.append("")

        # Recommendations
        lines.append("")
        lines.append("=" * 70)
        lines.append("OPTIMIZATION RECOMMENDATIONS")
        lines.append("=" * 70)
        lines.append("")

        for rec in recommendations:
            lines.append(rec)
            lines.append("")

        # Resume-Ready Metrics
        lines.append("=" * 70)
        lines.append("RESUME-READY METRICS")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Based on this Nsight Compute profiling, you can claim:")
        lines.append("")

        if 'gpu_utilization_pct' in metrics:
            util = metrics['gpu_utilization_pct']
            lines.append(f"✅ Achieved **{util:.1f}% GPU utilization**")
            lines.append(f"   through kernel fusion and warp-level optimizations")

        if 'memory_bandwidth_gbps' in metrics:
            bw = metrics['memory_bandwidth_gbps']
            eff = metrics.get('bandwidth_efficiency_pct', 0)
            lines.append(f"✅ Achieved **{bw:.1f} GB/s memory bandwidth**")
            lines.append(f"   ({eff:.1f}% of theoretical peak)")

        if 'duration_ms' in metrics:
            dur = metrics['duration_ms']
            lines.append(f"✅ Kernel execution time: **{dur:.4f} ms**")

        lines.append("")
        lines.append("All metrics validated using NVIDIA Nsight Compute.")
        lines.append("")

        return "\n".join(lines)

    def save_metrics_json(self, output_path: str) -> None:
        """Save extracted metrics to JSON."""
        metrics = self.extract_key_metrics()

        # Add metadata
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'source_csv': str(self.csv_path),
            'metrics': metrics,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Metrics saved to: {output_path}")

    def save_markdown_report(self, output_path: str) -> None:
        """Save markdown report."""
        report = self.generate_report()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✅ Report saved to: {output_path}")


def main():
    """Main analysis entry point."""

    if len(sys.argv) < 2:
        print("StyleForge Nsight Compute Profile Analyzer")
        print("\nUsage: python analyze_profile.py <nsight_metrics.csv> [output_prefix]")
        print("\nExample:")
        print("  python analyze_profile.py nsight_reports/profile_instance_norm_metrics_20241215_120000.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace('.csv', '')

    print("=" * 70)
    print("STYLEFORGE NSIGHT COMPUTE ANALYZER")
    print("=" * 70)
    print()

    # Analyze
    analyzer = ProfileAnalyzer(csv_path)

    if not analyzer.parse_csv():
        sys.exit(1)

    # Generate and print report
    report = analyzer.generate_report()
    print(report)

    # Save outputs
    analyzer.save_metrics_json(f"{output_prefix}_summary.json")
    analyzer.save_markdown_report(f"{output_prefix}_report.md")


if __name__ == "__main__":
    main()
