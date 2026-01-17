#!/bin/bash
#
# NVIDIA Nsight Compute profiling script for StyleForge CUDA kernels
#
# Usage: ./profile.sh [kernel_name] [options]
#
# Kernel names:
#   instance_norm    - Profile InstanceNorm kernel
#   conv_fusion      - Profile Conv+InstanceNorm+ReLU kernel
#   ffn              - Profile FFN kernel
#   pytorch_baseline  - Profile PyTorch baseline
#   all              - Profile all kernels
#
# Options:
#   --quick          - Run quick profiling (basic metrics only)
#   --full           - Run full profiling (default)
#   --memory         - Memory bandwidth focused profiling
#   --compute        - Compute utilization focused profiling
#
# Examples:
#   ./profile.sh instance_norm
#   ./profile.sh instance_norm --memory
#   ./profile.sh all --quick

set -e

# ============================================================================
# Configuration
# ============================================================================

KERNEL_MODE=${1:-instance_norm}
OUTPUT_DIR="nsight_reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse options
PROFILE_MODE="full"
for arg in "$@"; do
    case $arg in
        --quick)
            PROFILE_MODE="quick"
            shift
            ;;
        --full)
            PROFILE_MODE="full"
            shift
            ;;
        --memory)
            PROFILE_MODE="memory"
            shift
            ;;
        --compute)
            PROFILE_MODE="compute"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Check Nsight Compute
# ============================================================================

echo "========================================================================"
echo "NVIDIA Nsight Compute Profiling - StyleForge"
echo "========================================================================"
echo ""
echo "Kernel: $KERNEL_MODE"
echo "Profile mode: $PROFILE_MODE"
echo "Output directory: $OUTPUT_DIR"
echo ""

if ! command -v ncu &> /dev/null; then
    echo "❌ Nsight Compute (ncu) not found"
    echo ""
    echo "Install from: https://developer.nvidia.com/nsight-compute"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt-get install nvidia-nsight-compute"
    echo ""
    exit 1
fi

NCU_VERSION=$(ncu --version 2>&1 | head -1)
echo "✓ Nsight Compute found: $NCU_VERSION"
echo ""

# ============================================================================
# Base NCU Command
# ============================================================================

NCU_BASE="ncu"
NCU_OUTPUT="$OUTPUT_DIR/profile_${KERNEL_MODE}_${PROFILE_MODE}_${TIMESTAMP}"

# Add common flags
NCU_CMD="$NCU_BASE --export $NCU_OUTPUT"

# Select mode
case $PROFILE_MODE in
    quick)
        NCU_CMD="$NCU_CMD --set basic"
        echo "Running quick profiling (basic metrics)..."
        ;;
    full)
        NCU_CMD="$NCU_CMD --set full"
        echo "Running full profiling (all metrics)..."
        ;;
    memory)
        NCU_CMD="$NCU_CMD --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.avg,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
        echo "Running memory bandwidth profiling..."
        ;;
    compute)
        NCU_CMD="$NCU_CMD --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,smsp__average_warps_issue_stalled_long_scoreboard.pct,smsp__average_warps_issue_stalled_short_scoreboard.pct,smsp__warps_active.avg.pct,smsp__warps_issued.avg.pct"
        echo "Running compute utilization profiling..."
        ;;
esac

echo ""

# ============================================================================
# Run Profiling
# ============================================================================

echo "========================================================================"
echo "Running Profiling"
echo "========================================================================"
echo ""

# Run ncu with the profiling script
$NCU_CMD python profile_kernels.py $KERNEL_MODE

echo ""
echo "✅ Profiling complete!"
echo ""

# ============================================================================
# Generate CSV Report
# ============================================================================

echo "========================================================================"
echo "Generating CSV Report"
echo "========================================================================"

CSV_OUTPUT="$OUTPUT_DIR/profile_${KERNEL_MODE}_metrics_${TIMESTAMP}.csv"

ncu \
    --csv \
    --log-file "$CSV_OUTPUT" \
    --set full \
    python profile_kernels.py $KERNEL_MODE > /dev/null 2>&1

echo "✅ CSV report saved to: $CSV_OUTPUT"
echo ""

# ============================================================================
# Print Summary
# ============================================================================

echo "========================================================================"
echo "Profiling Summary"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  1. $NCU_OUTPUT.ncu-rep  (Nsight Compute report)"
echo "  2. $CSV_OUTPUT           (CSV metrics)"
echo ""
echo ""
echo "To view reports in Nsight Compute UI:"
echo "  ncu-ui $NCU_OUTPUT.ncu-rep"
echo ""
echo ""
echo "To analyze metrics:"
echo "  python analyze_profile.py $CSV_OUTPUT"
echo ""

# ============================================================================
# Auto-analyze if analyze_profile.py exists
# ============================================================================

if [ -f "analyze_profile.py" ]; then
    echo "========================================================================"
    echo "Auto-Analyzing Results"
    echo "========================================================================"
    echo ""

    python analyze_profile.py "$CSV_OUTPUT"
    echo ""
fi

# ============================================================================
# Done
# ============================================================================

echo "========================================================================"
echo "Profiling Complete!"
echo "========================================================================"
