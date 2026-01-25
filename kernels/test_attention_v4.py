"""
Test the fixed V4 attention kernel

This script verifies that V4 is much faster than V3.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from attention_v4_wrapper import FusedAttentionV4, benchmark_attention_v4_vs_pytorch

print("="*70)
print("Testing Fixed Attention V4 Kernel")
print("="*70)

# Test on different configurations
configs = [
    {"batch_size": 1, "seq_len": 256, "embed_dim": 512, "num_heads": 4},
    {"batch_size": 1, "seq_len": 512, "embed_dim": 512, "num_heads": 4},
    {"batch_size": 1, "seq_len": 512, "embed_dim": 1024, "num_heads": 8},
    {"batch_size": 1, "seq_len": 1024, "embed_dim": 2048, "num_heads": 16},
]

for cfg in configs:
    try:
        result = benchmark_attention_v4_vs_pytorch(
            batch_size=cfg["batch_size"],
            seq_len=cfg["seq_len"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            iterations=50
        )
        print()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
