"""
Thorough numerical validation of custom attention kernel

Tests edge cases and numerical stability across various configurations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from configs.llama2_7b import LLAMA2_7B
from models.custom_attention import CustomMultiHeadAttention, create_pytorch_baseline_attention
from models.utils import validate_attention_output


def test_numerical_stability():
    """Test numerical stability with extreme values"""

    print("="*70)
    print("Numerical Stability Tests")
    print("="*70)

    config = LLAMA2_7B
    seq_len = 512

    custom_attn = CustomMultiHeadAttention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
    ).cuda().eval()

    pytorch_attn = create_pytorch_baseline_attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
    ).cuda().eval()

    test_cases = [
        ("Normal distribution", lambda: torch.randn(1, seq_len, config.hidden_size).cuda()),
        ("Large values", lambda: torch.randn(1, seq_len, config.hidden_size).cuda() * 100),
        ("Small values", lambda: torch.randn(1, seq_len, config.hidden_size).cuda() * 0.01),
        ("Mixed values", lambda: torch.cat([
            torch.randn(1, seq_len//2, config.hidden_size).cuda() * 100,
            torch.randn(1, seq_len//2, config.hidden_size).cuda() * 0.01
        ], dim=1)),
        ("All zeros", lambda: torch.zeros(1, seq_len, config.hidden_size).cuda()),
        ("All ones", lambda: torch.ones(1, seq_len, config.hidden_size).cuda()),
    ]

    print(f"\nTesting {len(test_cases)} cases...")
    print("-"*70)

    all_passed = True

    for name, input_fn in test_cases:
        hidden_states = input_fn()

        with torch.no_grad():
            custom_output = custom_attn(hidden_states)
            pytorch_output, _ = pytorch_attn(hidden_states, hidden_states, hidden_states)

        is_close, max_error, mean_error = validate_attention_output(
            custom_output, pytorch_output
        )

        status = "PASS" if is_close else "FAIL"
        print(f"{name:<20} {status}  (max error: {max_error:.2e})")

        if not is_close:
            all_passed = False

    print("-"*70)
    if all_passed:
        print("All numerical stability tests passed!")
    else:
        print(" Some tests failed - review kernel implementation")

    return all_passed


def test_correctness_small():
    """Test correctness on a small, verifiable case"""

    print("\n" + "="*70)
    print("Small-Scale Correctness Test")
    print("="*70)

    # Small config for easy verification
    hidden_size = 128
    num_heads = 4
    seq_len = 16

    custom_attn = CustomMultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    ).cuda().eval()

    pytorch_attn = create_pytorch_baseline_attention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    ).cuda().eval()

    # Create input
    hidden_states = torch.randn(1, seq_len, hidden_size).cuda()

    with torch.no_grad():
        custom_output = custom_attn(hidden_states)
        pytorch_output, _ = pytorch_attn(hidden_states, hidden_states, hidden_states)

    is_close, max_error, mean_error = validate_attention_output(
        custom_output, pytorch_output, rtol=1e-3, atol=1e-5
    )

    print(f"\nOutput shape: {custom_output.shape}")
    print(f"Max error: {max_error:.2e}")
    print(f"Mean error: {mean_error:.2e}")

    if is_close:
        print("PASS: Outputs match within tolerance")
    else:
        print("FAIL: Outputs differ")

    return is_close


def test_sequence_lengths():
    """Test across different sequence lengths"""

    print("\n" + "="*70)
    print("Sequence Length Sweep")
    print("="*70)

    config = LLAMA2_7B
    seq_lengths = [128, 256, 512, 1024, 2048]

    custom_attn = CustomMultiHeadAttention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
    ).cuda().eval()

    pytorch_attn = create_pytorch_baseline_attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
    ).cuda().eval()

    print(f"\n{'Seq Len':<10} {'Max Error':<15} {'Mean Error':<15} {'Status':<10}")
    print("-"*50)

    all_passed = True
    for seq_len in seq_lengths:
        try:
            hidden_states = torch.randn(1, seq_len, config.hidden_size).cuda()

            with torch.no_grad():
                custom_output = custom_attn(hidden_states)
                pytorch_output, _ = pytorch_attn(hidden_states, hidden_states, hidden_states)

            is_close, max_error, mean_error = validate_attention_output(
                custom_output, pytorch_output
            )

            status = "PASS" if is_close else "FAIL"
            print(f"{seq_len:<10} {max_error:<15.2e} {mean_error:<15.2e} {status:<10}")

            if not is_close:
                all_passed = False
        except Exception as e:
            print(f"{seq_len:<10} {'ERROR':<15} {str(e):<30}")
            all_passed = False

    print("-"*50)
    if all_passed:
        print("All sequence length tests passed!")
    else:
        print(" Some tests failed")

    return all_passed


def run_all_validation_tests():
    """Run all validation tests"""

    if not torch.cuda.is_available():
        print("CUDA not available - skipping tests")
        return False

    print("\n" + "="*70)
    print("ATTENTION KERNEL VALIDATION SUITE")
    print("="*70)

    results = {
        'numerical_stability': test_numerical_stability(),
        'small_correctness': test_correctness_small(),
        'sequence_sweep': test_sequence_lengths(),
    }

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<30} {status}")

    all_passed = all(results.values())
    print("-"*50)
    if all_passed:
        print("All validation tests passed!")
    else:
        print("Some tests failed")

    return all_passed


if __name__ == "__main__":
    run_all_validation_tests()
