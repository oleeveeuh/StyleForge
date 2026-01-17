# StyleForge Verification Report Template

This template is used by the test runner to generate verification reports.

## Report Contents

When `run_verification_tests.py` is executed, it generates:

1. **verification_report.json** - Machine-readable JSON report
2. **verification_report.md** - Human-readable Markdown report
3. **test_outputs/** - Directory containing generated test images

## Test Modules

| Module | Description |
|--------|-------------|
| `test_model_loading` | Tests for loading pre-trained weights and architecture verification |
| `test_forward_pass` | Tests for correct output shapes and value ranges |
| `test_visual_quality` | Tests generating stylized images for manual inspection |
| `test_cuda_kernel_usage` | Tests for CUDA kernel loading and execution |
| `test_numerical_accuracy` | Tests for numerical stability and accuracy |
| `test_memory_leaks` | Tests for memory management |

## Success Criteria

A successful verification requires:

- [x] **All automated tests pass** - No test failures
- [ ] **Visual outputs look reasonable** - Manual inspection of `test_outputs/` images
- [ ] **Custom kernels are running** - Confirmed via CUDA kernel tests (if applicable)
- [ ] **No numerical issues** - No NaN, Inf, or extreme values
- [ ] **No memory leaks** - Memory growth within acceptable bounds

## Running Tests

```bash
# Run all tests
python run_verification_tests.py

# Run specific test module
python run_verification_tests.py --test model_loading

# Run with specific style
python run_verification_tests.py --style candy

# Skip visual tests (faster)
python run_verification_tests.py --skip-visual

# Run individual test file directly
python tests/test_model_loading.py --style candy
```

## Test Results Interpretation

### Output Symbols

- `✓` or `✅` - Test passed
- `❌` - Test failed
- `⚠️` - Warning or skipped test

### Common Issues

1. **Checkpoint not found**
   - Error: `Checkpoint not found: models/pretrained/{style}.pth`
   - Solution: Run `python download_models.py --all`

2. **CUDA not available**
   - Warning: `CUDA not available, skipping`
   - This is expected on CPU-only systems
   - CUDA-specific tests will be skipped

3. **Memory leak detected**
   - Error: `Memory leak detected: X MB growth`
   - May indicate a bug in memory management
   - Verify on your system as memory tracking can vary

## Manual Inspection Checklist

After running tests, manually inspect the generated images:

1. **Original images** - Reference inputs in `test_outputs/*_original.jpg`
2. **Stylized images** - Check each style output:
   - Colors are vibrant and match style expectations
   - No artifacts, corruption, or strange patterns
   - Edges and details are preserved reasonably
   - Overall aesthetic quality is acceptable

3. **Edge case images** - Check `test_outputs/edge_*.jpg`:
   - Solid colors are handled correctly
   - Gradients and patterns transfer properly

## Report File Format

### JSON Report Structure

```json
{
  "styleforge_verification_report": {
    "version": "0.1.0",
    "timestamp": "2024-01-01T12:00:00",
    "device": "cuda",
    "cuda_available": true,
    "results": {
      "summary": {
        "model_loading": {
          "passed": 5,
          "failed": 0,
          "skipped": 0
        },
        ...
      },
      "total_duration": 45.2,
      "start_time": "2024-01-01T12:00:00",
      "end_time": "2024-01-01T12:00:45"
    }
  }
}
```

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run verification tests
  run: |
    python download_models.py --style candy
    python run_verification_tests.py --skip-visual

- name: Upload test results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test_outputs/
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-01-17 | Initial test suite implementation |
