# Pre-compiled CUDA Kernels

Place your pre-compiled `.so` or `.pyd` files in this directory.

## For local development:

Copy your compiled kernel files here. They will be loaded automatically.

```bash
# Example: copy your compiled kernel
cp /path/to/fused_instance_norm.so kernels/prebuilt/
```

## For Hugging Face deployment:

The GitHub workflow **removes** this directory before pushing to Hugging Face
to avoid binary file rejection. Instead:

1. Upload kernels to `oliau/styleforge-kernels` dataset
2. Use `upload_kernels_to_dataset.py` script

The kernels will be downloaded at runtime on Hugging Face Spaces.
