# Running StyleForge in Colab via VS Code

## Using the VS Code Colab Extension

1. **Install the extension**
   - Search "Google Colab" in VS Code extensions
   - Install the official Google Colab extension

2. **Open the notebook**
   - Open `notebooks/demo.ipynb` in VS Code
   - Click the Colab icon in the top right to run on Colab

3. **First cell adds** (automatic when running on Colab)

Add this as the first cell in the notebook - it will clone/setup the repo automatically:

```python
# ============================================
# 📦 Colab Setup (runs only on Colab)
# ============================================

import os
import sys

# Check if running on Colab
if 'google.colab' in str(get_ipython()):
    print("🔄 Running on Colab - setting up environment...")

    # Clone repository (replace with your URL)
    !git clone https://github.com/yourusername/styleforge.git /content/StyleForge
    %cd /content/StyleForge

    # Install dependencies
    !pip install -q torch torchvision numpy matplotlib seaborn pybind11
    !pip install -q jupyter

    # Verify CUDA
    !nvidia-smi

    print("✅ Setup complete!")
else:
    print("🖥️  Running locally - using project files")
    # Add project root to path for local imports
    from pathlib import Path
    project_root = Path().absolute().parent
    while (project_root / 'models').exists() is False:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))
```

## Quick Start

1. Open `notebooks/demo.ipynb`
2. Click "Open in Colab" (VS Code extension)
3. Run all cells sequentially

## Notebook Structure

| Cell | Purpose |
|------|---------|
| 0 | Colab setup (clone repo, install deps) |
| 1 | Imports & path setup |
| 2 | CUDA environment check |
| 3 | Build & test baseline model |
| 4 | Run baseline benchmarks |
| 5 | Test CUDA kernel compilation |
| 6 | Test fused attention kernel |
| 7 | Progress summary |

## Tips

- **GPU Runtime**: Make sure Colab is set to GPU runtime
- **First Run**: CUDA kernels take 30-60s to compile on first run
- **Persisting Changes**: Commit changes to GitHub to save across sessions
