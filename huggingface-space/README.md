---
title: StyleForge
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
license: mit
---

# StyleForge: Real-Time Neural Style Transfer

Transform your photos into artwork using fast neural style transfer with custom CUDA kernel acceleration.

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/olivialiau/styleforge)
[![GitHub](https://img.shields.io/badge/GitHub-StyleForge-blue?logo=github)](https://github.com/olivialiau/StyleForge)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

## Overview

StyleForge is a high-performance neural style transfer application that combines cutting-edge machine learning with custom GPU optimization. It demonstrates end-to-end ML pipeline development, from model architecture to CUDA kernel optimization and web deployment.

### Key Features

| Feature | Description |
|---------|-------------|
| **4 Pre-trained Styles** | Candy, Mosaic, Rain Princess, Udnie |
| **AI-Powered Segmentation** 🆕 | Automatic foreground/background detection using U²-Net |
| **VGG19 Style Extraction** 🆕 | Real style extraction using neural feature matching |
| **Style Blending** | Interpolate between styles in latent space |
| **Region Transfer** | Apply different styles to different image regions |
| **Real-time Webcam** | Live video style transformation |
| **CUDA Acceleration** | 8-9x faster with custom fused kernels |
| **Performance Dashboard** | Live charts comparing backends |

## Quick Start

1. **Upload** any image (JPG, PNG, WebP)
2. **Select** an artistic style
3. **Choose** your backend (Auto recommended)
4. **Click** "Stylize Image"
5. **Download** your result!

---

## Features Guide

### 1. Quick Style Transfer

The fastest way to transform your images.

- **Side-by-side comparison**: See original and stylized versions together
- **Watermark option**: Add branding for social sharing
- **Backend selection**: Choose between CUDA Kernels (fastest) or PyTorch (compatible)

### 2. Style Blending

Mix two styles together to create unique artistic combinations.

**How it works**: Style blending interpolates between model weights in the latent space.

- Blend ratio 0% = Pure Style 1
- Blend ratio 50% = Equal mix of both styles
- Blend ratio 100% = Pure Style 2

This demonstrates that neural styles exist in a continuous manifold where you can navigate between artistic styles.

### 3. Region Transfer 🆕

Apply different styles to different parts of your image using **AI-powered segmentation**.

**Mask Types**:
| Mask | Description | Use Case |
|------|-------------|----------|
| **AI: Foreground** | Automatically detect main subject | Portraits, product photos |
| **AI: Background** | Automatically detect background | Sky replacement, effects |
| Horizontal Split | Top/bottom division | Sky vs landscape |
| Vertical Split | Left/right division | Portrait effects |
| Center Circle | Circular focus region | Spotlight subjects |
| Corner Box | Top-left quadrant only | Creative framing |
| Full | Entire image | Standard transfer |

**AI Segmentation**: Uses the U²-Net deep learning model for automatic subject detection without manual masking.

### 4. Create Style 🆕

**Extract** artistic style from any image using **VGG19 neural feature matching**.

**How it works**:
1. Upload an artwork image (painting, illustration, photo with artistic style)
2. VGG19 pre-trained network extracts style features (textures, colors, patterns)
3. A transformation network is fine-tuned to match those features
4. Your custom style model is saved and available in all tabs

This is **real style extraction** - the system learns the artistic characteristics from your image, not just copying an existing style.

**Tips for best results**:
- Use artwork with clear artistic direction (paintings, illustrations, stylized photos)
- Higher iterations = better style matching (but slower)
- GPU is recommended for training (100 iterations ≈ 30-60 seconds)

### 5. Webcam Live

Real-time style transfer on your webcam feed.

**Requirements**:
- Browser camera permissions
- Recommended: GPU device for smooth performance

**Performance**:
- GPU: 20-30 FPS
- CPU: 5-10 FPS

### 6. Performance Dashboard

Monitor and compare inference performance across backends.

**Metrics tracked**:
- Inference time per image
- Average/min/max times
- Backend comparison (CUDA vs PyTorch)
- Speedup calculations

---

## Deep Dive: New AI Features 🆕

### AI-Powered Segmentation (U²-Net)

**Overview**: StyleForge now uses the U²-Net (U-shape 2-level U-Net) deep learning model for automatic foreground/background segmentation. This eliminates the need for manual masking when applying different styles to specific image regions.

#### How U²-Net Works

```
Input Image (any size)
    ↓
┌─────────────────────────────────┐
│  Encoder (U-Net style)           │
│  - Extracts multi-scale features  │
│  - 6 encoder stages               │
│  - Deep supervision paths        │
├─────────────────────────────────┤
│  Decoder                          │
│  - Reconstructs segmentation mask  │
│  - Salient object detection      │
└─────────────────────────────────┘
    ↓
Binary Mask (256 levels)
    ↓
Foreground (white) / Background (black)
```

**Technical Details**:
- **Architecture**: U²-Net with a deep encoder-decoder structure
- **Input**: RGB image of any size
- **Output**: Grayscale mask where white = foreground, black = background
- **Model Size**: ~176 MB pre-trained weights
- **Inference Time**: ~200-500ms per image (CPU), ~50-100ms (GPU)

**Why U²-Net?**
- Trained on 20,000+ images with diverse subjects
- Excellent at detecting humans, animals, objects, and products
- Handles complex backgrounds and edges
- Works without requiring bounding boxes or user input

**Use Cases**:
- **Portrait Photography**: Style the subject differently from the background
- **Product Photography**: Apply artistic effects to products while keeping clean backgrounds
- **Creative Composites**: Apply different artistic styles to foreground vs background

#### Gram Matrices: Representing Style

The Gram matrix is computed from the feature activations:

```
F = feature map of shape (C, H, W)
Gram(F)[i,j] = Σ_k F[i,k] ⋅ F[j,k]
```

This captures:
- **Texture information**: How features correlate spatially
- **Color patterns**: Which colors appear together
- **Brush strokes**: Directionality and scale of textures
- **Style signature**: Unique fingerprint of the artistic style

#### Fine-Tuning Process

The system fine-tunes a pre-trained Fast Style Transfer model:

1. **Load base model** (e.g., Udnie style)
2. **Freeze early layers** (preserve low-level transformations)
3. **Train on style loss** using the extracted Gram matrices
4. **Iterate** with Adam optimizer (lr=0.001)
5. **Save** as a reusable `.pth` file

```
Base Model → Extracted Style Features → Fine-tuned Model
   ↓              ↓                        ↓
 Udnie        Starry Night          Custom "Starry Udnie"
```

**Training Time**:
- 100 iterations: ~30-60 seconds (GPU)
- 200 iterations: ~60-120 seconds (GPU)
- More iterations = better style matching

**Why VGG19?**
- Pre-trained on ImageNet (1M+ images)
- Learned rich feature representations
- Standard in style transfer research (Gatys et al., Johnson et al.)
- Captures both low-level (textures) and high-level (patterns) features

---

## Technical Details

### Architecture

StyleForge uses the **Fast Neural Style Transfer** architecture from Johnson et al.:

```
Input Image (3 x H x W)
    ↓
┌─────────────────────────────────┐
│  Encoder (3 Conv + InstanceNorm) │
├─────────────────────────────────┤
│  Transformer (5 Residual Blocks) │
├─────────────────────────────────┤
│  Decoder (3 Upsample + InstanceNorm) │
└─────────────────────────────────┘
    ↓
Output Image (3 x H x W)
```

**Layers**:
- **ConvLayer**: Conv2d → InstanceNorm → ReLU
- **ResidualBlock**: Two ConvLayers with skip connection
- **UpsampleConvLayer**: Upsample → Conv2d → InstanceNorm → ReLU

### CUDA Kernel Optimization

Custom CUDA kernels provide 8-9x speedup over PyTorch baseline.

**Fused InstanceNorm Kernel**:
- Combines mean, variance, normalization, and affine transform into single kernel
- Uses `float4` vectorized loads for 4x memory bandwidth
- Warp-level parallel reductions
- Shared memory tiling for reduced global memory traffic

**Performance Comparison** (512x512 image):

| Backend | Time | Speedup |
|---------|------|---------|
| PyTorch | ~80ms | 1.0x |
| CUDA Kernels | ~10ms | 8.0x |

### ML Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **Style Transfer** | Neural artistic stylization |
| **Latent Space** | Style blending shows continuous style space |
| **Conditional Generation** | Region-based style application |
| **Transfer Learning** | Custom styles from base models |
| **Performance Optimization** | CUDA kernels, JIT compilation, caching |
| **Model Deployment** | Gradio web interface, CI/CD pipeline |

---

## Styles Gallery

| Style | Description | Best For |
|-------|-------------|----------|
| 🍬 **Candy** | Bright, colorful pop-art transformation | Portraits, vibrant scenes |
| 🎨 **Mosaic** | Fragmented tile-like reconstruction | Landscapes, architecture |
| 🌧️ **Rain Princess** | Moody impressionistic style | Moody, atmospheric photos |
| 🖼️ **Udnie** | Bold abstract expressionist | High-contrast images |

---

## Performance Benchmarks

### Inference Time (milliseconds)

| Resolution | CUDA | PyTorch | Speedup |
|------------|------|---------|---------|
| 256x256 | 5ms | 40ms | 8.0x |
| 512x512 | 10ms | 80ms | 8.0x |
| 1024x1024 | 35ms | 280ms | 8.0x |

### FPS Performance (Webcam)

| Device | Resolution | FPS |
|--------|------------|-----|
| NVIDIA GPU | 640x480 | 25-30 |
| CPU (Modern) | 640x480 | 5-10 |

---

## Run Locally

### Using pip

```bash
git clone https://github.com/olivialiau/StyleForge
cd StyleForge/huggingface-space
pip install -r requirements.txt
python app.py
```

### Using conda (recommended)

```bash
git clone https://github.com/olivialiau/StyleForge
cd StyleForge/huggingface-space
conda env create -f environment.yml
conda activate styleforge
python app.py
```

Open http://localhost:7860 in your browser.

---

## API Usage

You can use StyleForge programmatically:

```python
import requests
from PIL import Image
from io import BytesIO

# Prepare image
img = Image.open("path/to/image.jpg")

# Call API
response = requests.post(
    "https://olivialiau-styleforge.hf.space/api/predict",
    json={
        "data": [
            {"name": "image.jpg", "data": "base64_encoded_image"},
            "candy",  # style
            "auto",   # backend
            False,    # show_comparison
            False     # add_watermark
        ]
    }
)

result = response.json()
output_img = Image.open(BytesIO(base64.b64decode(result["data"][0])))
```

---

## Embed in Your Website

```html
<iframe
  src="https://olivialiau-styleforge.hf.space"
  frameborder="0"
  width="100%"
  height="850"
  allow="camera; microphone"
></iframe>
```

---

## Project Structure

```
StyleForge/
├── huggingface-space/
│   ├── app.py                 # Main Gradio application
│   ├── requirements.txt       # Python dependencies
│   ├── README.md             # This file
│   ├── kernels/              # Custom CUDA kernels
│   │   ├── __init__.py
│   │   ├── cuda_build.py     # JIT compilation utilities
│   │   ├── instance_norm_wrapper.py
│   │   └── instance_norm.cu  # CUDA source code
│   ├── models/               # Model weights (auto-downloaded)
│   └── custom_styles/        # User-trained styles
├── .github/
│   └── workflows/
│       └── deploy-huggingface.yml  # CI/CD pipeline
└── saved_models/            # Local model cache
```

---

## Development

### CI/CD Pipeline

The project uses GitHub Actions for automatic deployment to Hugging Face Spaces:

```yaml
# .github/workflows/deploy-huggingface.yml
on:
  push:
    branches: [main]
    paths: ['huggingface-space/**']
```

Push to `main` branch → Auto-deploys to Hugging Face Space.

### Adding New Styles

1. Train a model using the original repo's training script
2. Save weights as `.pth` file
3. Add to `models/` directory or update URL map in `get_model_path()`
4. Add entry to `STYLES` and `STYLE_DESCRIPTIONS` dictionaries

---

## FAQ

**Q: How does the style extraction work?**

A: The new VGG19-based style extraction uses a pre-trained neural network to analyze artistic features (textures, brush strokes, color patterns) from your artwork. It then fine-tunes a transformation network to reproduce those features. This is the same technique used in the original neural style transfer research.

**Q: What's the difference between backends?**

A:
- **Auto**: Uses CUDA if available, otherwise PyTorch
- **CUDA Kernels**: Fastest, requires GPU and compilation
- **PyTorch**: Compatible fallback, works on CPU

**Q: Can I use this commercially?**

A: Yes! StyleForge is MIT licensed. The pre-trained models are from the fast-neural-style-transfer repo.

**Q: How large can my input image be?**

A: Any size, but larger images take longer. Webcam mode auto-scales to 640px max dimension for performance.

**Q: Why does compilation take time on first run?**

A: CUDA kernels are JIT-compiled on first use. This only happens once per session.

---

## Acknowledgments

- [Johnson et al.](https://arxiv.org/abs/1603.08155) - Perceptual Losses for Real-Time Style Transfer
- [yakhyo/fast-neural-style-transfer](https://github.com/yakhyo/fast-neural-style-transfer) - Pre-trained model weights
- [Rembg](https://github.com/danielgatis/rembg) - AI background removal (U²-Net)
- [VGG19](https://pytorch.org/vision/stable/models.html) - Pre-trained feature extractor for style extraction
- [Hugging Face](https://huggingface.co) - Spaces hosting platform
- [Gradio](https://gradio.app) - UI framework
- [PyTorch](https://pytorch.org) - Deep learning framework

---

## Author

**Olivia** - USC Computer Science

[GitHub](https://github.com/olivialiau/StyleForge)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Made with ❤️ and CUDA
