"""
Image loading, saving, and preprocessing utilities for StyleForge.

Supports:
- Loading images from files or URLs
- Resizing to multiple resolutions (256, 512, 1024)
- Converting between PIL, Tensor, and numpy formats
- Saving stylized results
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# Default preprocessing for Fast Style Transfer
# Normalizes to [0, 1] range (network output is also in this range)
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0, 1]
])

# Inverse transform for converting tensor back to PIL
INV_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
])


def load_image(
    path: Optional[Union[str, Path]] = None,
    url: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    size: Optional[int] = None,
    square: bool = False,
) -> Image.Image:
    """
    Load an image from file path, URL, or bytes.

    Args:
        path: Path to image file
        url: URL to download image from
        file_bytes: Raw image bytes (e.g., from file upload)
        size: If provided, resize to (size, size) or maintain aspect ratio
        square: If True, crop to square before resizing

    Returns:
        PIL Image in RGB format

    Example:
        >>> img = load_image("photo.jpg", size=512)
        >>> img = load_image(url="https://example.com/image.jpg", size=256)
    """
    if path:
        img = Image.open(path).convert('RGB')
    elif url:
        import requests
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
    elif file_bytes:
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    else:
        raise ValueError("Must provide one of: path, url, or file_bytes")

    # Handle resizing
    if size is not None:
        if square:
            # Center crop to square, then resize
            min_dim = min(img.size)
            left = (img.width - min_dim) // 2
            top = (img.height - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        else:
            # Maintain aspect ratio
            aspect = img.width / img.height
            if aspect > 1:
                new_width = size
                new_height = int(size / aspect)
            else:
                new_width = int(size * aspect)
                new_height = size
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def preprocess_image(
    img: Image.Image,
    size: Optional[int] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Convert PIL Image to tensor for model input.

    Args:
        img: PIL Image
        size: Optional resize dimension
        normalize: If True, normalize to [-1, 1], else [0, 1]

    Returns:
        Tensor of shape (1, 3, H, W)
    """
    if size is not None:
        img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Convert to tensor [0, 1]
    tensor = DEFAULT_TRANSFORM(img).unsqueeze(0)

    # Optionally normalize to [-1, 1]
    if normalize:
        tensor = tensor * 2 - 1

    return tensor


def postprocess_image(
    tensor: torch.Tensor,
    denormalize: bool = False,
) -> Image.Image:
    """
    Convert model output tensor to PIL Image.

    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W)
        denormalize: If True, convert from [-1, 1] to [0, 1]

    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Clamp and denormalize if needed
    if denormalize:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1)
    else:
        tensor = torch.clamp(tensor, 0, 1)

    return INV_TRANSFORM(tensor)


def save_image(
    img: Union[Image.Image, torch.Tensor],
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save image to file.

    Args:
        img: PIL Image or Tensor
        path: Output file path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img, torch.Tensor):
        img = postprocess_image(img)

    img.save(path, quality=quality, subsampling=0)


def tensor_to_numpy(
    tensor: torch.Tensor,
    denormalize: bool = False,
) -> np.ndarray:
    """
    Convert tensor to numpy array for visualization.

    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W)
        denormalize: If True, convert from [-1, 1] to [0, 1]

    Returns:
        numpy array of shape (H, W, 3) in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    if denormalize:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1)
    else:
        tensor = torch.clamp(tensor, 0, 1)

    return (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def resize_with_padding(
    img: Image.Image,
    target_size: int,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Resize image to fit within target_size while maintaining aspect ratio.
    Pads with background_color to fill the target size.

    Args:
        img: PIL Image
        target_size: Target width/height
        background_color: RGB tuple for padding

    Returns:
        PIL Image of size (target_size, target_size)
    """
    original_width, original_height = img.size

    # Calculate scaling factor
    scale = target_size / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize image
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with padding
    new_img = Image.new('RGB', (target_size, target_size), background_color)

    # Paste resized image centered
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    new_img.paste(img, (offset_x, offset_y))

    return new_img


def get_image_size_info(
    path: Union[str, Path],
) -> dict:
    """
    Get image information without loading the full image.

    Args:
        path: Path to image file

    Returns:
        Dict with width, height, mode, format, size_bytes
    """
    with Image.open(path) as img:
        return {
            'width': img.width,
            'height': img.height,
            'mode': img.mode,
            'format': img.format,
            'size_bytes': Path(path).stat().st_size,
        }


# Predefined sizes for different use cases
SIZE_SMALL = 256    # Fast processing
SIZE_MEDIUM = 512   # Balanced quality/speed
SIZE_LARGE = 1024   # High quality
SIZE_HD = 2048      # Very high quality
