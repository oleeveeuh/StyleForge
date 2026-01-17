"""
Vision Transformer-based Style Transfer Model

This module implements a ViT-based style transfer architecture that heavily utilizes
StyleForge's fused attention CUDA kernels for acceleration.

Architecture:
    Image → Patch Embedding → Positional Encoding
        → Encoder Blocks (6 blocks with custom attention)
        → Style Injection (AdaIN)
        → Decoder Blocks (6 blocks with custom attention)
        → Patch Unembedding → Stylized Image

Each transformer block uses:
    - CustomMultiheadAttention (fused_attention_v1 CUDA kernel)
    - FusedFFNWrapper (fused_ffn CUDA kernel when available)
    - Layer normalization and residual connections

Expected speedup: 8-15x on attention operations
Total attention calls per forward pass: 12 (6 encoder + 6 decoder)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.custom_attention_wrapper import (
    CustomMultiheadAttention,
    FusedFFNWrapper,
    get_attention_kernel_stats,
    print_attention_stats
)


class PatchEmbedding(nn.Module):
    """
    Convert image to sequence of patches.

    Args:
        image_size: Size of input image (H, W)
        patch_size: Size of each patch (P, P)
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension for patches
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Conv2d for patch embedding: more efficient than manual unfolding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, H, W]

        Returns:
            patches: [batch, num_patches, embed_dim]
        """
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Args:
        embed_dim: Embedding dimension
        max_patches: Maximum number of patches (for fixed encoding)
    """

    def __init__(self, embed_dim: int, max_patches: int = 256):  # 256 = 16x16 patches
        super().__init__()
        self.embed_dim = embed_dim

        # Create positional encoding matrix
        position = torch.arange(max_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_patches, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_patches, embed_dim]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, embed_dim]

        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """
    Transformer block using StyleForge CUDA kernels.

    Architecture:
        x → LayerNorm → CustomMultiheadAttention → Dropout → Add → x
        → LayerNorm → FusedFFN → Dropout → Add → output

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network hidden dimension
        dropout: Dropout probability
        use_cuda_kernels: If True, use CUDA kernels when available
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        use_cuda_kernels: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Multi-head attention with CUDA kernel
        self.attn = CustomMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=True,
            dropout=dropout,
            use_cuda_kernel=use_cuda_kernels
        )

        # Feed-forward network with CUDA kernel
        self.ffn = FusedFFNWrapper(
            embed_dim=embed_dim,
            hidden_dim=ffn_dim,
            bias=True,
            use_cuda_kernel=use_cuda_kernels
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            attn_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attn(x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) for style injection.

    AdaIN(content, style) = gamma(style) * norm(content) + beta(style)

    Args:
        num_features: Number of features (channels)
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            content: [batch, seq_len, channels] or [batch, channels, H, W]
            style: [batch, seq_len, channels] or [batch, channels, H, W]

        Returns:
            out: Normalized and style-modulated content
            params: Tuple of (gamma, beta) used
        """
        if content.dim() == 3:
            # [batch, seq_len, channels]
            mean_content = content.mean(dim=1, keepdim=True)
            std_content = content.std(dim=1, keepdim=True) + 1e-8
        else:
            # [batch, channels, H, W]
            mean_content = content.mean(dim=[2, 3], keepdim=True)
            std_content = content.std(dim=[2, 3], keepdim=True) + 1e-8

        # Compute style statistics
        if style.dim() == 3:
            mean_style = style.mean(dim=1, keepdim=True)
            std_style = style.std(dim=1, keepdim=True) + 1e-8
        else:
            mean_style = style.mean(dim=[2, 3], keepdim=True)
            std_style = style.std(dim=[2, 3], keepdim=True) + 1e-8

        # AdaIN formula
        gamma = std_style / std_content
        beta = mean_style - gamma * mean_content

        out = gamma * (content - mean_content) + beta

        return out, (gamma, beta)


class StyleEncoder(nn.Module):
    """
    Encode style image into style vectors.

    Args:
        in_channels: Number of input channels (3 for RGB)
        style_dim: Style embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        style_dim: int = 512
    ):
        super().__init__()

        # Simple CNN to extract style
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Style projection
        self.style_proj = nn.Linear(512, style_dim)

    def forward(self, style_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_image: [batch, 3, H, W]

        Returns:
            style: [batch, style_dim]
        """
        x = self.relu(self.conv1(style_image))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = self.global_pool(x).flatten(1)
        style = self.style_proj(x)

        return style


class UnpatchToImage(nn.Module):
    """
    Convert patch sequence back to image.

    Args:
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        out_channels: Number of output channels (3 for RGB)
        image_size: Output image size (H, W)
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 512,
        out_channels: int = 3,
        image_size: Tuple[int, int] = (256, 256)
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.image_size = image_size

        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Projection to RGB values per patch
        self.to_pixels = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, embed_dim]

        Returns:
            image: [batch, out_channels, H, W]
        """
        batch_size = x.size(0)

        # Project to pixels
        x = self.to_pixels(x)  # [B, num_patches, P*P*C]

        # Reshape to image
        x = x.reshape(
            batch_size,
            self.num_patches_h,
            self.num_patches_w,
            self.patch_size,
            self.patch_size,
            self.out_channels
        )

        # Rearrange to [B, C, H, W]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(
            batch_size,
            self.out_channels,
            self.num_patches_h * self.patch_size,
            self.num_patches_w * self.patch_size
        )

        return x


class StyleForgeTransformer(nn.Module):
    """
    Vision Transformer-based Style Transfer Model.

    This model heavily utilizes StyleForge's CUDA kernels:
    - CustomMultiheadAttention (fused_attention_v1) for 12+ attention calls
    - FusedFFNWrapper (fused_ffn) for feed-forward layers

    Architecture:
        Content Image → Patch Embedding → Encoder (6 blocks)
        Style Image → Style Encoder
        Content Features + Style → AdaIN → Decoder (6 blocks) → Output

    Args:
        image_size: Size of input/output images
        patch_size: Size of patches for ViT
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        num_encoder_blocks: Number of encoder transformer blocks
        num_decoder_blocks: Number of decoder transformer blocks
        ffn_dim: Feed-forward network hidden dimension
        style_dim: Style embedding dimension
        dropout: Dropout probability
        use_cuda_kernels: If True, use CUDA kernels when available
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
        ffn_dim: int = 2048,
        style_dim: int = 512,
        dropout: float = 0.1,
        use_cuda_kernels: bool = True
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks

        # Content encoding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        self.pos_encoding = PositionalEncoding(embed_dim)

        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_cuda_kernels=use_cuda_kernels
            )
            for _ in range(num_encoder_blocks)
        ])

        # Style encoder
        self.style_encoder = StyleEncoder(
            in_channels=3,
            style_dim=style_dim
        )

        # Style projection for AdaIN
        self.style_to_ada = nn.Sequential(
            nn.Linear(style_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_cuda_kernels=use_cuda_kernels
            )
            for _ in range(num_decoder_blocks)
        ])

        # AdaIN for style injection (applied after encoder, before decoder)
        self.adain = AdaptiveInstanceNorm(embed_dim)

        # Unpatch to image
        self.unpatch = UnpatchToImage(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=3,
            image_size=image_size
        )

        # Final projection to RGB range
        self.final_conv = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, 1)
        )

    def encode_content(self, content_image: torch.Tensor) -> torch.Tensor:
        """
        Encode content image into patch embeddings.

        Args:
            content_image: [batch, 3, H, W]

        Returns:
            content_features: [batch, num_patches, embed_dim]
        """
        x = self.patch_embed(content_image)
        x = self.pos_encoding(x)

        for block in self.encoder_blocks:
            x = block(x)

        return x

    def encode_style(self, style_image: torch.Tensor) -> torch.Tensor:
        """
        Encode style image into style vector.

        Args:
            style_image: [batch, 3, H, W]

        Returns:
            style: [batch, embed_dim]
        """
        style = self.style_encoder(style_image)
        style = self.style_to_ada(style)
        return style

    def forward(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply style transfer from style_image to content_image.

        Args:
            content_image: [batch, 3, H, W] - Content image (what to preserve)
            style_image: [batch, 3, H, W] - Style image (artistic style)

        Returns:
            output: [batch, 3, H, W] - Stylized image
        """
        # Encode content
        content_features = self.encode_content(content_image)

        # Encode style
        style_features = self.encode_style(style_image)

        # Expand style to match content spatial dimensions
        batch_size, num_patches, embed_dim = content_features.shape
        style_expanded = style_features.unsqueeze(1).expand(-1, num_patches, -1)

        # Apply AdaIN for style injection
        styled_features, _ = self.adain(content_features, style_expanded)

        # Decode with transformer blocks
        for block in self.decoder_blocks:
            styled_features = block(styled_features)

        # Convert patches back to image
        output = self.unpatch(styled_features)

        # Final projection
        output = self.final_conv(output)

        return output

    def get_kernel_stats(self) -> dict:
        """Get statistics about CUDA kernel usage."""
        return get_attention_kernel_stats(self)

    def print_kernel_stats(self) -> None:
        """Print CUDA kernel usage statistics."""
        print_attention_stats(self)


def create_styleforge_transformer(
    image_size: int = 256,
    patch_size: int = 16,
    embed_dim: int = 512,
    num_heads: int = 8,
    num_blocks: int = 6,
    ffn_dim: int = 2048,
    use_cuda_kernels: bool = True
) -> StyleForgeTransformer:
    """
    Factory function to create StyleForgeTransformer with default settings.

    Args:
        image_size: Size of input/output images
        patch_size: Size of patches
        embed_dim: Transformer embedding dimension
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks (encoder + decoder)
        ffn_dim: Feed-forward network hidden dimension
        use_cuda_kernels: If True, use CUDA kernels when available

    Returns:
        StyleForgeTransformer model
    """
    return StyleForgeTransformer(
        image_size=(image_size, image_size),
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_blocks=num_blocks,
        num_decoder_blocks=num_blocks,
        ffn_dim=ffn_dim,
        use_cuda_kernels=use_cuda_kernels
    )


# Predefined model configurations
STYLEFORGE_MODELS = {
    "small": {
        "image_size": 256,
        "patch_size": 16,
        "embed_dim": 256,
        "num_heads": 4,
        "num_blocks": 4,
        "ffn_dim": 1024,
    },
    "base": {
        "image_size": 256,
        "patch_size": 16,
        "embed_dim": 512,
        "num_heads": 8,
        "num_blocks": 6,
        "ffn_dim": 2048,
    },
    "large": {
        "image_size": 512,
        "patch_size": 16,
        "embed_dim": 768,
        "num_heads": 12,
        "num_blocks": 12,
        "ffn_dim": 3072,
    },
}


def create_model(variant: str = "base", use_cuda_kernels: bool = True) -> StyleForgeTransformer:
    """
    Create a StyleForgeTransformer model with predefined configuration.

    Args:
        variant: Model variant ("small", "base", or "large")
        use_cuda_kernels: If True, use CUDA kernels when available

    Returns:
        StyleForgeTransformer model
    """
    if variant not in STYLEFORGE_MODELS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(STYLEFORGE_MODELS.keys())}")

    config = STYLEFORGE_MODELS[variant].copy()
    # Map num_blocks to num_encoder_blocks and num_decoder_blocks
    num_blocks = config.pop("num_blocks", None)
    if num_blocks is not None:
        config["num_encoder_blocks"] = num_blocks
        config["num_decoder_blocks"] = num_blocks

    # Convert image_size from int to tuple if needed
    if "image_size" in config and isinstance(config["image_size"], int):
        config["image_size"] = (config["image_size"], config["image_size"])

    return StyleForgeTransformer(use_cuda_kernels=use_cuda_kernels, **config)
