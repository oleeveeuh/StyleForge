"""
StyleForge - Style Blender

Multi-style blending functionality for interpolating between
different artistic styles in weight space.
"""

import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class StyleBlender:
    """
    Blend multiple style models in weight space.

    Allows smooth interpolation between different artistic styles
    by blending their learned weights.

    Args:
        base_model: Base StyleTransferNetwork to use as template
        device: Device to load models on

    Example:
        >>> blender = StyleBlender(base_model)
        >>> blender.register_style('starry_night', 'checkpoints/starry_night.pth')
        >>> blender.register_style('picasso', 'checkpoints/picasso.pth')
        >>>
        >>> # Create 60% starry_night + 40% picasso blend
        >>> blended = blender.create_blended_model({
        ...     'starry_night': 0.6,
        ...     'picasso': 0.4
        ... })
    """

    def __init__(self, base_model: nn.Module, device: str = 'cuda'):
        self.base_model = base_model
        self.device = device
        self.style_checkpoints: Dict[str, Dict[str, torch.Tensor]] = {}

    def register_style(
        self,
        style_name: str,
        checkpoint_path: Optional[str] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        model: Optional[nn.Module] = None
    ):
        """
        Register a style checkpoint.

        Args:
            style_name: Name of the style (e.g., 'starry_night')
            checkpoint_path: Path to .pth file
            state_dict: Direct state dict
            model: Model to extract state dict from
        """
        if model is not None:
            state_dict = model.state_dict()

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

        if state_dict is None:
            # Use base model weights as placeholder
            state_dict = copy.deepcopy(self.base_model.state_dict())

        # Move to device
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(self.device)

        self.style_checkpoints[style_name] = state_dict
        print(f"✓ Registered style: {style_name}")

    def blend_styles(
        self,
        style_weights: Dict[str, float],
        normalize: bool = True
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Blend multiple styles in weight space.

        Args:
            style_weights: Dict mapping style names to blend weights
                           e.g., {'starry_night': 0.6, 'picasso': 0.4}
            normalize: Whether to normalize weights to sum to 1.0

        Returns:
            Blended state dict
        """
        # Normalize weights
        if normalize:
            total = sum(style_weights.values())
            if total > 0:
                style_weights = {k: v / total for k, v in style_weights.items()}

        print(f"\n🎨 Blending styles:")
        for style, weight in style_weights.items():
            print(f"   {style}: {weight:.1%}")

        # Initialize blended state dict
        blended_state = OrderedDict()

        # Get all parameter names from first style
        first_style = list(style_weights.keys())[0]
        param_names = self.style_checkpoints[first_style].keys()

        # Blend each parameter
        for param_name in param_names:
            blended_param = None

            for style_name, weight in style_weights.items():
                if style_name not in self.style_checkpoints:
                    print(f"   Warning: Style '{style_name}' not found, skipping")
                    continue

                style_param = self.style_checkpoints[style_name][param_name]

                if blended_param is None:
                    blended_param = weight * style_param.clone()
                else:
                    blended_param = blended_param + weight * style_param.clone()

            blended_state[param_name] = blended_param

        print(f"✓ Blended {len(blended_state)} parameters\n")

        return blended_state

    def create_blended_model(
        self,
        style_weights: Dict[str, float],
        normalize: bool = True
    ) -> nn.Module:
        """
        Create a new model with blended weights.

        Args:
            style_weights: Dict mapping style names to blend weights
            normalize: Whether to normalize weights

        Returns:
            Model with blended weights
        """
        blended_model = copy.deepcopy(self.base_model)
        blended_state = self.blend_styles(style_weights, normalize)
        blended_model.load_state_dict(blended_state)
        return blended_model

    def interpolate_styles(
        self,
        style_a: str,
        style_b: str,
        num_steps: int = 5
    ) -> List[Tuple[float, nn.Module]]:
        """
        Create interpolation between two styles.

        Args:
            style_a: First style name
            style_b: Second style name
            num_steps: Number of interpolation steps

        Returns:
            List of (alpha, model) tuples
        """
        models = []
        alphas = np.linspace(0, 1, num_steps)

        for alpha in alphas:
            blend = {style_a: 1 - alpha, style_b: alpha}
            model = self.create_blended_model(blend)
            models.append((alpha, model))

        return models

    def save_registry(self, filepath: str):
        """Save style registry to JSON file (metadata only, not weights)"""
        registry = {
            'styles': list(self.style_checkpoints.keys()),
            'base_model_type': type(self.base_model).__name__
        }

        with open(filepath, 'w') as f:
            json.dump(registry, f, indent=2)

        print(f"✓ Registry saved to {filepath}")


def create_placeholder_styles(
    model_class,
    styles: List[str],
    save_dir: str = 'checkpoints'
):
    """
    Create placeholder style checkpoints with random weights.

    Args:
        model_class: Model class to instantiate
        styles: List of style names
        save_dir: Directory to save checkpoints

    Returns:
        Dict mapping style names to checkpoint paths
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoints = {}

    for style in styles:
        # Create model with random weights
        model = model_class()

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'style_name': style,
            'trained': False,  # Mark as placeholder
            'config': {
                'model_type': type(model).__name__,
                'num_params': sum(p.numel() for p in model.parameters())
            }
        }

        ckpt_path = save_path / f'{style}.pth'
        torch.save(checkpoint, ckpt_path)
        checkpoints[style] = str(ckpt_path)
        print(f"✓ Created placeholder: {style}.pth")

    return checkpoints


def visualize_style_interpolation(
    blender: StyleBlender,
    style_a: str,
    style_b: str,
    test_image: torch.Tensor,
    num_steps: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize interpolation between two styles.

    Args:
        blender: StyleBlender instance
        style_a: First style name
        style_b: Second style name
        test_image: Input image [1, 3, H, W]
        num_steps: Number of interpolation steps
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt

    # Get interpolation models
    interp_models = blender.interpolate_styles(style_a, style_b, num_steps)

    # Process with each blend
    results = []
    with torch.no_grad():
        for alpha, model in interp_models:
            output = model(test_image)
            results.append((alpha, output))

    # Plot results
    fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))

    for idx, (alpha, output) in enumerate(results):
        ax = axes[idx]

        # Convert to displayable image
        img = output[0].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize [-1,1] to [0,1]

        ax.imshow(img)
        ax.set_title(f'{style_a.title()}: {1-alpha:.0%}\n{style_b.title()}: {alpha:.0%}',
                     fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Style Interpolation: {style_a.title()} → {style_b.title()}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")

    plt.show()
