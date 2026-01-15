"""
Update notebook with CELL 13 - Multi-Style Blending
"""
import json

# Read notebook
with open('notebooks/demo.ipynb', 'r') as f:
    nb = json.load(f)

# Find cells to replace
cells_to_replace = []
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 14: Multi-Style Blending' in source or ('MULTI-STYLE BLENDING' in source and 'StyleBlender' in source):
        cells_to_replace.append(i)

print(f"Cells to replace: {cells_to_replace}")

# New markdown header
new_cell_md = "## CELL 13: Multi-Style Blending"

# New code cell - split into parts to avoid triple-quote conflicts
part1 = r"""# ============================================
# 🎨 MULTI-STYLE BLENDING
# ============================================

print("Implementing multi-style blending...\n")
print("Allows interpolating between multiple artistic styles\n")

import copy
from collections import OrderedDict

# ----------------------------------------
# Style Blender Class
# ----------------------------------------

class StyleBlender:
    \"\"\"Blend multiple style models in weight space.\"\"\"

    def __init__(self, base_model):
        \"\"\"
        Args:
            base_model: Base StyleTransferNetwork to use as template
        \"\"\"
        self.base_model = base_model
        self.style_checkpoints = {}

    def register_style(self, style_name, checkpoint_path=None, state_dict=None):
        \"\"\"
        Register a style checkpoint

        Args:
            style_name: Name of the style (e.g., 'starry_night')
            checkpoint_path: Path to .pth file (optional)
            state_dict: Direct state dict (optional)
        \"\"\"
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

        if state_dict is None:
            state_dict = copy.deepcopy(self.base_model.state_dict())

        self.style_checkpoints[style_name] = state_dict
        print(f"✓ Registered style: {style_name}")

    def blend_styles(self, style_weights_dict, normalize=True):
        \"\"\"
        Blend multiple styles in weight space

        Args:
            style_weights_dict: Dict mapping style names to blend weights
                               e.g., {'starry_night': 0.6, 'picasso': 0.4}
            normalize: Whether to normalize weights to sum to 1.0

        Returns:
            Blended state dict
        \"\"\"
        if normalize:
            total = sum(style_weights_dict.values())
            style_weights_dict = {k: v/total for k, v in style_weights_dict.items()}

        print(f"\n🎨 Blending styles:")
        for style, weight in style_weights_dict.items():
            print(f"   {style}: {weight:.1%}")

        blended_state = OrderedDict()
        first_style = list(style_weights_dict.keys())[0]
        param_names = self.style_checkpoints[first_style].keys()

        for param_name in param_names:
            blended_param = None
            for style_name, weight in style_weights_dict.items():
                style_param = self.style_checkpoints[style_name][param_name]
                if blended_param is None:
                    blended_param = weight * style_param
                else:
                    blended_param = blended_param + weight * style_param
            blended_state[param_name] = blended_param

        print(f"✓ Blended {len(blended_state)} parameters\n")
        return blended_state

    def create_blended_model(self, style_weights_dict):
        \"\"\"
        Create a new model with blended weights

        Returns:
            Model with blended weights
        \"\"\"
        blended_model = copy.deepcopy(self.base_model)
        blended_state = self.blend_styles(style_weights_dict)
        blended_model.load_state_dict(blended_state)
        return blended_model
"""

part2 = """
# ----------------------------------------
# Create Style Checkpoints (Placeholders)
# ----------------------------------------

print("Creating placeholder style checkpoints...\\n")

styles = ['starry_night', 'picasso', 'monet', 'anime', 'cyberpunk', 'watercolor']

import os
checkpoint_dir = project_root / 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

for style in styles:
    style_model = OptimizedStyleTransferNetwork().cuda()
    checkpoint = {
        'model_state_dict': style_model.state_dict(),
        'style_name': style,
        'trained': False,
    }
    checkpoint_path = checkpoint_dir / f'{style}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Created placeholder: {style}.pth")

print("\\n💡 Note: Using random weights as placeholders")
print("   In production, train actual style transfer models\\n")

# ----------------------------------------
# Test Style Blending
# ----------------------------------------

print("🧪 Testing style blending...\\n")

blender = StyleBlender(OptimizedStyleTransferNetwork().cuda())

for style in styles:
    blender.register_style(style, checkpoint_path=str(checkpoint_dir / f'{style}.pth'))

print()

blend_dict = {'starry_night': 0.6, 'picasso': 0.4}
blended_model = blender.create_blended_model(blend_dict)

test_input = torch.randn(1, 3, 512, 512).cuda()
with torch.no_grad():
    output = blended_model(test_input)

print(f"✅ Blended model works!")
print(f"   Input:  {test_input.shape}")
print(f"   Output: {output.shape}\\n")
"""

part3 = """
# ----------------------------------------
# Create Blend Interpolation Grid
# ----------------------------------------

print("Creating blend interpolation examples...\\n")

def create_interpolation_grid(blender, style_a, style_b, num_steps=5):
    models = []
    alphas = np.linspace(0, 1, num_steps)
    for alpha in alphas:
        blend = {style_a: 1 - alpha, style_b: alpha}
        model = blender.create_blended_model(blend)
        models.append((alpha, model))
    return models

interp_models = create_interpolation_grid(blender, 'starry_night', 'picasso', num_steps=5)
print(f"✓ Created {len(interp_models)} interpolation steps\\n")

# ----------------------------------------
# Visualize Blending Results
# ----------------------------------------

print("Generating blend visualization...\\n")

test_img = torch.randn(1, 3, 256, 256).cuda()
results = []
with torch.no_grad():
    for alpha, model in interp_models:
        output = model(test_img)
        results.append((alpha, output))

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for idx, (alpha, output) in enumerate(results):
    ax = axes[idx]
    img = output[0].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    ax.imshow(img)
    ax.set_title(f'Starry Night {1-alpha:.0%}\\nPicasso {alpha:.0%}', fontsize=10)
    ax.axis('off')

plt.suptitle('Style Interpolation: Starry Night → Picasso', fontsize=14, fontweight='bold')
plt.tight_layout()

portfolio_dir = project_root / 'portfolio'
os.makedirs(portfolio_dir, exist_ok=True)
plt.savefig(portfolio_dir / 'style_interpolation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Visualization saved to {portfolio_dir / 'style_interpolation.png'}\\n")
"""

part4 = """
# ----------------------------------------
# Save Blender Code to File
# ----------------------------------------

blender_code = '\"""
StyleForge - Multi-Style Blending
Allows blending multiple artistic styles in weight space
\"""
import torch
import copy
from collections import OrderedDict

class StyleBlender:
    def __init__(self, base_model):
        self.base_model = base_model
        self.style_checkpoints = {}

    def register_style(self, style_name, checkpoint_path=None, state_dict=None):
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and \"model_state_dict\" in checkpoint:
                state_dict = checkpoint[\"model_state_dict\"]
            else:
                state_dict = checkpoint
        if state_dict is None:
            state_dict = copy.deepcopy(self.base_model.state_dict())
        self.style_checkpoints[style_name] = state_dict

    def blend_styles(self, style_weights_dict, normalize=True):
        if normalize:
            total = sum(style_weights_dict.values())
            style_weights_dict = {k: v/total for k, v in style_weights_dict.items()}
        blended_state = OrderedDict()
        first_style = list(style_weights_dict.keys())[0]
        param_names = self.style_checkpoints[first_style].keys()
        for param_name in param_names:
            blended_param = None
            for style_name, weight in style_weights_dict.items():
                style_param = self.style_checkpoints[style_name][param_name]
                if blended_param is None:
                    blended_param = weight * style_param
                else:
                    blended_param = blended_param + weight * style_param
            blended_state[param_name] = blended_param
        return blended_state

    def create_blended_model(self, style_weights_dict):
        blended_model = copy.deepcopy(self.base_model)
        blended_state = self.blend_styles(style_weights_dict)
        blended_model.load_state_dict(blended_state)
        return blended_model
'

blender_path = project_root / 'utils' / 'style_blender.py'
with open(blender_path, 'w') as f:
    f.write(blender_code)

print(f\"✓ Saved blender code to {blender_path}\")

# ----------------------------------------
# Summary
# ----------------------------------------

print(\"\\\\n\" + \"=\"*70)
print(\"  MULTI-STYLE BLENDING COMPLETE\")
print(\"=\"*70)

print(\"\"\"
╔══════════════════════════════════════════════════════════════╗
║              MULTI-STYLE BLENDING IMPLEMENTED               ║
╠══════════════════════════════════════════════════════════════╣
║  Features:                                                   ║
║    • Weight-space style blending                            ║
║    • Interpolate between any 2 styles                       ║
║    • Combine 3+ styles with custom weights                  ║
║    • Smooth transitions at customizable granularity         ║
║  Use Cases:                                                  ║
║    • Creative exploration of style combinations               ║
║    • Gradual transition between styles in video              ║
║    • Personalized style mixing                              ║
╚══════════════════════════════════════════════════════════════╝
\"\"\")
print(\"=\"*70)
print(\"\\\\n✅ Multi-style blending complete!\")
"""

# Combine all parts
new_cell_code = part1 + '\n' + part2 + '\n' + part3 + '\n' + part4

# Find where to insert
insert_index = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 13: Final Benchmark' in source:
        insert_index = i + 1
        break

if insert_index and cells_to_replace:
    # Remove old cells
    for idx in sorted(cells_to_replace, reverse=True):
        if idx < insert_index:
            insert_index -= 1
        del nb['cells'][idx]
        print(f"Deleted cell at index {idx}")

    # Insert new cells
    new_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": new_cell_md
    }
    new_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_cell_code
    }

    nb['cells'].insert(insert_index, new_code_cell)
    nb['cells'].insert(insert_index, new_md_cell)
    print(f"Inserted new cells at index {insert_index}")

# Save notebook
with open('notebooks/demo.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook updated successfully!")
print(f"Total cells now: {len(nb['cells'])}")
