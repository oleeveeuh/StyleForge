"""
Update notebook with CELL 15 - Regional Style Control
"""
import json

# Read notebook
with open('notebooks/demo.ipynb', 'r') as f:
    nb = json.load(f)

# New markdown header
new_cell_md = "## CELL 15: Regional Style Control"

# New code cell - split into parts to avoid quote conflicts
part1 = r"""# ============================================
# 🖌️ REGIONAL STYLE CONTROL
# ============================================

print("Implementing regional style control with masks...\n")
print("Allows applying style to specific image regions\n")

# ----------------------------------------
# Regional Styler Class
# ----------------------------------------

class RegionalStyler:
    \"\"\"Apply style transfer to specific regions using masks.\"\"\"

    def __init__(self, model):
        \"\"\"Initialize with base style transfer model.

        Args:
            model: Base style transfer model
        \"\"\"
        self.model = model

    def apply_regional_style(
        self,
        input_image,
        mask,
        style_strength=1.0,
        blur_radius=5
    ):
        \"\"\"Apply style only in masked regions.

        Args:
            input_image: [B, 3, H, W] Input image
            mask: [B, 1, H, W] Mask (0-1 float, 1 = apply style)
            style_strength: Overall style intensity
            blur_radius: Blur radius for smooth transitions

        Returns:
            Styled image with smooth blending
        \"\"\"
        with torch.no_grad():
            # Apply style to full image
            styled = self.model(input_image)

            # Optionally blur mask for smoother transitions
            if blur_radius > 0:
                mask = self._blur_mask(mask, blur_radius)

            # Blend: output = mask * styled + (1 - mask) * original
            # Apply style strength
            effective_mask = mask * style_strength
            output = effective_mask * styled + (1 - effective_mask) * input_image

            return output

    def _blur_mask(self, mask, radius):
        \"\"\"Apply Gaussian blur to mask for smooth transitions.\"\"\"
        # Simple box blur for smooth edges
        kernel_size = radius * 2 + 1
        blur = nn.AvgPool2d(kernel_size, stride=1, padding=radius)

        # Apply blur (may need to pad)
        blurred = blur(mask)

        return blurred

    def create_circular_mask(self, height, width, center, radius):
        \"\"\"Create circular mask.

        Args:
            height, width: Image dimensions
            center: (y, x) center coordinates
            radius: Circle radius in pixels

        Returns:
            [1, 1, H, W] mask tensor
        \"\"\"
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij'
        )

        cy, cx = center
        distance = torch.sqrt((y - cy)**2 + (x - cx)**2)
        mask = (distance <= radius).float()

        return mask.unsqueeze(0).unsqueeze(0).cuda()

    def create_gradient_mask(self, height, width, direction='horizontal'):
        \"\"\"Create gradient mask.

        Args:
            height, width: Image dimensions
            direction: 'horizontal', 'vertical', or 'radial'

        Returns:
            [1, 1, H, W] mask tensor
        \"\"\"
        if direction == 'horizontal':
            mask = torch.linspace(0, 1, width).repeat(height, 1)
        elif direction == 'vertical':
            mask = torch.linspace(0, 1, height).unsqueeze(1).repeat(1, width)
        elif direction == 'radial':
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, height),
                torch.linspace(-1, 1, width),
                indexing='ij'
            )
            mask = 1 - torch.sqrt(x**2 + y**2).clip(0, 1)

        return mask.unsqueeze(0).unsqueeze(0).cuda()
"""

part2 = """
# ----------------------------------------
# Test Regional Styling
# ----------------------------------------

print("🧪 Testing regional styling...\\n")

# Create test image
test_img = torch.randn(1, 3, 512, 512).cuda()

# Get a styled model
style_model = blender.create_blended_model({'starry_night': 1.0})

# Create regional styler
regional_styler = RegionalStyler(style_model)

# ----------------------------------------
# Test 1: Circular Mask
# ----------------------------------------

print("1️⃣  Testing circular mask...")

circular_mask = regional_styler.create_circular_mask(
    height=512,
    width=512,
    center=(256, 256),
    radius=150
)

output_circular = regional_styler.apply_regional_style(
    test_img,
    circular_mask,
    style_strength=1.0,
    blur_radius=10
)

print(f"   Output shape: {output_circular.shape} ✓\\n")

# ----------------------------------------
# Test 2: Gradient Mask
# ----------------------------------------

print("2️⃣  Testing gradient mask...")

gradient_mask = regional_styler.create_gradient_mask(
    height=512,
    width=512,
    direction='horizontal'
)

output_gradient = regional_styler.apply_regional_style(
    test_img,
    gradient_mask,
    style_strength=1.0,
    blur_radius=5
)

print(f"   Output shape: {output_gradient.shape} ✓\\n")

# ----------------------------------------
# Test 3: Custom Painted Mask
# ----------------------------------------

print("3️⃣  Testing custom painted mask...")

# Simulate user-painted mask (e.g., from brush strokes)
painted_mask = torch.zeros(1, 1, 512, 512).cuda()

# Add some \"brush strokes\" (rectangles as example)
painted_mask[0, 0, 100:200, 100:300] = 1.0
painted_mask[0, 0, 300:400, 200:400] = 1.0

output_painted = regional_styler.apply_regional_style(
    test_img,
    painted_mask,
    style_strength=0.8,
    blur_radius=15
)

print(f"   Output shape: {output_painted.shape} ✓\\n")
"""

part3 = """
# ----------------------------------------
# Visualize Regional Control
# ----------------------------------------

print("Creating visualization...\\n")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

test_cases = [
    ('Circular Mask', circular_mask, output_circular),
    ('Gradient Mask', gradient_mask, output_gradient),
    ('Painted Mask', painted_mask, output_painted)
]

for row, (name, mask, output) in enumerate(test_cases):
    # Input
    ax = axes[row, 0]
    img = test_img[0].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    ax.imshow(img)
    if row == 0:
        ax.set_title('Input Image', fontsize=11, fontweight='bold')
    ax.set_ylabel(name, fontsize=11, fontweight='bold')
    ax.axis('off')

    # Mask
    ax = axes[row, 1]
    mask_vis = mask[0, 0].cpu().numpy()
    ax.imshow(mask_vis, cmap='gray')
    if row == 0:
        ax.set_title('Mask\\n(White = Apply Style)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Full style (no mask)
    ax = axes[row, 2]
    with torch.no_grad():
        full_styled = style_model(test_img)
    img = full_styled[0].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    ax.imshow(img)
    if row == 0:
        ax.set_title('Full Style\\n(No Masking)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Regional result
    ax = axes[row, 3]
    img = output[0].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    ax.imshow(img)
    if row == 0:
        ax.set_title('Regional Result\\n(Masked)', fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('Regional Style Control Examples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(portfolio_dir / 'regional_control.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Visualization saved to portfolio/regional_control.png\\n")
"""

part4 = r"""
# ----------------------------------------
# Interactive Mask Builder
# ----------------------------------------

class InteractiveMaskBuilder:
    \"\"\"Helper for building masks programmatically.

    In web demo, this would be replaced with canvas drawing.
    \"\"\"

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.mask = torch.zeros(1, 1, height, width)

    def add_circle(self, center, radius, value=1.0):
        \"\"\"Add circular region to mask.\"\"\"
        y, x = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32),
            torch.arange(self.width, dtype=torch.float32),
            indexing='ij'
        )

        cy, cx = center
        distance = torch.sqrt((y - cy)**2 + (x - cx)**2)
        circle_mask = (distance <= radius).float() * value

        self.mask = torch.maximum(self.mask, circle_mask.unsqueeze(0).unsqueeze(0))

        return self

    def add_rectangle(self, top_left, bottom_right, value=1.0):
        \"\"\"Add rectangular region to mask.\"\"\"
        y1, x1 = top_left
        y2, x2 = bottom_right

        self.mask[0, 0, y1:y2, x1:x2] = value

        return self

    def blur(self, radius=5):
        \"\"\"Blur the mask for smooth edges.\"\"\"
        kernel_size = radius * 2 + 1
        blur_layer = nn.AvgPool2d(kernel_size, stride=1, padding=radius)
        self.mask = blur_layer(self.mask)

        return self

    def get_mask(self):
        \"\"\"Get final mask tensor.\"\"\"
        return self.mask.cuda()

# Test mask builder
print("🔧 Testing interactive mask builder...\n")

mask_builder = InteractiveMaskBuilder(512, 512)
mask_builder.add_circle((150, 150), 80)\
            .add_circle((350, 350), 100)\
            .add_rectangle((200, 250), (300, 400))\
            .blur(10)

complex_mask = mask_builder.get_mask()

output_complex = regional_styler.apply_regional_style(
    test_img,
    complex_mask,
    style_strength=1.0
)

print("✓ Complex mask created and applied\n")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
img = test_img[0].cpu().permute(1, 2, 0).numpy()
img = (img * 0.5 + 0.5).clip(0, 1)
ax.imshow(img)
ax.set_title('Input', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[1]
ax.imshow(complex_mask[0, 0].cpu().numpy(), cmap='viridis')
ax.set_title('Complex Mask\n(Multiple Regions)', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[2]
img = output_complex[0].cpu().permute(1, 2, 0).numpy()
img = (img * 0.5 + 0.5).clip(0, 1)
ax.imshow(img)
ax.set_title('Regional Result', fontsize=12, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig(portfolio_dir / 'complex_mask_example.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Complex mask example saved\n")
"""

part5 = """
# ----------------------------------------
# Save Regional Styler Code
# ----------------------------------------

regional_code = '''\"""
StyleForge - Regional Style Control
Apply style transfer to specific image regions using masks
\"""
import torch
import torch.nn as nn

class RegionalStyler:
    \"\"\"Regional style control with mask-based blending\"\"\"

    def __init__(self, model):
        self.model = model

    def apply_regional_style(self, input_image, mask, style_strength=1.0, blur_radius=5):
        with torch.no_grad():
            styled = self.model(input_image)
            if blur_radius > 0:
                mask = self._blur_mask(mask, blur_radius)
            effective_mask = mask * style_strength
            output = effective_mask * styled + (1 - effective_mask) * input_image
            return output

    def _blur_mask(self, mask, radius):
        kernel_size = radius * 2 + 1
        blur = nn.AvgPool2d(kernel_size, stride=1, padding=radius)
        return blur(mask)

    def create_circular_mask(self, height, width, center, radius):
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing='ij')
        cy, cx = center
        distance = torch.sqrt((y - cy)**2 + (x - cx)**2)
        mask = (distance <= radius).float()
        return mask.unsqueeze(0).unsqueeze(0).cuda()

    def create_gradient_mask(self, height, width, direction='horizontal'):
        if direction == 'horizontal':
            mask = torch.linspace(0, 1, width).repeat(height, 1)
        elif direction == 'vertical':
            mask = torch.linspace(0, 1, height).unsqueeze(1).repeat(1, width)
        elif direction == 'radial':
            y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
            mask = 1 - torch.sqrt(x**2 + y**2).clip(0, 1)
        return mask.unsqueeze(0).unsqueeze(0).cuda()

class InteractiveMaskBuilder:
    \"\"\"Build masks programmatically\"\"\"

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.mask = torch.zeros(1, 1, height, width)

    def add_circle(self, center, radius, value=1.0):
        y, x = torch.meshgrid(torch.arange(self.height, dtype=torch.float32),
                              torch.arange(self.width, dtype=torch.float32), indexing='ij')
        cy, cx = center
        distance = torch.sqrt((y - cy)**2 + (x - cx)**2)
        circle_mask = (distance <= radius).float() * value
        self.mask = torch.maximum(self.mask, circle_mask.unsqueeze(0).unsqueeze(0))
        return self

    def add_rectangle(self, top_left, bottom_right, value=1.0):
        y1, x1 = top_left
        y2, x2 = bottom_right
        self.mask[0, 0, y1:y2, x1:x2] = value
        return self

    def blur(self, radius=5):
        kernel_size = radius * 2 + 1
        blur_layer = nn.AvgPool2d(kernel_size, stride=1, padding=radius)
        self.mask = blur_layer(self.mask)
        return self

    def get_mask(self):
        return self.mask.cuda()
'''

regional_path = project_root / 'utils' / 'regional_styler.py'
with open(regional_path, 'w') as f:
    f.write(regional_code)

print(f"✓ Saved regional styler to {regional_path}")

# ----------------------------------------
# Summary
# ----------------------------------------

print("="*70)
print("  REGIONAL STYLE CONTROL COMPLETE")
print("="*70)

print()
print("Features:")
print("  - Apply style to specific regions using masks")
print("  - Circular, gradient, and custom painted masks")
print("  - Smooth blending with adjustable blur radius")
print("  - Style strength control")
print()
print("Mask Types:")
print("  - Circular: Radial region masking")
print("  - Gradient: Smooth horizontal/vertical/radial transitions")
print("  - Painted: User-defined brush strokes")
print("  - Complex: Multiple combined regions")
print()
print("Use Cases:")
print("  - Selective style application")
print("  - Smooth gradient transitions")
print("  - Face-only styling")
print("  - Background/foreground separation")
print()
print("="*70)
print("\\n✅ Regional control complete!")
"""

# Combine all parts
new_cell_code = part1 + '\n' + part2 + '\n' + part3 + '\n' + part4 + '\n' + part5

# Find where to insert (after CELL 14 Latent Space Interpolation)
insert_index = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 14: Latent Space Interpolation' in source:
        # Find the code cell after this markdown
        if i + 1 < len(nb['cells']) and nb['cells'][i + 1]['cell_type'] == 'code':
            insert_index = i + 2
        else:
            insert_index = i + 1
        break

if insert_index is None:
    # Fallback: insert at end
    insert_index = len(nb['cells'])

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
