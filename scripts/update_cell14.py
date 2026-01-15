"""
Update notebook with CELL 14 - Latent Space Interpolation
"""
import json

# Read notebook
with open('notebooks/demo.ipynb', 'r') as f:
    nb = json.load(f)

# New markdown header
new_cell_md = "## CELL 14: Latent Space Interpolation (Advanced)"

# New code cell
part1 = r"""# ============================================
# 🎨 LATENT SPACE INTERPOLATION
# ============================================

print("Implementing latent space interpolation...\n")
print("More sophisticated blending in activation space\n")

# ----------------------------------------
# Latent Interpolation
# ----------------------------------------

class LatentStyleBlender:
    \"\"\"Blend styles in latent/activation space.

    More sophisticated than weight-space blending.
    \"\"\"

    def __init__(self):
        self.style_models = {}

    def register_style_model(self, style_name, model):
        \"\"\"Register a complete model for a style.\"\"\"
        self.style_models[style_name] = model
        print(f"✓ Registered model for: {style_name}")

    def interpolate_in_latent_space(
        self,
        input_image,
        style_a_name,
        style_b_name,
        alpha=0.5,
        blend_point='transformer'
    ):
        \"\"\"Interpolate between two styles in activation space.

        Args:
            input_image: Input tensor
            style_a_name: First style name
            style_b_name: Second style name
            alpha: Blend factor (0 = all A, 1 = all B)
            blend_point: Where to blend ('encoder', 'transformer', 'all')

        Returns:
            Blended output image
        \"\"\"
        model_a = self.style_models[style_a_name]
        model_b = self.style_models[style_b_name]

        with torch.no_grad():
            # ----------------------------------------
            # Encode with both models
            # ----------------------------------------

            # Model A encoding
            x_a = input_image
            for layer in model_a.encoder:
                x_a = layer(x_a)

            # Model B encoding
            x_b = input_image
            for layer in model_b.encoder:
                x_b = layer(x_b)

            # Blend encoded features
            if blend_point in ['encoder', 'all']:
                x_blended = (1 - alpha) * x_a + alpha * x_b
            else:
                x_blended = x_a  # Use model A's encoding

            # ----------------------------------------
            # Transformer with interpolation
            # ----------------------------------------

            # Reshape for transformer
            B, C, H, W = x_blended.shape

            if blend_point in ['transformer', 'all']:
                # Process through both transformers and blend
                x_a_trans = x_a.flatten(2).transpose(1, 2)
                x_b_trans = x_b.flatten(2).transpose(1, 2)

                for block_a, block_b in zip(model_a.transformer_blocks,
                                           model_b.transformer_blocks):
                    x_a_trans = block_a(x_a_trans)
                    x_b_trans = block_b(x_b_trans)

                # Blend transformer outputs
                x_trans_blended = (1 - alpha) * x_a_trans + alpha * x_b_trans
                x_blended = x_trans_blended.transpose(1, 2).reshape(B, C, H, W)
            else:
                # Use blended encoding through model A's transformer
                x_trans = x_blended.flatten(2).transpose(1, 2)
                for block in model_a.transformer_blocks:
                    x_trans = block(x_trans)
                x_blended = x_trans.transpose(1, 2).reshape(B, C, H, W)

            # ----------------------------------------
            # Decode (using model A's decoder)
            # ----------------------------------------

            for layer in model_a.decoder:
                x_blended = layer(x_blended)

            output = model_a.final_activation(x_blended)

        return output
"""

part2 = """
# ----------------------------------------
# Test Latent Interpolation
# ----------------------------------------

print("🧪 Testing latent space interpolation...\\n")

# Create latent blender
latent_blender = LatentStyleBlender()

# Register two styles
style_a_model = blender.create_blended_model({'starry_night': 1.0})
style_b_model = blender.create_blended_model({'picasso': 1.0})

latent_blender.register_style_model('starry_night', style_a_model)
latent_blender.register_style_model('picasso', style_b_model)

print()

# Test interpolation at different alpha values
test_img = torch.randn(1, 3, 256, 256).cuda()

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []

print("Generating latent interpolations...")
for alpha in alphas:
    output = latent_blender.interpolate_in_latent_space(
        test_img,
        'starry_night',
        'picasso',
        alpha=alpha,
        blend_point='transformer'
    )
    results.append((alpha, output))
    print(f"  α={alpha:.2f} ✓")

print()
"""

part3 = """
# ----------------------------------------
# Visualize Latent Interpolation
# ----------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for idx, (alpha, output) in enumerate(results):
    ax = axes[idx]

    img = output[0].cpu().permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)

    ax.imshow(img)
    ax.set_title(f'α = {alpha:.2f}\\nStyle A {1-alpha:.0%} / Style B {alpha:.0%}',
                 fontsize=10)
    ax.axis('off')

plt.suptitle('Latent Space Interpolation (Transformer Blend)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(portfolio_dir / 'latent_interpolation.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Latent interpolation visualization saved\\n")
"""

part4 = """
# ----------------------------------------
# Compare Weight vs Latent Blending
# ----------------------------------------

print("📊 Comparing weight-space vs latent-space blending...\\n")

alpha_test = 0.5

# Weight-space blend
weight_blend_model = blender.create_blended_model({
    'starry_night': 0.5,
    'picasso': 0.5
})

with torch.no_grad():
    weight_blend_output = weight_blend_model(test_img)

# Latent-space blend
latent_blend_output = latent_blender.interpolate_in_latent_space(
    test_img,
    'starry_night',
    'picasso',
    alpha=0.5,
    blend_point='transformer'
)

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
ax = axes[0]
img = test_img[0].cpu().permute(1, 2, 0).numpy()
img = (img * 0.5 + 0.5).clip(0, 1)
ax.imshow(img)
ax.set_title('Input', fontsize=12, fontweight='bold')
ax.axis('off')

# Weight-space blend
ax = axes[1]
img = weight_blend_output[0].cpu().permute(1, 2, 0).numpy()
img = (img * 0.5 + 0.5).clip(0, 1)
ax.imshow(img)
ax.set_title('Weight-Space Blending\\n(Linear in Parameters)', fontsize=12)
ax.axis('off')

# Latent-space blend
ax = axes[2]
img = latent_blend_output[0].cpu().permute(1, 2, 0).numpy()
img = (img * 0.5 + 0.5).clip(0, 1)
ax.imshow(img)
ax.set_title('Latent-Space Blending\\n(Linear in Activations)', fontsize=12)
ax.axis('off')

plt.suptitle('Blending Method Comparison (50/50 Mix)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(portfolio_dir / 'blending_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✓ Comparison saved to portfolio/blending_comparison.png\\n")
"""

part5 = """
# ----------------------------------------
# Summary
# ----------------------------------------

print("="*70)
print("  LATENT SPACE INTERPOLATION COMPLETE")
print("="*70)

print()
print("Methods:")
print("  - Weight-Space Blending (CELL 13)")
print("    * Linear interpolation of model parameters")
print("    * Fast, single blended model")
print("    * Good for similar styles")
print()
print("  - Latent-Space Blending (CELL 14)")
print("    * Interpolation in activation space")
print("    * Can blend at different network depths")
print("    * More expressive for style combinations")
print()
print("Blend Points:")
print("  - 'encoder' - Blend after encoder")
print("  - 'transformer' - Blend after transformer blocks")
print("  - 'all' - Blend at multiple stages")
print()
print("Use Cases:")
print("  - Fine-grained style control")
print("  - Artistic style exploration")
print("  - Temporal coherence in video")
print()
print("="*70)
print("\\n✅ Latent space interpolation complete!")
"""

# Combine all parts
new_cell_code = part1 + '\n' + part2 + '\n' + part3 + '\n' + part4 + '\n' + part5

# Find where to insert (after CELL 13 Multi-Style Blending)
insert_index = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 13: Multi-Style Blending' in source:
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
