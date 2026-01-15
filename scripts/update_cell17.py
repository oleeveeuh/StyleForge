"""
Update notebook with CELL 17 - Temporal Coherence for Video
"""
import json

# Read notebook
with open('notebooks/demo.ipynb', 'r') as f:
    nb = json.load(f)

# New markdown header
new_cell_md = "## CELL 17: Temporal Coherence for Video"

# New code cell - split into parts
part1 = r"""# ============================================
# 🎬 TEMPORAL COHERENCE FOR VIDEO
# ============================================

print("Implementing temporal coherence for video stylization...\n")
print("Goal: Flicker-free, consistent video style transfer\n")

import cv2
from collections import deque

# ----------------------------------------
# Temporal Styler Class
# ----------------------------------------

class TemporalStyler:
    \"\"\"Apply style transfer to video with temporal coherence.

    Prevents flickering between frames.
    \"\"\"

    def __init__(self, model, blend_factor=0.7):
        \"\"\"Initialize temporal styler.

        Args:
            model: Style transfer model
            blend_factor: How much to blend with previous frame (0-1)
                         Higher = more temporal stability, less responsiveness
        \"\"\"
        self.model = model
        self.blend_factor = blend_factor
        self.previous_styled = None
        self.frame_buffer = deque(maxlen=3)  # Keep last 3 frames

    def reset(self):
        \"\"\"Reset temporal state (call at start of new video).\"\"\"
        self.previous_styled = None
        self.frame_buffer.clear()

    def process_frame(self, frame_tensor, use_optical_flow=False):
        \"\"\"Process single video frame with temporal coherence.

        Args:
            frame_tensor: [1, 3, H, W] Current frame
            use_optical_flow: Whether to use optical flow for warping

        Returns:
            Styled frame with temporal coherence
        \"\"\"
        with torch.no_grad():
            # Style current frame
            current_styled = self.model(frame_tensor)

            if self.previous_styled is None:
                # First frame - no blending
                output = current_styled
            else:
                # Blend with previous frame for temporal coherence
                if use_optical_flow and len(self.frame_buffer) >= 2:
                    # Warp previous styled frame using optical flow
                    warped_previous = self._warp_with_flow(
                        self.previous_styled,
                        self.frame_buffer[-2],
                        frame_tensor
                    )
                    output = self.blend_factor * warped_previous + \\\
                            (1 - self.blend_factor) * current_styled
                else:
                    # Simple temporal blending
                    output = self.blend_factor * self.previous_styled + \\\
                            (1 - self.blend_factor) * current_styled

            # Update state
            self.previous_styled = output.clone()
            self.frame_buffer.append(frame_tensor)

            return output

    def _warp_with_flow(self, previous_styled, previous_frame, current_frame):
        \"\"\"Warp previous styled frame using optical flow.

        This helps maintain consistency when there's motion.
        \"\"\"
        # Convert to numpy for OpenCV
        prev_np = previous_frame[0].cpu().permute(1, 2, 0).numpy()
        curr_np = current_frame[0].cpu().permute(1, 2, 0).numpy()

        # Normalize to 0-255 for optical flow
        prev_np = ((prev_np * 0.5 + 0.5) * 255).astype(np.uint8)
        curr_np = ((curr_np * 0.5 + 0.5) * 255).astype(np.uint8)

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Warp previous styled frame
        h, w = flow.shape[:2]
        flow_map = np.column_stack([
            (np.arange(w) + flow[:, :, 0]).flatten(),
            (np.arange(h)[:, None] + flow[:, :, 1]).flatten()
        ]).reshape(h, w, 2)

        # Convert styled frame to numpy
        styled_np = previous_styled[0].cpu().permute(1, 2, 0).numpy()
        styled_np = ((styled_np * 0.5 + 0.5) * 255).astype(np.uint8)

        # Warp
        warped = cv2.remap(
            styled_np,
            flow_map[:, :, 0].astype(np.float32),
            flow_map[:, :, 1].astype(np.float32),
            cv2.INTER_LINEAR
        )

        # Convert back to tensor
        warped = warped.astype(np.float32) / 255.0
        warped = (warped - 0.5) / 0.5
        warped_tensor = torch.from_numpy(warped).permute(2, 0, 1).unsqueeze(0).cuda()

        return warped_tensor
"""

part2 = """
# ----------------------------------------
# Video Processing Function
# ----------------------------------------

def process_video_file(
    video_path,
    output_path,
    model,
    use_temporal_coherence=True,
    use_optical_flow=False,
    blend_factor=0.7,
    max_frames=None
):
    \\\"\"\\\"Process entire video file with style transfer

    Args:
        video_path: Path to input video
        output_path: Path to save output video
        model: Style transfer model
        use_temporal_coherence: Whether to use temporal blending
        use_optical_flow: Whether to use optical flow warping
        blend_factor: Temporal blending factor
        max_frames: Maximum frames to process (None = all)

    Returns:
        Processing statistics
    \\\"\"\\\"
    print(f\"📹 Processing video: {video_path}\\n\")

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f\"Could not open video: {video_path}\")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f\"Video properties:\")
    print(f\"  Resolution: {width}×{height}\")
    print(f\"  FPS: {fps}\")
    print(f\"  Total frames: {total_frames}\")

    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f\"  Processing: {total_frames} frames\\n\")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create temporal styler
    if use_temporal_coherence:
        temporal_styler = TemporalStyler(model, blend_factor)
        temporal_styler.reset()

    # Process frames
    frame_times = []
    frame_idx = 0

    print(\"Processing frames...\")

    while True:
        ret, frame = cap.read()

        if not ret or (max_frames and frame_idx >= max_frames):
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 512×512 for model
        frame_resized = cv2.resize(frame_rgb, (512, 512))

        # To tensor
        frame_np = frame_resized.astype(np.float32) / 255.0
        frame_np = (frame_np - 0.5) / 0.5
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).cuda()

        # Style frame
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        if use_temporal_coherence:
            styled_tensor = temporal_styler.process_frame(
                frame_tensor,
                use_optical_flow=use_optical_flow
            )
        else:
            with torch.no_grad():
                styled_tensor = model(frame_tensor)

        end.record()
        torch.cuda.synchronize()

        frame_time = start.elapsed_time(end)
        frame_times.append(frame_time)

        # Convert back to numpy
        styled_np = styled_tensor[0].cpu().permute(1, 2, 0).numpy()
        styled_np = ((styled_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

        # Resize back to original size
        styled_resized = cv2.resize(styled_np, (width, height))

        # Convert RGB to BGR for OpenCV
        styled_bgr = cv2.cvtColor(styled_resized, cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(styled_bgr)

        frame_idx += 1

        if frame_idx % 10 == 0:
            avg_time = np.mean(frame_times[-10:])
            fps_current = 1000.0 / avg_time
            progress = frame_idx / total_frames * 100
            print(f\"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - \"
                  f\"{avg_time:.2f}ms/frame ({fps_current:.1f} FPS)\")

    # Cleanup
    cap.release()
    out.release()

    # Statistics
    stats = {
        'total_frames': frame_idx,
        'avg_latency_ms': np.mean(frame_times),
        'std_latency_ms': np.std(frame_times),
        'avg_fps': 1000.0 / np.mean(frame_times),
        'total_time_sec': sum(frame_times) / 1000.0,
        'temporal_coherence': use_temporal_coherence,
        'optical_flow': use_optical_flow
    }

    print(f\"\\n✅ Video processing complete!\")
    print(f\"   Output: {output_path}\")
    print(f\"   Average: {stats['avg_latency_ms']:.2f} ms/frame ({stats['avg_fps']:.1f} FPS)\")
    print(f\"   Total time: {stats['total_time_sec']:.1f} seconds\\n\")

    return stats
"""

part3 = """
# ----------------------------------------
# Test Temporal Coherence
# ----------------------------------------

print("🧪 Testing temporal coherence...\\n")

# Create test video (synthetic)
print("Creating synthetic test video...")

def create_test_video(output_path, num_frames=60, fps=30):
    \\\"\"\\\"Create a simple test video with moving circle\\\"\"\\\"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))

    for i in range(num_frames):
        # Create frame with moving circle
        frame = np.zeros((512, 512, 3), dtype=np.uint8)

        # Circle position moves
        cx = int(256 + 150 * np.sin(2 * np.pi * i / num_frames))
        cy = int(256 + 150 * np.cos(2 * np.pi * i / num_frames))

        cv2.circle(frame, (cx, cy), 50, (255, 255, 255), -1)
        cv2.circle(frame, (256, 256), 100, (128, 128, 128), 2)

        out.write(frame)

    out.release()
    print(f\"✓ Created test video: {output_path}\\n\")

test_video_path = portfolio_dir / 'test_video.mp4'
create_test_video(test_video_path, num_frames=60, fps=30)

# Get style model
style_model = blender.create_blended_model({'starry_night': 1.0})

# Process WITHOUT temporal coherence
print(\"1️⃣  Processing WITHOUT temporal coherence...\\n\")

output_no_temporal = portfolio_dir / 'styled_no_temporal.mp4'
stats_no_temporal = process_video_file(
    test_video_path,
    output_no_temporal,
    style_model,
    use_temporal_coherence=False,
    max_frames=60
)

# Process WITH temporal coherence (simple blending)
print(\"\\n2️⃣  Processing WITH temporal coherence (simple)...\\n\")

output_temporal_simple = portfolio_dir / 'styled_temporal_simple.mp4'
stats_temporal_simple = process_video_file(
    test_video_path,
    output_temporal_simple,
    style_model,
    use_temporal_coherence=True,
    use_optical_flow=False,
    blend_factor=0.7,
    max_frames=60
)
"""

part4 = """
# Process WITH temporal coherence + optical flow (demo only)
print(\"\\n3️⃣  Optical flow warping (advanced):\\n\")
print(\"   Optical flow warping provides better motion compensation\")
print(\"   but adds computational overhead. Enable for production use.\\n\")

# ----------------------------------------
# Compare Results
# ----------------------------------------

print(\"\\n📊 Temporal Coherence Comparison:\\n\")

print(\"Method          | FPS    | Latency (ms)\")
print(\"----------------|--------|-------------\")

print(f\"No Temporal     | {stats_no_temporal['avg_fps']:.1f}    | {stats_no_temporal['avg_latency_ms']:.2f}\")
print(f\"Simple Blending | {stats_temporal_simple['avg_fps']:.1f}    | {stats_temporal_simple['avg_latency_ms']:.2f}\")

print(\"\\nKey Insights:\")
print(\"  • Temporal blending reduces flickering between frames\")
print(\"  • Optical flow warping handles motion better\")
print(\"  • Higher blend_factor = more stability, less responsiveness\")
print(\"  • Typical blend_factor: 0.6-0.8 for video\")
"""

part5 = """
# ----------------------------------------
# Save Temporal Styler Code
# ----------------------------------------

temporal_code = '''\"\"\\\"
StyleForge - Temporal Coherence for Video

Prevents flickering in video style transfer
\"\"\\\"

import torch
import cv2
import numpy as np
from collections import deque

class TemporalStyler:
    \\\"\\\"\"Video style transfer with temporal coherence\\\"\\\"\\\"

    def __init__(self, model, blend_factor=0.7):
        self.model = model
        self.blend_factor = blend_factor
        self.previous_styled = None
        self.frame_buffer = deque(maxlen=3)

    def reset(self):
        self.previous_styled = None
        self.frame_buffer.clear()

    def process_frame(self, frame_tensor, use_optical_flow=False):
        with torch.no_grad():
            current_styled = self.model(frame_tensor)

            if self.previous_styled is None:
                output = current_styled
            else:
                if use_optical_flow and len(self.frame_buffer) >= 2:
                    warped = self._warp_with_flow(
                        self.previous_styled,
                        self.frame_buffer[-2],
                        frame_tensor
                    )
                    output = self.blend_factor * warped + (1 - self.blend_factor) * current_styled
                else:
                    output = self.blend_factor * self.previous_styled + (1 - self.blend_factor) * current_styled

            self.previous_styled = output.clone()
            self.frame_buffer.append(frame_tensor)
            return output

    def _warp_with_flow(self, previous_styled, previous_frame, current_frame):
        # Optical flow computation and warping
        prev_np = previous_frame[0].cpu().permute(1, 2, 0).numpy()
        curr_np = current_frame[0].cpu().permute(1, 2, 0).numpy()
        prev_np = ((prev_np * 0.5 + 0.5) * 255).astype(np.uint8)
        curr_np = ((curr_np * 0.5 + 0.5) * 255).astype(np.uint8)
        prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                              pyr_scale=0.5, levels=3, winsize=15,
                                              iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        h, w = flow.shape[:2]
        flow_map = np.column_stack([(np.arange(w) + flow[:,:,0]).flatten(),
                                     (np.arange(h)[:,None] + flow[:,:,1]).flatten()]).reshape(h,w,2)

        styled_np = previous_styled[0].cpu().permute(1,2,0).numpy()
        styled_np = ((styled_np * 0.5 + 0.5) * 255).astype(np.uint8)

        warped = cv2.remap(styled_np, flow_map[:,:,0].astype(np.float32),
                          flow_map[:,:,1].astype(np.float32), cv2.INTER_LINEAR)

        warped = warped.astype(np.float32) / 255.0
        warped = (warped - 0.5) / 0.5
        return torch.from_numpy(warped).permute(2,0,1).unsqueeze(0).cuda()

# Usage:
# styler = TemporalStyler(model, blend_factor=0.7)
# styler.reset()
# for frame in video:
#     styled = styler.process_frame(frame_tensor)
'''
'''

temporal_path = project_root / 'utils' / 'temporal_styler.py'
with open(temporal_path, 'w') as f:
    f.write(temporal_code)

print(f\"✓ Saved temporal styler to {temporal_path}\")

# ----------------------------------------
# Summary
# ----------------------------------------

print(\"=\"*70)
print(\"  TEMPORAL COHERENCE FOR VIDEO COMPLETE\")
print(\"=\"*70)

print()
print(\"Features:\")
print(\"  - Flicker-free video style transfer\")
print(\"  - Configurable temporal blending factor\")
print(\"  - Optional optical flow warping for motion compensation\")
print(\"  - Frame buffer for multi-frame consistency\")
print()
print(\"Methods:\")
print(\"  - No Temporal: Process each frame independently (fast, flickers)\")
print(\"  - Simple Blending: Blend adjacent frames (good for slow motion)\")
print(\"  - Optical Flow: Warp-based alignment (best for fast motion)\")
print()
print(\"Use Cases:\")
print(\"  - Video stylization with consistent style\")
print(\"  - Real-time video processing\")
print(\"  - Animation style transfer\")
print(\"  - Webcam applications\")
print()
print(\"=\"*70)
print(\"\\n✅ Temporal coherence implementation complete!\")
"""

# Combine all parts
new_cell_code = part1 + '\n' + part2 + '\n' + part3 + '\n' + part4 + '\n' + part5

# Find where to insert (after CELL 16 Gradio Web Interface)
insert_index = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 16: Gradio Web Interface' in source:
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
