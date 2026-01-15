"""
Update notebook with CELL 18 - Real-Time Webcam Demo
"""
import json

# Read notebook
with open('notebooks/demo.ipynb', 'r') as f:
    nb = json.load(f)

# New markdown header
new_cell_md = "## CELL 18: Real-Time Webcam Demo"

# New code cell - split into parts
part1 = r"""# ============================================
# 📷 REAL-TIME WEBCAM DEMO
# ============================================

print("Setting up real-time webcam demo...\n")

import threading
import queue
from IPython.display import display, HTML, clear_output
import matplotlib.animation as animation

# Note: In Colab, webcam access is limited
# This code demonstrates the approach - works better locally or on deployed server

# ----------------------------------------
# Webcam Processor Class
# ----------------------------------------

class WebcamStyler:
    \"\"\"Real-time webcam style transfer.\"\"\"

    def __init__(self, model, target_fps=30):
        \"\"\"Initialize webcam styler.

        Args:
            model: Style transfer model
            target_fps: Target frames per second
        \"\"\"
        self.model = model
        self.target_fps = target_fps
        self.frame_time_target = 1.0 / target_fps

        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.stats_queue = queue.Queue(maxsize=10)

        self.temporal_styler = TemporalStyler(model, blend_factor=0.5)

    def process_webcam(
        self,
        camera_id=0,
        display_size=(640, 480),
        use_temporal=True
    ):
        \"\"\"Process webcam feed in real-time.

        Args:
            camera_id: Webcam device ID
            display_size: Display resolution
            use_temporal: Use temporal coherence
        \"\"\"
        print(f"🎥 Opening webcam (device {camera_id})...\\n")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("❌ Could not open webcam")
            print("   (Note: Webcam access may be limited in Colab)")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])

        print("✅ Webcam opened")
        print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"   Target FPS: {self.target_fps}\\n")
        print("Press 'q' to quit\\n")

        if use_temporal:
            self.temporal_styler.reset()

        # Warmup
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, 512, 512).cuda()
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(dummy_input)
        print("✓ Warmup complete\\n")

        # Processing loop
        frame_count = 0
        fps_history = deque(maxlen=30)

        print("🎬 Starting real-time processing...")
        print("="*60)

        try:
            while True:
                loop_start = time.time()

                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to 512×512 for model
                frame_resized = cv2.resize(frame_rgb, (512, 512))

                # To tensor
                frame_np = frame_resized.astype(np.float32) / 255.0
                frame_np = (frame_np - 0.5) / 0.5
                frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).cuda()

                # Style transfer
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

                if use_temporal:
                    styled_tensor = self.temporal_styler.process_frame(frame_tensor)
                else:
                    with torch.no_grad():
                        styled_tensor = self.model(frame_tensor)

                end.record()
                torch.cuda.synchronize()

                process_time = start.elapsed_time(end) / 1000.0  # Convert to seconds

                # Convert back to display format
                styled_np = styled_tensor[0].cpu().permute(1, 2, 0).numpy()
                styled_np = ((styled_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

                # Resize back
                styled_display = cv2.resize(styled_np, display_size)
                frame_display = cv2.resize(frame_rgb, display_size)

                # Create side-by-side display
                combined = np.hstack([frame_display, styled_display])

                # Add FPS overlay
                current_fps = 1.0 / process_time if process_time > 0 else 0
                fps_history.append(current_fps)
                avg_fps = np.mean(fps_history)

                cv2.putText(
                    combined,
                    f'FPS: {avg_fps:.1f}  |  Latency: {process_time*1000:.1f}ms',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    combined,
                    'Original',
                    (10, display_size[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    combined,
                    'Styled (CUDA Optimized)',
                    (display_size[0] + 10, display_size[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # Display (Note: cv2.imshow doesn't work in Colab)
                # For Colab, we'd need to use different display method
                # cv2.imshow('StyleForge - Real-Time', combined)

                # For demonstration, save frame to show it works
                if frame_count % 30 == 0:  # Save every 30 frames
                    cv2.imwrite(
                        str(portfolio_dir / f'webcam_frame_{frame_count}.jpg'),
                        cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                    )

                frame_count += 1

                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: {avg_fps:.1f} FPS, "
                          f"{process_time*1000:.1f}ms latency")

                # Check for quit (works in local OpenCV window)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # Limit for demo in Colab
                if frame_count >= 90:  # Process 3 seconds
                    break

                # Frame rate limiting
                loop_time = time.time() - loop_start
                if loop_time < self.frame_time_target:
                    time.sleep(self.frame_time_target - loop_time)

        finally:
            cap.release()
            # cv2.destroyAllWindows()

            print("\\n" + "="*60)
            print(f"✅ Processed {frame_count} frames")
            print(f"   Average FPS: {np.mean(fps_history):.1f}")
            print(f"   Average latency: {np.mean([1/f for f in fps_history if f > 0])*1000:.1f}ms")
"""

part2 = """
# ----------------------------------------
# Alternative: Image Sequence Demo
# ----------------------------------------

print("💡 Webcam demo code ready (works best locally/deployed)\\n")
print("   In Colab, webcam access is limited")
print("   Creating alternative demo with image sequence...\\n")

def create_demo_sequence():
    \\\"\\\"Create a demo showing real-time capability

    Using static images instead of webcam
    \\\"\\\"\"
    print("Creating demo frames...\\n")

    # Create test images
    test_images = []
    for i in range(10):
        img = torch.randn(1, 3, 512, 512).cuda()
        test_images.append(img)

    # Process with timing
    style_model = blender.create_blended_model({'starry_night': 1.0})
    temporal_styler = TemporalStyler(style_model, blend_factor=0.6)
    temporal_styler.reset()

    results = []
    times = []

    print("Processing frames...")
    for i, img in enumerate(test_images):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        styled = temporal_styler.process_frame(img, use_optical_flow=False)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        results.append(styled)
        times.append(elapsed)

        print(f"  Frame {i+1}/10: {elapsed:.2f}ms ({1000/elapsed:.1f} FPS)")

    avg_time = np.mean(times)
    avg_fps = 1000 / avg_time

    print(f"\\n✅ Average: {avg_time:.2f}ms ({avg_fps:.1f} FPS)")

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, styled in enumerate(results):
        img = styled[0].cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f'Frame {i+1}\\n{times[i]:.1f}ms', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Real-Time Processing Demo - Average: {avg_fps:.1f} FPS',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(portfolio_dir / 'realtime_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return avg_fps

demo_fps = create_demo_sequence()

print(f"\\n✅ Real-time demo complete!")
print(f"   Achieved: {demo_fps:.1f} FPS")

if demo_fps >= 30:
    print(f"   🎉 Real-time performance achieved (>30 FPS)!")
elif demo_fps >= 24:
    print(f"   ✅ Smooth video performance (>24 FPS)")
else:
    print(f"   ⚠️  Below real-time threshold")
"""

part3 = """
# ----------------------------------------
# Save Webcam Code
# ----------------------------------------

webcam_code = '''\"\"\\\"
StyleForge - Real-Time Webcam Demo

Process webcam feed in real-time with style transfer
\"\"\\\"

import cv2
import torch
import numpy as np
from collections import deque

class WebcamStyler:
    \\\"\\\"\"Real-time webcam style transfer\\\"\\\"\\\"

    def __init__(self, model, target_fps=30):
        self.model = model
        self.target_fps = target_fps
        self.frame_time_target = 1.0 / target_fps

    def process_webcam(self, camera_id=0, display_size=(640, 480), use_temporal=True):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f\"Could not open webcam {camera_id}\")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])

        temporal = TemporalStyler(self.model, blend_factor=0.6)
        temporal.reset()

        fps_history = deque(maxlen=30)
        print(\"Starting webcam processing... (press 'q' to quit)\")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (512, 512))

            frame_np = frame_resized.astype(np.float32) / 255.0
            frame_np = (frame_np - 0.5) / 0.5
            frame_tensor = torch.from_numpy(frame_np).permute(2,0,1).unsqueeze(0).cuda()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            styled = temporal.process_frame(frame_tensor)
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end)
            fps = 1000.0 / elapsed_ms
            fps_history.append(fps)

            styled_np = styled[0].cpu().permute(1,2,0).numpy()
            styled_np = ((styled_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            styled_display = cv2.resize(styled_np, display_size)

            cv2.putText(styled_display, f'FPS: {np.mean(fps_history):.1f}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('StyleForge Real-Time', cv2.cvtColor(styled_display, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f\"Average FPS: {np.mean(fps_history):.1f}\")

# Usage:
# model = OptimizedStyleTransferNetwork().cuda().eval()
# webcam = WebcamStyler(model, target_fps=30)
# webcam.process_webcam(camera_id=0)
'''
'''

webcam_path = project_root / 'utils' / 'webcam_styler.py'
with open(webcam_path, 'w') as f:
    f.write(webcam_code)

print(f\"✓ Saved webcam styler to {webcam_path}\")

# ----------------------------------------
# Summary
# ----------------------------------------

print(\"=\"*70)
print(\"  REAL-TIME WEBCAM DEMO COMPLETE\")
print(\"=\"*70)

print()
print(\"Features:\")
print(\"  - Real-time webcam style transfer\")
print(\"  - Temporal coherence for stable video\")
print(\"  - FPS tracking and display\")
print(\"  - Side-by-side comparison view\")
print()
print(\"Performance Targets:\")
print(\"  - 60 FPS: Ultra-smooth (high-end GPUs)\")
print(\"  - 30 FPS: Real-time standard (RTX 3060+)\")
print(\"  - 24 FPS: Smooth video (GTX 1660+)\")
print()
print(\"Deployment Options:\")
print(\"  - Local: cv2.imshow() window\")
print(\"  - Web: Flask/FastAPI + WebSocket streaming\")
print(\"  - Mobile: TorchScript + CoreML\")
print(\"  - Edge: ONNX Runtime + TensorRT\")
print()
print(\"=\"*70)
print(\"\\n✅ Real-time webcam demo complete!\")
"""

# Combine all parts
new_cell_code = part1 + '\n' + part2 + '\n' + part3

# Find where to insert (after CELL 17 Temporal Coherence)
insert_index = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 17: Temporal Coherence' in source:
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
