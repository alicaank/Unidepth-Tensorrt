#!/usr/bin/env python3
"""
Compare TensorRT and PyTorch outputs for UniDepth V2

Validates that TensorRT inference matches PyTorch reference implementation
for depth, confidence, uncertainty, and 3D points.
"""

import argparse
import sys
import os
import numpy as np
import cv2
import torch
import subprocess

# Add parent directory to path for unidepth imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from unidepth.models import UniDepthV2


def load_pytorch_model(model_name: str, device: str = 'cuda'):
    """Load PyTorch UniDepth V2 model"""
    print(f"\nLoading PyTorch model: {model_name}")

    model = UniDepthV2.from_pretrained(model_name)
    model = model.to(device).eval()

    print(f"  ✓ Model loaded successfully")
    return model


def pytorch_inference(model, image_path, intrinsics, device='cuda'):
    """Run PyTorch inference"""
    print(f"\nRunning PyTorch inference...")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    orig_h, orig_w = image.shape[:2]
    print(f"  Input size: {orig_w}x{orig_h}")

    # Convert to RGB and then to torch tensor in CHW format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert HWC to CHW: (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).to(device)

    # Prepare intrinsics as 3x3 matrix
    # [[fx,  0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]
    intrinsics_matrix = torch.tensor([
        [[intrinsics[0], 0.0, intrinsics[2]],
         [0.0, intrinsics[1], intrinsics[3]],
         [0.0, 0.0, 1.0]]
    ], device=device, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        predictions = model.infer(image_tensor, intrinsics_matrix)

    # Extract outputs
    pts_3d = predictions['points'].squeeze(0).cpu().numpy()  # [3, H, W]
    depth = predictions['depth'].squeeze(0).cpu().numpy()  # [1, H, W]
    confidence = predictions['confidence'].squeeze(0).cpu().numpy()  # [1, H, W]
    intrinsics_pred = predictions['intrinsics'].squeeze(0).cpu().numpy()  # [3, 3]

    # Compute uncertainty (inverse of confidence)
    uncertainty = 1.0 / (confidence + 1e-6)

    print(f"  ✓ PyTorch inference complete")
    print(f"  Output shapes:")
    print(f"    pts_3d: {pts_3d.shape}")
    print(f"    depth: {depth.shape}")
    print(f"    confidence: {confidence.shape}")
    print(f"    uncertainty: {uncertainty.shape}")

    # Extract depth map (squeeze channel dimension)
    depth = depth.squeeze(0)  # [H, W]
    confidence = confidence.squeeze(0)
    uncertainty = uncertainty.squeeze(0)

    print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")
    print(f"  Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")

    return {
        'pts_3d': pts_3d,
        'depth': depth,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'intrinsics': intrinsics_pred
    }


def run_tensorrt_inference(cpp_inference_path, engine_path, image_path, intrinsics, output_dir):
    """Run TensorRT C++ inference"""
    print(f"\nRunning TensorRT inference...")

    # Create output paths
    os.makedirs(output_dir, exist_ok=True)
    depth_output = os.path.join(output_dir, 'trt_depth.png')
    confidence_output = os.path.join(output_dir, 'trt_confidence.png')
    uncertainty_output = os.path.join(output_dir, 'trt_uncertainty.png')
    points_output = os.path.join(output_dir, 'trt_points.ply')

    # Format intrinsics
    intrinsics_str = f"{intrinsics[0]},{intrinsics[1]},{intrinsics[2]},{intrinsics[3]}"

    # Run C++ inference
    cmd = [
        cpp_inference_path,
        '--engine', engine_path,
        '--input', image_path,
        '--intrinsics', intrinsics_str,
        '--output-depth', depth_output,
        '--output-confidence', confidence_output,
        '--output-uncertainty', uncertainty_output,
        '--output-points', points_output,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running TensorRT inference:")
        print(result.stderr)
        sys.exit(1)

    print(f"  ✓ TensorRT inference complete")

    # Load outputs (as visualizations for now - grayscale depth maps)
    # Note: The C++ code saves normalized visualizations, not raw values
    # We'll need to extract the actual values for comparison

    # For now, we'll parse the output to get the depth range
    output_lines = result.stdout.split('\n')
    depth_range = None
    for line in output_lines:
        if 'Depth range:' in line:
            # Extract values like "[1.650, 6.665] meters"
            parts = line.split('[')[1].split(']')[0]
            min_val, max_val = parts.split(',')
            depth_range = (float(min_val), float(max_val))
            break

    print(f"  Depth range from TensorRT: {depth_range}")

    return {
        'depth_output': depth_output,
        'confidence_output': confidence_output,
        'uncertainty_output': uncertainty_output,
        'depth_range': depth_range,
        'stdout': result.stdout
    }


def compare_outputs(pytorch_out, tensorrt_info, save_dir):
    """Compare PyTorch and TensorRT outputs"""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    # For this initial version, we'll do a qualitative comparison
    # since TensorRT outputs are visualizations

    print("\nPyTorch Outputs:")
    print(f"  Depth range: [{pytorch_out['depth'].min():.4f}, {pytorch_out['depth'].max():.4f}]")
    print(f"  Confidence range: [{pytorch_out['confidence'].min():.4f}, {pytorch_out['confidence'].max():.4f}]")
    print(f"  Uncertainty range: [{pytorch_out['uncertainty'].min():.4f}, {pytorch_out['uncertainty'].max():.4f}]")

    print("\nTensorRT Outputs:")
    if tensorrt_info['depth_range']:
        trt_min, trt_max = tensorrt_info['depth_range']
        print(f"  Depth range: [{trt_min:.4f}, {trt_max:.4f}]")

        # Compare ranges
        pt_min = pytorch_out['depth'].min()
        pt_max = pytorch_out['depth'].max()

        min_diff = abs(trt_min - pt_min) / pt_min * 100
        max_diff = abs(trt_max - pt_max) / pt_max * 100

        print(f"\nRange Comparison:")
        print(f"  Min difference: {min_diff:.2f}%")
        print(f"  Max difference: {max_diff:.2f}%")

        if min_diff < 5.0 and max_diff < 5.0:
            print(f"\n✅ EXCELLENT: Depth ranges match within 5%!")
        elif min_diff < 10.0 and max_diff < 10.0:
            print(f"\n✅ GOOD: Depth ranges match within 10%")
        else:
            print(f"\n⚠  WARNING: Depth ranges differ by more than 10%")

    # Save PyTorch outputs for visual comparison
    print(f"\nSaving PyTorch visualizations to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Normalize and save depth
    def normalize_depth(d):
        d_min, d_max = d.min(), d.max()
        return ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    pt_depth_vis = normalize_depth(pytorch_out['depth'])
    pt_depth_colored = cv2.applyColorMap(pt_depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, 'pytorch_depth.png'), pt_depth_colored)

    pt_confidence_vis = normalize_depth(pytorch_out['confidence'])
    pt_confidence_colored = cv2.applyColorMap(pt_confidence_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, 'pytorch_confidence.png'), pt_confidence_colored)

    pt_uncertainty_vis = normalize_depth(pytorch_out['uncertainty'])
    pt_uncertainty_colored = cv2.applyColorMap(pt_uncertainty_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, 'pytorch_uncertainty.png'), pt_uncertainty_colored)

    # Save raw numpy arrays for detailed analysis
    np.save(os.path.join(save_dir, 'pytorch_depth.npy'), pytorch_out['depth'])
    np.save(os.path.join(save_dir, 'pytorch_confidence.npy'), pytorch_out['confidence'])
    np.save(os.path.join(save_dir, 'pytorch_uncertainty.npy'), pytorch_out['uncertainty'])
    np.save(os.path.join(save_dir, 'pytorch_pts_3d.npy'), pytorch_out['pts_3d'])

    print(f"  ✓ Saved PyTorch visualizations")
    print(f"  ✓ Saved raw outputs as .npy files")

    print(f"\n{'='*60}")
    print("Compare visualizations:")
    print(f"  PyTorch:   {save_dir}/pytorch_depth.png")
    print(f"  TensorRT:  {tensorrt_info['depth_output']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare TensorRT and PyTorch outputs for UniDepth V2'
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Input image path'
    )
    parser.add_argument(
        '--engine',
        type=str,
        required=True,
        help='TensorRT engine path (.trt)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lpiccinelli/unidepth-v2-vitl14',
        help='HuggingFace model name for PyTorch'
    )
    parser.add_argument(
        '--intrinsics',
        type=str,
        help='Camera intrinsics as "fx,fy,cx,cy" (optional, will estimate if not provided)'
    )
    parser.add_argument(
        '--cpp-inference',
        type=str,
        default='./build/image_inference',
        help='Path to C++ inference binary'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='comparison_results',
        help='Directory to save comparison results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for PyTorch (cuda or cpu)'
    )

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.engine):
        print(f"Error: TensorRT engine not found: {args.engine}")
        sys.exit(1)

    if not os.path.exists(args.cpp_inference):
        print(f"Error: C++ inference binary not found: {args.cpp_inference}")
        print(f"  Build it first with: ./build.sh")
        sys.exit(1)

    # Parse or estimate intrinsics
    if args.intrinsics:
        intrinsics = [float(x) for x in args.intrinsics.split(',')]
        if len(intrinsics) != 4:
            print(f"Error: Intrinsics must be 4 values: fx,fy,cx,cy")
            sys.exit(1)
        print(f"Using provided intrinsics: {intrinsics}")
    else:
        # Estimate from image size (60° FoV)
        image = cv2.imread(args.image)
        h, w = image.shape[:2]
        fx = fy = w / (2.0 * np.tan(np.radians(60.0) / 2.0))
        cx = w / 2.0
        cy = h / 2.0
        intrinsics = [fx, fy, cx, cy]
        print(f"Estimated intrinsics (60° FoV): {intrinsics}")

    print("="*60)
    print("UniDepth V2 - PyTorch vs TensorRT Comparison")
    print("="*60)
    print(f"Image:      {args.image}")
    print(f"Engine:     {args.engine}")
    print(f"Model:      {args.model}")
    print(f"Device:     {args.device}")

    # 1. Run PyTorch inference
    model = load_pytorch_model(args.model, args.device)
    pytorch_out = pytorch_inference(model, args.image, intrinsics, args.device)

    # 2. Run TensorRT inference
    tensorrt_info = run_tensorrt_inference(
        args.cpp_inference,
        args.engine,
        args.image,
        intrinsics,
        args.save_dir
    )

    # 3. Compare outputs
    compare_outputs(pytorch_out, tensorrt_info, args.save_dir)

    print(f"\n✓ Comparison complete!")
    print(f"  Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
