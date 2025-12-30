#!/usr/bin/env python3
"""
ONNX Export Script for UniDepth V2 with Camera Input and Uncertainty
Extends UniDepth's export.py to support camera intrinsics input and uncertainty output.
"""

import argparse
import json
import os
import sys
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import shape_inference
import huggingface_hub

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from unidepth.models.unidepthv2 import UniDepthV2


# Monkey patch F.interpolate to disable antialiasing for ONNX export
# TensorRT doesn't support antialiasing in Resize operations
_original_interpolate = F.interpolate

def interpolate_no_antialias(*args, **kwargs):
    """Wrapper for F.interpolate that removes antialias argument"""
    # Remove antialias argument if present
    kwargs.pop('antialias', None)
    return _original_interpolate(*args, **kwargs)

# Apply patch
F.interpolate = interpolate_no_antialias


class UniDepthV2ONNXWithIntrinsics(UniDepthV2):
    """
    ONNX-compatible wrapper for UniDepth V2 with camera intrinsics input

    Inputs:
        - rgb: [B, 3, H, W] - RGB image (normalized with ImageNet stats)
        - intrinsics: [B, 4] - Camera intrinsics [fx, fy, cx, cy]

    Outputs:
        - pts_3d: [B, 3, H, W] - 3D points in camera coordinates
        - depth: [B, 1, H, W] - Depth map (Z coordinate)
        - confidence: [B, 1, H, W] - Confidence map
        - uncertainty: [B, 1, H, W] - Uncertainty (1/confidence)
        - pred_intrinsics: [B, 4] - Predicted intrinsics
    """

    def __init__(self, config, eps: float = 1e-6, **kwargs):
        super().__init__(config, eps)

    def forward(self, rgbs, intrinsics):
        """
        Forward pass with camera intrinsics

        Args:
            rgbs: [B, 3, H, W] - RGB image (already normalized)
            intrinsics: [B, 4] - Camera intrinsics [fx, fy, cx, cy]

        Returns:
            pts_3d, depth, confidence, uncertainty, pred_intrinsics
        """
        B, _, H, W = rgbs.shape
        device = rgbs.device
        dtype = rgbs.dtype

        # Convert intrinsics to camera rays
        # This matches the preprocessing in UniDepth's infer() method
        fx = intrinsics[:, 0:1, None, None]  # [B, 1, 1, 1]
        fy = intrinsics[:, 1:2, None, None]
        cx = intrinsics[:, 2:3, None, None]
        cy = intrinsics[:, 3:4, None, None]

        # Create pixel coordinate grids
        y_coords = torch.arange(H, device=device, dtype=dtype)
        x_coords = torch.arange(W, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Expand to batch
        x_grid = x_grid[None, None, :, :].expand(B, 1, H, W)
        y_grid = y_grid[None, None, :, :].expand(B, 1, H, W)

        # Compute ray directions (inverse projection)
        rays_x = (x_grid - cx) / fx
        rays_y = (y_grid - cy) / fy
        rays_z = torch.ones_like(rays_x)

        # Stack and normalize rays
        camera_rays = torch.cat([rays_x, rays_y, rays_z], dim=1)  # [B, 3, H, W]
        camera_rays = F.normalize(camera_rays, p=2, dim=1)

        # Forward pass through encoder
        features, tokens = self.pixel_encoder(rgbs)

        # Prepare inputs for decoder (matches UniDepthV2ONNXcam)
        inputs = {}
        inputs["image"] = rgbs
        inputs["rays"] = camera_rays
        inputs["features"] = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        inputs["tokens"] = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]

        # Decode
        outputs = self.pixel_decoder(inputs, [])
        outputs["rays"] = outputs["rays"].permute(0, 2, 1).reshape(B, 3, H, W)

        # Compute 3D points
        pts_3d = outputs["rays"] * outputs["radius"]

        # Extract depth (Z coordinate)
        depth = pts_3d[:, 2:3, :, :]

        # Get confidence and intrinsics
        confidence = outputs["confidence"]
        pred_intrinsics = outputs["intrinsics"]

        # Compute uncertainty (inverse of confidence)
        uncertainty = 1.0 / (confidence + 1e-6)

        return pts_3d, depth, confidence, uncertainty, pred_intrinsics


def export_to_onnx(
    model_name: str,
    output_path: str,
    input_height: int = 518,
    input_width: int = 518,
    opset_version: int = 17,
    simplify: bool = True,
    verbose: bool = False
):
    """
    Export UniDepth V2 model to ONNX format with camera intrinsics input

    Args:
        model_name: HuggingFace model name (e.g., "lpiccinelli/unidepth-v2-vitl14")
        output_path: Path to save ONNX model
        input_height: Input image height (must be multiple of 14)
        input_width: Input image width (must be multiple of 14)
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model
        verbose: Enable verbose logging
    """

    # Validate and round input size to multiple of 14
    input_height = 14 * ceil(input_height / 14)
    input_width = 14 * ceil(input_width / 14)

    print(f"Exporting UniDepth V2 with camera intrinsics to ONNX...")
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")
    print(f"Input size: {input_height}x{input_width}")
    print(f"Opset version: {opset_version}")

    # Extract version and backbone from model name
    # e.g., "lpiccinelli/unidepth-v2-vitl14" -> version="v2", backbone="vitl"
    parts = model_name.split("/")[-1].split("-")
    version = parts[1]  # "v2"
    backbone = parts[2].replace("14", "")  # "vitl"

    print(f"\nLoading model configuration...")
    print(f"  Version: {version}")
    print(f"  Backbone: {backbone}")

    # Load config
    config_path = os.path.join(
        os.path.dirname(__file__),
        f"../../configs/config_{version}_{backbone}14.json"
    )

    with open(config_path) as f:
        config = json.load(f)

    # Disable efficient attention for ONNX export
    config["training"]["export"] = True
    print(f"  ✓ Config loaded (export mode enabled)")

    # Create ONNX model
    print(f"\nCreating ONNX-compatible model...")
    model = UniDepthV2ONNXWithIntrinsics(config)

    # Load pretrained weights
    print(f"Downloading pretrained weights from HuggingFace...")
    checkpoint_path = huggingface_hub.hf_hub_download(
        repo_id=model_name,
        filename="pytorch_model.bin",
        repo_type="model",
    )

    info = model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
    print(f"  ✓ Weights loaded")
    if info.missing_keys:
        print(f"    Missing keys: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"    Unexpected keys: {info.unexpected_keys}")

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    print(f"  ✓ Model ready on {device}")

    # Create dummy inputs
    dummy_rgb = torch.randn(1, 3, input_height, input_width, device=device)
    dummy_intrinsics = torch.tensor(
        [[input_width * 1.0, input_height * 1.0, input_width / 2.0, input_height / 2.0]],
        device=device,
        dtype=dummy_rgb.dtype
    )

    print(f"\nTesting forward pass...")
    with torch.no_grad():
        pts_3d, depth, confidence, uncertainty, pred_intrinsics = model(
            dummy_rgb,
            dummy_intrinsics
        )

    print(f"  Output shapes:")
    print(f"    pts_3d: {pts_3d.shape}")
    print(f"    depth: {depth.shape}")
    print(f"    confidence: {confidence.shape}")
    print(f"    uncertainty: {uncertainty.shape}")
    print(f"    pred_intrinsics: {pred_intrinsics.shape}")
    print(f"  ✓ Forward pass successful")

    # Export to ONNX
    print(f"\nExporting to ONNX...")

    dynamic_axes = {
        'rgb': {0: 'batch'},
        'intrinsics': {0: 'batch'},
        'pts_3d': {0: 'batch'},
        'depth': {0: 'batch'},
        'confidence': {0: 'batch'},
        'uncertainty': {0: 'batch'},
        'pred_intrinsics': {0: 'batch'},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_rgb, dummy_intrinsics),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['rgb', 'intrinsics'],
            output_names=['pts_3d', 'depth', 'confidence', 'uncertainty', 'pred_intrinsics'],
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

    print(f"  ✓ ONNX export complete")

    # Load and verify ONNX model
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX model verification passed")

    # Apply shape inference
    print(f"Applying shape inference...")
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)

    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print(f"Simplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, output_path)
                print(f"  ✓ ONNX model simplified successfully")
            else:
                print(f"  ⚠ Warning: ONNX simplification check failed")
        except ImportError:
            print(f"  ⚠ Warning: onnx-simplifier not installed, skipping")

    # Print model info
    print(f"\nModel Information:")
    print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
    print(f"  Nodes: {len(onnx_model.graph.node)}")

    print(f"\n✓ Export complete!")
    print(f"  File: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Export UniDepth V2 to ONNX with camera intrinsics and uncertainty'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='lpiccinelli/unidepth-v2-vitl14',
        choices=[
            'lpiccinelli/unidepth-v2-vits14',
            'lpiccinelli/unidepth-v2-vitb14',
            'lpiccinelli/unidepth-v2-vitl14',
        ],
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save ONNX model (.onnx file)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=518,
        help='Input image height (will be rounded to multiple of 14)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=518,
        help='Input image width (will be rounded to multiple of 14)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX model simplification'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    # Export to ONNX
    export_to_onnx(
        model_name=args.model,
        output_path=args.output,
        input_height=args.height,
        input_width=args.width,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
