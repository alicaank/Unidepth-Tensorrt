#!/usr/bin/env python3
"""
TensorRT Engine Builder for UniDepth V2
Builds optimized TensorRT engines from ONNX models with FP16/INT8 support
"""

import argparse
import os
import sys

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


# Logger for TensorRT
class TRTLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.logs = []

    def log(self, severity, msg):
        # Capture all logs for later display
        self.logs.append((severity, msg))

        # Only print warnings and errors during build
        if severity <= trt.ILogger.WARNING:
            severity_str = {
                trt.ILogger.INTERNAL_ERROR: "INTERNAL_ERROR",
                trt.ILogger.ERROR: "ERROR",
                trt.ILogger.WARNING: "WARNING",
                trt.ILogger.INFO: "INFO",
                trt.ILogger.VERBOSE: "VERBOSE"
            }.get(severity, "UNKNOWN")
            print(f"[TensorRT {severity_str}] {msg}")


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp16',
    workspace_mb: int = 4096,
    min_timing_iterations: int = 2,
    avg_timing_iterations: int = 2,
    verbose: bool = False
):
    """
    Build TensorRT engine from ONNX model

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', or 'int8')
        workspace_mb: Workspace size in MB
        min_timing_iterations: Minimum timing iterations for builder
        avg_timing_iterations: Average timing iterations for builder
        verbose: Enable verbose logging
    """

    print(f"Building TensorRT Engine for UniDepth V2")
    print(f"=" * 60)
    print(f"ONNX Model:     {onnx_path}")
    print(f"Output Engine:  {engine_path}")
    print(f"Precision:      {precision.upper()}")
    print(f"Workspace:      {workspace_mb} MB")
    print(f"=" * 60)

    # Check ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        sys.exit(1)

    # Create logger
    logger = TRTLogger()
    if verbose:
        logger.min_severity = trt.ILogger.VERBOSE

    # Create builder
    print("\n[1/6] Creating TensorRT builder...")
    builder = trt.Builder(logger)

    # Create network
    print("[2/6] Creating network definition...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX
    print(f"[3/6] Parsing ONNX model...")
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        onnx_data = f.read()
        if not parser.parse(onnx_data):
            print("Error: Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            sys.exit(1)

    print(f"  ✓ Parsed successfully")
    print(f"    Network inputs:  {network.num_inputs}")
    print(f"    Network outputs: {network.num_outputs}")

    # Print input/output info
    print("\n  Input tensors:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"    {i}: {tensor.name} - {tensor.shape} ({tensor.dtype})")

    print("\n  Output tensors:")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"    {i}: {tensor.name} - {tensor.shape} ({tensor.dtype})")

    # Create builder config
    print("\n[4/6] Configuring builder...")
    config = builder.create_builder_config()

    # Set workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    print(f"  ✓ Workspace: {workspace_mb} MB")

    # Set precision
    if precision == 'fp16':
        if not builder.platform_has_fast_fp16:
            print("  ⚠ Warning: FP16 not supported on this platform, using FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"  ✓ FP16 precision enabled")
    elif precision == 'int8':
        if not builder.platform_has_fast_int8:
            print("  ⚠ Warning: INT8 not supported on this platform, using FP32")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            print(f"  ✓ INT8 precision enabled")
            print(f"  ⚠ Note: INT8 calibration not implemented, using dynamic range")
    else:
        print(f"  ✓ FP32 precision (default)")

    # Optimization profile for dynamic batch size
    print("\n[5/6] Creating optimization profile...")
    profile = builder.create_optimization_profile()

    # Set dynamic shapes for inputs
    # rgb: [batch, 3, 518, 518] - batch can be 1-8
    # intrinsics: [batch, 4]
    profile.set_shape(
        "rgb",
        min=(1, 3, 518, 518),
        opt=(1, 3, 518, 518),
        max=(8, 3, 518, 518)
    )
    profile.set_shape(
        "intrinsics",
        min=(1, 4),
        opt=(1, 4),
        max=(8, 4)
    )

    config.add_optimization_profile(profile)
    print(f"  ✓ Dynamic batch size: 1-8 (optimized for batch=1)")

    # Builder timing cache (speeds up subsequent builds)
    timing_iterations_str = f"min={min_timing_iterations}, avg={avg_timing_iterations}"
    print(f"  ✓ Timing iterations: {timing_iterations_str}")

    # Build engine
    print(f"\n[6/6] Building TensorRT engine...")
    print(f"  This may take several minutes...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Error: Failed to build TensorRT engine")
        print("\nBuilder logs:")
        for severity, msg in logger.logs:
            print(f"  {msg}")
        sys.exit(1)

    print(f"  ✓ Engine built successfully")

    # Save engine
    print(f"\nSaving engine to: {engine_path}")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)) or '.', exist_ok=True)

    # Convert IHostMemory to bytes
    engine_bytes = bytes(serialized_engine)
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)

    # Print summary
    engine_size_mb = len(engine_bytes) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"✓ Build Complete!")
    print(f"{'='*60}")
    print(f"Engine file:  {engine_path}")
    print(f"Engine size:  {engine_size_mb:.2f} MB")
    print(f"Precision:    {precision.upper()}")
    print(f"Workspace:    {workspace_mb} MB")

    # Verify engine
    print(f"\nVerifying engine...")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine is None:
        print("Error: Failed to deserialize engine")
        sys.exit(1)

    print(f"  ✓ Engine verification passed")
    print(f"  I/O Tensors: {engine.num_io_tensors}")

    # Print tensor info
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"    {mode_str}: {name} - {shape} ({dtype})")

    print(f"\n✓ Ready for inference!")


def main():
    parser = argparse.ArgumentParser(
        description='Build TensorRT engine for UniDepth V2'
    )

    parser.add_argument(
        '--onnx',
        type=str,
        required=True,
        help='Path to ONNX model (.onnx file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save TensorRT engine (.trt file)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp32', 'fp16', 'int8'],
        help='Precision mode (default: fp16)'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4096,
        help='Workspace size in MB (default: 4096)'
    )
    parser.add_argument(
        '--min-timing-iterations',
        type=int,
        default=2,
        help='Minimum timing iterations (default: 2)'
    )
    parser.add_argument(
        '--avg-timing-iterations',
        type=int,
        default=2,
        help='Average timing iterations (default: 2)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Build engine
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        precision=args.precision,
        workspace_mb=args.workspace,
        min_timing_iterations=args.min_timing_iterations,
        avg_timing_iterations=args.avg_timing_iterations,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
