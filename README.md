# UniDepth V2 TensorRT (C++ Inference)

High-performance C++/CUDA + TensorRT implementation of **UniDepth V2** with **camera intrinsics input** and **uncertainty/confidence outputs**.

- **C++ API**: `include/unidepth_v2.h`
- **CUDA kernels**: `kernels/`
- **CLI demo**: `examples/image_inference.cpp` (built as `build/image_inference`)
- **Python tooling** (optional): export ONNX + build TensorRT engines


## What you get

### Inputs

- **RGB image**: any size (resized internally to `518x518`)
- **Camera intrinsics**: `fx, fy, cx, cy`

### Outputs

- **Depth**: metric depth in meters
- **Confidence**: higher is better
- **Uncertainty**: higher is worse (computed as `1 / (confidence + 1e-6)`)
- **Point cloud**: `PLY` with `H*W` vertices

## Optimizations

- FP16 engines
- CUDA Graph support (optional)
- GPU preprocessing / postprocessing (CUDA kernels)

## Prerequisites

- CUDA Toolkit
- TensorRT (set `TensorRT_ROOT` if not installed in a default location)
- OpenCV development headers
- CMake >= 3.18
- A C++17 compiler

## Build (C++)

```bash
./build.sh
```

This produces:

- `build/image_inference`

## Run image inference

```bash
./build/image_inference \
  --engine engines/unidepth_v2_vitl14_fp16.trt \
  --input /path/to/image.png \
  --intrinsics "525.0,525.0,319.5,239.5" \
  --output-depth depth.png
```

Additional optional flags:

- `--output-uncertainty uncertainty.png`
- `--output-confidence confidence.png`
- `--output-points points.ply`
- `--no-cuda-graph`
- `--benchmark N`

## Example output

Example console output (ViT-L14 FP16 engine; values depend on the image and GPU):

```text
========================================
UniDepth V2 TensorRT Inference
========================================
Engine:  engines/unidepth_v2_vitl14_fp16.trt
Input:   assets/demo/rgb.png

Image size: 640x480
Using provided intrinsics:
  fx: 525
  fy: 525
  cx: 319.5
  cy: 239.5

Running inference...
Inference time: 25 ms

Output statistics:
  Depth range:       [1.595, 6.536] meters
  Confidence range:  [0.484, 4.360]
  Uncertainty range: [0.332, 2.064]

Saving outputs...
  Depth visualization: depth.png
  Uncertainty map: uncertainty.png
  Confidence map: confidence.png
Saved point cloud: points.ply (307200 points)

✓ Done!
```

Generated files (when using default flags):

- `depth.png`
- `uncertainty.png`
- `confidence.png`
- `points.ply`

## Performance (ViT-L14, RTX 5070)

This repo intentionally does **not** publish estimated speeds.

| Metric | Value |
|---|---|
| Inference | 25 ms |

Accuracy summary (PyTorch vs TensorRT FP16, same input):

| Metric | Value |
|---|---|
| Depth range difference (min) | 2.55% |
| Depth range difference (max) | 1.83% |

## Confidence and uncertainty

- **Confidence**: higher is better
- **Uncertainty**: higher is worse (`1 / (confidence + 1e-6)`)

## Camera intrinsics notes

Intrinsics are provided as:

- `fx,fy,cx,cy` (pixel units)

If you don’t pass `--intrinsics`, the demo estimates intrinsics from image size using a 60° horizontal field-of-view.

## (Optional) Build an engine from HuggingFace weights

### 1) Export ONNX

```bash
python3 python/export_onnx.py \
  --model lpiccinelli/unidepth-v2-vitl14 \
  --output models/unidepth_v2_vitl14_camera.onnx \
  --height 518 \
  --width 518
```

### 2) Build TensorRT engine

```bash
python3 python/build_engine.py \
  --onnx models/unidepth_v2_vitl14_camera.onnx \
  --output engines/unidepth_v2_vitl14_fp16.trt \
  --precision fp16 \
  --workspace 6144
```


