/**
 * Preprocessing kernel declarations
 */

#pragma once

#include <cuda_runtime.h>

namespace unidepth {
namespace kernels {

/**
 * Preprocess image on GPU
 * - Resize to target size (bilinear interpolation)
 * - BGR to RGB conversion
 * - Normalize with ImageNet mean/std
 * - Convert HWC to CHW
 */
void preprocess(
    const unsigned char* d_input,
    float* d_output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    cudaStream_t stream = nullptr
);

} // namespace kernels
} // namespace unidepth
