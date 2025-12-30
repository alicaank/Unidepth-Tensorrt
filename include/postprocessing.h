/**
 * Postprocessing kernel declarations
 */

#pragma once

#include <cuda_runtime.h>

namespace unidepth {
namespace kernels {

/**
 * Resize single-channel map (depth/confidence/uncertainty) using bilinear interpolation
 */
void resizeBilinear(
    const float* d_input,
    float* d_output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    cudaStream_t stream = nullptr
);

/**
 * Resize 3-channel 3D points using bilinear interpolation
 */
void resize3DPoints(
    const float* d_input,
    float* d_output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    cudaStream_t stream = nullptr
);

/**
 * Find min/max values in array (for normalization)
 */
void findMinMax(
    const float* d_input,
    float& min_val,
    float& max_val,
    int size,
    cudaStream_t stream = nullptr
);

/**
 * Normalize float array to uint8 grayscale [0, 255]
 */
void normalizeToGrayscale(
    const float* d_input,
    unsigned char* d_output,
    float min_val,
    float max_val,
    int size,
    cudaStream_t stream = nullptr
);

/**
 * Apply Jet colormap to grayscale image
 */
void applyColormap(
    const unsigned char* d_input_gray,
    unsigned char* d_output_bgr,
    int size,
    cudaStream_t stream = nullptr
);

} // namespace kernels
} // namespace unidepth
