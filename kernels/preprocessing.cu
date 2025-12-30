/**
 * CUDA Preprocessing Kernels for UniDepth V2
 *
 * Fused preprocessing pipeline:
 * 1. Bilinear resize BGR input to 518x518
 * 2. Convert BGR to RGB
 * 3. Normalize with ImageNet mean/std
 * 4. Convert HWC to CHW format
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

namespace unidepth {
namespace kernels {

// ImageNet normalization constants
__constant__ float MEAN[3] = {0.485f, 0.456f, 0.406f};  // RGB
__constant__ float STD[3]  = {0.229f, 0.224f, 0.225f};   // RGB

/**
 * Fused preprocessing kernel
 *
 * @param input BGR image (HWC, uint8)
 * @param output Preprocessed RGB image (CHW, float32, normalized)
 * @param src_width Original image width
 * @param src_height Original image height
 * @param dst_width Target width (518)
 * @param dst_height Target height (518)
 */
__global__ void preprocessKernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    // Compute source coordinates for bilinear interpolation
    float src_x = (x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y = (y + 0.5f) * src_height / dst_height - 0.5f;

    // Clamp to valid range
    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

    // Get integer coordinates
    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    // Compute interpolation weights
    float wx = src_x - x0;
    float wy = src_y - y0;

    // Bilinear interpolation for each channel
    // Input is BGR (OpenCV format), need to convert to RGB
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        // OpenCV BGR -> RGB: B=2, G=1, R=0 -> R=0, G=1, B=2
        int src_c = 2 - c;  // Reverse channel order

        // Sample 4 neighbors
        float v00 = input[(y0 * src_width + x0) * 3 + src_c];
        float v01 = input[(y0 * src_width + x1) * 3 + src_c];
        float v10 = input[(y1 * src_width + x0) * 3 + src_c];
        float v11 = input[(y1 * src_width + x1) * 3 + src_c];

        // Bilinear interpolation
        float v0 = v00 * (1.0f - wx) + v01 * wx;
        float v1 = v10 * (1.0f - wx) + v11 * wx;
        float v = v0 * (1.0f - wy) + v1 * wy;

        // Normalize: [0, 255] -> [0, 1] -> normalize with ImageNet stats
        v = v / 255.0f;
        v = (v - MEAN[c]) / STD[c];

        // Write to output in CHW format
        // Output layout: [C, H, W]
        output[c * dst_height * dst_width + y * dst_width + x] = v;
    }
}

/**
 * Host function to launch preprocessing
 */
void preprocess(
    const unsigned char* d_input,
    float* d_output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (dst_width + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y
    );

    preprocessKernel<<<grid, block, 0, stream>>>(
        d_input,
        d_output,
        src_width,
        src_height,
        dst_width,
        dst_height
    );
}

} // namespace kernels
} // namespace unidepth
