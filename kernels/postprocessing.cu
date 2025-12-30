/**
 * CUDA Postprocessing Kernels for UniDepth V2
 *
 * Handles resizing and visualization of all outputs:
 * - Depth map
 * - 3D points
 * - Confidence map
 * - Uncertainty map
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <vector>
#include <algorithm>

namespace unidepth {
namespace kernels {

/**
 * Resize depth/confidence/uncertainty maps using bilinear interpolation
 */
__global__ void resizeBilinearKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    // Compute source coordinates
    float src_x = (x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y = (y + 0.5f) * src_height / dst_height - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float wx = src_x - x0;
    float wy = src_y - y0;

    // Bilinear interpolation
    float v00 = input[y0 * src_width + x0];
    float v01 = input[y0 * src_width + x1];
    float v10 = input[y1 * src_width + x0];
    float v11 = input[y1 * src_width + x1];

    float v0 = v00 * (1.0f - wx) + v01 * wx;
    float v1 = v10 * (1.0f - wx) + v11 * wx;
    float v = v0 * (1.0f - wy) + v1 * wy;

    output[y * dst_width + x] = v;
}

/**
 * Resize 3D points (3 channels) using bilinear interpolation
 */
__global__ void resize3DPointsKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    // Compute source coordinates
    float src_x = (x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y = (y + 0.5f) * src_height / dst_height - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

    int x0 = static_cast<int>(src_x);
    int y0 = static_cast<int>(src_y);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float wx = src_x - x0;
    float wy = src_y - y0;

    // Interpolate each channel (X, Y, Z)
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        int offset = c * src_height * src_width;

        float v00 = input[offset + y0 * src_width + x0];
        float v01 = input[offset + y0 * src_width + x1];
        float v10 = input[offset + y1 * src_width + x0];
        float v11 = input[offset + y1 * src_width + x1];

        float v0 = v00 * (1.0f - wx) + v01 * wx;
        float v1 = v10 * (1.0f - wx) + v11 * wx;
        float v = v0 * (1.0f - wy) + v1 * wy;

        int out_offset = c * dst_height * dst_width;
        output[out_offset + y * dst_width + x] = v;
    }
}

/**
 * Min/Max reduction kernel for normalization
 */
__global__ void minMaxReductionKernel(
    const float* __restrict__ input,
    float* min_out,
    float* max_out,
    int size
) {
    extern __shared__ float shared_data[];
    float* shared_min = shared_data;
    float* shared_max = shared_data + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float min_val = (idx < size) ? input[idx] : INFINITY;
    float max_val = (idx < size) ? input[idx] : -INFINITY;

    shared_min[tid] = min_val;
    shared_max[tid] = max_val;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (idx + stride) < size) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_out[blockIdx.x] = shared_min[0];
        max_out[blockIdx.x] = shared_max[0];
    }
}

/**
 * Normalize and convert to uint8 grayscale
 */
__global__ void normalizeToGrayscaleKernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float min_val,
    float max_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    float val = input[idx];
    float normalized = (val - min_val) / (max_val - min_val + 1e-6f);
    normalized = fmaxf(0.0f, fminf(normalized, 1.0f));

    output[idx] = static_cast<unsigned char>(normalized * 255.0f);
}

/**
 * Apply colormap (Jet) to normalized depth/uncertainty
 */
__global__ void applyColormapKernel(
    const unsigned char* __restrict__ input_gray,
    unsigned char* __restrict__ output_bgr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    // Jet colormap implementation
    float value = input_gray[idx] / 255.0f;

    float r, g, b;
    if (value < 0.25f) {
        r = 0.0f;
        g = 4.0f * value;
        b = 1.0f;
    } else if (value < 0.5f) {
        r = 0.0f;
        g = 1.0f;
        b = 1.0f - 4.0f * (value - 0.25f);
    } else if (value < 0.75f) {
        r = 4.0f * (value - 0.5f);
        g = 1.0f;
        b = 0.0f;
    } else {
        r = 1.0f;
        g = 1.0f - 4.0f * (value - 0.75f);
        b = 0.0f;
    }

    // OpenCV uses BGR format
    output_bgr[idx * 3 + 0] = static_cast<unsigned char>(b * 255.0f);
    output_bgr[idx * 3 + 1] = static_cast<unsigned char>(g * 255.0f);
    output_bgr[idx * 3 + 2] = static_cast<unsigned char>(r * 255.0f);
}

/**
 * Host functions
 */

void resizeBilinear(
    const float* d_input,
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

    resizeBilinearKernel<<<grid, block, 0, stream>>>(
        d_input, d_output,
        src_width, src_height,
        dst_width, dst_height
    );
}

void resize3DPoints(
    const float* d_input,
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

    resize3DPointsKernel<<<grid, block, 0, stream>>>(
        d_input, d_output,
        src_width, src_height,
        dst_width, dst_height
    );
}

void findMinMax(
    const float* d_input,
    float& min_val,
    float& max_val,
    int size,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    float *d_min_blocks, *d_max_blocks;
    cudaMalloc(&d_min_blocks, num_blocks * sizeof(float));
    cudaMalloc(&d_max_blocks, num_blocks * sizeof(float));

    size_t shared_size = 2 * block_size * sizeof(float);
    minMaxReductionKernel<<<num_blocks, block_size, shared_size, stream>>>(
        d_input, d_min_blocks, d_max_blocks, size
    );

    // Copy partial results to host
    std::vector<float> h_min_blocks(num_blocks);
    std::vector<float> h_max_blocks(num_blocks);

    cudaStreamSynchronize(stream);
    cudaMemcpy(h_min_blocks.data(), d_min_blocks, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_blocks.data(), d_max_blocks, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    min_val = *std::min_element(h_min_blocks.begin(), h_min_blocks.end());
    max_val = *std::max_element(h_max_blocks.begin(), h_max_blocks.end());

    cudaFree(d_min_blocks);
    cudaFree(d_max_blocks);
}

void normalizeToGrayscale(
    const float* d_input,
    unsigned char* d_output,
    float min_val,
    float max_val,
    int size,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    normalizeToGrayscaleKernel<<<num_blocks, block_size, 0, stream>>>(
        d_input, d_output, min_val, max_val, size
    );
}

void applyColormap(
    const unsigned char* d_input_gray,
    unsigned char* d_output_bgr,
    int size,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    applyColormapKernel<<<num_blocks, block_size, 0, stream>>>(
        d_input_gray, d_output_bgr, size
    );
}

} // namespace kernels
} // namespace unidepth
