/**
 * UniDepth V2 TensorRT Inference Engine
 *
 * High-performance C++ TensorRT implementation for UniDepth V2
 * with camera intrinsics input and uncertainty estimation.
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

namespace unidepth {

/**
 * Camera intrinsics parameters
 */
struct CameraIntrinsics {
    float fx;  // Focal length in x
    float fy;  // Focal length in y
    float cx;  // Principal point x
    float cy;  // Principal point y

    CameraIntrinsics() : fx(0), fy(0), cx(0), cy(0) {}
    CameraIntrinsics(float fx_, float fy_, float cx_, float cy_)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    // Create from image dimensions assuming 60° horizontal FoV
    static CameraIntrinsics fromImageSize(int width, int height) {
        float fx = width / (2.0f * std::tan(60.0f * M_PI / 360.0f));
        float fy = fx;  // Assume square pixels
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        return CameraIntrinsics(fx, fy, cx, cy);
    }
};

/**
 * UniDepth V2 inference outputs
 */
struct UniDepthOutput {
    cv::Mat pts_3d;          // [3, H, W] - 3D points in camera coordinates (CV_32FC3)
    cv::Mat depth;           // [1, H, W] - Depth map in meters (CV_32FC1)
    cv::Mat confidence;      // [1, H, W] - Confidence map (CV_32FC1)
    cv::Mat uncertainty;     // [1, H, W] - Uncertainty map (CV_32FC1)
    cv::Mat pred_intrinsics; // [3, 3] - Predicted camera intrinsics matrix (CV_32FC1)
};

/**
 * UniDepth V2 TensorRT Inference Engine
 */
class UniDepthV2 {
public:
    /**
     * Constructor
     * @param engine_path Path to TensorRT engine file (.trt)
     * @param use_cuda_graph Enable CUDA Graph optimization (default: true)
     */
    explicit UniDepthV2(const std::string& engine_path, bool use_cuda_graph = true);

    /**
     * Destructor
     */
    ~UniDepthV2();

    // Delete copy constructor and assignment operator
    UniDepthV2(const UniDepthV2&) = delete;
    UniDepthV2& operator=(const UniDepthV2&) = delete;

    /**
     * Run inference on a single image
     * @param image Input RGB image (any size, will be resized to 518x518)
     * @param intrinsics Camera intrinsics parameters
     * @param output Output structure containing all predictions
     * @param stream CUDA stream (optional, uses default stream if nullptr)
     */
    void infer(
        const cv::Mat& image,
        const CameraIntrinsics& intrinsics,
        UniDepthOutput& output,
        cudaStream_t stream = nullptr
    );

    /**
     * Get input size
     */
    int getInputHeight() const { return input_height_; }
    int getInputWidth() const { return input_width_; }

    /**
     * Get model info
     */
    std::string getModelInfo() const;

private:
    // TensorRT components
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    // CUDA Graph (optional)
    bool use_cuda_graph_;
    cudaGraph_t cuda_graph_;
    cudaGraphExec_t cuda_graph_exec_;
    bool graph_captured_;

    // Model dimensions
    int input_height_;   // 518
    int input_width_;    // 518
    int output_height_;  // 518
    int output_width_;   // 518

    // Device buffers (managed with unique_ptr for RAII)
    struct CudaDeleter {
        void operator()(void* ptr) const {
            if (ptr) cudaFree(ptr);
        }
    };

    std::unique_ptr<void, CudaDeleter> d_input_;           // [1, 3, 518, 518]
    std::unique_ptr<void, CudaDeleter> d_intrinsics_;      // [1, 4]
    std::unique_ptr<void, CudaDeleter> d_pts_3d_;          // [1, 3, 518, 518]
    std::unique_ptr<void, CudaDeleter> d_depth_;           // [1, 1, 518, 518]
    std::unique_ptr<void, CudaDeleter> d_confidence_;      // [1, 1, 518, 518]
    std::unique_ptr<void, CudaDeleter> d_uncertainty_;     // [1, 1, 518, 518]
    std::unique_ptr<void, CudaDeleter> d_pred_intrinsics_; // [1, 3, 3]

    // Internal methods
    void loadEngine(const std::string& engine_path);
    void allocateBuffers();
    void preprocessImage(const cv::Mat& image, cudaStream_t stream);
    void executeInference(cudaStream_t stream);
    void postprocessOutputs(
        const cv::Mat& original_image,
        UniDepthOutput& output,
        cudaStream_t stream
    );

    // Helper for tensor name lookup
    int getTensorIndex(const std::string& name) const;
};

} // namespace unidepth
