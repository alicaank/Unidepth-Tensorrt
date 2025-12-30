/**
 * UniDepth V2 TensorRT Inference Engine Implementation
 */

#include "unidepth_v2.h"
#include "preprocessing.h"
#include "postprocessing.h"

#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace unidepth {

// TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only print warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

/**
 * Constructor
 */
UniDepthV2::UniDepthV2(const std::string& engine_path, bool use_cuda_graph)
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      use_cuda_graph_(use_cuda_graph),
      cuda_graph_(nullptr),
      cuda_graph_exec_(nullptr),
      graph_captured_(false),
      input_height_(518),
      input_width_(518),
      output_height_(518),
      output_width_(518) {

    loadEngine(engine_path);
    allocateBuffers();

    std::cout << "UniDepth V2 TensorRT engine loaded successfully" << std::endl;
    std::cout << "  Input size: " << input_width_ << "x" << input_height_ << std::endl;
    std::cout << "  CUDA Graph: " << (use_cuda_graph_ ? "Enabled" : "Disabled") << std::endl;
}

/**
 * Destructor
 */
UniDepthV2::~UniDepthV2() {
    // CUDA Graph cleanup
    if (cuda_graph_exec_) {
        cudaGraphExecDestroy(cuda_graph_exec_);
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
    }

    // TensorRT cleanup (use delete for TensorRT 10+)
    if (context_) {
        delete context_;
    }
    if (engine_) {
        delete engine_;
    }
    if (runtime_) {
        delete runtime_;
    }
}

/**
 * Load TensorRT engine from file
 */
void UniDepthV2::loadEngine(const std::string& engine_path) {
    // Read engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Deserialize engine
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    // Create execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }

    // Print engine info
    std::cout << "Engine loaded: " << engine_path << std::endl;
    std::cout << "  I/O Tensors: " << engine_->getNbIOTensors() << std::endl;

    // Verify expected tensors
    const char* expected_inputs[] = {"rgb", "intrinsics"};
    const char* expected_outputs[] = {"pts_3d", "depth", "confidence", "uncertainty", "pred_intrinsics"};

    for (const char* name : expected_inputs) {
        auto dims = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);
        if (mode != nvinfer1::TensorIOMode::kINPUT) {
            throw std::runtime_error(std::string("Expected input tensor not found: ") + name);
        }
    }

    for (const char* name : expected_outputs) {
        auto dims = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);
        if (mode != nvinfer1::TensorIOMode::kOUTPUT) {
            throw std::runtime_error(std::string("Expected output tensor not found: ") + name);
        }
    }
}

/**
 * Allocate device buffers
 */
void UniDepthV2::allocateBuffers() {
    // Get tensor dimensions
    auto rgb_dims = engine_->getTensorShape("rgb");
    auto intrinsics_dims = engine_->getTensorShape("intrinsics");
    auto pts_3d_dims = engine_->getTensorShape("pts_3d");
    auto depth_dims = engine_->getTensorShape("depth");
    auto confidence_dims = engine_->getTensorShape("confidence");
    auto uncertainty_dims = engine_->getTensorShape("uncertainty");
    auto pred_intrinsics_dims = engine_->getTensorShape("pred_intrinsics");

    // Calculate sizes (batch size = 1)
    size_t rgb_size = 1 * 3 * input_height_ * input_width_ * sizeof(float);
    size_t intrinsics_size = 1 * 4 * sizeof(float);
    size_t pts_3d_size = 1 * 3 * output_height_ * output_width_ * sizeof(float);
    size_t depth_size = 1 * 1 * output_height_ * output_width_ * sizeof(float);
    size_t confidence_size = 1 * 1 * output_height_ * output_width_ * sizeof(float);
    size_t uncertainty_size = 1 * 1 * output_height_ * output_width_ * sizeof(float);
    size_t pred_intrinsics_size = 1 * 3 * 3 * sizeof(float);

    // Allocate device memory
    void* d_rgb_ptr;
    void* d_intrinsics_ptr;
    void* d_pts_3d_ptr;
    void* d_depth_ptr;
    void* d_confidence_ptr;
    void* d_uncertainty_ptr;
    void* d_pred_intrinsics_ptr;

    cudaMalloc(&d_rgb_ptr, rgb_size);
    cudaMalloc(&d_intrinsics_ptr, intrinsics_size);
    cudaMalloc(&d_pts_3d_ptr, pts_3d_size);
    cudaMalloc(&d_depth_ptr, depth_size);
    cudaMalloc(&d_confidence_ptr, confidence_size);
    cudaMalloc(&d_uncertainty_ptr, uncertainty_size);
    cudaMalloc(&d_pred_intrinsics_ptr, pred_intrinsics_size);

    // Assign to unique_ptr (RAII)
    d_input_.reset(d_rgb_ptr);
    d_intrinsics_.reset(d_intrinsics_ptr);
    d_pts_3d_.reset(d_pts_3d_ptr);
    d_depth_.reset(d_depth_ptr);
    d_confidence_.reset(d_confidence_ptr);
    d_uncertainty_.reset(d_uncertainty_ptr);
    d_pred_intrinsics_.reset(d_pred_intrinsics_ptr);

    std::cout << "Device buffers allocated" << std::endl;
}

/**
 * Preprocess image on GPU
 */
void UniDepthV2::preprocessImage(const cv::Mat& image, cudaStream_t stream) {
    // Upload image to device
    void* d_image_bgr;
    size_t image_size = image.rows * image.cols * 3 * sizeof(unsigned char);
    cudaMalloc(&d_image_bgr, image_size);
    cudaMemcpyAsync(d_image_bgr, image.data, image_size, cudaMemcpyHostToDevice, stream);

    // Run preprocessing kernel
    kernels::preprocess(
        static_cast<unsigned char*>(d_image_bgr),
        static_cast<float*>(d_input_.get()),
        image.cols,
        image.rows,
        input_width_,
        input_height_,
        stream
    );

    cudaFree(d_image_bgr);
}

/**
 * Execute TensorRT inference
 */
void UniDepthV2::executeInference(cudaStream_t stream) {
    // Set tensor addresses
    context_->setTensorAddress("rgb", d_input_.get());
    context_->setTensorAddress("intrinsics", d_intrinsics_.get());
    context_->setTensorAddress("pts_3d", d_pts_3d_.get());
    context_->setTensorAddress("depth", d_depth_.get());
    context_->setTensorAddress("confidence", d_confidence_.get());
    context_->setTensorAddress("uncertainty", d_uncertainty_.get());
    context_->setTensorAddress("pred_intrinsics", d_pred_intrinsics_.get());

    // Execute inference
    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Failed to enqueue inference");
    }
}

/**
 * Postprocess outputs
 */
void UniDepthV2::postprocessOutputs(
    const cv::Mat& original_image,
    UniDepthOutput& output,
    cudaStream_t stream
) {
    int orig_width = original_image.cols;
    int orig_height = original_image.rows;

    // Allocate temporary device buffers for resized outputs
    void *d_pts_3d_resized, *d_depth_resized, *d_confidence_resized, *d_uncertainty_resized;

    size_t pts_3d_resized_size = 3 * orig_height * orig_width * sizeof(float);
    size_t map_resized_size = orig_height * orig_width * sizeof(float);

    cudaMalloc(&d_pts_3d_resized, pts_3d_resized_size);
    cudaMalloc(&d_depth_resized, map_resized_size);
    cudaMalloc(&d_confidence_resized, map_resized_size);
    cudaMalloc(&d_uncertainty_resized, map_resized_size);

    // Resize outputs to original dimensions
    kernels::resize3DPoints(
        static_cast<float*>(d_pts_3d_.get()),
        static_cast<float*>(d_pts_3d_resized),
        output_width_, output_height_,
        orig_width, orig_height,
        stream
    );

    kernels::resizeBilinear(
        static_cast<float*>(d_depth_.get()),
        static_cast<float*>(d_depth_resized),
        output_width_, output_height_,
        orig_width, orig_height,
        stream
    );

    kernels::resizeBilinear(
        static_cast<float*>(d_confidence_.get()),
        static_cast<float*>(d_confidence_resized),
        output_width_, output_height_,
        orig_width, orig_height,
        stream
    );

    kernels::resizeBilinear(
        static_cast<float*>(d_uncertainty_.get()),
        static_cast<float*>(d_uncertainty_resized),
        output_width_, output_height_,
        orig_width, orig_height,
        stream
    );

    // Synchronize before copying to host
    cudaStreamSynchronize(stream);

    // Allocate host output matrices
    output.pts_3d = cv::Mat(orig_height, orig_width, CV_32FC3);
    output.depth = cv::Mat(orig_height, orig_width, CV_32FC1);
    output.confidence = cv::Mat(orig_height, orig_width, CV_32FC1);
    output.uncertainty = cv::Mat(orig_height, orig_width, CV_32FC1);
    output.pred_intrinsics = cv::Mat(3, 3, CV_32FC1);

    // Copy outputs from device to host
    cudaMemcpy(output.pts_3d.data, d_pts_3d_resized, pts_3d_resized_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output.depth.data, d_depth_resized, map_resized_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output.confidence.data, d_confidence_resized, map_resized_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output.uncertainty.data, d_uncertainty_resized, map_resized_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(output.pred_intrinsics.data, d_pred_intrinsics_.get(), 9 * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup temporary buffers
    cudaFree(d_pts_3d_resized);
    cudaFree(d_depth_resized);
    cudaFree(d_confidence_resized);
    cudaFree(d_uncertainty_resized);
}

/**
 * Run inference
 */
void UniDepthV2::infer(
    const cv::Mat& image,
    const CameraIntrinsics& intrinsics,
    UniDepthOutput& output,
    cudaStream_t stream
) {
    cudaStream_t inference_stream = stream ? stream : 0;

    // Upload camera intrinsics to device
    float h_intrinsics[4] = {intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy};
    cudaMemcpyAsync(
        d_intrinsics_.get(),
        h_intrinsics,
        4 * sizeof(float),
        cudaMemcpyHostToDevice,
        inference_stream
    );

    // Preprocess image
    preprocessImage(image, inference_stream);

    // Execute inference (with optional CUDA Graph)
    if (use_cuda_graph_ && !graph_captured_) {
        // Capture CUDA Graph on first run
        cudaStreamBeginCapture(inference_stream, cudaStreamCaptureModeGlobal);
        executeInference(inference_stream);
        cudaStreamEndCapture(inference_stream, &cuda_graph_);
        cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0);
        graph_captured_ = true;

        std::cout << "CUDA Graph captured" << std::endl;
    }

    if (use_cuda_graph_ && graph_captured_) {
        // Execute captured graph
        cudaGraphLaunch(cuda_graph_exec_, inference_stream);
    } else {
        // Execute normally
        executeInference(inference_stream);
    }

    // Postprocess outputs
    postprocessOutputs(image, output, inference_stream);

    cudaStreamSynchronize(inference_stream);
}

/**
 * Get model info
 */
std::string UniDepthV2::getModelInfo() const {
    std::stringstream ss;
    ss << "UniDepth V2 TensorRT Model\n";
    ss << "  Input size: " << input_width_ << "x" << input_height_ << "\n";
    ss << "  Output size: " << output_width_ << "x" << output_height_ << "\n";
    ss << "  I/O tensors: " << engine_->getNbIOTensors() << "\n";
    ss << "  CUDA Graph: " << (use_cuda_graph_ ? "Enabled" : "Disabled") << "\n";
    return ss.str();
}

} // namespace unidepth
