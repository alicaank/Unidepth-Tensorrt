/**
 * UniDepth V2 Image Inference Example
 *
 * Demonstrates how to use the UniDepth V2 TensorRT engine for:
 * - Metric depth estimation
 * - 3D point cloud generation
 * - Uncertainty estimation
 */

#include "unidepth_v2.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace unidepth;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --engine PATH          TensorRT engine file (required)\n";
    std::cout << "  --input PATH           Input image file (required)\n";
    std::cout << "  --intrinsics FX,FY,CX,CY  Camera intrinsics (optional)\n";
    std::cout << "  --output-depth PATH    Output depth map visualization (default: depth.png)\n";
    std::cout << "  --output-uncertainty PATH  Output uncertainty map (default: uncertainty.png)\n";
    std::cout << "  --output-confidence PATH   Output confidence map (default: confidence.png)\n";
    std::cout << "  --output-points PATH   Output 3D point cloud PLY file (default: points.ply)\n";
    std::cout << "  --output-raw PATH      Output raw depth values as .npy (optional)\n";
    std::cout << "  --no-cuda-graph        Disable CUDA Graph optimization\n";
    std::cout << "  --benchmark N          Run N iterations for benchmarking\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " \\\n";
    std::cout << "    --engine engines/unidepth_v2_vitl14_fp16.trt \\\n";
    std::cout << "    --input image.jpg \\\n";
    std::cout << "    --intrinsics \"525.0,525.0,319.5,239.5\" \\\n";
    std::cout << "    --output-depth depth.png\n";
}

bool parseIntrinsics(const std::string& str, CameraIntrinsics& intrinsics) {
    std::istringstream iss(str);
    std::string token;
    std::vector<float> values;

    while (std::getline(iss, token, ',')) {
        try {
            values.push_back(std::stof(token));
        } catch (...) {
            return false;
        }
    }

    if (values.size() != 4) return false;

    intrinsics.fx = values[0];
    intrinsics.fy = values[1];
    intrinsics.cx = values[2];
    intrinsics.cy = values[3];

    return true;
}

void savePointCloud(const std::string& filename, const cv::Mat& pts_3d) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open PLY file: " << filename << std::endl;
        return;
    }

    int width = pts_3d.cols;
    int height = pts_3d.rows;
    int num_points = width * height;

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << num_points << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";

    // Write points
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3f point = pts_3d.at<cv::Vec3f>(y, x);
            file << point[0] << " " << point[1] << " " << point[2] << "\n";
        }
    }

    file.close();
    std::cout << "Saved point cloud: " << filename << " (" << num_points << " points)" << std::endl;
}

cv::Mat visualizeDepth(const cv::Mat& depth) {
    // Normalize and apply colormap
    cv::Mat depth_normalized;
    double min_val, max_val;
    cv::minMaxLoc(depth, &min_val, &max_val);

    depth.convertTo(depth_normalized, CV_8U, 255.0 / (max_val - min_val), -255.0 * min_val / (max_val - min_val));

    cv::Mat depth_colored;
    cv::applyColorMap(depth_normalized, depth_colored, cv::COLORMAP_JET);

    return depth_colored;
}

int main(int argc, char** argv) {
    // Parse arguments
    std::string engine_path;
    std::string input_path;
    std::string output_depth = "depth.png";
    std::string output_uncertainty = "uncertainty.png";
    std::string output_confidence = "confidence.png";
    std::string output_points = "points.ply";
    std::string output_raw;
    std::string intrinsics_str;
    bool use_cuda_graph = true;
    int benchmark_iterations = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--intrinsics" && i + 1 < argc) {
            intrinsics_str = argv[++i];
        } else if (arg == "--output-depth" && i + 1 < argc) {
            output_depth = argv[++i];
        } else if (arg == "--output-uncertainty" && i + 1 < argc) {
            output_uncertainty = argv[++i];
        } else if (arg == "--output-confidence" && i + 1 < argc) {
            output_confidence = argv[++i];
        } else if (arg == "--output-points" && i + 1 < argc) {
            output_points = argv[++i];
        } else if (arg == "--output-raw" && i + 1 < argc) {
            output_raw = argv[++i];
        } else if (arg == "--no-cuda-graph") {
            use_cuda_graph = false;
        } else if (arg == "--benchmark" && i + 1 < argc) {
            benchmark_iterations = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Validate arguments
    if (engine_path.empty() || input_path.empty()) {
        std::cerr << "Error: Missing required arguments\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    try {
        std::cout << "========================================" << std::endl;
        std::cout << "UniDepth V2 TensorRT Inference" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Engine:  " << engine_path << std::endl;
        std::cout << "Input:   " << input_path << std::endl;
        std::cout << std::endl;

        // Load image
        cv::Mat image = cv::imread(input_path);
        if (image.empty()) {
            std::cerr << "Error: Failed to load image: " << input_path << std::endl;
            return 1;
        }

        std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

        // Parse or estimate camera intrinsics
        CameraIntrinsics intrinsics;
        if (!intrinsics_str.empty()) {
            if (!parseIntrinsics(intrinsics_str, intrinsics)) {
                std::cerr << "Error: Invalid intrinsics format. Expected: fx,fy,cx,cy" << std::endl;
                return 1;
            }
            std::cout << "Using provided intrinsics:" << std::endl;
        } else {
            intrinsics = CameraIntrinsics::fromImageSize(image.cols, image.rows);
            std::cout << "Estimating intrinsics (60° FoV):" << std::endl;
        }

        std::cout << "  fx: " << intrinsics.fx << std::endl;
        std::cout << "  fy: " << intrinsics.fy << std::endl;
        std::cout << "  cx: " << intrinsics.cx << std::endl;
        std::cout << "  cy: " << intrinsics.cy << std::endl;
        std::cout << std::endl;

        // Load model
        UniDepthV2 model(engine_path, use_cuda_graph);
        std::cout << std::endl;

        // Run inference
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        UniDepthOutput output;
        model.infer(image, intrinsics, output);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;

        // Print output statistics
        double depth_min, depth_max;
        cv::minMaxLoc(output.depth, &depth_min, &depth_max);

        double conf_min, conf_max;
        cv::minMaxLoc(output.confidence, &conf_min, &conf_max);

        double uncert_min, uncert_max;
        cv::minMaxLoc(output.uncertainty, &uncert_min, &uncert_max);

        std::cout << "Output statistics:" << std::endl;
        std::cout << "  Depth range:       [" << std::fixed << std::setprecision(3)
                  << depth_min << ", " << depth_max << "] meters" << std::endl;
        std::cout << "  Confidence range:  [" << conf_min << ", " << conf_max << "]" << std::endl;
        std::cout << "  Uncertainty range: [" << uncert_min << ", " << uncert_max << "]" << std::endl;
        std::cout << std::endl;

        // Save outputs
        std::cout << "Saving outputs..." << std::endl;

        cv::Mat depth_vis = visualizeDepth(output.depth);
        cv::imwrite(output_depth, depth_vis);
        std::cout << "  Depth visualization: " << output_depth << std::endl;

        cv::Mat uncertainty_vis = visualizeDepth(output.uncertainty);
        cv::imwrite(output_uncertainty, uncertainty_vis);
        std::cout << "  Uncertainty map: " << output_uncertainty << std::endl;

        cv::Mat confidence_vis = visualizeDepth(output.confidence);
        cv::imwrite(output_confidence, confidence_vis);
        std::cout << "  Confidence map: " << output_confidence << std::endl;

        savePointCloud(output_points, output.pts_3d);

        // Benchmark if requested
        if (benchmark_iterations > 0) {
            std::cout << "\nRunning benchmark (" << benchmark_iterations << " iterations)..." << std::endl;

            auto bench_start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < benchmark_iterations; ++i) {
                model.infer(image, intrinsics, output);
            }

            auto bench_end = std::chrono::high_resolution_clock::now();
            auto bench_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start);

            double avg_time = static_cast<double>(bench_duration.count()) / benchmark_iterations;
            double fps = 1000.0 / avg_time;

            std::cout << "Average inference time: " << std::fixed << std::setprecision(2)
                      << avg_time << " ms" << std::endl;
            std::cout << "Throughput: " << fps << " FPS" << std::endl;
        }

        std::cout << "\n✓ Done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
