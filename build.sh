#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}UniDepth V2 TensorRT Build Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-75;80;86;89}"
BUILD_DIR="build"

echo "Build configuration:"
echo "  Build type:         $BUILD_TYPE"
echo "  CUDA architectures: $CUDA_ARCH"
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA not found. Please install CUDA toolkit.${NC}"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo -e "  ${GREEN}✓${NC} CUDA $CUDA_VERSION"

# Check OpenCV
if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
    echo -e "${YELLOW}Warning: OpenCV pkg-config not found${NC}"
    echo "  Build may fail if OpenCV is not installed"
else
    OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null)
    echo -e "  ${GREEN}✓${NC} OpenCV $OPENCV_VERSION"
fi

echo ""

# Create build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DBUILD_EXAMPLES=ON \
    -DENABLE_CUDA_GRAPH=ON

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Build complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Binaries:"
ls -lh image_inference 2>/dev/null || echo "  (No binaries found)"

echo ""
echo "To run inference:"
echo "  ./build/image_inference \\"
echo "    --engine engines/unidepth_v2_vitl14_fp16.trt \\"
echo "    --input image.jpg \\"
echo "    --intrinsics \"525.0,525.0,319.5,239.5\""
echo ""
