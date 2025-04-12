#!/bin/bash
# TinyAI Priority Test Execution Script
# This script runs tests in the priority order defined in the test plan

echo "===== TinyAI Priority Test Execution ====="
echo

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure and build the project
echo "Configuring and building..."
cmake ..
if [ $? -ne 0 ]; then
  echo "ERROR: Configuration failed"
  exit 1
fi

cmake --build .
if [ $? -ne 0 ]; then
  echo "ERROR: Build failed"
  exit 1
fi

echo
echo "===== Running tests in priority order ====="
echo

# 1. Core Memory Management Tests
echo "------ Memory Management Tests ------"
./tinyai_tests memory
if [ $? -ne 0 ]; then
  echo "ERROR: Memory management tests failed"
  exit 1
fi

# Out-of-Memory Handling Test
echo "------ Out-of-Memory Handling Test ------"
./oom_handling_test
if [ $? -ne 0 ]; then
  echo "ERROR: Out-of-memory handling test failed"
  exit 1
fi

# Memory-Mapped Model Test
echo "------ Memory-Mapped Model Test ------"
./mmap_loader_test
if [ $? -ne 0 ]; then
  echo "ERROR: Memory-mapped model test failed"
  exit 1
fi

# 2. 4-bit Quantization Tests
echo "------ Quantization Tests ------"
./tinyai_tests quantize
if [ $? -ne 0 ]; then
  echo "ERROR: Quantization tests failed"
  exit 1
fi

# Sparse Matrix Tests
echo "------ Sparse Matrix Tests ------"
./sparse_matrix_test
if [ $? -ne 0 ]; then
  echo "ERROR: Sparse matrix tests failed"
  exit 1
fi

# 3. SIMD Acceleration Tests
echo "------ SIMD Acceleration Tests ------"
./tinyai_tests simd
if [ $? -ne 0 ]; then
  echo "ERROR: SIMD acceleration tests failed"
  exit 1
fi

# SIMD Benchmark Test
echo "------ SIMD Benchmark Test ------"
./simd_benchmark_test
if [ $? -ne 0 ]; then
  echo "ERROR: SIMD benchmark test failed"
  exit 1
fi

# Attention Mechanism Test
echo "------ Attention Mechanism Test ------"
./attention_test
if [ $? -ne 0 ]; then
  echo "ERROR: Attention mechanism test failed"
  exit 1
fi

# 4. Text Generation Tests
echo "------ Text Generation Tests ------"
./tinyai_tests generate
if [ $? -ne 0 ]; then
  echo "ERROR: Text generation tests failed"
  exit 1
fi

# 5. Image Model Tests
echo "------ Image Model Tests ------"
./image_test
if [ $? -ne 0 ]; then
  echo "ERROR: Image model tests failed"
  exit 1
fi

# 6. Multimodal Capabilities Tests
echo "------ Multimodal Capabilities Tests ------"
./multimodal_test
if [ $? -ne 0 ]; then
  echo "ERROR: Multimodal capabilities tests failed"
  exit 1
fi

echo
echo "===== All priority tests completed successfully ====="
echo

cd ..
exit 0 