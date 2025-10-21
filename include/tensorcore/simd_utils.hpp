#pragma once

#include <vector>
#include <cstdint>

// Only include x86 intrinsics on x86/x64 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>  // For AVX/SSE intrinsics
#include <x86intrin.h>  // For CPU feature detection
#define TENSORCORE_SIMD_AVAILABLE 1
#else
#define TENSORCORE_SIMD_AVAILABLE 0
#endif

namespace tensorcore {

/**
 * @brief SIMD utilities for vectorized operations
 * 
 * This module provides SIMD-optimized implementations of common tensor operations
 * using AVX2, AVX, and SSE instructions for maximum performance.
 */
class SIMDUtils {
public:
    // CPU feature detection
    static bool has_avx2();
    static bool has_avx();
    static bool has_sse4_1();
    
    // Vectorized operations
    static void vectorized_add(const double* a, const double* b, double* result, size_t size);
    static void vectorized_subtract(const double* a, const double* b, double* result, size_t size);
    static void vectorized_multiply(const double* a, const double* b, double* result, size_t size);
    static void vectorized_divide(const double* a, const double* b, double* result, size_t size);
    
    // Scalar operations
    static void vectorized_add_scalar(const double* a, double scalar, double* result, size_t size);
    static void vectorized_multiply_scalar(const double* a, double scalar, double* result, size_t size);
    
    // Mathematical functions
    static void vectorized_sqrt(const double* a, double* result, size_t size);
    static void vectorized_exp(const double* a, double* result, size_t size);
    static void vectorized_log(const double* a, double* result, size_t size);
    static void vectorized_sin(const double* a, double* result, size_t size);
    static void vectorized_cos(const double* a, double* result, size_t size);
    static void vectorized_tanh(const double* a, double* result, size_t size);
    
    // Reduction operations
    static double vectorized_sum(const double* a, size_t size);
    static double vectorized_max(const double* a, size_t size);
    static double vectorized_min(const double* a, size_t size);
    static double vectorized_mean(const double* a, size_t size);
    
    // Matrix operations
    static void vectorized_matmul(const double* a, const double* b, double* result,
                                 size_t m, size_t n, size_t k);
    
    // Activation functions
    static void vectorized_relu(const double* a, double* result, size_t size);
    static void vectorized_sigmoid(const double* a, double* result, size_t size);
    static void vectorized_softmax(const double* a, double* result, size_t size);
    
    // Convolution operations
    static void vectorized_conv2d(const double* input, const double* weights, double* output,
                                 size_t batch_size, size_t input_channels, size_t input_height, size_t input_width,
                                 size_t output_channels, size_t kernel_size, size_t stride, size_t padding);
    
    // Pooling operations
    static void vectorized_maxpool2d(const double* input, double* output,
                                    size_t batch_size, size_t channels, size_t input_height, size_t input_width,
                                    size_t kernel_size, size_t stride, size_t padding);
    
    static void vectorized_avgpool2d(const double* input, double* output,
                                     size_t batch_size, size_t channels, size_t input_height, size_t input_width,
                                     size_t kernel_size, size_t stride, size_t padding);

private:
    // Internal helper functions
    static void fallback_add(const double* a, const double* b, double* result, size_t size);
    static void fallback_multiply(const double* a, const double* b, double* result, size_t size);
    static void fallback_sqrt(const double* a, double* result, size_t size);
    
    // AVX2 implementations
    static void avx2_add(const double* a, const double* b, double* result, size_t size);
    static void avx2_multiply(const double* a, const double* b, double* result, size_t size);
    static void avx2_sqrt(const double* a, double* result, size_t size);
    
    // SSE implementations
    static void sse_add(const double* a, const double* b, double* result, size_t size);
    static void sse_multiply(const double* a, const double* b, double* result, size_t size);
    static void sse_sqrt(const double* a, double* result, size_t size);
};

} // namespace tensorcore
