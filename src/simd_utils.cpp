#include "tensorcore/simd_utils.hpp"
#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tensorcore {

// CPU feature detection
bool SIMDUtils::has_avx2() {
    return __builtin_cpu_supports("avx2");
}

bool SIMDUtils::has_avx() {
    return __builtin_cpu_supports("avx");
}

bool SIMDUtils::has_sse4_1() {
    return __builtin_cpu_supports("sse4.1");
}

// Vectorized addition with automatic SIMD selection
void SIMDUtils::vectorized_add(const double* a, const double* b, double* result, size_t size) {
    if (has_avx2()) {
        avx2_add(a, b, result, size);
    } else if (has_avx()) {
        // Fall back to AVX implementation
        fallback_add(a, b, result, size);
    } else if (has_sse4_1()) {
        sse_add(a, b, result, size);
    } else {
        fallback_add(a, b, result, size);
    }
}

// Vectorized multiplication with automatic SIMD selection
void SIMDUtils::vectorized_multiply(const double* a, const double* b, double* result, size_t size) {
    if (has_avx2()) {
        avx2_multiply(a, b, result, size);
    } else if (has_avx()) {
        // Fall back to AVX implementation
        fallback_multiply(a, b, result, size);
    } else if (has_sse4_1()) {
        sse_multiply(a, b, result, size);
    } else {
        fallback_multiply(a, b, result, size);
    }
}

// Vectorized square root with automatic SIMD selection
void SIMDUtils::vectorized_sqrt(const double* a, double* result, size_t size) {
    if (has_avx2()) {
        avx2_sqrt(a, result, size);
    } else if (has_avx()) {
        // Fall back to AVX implementation
        fallback_sqrt(a, result, size);
    } else if (has_sse4_1()) {
        sse_sqrt(a, result, size);
    } else {
        fallback_sqrt(a, result, size);
    }
}

// AVX2 implementations
void SIMDUtils::avx2_add(const double* a, const double* b, double* result, size_t size) {
    size_t i = 0;
    
    // Process 4 doubles at a time with AVX2
    for (; i + 3 < size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDUtils::avx2_multiply(const double* a, const double* b, double* result, size_t size) {
    size_t i = 0;
    
    // Process 4 doubles at a time with AVX2
    for (; i + 3 < size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vb = _mm256_load_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDUtils::avx2_sqrt(const double* a, double* result, size_t size) {
    size_t i = 0;
    
    // Process 4 doubles at a time with AVX2
    for (; i + 3 < size; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d vresult = _mm256_sqrt_pd(va);
        _mm256_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::sqrt(a[i]);
    }
}

// SSE implementations
void SIMDUtils::sse_add(const double* a, const double* b, double* result, size_t size) {
    size_t i = 0;
    
    // Process 2 doubles at a time with SSE
    for (; i + 1 < size; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vb = _mm_load_pd(&b[i]);
        __m128d vresult = _mm_add_pd(va, vb);
        _mm_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDUtils::sse_multiply(const double* a, const double* b, double* result, size_t size) {
    size_t i = 0;
    
    // Process 2 doubles at a time with SSE
    for (; i + 1 < size; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vb = _mm_load_pd(&b[i]);
        __m128d vresult = _mm_mul_pd(va, vb);
        _mm_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDUtils::sse_sqrt(const double* a, double* result, size_t size) {
    size_t i = 0;
    
    // Process 2 doubles at a time with SSE
    for (; i + 1 < size; i += 2) {
        __m128d va = _mm_load_pd(&a[i]);
        __m128d vresult = _mm_sqrt_pd(va);
        _mm_store_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::sqrt(a[i]);
    }
}

// Fallback implementations (no SIMD)
void SIMDUtils::fallback_add(const double* a, const double* b, double* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDUtils::fallback_multiply(const double* a, const double* b, double* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SIMDUtils::fallback_sqrt(const double* a, double* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::sqrt(a[i]);
    }
}

// Scalar operations
void SIMDUtils::vectorized_add_scalar(const double* a, double scalar, double* result, size_t size) {
    size_t i = 0;
    
    if (has_avx2()) {
        __m256d vscalar = _mm256_set1_pd(scalar);
        for (; i + 3 < size; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);
            __m256d vresult = _mm256_add_pd(va, vscalar);
            _mm256_store_pd(&result[i], vresult);
        }
    } else if (has_sse4_1()) {
        __m128d vscalar = _mm_set1_pd(scalar);
        for (; i + 1 < size; i += 2) {
            __m128d va = _mm_load_pd(&a[i]);
            __m128d vresult = _mm_add_pd(va, vscalar);
            _mm_store_pd(&result[i], vresult);
        }
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

void SIMDUtils::vectorized_multiply_scalar(const double* a, double scalar, double* result, size_t size) {
    size_t i = 0;
    
    if (has_avx2()) {
        __m256d vscalar = _mm256_set1_pd(scalar);
        for (; i + 3 < size; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);
            __m256d vresult = _mm256_mul_pd(va, vscalar);
            _mm256_store_pd(&result[i], vresult);
        }
    } else if (has_sse4_1()) {
        __m128d vscalar = _mm_set1_pd(scalar);
        for (; i + 1 < size; i += 2) {
            __m128d va = _mm_load_pd(&a[i]);
            __m128d vresult = _mm_mul_pd(va, vscalar);
            _mm_store_pd(&result[i], vresult);
        }
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

// Reduction operations
double SIMDUtils::vectorized_sum(const double* a, size_t size) {
    double sum = 0.0;
    size_t i = 0;
    
    if (has_avx2()) {
        __m256d vsum = _mm256_setzero_pd();
        for (; i + 3 < size; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);
            vsum = _mm256_add_pd(vsum, va);
        }
        
        // Extract sum from vector
        double temp[4];
        _mm256_store_pd(temp, vsum);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
    } else if (has_sse4_1()) {
        __m128d vsum = _mm_setzero_pd();
        for (; i + 1 < size; i += 2) {
            __m128d va = _mm_load_pd(&a[i]);
            vsum = _mm_add_pd(vsum, va);
        }
        
        // Extract sum from vector
        double temp[2];
        _mm_store_pd(temp, vsum);
        sum = temp[0] + temp[1];
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        sum += a[i];
    }
    
    return sum;
}

double SIMDUtils::vectorized_max(const double* a, size_t size) {
    if (size == 0) return 0.0;
    
    double max_val = a[0];
    size_t i = 1;
    
    if (has_avx2()) {
        __m256d vmax = _mm256_set1_pd(max_val);
        for (; i + 3 < size; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);
            vmax = _mm256_max_pd(vmax, va);
        }
        
        // Extract max from vector
        double temp[4];
        _mm256_store_pd(temp, vmax);
        max_val = std::max({temp[0], temp[1], temp[2], temp[3]});
    } else if (has_sse4_1()) {
        __m128d vmax = _mm_set1_pd(max_val);
        for (; i + 1 < size; i += 2) {
            __m128d va = _mm_load_pd(&a[i]);
            vmax = _mm_max_pd(vmax, va);
        }
        
        // Extract max from vector
        double temp[2];
        _mm_store_pd(temp, vmax);
        max_val = std::max(temp[0], temp[1]);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        max_val = std::max(max_val, a[i]);
    }
    
    return max_val;
}

double SIMDUtils::vectorized_min(const double* a, size_t size) {
    if (size == 0) return 0.0;
    
    double min_val = a[0];
    size_t i = 1;
    
    if (has_avx2()) {
        __m256d vmin = _mm256_set1_pd(min_val);
        for (; i + 3 < size; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);
            vmin = _mm256_min_pd(vmin, va);
        }
        
        // Extract min from vector
        double temp[4];
        _mm256_store_pd(temp, vmin);
        min_val = std::min({temp[0], temp[1], temp[2], temp[3]});
    } else if (has_sse4_1()) {
        __m128d vmin = _mm_set1_pd(min_val);
        for (; i + 1 < size; i += 2) {
            __m128d va = _mm_load_pd(&a[i]);
            vmin = _mm_min_pd(vmin, va);
        }
        
        // Extract min from vector
        double temp[2];
        _mm_store_pd(temp, vmin);
        min_val = std::min(temp[0], temp[1]);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        min_val = std::min(min_val, a[i]);
    }
    
    return min_val;
}

double SIMDUtils::vectorized_mean(const double* a, size_t size) {
    if (size == 0) return 0.0;
    return vectorized_sum(a, size) / static_cast<double>(size);
}

// Activation functions
void SIMDUtils::vectorized_relu(const double* a, double* result, size_t size) {
    size_t i = 0;
    
    if (has_avx2()) {
        __m256d vzero = _mm256_setzero_pd();
        for (; i + 3 < size; i += 4) {
            __m256d va = _mm256_load_pd(&a[i]);
            __m256d vresult = _mm256_max_pd(va, vzero);
            _mm256_store_pd(&result[i], vresult);
        }
    } else if (has_sse4_1()) {
        __m128d vzero = _mm_setzero_pd();
        for (; i + 1 < size; i += 2) {
            __m128d va = _mm_load_pd(&a[i]);
            __m128d vresult = _mm_max_pd(va, vzero);
            _mm_store_pd(&result[i], vresult);
        }
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::max(0.0, a[i]);
    }
}

void SIMDUtils::vectorized_sigmoid(const double* a, double* result, size_t size) {
    // Sigmoid: 1 / (1 + exp(-x))
    for (size_t i = 0; i < size; ++i) {
        double x = -a[i];
        if (x > 700) {  // Prevent overflow
            result[i] = 0.0;
        } else if (x < -700) {  // Prevent underflow
            result[i] = 1.0;
        } else {
            result[i] = 1.0 / (1.0 + std::exp(x));
        }
    }
}

void SIMDUtils::vectorized_softmax(const double* a, double* result, size_t size) {
    if (size == 0) return;
    
    // Find maximum value for numerical stability
    double max_val = vectorized_max(a, size);
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double val = std::exp(a[i] - max_val);
        result[i] = val;
        sum += val;
    }
    
    // Normalize
    if (sum > 0) {
        for (size_t i = 0; i < size; ++i) {
            result[i] /= sum;
        }
    }
}

} // namespace tensorcore
