#include "tensorcore/tensor.hpp"
#include "tensorcore/simd_utils.hpp"
#include "tensorcore/memory_pool.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using namespace tensorcore;

void benchmark_tensor_operations() {
    std::cout << "=== SIMD Performance Benchmark ===" << std::endl;
    
    // Test different tensor sizes
    std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};
    
    for (size_t size : sizes) {
        std::cout << "\nTesting with " << size << " elements:" << std::endl;
        
        // Create test data
        std::vector<double> a_data(size), b_data(size), result_data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        
        for (size_t i = 0; i < size; ++i) {
            a_data[i] = dis(gen);
            b_data[i] = dis(gen);
        }
        
        // Test addition
        auto start = std::chrono::high_resolution_clock::now();
        SIMDUtils::vectorized_add(a_data.data(), b_data.data(), result_data.data(), size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  SIMD Addition: " << duration.count() << " μs" << std::endl;
        
        // Test multiplication
        start = std::chrono::high_resolution_clock::now();
        SIMDUtils::vectorized_multiply(a_data.data(), b_data.data(), result_data.data(), size);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  SIMD Multiplication: " << duration.count() << " μs" << std::endl;
        
        // Test square root
        start = std::chrono::high_resolution_clock::now();
        SIMDUtils::vectorized_sqrt(a_data.data(), result_data.data(), size);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  SIMD Square Root: " << duration.count() << " μs" << std::endl;
        
        // Test sum reduction
        start = std::chrono::high_resolution_clock::now();
        double sum = SIMDUtils::vectorized_sum(a_data.data(), size);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  SIMD Sum: " << duration.count() << " μs (result: " << sum << ")" << std::endl;
    }
}

void benchmark_memory_pool() {
    std::cout << "\n=== Memory Pool Performance Benchmark ===" << std::endl;
    
    const size_t num_allocations = 10000;
    const size_t allocation_size = 1024;
    
    // Test with memory pool
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<PooledAllocation> allocations;
    allocations.reserve(num_allocations);
    
    for (size_t i = 0; i < num_allocations; ++i) {
        allocations.emplace_back(allocation_size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto pool_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Memory Pool Allocations: " << pool_duration.count() << " μs" << std::endl;
    
    // Test with standard malloc
    start = std::chrono::high_resolution_clock::now();
    std::vector<void*> malloc_ptrs;
    malloc_ptrs.reserve(num_allocations);
    
    for (size_t i = 0; i < num_allocations; ++i) {
        malloc_ptrs.push_back(std::malloc(allocation_size));
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto malloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Standard Malloc: " << malloc_duration.count() << " μs" << std::endl;
    
    // Cleanup
    for (void* ptr : malloc_ptrs) {
        std::free(ptr);
    }
    
    double speedup = static_cast<double>(malloc_duration.count()) / static_cast<double>(pool_duration.count());
    std::cout << "Memory Pool Speedup: " << speedup << "x" << std::endl;
}

void benchmark_tensor_operations_with_simd() {
    std::cout << "\n=== Tensor Operations with SIMD ===" << std::endl;
    
    // Create large tensors
    Tensor a({1000, 1000}, 1.0);
    Tensor b({1000, 1000}, 2.0);
    
    // Test tensor addition
    auto start = std::chrono::high_resolution_clock::now();
    Tensor c = a + b;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Tensor Addition (1000x1000): " << duration.count() << " μs" << std::endl;
    
    // Test tensor multiplication
    start = std::chrono::high_resolution_clock::now();
    Tensor d = a * b;
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Tensor Multiplication (1000x1000): " << duration.count() << " μs" << std::endl;
    
    // Test tensor square root
    start = std::chrono::high_resolution_clock::now();
    Tensor e = a.sqrt();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Tensor Square Root (1000x1000): " << duration.count() << " μs" << std::endl;
}

void test_cpu_features() {
    std::cout << "\n=== CPU Feature Detection ===" << std::endl;
    
    std::cout << "AVX2 Support: " << (SIMDUtils::has_avx2() ? "Yes" : "No") << std::endl;
    std::cout << "AVX Support: " << (SIMDUtils::has_avx() ? "Yes" : "No") << std::endl;
    std::cout << "SSE4.1 Support: " << (SIMDUtils::has_sse4_1() ? "Yes" : "No") << std::endl;
    
    // Test memory pool statistics
    auto& pool = get_global_memory_pool();
    std::cout << "\nMemory Pool Statistics:" << std::endl;
    std::cout << "Total Allocated: " << pool.get_total_allocated() << " bytes" << std::endl;
    std::cout << "Total Capacity: " << pool.get_total_capacity() << " bytes" << std::endl;
    std::cout << "Utilization Ratio: " << pool.get_utilization_ratio() << std::endl;
    std::cout << "Allocation Count: " << pool.get_allocation_count() << std::endl;
    std::cout << "Deallocation Count: " << pool.get_deallocation_count() << std::endl;
}

int main() {
    try {
        test_cpu_features();
        benchmark_tensor_operations();
        benchmark_memory_pool();
        benchmark_tensor_operations_with_simd();
        
        std::cout << "\n🎉 All SIMD performance tests completed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
