#include "tensorcore/tensor.hpp"
#include "tensorcore/operations.hpp"
#include "tensorcore/activations.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/utils.hpp"
#include <iostream>
#include <cassert>
#include <stdexcept>

using namespace tensorcore;

void test_empty_tensor_operations() {
    std::cout << "Testing empty tensor operations..." << std::endl;
    
    // Test empty tensor creation
    Tensor empty_tensor;
    assert(empty_tensor.size() == 0);
    assert(empty_tensor.shape().empty());
    
    // Test operations on empty tensors
    Tensor empty_tensor2;
    Tensor result = empty_tensor + empty_tensor2;
    assert(result.size() == 0);
    
    // Test scalar operations on empty tensors
    Tensor scalar_result = empty_tensor + 5.0;
    assert(scalar_result.size() == 0);
    
    std::cout << "  ✓ Empty tensor operations tests passed" << std::endl;
}

void test_single_element_tensor() {
    std::cout << "Testing single element tensor..." << std::endl;
    
    Tensor single({1}, 42.0);
    assert(single.size() == 1);
    assert(single[0] == 42.0);
    
    // Test operations on single element tensors
    Tensor single2({1}, 8.0);
    Tensor result = single + single2;
    assert(result.size() == 1);
    assert(result[0] == 50.0);
    
    // Test scalar operations
    Tensor scalar_result = single * 2.0;
    assert(scalar_result.size() == 1);
    assert(scalar_result[0] == 84.0);
    
    std::cout << "  ✓ Single element tensor tests passed" << std::endl;
}

void test_large_tensor_operations() {
    std::cout << "Testing large tensor operations..." << std::endl;
    
    const size_t large_size = 1000000;
    Tensor large_tensor({large_size}, 1.0);
    
    // Test that large tensor operations work
    Tensor result = large_tensor + large_tensor;
    assert(result.size() == large_size);
    for (size_t i = 0; i < std::min(size_t(100), large_size); ++i) {
        assert(result[i] == 2.0);
    }
    
    // Test mathematical operations on large tensors
    Tensor sum_result = large_tensor.sum();
    assert(sum_result.size() == 1);
    assert(sum_result[0] == static_cast<double>(large_size));
    
    std::cout << "  ✓ Large tensor operations tests passed" << std::endl;
}

void test_shape_mismatch_errors() {
    std::cout << "Testing shape mismatch error handling..." << std::endl;
    
    Tensor a({2, 3}, 1.0);
    Tensor b({3, 2}, 2.0);
    
    // Test that operations with incompatible shapes throw errors
    bool exception_thrown = false;
    try {
        Tensor c = a + b; // This should throw an error
    } catch (const std::exception& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "  ✓ Shape mismatch error handling tests passed" << std::endl;
}

void test_invalid_indices() {
    std::cout << "Testing invalid index handling..." << std::endl;
    
    Tensor t({3, 4}, 1.0);
    
    // Test that accessing invalid indices throws errors
    bool exception_thrown = false;
    try {
        double value = t[100]; // This should throw an error
        (void)value; // Suppress unused variable warning
    } catch (const std::exception& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "  ✓ Invalid index handling tests passed" << std::endl;
}

void test_division_by_zero() {
    std::cout << "Testing division by zero handling..." << std::endl;
    
    Tensor a({3}, 1.0);
    Tensor b({3}, 0.0);
    
    // Test that division by zero produces infinity or NaN
    Tensor result = a / b;
    for (size_t i = 0; i < result.size(); ++i) {
        assert(std::isinf(result[i]) || std::isnan(result[i]));
    }
    
    std::cout << "  ✓ Division by zero handling tests passed" << std::endl;
}

void test_sqrt_negative() {
    std::cout << "Testing sqrt of negative numbers..." << std::endl;
    
    Tensor a({3}, -1.0);
    
    // Test that sqrt of negative numbers produces NaN
    Tensor result = a.sqrt();
    for (size_t i = 0; i < result.size(); ++i) {
        assert(std::isnan(result[i]));
    }
    
    std::cout << "  ✓ Sqrt of negative numbers tests passed" << std::endl;
}

void test_log_zero() {
    std::cout << "Testing log of zero..." << std::endl;
    
    Tensor a({3}, 0.0);
    
    // Test that log of zero produces negative infinity
    Tensor result = a.log();
    for (size_t i = 0; i < result.size(); ++i) {
        assert(std::isinf(result[i]) && result[i] < 0);
    }
    
    std::cout << "  ✓ Log of zero tests passed" << std::endl;
}

void test_log_negative() {
    std::cout << "Testing log of negative numbers..." << std::endl;
    
    Tensor a({3}, -1.0);
    
    // Test that log of negative numbers produces NaN
    Tensor result = a.log();
    for (size_t i = 0; i < result.size(); ++i) {
        assert(std::isnan(result[i]));
    }
    
    std::cout << "  ✓ Log of negative numbers tests passed" << std::endl;
}

void test_softmax_edge_cases() {
    std::cout << "Testing softmax edge cases..." << std::endl;
    
    // Test softmax with very large values
    Tensor large_values({3}, 1000.0);
    Tensor softmax_large = softmax(large_values);
    double sum = 0.0;
    for (size_t i = 0; i < softmax_large.size(); ++i) {
        assert(softmax_large[i] > 0.0);
        sum += softmax_large[i];
    }
    assert(std::abs(sum - 1.0) < 1e-10);
    
    // Test softmax with very small values
    Tensor small_values({3}, -1000.0);
    Tensor softmax_small = softmax(small_values);
    sum = 0.0;
    for (size_t i = 0; i < softmax_small.size(); ++i) {
        assert(softmax_small[i] > 0.0);
        sum += softmax_small[i];
    }
    assert(std::abs(sum - 1.0) < 1e-10);
    
    std::cout << "  ✓ Softmax edge cases tests passed" << std::endl;
}

void test_loss_function_edge_cases() {
    std::cout << "Testing loss function edge cases..." << std::endl;
    
    // Test loss functions with identical predictions and targets
    Tensor predictions({3}, 1.0);
    Tensor targets({3}, 1.0);
    
    Tensor mse = mse_loss(predictions, targets);
    assert(mse[0] == 0.0);
    
    Tensor mae = mae_loss(predictions, targets);
    assert(mae[0] == 0.0);
    
    // Test loss functions with very different predictions and targets
    Tensor predictions2({3}, 0.0);
    Tensor targets2({3}, 1000.0);
    
    Tensor mse2 = mse_loss(predictions2, targets2);
    assert(mse2[0] == 1000000.0); // (0-1000)^2 = 1000000
    
    std::cout << "  ✓ Loss function edge cases tests passed" << std::endl;
}

void test_memory_efficiency() {
    std::cout << "Testing memory efficiency..." << std::endl;
    
    // Test that tensors don't leak memory
    const size_t iterations = 1000;
    const size_t tensor_size = 10000;
    
    for (size_t i = 0; i < iterations; ++i) {
        Tensor t({tensor_size}, static_cast<double>(i));
        Tensor result = t + t;
        // Destructors should be called automatically
    }
    
    std::cout << "  ✓ Memory efficiency tests passed" << std::endl;
}

void test_copy_semantics() {
    std::cout << "Testing copy semantics..." << std::endl;
    
    Tensor original({3, 4}, 5.0);
    Tensor copy = original;
    
    // Test that copy is independent
    copy[0] = 10.0;
    assert(original[0] == 5.0);
    assert(copy[0] == 10.0);
    
    // Test that copy has same shape
    assert(original.shape() == copy.shape());
    assert(original.size() == copy.size());
    
    std::cout << "  ✓ Copy semantics tests passed" << std::endl;
}

void test_move_semantics() {
    std::cout << "Testing move semantics..." << std::endl;
    
    Tensor original({3, 4}, 5.0);
    Tensor moved = std::move(original);
    
    // Test that moved tensor has the data
    assert(moved.size() == 12);
    assert(moved[0] == 5.0);
    
    // Test that original tensor is in valid state
    assert(original.size() == 0);
    assert(original.shape().empty());
    
    std::cout << "  ✓ Move semantics tests passed" << std::endl;
}

int main() {
    std::cout << "Running TensorCore edge case tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        test_empty_tensor_operations();
        test_single_element_tensor();
        test_large_tensor_operations();
        test_shape_mismatch_errors();
        test_invalid_indices();
        test_division_by_zero();
        test_sqrt_negative();
        test_log_zero();
        test_log_negative();
        test_softmax_edge_cases();
        test_loss_function_edge_cases();
        test_memory_efficiency();
        test_copy_semantics();
        test_move_semantics();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "🎉 All edge case tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Edge case test failed: " << e.what() << std::endl;
        return 1;
    }
}
