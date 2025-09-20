#include "tensorcore/tensor.hpp"
#include "tensorcore/operations.hpp"
#include "tensorcore/activations.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/utils.hpp"
#include <iostream>
#include <cassert>

using namespace tensorcore;

int main() {
    std::cout << "Testing TensorCore basic functionality..." << std::endl;
    
    try {
        // Test tensor creation
        std::cout << "1. Testing tensor creation..." << std::endl;
        Tensor a = {1, 2, 3, 4};
        Tensor b = {5, 6, 7, 8};
        std::cout << "   Tensor a: " << a.to_string() << std::endl;
        std::cout << "   Tensor b: " << b.to_string() << std::endl;
        
        // Test basic operations
        std::cout << "2. Testing basic operations..." << std::endl;
        Tensor c = a + b;
        std::cout << "   a + b = " << c.to_string() << std::endl;
        
        Tensor d = a * 2.0;
        std::cout << "   a * 2 = " << d.to_string() << std::endl;
        
        // Test mathematical functions
        std::cout << "3. Testing mathematical functions..." << std::endl;
        Tensor e = sin(a);
        std::cout << "   sin(a) = " << e.to_string() << std::endl;
        
        Tensor f = exp(a);
        std::cout << "   exp(a) = " << f.to_string() << std::endl;
        
        // Test activation functions
        std::cout << "4. Testing activation functions..." << std::endl;
        Tensor g = relu(a);
        std::cout << "   relu(a) = " << g.to_string() << std::endl;
        
        Tensor h = sigmoid(a);
        std::cout << "   sigmoid(a) = " << h.to_string() << std::endl;
        
        // Test loss functions
        std::cout << "5. Testing loss functions..." << std::endl;
        Tensor predictions = {0.1, 0.3, 0.6};
        Tensor targets = {0.0, 0.0, 1.0};
        Tensor loss = cross_entropy_loss(predictions, targets);
        std::cout << "   Cross-entropy loss = " << loss.to_string() << std::endl;
        
        // Test utility functions
        std::cout << "6. Testing utility functions..." << std::endl;
        Tensor zeros_tensor = create_zeros({3, 4});
        std::cout << "   zeros(3, 4) = " << zeros_tensor.to_string() << std::endl;
        
        Tensor ones_tensor = create_ones({2, 3});
        std::cout << "   ones(2, 3) = " << ones_tensor.to_string() << std::endl;
        
        // Test matrix operations
        std::cout << "7. Testing matrix operations..." << std::endl;
        Tensor matrix_a = {{1, 2}, {3, 4}};
        Tensor matrix_b = {{5, 6}, {7, 8}};
        Tensor matrix_c = matrix_a.matmul(matrix_b);
        std::cout << "   Matrix multiplication result = " << matrix_c.to_string() << std::endl;
        
        // Test reductions
        std::cout << "8. Testing reductions..." << std::endl;
        Tensor sum_result = a.sum();
        std::cout << "   sum(a) = " << sum_result.to_string() << std::endl;
        
        Tensor mean_result = a.mean();
        std::cout << "   mean(a) = " << mean_result.to_string() << std::endl;
        
        std::cout << "All tests passed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
