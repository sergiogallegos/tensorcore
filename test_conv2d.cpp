#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/optimizers.hpp"
#include <iostream>
#include <cassert>

using namespace tensorcore;

void test_conv2d_forward() {
    std::cout << "Testing Conv2D forward pass..." << std::endl;
    
    // Create a simple Conv2D layer
    Conv2D conv(1, 2, 3, 1, 0, true, "relu"); // 1 input channel, 2 output channels, 3x3 kernel
    
    // Create input (batch_size=1, channels=1, height=5, width=5)
    Tensor input({1, 1, 5, 5}, {1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 10.0,
                                 11.0, 12.0, 13.0, 14.0, 15.0,
                                 16.0, 17.0, 18.0, 19.0, 20.0,
                                 21.0, 22.0, 23.0, 24.0, 25.0});
    
    std::cout << "Input shape: " << input.shape()[0] << "x" << input.shape()[1] 
              << "x" << input.shape()[2] << "x" << input.shape()[3] << std::endl;
    
    // Forward pass
    Tensor output = conv.forward(input);
    
    std::cout << "Output shape: " << output.shape()[0] << "x" << output.shape()[1] 
              << "x" << output.shape()[2] << "x" << output.shape()[3] << std::endl;
    
    // Verify output shape: (1, 2, 3, 3) for 5x5 input with 3x3 kernel, stride 1, padding 0
    assert(output.shape()[0] == 1); // batch size
    assert(output.shape()[1] == 2); // output channels
    assert(output.shape()[2] == 3); // output height
    assert(output.shape()[3] == 3); // output width
    
    std::cout << "✓ Conv2D forward pass test passed" << std::endl;
}

void test_maxpool2d_forward() {
    std::cout << "Testing MaxPool2D forward pass..." << std::endl;
    
    // Create a simple MaxPool2D layer
    MaxPool2D pool(2, 2, 0); // 2x2 kernel, stride 2, no padding
    
    // Create input (batch_size=1, channels=1, height=4, width=4)
    Tensor input({1, 1, 4, 4}, {1.0, 2.0, 3.0, 4.0,
                                 5.0, 6.0, 7.0, 8.0,
                                 9.0, 10.0, 11.0, 12.0,
                                 13.0, 14.0, 15.0, 16.0});
    
    std::cout << "Input shape: " << input.shape()[0] << "x" << input.shape()[1] 
              << "x" << input.shape()[2] << "x" << input.shape()[3] << std::endl;
    
    // Forward pass
    Tensor output = pool.forward(input);
    
    std::cout << "Output shape: " << output.shape()[0] << "x" << output.shape()[1] 
              << "x" << output.shape()[2] << "x" << output.shape()[3] << std::endl;
    
    // Verify output shape: (1, 1, 2, 2) for 4x4 input with 2x2 kernel, stride 2
    assert(output.shape()[0] == 1); // batch size
    assert(output.shape()[1] == 1); // channels
    assert(output.shape()[2] == 2); // output height
    assert(output.shape()[3] == 2); // output width
    
    // Check that max pooling worked correctly
    // Top-left: max(1,2,5,6) = 6
    // Top-right: max(3,4,7,8) = 8
    // Bottom-left: max(9,10,13,14) = 14
    // Bottom-right: max(11,12,15,16) = 16
    assert(std::abs(output({0, 0, 0, 0}) - 6.0) < 1e-10);
    assert(std::abs(output({0, 0, 0, 1}) - 8.0) < 1e-10);
    assert(std::abs(output({0, 0, 1, 0}) - 14.0) < 1e-10);
    assert(std::abs(output({0, 0, 1, 1}) - 16.0) < 1e-10);
    
    std::cout << "✓ MaxPool2D forward pass test passed" << std::endl;
}

void test_conv2d_backward() {
    std::cout << "Testing Conv2D backward pass..." << std::endl;
    
    // Create a simple Conv2D layer
    Conv2D conv(1, 1, 3, 1, 0, true, "relu"); // 1 input channel, 1 output channel, 3x3 kernel
    
    // Create input
    Tensor input({1, 1, 3, 3}, {1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0});
    
    // Forward pass
    Tensor output = conv.forward(input);
    
    // Create gradient output
    Tensor grad_output({1, 1, 1, 1}, {1.0}); // 1x1 output for 3x3 input with 3x3 kernel
    
    // Backward pass
    Tensor input_grad = conv.backward(grad_output);
    
    std::cout << "Input gradient shape: " << input_grad.shape()[0] << "x" << input_grad.shape()[1] 
              << "x" << input_grad.shape()[2] << "x" << input_grad.shape()[3] << std::endl;
    
    // Verify gradient shape matches input
    assert(input_grad.shape() == input.shape());
    
    // Check that gradients are computed
    auto grads = conv.get_gradients();
    assert(grads.size() == 2); // weights and bias
    
    std::cout << "✓ Conv2D backward pass test passed" << std::endl;
}

int main() {
    std::cout << "Testing TensorCore Conv2D and MaxPool2D" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        test_conv2d_forward();
        test_maxpool2d_forward();
        test_conv2d_backward();
        
        std::cout << "======================================" << std::endl;
        std::cout << "🎉 All Conv2D and MaxPool2D tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
