#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include <iostream>

using namespace tensorcore;

int main() {
    std::cout << "Testing simple Conv2D..." << std::endl;
    
    // Create a simple Conv2D layer
    Conv2D conv(1, 1, 3, 1, 0, true, "relu"); // 1 input channel, 1 output channel, 3x3 kernel
    
    // Create input (batch_size=1, channels=1, height=3, width=3)
    Tensor input({1, 1, 3, 3}, {1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0});
    
    std::cout << "Input shape: ";
    for (size_t i = 0; i < input.shape().size(); ++i) {
        std::cout << input.shape()[i];
        if (i < input.shape().size() - 1) std::cout << "x";
    }
    std::cout << std::endl;
    
    std::cout << "Input data: ";
    for (size_t i = 0; i < input.size(); ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    
    try {
        // Forward pass
        Tensor output = conv.forward(input);
        
        std::cout << "Output shape: ";
        for (size_t i = 0; i < output.shape().size(); ++i) {
            std::cout << output.shape()[i];
            if (i < output.shape().size() - 1) std::cout << "x";
        }
        std::cout << std::endl;
        
        std::cout << "Output data: ";
        for (size_t i = 0; i < output.size(); ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "✓ Conv2D test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
