#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include <iostream>

using namespace tensorcore;

int main() {
    std::cout << "Debugging Conv2D..." << std::endl;
    
    // Create a simple Conv2D layer
    Conv2D conv(1, 1, 3, 1, 0, true, "relu"); // 1 input channel, 1 output channel, 3x3 kernel
    
    std::cout << "Conv2D created successfully" << std::endl;
    
    // Create input (batch_size=1, channels=1, height=3, width=3)
    Tensor input({1, 1, 3, 3}, {1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0});
    
    std::cout << "Input created successfully" << std::endl;
    std::cout << "Input shape: ";
    for (size_t i = 0; i < input.shape().size(); ++i) {
        std::cout << input.shape()[i];
        if (i < input.shape().size() - 1) std::cout << "x";
    }
    std::cout << std::endl;
    
    // Try to access input with correct indexing
    try {
        double val = input({0, 0, 0, 0});
        std::cout << "Input[0,0,0,0] = " << val << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error accessing input: " << e.what() << std::endl;
    }
    
    return 0;
}
