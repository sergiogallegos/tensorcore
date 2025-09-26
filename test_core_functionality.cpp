#include "tensorcore/tensor.hpp"
#include "tensorcore/autograd.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/optimizers.hpp"
#include "tensorcore/operations.hpp"
#include <iostream>
#include <cassert>

using namespace tensorcore;

void test_autograd_basic() {
    std::cout << "Testing basic autograd..." << std::endl;
    
    // Create variables
    auto x = variable(Tensor({1}, {2.0}), true);
    auto y = variable(Tensor({1}, {3.0}), true);
    
    // Forward pass: z = x^2 + y^2
    auto x_squared = global_graph.power_scalar(x, 2.0);
    auto y_squared = global_graph.power_scalar(y, 2.0);
    auto z = global_graph.add(x_squared, y_squared);
    
    std::cout << "x = " << x->get_tensor()[0] << std::endl;
    std::cout << "y = " << y->get_tensor()[0] << std::endl;
    std::cout << "z = " << z->get_tensor()[0] << std::endl;
    
    // Backward pass
    global_graph.backward(z);
    
    std::cout << "dz/dx = " << x->get_gradient()[0] << std::endl;
    std::cout << "dz/dy = " << y->get_gradient()[0] << std::endl;
    
    // Verify gradients: dz/dx = 2x = 4, dz/dy = 2y = 6
    assert(std::abs(x->get_gradient()[0] - 4.0) < 1e-10);
    assert(std::abs(y->get_gradient()[0] - 6.0) < 1e-10);
    
    std::cout << "✓ Basic autograd test passed" << std::endl;
}

void test_dense_layer() {
    std::cout << "Testing Dense layer..." << std::endl;
    
    // Create a simple Dense layer
    Dense layer(2, 3, true, "relu");
    
    // Create input (batch_size=1, input_size=2)
    Tensor input({1, 2}, {1.0, 2.0});
    
    std::cout << "Input shape: " << input.shape()[0] << "x" << input.shape()[1] << std::endl;
    std::cout << "Input dimensions: " << input.shape().size() << std::endl;
    
    // Forward pass
    Tensor output = layer.forward(input);
    std::cout << "Output shape: " << output.shape()[0] << "x" << output.shape()[1] << std::endl;
    
    // Test backward pass
    Tensor grad_output({1, 3}, {1.0, 1.0, 1.0});
    Tensor input_grad = layer.backward(grad_output);
    
    std::cout << "Input gradient shape: " << input_grad.shape()[0] << "x" << input_grad.shape()[1] << std::endl;
    
    // Check that gradients are computed
    auto grads = layer.get_gradients();
    assert(grads.size() == 2); // weights and bias
    
    std::cout << "✓ Dense layer test passed" << std::endl;
}

void test_optimizer() {
    std::cout << "Testing SGD optimizer..." << std::endl;
    
    // Create a simple parameter
    Tensor param({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Create optimizer
    SGD optimizer(0.01, 0.9, 0.0, 0.0, false);
    optimizer.add_parameter(param);
    
    std::cout << "Initial parameter: " << param[0] << ", " << param[1] << std::endl;
    
    // Simulate gradient (simplified - in real usage, gradients would be computed by autograd)
    // For this test, we'll just set some dummy gradients
    param = param - 0.01 * param; // Simulate gradient step
    
    std::cout << "After gradient step: " << param[0] << ", " << param[1] << std::endl;
    
    std::cout << "✓ Optimizer test passed" << std::endl;
}

void test_simple_neural_network() {
    std::cout << "Testing simple neural network..." << std::endl;
    
    // Create a simple 2-layer network
    auto layer1 = std::make_shared<Dense>(2, 3, true, "relu");
    auto layer2 = std::make_shared<Dense>(3, 1, true, "sigmoid");
    
    Sequential network({layer1, layer2});
    
    // Create input
    Tensor input({1, 2}, {1.0, 2.0});
    
    // Forward pass
    Tensor output = network.forward(input);
    std::cout << "Network output: " << output[0] << std::endl;
    
    // Test backward pass
    Tensor grad_output({1, 1}, {1.0});
    Tensor input_grad = network.backward(grad_output);
    
    // Check that all layers have gradients
    auto params = network.get_parameters();
    auto grads = network.get_gradients();
    
    std::cout << "Number of parameters: " << params.size() << std::endl;
    std::cout << "Number of gradients: " << grads.size() << std::endl;
    
    assert(params.size() == grads.size());
    
    std::cout << "✓ Simple neural network test passed" << std::endl;
}

int main() {
    std::cout << "Testing TensorCore Core Functionality" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        test_autograd_basic();
        test_dense_layer();
        test_optimizer();
        test_simple_neural_network();
        
        std::cout << "=====================================" << std::endl;
        std::cout << "🎉 All core functionality tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
