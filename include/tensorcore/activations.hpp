#pragma once

#include "tensor.hpp"

namespace tensorcore {

/**
 * @brief Activation functions for neural networks
 * 
 * This module provides various activation functions commonly used in neural networks,
 * including their forward and backward (gradient) implementations.
 */

// Basic activation functions
Tensor relu(const Tensor& x);
Tensor leaky_relu(const Tensor& x, double alpha = 0.01);
Tensor elu(const Tensor& x, double alpha = 1.0);
Tensor gelu(const Tensor& x);
Tensor swish(const Tensor& x, double beta = 1.0);
Tensor mish(const Tensor& x);

// Sigmoid family
Tensor sigmoid(const Tensor& x);
Tensor hard_sigmoid(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor hard_tanh(const Tensor& x, double min_val = -1.0, double max_val = 1.0);

// Softmax family
Tensor softmax(const Tensor& x);
Tensor softmax(const Tensor& x, int axis);
Tensor log_softmax(const Tensor& x);
Tensor log_softmax(const Tensor& x, int axis);
Tensor softplus(const Tensor& x, double beta = 1.0);
Tensor softsign(const Tensor& x);

// Gaussian family
Tensor gaussian(const Tensor& x, double mean = 0.0, double std = 1.0);
Tensor gaussian_noise(const Tensor& x, double std = 1.0);

// Other activation functions
Tensor identity(const Tensor& x);
Tensor step(const Tensor& x);
Tensor ramp(const Tensor& x);
Tensor bent_identity(const Tensor& x);
Tensor silu(const Tensor& x);
Tensor celu(const Tensor& x, double alpha = 1.0);

// Gradient functions (for backpropagation)
Tensor relu_grad(const Tensor& x, const Tensor& grad_output);
Tensor leaky_relu_grad(const Tensor& x, const Tensor& grad_output, double alpha = 0.01);
Tensor elu_grad(const Tensor& x, const Tensor& grad_output, double alpha = 1.0);
Tensor gelu_grad(const Tensor& x, const Tensor& grad_output);
Tensor swish_grad(const Tensor& x, const Tensor& grad_output, double beta = 1.0);
Tensor mish_grad(const Tensor& x, const Tensor& grad_output);

Tensor sigmoid_grad(const Tensor& x, const Tensor& grad_output);
Tensor hard_sigmoid_grad(const Tensor& x, const Tensor& grad_output);
Tensor tanh_grad(const Tensor& x, const Tensor& grad_output);
Tensor hard_tanh_grad(const Tensor& x, const Tensor& grad_output, 
                      double min_val = -1.0, double max_val = 1.0);

Tensor softmax_grad(const Tensor& x, const Tensor& grad_output);
Tensor softmax_grad(const Tensor& x, const Tensor& grad_output, int axis);
Tensor log_softmax_grad(const Tensor& x, const Tensor& grad_output);
Tensor log_softmax_grad(const Tensor& x, const Tensor& grad_output, int axis);
Tensor softplus_grad(const Tensor& x, const Tensor& grad_output, double beta = 1.0);
Tensor softsign_grad(const Tensor& x, const Tensor& grad_output);

Tensor gaussian_grad(const Tensor& x, const Tensor& grad_output, 
                     double mean = 0.0, double std = 1.0);
Tensor gaussian_noise_grad(const Tensor& x, const Tensor& grad_output, double std = 1.0);

Tensor identity_grad(const Tensor& x, const Tensor& grad_output);
Tensor step_grad(const Tensor& x, const Tensor& grad_output);
Tensor ramp_grad(const Tensor& x, const Tensor& grad_output);
Tensor bent_identity_grad(const Tensor& x, const Tensor& grad_output);
Tensor silu_grad(const Tensor& x, const Tensor& grad_output);
Tensor celu_grad(const Tensor& x, const Tensor& grad_output, double alpha = 1.0);

// Activation function class for easy switching
class ActivationFunction {
public:
    using forward_func = std::function<Tensor(const Tensor&)>;
    using backward_func = std::function<Tensor(const Tensor&, const Tensor&)>;
    
    ActivationFunction(forward_func forward, backward_func backward)
        : forward_(forward), backward_(backward) {}
    
    Tensor forward(const Tensor& x) const { return forward_(x); }
    Tensor backward(const Tensor& x, const Tensor& grad_output) const { 
        return backward_(x, grad_output); 
    }
    
private:
    forward_func forward_;
    backward_func backward_;
};

// Predefined activation functions
extern const ActivationFunction ReLU;
extern const ActivationFunction LeakyReLU;
extern const ActivationFunction ELU;
extern const ActivationFunction GELU;
extern const ActivationFunction Swish;
extern const ActivationFunction Mish;
extern const ActivationFunction Sigmoid;
extern const ActivationFunction HardSigmoid;
extern const ActivationFunction Tanh;
extern const ActivationFunction HardTanh;
extern const ActivationFunction Softmax;
extern const ActivationFunction LogSoftmax;
extern const ActivationFunction Softplus;
extern const ActivationFunction Softsign;
extern const ActivationFunction Gaussian;
extern const ActivationFunction Identity;
extern const ActivationFunction Step;
extern const ActivationFunction Ramp;
extern const ActivationFunction BentIdentity;
extern const ActivationFunction SiLU;
extern const ActivationFunction CELU;

// Utility functions
std::string get_activation_name(const ActivationFunction& activation);
ActivationFunction get_activation_by_name(const std::string& name);
std::vector<std::string> get_available_activations();

} // namespace tensorcore
