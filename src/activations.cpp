#include "tensorcore/activations.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tensorcore {

// Basic activation functions

Tensor leaky_relu(const Tensor& x, double alpha) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (x[i] > 0.0) ? x[i] : alpha * x[i];
    }
    return result;
}

Tensor elu(const Tensor& x, double alpha) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (x[i] > 0.0) ? x[i] : alpha * (std::exp(x[i]) - 1.0);
    }
    return result;
}

Tensor gelu(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        double x_val = x[i];
        double x_cubed = x_val * x_val * x_val;
        double inner = std::sqrt(2.0 / M_PI) * (x_val + 0.044715 * x_cubed);
        result[i] = x_val * 0.5 * (1.0 + std::tanh(inner));
    }
    return result;
}

Tensor swish(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] / (1.0 + std::exp(-x[i]));
    }
    return result;
}

Tensor swish(const Tensor& x, double beta) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] / (1.0 + std::exp(-beta * x[i]));
    }
    return result;
}

// Advanced activation functions

Tensor softmax(const Tensor& x) {
    if (x.shape().size() != 1) {
        throw std::invalid_argument("Softmax requires 1D tensor");
    }
    
    // Find maximum for numerical stability
    double max_val = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Compute exponentials
    Tensor exp_x = x;
    double sum_exp = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] = std::exp(x[i] - max_val);
        sum_exp += exp_x[i];
    }
    
    // Normalize
    Tensor result = exp_x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = exp_x[i] / sum_exp;
    }
    
    return result;
}

Tensor log_softmax(const Tensor& x) {
    if (x.shape().size() != 1) {
        throw std::invalid_argument("Log-softmax requires 1D tensor");
    }
    
    // Find maximum for numerical stability
    double max_val = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Compute log-sum-exp
    double log_sum_exp = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        log_sum_exp += std::exp(x[i] - max_val);
    }
    log_sum_exp = max_val + std::log(log_sum_exp);
    
    // Compute log-softmax
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] - log_sum_exp;
    }
    
    return result;
}

// Specialized activation functions
Tensor mish(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        double x_val = x[i];
        result[i] = x_val * std::tanh(std::log(1.0 + std::exp(x_val)));
    }
    return result;
}

Tensor hard_sigmoid(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        double x_val = x[i];
        result[i] = std::max(0.0, std::min(1.0, 0.2 * x_val + 0.5));
    }
    return result;
}

Tensor hard_tanh(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        double x_val = x[i];
        result[i] = std::max(-1.0, std::min(1.0, x_val));
    }
    return result;
}

Tensor hard_tanh(const Tensor& x, double min_val, double max_val) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        double x_val = x[i];
        result[i] = std::max(min_val, std::min(max_val, x_val));
    }
    return result;
}

Tensor selu(const Tensor& x) {
    const double alpha = 1.6732632423543772848170429916717;
    const double scale = 1.0507009873554804934193349852946;
    
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        double x_val = x[i];
        if (x_val > 0.0) {
            result[i] = scale * x_val;
        } else {
            result[i] = scale * alpha * (std::exp(x_val) - 1.0);
        }
    }
    return result;
}

// Activation function derivatives (for backpropagation)
Tensor relu_grad(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (x[i] > 0.0) ? 1.0 : 0.0;
    }
    return result;
}

Tensor leaky_relu_grad(const Tensor& x, double alpha) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (x[i] > 0.0) ? 1.0 : alpha;
    }
    return result;
}

Tensor elu_grad(const Tensor& x, double alpha) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] > 0.0) {
            result[i] = 1.0;
        } else {
            result[i] = alpha * std::exp(x[i]);
        }
    }
    return result;
}

Tensor sigmoid_grad(const Tensor& x) {
    Tensor sigmoid_x = sigmoid(x);
    Tensor result = sigmoid_x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid_x[i] * (1.0 - sigmoid_x[i]);
    }
    return result;
}

Tensor tanh_grad(const Tensor& x) {
    Tensor tanh_x = tanh(x);
    Tensor result = tanh_x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = 1.0 - tanh_x[i] * tanh_x[i];
    }
    return result;
}

Tensor swish_grad(const Tensor& x) {
    Tensor sigmoid_x = sigmoid(x);
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid_x[i] + x[i] * sigmoid_x[i] * (1.0 - sigmoid_x[i]);
    }
    return result;
}

// Utility functions for activation functions
Tensor softplus(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::log(1.0 + std::exp(x[i]));
    }
    return result;
}

Tensor softsign(const Tensor& x) {
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] / (1.0 + std::abs(x[i]));
    }
    return result;
}

Tensor silu(const Tensor& x) {
    // SiLU is the same as Swish
    return swish(x, 1.0);
}

Tensor glu(const Tensor& x) {
    if (x.shape().size() != 1 || x.size() % 2 != 0) {
        throw std::invalid_argument("GLU requires 1D tensor with even size");
    }
    
    size_t half_size = x.size() / 2;
    Tensor::shape_type shape = {half_size};
    Tensor result(shape);
    
    for (size_t i = 0; i < half_size; ++i) {
        Tensor single_val({1}, x[i]);
        double gate = sigmoid(single_val)[0];
        result[i] = gate * x[i + half_size];
    }
    
    return result;
}

Tensor geglu(const Tensor& x) {
    if (x.shape().size() != 1 || x.size() % 2 != 0) {
        throw std::invalid_argument("GEGLU requires 1D tensor with even size");
    }
    
    size_t half_size = x.size() / 2;
    Tensor::shape_type shape = {half_size};
    Tensor result(shape);
    
    for (size_t i = 0; i < half_size; ++i) {
        Tensor single_val({1}, x[i]);
        double gate = gelu(single_val)[0];
        result[i] = gate * x[i + half_size];
    }
    
    return result;
}

// Activation function selection helper
std::string get_activation_name([[maybe_unused]] const std::function<Tensor(const Tensor&)>& activation) {
    // This is a simple approach - in practice, you might want to use function pointers
    // or a more sophisticated method to identify activation functions
    return "unknown";
}

// Activation function factory
std::function<Tensor(const Tensor&)> create_activation(const std::string& name) {
    if (name == "relu") {
        return relu;
    } else if (name == "leaky_relu") {
        return [](const Tensor& x) { return leaky_relu(x, 0.01); };
    } else if (name == "elu") {
        return [](const Tensor& x) { return elu(x, 1.0); };
    } else if (name == "gelu") {
        return gelu;
    } else if (name == "swish") {
        return [](const Tensor& x) { return swish(x, 1.0); };
    } else if (name == "sigmoid") {
        return sigmoid;
    } else if (name == "tanh") {
        return tanh;
    } else if (name == "softmax") {
        return [](const Tensor& x) { return softmax(x); };
    } else if (name == "mish") {
        return mish;
    } else if (name == "hard_sigmoid") {
        return hard_sigmoid;
    } else if (name == "hard_tanh") {
        return [](const Tensor& x) { return hard_tanh(x, -1.0, 1.0); };
    } else if (name == "selu") {
        return selu;
    } else {
        throw std::invalid_argument("Unknown activation function: " + name);
    }
}

// Predefined activation functions
const ActivationFunction ReLU(
    [](const Tensor& x) { return relu(x); },
    [](const Tensor& x, const Tensor& grad_output) { return grad_output * relu_grad(x); }
);

const ActivationFunction Sigmoid(
    [](const Tensor& x) { return sigmoid(x); },
    [](const Tensor& x, const Tensor& grad_output) { return grad_output * sigmoid_grad(x); }
);

const ActivationFunction Tanh(
    [](const Tensor& x) { return tanh(x); },
    [](const Tensor& x, const Tensor& grad_output) { return grad_output * tanh_grad(x); }
);

const ActivationFunction Identity(
    [](const Tensor& x) { return x; },
    []([[maybe_unused]] const Tensor& x, const Tensor& grad_output) { return grad_output; }
);

} // namespace tensorcore
