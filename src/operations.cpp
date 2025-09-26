#include "tensorcore/operations.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace tensorcore {

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] += b[i];
    }
    return result;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] -= b[i];
    }
    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for multiplication");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] *= b[i];
    }
    return result;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        if (b[i] == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        result[i] /= b[i];
    }
    return result;
}

Tensor power(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for power operation");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::pow(a[i], b[i]);
    }
    return result;
}

Tensor mod(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for modulo operation");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        if (b[i] == 0.0) {
            throw std::runtime_error("Modulo by zero");
        }
        result[i] = std::fmod(a[i], b[i]);
    }
    return result;
}

// Scalar operations
Tensor add_scalar(const Tensor& tensor, double scalar) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] += scalar;
    }
    return result;
}

Tensor subtract_scalar(const Tensor& tensor, double scalar) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] -= scalar;
    }
    return result;
}

Tensor multiply_scalar(const Tensor& tensor, double scalar) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] *= scalar;
    }
    return result;
}

Tensor divide_scalar(const Tensor& tensor, double scalar) {
    if (scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] /= scalar;
    }
    return result;
}

Tensor power_scalar(const Tensor& tensor, double scalar) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::pow(tensor[i], scalar);
    }
    return result;
}

// Trigonometric functions
Tensor sin(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::sin(tensor[i]);
    }
    return result;
}

Tensor cos(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::cos(tensor[i]);
    }
    return result;
}

Tensor tan(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::tan(tensor[i]);
    }
    return result;
}

Tensor asin(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] < -1.0 || tensor[i] > 1.0) {
            throw std::domain_error("asin: input must be in range [-1, 1]");
        }
        result[i] = std::asin(tensor[i]);
    }
    return result;
}

Tensor acos(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] < -1.0 || tensor[i] > 1.0) {
            throw std::domain_error("acos: input must be in range [-1, 1]");
        }
        result[i] = std::acos(tensor[i]);
    }
    return result;
}

Tensor atan(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::atan(tensor[i]);
    }
    return result;
}

Tensor atan2(const Tensor& y, const Tensor& x) {
    if (y.shape() != x.shape()) {
        throw std::invalid_argument("Tensor shapes must match for atan2");
    }
    
    Tensor result = y;
    for (size_t i = 0; i < y.size(); ++i) {
        result[i] = std::atan2(y[i], x[i]);
    }
    return result;
}

// Hyperbolic functions
Tensor sinh(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::sinh(tensor[i]);
    }
    return result;
}

Tensor cosh(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::cosh(tensor[i]);
    }
    return result;
}


Tensor asinh(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::asinh(tensor[i]);
    }
    return result;
}

Tensor acosh(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] < 1.0) {
            throw std::domain_error("acosh: input must be >= 1");
        }
        result[i] = std::acosh(tensor[i]);
    }
    return result;
}

Tensor atanh(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] <= -1.0 || tensor[i] >= 1.0) {
            throw std::domain_error("atanh: input must be in range (-1, 1)");
        }
        result[i] = std::atanh(tensor[i]);
    }
    return result;
}

// Logarithmic and exponential functions
Tensor log(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] <= 0.0) {
            throw std::domain_error("log: input must be positive");
        }
        result[i] = std::log(tensor[i]);
    }
    return result;
}

Tensor log2(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] <= 0.0) {
            throw std::domain_error("log2: input must be positive");
        }
        result[i] = std::log2(tensor[i]);
    }
    return result;
}

Tensor log10(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] <= 0.0) {
            throw std::domain_error("log10: input must be positive");
        }
        result[i] = std::log10(tensor[i]);
    }
    return result;
}

Tensor exp(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::exp(tensor[i]);
    }
    return result;
}

Tensor exp2(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::exp2(tensor[i]);
    }
    return result;
}

Tensor expm1(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::expm1(tensor[i]);
    }
    return result;
}

Tensor log1p(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] <= -1.0) {
            throw std::domain_error("log1p: input must be > -1");
        }
        result[i] = std::log1p(tensor[i]);
    }
    return result;
}

// Power and root functions
Tensor sqrt(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] < 0.0) {
            throw std::domain_error("sqrt: input must be non-negative");
        }
        result[i] = std::sqrt(tensor[i]);
    }
    return result;
}

Tensor cbrt(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::cbrt(tensor[i]);
    }
    return result;
}

Tensor square(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = tensor[i] * tensor[i];
    }
    return result;
}

Tensor reciprocal(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] == 0.0) {
            throw std::runtime_error("reciprocal: division by zero");
        }
        result[i] = 1.0 / tensor[i];
    }
    return result;
}

Tensor rsqrt(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] <= 0.0) {
            throw std::domain_error("rsqrt: input must be positive");
        }
        result[i] = 1.0 / std::sqrt(tensor[i]);
    }
    return result;
}

// Rounding functions
Tensor floor(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::floor(tensor[i]);
    }
    return result;
}

Tensor ceil(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::ceil(tensor[i]);
    }
    return result;
}

Tensor round(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::round(tensor[i]);
    }
    return result;
}

Tensor trunc(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::trunc(tensor[i]);
    }
    return result;
}

Tensor rint(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::rint(tensor[i]);
    }
    return result;
}

// Absolute value and sign functions
Tensor abs(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::abs(tensor[i]);
    }
    return result;
}

Tensor fabs(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::fabs(tensor[i]);
    }
    return result;
}

Tensor sign(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (tensor[i] > 0.0) {
            result[i] = 1.0;
        } else if (tensor[i] < 0.0) {
            result[i] = -1.0;
        } else {
            result[i] = 0.0;
        }
    }
    return result;
}

Tensor copysign(const Tensor& x, const Tensor& y) {
    if (x.shape() != y.shape()) {
        throw std::invalid_argument("Tensor shapes must match for copysign");
    }
    
    Tensor result = x;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::copysign(x[i], y[i]);
    }
    return result;
}

// Comparison operations
Tensor equal(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for comparison");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] == b[i]) ? 1.0 : 0.0;
    }
    return result;
}

Tensor not_equal(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for comparison");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] != b[i]) ? 1.0 : 0.0;
    }
    return result;
}

Tensor less(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for comparison");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] < b[i]) ? 1.0 : 0.0;
    }
    return result;
}

Tensor greater(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for comparison");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] > b[i]) ? 1.0 : 0.0;
    }
    return result;
}

Tensor less_equal(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for comparison");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] <= b[i]) ? 1.0 : 0.0;
    }
    return result;
}

Tensor greater_equal(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for comparison");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = (a[i] >= b[i]) ? 1.0 : 0.0;
    }
    return result;
}

// Maximum and minimum operations
Tensor maximum(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for maximum");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::max(a[i], b[i]);
    }
    return result;
}

Tensor minimum(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Tensor shapes must match for minimum");
    }
    
    Tensor result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = std::min(a[i], b[i]);
    }
    return result;
}

Tensor maximum_scalar(const Tensor& tensor, double scalar) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::max(tensor[i], scalar);
    }
    return result;
}

Tensor minimum_scalar(const Tensor& tensor, double scalar) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::min(tensor[i], scalar);
    }
    return result;
}

// Clipping operations
Tensor clip(const Tensor& tensor, double min_val, double max_val) {
    if (min_val > max_val) {
        throw std::invalid_argument("min_val must be <= max_val");
    }
    
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::max(min_val, std::min(max_val, tensor[i]));
    }
    return result;
}

Tensor clip_min(const Tensor& tensor, double min_val) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::max(min_val, tensor[i]);
    }
    return result;
}

Tensor clip_max(const Tensor& tensor, double max_val) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::min(max_val, tensor[i]);
    }
    return result;
}

// Missing functions needed for autograd
Tensor tanh(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::tanh(tensor[i]);
    }
    return result;
}

Tensor sigmoid(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = 1.0 / (1.0 + std::exp(-tensor[i]));
    }
    return result;
}

Tensor relu(const Tensor& tensor) {
    Tensor result = tensor;
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::max(0.0, tensor[i]);
    }
    return result;
}

} // namespace tensorcore
