#include "tensorcore/utils.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

namespace tensorcore {

// Random number generation
static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

void set_seed(unsigned int seed) {
    rng.seed(seed);
}

Tensor random_normal(const Tensor::shape_type& shape, double mean, double std) {
    if (std < 0.0) {
        throw std::invalid_argument("Standard deviation must be non-negative");
    }
    
    Tensor result(shape);
    std::normal_distribution<double> dist(mean, std);
    
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = dist(rng);
    }
    
    return result;
}

Tensor random_uniform(const Tensor::shape_type& shape, double min, double max) {
    if (min >= max) {
        throw std::invalid_argument("min must be less than max");
    }
    
    Tensor result(shape);
    std::uniform_real_distribution<double> dist(min, max);
    
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = dist(rng);
    }
    
    return result;
}

// Tensor creation utilities
Tensor zeros(const Tensor::shape_type& shape) {
    return Tensor(shape, 0.0);
}

Tensor ones(const Tensor::shape_type& shape) {
    return Tensor(shape, 1.0);
}

Tensor create_zeros(const Tensor::shape_type& shape) {
    return zeros(shape);
}

Tensor create_ones(const Tensor::shape_type& shape) {
    return ones(shape);
}

Tensor eye(size_t size) {
    Tensor::shape_type shape = {size, size};
    Tensor result(shape);
    for (size_t i = 0; i < size; ++i) {
        result({i, i}) = 1.0;
    }
    return result;
}

Tensor eye(size_t rows, size_t cols) {
    Tensor::shape_type shape = {rows, cols};
    Tensor result(shape);
    size_t min_dim = std::min(rows, cols);
    for (size_t i = 0; i < min_dim; ++i) {
        result({i, i}) = 1.0;
    }
    return result;
}

Tensor arange(double start, double stop, double step) {
    if (step == 0.0) {
        throw std::invalid_argument("step cannot be zero");
    }
    
    std::vector<double> data;
    double current = start;
    while ((step > 0.0 && current < stop) || (step < 0.0 && current > stop)) {
        data.push_back(current);
        current += step;
    }
    
    return Tensor({data.size()}, data);
}

Tensor linspace(double start, double stop, size_t num) {
    if (num < 2) {
        throw std::invalid_argument("num must be at least 2");
    }
    
    std::vector<double> data;
    double step = (stop - start) / (num - 1);
    
    for (size_t i = 0; i < num; ++i) {
        data.push_back(start + i * step);
    }
    
    return Tensor({data.size()}, data);
}

Tensor create_tensor(const std::vector<double>& data, const std::vector<size_t>& shape, bool requires_grad) {
    if (shape.empty()) {
        // Infer shape from data
        return Tensor({data.size()}, data);
    } else {
        // Use provided shape
        size_t expected_size = 1;
        for (size_t dim : shape) {
            expected_size *= dim;
        }
        
        if (data.size() != expected_size) {
            throw std::invalid_argument("Data size does not match provided shape");
        }
        
        Tensor result(shape, data);
        result.set_requires_grad(requires_grad);
        return result;
    }
}

// Mathematical utilities
double log_sum_exp(const Tensor& x) {
    if (x.size() == 0) {
        throw std::invalid_argument("Cannot compute log_sum_exp of empty tensor");
    }
    
    // Find maximum for numerical stability
    double max_val = x[0];
    for (size_t i = 1; i < x.size(); ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Compute log-sum-exp
    double sum_exp = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum_exp += std::exp(x[i] - max_val);
    }
    
    return max_val + std::log(sum_exp);
}



// Broadcasting utilities
bool is_broadcastable(const Tensor::shape_type& shape1, const Tensor::shape_type& shape2) {
    if (shape1.empty() || shape2.empty()) {
        return true;
    }
    
    size_t max_dims = std::max(shape1.size(), shape2.size());
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    
    return true;
}

Tensor::shape_type broadcast_shape(const Tensor::shape_type& shape1, const Tensor::shape_type& shape2) {
    if (!is_broadcastable(shape1, shape2)) {
        throw std::invalid_argument("Shapes are not broadcastable");
    }
    
    size_t max_dims = std::max(shape1.size(), shape2.size());
    Tensor::shape_type result(max_dims);
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
        
        result[max_dims - 1 - i] = std::max(dim1, dim2);
    }
    
    return result;
}

// Memory utilities
size_t get_memory_usage(const Tensor& tensor) {
    return tensor.size() * sizeof(Tensor::value_type);
}

size_t get_total_memory_usage(const std::vector<Tensor>& tensors) {
    size_t total = 0;
    for (const auto& tensor : tensors) {
        total += get_memory_usage(tensor);
    }
    return total;
}

// Performance utilities
void benchmark_operation(const std::function<void()>& operation, const std::string& name, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        operation();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time = static_cast<double>(duration.count()) / iterations;
    std::cout << name << " (avg over " << iterations << " iterations): " 
              << avg_time << " microseconds" << std::endl;
}

// Debugging utilities
void print_tensor_info(const Tensor& tensor, const std::string& name) {
    std::cout << "Tensor: " << name << std::endl;
    std::cout << "  Shape: [";
    for (size_t i = 0; i < tensor.shape().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tensor.shape()[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  Size: " << tensor.size() << std::endl;
    std::cout << "  Memory: " << get_memory_usage(tensor) << " bytes" << std::endl;
    std::cout << "  Requires grad: " << (tensor.requires_grad() ? "true" : "false") << std::endl;
    
    // Print first few elements
    std::cout << "  Data: [";
    size_t print_count = std::min(tensor.size(), static_cast<size_t>(10));
    for (size_t i = 0; i < print_count; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tensor[i];
    }
    if (tensor.size() > 10) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

// Validation utilities
bool validate_tensor(const Tensor& tensor) {
    // Check for NaN values
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (std::isnan(tensor[i])) {
            return false;
        }
    }
    
    // Check for infinite values
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (std::isinf(tensor[i])) {
            return false;
        }
    }
    
    return true;
}

bool validate_tensors(const std::vector<Tensor>& tensors) {
    for (const auto& tensor : tensors) {
        if (!validate_tensor(tensor)) {
            return false;
        }
    }
    return true;
}

// Gradient utilities
void zero_gradients(std::vector<Tensor>& tensors) {
    for (auto& tensor : tensors) {
        if (tensor.requires_grad()) {
            // TODO: Implement gradient zeroing when gradients are implemented
        }
    }
}

void clip_gradients(std::vector<Tensor>& tensors, double max_norm) {
    if (max_norm <= 0.0) {
        throw std::invalid_argument("max_norm must be positive");
    }
    
    // TODO: Implement gradient clipping when gradients are implemented
}

} // namespace tensorcore
