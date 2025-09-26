#pragma once

#include <vector>
#include <memory>
#include <string>
#include <initializer_list>
#include <functional>
#include <cstdint>

namespace tensorcore {

/**
 * @brief Main tensor class for multi-dimensional arrays
 * 
 * This is the core class that represents multi-dimensional arrays similar to
 * NumPy arrays or PyTorch tensors. It provides efficient storage and operations
 * for numerical computations.
 */
class Tensor {
public:
    using value_type = double;
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    using data_type = std::vector<value_type>;

    // Constructors
    Tensor();
    explicit Tensor(const shape_type& shape);
    Tensor(const shape_type& shape, const data_type& data);
    Tensor(const shape_type& shape, value_type fill_value);
    Tensor(std::initializer_list<value_type> data);
    Tensor(std::initializer_list<std::initializer_list<value_type>> data);
    
    // Copy and move constructors
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor() = default;
    
    // Assignment operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Access operators
    value_type& operator[](size_type index);
    const value_type& operator[](size_type index) const;
    value_type& operator()(const std::vector<size_type>& indices);
    const value_type& operator()(const std::vector<size_type>& indices) const;
    
    // Arithmetic operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;
    
    // Scalar operations
    Tensor operator+(value_type scalar) const;
    Tensor operator-(value_type scalar) const;
    Tensor operator*(value_type scalar) const;
    Tensor operator/(value_type scalar) const;
    
    // Friend functions for scalar operations
    friend Tensor operator*(value_type scalar, const Tensor& tensor);
    
    // In-place operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    Tensor& operator+=(value_type scalar);
    Tensor& operator-=(value_type scalar);
    Tensor& operator*=(value_type scalar);
    Tensor& operator/=(value_type scalar);
    
    // Comparison operators
    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;
    
    // Getters
    const shape_type& shape() const { return shape_; }
    size_type ndim() const { return shape_.size(); }
    size_type size() const { return size_; }
    const data_type& data() const { return data_; }
    data_type& data() { return data_; }
    bool requires_grad() const { return requires_grad_; }
    
    // Setters
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    // Shape operations
    Tensor reshape(const shape_type& new_shape) const;
    Tensor transpose() const;
    Tensor transpose(const std::vector<int>& axes) const;
    Tensor squeeze() const;
    Tensor squeeze(int axis) const;
    Tensor unsqueeze(int axis) const;
    
    // Mathematical operations
    Tensor sum() const;
    Tensor sum(int axis) const;
    Tensor mean() const;
    Tensor mean(int axis) const;
    Tensor max() const;
    Tensor max(int axis) const;
    Tensor min() const;
    Tensor min(int axis) const;
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(value_type exponent) const;
    
    // Linear algebra operations
    Tensor matmul(const Tensor& other) const;
    Tensor dot(const Tensor& other) const;
    Tensor norm() const;
    Tensor norm(int axis) const;
    Tensor var() const;
    Tensor var(int axis) const;
    
    // Utility functions
    Tensor copy() const;
    void fill(value_type value);
    void random_normal(value_type mean = 0.0, value_type std = 1.0);
    void random_uniform(value_type min = 0.0, value_type max = 1.0);
    
    // String representation
    std::string to_string() const;
    void print() const;
    
    // Memory management
    void* data_ptr() { return data_.data(); }
    const void* data_ptr() const { return data_.data(); }
    
    // Broadcasting
    bool is_broadcastable(const Tensor& other) const;
    Tensor broadcast_to(const shape_type& target_shape) const;
    
    // Indexing
    Tensor slice(const std::vector<std::pair<size_type, size_type>>& ranges) const;
    Tensor index(const std::vector<size_type>& indices) const;
    
private:
    shape_type shape_;           // Shape of the tensor
    size_type size_;             // Total number of elements
    data_type data_;             // Actual data storage
    bool requires_grad_;         // Whether gradients are required
    std::shared_ptr<Tensor> grad_; // Gradient tensor
    
    // Helper functions
    size_type compute_size(const shape_type& shape) const;
    size_type compute_index(const std::vector<size_type>& indices) const;
    std::vector<size_type> compute_indices(size_type index) const;
    bool is_valid_shape(const shape_type& shape) const;
    void validate_indices(const std::vector<size_type>& indices) const;
    
    // Broadcasting helpers
    shape_type compute_broadcast_shape(const shape_type& shape1, const shape_type& shape2) const;
    std::vector<size_type> compute_broadcast_strides(const shape_type& shape) const;
};

// Global operators for scalar operations
Tensor operator+(double scalar, const Tensor& tensor);
Tensor operator-(double scalar, const Tensor& tensor);
Tensor operator*(double scalar, const Tensor& tensor);
Tensor operator/(double scalar, const Tensor& tensor);


} // namespace tensorcore
