#include "tensorcore/tensor.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace tensorcore {

// Constructors
Tensor::Tensor() : shape_({}), size_(0), requires_grad_(false) {}

Tensor::Tensor(const shape_type& shape) 
    : shape_(shape), requires_grad_(false) {
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_type>());
    data_.resize(size_, 0.0);
}

Tensor::Tensor(const shape_type& shape, const data_type& data)
    : shape_(shape), data_(data), requires_grad_(false) {
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_type>());
    if (data_.size() != size_) {
        throw std::invalid_argument("Data size does not match tensor shape");
    }
}

Tensor::Tensor(const shape_type& shape, value_type fill_value)
    : shape_(shape), requires_grad_(false) {
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_type>());
    data_.resize(size_, fill_value);
}

Tensor::Tensor(std::initializer_list<value_type> data)
    : data_(data), requires_grad_(false) {
    shape_ = {data.size()};
    size_ = data.size();
}

Tensor::Tensor(std::initializer_list<std::initializer_list<value_type>> data)
    : requires_grad_(false) {
    if (data.size() == 0) {
        shape_ = {0, 0};
        size_ = 0;
        return;
    }
    
    size_t rows = data.size();
    size_t cols = data.begin()->size();
    
    // Check that all rows have the same size
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same size");
        }
    }
    
    shape_ = {rows, cols};
    size_ = rows * cols;
    data_.reserve(size_);
    
    for (const auto& row : data) {
        data_.insert(data_.end(), row.begin(), row.end());
    }
}

// Copy and move constructors
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), data_(other.data_), size_(other.size_), requires_grad_(other.requires_grad_) {}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), data_(std::move(other.data_)), 
      size_(other.size_), requires_grad_(other.requires_grad_) {
    other.size_ = 0;
    other.requires_grad_ = false;
}

// Assignment operators
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        data_ = other.data_;
        size_ = other.size_;
        requires_grad_ = other.requires_grad_;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
        size_ = other.size_;
        requires_grad_ = other.requires_grad_;
        other.size_ = 0;
        other.requires_grad_ = false;
    }
    return *this;
}

// Access operators
Tensor::value_type& Tensor::operator[](size_type index) {
    if (index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

const Tensor::value_type& Tensor::operator[](size_type index) const {
    if (index >= size_) {
        throw std::out_of_range("Index out of range");
    }
    return data_[index];
}

Tensor::value_type& Tensor::operator()(const std::vector<size_type>& indices) {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    size_type index = 0;
    size_type stride = 1;
    
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return data_[index];
}

const Tensor::value_type& Tensor::operator()(const std::vector<size_type>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    size_type index = 0;
    size_type stride = 1;
    
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return data_[index];
}

// Arithmetic operators
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] += other[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] -= other[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for multiplication");
    }
    
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] *= other[i];
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }
    
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        if (other[i] == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        result[i] /= other[i];
    }
    return result;
}

Tensor Tensor::operator-() const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] = -result[i];
    }
    return result;
}

// Scalar operations
Tensor Tensor::operator+(value_type scalar) const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] += scalar;
    }
    return result;
}

Tensor Tensor::operator-(value_type scalar) const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] -= scalar;
    }
    return result;
}

Tensor Tensor::operator*(value_type scalar) const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] *= scalar;
    }
    return result;
}

Tensor operator*(Tensor::value_type scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor Tensor::operator/(value_type scalar) const {
    if (scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] /= scalar;
    }
    return result;
}

// In-place operations
Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    for (size_type i = 0; i < size_; ++i) {
        data_[i] += other[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    for (size_type i = 0; i < size_; ++i) {
        data_[i] -= other[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for multiplication");
    }
    
    for (size_type i = 0; i < size_; ++i) {
        data_[i] *= other[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }
    
    for (size_type i = 0; i < size_; ++i) {
        if (other[i] == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        data_[i] /= other[i];
    }
    return *this;
}

Tensor& Tensor::operator+=(value_type scalar) {
    for (size_type i = 0; i < size_; ++i) {
        data_[i] += scalar;
    }
    return *this;
}

Tensor& Tensor::operator-=(value_type scalar) {
    for (size_type i = 0; i < size_; ++i) {
        data_[i] -= scalar;
    }
    return *this;
}

Tensor& Tensor::operator*=(value_type scalar) {
    for (size_type i = 0; i < size_; ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

Tensor& Tensor::operator/=(value_type scalar) {
    if (scalar == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    
    for (size_type i = 0; i < size_; ++i) {
        data_[i] /= scalar;
    }
    return *this;
}

// Comparison operators
bool Tensor::operator==(const Tensor& other) const {
    if (shape_ != other.shape_) {
        return false;
    }
    
    for (size_type i = 0; i < size_; ++i) {
        if (data_[i] != other[i]) {
            return false;
        }
    }
    return true;
}

bool Tensor::operator!=(const Tensor& other) const {
    return !(*this == other);
}

// Shape operations
Tensor Tensor::reshape(const shape_type& new_shape) const {
    size_type new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_type>());
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have the same total number of elements");
    }
    
    Tensor result = *this;
    result.shape_ = new_shape;
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Transpose requires 2D tensor");
    }
    
    Tensor::shape_type new_shape = {shape_[1], shape_[0]};
    Tensor result(new_shape);
    for (size_type i = 0; i < shape_[0]; ++i) {
        for (size_type j = 0; j < shape_[1]; ++j) {
            result({j, i}) = (*this)({i, j});
        }
    }
    return result;
}

Tensor Tensor::transpose(const std::vector<int>& axes) const {
    if (axes.size() != shape_.size()) {
        throw std::invalid_argument("Number of axes must match tensor dimensions");
    }
    
    // Validate axes
    std::vector<bool> used(shape_.size(), false);
    for (int axis : axes) {
        if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
            throw std::invalid_argument("Invalid axis");
        }
        if (used[axis]) {
            throw std::invalid_argument("Duplicate axis");
        }
        used[axis] = true;
    }
    
    shape_type new_shape(shape_.size());
    for (size_type i = 0; i < axes.size(); ++i) {
        new_shape[i] = shape_[axes[i]];
    }
    
    Tensor result(new_shape);
    // TODO: Implement actual transposition logic
    return result;
}

Tensor Tensor::squeeze() const {
    shape_type new_shape;
    for (size_type dim : shape_) {
        if (dim != 1) {
            new_shape.push_back(dim);
        }
    }
    
    if (new_shape.empty()) {
        new_shape = {1};
    }
    
    Tensor result = *this;
    result.shape_ = new_shape;
    return result;
}

Tensor Tensor::squeeze(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    if (shape_[axis] != 1) {
        return *this;  // Nothing to squeeze
    }
    
    shape_type new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result = *this;
    result.shape_ = new_shape;
    return result;
}

Tensor Tensor::unsqueeze(int axis) const {
    if (axis < 0 || axis > static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    shape_type new_shape = shape_;
    new_shape.insert(new_shape.begin() + axis, 1);
    
    Tensor result = *this;
    result.shape_ = new_shape;
    return result;
}

// Mathematical operations
Tensor Tensor::sum() const {
    value_type result = 0.0;
    for (size_type i = 0; i < size_; ++i) {
        result += data_[i];
    }
    return Tensor({1}, {result});
}

Tensor Tensor::sum(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    shape_type new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result(new_shape);
    
    // TODO: Implement axis-wise summation
    return result;
}

Tensor Tensor::mean() const {
    value_type sum_val = 0.0;
    for (size_type i = 0; i < size_; ++i) {
        sum_val += data_[i];
    }
    return Tensor({1}, {sum_val / static_cast<value_type>(size_)});
}

Tensor Tensor::mean(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    shape_type new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result(new_shape);
    
    // TODO: Implement axis-wise mean
    return result;
}

Tensor Tensor::max() const {
    if (size_ == 0) {
        throw std::runtime_error("Cannot find max of empty tensor");
    }
    
    value_type result = data_[0];
    for (size_type i = 1; i < size_; ++i) {
        if (data_[i] > result) {
            result = data_[i];
        }
    }
    return Tensor({1}, {result});
}

Tensor Tensor::max(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    shape_type new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result(new_shape);
    
    // TODO: Implement axis-wise max
    return result;
}

Tensor Tensor::min() const {
    if (size_ == 0) {
        throw std::runtime_error("Cannot find min of empty tensor");
    }
    
    value_type result = data_[0];
    for (size_type i = 1; i < size_; ++i) {
        if (data_[i] < result) {
            result = data_[i];
        }
    }
    return Tensor({1}, {result});
}

Tensor Tensor::min(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    shape_type new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result(new_shape);
    
    // TODO: Implement axis-wise min
    return result;
}

Tensor Tensor::abs() const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] = std::abs(data_[i]);
    }
    return result;
}

Tensor Tensor::sqrt() const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        if (data_[i] < 0) {
            throw std::domain_error("Cannot compute square root of negative number");
        }
        result[i] = std::sqrt(data_[i]);
    }
    return result;
}

Tensor Tensor::exp() const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] = std::exp(data_[i]);
    }
    return result;
}

Tensor Tensor::log() const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        if (data_[i] <= 0) {
            throw std::domain_error("Cannot compute logarithm of non-positive number");
        }
        result[i] = std::log(data_[i]);
    }
    return result;
}

Tensor Tensor::pow(value_type exponent) const {
    Tensor result = *this;
    for (size_type i = 0; i < size_; ++i) {
        result[i] = std::pow(data_[i], exponent);
    }
    return result;
}

// Linear algebra operations
Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }
    
    Tensor::shape_type new_shape = {shape_[0], other.shape_[1]};
    Tensor result(new_shape);
    
    for (size_type i = 0; i < shape_[0]; ++i) {
        for (size_type j = 0; j < other.shape_[1]; ++j) {
            value_type sum = 0.0;
            for (size_type k = 0; k < shape_[1]; ++k) {
                sum += (*this)({i, k}) * other({k, j});
            }
            result({i, j}) = sum;
        }
    }
    
    return result;
}

Tensor Tensor::dot(const Tensor& other) const {
    if (shape_.size() != 1 || other.shape_.size() != 1) {
        throw std::invalid_argument("Dot product requires 1D tensors");
    }
    
    if (shape_[0] != other.shape_[0]) {
        throw std::invalid_argument("Tensors must have the same size for dot product");
    }
    
    value_type result = 0.0;
    for (size_type i = 0; i < shape_[0]; ++i) {
        result += data_[i] * other[i];
    }
    
    return Tensor({1}, {result});
}

Tensor Tensor::norm() const {
    value_type sum_squares = 0.0;
    for (size_type i = 0; i < size_; ++i) {
        sum_squares += data_[i] * data_[i];
    }
    return Tensor({1}, {std::sqrt(sum_squares)});
}

Tensor Tensor::norm(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    shape_type new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result(new_shape);
    
    // TODO: Implement axis-wise norm
    return result;
}

Tensor Tensor::var() const {
    Tensor mean_val = mean();
    Tensor diff = *this - mean_val;
    Tensor squared_diff = diff * diff;
    return squared_diff.mean();
}

Tensor Tensor::var(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Invalid axis");
    }
    
    shape_type new_shape = shape_;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result(new_shape);
    
    // TODO: Implement axis-wise variance
    return result;
}

// Utility functions
Tensor Tensor::copy() const {
    return Tensor(*this);
}

void Tensor::fill(value_type value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::random_normal(value_type mean, value_type std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<value_type> dist(mean, std);
    
    for (size_type i = 0; i < size_; ++i) {
        data_[i] = dist(gen);
    }
}

void Tensor::random_uniform(value_type min, value_type max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<value_type> dist(min, max);
    
    for (size_type i = 0; i < size_; ++i) {
        data_[i] = dist(gen);
    }
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=";
    oss << "[";
    for (size_type i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], data=[";
    for (size_type i = 0; i < std::min(size_, static_cast<size_type>(10)); ++i) {
        if (i > 0) oss << ", ";
        oss << data_[i];
    }
    if (size_ > 10) {
        oss << ", ...";
    }
    oss << "])";
    return oss.str();
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}


bool Tensor::is_broadcastable(const Tensor& other) const {
    // Simple implementation - check if shapes are compatible
    if (shape_.size() == 0 || other.shape_.size() == 0) {
        return true;
    }
    
    // Check if one tensor can be broadcast to the other
    size_type max_dims = std::max(shape_.size(), other.shape_.size());
    
    for (size_type i = 0; i < max_dims; ++i) {
        size_type dim1 = (i < shape_.size()) ? shape_[shape_.size() - 1 - i] : 1;
        size_type dim2 = (i < other.shape_.size()) ? other.shape_[other.shape_.size() - 1 - i] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    
    return true;
}

Tensor Tensor::broadcast_to(const shape_type& target_shape) const {
    // Simple implementation - just return a copy for now
    // TODO: Implement actual broadcasting
    return *this;
}

Tensor Tensor::slice(const std::vector<std::pair<size_type, size_type>>& ranges) const {
    // Simple implementation - return a copy for now
    // TODO: Implement actual slicing
    return *this;
}

Tensor Tensor::index(const std::vector<size_type>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    
    size_type index = 0;
    size_type stride = 1;
    
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return Tensor({1}, {data_[index]});
}

} // namespace tensorcore
