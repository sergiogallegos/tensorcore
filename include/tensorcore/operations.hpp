#pragma once

#include "tensor.hpp"
#include <functional>

namespace tensorcore {

/**
 * @brief Mathematical operations for tensors
 * 
 * This module provides various mathematical operations that can be performed
 * on tensors, including element-wise operations, reductions, and linear algebra.
 */

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);
Tensor power(const Tensor& a, const Tensor& b);
Tensor mod(const Tensor& a, const Tensor& b);

// Scalar operations
Tensor add_scalar(const Tensor& tensor, double scalar);
Tensor subtract_scalar(const Tensor& tensor, double scalar);
Tensor multiply_scalar(const Tensor& tensor, double scalar);
Tensor divide_scalar(const Tensor& tensor, double scalar);
Tensor power_scalar(const Tensor& tensor, double scalar);

// Trigonometric functions
Tensor sin(const Tensor& tensor);
Tensor cos(const Tensor& tensor);
Tensor tan(const Tensor& tensor);
Tensor asin(const Tensor& tensor);
Tensor acos(const Tensor& tensor);
Tensor atan(const Tensor& tensor);
Tensor atan2(const Tensor& y, const Tensor& x);

// Hyperbolic functions
Tensor sinh(const Tensor& tensor);
Tensor cosh(const Tensor& tensor);
Tensor tanh(const Tensor& tensor);
Tensor asinh(const Tensor& tensor);
Tensor acosh(const Tensor& tensor);
Tensor atanh(const Tensor& tensor);

// Logarithmic and exponential functions
Tensor log(const Tensor& tensor);
Tensor log2(const Tensor& tensor);
Tensor log10(const Tensor& tensor);
Tensor exp(const Tensor& tensor);
Tensor exp2(const Tensor& tensor);
Tensor expm1(const Tensor& tensor);
Tensor log1p(const Tensor& tensor);

// Power and root functions
Tensor sqrt(const Tensor& tensor);
Tensor cbrt(const Tensor& tensor);
Tensor square(const Tensor& tensor);
Tensor reciprocal(const Tensor& tensor);
Tensor rsqrt(const Tensor& tensor); // reciprocal square root

// Rounding functions
Tensor floor(const Tensor& tensor);
Tensor ceil(const Tensor& tensor);
Tensor round(const Tensor& tensor);
Tensor trunc(const Tensor& tensor);
Tensor rint(const Tensor& tensor);

// Absolute value and sign functions
Tensor abs(const Tensor& tensor);
Tensor fabs(const Tensor& tensor);
Tensor sign(const Tensor& tensor);
Tensor copysign(const Tensor& x, const Tensor& y);

// Comparison operations
Tensor equal(const Tensor& a, const Tensor& b);
Tensor not_equal(const Tensor& a, const Tensor& b);
Tensor less(const Tensor& a, const Tensor& b);
Tensor less_equal(const Tensor& a, const Tensor& b);
Tensor greater(const Tensor& a, const Tensor& b);
Tensor greater_equal(const Tensor& a, const Tensor& b);

// Logical operations
Tensor logical_and(const Tensor& a, const Tensor& b);
Tensor logical_or(const Tensor& a, const Tensor& b);
Tensor logical_xor(const Tensor& a, const Tensor& b);
Tensor logical_not(const Tensor& tensor);

// Reduction operations
Tensor sum(const Tensor& tensor);
Tensor sum(const Tensor& tensor, int axis);
Tensor sum(const Tensor& tensor, const std::vector<int>& axes);
Tensor mean(const Tensor& tensor);
Tensor mean(const Tensor& tensor, int axis);
Tensor mean(const Tensor& tensor, const std::vector<int>& axes);
Tensor max(const Tensor& tensor);
Tensor max(const Tensor& tensor, int axis);
Tensor max(const Tensor& tensor, const std::vector<int>& axes);
Tensor min(const Tensor& tensor);
Tensor min(const Tensor& tensor, int axis);
Tensor min(const Tensor& tensor, const std::vector<int>& axes);
Tensor argmax(const Tensor& tensor);
Tensor argmax(const Tensor& tensor, int axis);
Tensor argmin(const Tensor& tensor);
Tensor argmin(const Tensor& tensor, int axis);
Tensor prod(const Tensor& tensor);
Tensor prod(const Tensor& tensor, int axis);
Tensor prod(const Tensor& tensor, const std::vector<int>& axes);
Tensor std(const Tensor& tensor);
Tensor std(const Tensor& tensor, int axis);
Tensor var(const Tensor& tensor);
Tensor var(const Tensor& tensor, int axis);

// Linear algebra operations
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor dot(const Tensor& a, const Tensor& b);
Tensor outer(const Tensor& a, const Tensor& b);
Tensor cross(const Tensor& a, const Tensor& b);
Tensor norm(const Tensor& tensor);
Tensor norm(const Tensor& tensor, int axis);
Tensor norm(const Tensor& tensor, double p);
Tensor norm(const Tensor& tensor, int axis, double p);

// Matrix operations
Tensor transpose(const Tensor& tensor);
Tensor transpose(const Tensor& tensor, const std::vector<int>& axes);
Tensor conjugate(const Tensor& tensor);
Tensor hermitian(const Tensor& tensor);
Tensor trace(const Tensor& tensor);
Tensor det(const Tensor& tensor);
Tensor inv(const Tensor& tensor);
Tensor pinv(const Tensor& tensor);
Tensor solve(const Tensor& A, const Tensor& b);
Tensor lstsq(const Tensor& A, const Tensor& b);

// Eigenvalue and SVD
std::pair<Tensor, Tensor> eig(const Tensor& tensor);
std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& tensor);
std::pair<Tensor, Tensor> eigh(const Tensor& tensor);

// Broadcasting operations
Tensor broadcast_to(const Tensor& tensor, const Tensor::shape_type& shape);
Tensor broadcast_add(const Tensor& a, const Tensor& b);
Tensor broadcast_multiply(const Tensor& a, const Tensor& b);

// Utility operations
Tensor concatenate(const std::vector<Tensor>& tensors, int axis = 0);
Tensor stack(const std::vector<Tensor>& tensors, int axis = 0);
Tensor split(const Tensor& tensor, int axis, const std::vector<int>& sizes);
Tensor tile(const Tensor& tensor, const std::vector<int>& reps);
Tensor repeat(const Tensor& tensor, const std::vector<int>& reps);
Tensor pad(const Tensor& tensor, const std::vector<std::pair<int, int>>& padding, double value = 0.0);

// Statistical operations
Tensor histogram(const Tensor& tensor, int bins = 10);
Tensor percentile(const Tensor& tensor, double q);
Tensor quantile(const Tensor& tensor, const std::vector<double>& q);
Tensor median(const Tensor& tensor);
Tensor median(const Tensor& tensor, int axis);

// Window functions
Tensor hamming(int size);
Tensor hanning(int size);
Tensor blackman(int size);
Tensor bartlett(int size);

// Convolution operations
Tensor conv1d(const Tensor& input, const Tensor& kernel, int stride = 1, int padding = 0);
Tensor conv2d(const Tensor& input, const Tensor& kernel, const std::vector<int>& stride = {1, 1}, 
              const std::vector<int>& padding = {0, 0});
Tensor max_pool1d(const Tensor& input, int kernel_size, int stride = 1, int padding = 0);
Tensor max_pool2d(const Tensor& input, const std::vector<int>& kernel_size, 
                  const std::vector<int>& stride = {1, 1}, const std::vector<int>& padding = {0, 0});

// Gradient operations (for automatic differentiation)
Tensor gradient(const Tensor& tensor, const Tensor& x);
Tensor hessian(const Tensor& tensor, const Tensor& x);

} // namespace tensorcore
