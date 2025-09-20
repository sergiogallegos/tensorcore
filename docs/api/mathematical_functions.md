# Mathematical Functions

This document provides comprehensive documentation for all mathematical functions in TensorCore.

## Table of Contents

1. [Trigonometric Functions](#trigonometric-functions)
2. [Hyperbolic Functions](#hyperbolic-functions)
3. [Exponential and Logarithmic Functions](#exponential-and-logarithmic-functions)
4. [Power and Root Functions](#power-and-root-functions)
5. [Rounding Functions](#rounding-functions)
6. [Absolute Value and Sign Functions](#absolute-value-and-sign-functions)
7. [Numerical Stability](#numerical-stability)

## Trigonometric Functions

### `sin`

```cpp
/**
 * @brief Computes the sine of each element
 * 
 * @details This function computes the sine of each element in the input tensor.
 * The operation is mathematically defined as:
 * 
 *     result[i] = sin(x[i])
 * 
 * The input is expected to be in radians.
 * 
 * @param x Input tensor (in radians)
 * @return Tensor containing sine values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, M_PI/2, M_PI, 3*M_PI/2};
 * Tensor result = sin(x);  // {0, 1, 0, -1}
 * 
 * // Convert degrees to radians
 * Tensor degrees = {0, 30, 45, 60, 90};
 * Tensor radians = degrees * M_PI / 180;
 * Tensor result2 = sin(radians);  // {0, 0.5, 0.707, 0.866, 1}
 * ```
 * 
 * @see cos, tan, asin
 * @since 1.0.0
 */
Tensor sin(const Tensor& x);
```

### `cos`

```cpp
/**
 * @brief Computes the cosine of each element
 * 
 * @details This function computes the cosine of each element in the input tensor.
 * The operation is mathematically defined as:
 * 
 *     result[i] = cos(x[i])
 * 
 * The input is expected to be in radians.
 * 
 * @param x Input tensor (in radians)
 * @return Tensor containing cosine values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, M_PI/2, M_PI, 3*M_PI/2};
 * Tensor result = cos(x);  // {1, 0, -1, 0}
 * 
 * // Convert degrees to radians
 * Tensor degrees = {0, 30, 45, 60, 90};
 * Tensor radians = degrees * M_PI / 180;
 * Tensor result2 = cos(radians);  // {1, 0.866, 0.707, 0.5, 0}
 * ```
 * 
 * @see sin, tan, acos
 * @since 1.0.0
 */
Tensor cos(const Tensor& x);
```

### `tan`

```cpp
/**
 * @brief Computes the tangent of each element
 * 
 * @details This function computes the tangent of each element in the input tensor.
 * The operation is mathematically defined as:
 * 
 *     result[i] = tan(x[i])
 * 
 * The input is expected to be in radians.
 * 
 * @param x Input tensor (in radians)
 * @return Tensor containing tangent values
 * 
 * @note Returns infinity for values where cos(x) = 0
 * 
 * @example
 * ```cpp
 * Tensor x = {0, M_PI/4, M_PI/2, M_PI};
 * Tensor result = tan(x);  // {0, 1, inf, 0}
 * 
 * // Convert degrees to radians
 * Tensor degrees = {0, 30, 45, 60};
 * Tensor radians = degrees * M_PI / 180;
 * Tensor result2 = tan(radians);  // {0, 0.577, 1, 1.732}
 * ```
 * 
 * @see sin, cos, atan
 * @since 1.0.0
 */
Tensor tan(const Tensor& x);
```

### `asin`

```cpp
/**
 * @brief Computes the arcsine of each element
 * 
 * @details This function computes the arcsine (inverse sine) of each element
 * in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = arcsin(x[i])
 * 
 * The input must be in the range [-1, 1], and the output is in radians.
 * 
 * @param x Input tensor (must be in range [-1, 1])
 * @return Tensor containing arcsine values (in radians)
 * 
 * @throws std::domain_error if any element is outside [-1, 1]
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 0.5, 0.707, 1};
 * Tensor result = asin(x);  // {0, π/6, π/4, π/2}
 * 
 * // Convert result to degrees
 * Tensor degrees = result * 180 / M_PI;  // {0, 30, 45, 90}
 * ```
 * 
 * @see sin, acos, atan
 * @since 1.0.0
 */
Tensor asin(const Tensor& x);
```

### `acos`

```cpp
/**
 * @brief Computes the arccosine of each element
 * 
 * @details This function computes the arccosine (inverse cosine) of each element
 * in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = arccos(x[i])
 * 
 * The input must be in the range [-1, 1], and the output is in radians.
 * 
 * @param x Input tensor (must be in range [-1, 1])
 * @return Tensor containing arccosine values (in radians)
 * 
 * @throws std::domain_error if any element is outside [-1, 1]
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 0.866, 0.707, 0};
 * Tensor result = acos(x);  // {0, π/6, π/4, π/2}
 * 
 * // Convert result to degrees
 * Tensor degrees = result * 180 / M_PI;  // {0, 30, 45, 90}
 * ```
 * 
 * @see cos, asin, atan
 * @since 1.0.0
 */
Tensor acos(const Tensor& x);
```

### `atan`

```cpp
/**
 * @brief Computes the arctangent of each element
 * 
 * @details This function computes the arctangent (inverse tangent) of each element
 * in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = arctan(x[i])
 * 
 * The output is in radians and in the range [-π/2, π/2].
 * 
 * @param x Input tensor
 * @return Tensor containing arctangent values (in radians)
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 1.732, 1000};
 * Tensor result = atan(x);  // {0, π/4, π/3, ≈π/2}
 * 
 * // Convert result to degrees
 * Tensor degrees = result * 180 / M_PI;  // {0, 45, 60, ≈90}
 * ```
 * 
 * @see tan, asin, acos
 * @since 1.0.0
 */
Tensor atan(const Tensor& x);
```

### `atan2`

```cpp
/**
 * @brief Computes the arctangent of y/x in the correct quadrant
 * 
 * @details This function computes the arctangent of y/x, but uses the signs
 * of both arguments to determine the quadrant of the result. This is more
 * accurate than atan(y/x) because it handles the case where x = 0.
 * 
 * The operation is mathematically defined as:
 * 
 *     result[i] = atan2(y[i], x[i])
 * 
 * The output is in radians and in the range [-π, π].
 * 
 * @param y Y-coordinate tensor
 * @param x X-coordinate tensor
 * @return Tensor containing arctangent values (in radians)
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor y = {1, 1, -1, -1};
 * Tensor x = {1, -1, 1, -1};
 * Tensor result = atan2(y, x);  // {π/4, 3π/4, -π/4, -3π/4}
 * 
 * // Convert result to degrees
 * Tensor degrees = result * 180 / M_PI;  // {45, 135, -45, -135}
 * ```
 * 
 * @see atan, sin, cos
 * @since 1.0.0
 */
Tensor atan2(const Tensor& y, const Tensor& x);
```

## Hyperbolic Functions

### `sinh`

```cpp
/**
 * @brief Computes the hyperbolic sine of each element
 * 
 * @details This function computes the hyperbolic sine of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = sinh(x[i]) = (exp(x[i]) - exp(-x[i])) / 2
 * 
 * @param x Input tensor
 * @return Tensor containing hyperbolic sine values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 2, -1};
 * Tensor result = sinh(x);  // {0, 1.175, 3.627, -1.175}
 * ```
 * 
 * @see cosh, tanh, asinh
 * @since 1.0.0
 */
Tensor sinh(const Tensor& x);
```

### `cosh`

```cpp
/**
 * @brief Computes the hyperbolic cosine of each element
 * 
 * @details This function computes the hyperbolic cosine of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = cosh(x[i]) = (exp(x[i]) + exp(-x[i])) / 2
 * 
 * @param x Input tensor
 * @return Tensor containing hyperbolic cosine values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 2, -1};
 * Tensor result = cosh(x);  // {1, 1.543, 3.762, 1.543}
 * ```
 * 
 * @see sinh, tanh, acosh
 * @since 1.0.0
 */
Tensor cosh(const Tensor& x);
```

### `tanh`

```cpp
/**
 * @brief Computes the hyperbolic tangent of each element
 * 
 * @details This function computes the hyperbolic tangent of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = tanh(x[i]) = sinh(x[i]) / cosh(x[i])
 * 
 * The output is in the range [-1, 1].
 * 
 * @param x Input tensor
 * @return Tensor containing hyperbolic tangent values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 2, -1};
 * Tensor result = tanh(x);  // {0, 0.762, 0.964, -0.762}
 * ```
 * 
 * @see sinh, cosh, atanh
 * @since 1.0.0
 */
Tensor tanh(const Tensor& x);
```

### `asinh`

```cpp
/**
 * @brief Computes the inverse hyperbolic sine of each element
 * 
 * @details This function computes the inverse hyperbolic sine of each element
 * in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = asinh(x[i]) = ln(x[i] + sqrt(x[i]² + 1))
 * 
 * @param x Input tensor
 * @return Tensor containing inverse hyperbolic sine values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 2, -1};
 * Tensor result = asinh(x);  // {0, 0.881, 1.444, -0.881}
 * ```
 * 
 * @see sinh, acosh, atanh
 * @since 1.0.0
 */
Tensor asinh(const Tensor& x);
```

### `acosh`

```cpp
/**
 * @brief Computes the inverse hyperbolic cosine of each element
 * 
 * @details This function computes the inverse hyperbolic cosine of each element
 * in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = acosh(x[i]) = ln(x[i] + sqrt(x[i]² - 1))
 * 
 * The input must be in the range [1, ∞).
 * 
 * @param x Input tensor (must be >= 1)
 * @return Tensor containing inverse hyperbolic cosine values
 * 
 * @throws std::domain_error if any element is < 1
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 2, 3, 4};
 * Tensor result = acosh(x);  // {0, 1.317, 1.763, 2.063}
 * ```
 * 
 * @see cosh, asinh, atanh
 * @since 1.0.0
 */
Tensor acosh(const Tensor& x);
```

### `atanh`

```cpp
/**
 * @brief Computes the inverse hyperbolic tangent of each element
 * 
 * @details This function computes the inverse hyperbolic tangent of each element
 * in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = atanh(x[i]) = 0.5 * ln((1 + x[i]) / (1 - x[i]))
 * 
 * The input must be in the range (-1, 1).
 * 
 * @param x Input tensor (must be in range (-1, 1))
 * @return Tensor containing inverse hyperbolic tangent values
 * 
 * @throws std::domain_error if any element is outside (-1, 1)
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 0.5, -0.5, 0.9};
 * Tensor result = atanh(x);  // {0, 0.549, -0.549, 1.472}
 * ```
 * 
 * @see tanh, asinh, acosh
 * @since 1.0.0
 */
Tensor atanh(const Tensor& x);
```

## Exponential and Logarithmic Functions

### `exp`

```cpp
/**
 * @brief Computes the exponential of each element
 * 
 * @details This function computes e raised to the power of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = exp(x[i]) = e^x[i]
 * 
 * @param x Input tensor
 * @return Tensor containing exponential values
 * 
 * @note Returns infinity for very large positive values
 * @note Returns 0 for very large negative values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 2, -1};
 * Tensor result = exp(x);  // {1, 2.718, 7.389, 0.368}
 * 
 * // Softmax computation
 * Tensor logits = {1, 2, 3};
 * Tensor exp_logits = exp(logits);
 * Tensor softmax = exp_logits / exp_logits.sum();
 * ```
 * 
 * @see log, exp2, expm1
 * @since 1.0.0
 */
Tensor exp(const Tensor& x);
```

### `log`

```cpp
/**
 * @brief Computes the natural logarithm of each element
 * 
 * @details This function computes the natural logarithm (base e) of each element
 * in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = ln(x[i])
 * 
 * The input must be positive.
 * 
 * @param x Input tensor (must be positive)
 * @return Tensor containing natural logarithm values
 * 
 * @throws std::domain_error if any element is <= 0
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 2.718, 7.389, 0.368};
 * Tensor result = log(x);  // {0, 1, 2, -1}
 * 
 * // Cross-entropy loss computation
 * Tensor predictions = {0.1, 0.3, 0.6};
 * Tensor loss = -log(predictions).sum();
 * ```
 * 
 * @see exp, log2, log10
 * @since 1.0.0
 */
Tensor log(const Tensor& x);
```

### `log2`

```cpp
/**
 * @brief Computes the base-2 logarithm of each element
 * 
 * @details This function computes the base-2 logarithm of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = log₂(x[i]) = ln(x[i]) / ln(2)
 * 
 * The input must be positive.
 * 
 * @param x Input tensor (must be positive)
 * @return Tensor containing base-2 logarithm values
 * 
 * @throws std::domain_error if any element is <= 0
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 2, 4, 8, 16};
 * Tensor result = log2(x);  // {0, 1, 2, 3, 4}
 * ```
 * 
 * @see log, log10, exp
 * @since 1.0.0
 */
Tensor log2(const Tensor& x);
```

### `log10`

```cpp
/**
 * @brief Computes the base-10 logarithm of each element
 * 
 * @details This function computes the base-10 logarithm of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = log₁₀(x[i]) = ln(x[i]) / ln(10)
 * 
 * The input must be positive.
 * 
 * @param x Input tensor (must be positive)
 * @return Tensor containing base-10 logarithm values
 * 
 * @throws std::domain_error if any element is <= 0
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 10, 100, 1000};
 * Tensor result = log10(x);  // {0, 1, 2, 3}
 * ```
 * 
 * @see log, log2, exp
 * @since 1.0.0
 */
Tensor log10(const Tensor& x);
```

### `exp2`

```cpp
/**
 * @brief Computes 2 raised to the power of each element
 * 
 * @details This function computes 2 raised to the power of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = 2^x[i]
 * 
 * @param x Input tensor
 * @return Tensor containing 2^x values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 2, 3, 4};
 * Tensor result = exp2(x);  // {1, 2, 4, 8, 16}
 * ```
 * 
 * @see exp, log2, pow
 * @since 1.0.0
 */
Tensor exp2(const Tensor& x);
```

### `expm1`

```cpp
/**
 * @brief Computes exp(x) - 1 for numerical stability
 * 
 * @details This function computes exp(x) - 1, but with better numerical
 * stability for small values of x. The operation is mathematically defined as:
 * 
 *     result[i] = exp(x[i]) - 1
 * 
 * This function is more accurate than exp(x) - 1 when x is close to zero.
 * 
 * @param x Input tensor
 * @return Tensor containing exp(x) - 1 values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 0.001, 0.01, 0.1};
 * Tensor result = expm1(x);  // {0, 0.001, 0.010, 0.105}
 * ```
 * 
 * @see exp, log1p
 * @since 1.0.0
 */
Tensor expm1(const Tensor& x);
```

### `log1p`

```cpp
/**
 * @brief Computes log(1 + x) for numerical stability
 * 
 * @details This function computes log(1 + x), but with better numerical
 * stability for small values of x. The operation is mathematically defined as:
 * 
 *     result[i] = ln(1 + x[i])
 * 
 * This function is more accurate than log(1 + x) when x is close to zero.
 * 
 * @param x Input tensor (must be > -1)
 * @return Tensor containing log(1 + x) values
 * 
 * @throws std::domain_error if any element is <= -1
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 0.001, 0.01, 0.1};
 * Tensor result = log1p(x);  // {0, 0.001, 0.010, 0.095}
 * ```
 * 
 * @see log, expm1
 * @since 1.0.0
 */
Tensor log1p(const Tensor& x);
```

## Power and Root Functions

### `sqrt`

```cpp
/**
 * @brief Computes the square root of each element
 * 
 * @details This function computes the square root of each element in the input
 * tensor. The operation is mathematically defined as:
 * 
 *     result[i] = √x[i]
 * 
 * The input must be non-negative.
 * 
 * @param x Input tensor (must be non-negative)
 * @return Tensor containing square root values
 * 
 * @throws std::domain_error if any element is negative
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 4, 9, 16};
 * Tensor result = sqrt(x);  // {0, 1, 2, 3, 4}
 * 
 * // L2 norm computation
 * Tensor vector = {3, 4, 5};
 * Tensor norm = sqrt((vector * vector).sum());  // 7.071
 * ```
 * 
 * @see cbrt, square, pow
 * @since 1.0.0
 */
Tensor sqrt(const Tensor& x);
```

### `cbrt`

```cpp
/**
 * @brief Computes the cube root of each element
 * 
 * @details This function computes the cube root of each element in the input
 * tensor. The operation is mathematically defined as:
 * 
 *     result[i] = ∛x[i]
 * 
 * @param x Input tensor
 * @return Tensor containing cube root values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 8, 27, 64};
 * Tensor result = cbrt(x);  // {0, 1, 2, 3, 4}
 * ```
 * 
 * @see sqrt, square, pow
 * @since 1.0.0
 */
Tensor cbrt(const Tensor& x);
```

### `square`

```cpp
/**
 * @brief Computes the square of each element
 * 
 * @details This function computes the square of each element in the input
 * tensor. The operation is mathematically defined as:
 * 
 *     result[i] = x[i]²
 * 
 * @param x Input tensor
 * @return Tensor containing squared values
 * 
 * @example
 * ```cpp
 * Tensor x = {0, 1, 2, 3, 4};
 * Tensor result = square(x);  // {0, 1, 4, 9, 16}
 * 
 * // Mean squared error computation
 * Tensor predictions = {1, 2, 3};
 * Tensor targets = {1.1, 1.9, 3.1};
 * Tensor mse = (predictions - targets).square().mean();
 * ```
 * 
 * @see sqrt, cbrt, pow
 * @since 1.0.0
 */
Tensor square(const Tensor& x);
```

### `reciprocal`

```cpp
/**
 * @brief Computes the reciprocal of each element
 * 
 * @details This function computes the reciprocal (1/x) of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = 1 / x[i]
 * 
 * The input must be non-zero.
 * 
 * @param x Input tensor (must be non-zero)
 * @return Tensor containing reciprocal values
 * 
 * @throws std::runtime_error if any element is zero
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 2, 4, 8};
 * Tensor result = reciprocal(x);  // {1, 0.5, 0.25, 0.125}
 * ```
 * 
 * @see sqrt, rsqrt
 * @since 1.0.0
 */
Tensor reciprocal(const Tensor& x);
```

### `rsqrt`

```cpp
/**
 * @brief Computes the reciprocal square root of each element
 * 
 * @details This function computes the reciprocal square root (1/√x) of each
 * element in the input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = 1 / √x[i]
 * 
 * The input must be positive.
 * 
 * @param x Input tensor (must be positive)
 * @return Tensor containing reciprocal square root values
 * 
 * @throws std::domain_error if any element is <= 0
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 4, 9, 16};
 * Tensor result = rsqrt(x);  // {1, 0.5, 0.333, 0.25}
 * ```
 * 
 * @see sqrt, reciprocal
 * @since 1.0.0
 */
Tensor rsqrt(const Tensor& x);
```

## Rounding Functions

### `floor`

```cpp
/**
 * @brief Rounds each element down to the nearest integer
 * 
 * @details This function rounds each element down to the nearest integer.
 * The operation is mathematically defined as:
 * 
 *     result[i] = ⌊x[i]⌋
 * 
 * @param x Input tensor
 * @return Tensor containing floor values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2.7, -1.5, -0.5, 0.5, 1.5, 2.7};
 * Tensor result = floor(x);  // {-3, -2, -1, 0, 1, 2}
 * ```
 * 
 * @see ceil, round, trunc
 * @since 1.0.0
 */
Tensor floor(const Tensor& x);
```

### `ceil`

```cpp
/**
 * @brief Rounds each element up to the nearest integer
 * 
 * @details This function rounds each element up to the nearest integer.
 * The operation is mathematically defined as:
 * 
 *     result[i] = ⌈x[i]⌉
 * 
 * @param x Input tensor
 * @return Tensor containing ceiling values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2.7, -1.5, -0.5, 0.5, 1.5, 2.7};
 * Tensor result = ceil(x);  // {-2, -1, 0, 1, 2, 3}
 * ```
 * 
 * @see floor, round, trunc
 * @since 1.0.0
 */
Tensor ceil(const Tensor& x);
```

### `round`

```cpp
/**
 * @brief Rounds each element to the nearest integer
 * 
 * @details This function rounds each element to the nearest integer.
 * The operation is mathematically defined as:
 * 
 *     result[i] = round(x[i])
 * 
 * @param x Input tensor
 * @return Tensor containing rounded values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2.7, -1.5, -0.5, 0.5, 1.5, 2.7};
 * Tensor result = round(x);  // {-3, -2, 0, 0, 2, 3}
 * ```
 * 
 * @see floor, ceil, trunc
 * @since 1.0.0
 */
Tensor round(const Tensor& x);
```

### `trunc`

```cpp
/**
 * @brief Truncates each element towards zero
 * 
 * @details This function truncates each element towards zero, removing the
 * fractional part. The operation is mathematically defined as:
 * 
 *     result[i] = trunc(x[i])
 * 
 * @param x Input tensor
 * @return Tensor containing truncated values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2.7, -1.5, -0.5, 0.5, 1.5, 2.7};
 * Tensor result = trunc(x);  // {-2, -1, 0, 0, 1, 2}
 * ```
 * 
 * @see floor, ceil, round
 * @since 1.0.0
 */
Tensor trunc(const Tensor& x);
```

### `rint`

```cpp
/**
 * @brief Rounds each element to the nearest integer using current rounding mode
 * 
 * @details This function rounds each element to the nearest integer using the
 * current floating-point rounding mode. This is similar to round() but uses
 * the system's rounding mode.
 * 
 * @param x Input tensor
 * @return Tensor containing rounded values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2.7, -1.5, -0.5, 0.5, 1.5, 2.7};
 * Tensor result = rint(x);  // Depends on rounding mode
 * ```
 * 
 * @see floor, ceil, round, trunc
 * @since 1.0.0
 */
Tensor rint(const Tensor& x);
```

## Absolute Value and Sign Functions

### `abs`

```cpp
/**
 * @brief Computes the absolute value of each element
 * 
 * @details This function computes the absolute value of each element in the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = |x[i]|
 * 
 * @param x Input tensor
 * @return Tensor containing absolute values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = abs(x);  // {2, 1, 0, 1, 2}
 * 
 * // L1 norm computation
 * Tensor vector = {-3, 4, -5};
 * Tensor l1_norm = abs(vector).sum();  // 12
 * ```
 * 
 * @see fabs, sign
 * @since 1.0.0
 */
Tensor abs(const Tensor& x);
```

### `fabs`

```cpp
/**
 * @brief Computes the absolute value of each element (floating-point)
 * 
 * @details This function computes the absolute value of each element in the
 * input tensor, specifically designed for floating-point values. It's
 * equivalent to abs() but may have different performance characteristics.
 * 
 * @param x Input tensor
 * @return Tensor containing absolute values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2.5, -1.0, 0.0, 1.0, 2.5};
 * Tensor result = fabs(x);  // {2.5, 1.0, 0.0, 1.0, 2.5}
 * ```
 * 
 * @see abs, sign
 * @since 1.0.0
 */
Tensor fabs(const Tensor& x);
```

### `sign`

```cpp
/**
 * @brief Computes the sign of each element
 * 
 * @details This function computes the sign of each element in the input tensor.
 * The operation is mathematically defined as:
 * 
 *     result[i] = sign(x[i]) = {1 if x[i] > 0, 0 if x[i] = 0, -1 if x[i] < 0}
 * 
 * @param x Input tensor
 * @return Tensor containing sign values
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = sign(x);  // {-1, -1, 0, 1, 1}
 * ```
 * 
 * @see abs, fabs
 * @since 1.0.0
 */
Tensor sign(const Tensor& x);
```

### `copysign`

```cpp
/**
 * @brief Copies the sign of y to the magnitude of x
 * 
 * @details This function copies the sign of each element in y to the magnitude
 * of each element in x. The operation is mathematically defined as:
 * 
 *     result[i] = copysign(x[i], y[i]) = |x[i]| * sign(y[i])
 * 
 * @param x Input tensor (magnitude)
 * @param y Input tensor (sign)
 * @return Tensor with magnitude of x and sign of y
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 2, 3, 4};
 * Tensor y = {-1, 1, -1, 1};
 * Tensor result = copysign(x, y);  // {-1, 2, -3, 4}
 * ```
 * 
 * @see abs, sign
 * @since 1.0.0
 */
Tensor copysign(const Tensor& x, const Tensor& y);
```

## Numerical Stability

### Softmax Implementation

```cpp
/**
 * @brief Computes the softmax function with numerical stability
 * 
 * @details The softmax function converts a vector of real numbers into a
 * probability distribution. It's defined as:
 * 
 *     softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
 * 
 * The subtraction of max(x) is crucial for numerical stability, preventing
 * overflow when computing exponentials of large numbers.
 * 
 * @param x Input tensor of shape (N,) or (..., N)
 * @return Tensor of same shape as input, with values in [0,1] that sum to 1
 * 
 * @note This implementation uses the numerically stable version to prevent
 *       overflow/underflow issues common in naive implementations
 * 
 * @example
 * ```cpp
 * Tensor x = {1.0, 2.0, 3.0};
 * Tensor result = softmax(x);  // ≈ {0.09, 0.24, 0.67}
 * 
 * // Verify probabilities sum to 1
 * double sum = result.sum();  // ≈ 1.0
 * ```
 * 
 * @see log_softmax, sigmoid
 * @since 1.0.0
 */
Tensor softmax(const Tensor& x);
```

### Log-Sum-Exp Trick

```cpp
/**
 * @brief Computes log(Σ exp(x_i)) with numerical stability
 * 
 * @details This function computes the logarithm of the sum of exponentials
 * with numerical stability. It's commonly used in softmax and log-likelihood
 * computations.
 * 
 * @param x Input tensor
 * @return Scalar value of log(Σ exp(x_i))
 * 
 * @example
 * ```cpp
 * Tensor x = {1.0, 2.0, 3.0};
 * double result = log_sum_exp(x);  // ≈ 3.407
 * ```
 * 
 * @see softmax, expm1
 * @since 1.0.0
 */
double log_sum_exp(const Tensor& x);
```

## Common Patterns

### Activation Functions

```cpp
// ReLU activation
Tensor relu(const Tensor& x) {
    return max(x, 0.0);
}

// Leaky ReLU activation
Tensor leaky_relu(const Tensor& x, double alpha = 0.01) {
    return max(x, alpha * x);
}

// ELU activation
Tensor elu(const Tensor& x, double alpha = 1.0) {
    return where(x > 0, x, alpha * (exp(x) - 1));
}
```

### Loss Functions

```cpp
// Mean Squared Error
Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    return (predictions - targets).square().mean();
}

// Cross-Entropy Loss
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    Tensor log_softmax = log_softmax(logits);
    return -(log_softmax * targets).sum();
}
```

### Normalization

```cpp
// L2 normalization
Tensor l2_normalize(const Tensor& x) {
    Tensor norm = sqrt((x * x).sum());
    return x / norm;
}

// Min-max normalization
Tensor min_max_normalize(const Tensor& x) {
    Tensor min_val = x.min();
    Tensor max_val = x.max();
    return (x - min_val) / (max_val - min_val);
}
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's mathematical functions for their machine learning projects.
