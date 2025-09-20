# Activation Functions

This document provides comprehensive documentation for all activation functions in TensorCore.

## Table of Contents

1. [Basic Activation Functions](#basic-activation-functions)
2. [Advanced Activation Functions](#advanced-activation-functions)
3. [Specialized Activation Functions](#specialized-activation-functions)
4. [Activation Function Properties](#activation-function-properties)
5. [Performance Considerations](#performance-considerations)

## Basic Activation Functions

### `relu`

```cpp
/**
 * @brief Rectified Linear Unit activation function
 * 
 * @details This function computes the ReLU activation, which is defined as:
 * 
 *     ReLU(x) = max(0, x) = {x if x > 0, 0 if x ≤ 0}
 * 
 * ReLU is one of the most popular activation functions in deep learning
 * due to its simplicity and effectiveness. It introduces non-linearity
 * while being computationally efficient.
 * 
 * @param x Input tensor
 * @return Tensor with ReLU activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = relu(x);  // {0, 0, 0, 1, 2}
 * 
 * // In neural network forward pass
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = relu(logits);
 * ```
 * 
 * @see leaky_relu, elu, gelu
 * @since 1.0.0
 */
Tensor relu(const Tensor& x);
```

### `leaky_relu`

```cpp
/**
 * @brief Leaky Rectified Linear Unit activation function
 * 
 * @details This function computes the Leaky ReLU activation, which is defined as:
 * 
 *     LeakyReLU(x) = {x if x > 0, αx if x ≤ 0}
 * 
 * where α is a small positive constant (typically 0.01). Leaky ReLU addresses
 * the "dying ReLU" problem by allowing small negative values to pass through.
 * 
 * @param x Input tensor
 * @param alpha Negative slope coefficient (default: 0.01)
 * @return Tensor with Leaky ReLU activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = leaky_relu(x, 0.01);  // {-0.02, -0.01, 0, 1, 2}
 * 
 * // In neural network forward pass
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = leaky_relu(logits, 0.01);
 * ```
 * 
 * @see relu, elu, gelu
 * @since 1.0.0
 */
Tensor leaky_relu(const Tensor& x, double alpha = 0.01);
```

### `elu`

```cpp
/**
 * @brief Exponential Linear Unit activation function
 * 
 * @details This function computes the ELU activation, which is defined as:
 * 
 *     ELU(x) = {x if x > 0, α(e^x - 1) if x ≤ 0}
 * 
 * where α is a positive constant (typically 1.0). ELU has smooth negative
 * values and can produce negative outputs, which helps with the "dying ReLU"
 * problem while maintaining computational efficiency.
 * 
 * @param x Input tensor
 * @param alpha Negative slope coefficient (default: 1.0)
 * @return Tensor with ELU activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = elu(x, 1.0);  // {-0.865, -0.632, 0, 1, 2}
 * 
 * // In neural network forward pass
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = elu(logits, 1.0);
 * ```
 * 
 * @see relu, leaky_relu, gelu
 * @since 1.0.0
 */
Tensor elu(const Tensor& x, double alpha = 1.0);
```

### `gelu`

```cpp
/**
 * @brief Gaussian Error Linear Unit activation function
 * 
 * @details This function computes the GELU activation, which is defined as:
 * 
 *     GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
 * 
 * where Φ(x) is the cumulative distribution function of the standard normal
 * distribution. GELU is smooth, non-monotonic, and has been shown to work
 * well in transformer models.
 * 
 * @param x Input tensor
 * @return Tensor with GELU activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = gelu(x);  // {-0.045, -0.159, 0, 0.841, 1.955}
 * 
 * // In transformer forward pass
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = gelu(logits);
 * ```
 * 
 * @see relu, elu, swish
 * @since 1.0.0
 */
Tensor gelu(const Tensor& x);
```

### `swish`

```cpp
/**
 * @brief Swish activation function
 * 
 * @details This function computes the Swish activation, which is defined as:
 * 
 *     Swish(x) = x * sigmoid(x) = x / (1 + e^(-x))
 * 
 * Swish is a smooth, non-monotonic activation function that has been shown
 * to work well in deep networks. It's self-gated and can produce negative
 * outputs.
 * 
 * @param x Input tensor
 * @return Tensor with Swish activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = swish(x);  // {-0.238, -0.269, 0, 0.731, 1.762}
 * 
 * // In neural network forward pass
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = swish(logits);
 * ```
 * 
 * @see gelu, sigmoid, tanh
 * @since 1.0.0
 */
Tensor swish(const Tensor& x);
```

## Advanced Activation Functions

### `sigmoid`

```cpp
/**
 * @brief Sigmoid activation function
 * 
 * @details This function computes the sigmoid activation, which is defined as:
 * 
 *     sigmoid(x) = 1 / (1 + e^(-x))
 * 
 * The sigmoid function maps any real number to the range (0, 1). It's commonly
 * used in binary classification problems and as a gating mechanism in LSTM cells.
 * 
 * @param x Input tensor
 * @return Tensor with sigmoid activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = sigmoid(x);  // {0.119, 0.269, 0.5, 0.731, 0.881}
 * 
 * // In binary classification
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor probabilities = sigmoid(logits);
 * ```
 * 
 * @see tanh, softmax, swish
 * @since 1.0.0
 */
Tensor sigmoid(const Tensor& x);
```

### `tanh`

```cpp
/**
 * @brief Hyperbolic tangent activation function
 * 
 * @details This function computes the tanh activation, which is defined as:
 * 
 *     tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * 
 * The tanh function maps any real number to the range (-1, 1). It's commonly
 * used in RNNs and as an alternative to sigmoid due to its zero-centered output.
 * 
 * @param x Input tensor
 * @return Tensor with tanh activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = tanh(x);  // {-0.964, -0.762, 0, 0.762, 0.964}
 * 
 * // In RNN forward pass
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = tanh(logits);
 * ```
 * 
 * @see sigmoid, softmax, swish
 * @since 1.0.0
 */
Tensor tanh(const Tensor& x);
```

### `softmax`

```cpp
/**
 * @brief Softmax activation function
 * 
 * @details This function computes the softmax activation, which is defined as:
 * 
 *     softmax(x_i) = e^(x_i) / Σ(j=0 to n-1) e^(x_j)
 * 
 * The softmax function converts a vector of real numbers into a probability
 * distribution. It's commonly used in multi-class classification problems
 * and attention mechanisms.
 * 
 * @param x Input tensor
 * @return Tensor with softmax activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 2, 3};
 * Tensor result = softmax(x);  // {0.090, 0.245, 0.665}
 * 
 * // In multi-class classification
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor probabilities = softmax(logits);
 * ```
 * 
 * @see sigmoid, tanh, log_softmax
 * @since 1.0.0
 */
Tensor softmax(const Tensor& x);
```

### `log_softmax`

```cpp
/**
 * @brief Log-softmax activation function
 * 
 * @details This function computes the log-softmax activation, which is defined as:
 * 
 *     log_softmax(x_i) = log(softmax(x_i)) = x_i - log(Σ(j=0 to n-1) e^(x_j))
 * 
 * The log-softmax function is numerically stable and commonly used in
 * cross-entropy loss computations. It's equivalent to log(softmax(x)) but
 * computed more efficiently.
 * 
 * @param x Input tensor
 * @return Tensor with log-softmax activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {1, 2, 3};
 * Tensor result = log_softmax(x);  // {-2.408, -1.408, -0.408}
 * 
 * // In cross-entropy loss computation
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor log_probs = log_softmax(logits);
 * Tensor loss = -(log_probs * targets).sum();
 * ```
 * 
 * @see softmax, sigmoid, tanh
 * @since 1.0.0
 */
Tensor log_softmax(const Tensor& x);
```

## Specialized Activation Functions

### `mish`

```cpp
/**
 * @brief Mish activation function
 * 
 * @details This function computes the Mish activation, which is defined as:
 * 
 *     Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
 * 
 * Mish is a smooth, non-monotonic activation function that has been shown
 * to work well in deep networks. It's self-gated and can produce negative
 * outputs.
 * 
 * @param x Input tensor
 * @return Tensor with Mish activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = mish(x);  // {-0.252, -0.303, 0, 0.865, 1.944}
 * 
 * // In neural network forward pass
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = mish(logits);
 * ```
 * 
 * @see swish, gelu, elu
 * @since 1.0.0
 */
Tensor mish(const Tensor& x);
```

### `hard_sigmoid`

```cpp
/**
 * @brief Hard sigmoid activation function
 * 
 * @details This function computes the hard sigmoid activation, which is defined as:
 * 
 *     hard_sigmoid(x) = max(0, min(1, 0.2x + 0.5))
 * 
 * Hard sigmoid is a piecewise linear approximation of the sigmoid function.
 * It's computationally efficient and commonly used in quantized neural networks.
 * 
 * @param x Input tensor
 * @return Tensor with hard sigmoid activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = hard_sigmoid(x);  // {0.1, 0.3, 0.5, 0.7, 0.9}
 * 
 * // In quantized neural network
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = hard_sigmoid(logits);
 * ```
 * 
 * @see sigmoid, hard_tanh, relu
 * @since 1.0.0
 */
Tensor hard_sigmoid(const Tensor& x);
```

### `hard_tanh`

```cpp
/**
 * @brief Hard tanh activation function
 * 
 * @details This function computes the hard tanh activation, which is defined as:
 * 
 *     hard_tanh(x) = max(-1, min(1, x))
 * 
 * Hard tanh is a piecewise linear approximation of the tanh function.
 * It's computationally efficient and commonly used in quantized neural networks.
 * 
 * @param x Input tensor
 * @return Tensor with hard tanh activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = hard_tanh(x);  // {-1, -1, 0, 1, 1}
 * 
 * // In quantized neural network
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = hard_tanh(logits);
 * ```
 * 
 * @see tanh, hard_sigmoid, relu
 * @since 1.0.0
 */
Tensor hard_tanh(const Tensor& x);
```

### `selu`

```cpp
/**
 * @brief Scaled Exponential Linear Unit activation function
 * 
 * @details This function computes the SELU activation, which is defined as:
 * 
 *     SELU(x) = λ * {x if x > 0, α(e^x - 1) if x ≤ 0}
 * 
 * where λ ≈ 1.0507 and α ≈ 1.6733. SELU is designed to be self-normalizing,
 * meaning that the output distribution remains approximately normal with
 * zero mean and unit variance.
 * 
 * @param x Input tensor
 * @return Tensor with SELU activation applied
 * 
 * @example
 * ```cpp
 * Tensor x = {-2, -1, 0, 1, 2};
 * Tensor result = selu(x);  // {-1.520, -1.111, 0, 1.051, 2.101}
 * 
 * // In self-normalizing neural network
 * Tensor logits = matmul(input, weights) + bias;
 * Tensor activations = selu(logits);
 * ```
 * 
 * @see elu, gelu, mish
 * @since 1.0.0
 */
Tensor selu(const Tensor& x);
```

## Activation Function Properties

### Mathematical Properties

| Function | Range | Monotonic | Smooth | Zero-centered | Sparse |
|----------|-------|-----------|--------|---------------|--------|
| ReLU | [0, ∞) | Yes | No | No | Yes |
| Leaky ReLU | (-∞, ∞) | Yes | No | No | No |
| ELU | (-α, ∞) | Yes | Yes | No | No |
| GELU | (-∞, ∞) | No | Yes | Yes | No |
| Swish | (-∞, ∞) | No | Yes | No | No |
| Sigmoid | (0, 1) | Yes | Yes | No | No |
| Tanh | (-1, 1) | Yes | Yes | Yes | No |
| Softmax | (0, 1) | Yes | Yes | No | No |

### Gradient Properties

```cpp
// ReLU gradient
Tensor x = {-2, -1, 0, 1, 2};
Tensor grad = relu_grad(x);  // {0, 0, 0, 1, 1}

// Sigmoid gradient
Tensor grad_sigmoid = sigmoid_grad(x);  // {0.105, 0.197, 0.25, 0.197, 0.105}

// Tanh gradient
Tensor grad_tanh = tanh_grad(x);  // {0.071, 0.420, 1, 0.420, 0.071}
```

### Numerical Stability

```cpp
// Softmax with numerical stability
Tensor x = {1000, 1001, 1002};  // Large values
Tensor result = softmax(x);     // Numerically stable implementation

// Log-softmax for cross-entropy
Tensor logits = {1, 2, 3};
Tensor log_probs = log_softmax(logits);  // More stable than log(softmax(x))
```

## Performance Considerations

### Computational Complexity

| Function | Complexity | Notes |
|----------|------------|-------|
| ReLU | O(n) | Simple comparison |
| Leaky ReLU | O(n) | Simple comparison |
| ELU | O(n) | Requires exp() |
| GELU | O(n) | Requires erf() |
| Swish | O(n) | Requires sigmoid() |
| Sigmoid | O(n) | Requires exp() |
| Tanh | O(n) | Requires exp() |
| Softmax | O(n) | Requires exp() and sum |

### Memory Usage

- **In-place Operations**: Some activations can be computed in-place
- **Temporary Storage**: Functions like softmax require temporary storage
- **Gradient Storage**: Gradient computation may require additional memory

### Best Practices

```cpp
// Good: Use appropriate activation for the task
Tensor logits = matmul(input, weights) + bias;

// For binary classification
Tensor probabilities = sigmoid(logits);

// For multi-class classification
Tensor probabilities = softmax(logits);

// For hidden layers
Tensor activations = relu(logits);  // or gelu, elu, etc.

// Good: Use log_softmax for cross-entropy loss
Tensor log_probs = log_softmax(logits);
Tensor loss = -(log_probs * targets).sum();

// Good: Use in-place operations when possible
Tensor x = {1, 2, 3, 4, 5};
x = relu(x);  // In-place operation
```

## Common Patterns

### Neural Network Forward Pass

```cpp
// Multi-layer perceptron
Tensor input = {1, 2, 3, 4};
Tensor hidden1 = relu(matmul(input, weights1) + bias1);
Tensor hidden2 = relu(matmul(hidden1, weights2) + bias2);
Tensor output = softmax(matmul(hidden2, weights3) + bias3);

// Convolutional neural network
Tensor conv_out = relu(conv2d(input, conv_weights));
Tensor pooled = max_pool2d(conv_out, 2);
Tensor flattened = pooled.reshape({-1});
Tensor output = softmax(matmul(flattened, fc_weights) + fc_bias);
```

### Attention Mechanisms

```cpp
// Self-attention
Tensor queries = matmul(input, W_q);
Tensor keys = matmul(input, W_k);
Tensor values = matmul(input, W_v);

Tensor scores = matmul(queries, keys.transpose()) / sqrt(d_k);
Tensor attention_weights = softmax(scores);
Tensor output = matmul(attention_weights, values);
```

### Recurrent Neural Networks

```cpp
// LSTM cell
Tensor f_t = sigmoid(matmul([h_prev, x_t], W_f) + b_f);  // Forget gate
Tensor i_t = sigmoid(matmul([h_prev, x_t], W_i) + b_i);  // Input gate
Tensor o_t = sigmoid(matmul([h_prev, x_t], W_o) + b_o);  // Output gate
Tensor c_tilde = tanh(matmul([h_prev, x_t], W_c) + b_c); // Candidate values
Tensor c_t = f_t * c_prev + i_t * c_tilde;              // Cell state
Tensor h_t = o_t * tanh(c_t);                           // Hidden state
```

### Activation Function Selection Guide

```cpp
// For hidden layers in deep networks
Tensor activations = gelu(logits);  // Modern choice, good for transformers
// or
Tensor activations = relu(logits);  // Classic choice, simple and effective

// For binary classification output
Tensor probabilities = sigmoid(logits);

// For multi-class classification output
Tensor probabilities = softmax(logits);

// For regression output
Tensor output = logits;  // Linear activation (no activation)

// For quantized networks
Tensor activations = hard_sigmoid(logits);  // or hard_tanh
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's activation functions for their machine learning projects.
