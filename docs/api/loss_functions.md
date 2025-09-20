# Loss Functions

This document provides comprehensive documentation for all loss functions in TensorCore.

## Table of Contents

1. [Regression Loss Functions](#regression-loss-functions)
2. [Classification Loss Functions](#classification-loss-functions)
3. [Specialized Loss Functions](#specialized-loss-functions)
4. [Loss Function Properties](#loss-function-properties)
5. [Performance Considerations](#performance-considerations)

## Regression Loss Functions

### `mse_loss`

```cpp
/**
 * @brief Mean Squared Error loss function
 * 
 * @details This function computes the mean squared error between predictions
 * and targets. The operation is mathematically defined as:
 * 
 *     MSE = (1/n) * Σ(i=0 to n-1) (predictions[i] - targets[i])²
 * 
 * MSE is commonly used for regression problems and is sensitive to outliers.
 * It penalizes large errors more heavily than small errors.
 * 
 * @param predictions Predicted values
 * @param targets Target values
 * @return Scalar MSE loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * 
 * @example
 * ```cpp
 * Tensor predictions = {1.0, 2.0, 3.0, 4.0};
 * Tensor targets = {1.1, 1.9, 3.1, 3.9};
 * double loss = mse_loss(predictions, targets);  // ≈ 0.01
 * 
 * // In training loop
 * Tensor pred = model.forward(input);
 * double loss = mse_loss(pred, targets);
 * loss.backward();
 * ```
 * 
 * @see mae_loss, huber_loss, smooth_l1_loss
 * @since 1.0.0
 */
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);
```

### `mae_loss`

```cpp
/**
 * @brief Mean Absolute Error loss function
 * 
 * @details This function computes the mean absolute error between predictions
 * and targets. The operation is mathematically defined as:
 * 
 *     MAE = (1/n) * Σ(i=0 to n-1) |predictions[i] - targets[i]|
 * 
 * MAE is less sensitive to outliers than MSE and provides a more robust
 * measure of error for regression problems.
 * 
 * @param predictions Predicted values
 * @param targets Target values
 * @return Scalar MAE loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * 
 * @example
 * ```cpp
 * Tensor predictions = {1.0, 2.0, 3.0, 4.0};
 * Tensor targets = {1.1, 1.9, 3.1, 3.9};
 * double loss = mae_loss(predictions, targets);  // ≈ 0.1
 * 
 * // In training loop
 * Tensor pred = model.forward(input);
 * double loss = mae_loss(pred, targets);
 * loss.backward();
 * ```
 * 
 * @see mse_loss, huber_loss, smooth_l1_loss
 * @since 1.0.0
 */
Tensor mae_loss(const Tensor& predictions, const Tensor& targets);
```

### `huber_loss`

```cpp
/**
 * @brief Huber loss function
 * 
 * @details This function computes the Huber loss between predictions and targets.
 * The operation is mathematically defined as:
 * 
 *     Huber(x) = {0.5 * x² if |x| ≤ δ, δ * (|x| - 0.5 * δ) otherwise}
 * 
 * where x = predictions - targets and δ is the threshold parameter.
 * Huber loss combines the benefits of MSE and MAE, being less sensitive
 * to outliers than MSE while remaining smooth.
 * 
 * @param predictions Predicted values
 * @param targets Target values
 * @param delta Threshold parameter (default: 1.0)
 * @return Scalar Huber loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * 
 * @example
 * ```cpp
 * Tensor predictions = {1.0, 2.0, 3.0, 4.0};
 * Tensor targets = {1.1, 1.9, 3.1, 3.9};
 * double loss = huber_loss(predictions, targets, 1.0);  // ≈ 0.01
 * 
 * // In training loop
 * Tensor pred = model.forward(input);
 * double loss = huber_loss(pred, targets, 1.0);
 * loss.backward();
 * ```
 * 
 * @see mse_loss, mae_loss, smooth_l1_loss
 * @since 1.0.0
 */
Tensor huber_loss(const Tensor& predictions, const Tensor& targets, double delta = 1.0);
```

### `smooth_l1_loss`

```cpp
/**
 * @brief Smooth L1 loss function
 * 
 * @details This function computes the smooth L1 loss between predictions and
 * targets. The operation is mathematically defined as:
 * 
 *     SmoothL1(x) = {0.5 * x² if |x| < 1, |x| - 0.5 otherwise}
 * 
 * where x = predictions - targets. Smooth L1 loss is similar to Huber loss
 * with δ = 1 and is commonly used in object detection tasks.
 * 
 * @param predictions Predicted values
 * @param targets Target values
 * @return Scalar smooth L1 loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * 
 * @example
 * ```cpp
 * Tensor predictions = {1.0, 2.0, 3.0, 4.0};
 * Tensor targets = {1.1, 1.9, 3.1, 3.9};
 * double loss = smooth_l1_loss(predictions, targets);  // ≈ 0.01
 * 
 * // In object detection training
 * Tensor bbox_pred = model.forward(input);
 * double loss = smooth_l1_loss(bbox_pred, bbox_targets);
 * loss.backward();
 * ```
 * 
 * @see mse_loss, mae_loss, huber_loss
 * @since 1.0.0
 */
Tensor smooth_l1_loss(const Tensor& predictions, const Tensor& targets);
```

## Classification Loss Functions

### `cross_entropy_loss`

```cpp
/**
 * @brief Cross-entropy loss function
 * 
 * @details This function computes the cross-entropy loss between predictions
 * and targets. The operation is mathematically defined as:
 * 
 *     CE = -Σ(i=0 to n-1) targets[i] * log(predictions[i])
 * 
 * Cross-entropy loss is commonly used for multi-class classification problems.
 * It measures the difference between the predicted probability distribution
 * and the true probability distribution.
 * 
 * @param predictions Predicted probabilities (must sum to 1)
 * @param targets Target probabilities (must sum to 1)
 * @return Scalar cross-entropy loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * @throws std::domain_error if predictions contain non-positive values
 * 
 * @example
 * ```cpp
 * Tensor predictions = {0.1, 0.3, 0.6};  // Predicted probabilities
 * Tensor targets = {0.0, 0.0, 1.0};      // True class (one-hot)
 * double loss = cross_entropy_loss(predictions, targets);  // ≈ 0.511
 * 
 * // In training loop
 * Tensor logits = model.forward(input);
 * Tensor probs = softmax(logits);
 * double loss = cross_entropy_loss(probs, targets);
 * loss.backward();
 * ```
 * 
 * @see binary_cross_entropy_loss, categorical_cross_entropy_loss
 * @since 1.0.0
 */
Tensor cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
```

### `binary_cross_entropy_loss`

```cpp
/**
 * @brief Binary cross-entropy loss function
 * 
 * @details This function computes the binary cross-entropy loss between
 * predictions and targets. The operation is mathematically defined as:
 * 
 *     BCE = -(1/n) * Σ(i=0 to n-1) [targets[i] * log(predictions[i]) + 
 *                                   (1 - targets[i]) * log(1 - predictions[i])]
 * 
 * Binary cross-entropy loss is used for binary classification problems
 * where targets are in the range [0, 1].
 * 
 * @param predictions Predicted probabilities (must be in [0, 1])
 * @param targets Target probabilities (must be in [0, 1])
 * @return Scalar binary cross-entropy loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * @throws std::domain_error if predictions are outside [0, 1]
 * 
 * @example
 * ```cpp
 * Tensor predictions = {0.1, 0.7, 0.9, 0.3};  // Predicted probabilities
 * Tensor targets = {0.0, 1.0, 1.0, 0.0};      // True labels
 * double loss = binary_cross_entropy_loss(predictions, targets);  // ≈ 0.693
 * 
 * // In training loop
 * Tensor logits = model.forward(input);
 * Tensor probs = sigmoid(logits);
 * double loss = binary_cross_entropy_loss(probs, targets);
 * loss.backward();
 * ```
 * 
 * @see cross_entropy_loss, categorical_cross_entropy_loss
 * @since 1.0.0
 */
Tensor binary_cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
```

### `categorical_cross_entropy_loss`

```cpp
/**
 * @brief Categorical cross-entropy loss function
 * 
 * @details This function computes the categorical cross-entropy loss between
 * predictions and targets. The operation is mathematically defined as:
 * 
 *     CCE = -Σ(i=0 to n-1) targets[i] * log(predictions[i])
 * 
 * Categorical cross-entropy loss is used for multi-class classification
 * problems where targets are one-hot encoded vectors.
 * 
 * @param predictions Predicted probabilities (must sum to 1 along last axis)
 * @param targets Target one-hot vectors (must sum to 1 along last axis)
 * @return Scalar categorical cross-entropy loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * @throws std::domain_error if predictions contain non-positive values
 * 
 * @example
 * ```cpp
 * Tensor predictions = {{0.1, 0.3, 0.6}, {0.2, 0.5, 0.3}};  // (2, 3)
 * Tensor targets = {{0, 0, 1}, {0, 1, 0}};                  // (2, 3)
 * double loss = categorical_cross_entropy_loss(predictions, targets);
 * 
 * // In training loop
 * Tensor logits = model.forward(input);
 * Tensor probs = softmax(logits);
 * double loss = categorical_cross_entropy_loss(probs, targets);
 * loss.backward();
 * ```
 * 
 * @see cross_entropy_loss, binary_cross_entropy_loss
 * @since 1.0.0
 */
Tensor categorical_cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
```

### `sparse_categorical_cross_entropy_loss`

```cpp
/**
 * @brief Sparse categorical cross-entropy loss function
 * 
 * @details This function computes the sparse categorical cross-entropy loss
 * between predictions and targets. The operation is mathematically defined as:
 * 
 *     SparseCCE = -Σ(i=0 to n-1) log(predictions[i, targets[i]])
 * 
 * Sparse categorical cross-entropy loss is used for multi-class classification
 * problems where targets are integer class indices rather than one-hot vectors.
 * 
 * @param predictions Predicted probabilities (must sum to 1 along last axis)
 * @param targets Target class indices (must be in range [0, num_classes))
 * @return Scalar sparse categorical cross-entropy loss value
 * 
 * @throws ShapeError if predictions and targets have incompatible shapes
 * @throws std::domain_error if predictions contain non-positive values
 * 
 * @example
 * ```cpp
 * Tensor predictions = {{0.1, 0.3, 0.6}, {0.2, 0.5, 0.3}};  // (2, 3)
 * Tensor targets = {2, 1};                                   // (2,) - class indices
 * double loss = sparse_categorical_cross_entropy_loss(predictions, targets);
 * 
 * // In training loop
 * Tensor logits = model.forward(input);
 * Tensor probs = softmax(logits);
 * double loss = sparse_categorical_cross_entropy_loss(probs, targets);
 * loss.backward();
 * ```
 * 
 * @see cross_entropy_loss, categorical_cross_entropy_loss
 * @since 1.0.0
 */
Tensor sparse_categorical_cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
```

## Specialized Loss Functions

### `hinge_loss`

```cpp
/**
 * @brief Hinge loss function
 * 
 * @details This function computes the hinge loss between predictions and targets.
 * The operation is mathematically defined as:
 * 
 *     Hinge = max(0, 1 - predictions * targets)
 * 
 * Hinge loss is commonly used in support vector machines and is less sensitive
 * to outliers than cross-entropy loss.
 * 
 * @param predictions Predicted values
 * @param targets Target values (must be ±1)
 * @return Scalar hinge loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * 
 * @example
 * ```cpp
 * Tensor predictions = {0.5, 1.5, -0.5, -1.5};  // Predicted values
 * Tensor targets = {1, 1, -1, -1};               // True labels (±1)
 * double loss = hinge_loss(predictions, targets);  // ≈ 0.5
 * 
 * // In SVM training
 * Tensor logits = model.forward(input);
 * double loss = hinge_loss(logits, targets);
 * loss.backward();
 * ```
 * 
 * @see squared_hinge_loss, cross_entropy_loss
 * @since 1.0.0
 */
Tensor hinge_loss(const Tensor& predictions, const Tensor& targets);
```

### `squared_hinge_loss`

```cpp
/**
 * @brief Squared hinge loss function
 * 
 * @details This function computes the squared hinge loss between predictions
 * and targets. The operation is mathematically defined as:
 * 
 *     SquaredHinge = max(0, 1 - predictions * targets)²
 * 
 * Squared hinge loss is smoother than hinge loss and is commonly used in
 * support vector machines for better convergence.
 * 
 * @param predictions Predicted values
 * @param targets Target values (must be ±1)
 * @return Scalar squared hinge loss value
 * 
 * @throws ShapeError if predictions and targets have different shapes
 * 
 * @example
 * ```cpp
 * Tensor predictions = {0.5, 1.5, -0.5, -1.5};  // Predicted values
 * Tensor targets = {1, 1, -1, -1};               // True labels (±1)
 * double loss = squared_hinge_loss(predictions, targets);  // ≈ 0.25
 * 
 * // In SVM training
 * Tensor logits = model.forward(input);
 * double loss = squared_hinge_loss(logits, targets);
 * loss.backward();
 * ```
 * 
 * @see hinge_loss, cross_entropy_loss
 * @since 1.0.0
 */
Tensor squared_hinge_loss(const Tensor& predictions, const Tensor& targets);
```

### `kl_divergence_loss`

```cpp
/**
 * @brief Kullback-Leibler divergence loss function
 * 
 * @details This function computes the KL divergence between two probability
 * distributions. The operation is mathematically defined as:
 * 
 *     KL(P||Q) = Σ(i=0 to n-1) P[i] * log(P[i] / Q[i])
 * 
 * KL divergence measures how much one probability distribution differs from
 * another. It's commonly used in variational autoencoders and knowledge distillation.
 * 
 * @param p First probability distribution
 * @param q Second probability distribution
 * @return Scalar KL divergence value
 * 
 * @throws ShapeError if p and q have different shapes
 * @throws std::domain_error if p or q contain non-positive values
 * 
 * @example
 * ```cpp
 * Tensor p = {0.1, 0.3, 0.6};  // True distribution
 * Tensor q = {0.2, 0.3, 0.5};  // Approximate distribution
 * double loss = kl_divergence_loss(p, q);  // ≈ 0.085
 * 
 * // In VAE training
 * Tensor true_dist = softmax(true_logits);
 * Tensor approx_dist = softmax(approx_logits);
 * double loss = kl_divergence_loss(true_dist, approx_dist);
 * ```
 * 
 * @see cross_entropy_loss, js_divergence_loss
 * @since 1.0.0
 */
Tensor kl_divergence_loss(const Tensor& p, const Tensor& q);
```

### `js_divergence_loss`

```cpp
/**
 * @brief Jensen-Shannon divergence loss function
 * 
 * @details This function computes the Jensen-Shannon divergence between two
 * probability distributions. The operation is mathematically defined as:
 * 
 *     JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
 * 
 * where M = 0.5 * (P + Q). JS divergence is symmetric and bounded in [0, 1],
 * making it more suitable for some applications than KL divergence.
 * 
 * @param p First probability distribution
 * @param q Second probability distribution
 * @return Scalar JS divergence value
 * 
 * @throws ShapeError if p and q have different shapes
 * @throws std::domain_error if p or q contain non-positive values
 * 
 * @example
 * ```cpp
 * Tensor p = {0.1, 0.3, 0.6};  // First distribution
 * Tensor q = {0.2, 0.3, 0.5};  // Second distribution
 * double loss = js_divergence_loss(p, q);  // ≈ 0.043
 * 
 * // In GAN training
 * Tensor real_dist = softmax(real_logits);
 * Tensor fake_dist = softmax(fake_logits);
 * double loss = js_divergence_loss(real_dist, fake_dist);
 * ```
 * 
 * @see kl_divergence_loss, cross_entropy_loss
 * @since 1.0.0
 */
Tensor js_divergence_loss(const Tensor& p, const Tensor& q);
```

## Loss Function Properties

### Mathematical Properties

| Function | Range | Smooth | Convex | Robust to Outliers |
|----------|-------|--------|--------|-------------------|
| MSE | [0, ∞) | Yes | Yes | No |
| MAE | [0, ∞) | No | Yes | Yes |
| Huber | [0, ∞) | Yes | Yes | Yes |
| Smooth L1 | [0, ∞) | Yes | Yes | Yes |
| Cross-entropy | [0, ∞) | Yes | Yes | No |
| Binary CE | [0, ∞) | Yes | Yes | No |
| Hinge | [0, ∞) | No | Yes | Yes |
| Squared Hinge | [0, ∞) | Yes | Yes | Yes |
| KL Divergence | [0, ∞) | Yes | No | No |
| JS Divergence | [0, 1] | Yes | No | No |

### Gradient Properties

```cpp
// MSE gradient
Tensor predictions = {1.0, 2.0, 3.0};
Tensor targets = {1.1, 1.9, 3.1};
Tensor grad = 2 * (predictions - targets);  // MSE gradient

// Cross-entropy gradient
Tensor probs = {0.1, 0.3, 0.6};
Tensor targets = {0.0, 0.0, 1.0};
Tensor grad = probs - targets;  // Cross-entropy gradient

// Hinge gradient
Tensor predictions = {0.5, 1.5, -0.5};
Tensor targets = {1, 1, -1};
Tensor grad = where(predictions * targets < 1, -targets, 0);  // Hinge gradient
```

### Numerical Stability

```cpp
// Cross-entropy with numerical stability
Tensor logits = {1000, 1001, 1002};  // Large values
Tensor log_probs = log_softmax(logits);  // Numerically stable
Tensor loss = -(log_probs * targets).sum();

// Binary cross-entropy with numerical stability
Tensor logits = {1000, -1000};  // Large values
Tensor probs = sigmoid(logits);  // Numerically stable
Tensor loss = binary_cross_entropy_loss(probs, targets);
```

## Performance Considerations

### Computational Complexity

| Function | Complexity | Notes |
|----------|------------|-------|
| MSE | O(n) | Simple arithmetic |
| MAE | O(n) | Simple arithmetic |
| Huber | O(n) | Conditional operations |
| Smooth L1 | O(n) | Conditional operations |
| Cross-entropy | O(n) | Requires log() |
| Binary CE | O(n) | Requires log() |
| Hinge | O(n) | Simple arithmetic |
| Squared Hinge | O(n) | Simple arithmetic |
| KL Divergence | O(n) | Requires log() |
| JS Divergence | O(n) | Requires log() |

### Memory Usage

- **Temporary Storage**: Some functions require temporary storage for intermediate calculations
- **Gradient Storage**: Gradient computation may require additional memory
- **Numerical Stability**: Some functions use additional memory for numerical stability

### Best Practices

```cpp
// Good: Use appropriate loss function for the task
Tensor predictions = model.forward(input);

// For regression
double loss = mse_loss(predictions, targets);  // or mae_loss, huber_loss

// For binary classification
double loss = binary_cross_entropy_loss(predictions, targets);

// For multi-class classification
double loss = cross_entropy_loss(predictions, targets);

// Good: Use numerical stability when needed
Tensor logits = model.forward(input);
Tensor log_probs = log_softmax(logits);
double loss = -(log_probs * targets).sum();  // More stable than cross_entropy_loss

// Good: Use reduction when appropriate
Tensor batch_losses = mse_loss(predictions, targets, reduction="none");
double loss = batch_losses.mean();  // Mean over batch
```

## Common Patterns

### Training Loop

```cpp
// Standard training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        Tensor predictions = model.forward(batch.input);
        double loss = loss_function(predictions, batch.targets);
        
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}

// Multi-task learning
Tensor task1_pred = model.forward(batch.input);
Tensor task2_pred = model.forward(batch.input);

double loss1 = mse_loss(task1_pred, batch.targets1);
double loss2 = cross_entropy_loss(task2_pred, batch.targets2);
double total_loss = 0.5 * loss1 + 0.5 * loss2;

total_loss.backward();
```

### Loss Function Selection Guide

```cpp
// For regression problems
Tensor predictions = model.forward(input);

// Use MSE for smooth, continuous targets
double loss = mse_loss(predictions, targets);

// Use MAE for robust regression
double loss = mae_loss(predictions, targets);

// Use Huber for balanced approach
double loss = huber_loss(predictions, targets, 1.0);

// For classification problems
Tensor logits = model.forward(input);

// Use cross-entropy for multi-class
Tensor probs = softmax(logits);
double loss = cross_entropy_loss(probs, targets);

// Use binary cross-entropy for binary classification
Tensor probs = sigmoid(logits);
double loss = binary_cross_entropy_loss(probs, targets);

// Use hinge loss for SVM
double loss = hinge_loss(logits, targets);
```

### Custom Loss Functions

```cpp
// Custom weighted loss
Tensor predictions = model.forward(input);
Tensor weights = {1.0, 2.0, 0.5, 1.5};  // Sample weights
Tensor weighted_loss = weights * (predictions - targets).square();
double loss = weighted_loss.mean();

// Custom focal loss for imbalanced classification
Tensor probs = sigmoid(logits);
Tensor alpha = 0.25;
Tensor gamma = 2.0;
Tensor focal_loss = -alpha * (1 - probs).pow(gamma) * targets * log(probs) -
                    (1 - alpha) * probs.pow(gamma) * (1 - targets) * log(1 - probs);
double loss = focal_loss.mean();
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's loss functions for their machine learning projects.
