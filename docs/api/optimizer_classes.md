# Optimizer Classes

This document provides comprehensive documentation for all optimizer classes in TensorCore.

## Table of Contents

1. [Basic Optimizers](#basic-optimizers)
2. [Advanced Optimizers](#advanced-optimizers)
3. [Adaptive Optimizers](#adaptive-optimizers)
4. [Optimizer Properties](#optimizer-properties)
5. [Performance Considerations](#performance-considerations)

## Basic Optimizers

### `SGD`

```cpp
/**
 * @brief Stochastic Gradient Descent optimizer
 * 
 * @details This class implements the standard SGD optimizer with optional
 * momentum and weight decay. The update rule is:
 * 
 *     v_t = μ * v_{t-1} + g_t
 *     θ_t = θ_{t-1} - lr * v_t
 * 
 * where μ is the momentum coefficient, g_t is the gradient at time t,
 * and lr is the learning rate.
 * 
 * @param lr Learning rate (default: 0.01)
 * @param momentum Momentum coefficient (default: 0.0)
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * @param dampening Dampening for momentum (default: 0.0)
 * @param nesterov Whether to use Nesterov momentum (default: false)
 * 
 * @example
 * ```cpp
 * // Basic SGD
 * SGD optimizer(0.01);
 * 
 * // SGD with momentum
 * SGD optimizer(0.01, 0.9);
 * 
 * // SGD with weight decay
 * SGD optimizer(0.01, 0.0, 1e-4);
 * 
 * // Nesterov momentum
 * SGD optimizer(0.01, 0.9, 0.0, 0.0, true);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see Adam, RMSprop, Adagrad
 * @since 1.0.0
 */
class SGD {
public:
    SGD(double lr = 0.01, double momentum = 0.0, double weight_decay = 0.0, 
        double dampening = 0.0, bool nesterov = false);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

### `Adam`

```cpp
/**
 * @brief Adam optimizer
 * 
 * @details This class implements the Adam optimizer, which combines the
 * benefits of AdaGrad and RMSProp. The update rule is:
 * 
 *     m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 *     v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 *     m̂_t = m_t / (1 - β₁ᵗ)
 *     v̂_t = v_t / (1 - β₂ᵗ)
 *     θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
 * 
 * where β₁ and β₂ are exponential decay rates, and ε is a small constant.
 * 
 * @param lr Learning rate (default: 0.001)
 * @param betas Tuple of (β₁, β₂) (default: (0.9, 0.999))
 * @param eps Small constant for numerical stability (default: 1e-8)
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * @param amsgrad Whether to use AMSGrad variant (default: false)
 * 
 * @example
 * ```cpp
 * // Basic Adam
 * Adam optimizer(0.001);
 * 
 * // Adam with custom parameters
 * Adam optimizer(0.001, {0.9, 0.999}, 1e-8, 1e-4);
 * 
 * // AMSGrad variant
 * Adam optimizer(0.001, {0.9, 0.999}, 1e-8, 0.0, true);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see SGD, RMSprop, Adagrad
 * @since 1.0.0
 */
class Adam {
public:
    Adam(double lr = 0.001, std::pair<double, double> betas = {0.9, 0.999}, 
         double eps = 1e-8, double weight_decay = 0.0, bool amsgrad = false);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

### `RMSprop`

```cpp
/**
 * @brief RMSprop optimizer
 * 
 * @details This class implements the RMSprop optimizer, which uses a moving
 * average of squared gradients to normalize the learning rate. The update rule is:
 * 
 *     v_t = α * v_{t-1} + (1 - α) * g_t²
 *     θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
 * 
 * where α is the decay rate and ε is a small constant.
 * 
 * @param lr Learning rate (default: 0.01)
 * @param alpha Decay rate for moving average (default: 0.99)
 * @param eps Small constant for numerical stability (default: 1e-8)
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * @param momentum Momentum coefficient (default: 0.0)
 * @param centered Whether to use centered RMSprop (default: false)
 * 
 * @example
 * ```cpp
 * // Basic RMSprop
 * RMSprop optimizer(0.01);
 * 
 * // RMSprop with momentum
 * RMSprop optimizer(0.01, 0.99, 1e-8, 0.0, 0.9);
 * 
 * // Centered RMSprop
 * RMSprop optimizer(0.01, 0.99, 1e-8, 0.0, 0.0, true);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see SGD, Adam, Adagrad
 * @since 1.0.0
 */
class RMSprop {
public:
    RMSprop(double lr = 0.01, double alpha = 0.99, double eps = 1e-8, 
            double weight_decay = 0.0, double momentum = 0.0, bool centered = false);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

### `Adagrad`

```cpp
/**
 * @brief Adagrad optimizer
 * 
 * @details This class implements the Adagrad optimizer, which adapts the
 * learning rate based on the historical gradient information. The update rule is:
 * 
 *     v_t = v_{t-1} + g_t²
 *     θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
 * 
 * where ε is a small constant. Adagrad works well for sparse gradients
 * but may have diminishing learning rates.
 * 
 * @param lr Learning rate (default: 0.01)
 * @param eps Small constant for numerical stability (default: 1e-10)
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * @param lr_decay Learning rate decay (default: 0.0)
 * 
 * @example
 * ```cpp
 * // Basic Adagrad
 * Adagrad optimizer(0.01);
 * 
 * // Adagrad with weight decay
 * Adagrad optimizer(0.01, 1e-10, 1e-4);
 * 
 * // Adagrad with learning rate decay
 * Adagrad optimizer(0.01, 1e-10, 0.0, 1e-6);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see SGD, Adam, RMSprop
 * @since 1.0.0
 */
class Adagrad {
public:
    Adagrad(double lr = 0.01, double eps = 1e-10, double weight_decay = 0.0, 
            double lr_decay = 0.0);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

## Advanced Optimizers

### `AdamW`

```cpp
/**
 * @brief AdamW optimizer
 * 
 * @details This class implements the AdamW optimizer, which is a variant of
 * Adam that decouples weight decay from gradient updates. The update rule is:
 * 
 *     m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 *     v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 *     m̂_t = m_t / (1 - β₁ᵗ)
 *     v̂_t = v_t / (1 - β₂ᵗ)
 *     θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
 * 
 * where λ is the weight decay coefficient. AdamW often performs better
 * than Adam for training deep neural networks.
 * 
 * @param lr Learning rate (default: 0.001)
 * @param betas Tuple of (β₁, β₂) (default: (0.9, 0.999))
 * @param eps Small constant for numerical stability (default: 1e-8)
 * @param weight_decay Weight decay coefficient (default: 0.01)
 * @param amsgrad Whether to use AMSGrad variant (default: false)
 * 
 * @example
 * ```cpp
 * // Basic AdamW
 * AdamW optimizer(0.001);
 * 
 * // AdamW with custom weight decay
 * AdamW optimizer(0.001, {0.9, 0.999}, 1e-8, 0.01);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see Adam, SGD, RMSprop
 * @since 1.0.0
 */
class AdamW {
public:
    AdamW(double lr = 0.001, std::pair<double, double> betas = {0.9, 0.999}, 
          double eps = 1e-8, double weight_decay = 0.01, bool amsgrad = false);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

### `AdaDelta`

```cpp
/**
 * @brief AdaDelta optimizer
 * 
 * @details This class implements the AdaDelta optimizer, which is an extension
 * of Adagrad that addresses the diminishing learning rate problem. The update rule is:
 * 
 *     v_t = ρ * v_{t-1} + (1 - ρ) * g_t²
 *     Δθ_t = -√(E[Δθ²]_{t-1} + ε) / √(v_t + ε) * g_t
 *     E[Δθ²]_t = ρ * E[Δθ²]_{t-1} + (1 - ρ) * Δθ_t²
 *     θ_t = θ_{t-1} + Δθ_t
 * 
 * where ρ is the decay rate and ε is a small constant.
 * 
 * @param lr Learning rate (default: 1.0)
 * @param rho Decay rate for moving averages (default: 0.9)
 * @param eps Small constant for numerical stability (default: 1e-6)
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * 
 * @example
 * ```cpp
 * // Basic AdaDelta
 * AdaDelta optimizer(1.0);
 * 
 * // AdaDelta with custom parameters
 * AdaDelta optimizer(1.0, 0.9, 1e-6, 1e-4);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see Adam, RMSprop, Adagrad
 * @since 1.0.0
 */
class AdaDelta {
public:
    AdaDelta(double lr = 1.0, double rho = 0.9, double eps = 1e-6, 
             double weight_decay = 0.0);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

### `AdaMax`

```cpp
/**
 * @brief AdaMax optimizer
 * 
 * @details This class implements the AdaMax optimizer, which is a variant of
 * Adam that uses the infinity norm instead of the L2 norm. The update rule is:
 * 
 *     m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 *     u_t = max(β₂ * u_{t-1}, |g_t|)
 *     m̂_t = m_t / (1 - β₁ᵗ)
 *     θ_t = θ_{t-1} - lr * m̂_t / u_t
 * 
 * where β₁ and β₂ are exponential decay rates. AdaMax can be more stable
 * than Adam in some cases.
 * 
 * @param lr Learning rate (default: 0.002)
 * @param betas Tuple of (β₁, β₂) (default: (0.9, 0.999))
 * @param eps Small constant for numerical stability (default: 1e-8)
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * 
 * @example
 * ```cpp
 * // Basic AdaMax
 * AdaMax optimizer(0.002);
 * 
 * // AdaMax with custom parameters
 * AdaMax optimizer(0.002, {0.9, 0.999}, 1e-8, 1e-4);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see Adam, AdamW, RMSprop
 * @since 1.0.0
 */
class AdaMax {
public:
    AdaMax(double lr = 0.002, std::pair<double, double> betas = {0.9, 0.999}, 
           double eps = 1e-8, double weight_decay = 0.0);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

## Adaptive Optimizers

### `RAdam`

```cpp
/**
 * @brief RAdam optimizer
 * 
 * @details This class implements the RAdam optimizer, which is a variant of
 * Adam that uses a rectified variance estimate. RAdam addresses the problem
 * of high variance in the early stages of training by using a warmup period.
 * 
 * @param lr Learning rate (default: 0.001)
 * @param betas Tuple of (β₁, β₂) (default: (0.9, 0.999))
 * @param eps Small constant for numerical stability (default: 1e-8)
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * 
 * @example
 * ```cpp
 * // Basic RAdam
 * RAdam optimizer(0.001);
 * 
 * // RAdam with custom parameters
 * RAdam optimizer(0.001, {0.9, 0.999}, 1e-8, 1e-4);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see Adam, AdamW, AdaMax
 * @since 1.0.0
 */
class RAdam {
public:
    RAdam(double lr = 0.001, std::pair<double, double> betas = {0.9, 0.999}, 
          double eps = 1e-8, double weight_decay = 0.0);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

### `Lion`

```cpp
/**
 * @brief Lion optimizer
 * 
 * @details This class implements the Lion optimizer, which is a memory-efficient
 * optimizer that uses only momentum and sign-based updates. Lion is designed
 * to be simple, memory-efficient, and effective for large-scale training.
 * 
 * @param lr Learning rate (default: 0.0001)
 * @param betas Tuple of (β₁, β₂) (default: (0.9, 0.99))
 * @param weight_decay Weight decay coefficient (default: 0.0)
 * 
 * @example
 * ```cpp
 * // Basic Lion
 * Lion optimizer(0.0001);
 * 
 * // Lion with custom parameters
 * Lion optimizer(0.0001, {0.9, 0.99}, 1e-4);
 * 
 * // Training loop
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         Tensor loss = model.forward(batch.input, batch.targets);
 *         loss.backward();
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * ```
 * 
 * @see Adam, AdamW, RAdam
 * @since 1.0.0
 */
class Lion {
public:
    Lion(double lr = 0.0001, std::pair<double, double> betas = {0.9, 0.99}, 
         double weight_decay = 0.0);
    
    void step();
    void zero_grad();
    void set_lr(double lr);
    double get_lr() const;
};
```

## Optimizer Properties

### Mathematical Properties

| Optimizer | Memory | Convergence | Robustness | Hyperparameters |
|-----------|--------|-------------|------------|-----------------|
| SGD | Low | Slow | High | lr, momentum |
| Adam | High | Fast | Medium | lr, β₁, β₂, ε |
| RMSprop | Medium | Fast | Medium | lr, α, ε |
| Adagrad | High | Medium | Low | lr, ε |
| AdamW | High | Fast | High | lr, β₁, β₂, ε, λ |
| AdaDelta | Medium | Medium | Medium | lr, ρ, ε |
| AdaMax | High | Fast | Medium | lr, β₁, β₂, ε |
| RAdam | High | Fast | High | lr, β₁, β₂, ε |
| Lion | Low | Fast | High | lr, β₁, β₂ |

### Learning Rate Schedules

```cpp
// Step decay
class StepLR {
public:
    StepLR(Optimizer& optimizer, int step_size, double gamma = 0.1);
    void step();
};

// Exponential decay
class ExponentialLR {
public:
    ExponentialLR(Optimizer& optimizer, double gamma);
    void step();
};

// Cosine annealing
class CosineAnnealingLR {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, double eta_min = 0.0);
    void step();
};

// Usage
Adam optimizer(0.001);
StepLR scheduler(optimizer, 30, 0.1);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        Tensor loss = model.forward(batch.input, batch.targets);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
    scheduler.step();  // Update learning rate
}
```

### Gradient Clipping

```cpp
// Gradient clipping
void clip_grad_norm(std::vector<Tensor>& parameters, double max_norm) {
    double total_norm = 0.0;
    for (auto& param : parameters) {
        total_norm += param.grad().square().sum();
    }
    total_norm = sqrt(total_norm);
    
    if (total_norm > max_norm) {
        double clip_coef = max_norm / (total_norm + 1e-6);
        for (auto& param : parameters) {
            param.grad() *= clip_coef;
        }
    }
}

// Usage in training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        Tensor loss = model.forward(batch.input, batch.targets);
        loss.backward();
        
        // Clip gradients
        clip_grad_norm(model.parameters(), 1.0);
        
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

## Performance Considerations

### Computational Complexity

| Optimizer | Per-parameter | Per-iteration | Memory |
|-----------|---------------|---------------|--------|
| SGD | O(1) | O(n) | O(n) |
| Adam | O(1) | O(n) | O(2n) |
| RMSprop | O(1) | O(n) | O(n) |
| Adagrad | O(1) | O(n) | O(n) |
| AdamW | O(1) | O(n) | O(2n) |
| AdaDelta | O(1) | O(n) | O(2n) |
| AdaMax | O(1) | O(n) | O(2n) |
| RAdam | O(1) | O(n) | O(2n) |
| Lion | O(1) | O(n) | O(n) |

### Memory Usage

- **SGD**: Minimal memory usage, only stores momentum
- **Adam**: High memory usage, stores first and second moments
- **RMSprop**: Medium memory usage, stores squared gradients
- **Adagrad**: High memory usage, accumulates squared gradients
- **Lion**: Low memory usage, only stores momentum

### Best Practices

```cpp
// Good: Choose appropriate optimizer for the task
// For computer vision
Adam optimizer(0.001);

// For natural language processing
AdamW optimizer(0.001, {0.9, 0.999}, 1e-8, 0.01);

// For large-scale training
Lion optimizer(0.0001, {0.9, 0.99}, 1e-4);

// Good: Use learning rate scheduling
Adam optimizer(0.001);
StepLR scheduler(optimizer, 30, 0.1);

// Good: Use gradient clipping for stability
clip_grad_norm(model.parameters(), 1.0);

// Good: Use appropriate weight decay
AdamW optimizer(0.001, {0.9, 0.999}, 1e-8, 0.01);  // Decoupled weight decay
```

## Common Patterns

### Training Loop

```cpp
// Standard training loop
void train_model(Model& model, DataLoader& dataloader, Optimizer& optimizer) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model.train();
        double epoch_loss = 0.0;
        
        for (auto& batch : dataloader) {
            Tensor predictions = model.forward(batch.input);
            Tensor loss = loss_function(predictions, batch.targets);
            
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
            
            epoch_loss += loss.item();
        }
        
        std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss / dataloader.size() << std::endl;
    }
}
```

### Multi-task Learning

```cpp
// Multi-task learning with different optimizers
Adam task1_optimizer(0.001);
AdamW task2_optimizer(0.001, {0.9, 0.999}, 1e-8, 0.01);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        // Task 1
        Tensor task1_pred = model.task1_forward(batch.input);
        Tensor task1_loss = mse_loss(task1_pred, batch.task1_targets);
        task1_loss.backward();
        task1_optimizer.step();
        task1_optimizer.zero_grad();
        
        // Task 2
        Tensor task2_pred = model.task2_forward(batch.input);
        Tensor task2_loss = cross_entropy_loss(task2_pred, batch.task2_targets);
        task2_loss.backward();
        task2_optimizer.step();
        task2_optimizer.zero_grad();
    }
}
```

### Optimizer Selection Guide

```cpp
// For computer vision (CNNs)
Adam optimizer(0.001);

// For natural language processing (Transformers)
AdamW optimizer(0.001, {0.9, 0.999}, 1e-8, 0.01);

// For recurrent neural networks
RMSprop optimizer(0.01, 0.99, 1e-8, 0.0, 0.9);

// For sparse data
Adagrad optimizer(0.01);

// For large-scale training
Lion optimizer(0.0001, {0.9, 0.99}, 1e-4);

// For fine-tuning
AdamW optimizer(0.0001, {0.9, 0.999}, 1e-8, 0.01);
```

### Custom Optimizer

```cpp
// Custom optimizer example
class CustomOptimizer {
public:
    CustomOptimizer(double lr, double momentum = 0.9) : lr_(lr), momentum_(momentum) {}
    
    void step() {
        for (auto& param : parameters_) {
            if (param.requires_grad()) {
                Tensor& grad = param.grad();
                Tensor& momentum = momentum_[param];
                
                momentum = momentum_ * momentum + (1 - momentum_) * grad;
                param.data() -= lr_ * momentum;
            }
        }
    }
    
    void zero_grad() {
        for (auto& param : parameters_) {
            if (param.requires_grad()) {
                param.grad().zero_();
            }
        }
    }
    
private:
    double lr_;
    double momentum_;
    std::vector<Tensor> parameters_;
    std::unordered_map<Tensor*, Tensor> momentum_;
};
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's optimizer classes for their machine learning projects.
