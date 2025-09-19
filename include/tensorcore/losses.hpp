#pragma once

#include "tensor.hpp"

namespace tensorcore {

/**
 * @brief Loss functions for machine learning
 * 
 * This module provides various loss functions commonly used in machine learning,
 * including regression, classification, and specialized losses.
 */

// Regression losses
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);
Tensor mae_loss(const Tensor& predictions, const Tensor& targets);
Tensor huber_loss(const Tensor& predictions, const Tensor& targets, double delta = 1.0);
Tensor smooth_l1_loss(const Tensor& predictions, const Tensor& targets, double beta = 1.0);
Tensor poisson_loss(const Tensor& predictions, const Tensor& targets);
Tensor cosine_similarity_loss(const Tensor& predictions, const Tensor& targets);

// Classification losses
Tensor cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
Tensor binary_cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
Tensor categorical_cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
Tensor sparse_categorical_cross_entropy_loss(const Tensor& predictions, const Tensor& targets);
Tensor focal_loss(const Tensor& predictions, const Tensor& targets, double alpha = 1.0, double gamma = 2.0);
Tensor dice_loss(const Tensor& predictions, const Tensor& targets, double smooth = 1.0);

// Hinge losses
Tensor hinge_loss(const Tensor& predictions, const Tensor& targets, double margin = 1.0);
Tensor squared_hinge_loss(const Tensor& predictions, const Tensor& targets, double margin = 1.0);
Tensor categorical_hinge_loss(const Tensor& predictions, const Tensor& targets);

// KL divergence losses
Tensor kl_divergence_loss(const Tensor& predictions, const Tensor& targets);
Tensor js_divergence_loss(const Tensor& predictions, const Tensor& targets);

// Wasserstein losses
Tensor wasserstein_loss(const Tensor& predictions, const Tensor& targets);
Tensor earth_mover_distance_loss(const Tensor& predictions, const Tensor& targets);

// Gradient functions (for backpropagation)
Tensor mse_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor mae_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor huber_loss_grad(const Tensor& predictions, const Tensor& targets, double delta = 1.0);
Tensor smooth_l1_loss_grad(const Tensor& predictions, const Tensor& targets, double beta = 1.0);
Tensor poisson_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor cosine_similarity_loss_grad(const Tensor& predictions, const Tensor& targets);

Tensor cross_entropy_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor binary_cross_entropy_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor categorical_cross_entropy_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor sparse_categorical_cross_entropy_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor focal_loss_grad(const Tensor& predictions, const Tensor& targets, double alpha = 1.0, double gamma = 2.0);
Tensor dice_loss_grad(const Tensor& predictions, const Tensor& targets, double smooth = 1.0);

Tensor hinge_loss_grad(const Tensor& predictions, const Tensor& targets, double margin = 1.0);
Tensor squared_hinge_loss_grad(const Tensor& predictions, const Tensor& targets, double margin = 1.0);
Tensor categorical_hinge_loss_grad(const Tensor& predictions, const Tensor& targets);

Tensor kl_divergence_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor js_divergence_loss_grad(const Tensor& predictions, const Tensor& targets);

Tensor wasserstein_loss_grad(const Tensor& predictions, const Tensor& targets);
Tensor earth_mover_distance_loss_grad(const Tensor& predictions, const Tensor& targets);

// Loss function class for easy switching
class LossFunction {
public:
    using forward_func = std::function<Tensor(const Tensor&, const Tensor&)>;
    using backward_func = std::function<Tensor(const Tensor&, const Tensor&)>;
    
    LossFunction(forward_func forward, backward_func backward)
        : forward_(forward), backward_(backward) {}
    
    Tensor forward(const Tensor& predictions, const Tensor& targets) const { 
        return forward_(predictions, targets); 
    }
    Tensor backward(const Tensor& predictions, const Tensor& targets) const { 
        return backward_(predictions, targets); 
    }
    
private:
    forward_func forward_;
    backward_func backward_;
};

// Predefined loss functions
extern const LossFunction MSELoss;
extern const LossFunction MAELoss;
extern const LossFunction HuberLoss;
extern const LossFunction SmoothL1Loss;
extern const LossFunction PoissonLoss;
extern const LossFunction CosineSimilarityLoss;
extern const LossFunction CrossEntropyLoss;
extern const LossFunction BinaryCrossEntropyLoss;
extern const LossFunction CategoricalCrossEntropyLoss;
extern const LossFunction SparseCategoricalCrossEntropyLoss;
extern const LossFunction FocalLoss;
extern const LossFunction DiceLoss;
extern const LossFunction HingeLoss;
extern const LossFunction SquaredHingeLoss;
extern const LossFunction CategoricalHingeLoss;
extern const LossFunction KLDivergenceLoss;
extern const LossFunction JSDivergenceLoss;
extern const LossFunction WassersteinLoss;
extern const LossFunction EarthMoverDistanceLoss;

// Utility functions
std::string get_loss_name(const LossFunction& loss);
LossFunction get_loss_by_name(const std::string& name);
std::vector<std::string> get_available_losses();

// Loss reduction modes
enum class Reduction {
    NONE,    // Return loss for each sample
    MEAN,    // Return mean of losses
    SUM      // Return sum of losses
};

// Loss function with reduction
Tensor compute_loss(const LossFunction& loss, const Tensor& predictions, 
                   const Tensor& targets, Reduction reduction = Reduction::MEAN);

// Multi-task loss
Tensor multi_task_loss(const std::vector<std::pair<LossFunction, double>>& losses,
                      const std::vector<Tensor>& predictions,
                      const std::vector<Tensor>& targets);

// Regularization losses
Tensor l1_regularization(const Tensor& weights, double lambda = 0.01);
Tensor l2_regularization(const Tensor& weights, double lambda = 0.01);
Tensor elastic_net_regularization(const Tensor& weights, double l1_lambda = 0.01, double l2_lambda = 0.01);

// Gradient penalties
Tensor gradient_penalty(const Tensor& predictions, const Tensor& targets, double lambda = 10.0);
Tensor spectral_norm_penalty(const Tensor& weights, double lambda = 0.01);

} // namespace tensorcore
