#include "tensorcore/losses.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace tensorcore {

// Regression loss functions
Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = predictions[i] - targets[i];
        sum_squared_error += error * error;
    }
    
    return Tensor({1}, sum_squared_error / static_cast<double>(predictions.size()));
}

Tensor mae_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    double sum_absolute_error = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        sum_absolute_error += std::abs(predictions[i] - targets[i]);
    }
    
    return Tensor({1}, sum_absolute_error / static_cast<double>(predictions.size()));
}

Tensor huber_loss(const Tensor& predictions, const Tensor& targets, double delta) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    if (delta <= 0.0) {
        throw std::invalid_argument("Delta must be positive");
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = std::abs(predictions[i] - targets[i]);
        if (error <= delta) {
            sum_loss += 0.5 * error * error;
        } else {
            sum_loss += delta * (error - 0.5 * delta);
        }
    }
    
    return Tensor({1}, sum_loss / static_cast<double>(predictions.size()));
}

Tensor smooth_l1_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = std::abs(predictions[i] - targets[i]);
        if (error < 1.0) {
            sum_loss += 0.5 * error * error;
        } else {
            sum_loss += error - 0.5;
        }
    }
    
    return Tensor({1}, sum_loss / static_cast<double>(predictions.size()));
}

Tensor smooth_l1_loss(const Tensor& predictions, const Tensor& targets, double beta) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double error = std::abs(predictions[i] - targets[i]);
        if (error < beta) {
            sum_loss += 0.5 * error * error / beta;
        } else {
            sum_loss += error - 0.5 * beta;
        }
    }
    
    return Tensor({1}, sum_loss / static_cast<double>(predictions.size()));
}

// Classification loss functions
Tensor cross_entropy_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that predictions are valid probabilities
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] <= 0.0 || predictions[i] > 1.0) {
            throw std::domain_error("Predictions must be valid probabilities in (0, 1]");
        }
    }
    
    // Check that targets are valid probabilities
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0.0 || targets[i] > 1.0) {
            throw std::domain_error("Targets must be valid probabilities in [0, 1]");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        sum_loss += targets[i] * std::log(predictions[i]);
    }
    
    return Tensor({1}, -sum_loss);
}

Tensor binary_cross_entropy_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that predictions are valid probabilities
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] <= 0.0 || predictions[i] >= 1.0) {
            throw std::domain_error("Predictions must be valid probabilities in (0, 1)");
        }
    }
    
    // Check that targets are valid probabilities
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0.0 || targets[i] > 1.0) {
            throw std::domain_error("Targets must be valid probabilities in [0, 1]");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double p = predictions[i];
        double t = targets[i];
        sum_loss += t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
    }
    
    return Tensor({1}, -sum_loss / static_cast<double>(predictions.size()));
}

Tensor categorical_cross_entropy_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that predictions are valid probabilities
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] <= 0.0 || predictions[i] > 1.0) {
            throw std::domain_error("Predictions must be valid probabilities in (0, 1]");
        }
    }
    
    // Check that targets are valid probabilities
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0.0 || targets[i] > 1.0) {
            throw std::domain_error("Targets must be valid probabilities in [0, 1]");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        sum_loss += targets[i] * std::log(predictions[i]);
    }
    
    return Tensor({1}, -sum_loss);
}

Tensor sparse_categorical_cross_entropy_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape().size() != 2 || targets.shape().size() != 1) {
        throw std::invalid_argument("Sparse categorical cross-entropy requires 2D predictions and 1D targets");
    }
    
    if (predictions.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("Batch sizes must match");
    }
    
    // Check that predictions are valid probabilities
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] <= 0.0 || predictions[i] > 1.0) {
            throw std::domain_error("Predictions must be valid probabilities in (0, 1]");
        }
    }
    
    // Check that targets are valid class indices
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0.0 || targets[i] >= static_cast<double>(predictions.shape()[1])) {
            throw std::domain_error("Targets must be valid class indices");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.shape()[0]; ++i) {
        size_t target_class = static_cast<size_t>(targets[i]);
        sum_loss += std::log(predictions({i, target_class}));
    }
    
    return Tensor({1}, -sum_loss / static_cast<double>(predictions.shape()[0]));
}

// Specialized loss functions
Tensor hinge_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that targets are ±1
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] != 1.0 && targets[i] != -1.0) {
            throw std::domain_error("Targets must be ±1 for hinge loss");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double margin = 1.0 - predictions[i] * targets[i];
        sum_loss += std::max(0.0, margin);
    }
    
    return Tensor({1}, sum_loss / static_cast<double>(predictions.size()));
}

Tensor hinge_loss(const Tensor& predictions, const Tensor& targets, double margin) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that targets are ±1
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] != 1.0 && targets[i] != -1.0) {
            throw std::domain_error("Targets must be ±1 for hinge loss");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double loss_margin = margin - predictions[i] * targets[i];
        sum_loss += std::max(0.0, loss_margin);
    }
    
    return Tensor({1}, sum_loss / static_cast<double>(predictions.size()));
}

Tensor squared_hinge_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that targets are ±1
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] != 1.0 && targets[i] != -1.0) {
            throw std::domain_error("Targets must be ±1 for squared hinge loss");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double margin = 1.0 - predictions[i] * targets[i];
        double loss = std::max(0.0, margin);
        sum_loss += loss * loss;
    }
    
    return Tensor({1}, sum_loss / static_cast<double>(predictions.size()));
}

Tensor squared_hinge_loss(const Tensor& predictions, const Tensor& targets, double margin) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that targets are ±1
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] != 1.0 && targets[i] != -1.0) {
            throw std::domain_error("Targets must be ±1 for squared hinge loss");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double loss_margin = margin - predictions[i] * targets[i];
        double loss = std::max(0.0, loss_margin);
        sum_loss += loss * loss;
    }
    
    return Tensor({1}, sum_loss / static_cast<double>(predictions.size()));
}

Tensor kl_divergence_loss(const Tensor& p, const Tensor& q) {
    if (p.shape() != q.shape()) {
        throw std::invalid_argument("Distributions must have the same shape");
    }
    
    // Check that p and q are valid probabilities
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] <= 0.0 || p[i] > 1.0) {
            throw std::domain_error("p must be valid probabilities in (0, 1]");
        }
        if (q[i] <= 0.0 || q[i] > 1.0) {
            throw std::domain_error("q must be valid probabilities in (0, 1]");
        }
    }
    
    double sum_kl = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        sum_kl += p[i] * std::log(p[i] / q[i]);
    }
    
    return Tensor({1}, sum_kl);
}

Tensor js_divergence_loss(const Tensor& p, const Tensor& q) {
    if (p.shape() != q.shape()) {
        throw std::invalid_argument("Distributions must have the same shape");
    }
    
    // Check that p and q are valid probabilities
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] <= 0.0 || p[i] > 1.0) {
            throw std::domain_error("p must be valid probabilities in (0, 1]");
        }
        if (q[i] <= 0.0 || q[i] > 1.0) {
            throw std::domain_error("q must be valid probabilities in (0, 1]");
        }
    }
    
    // Compute M = 0.5 * (p + q)
    Tensor m = p;
    for (size_t i = 0; i < p.size(); ++i) {
        m[i] = 0.5 * (p[i] + q[i]);
    }
    
    // Compute JS divergence = 0.5 * KL(p||M) + 0.5 * KL(q||M)
    double kl_p_m = 0.0;
    double kl_q_m = 0.0;
    
    for (size_t i = 0; i < p.size(); ++i) {
        kl_p_m += p[i] * std::log(p[i] / m[i]);
        kl_q_m += q[i] * std::log(q[i] / m[i]);
    }
    
    double js_div = 0.5 * kl_p_m + 0.5 * kl_q_m;
    return Tensor({1}, js_div);
}

// Loss function utilities
Tensor focal_loss(const Tensor& predictions, const Tensor& targets, double alpha, double gamma) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that predictions are valid probabilities
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] <= 0.0 || predictions[i] >= 1.0) {
            throw std::domain_error("Predictions must be valid probabilities in (0, 1)");
        }
    }
    
    // Check that targets are valid probabilities
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0.0 || targets[i] > 1.0) {
            throw std::domain_error("Targets must be valid probabilities in [0, 1]");
        }
    }
    
    double sum_loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        double p = predictions[i];
        double t = targets[i];
        double ce = t * std::log(p) + (1.0 - t) * std::log(1.0 - p);
        double p_t = (t == 1.0) ? p : (1.0 - p);
        double alpha_t = (t == 1.0) ? alpha : (1.0 - alpha);
        double focal_weight = alpha_t * std::pow(1.0 - p_t, gamma);
        sum_loss += focal_weight * ce;
    }
    
    return Tensor({1}, -sum_loss / static_cast<double>(predictions.size()));
}

Tensor dice_loss(const Tensor& predictions, const Tensor& targets, double smooth) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    // Check that predictions are valid probabilities
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] < 0.0 || predictions[i] > 1.0) {
            throw std::domain_error("Predictions must be valid probabilities in [0, 1]");
        }
    }
    
    // Check that targets are valid probabilities
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0.0 || targets[i] > 1.0) {
            throw std::domain_error("Targets must be valid probabilities in [0, 1]");
        }
    }
    
    double intersection = 0.0;
    double sum_p = 0.0;
    double sum_t = 0.0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        intersection += predictions[i] * targets[i];
        sum_p += predictions[i];
        sum_t += targets[i];
    }
    
    double dice = (2.0 * intersection + smooth) / (sum_p + sum_t + smooth);
    return Tensor({1}, 1.0 - dice);
}

// Loss function selection helper
std::string get_loss_name([[maybe_unused]] const std::function<Tensor(const Tensor&, const Tensor&)>& loss_func) {
    // This is a simple approach - in practice, you might want to use function pointers
    // or a more sophisticated method to identify loss functions
    return "unknown";
}

// Loss function factory
std::function<Tensor(const Tensor&, const Tensor&)> create_loss(const std::string& name) {
    if (name == "mse") {
        return mse_loss;
    } else if (name == "mae") {
        return mae_loss;
    } else if (name == "huber") {
        return [](const Tensor& p, const Tensor& t) { return huber_loss(p, t, 1.0); };
    } else if (name == "smooth_l1") {
        return [](const Tensor& p, const Tensor& t) { return smooth_l1_loss(p, t, 1.0); };
    } else if (name == "cross_entropy") {
        return cross_entropy_loss;
    } else if (name == "binary_cross_entropy") {
        return binary_cross_entropy_loss;
    } else if (name == "categorical_cross_entropy") {
        return categorical_cross_entropy_loss;
    } else if (name == "sparse_categorical_cross_entropy") {
        return sparse_categorical_cross_entropy_loss;
    } else if (name == "hinge") {
        return [](const Tensor& p, const Tensor& t) { return hinge_loss(p, t, 1.0); };
    } else if (name == "squared_hinge") {
        return [](const Tensor& p, const Tensor& t) { return squared_hinge_loss(p, t, 1.0); };
    } else if (name == "kl_divergence") {
        return kl_divergence_loss;
    } else if (name == "js_divergence") {
        return js_divergence_loss;
    } else {
        throw std::invalid_argument("Unknown loss function: " + name);
    }
}

} // namespace tensorcore
