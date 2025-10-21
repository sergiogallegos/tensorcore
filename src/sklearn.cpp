#include "tensorcore/sklearn.hpp"
#include "tensorcore/operations.hpp"
#include "tensorcore/utils.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <set>
#include <map>
#include <limits>

namespace tensorcore {
namespace sklearn {

// ============================================================================
// LinearRegression Implementation
// ============================================================================

LinearRegression::LinearRegression(bool fit_intercept, bool normalize, bool copy_X, int n_jobs)
    : fit_intercept_(fit_intercept), normalize_(normalize), copy_X_(copy_X), n_jobs_(n_jobs),
      fitted_(false) {
}

void LinearRegression::fit(const Tensor& X, const Tensor& y) {
    if (X.shape().size() != 2) {
        throw std::invalid_argument("X must be a 2D tensor");
    }
    if (y.shape().size() != 1) {
        throw std::invalid_argument("y must be a 1D tensor");
    }
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    [[maybe_unused]] int n_features = X.shape()[1];
    
    Tensor X_work = copy_X_ ? X.copy() : X;
    Tensor y_work = y.copy();
    
    // Normalize features if requested
    if (normalize_) {
        Tensor X_mean = X_work.mean();
        Tensor X_std = X_work.std();
        X_work = (X_work - X_mean) / X_std;
    }
    
    if (fit_intercept_) {
        // Add bias term (column of ones)
        Tensor ones = tensorcore::ones({static_cast<size_t>(n_samples), 1});
        X_work = concatenate({ones, X_work}, 1);
        n_features += 1;
    }
    
    // Normal equation: θ = (X^T X)^(-1) X^T y
    Tensor XTX = X_work.transpose().matmul(X_work);
    Tensor XTy = X_work.transpose().matmul(y_work.reshape({static_cast<size_t>(-1), 1}));
    
    // Solve for parameters
    Tensor theta = XTX.inv().matmul(XTy);
    
    if (fit_intercept_) {
        intercept_ = Tensor({1}, theta[0]);
        coef_ = theta.slice({{1, theta.shape()[0]}});
    } else {
        intercept_ = tensorcore::zeros({1});
        coef_ = theta;
    }
    
    fitted_ = true;
}

Tensor LinearRegression::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    Tensor X_work = copy_X_ ? X.copy() : X;
    
    // Normalize features if they were normalized during training
    if (normalize_) {
        // Note: In a real implementation, we'd store the training statistics
        // For now, we'll assume features are already normalized
    }
    
    Tensor predictions = X_work.matmul(coef_.reshape({static_cast<size_t>(-1), 1}));
    if (fit_intercept_) {
        predictions = predictions + intercept_;
    }
    
    return predictions.flatten();
}

double LinearRegression::score(const Tensor& X, const Tensor& y) const {
    Tensor y_pred = const_cast<LinearRegression*>(this)->predict(X);
    Tensor residuals = y - y_pred;
    Tensor ss_res = (residuals * residuals).sum();
    Tensor ss_tot = ((y - y.mean()) * (y - y.mean())).sum();
    return 1.0 - (ss_res.item() / ss_tot.item());
}

// ============================================================================
// Ridge Implementation
// ============================================================================

Ridge::Ridge(double alpha, bool fit_intercept, bool normalize, int max_iter, double tol)
    : alpha_(alpha), fit_intercept_(fit_intercept), normalize_(normalize), 
      max_iter_(max_iter), tol_(tol), fitted_(false) {
}

void Ridge::fit(const Tensor& X, const Tensor& y) {
    if (X.shape().size() != 2) {
        throw std::invalid_argument("X must be a 2D tensor");
    }
    if (y.shape().size() != 1) {
        throw std::invalid_argument("y must be a 1D tensor");
    }
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    [[maybe_unused]] int n_features = X.shape()[1];
    
    Tensor X_work = X.copy();
    Tensor y_work = y.copy();
    
    // Normalize features if requested
    if (normalize_) {
        Tensor X_mean = X_work.mean();
        Tensor X_std = X_work.std();
        X_work = (X_work - X_mean) / X_std;
    }
    
    if (fit_intercept_) {
        // Add bias term (column of ones)
        Tensor ones = tensorcore::ones({static_cast<size_t>(n_samples), 1});
        X_work = concatenate({ones, X_work}, 1);
        n_features += 1;
    }
    
    // Ridge regression: θ = (X^T X + αI)^(-1) X^T y
    Tensor XTX = X_work.transpose().matmul(X_work);
    Tensor I = tensorcore::eye(n_features) * alpha_;
    Tensor XTy = X_work.transpose().matmul(y_work.reshape({static_cast<size_t>(-1), 1}));
    
    // Solve for parameters
    Tensor theta = (XTX + I).inv().matmul(XTy);
    
    if (fit_intercept_) {
        intercept_ = Tensor({1}, theta[0]);
        coef_ = theta.slice({{1, theta.shape()[0]}});
    } else {
        intercept_ = tensorcore::zeros({1});
        coef_ = theta;
    }
    
    fitted_ = true;
}

Tensor Ridge::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    Tensor X_work = X.copy();
    
    // Normalize features if they were normalized during training
    if (normalize_) {
        // Note: In a real implementation, we'd store the training statistics
    }
    
    Tensor predictions = X_work.matmul(coef_.reshape({static_cast<size_t>(-1), 1}));
    if (fit_intercept_) {
        predictions = predictions + intercept_;
    }
    
    return predictions.flatten();
}

double Ridge::score(const Tensor& X, const Tensor& y) const {
    Tensor y_pred = const_cast<Ridge*>(this)->predict(X);
    Tensor residuals = y - y_pred;
    Tensor ss_res = (residuals * residuals).sum();
    Tensor ss_tot = ((y - y.mean()) * (y - y.mean())).sum();
    return 1.0 - (ss_res.item() / ss_tot.item());
}

// ============================================================================
// Lasso Implementation (Simplified - using coordinate descent)
// ============================================================================

Lasso::Lasso(double alpha, bool fit_intercept, bool normalize, int max_iter, double tol)
    : alpha_(alpha), fit_intercept_(fit_intercept), normalize_(normalize), 
      max_iter_(max_iter), tol_(tol), fitted_(false) {
}

void Lasso::fit(const Tensor& X, const Tensor& y) {
    if (X.shape().size() != 2) {
        throw std::invalid_argument("X must be a 2D tensor");
    }
    if (y.shape().size() != 1) {
        throw std::invalid_argument("y must be a 1D tensor");
    }
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    [[maybe_unused]] int n_features = X.shape()[1];
    
    Tensor X_work = X.copy();
    Tensor y_work = y.copy();
    
    // Normalize features if requested
    if (normalize_) {
        Tensor X_mean = X_work.mean();
        Tensor X_std = X_work.std();
        X_work = (X_work - X_mean) / X_std;
    }
    
    // Initialize coefficients
    coef_ = tensorcore::zeros({static_cast<size_t>(n_features)});
    if (fit_intercept_) {
        intercept_ = y_work.mean();
    } else {
        intercept_ = tensorcore::zeros({1});
    }
    
    // Coordinate descent algorithm
    for (int iter = 0; iter < max_iter_; ++iter) {
        Tensor coef_old = coef_.copy();
        
        for (int j = 0; j < n_features; ++j) {
            // Compute residual without feature j
            Tensor residual = y_work - X_work.matmul(coef_.reshape({static_cast<size_t>(-1), 1})).flatten();
            if (fit_intercept_) {
                residual = residual - intercept_;
            }
            
            // Update coefficient j
            Tensor X_j = X_work.slice({{{1, 1}, {j, j+1}}}).flatten();
            double rho = (residual * X_j).sum().item();
            double norm = (X_j * X_j).sum().item();
            
            if (norm > 0) {
                double coef_j = rho / norm;
                double soft_threshold = alpha_ / norm;
                
                if (coef_j > soft_threshold) {
                    coef_[j] = coef_j - soft_threshold;
                } else if (coef_j < -soft_threshold) {
                    coef_[j] = coef_j + soft_threshold;
                } else {
                    coef_[j] = 0.0;
                }
            }
        }
        
        // Check convergence
        Tensor coef_diff = coef_ - coef_old;
        double max_change = coef_diff.abs().max().item();
        if (max_change < tol_) {
            break;
        }
    }
    
    fitted_ = true;
}

Tensor Lasso::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    Tensor X_work = X.copy();
    
    // Normalize features if they were normalized during training
    if (normalize_) {
        // Note: In a real implementation, we'd store the training statistics
    }
    
    Tensor predictions = X_work.matmul(coef_.reshape({static_cast<size_t>(-1), 1}));
    if (fit_intercept_) {
        predictions = predictions + intercept_;
    }
    
    return predictions.flatten();
}

double Lasso::score(const Tensor& X, const Tensor& y) const {
    Tensor y_pred = const_cast<Lasso*>(this)->predict(X);
    Tensor residuals = y - y_pred;
    Tensor ss_res = (residuals * residuals).sum();
    Tensor ss_tot = ((y - y.mean()) * (y - y.mean())).sum();
    return 1.0 - (ss_res.item() / ss_tot.item());
}

// ============================================================================
// ElasticNet Implementation
// ============================================================================

ElasticNet::ElasticNet(double alpha, double l1_ratio, bool fit_intercept, bool normalize, 
                       int max_iter, double tol)
    : alpha_(alpha), l1_ratio_(l1_ratio), fit_intercept_(fit_intercept), 
      normalize_(normalize), max_iter_(max_iter), tol_(tol), fitted_(false) {
}

void ElasticNet::fit(const Tensor& X, const Tensor& y) {
    if (X.shape().size() != 2) {
        throw std::invalid_argument("X must be a 2D tensor");
    }
    if (y.shape().size() != 1) {
        throw std::invalid_argument("y must be a 1D tensor");
    }
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    [[maybe_unused]] int n_features = X.shape()[1];
    
    Tensor X_work = X.copy();
    Tensor y_work = y.copy();
    
    // Normalize features if requested
    if (normalize_) {
        Tensor X_mean = X_work.mean();
        Tensor X_std = X_work.std();
        X_work = (X_work - X_mean) / X_std;
    }
    
    // Initialize coefficients
    coef_ = tensorcore::zeros({static_cast<size_t>(n_features)});
    if (fit_intercept_) {
        intercept_ = y_work.mean();
    } else {
        intercept_ = tensorcore::zeros({1});
    }
    
    // Coordinate descent algorithm with elastic net penalty
    double alpha_l1 = alpha_ * l1_ratio_;
    double alpha_l2 = alpha_ * (1.0 - l1_ratio_);
    
    for (int iter = 0; iter < max_iter_; ++iter) {
        Tensor coef_old = coef_.copy();
        
        for (int j = 0; j < n_features; ++j) {
            // Compute residual without feature j
            Tensor residual = y_work - X_work.matmul(coef_.reshape({static_cast<size_t>(-1), 1})).flatten();
            if (fit_intercept_) {
                residual = residual - intercept_;
            }
            
            // Update coefficient j
            Tensor X_j = X_work.slice({{{1, 1}, {j, j+1}}}).flatten();
            double rho = (residual * X_j).sum().item();
            double norm = (X_j * X_j).sum().item();
            
            if (norm > 0) {
                double coef_j = rho / (norm + alpha_l2);
                double soft_threshold = alpha_l1 / (norm + alpha_l2);
                
                if (coef_j > soft_threshold) {
                    coef_[j] = coef_j - soft_threshold;
                } else if (coef_j < -soft_threshold) {
                    coef_[j] = coef_j + soft_threshold;
                } else {
                    coef_[j] = 0.0;
                }
            }
        }
        
        // Check convergence
        Tensor coef_diff = coef_ - coef_old;
        double max_change = coef_diff.abs().max().item();
        if (max_change < tol_) {
            break;
        }
    }
    
    fitted_ = true;
}

Tensor ElasticNet::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    Tensor X_work = X.copy();
    
    // Normalize features if they were normalized during training
    if (normalize_) {
        // Note: In a real implementation, we'd store the training statistics
    }
    
    Tensor predictions = X_work.matmul(coef_.reshape({static_cast<size_t>(-1), 1}));
    if (fit_intercept_) {
        predictions = predictions + intercept_;
    }
    
    return predictions.flatten();
}

double ElasticNet::score(const Tensor& X, const Tensor& y) const {
    Tensor y_pred = const_cast<ElasticNet*>(this)->predict(X);
    Tensor residuals = y - y_pred;
    Tensor ss_res = (residuals * residuals).sum();
    Tensor ss_tot = ((y - y.mean()) * (y - y.mean())).sum();
    return 1.0 - (ss_res.item() / ss_tot.item());
}

// ============================================================================
// LogisticRegression Implementation (Simplified)
// ============================================================================

LogisticRegression::LogisticRegression(const std::string& penalty, double C, bool fit_intercept, 
                                      int max_iter, double tol, const std::string& multi_class)
    : penalty_(penalty), C_(C), fit_intercept_(fit_intercept), max_iter_(max_iter), 
      tol_(tol), multi_class_(multi_class), fitted_(false) {
}

void LogisticRegression::fit(const Tensor& X, const Tensor& y) {
    if (X.shape().size() != 2) {
        throw std::invalid_argument("X must be a 2D tensor");
    }
    if (y.shape().size() != 1) {
        throw std::invalid_argument("y must be a 1D tensor");
    }
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    // Get unique classes
    std::vector<int> unique_classes;
    for (int i = 0; i < static_cast<int>(y.shape()[0]); ++i) {
        int class_val = static_cast<int>(y[i]);
        if (std::find(unique_classes.begin(), unique_classes.end(), class_val) == unique_classes.end()) {
            unique_classes.push_back(class_val);
        }
    }
    std::sort(unique_classes.begin(), unique_classes.end());
    classes_ = unique_classes;
    
    int n_classes = classes_.size();
    [[maybe_unused]] int n_samples = X.shape()[0];
    [[maybe_unused]] int n_features = X.shape()[1];
    
    if (n_classes == 2) {
        // Binary classification
        Tensor y_binary = y.copy();
        for (int i = 0; i < static_cast<int>(y_binary.shape()[0]); ++i) {
            y_binary[i] = (y_binary[i] == classes_[1]) ? 1.0 : 0.0;
        }
        
        // Initialize coefficients
        coef_ = tensorcore::zeros({1, static_cast<size_t>(n_features)});
        if (fit_intercept_) {
            intercept_ = tensorcore::zeros({1});
        } else {
            intercept_ = tensorcore::zeros({1});
        }
        
        // Gradient descent for logistic regression
        double learning_rate = 0.01;
        for (int iter = 0; iter < max_iter_; ++iter) {
            // Forward pass
            Tensor z = X.matmul(coef_.transpose()).flatten();
            if (fit_intercept_) {
                z = z + intercept_[0];
            }
            
            // Sigmoid function
            Tensor sigmoid_z = 1.0 / (1.0 + tensorcore::exp(-z));
            
            // Compute loss and gradients
            Tensor loss = -(y_binary * tensorcore::log(sigmoid_z + 1e-15) + 
                           (1.0 - y_binary) * tensorcore::log(1.0 - sigmoid_z + 1e-15)).mean();
            
            Tensor error = sigmoid_z - y_binary;
            Tensor grad_coef = (X.transpose().matmul(error.reshape({static_cast<size_t>(-1), 1})) / n_samples).transpose();
            
            // Add regularization
            if (penalty_ == "l2") {
                grad_coef = grad_coef + (coef_ / C_);
            }
            
            // Update parameters
            coef_ = coef_ - learning_rate * grad_coef;
            if (fit_intercept_) {
                Tensor grad_intercept = error.mean();
                intercept_[0] = intercept_[0] - learning_rate * grad_intercept.item();
            }
        }
    } else {
        // Multiclass classification (One-vs-Rest)
        coef_ = tensorcore::zeros({static_cast<size_t>(n_classes), static_cast<size_t>(n_features)});
        intercept_ = tensorcore::zeros({static_cast<size_t>(n_classes)});
        
        for (int c = 0; c < n_classes; ++c) {
            // Create binary labels for class c
            Tensor y_binary = tensorcore::zeros({static_cast<size_t>(n_samples)});
            for (int i = 0; i < n_samples; ++i) {
                y_binary[i] = (static_cast<int>(y[i]) == classes_[c]) ? 1.0 : 0.0;
            }
            
            // Train binary classifier for class c
            Tensor coef_c = tensorcore::zeros({1, static_cast<size_t>(n_features)});
            Tensor intercept_c = tensorcore::zeros({1});
            
            double learning_rate = 0.01;
            for (int iter = 0; iter < max_iter_; ++iter) {
                Tensor z = X.matmul(coef_c.transpose()).flatten() + intercept_c[0];
                Tensor sigmoid_z = 1.0 / (1.0 + tensorcore::exp(-z));
                
                Tensor error = sigmoid_z - y_binary;
                Tensor grad_coef = (X.transpose().matmul(error.reshape({static_cast<size_t>(-1), 1})) / n_samples).transpose();
                
                if (penalty_ == "l2") {
                    grad_coef = grad_coef + (coef_c / C_);
                }
                
                coef_c = coef_c - learning_rate * grad_coef;
                Tensor grad_intercept = error.mean();
                intercept_c[0] = intercept_c[0] - learning_rate * grad_intercept.item();
            }
            
            coef_[c] = coef_c[0];
            intercept_[c] = intercept_c[0];
        }
    }
    
    fitted_ = true;
}

Tensor LogisticRegression::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    int n_classes = classes_.size();
    
    if (n_classes == 2) {
        // Binary classification
        Tensor z = X.matmul(coef_.transpose()).flatten();
        if (fit_intercept_) {
            z = z + intercept_[0];
        }
        Tensor probabilities = 1.0 / (1.0 + tensorcore::exp(-z));
        
        Tensor predictions = tensorcore::zeros({static_cast<size_t>(n_samples)});
        for (int i = 0; i < n_samples; ++i) {
            predictions[i] = (probabilities[i] > 0.5) ? classes_[1] : classes_[0];
        }
        return predictions;
    } else {
        // Multiclass classification
        Tensor scores = X.matmul(coef_.transpose());
        if (fit_intercept_) {
            for (int i = 0; i < n_samples; ++i) {
                for (int c = 0; c < n_classes; ++c) {
                    scores[i * n_classes + c] = scores[i * n_classes + c] + intercept_[c];
                }
            }
        }
        
        Tensor predictions = tensorcore::zeros({static_cast<size_t>(n_samples)});
        for (int i = 0; i < n_samples; ++i) {
            int max_class = 0;
            double max_score = scores[i * n_classes];
            for (int c = 1; c < n_classes; ++c) {
                if (scores[i * n_classes + c] > max_score) {
                    max_score = scores[i * n_classes + c];
                    max_class = c;
                }
            }
            predictions[i] = classes_[max_class];
        }
        return predictions;
    }
}

Tensor LogisticRegression::predict_proba(const Tensor& X) {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    int n_classes = classes_.size();
    
    if (n_classes == 2) {
        // Binary classification
        Tensor z = X.matmul(coef_.transpose()).flatten();
        if (fit_intercept_) {
            z = z + intercept_[0];
        }
        Tensor probabilities = 1.0 / (1.0 + tensorcore::exp(-z));
        
        Tensor proba = tensorcore::zeros({static_cast<size_t>(n_samples), 2});
        for (int i = 0; i < n_samples; ++i) {
            double p = probabilities[i];
            proba[i * 2] = 1.0 - p;      // Probability of class 0
            proba[i * 2 + 1] = p;        // Probability of class 1
        }
        return proba;
    } else {
        // Multiclass classification
        Tensor scores = X.matmul(coef_.transpose());
        if (fit_intercept_) {
            for (int i = 0; i < n_samples; ++i) {
                for (int c = 0; c < n_classes; ++c) {
                    scores[i * n_classes + c] = scores[i * n_classes + c] + intercept_[c];
                }
            }
        }
        
        // Softmax
        Tensor proba = tensorcore::zeros({static_cast<size_t>(n_samples), static_cast<size_t>(n_classes)});
        for (int i = 0; i < n_samples; ++i) {
            Tensor row_scores = scores.slice({{{0, 1}, {i, i+1}}}).flatten();
            Tensor exp_scores = tensorcore::exp(row_scores - row_scores.max());
            Tensor softmax_scores = exp_scores / exp_scores.sum();
            
            for (int c = 0; c < n_classes; ++c) {
                proba[i * n_classes + c] = softmax_scores[c];
            }
        }
        return proba;
    }
}

double LogisticRegression::score(const Tensor& X, const Tensor& y) const {
    Tensor y_pred = const_cast<LogisticRegression*>(this)->predict(X);
    int correct = 0;
    for (int i = 0; i < static_cast<int>(y.shape()[0]); ++i) {
        if (static_cast<int>(y[i]) == static_cast<int>(y_pred[i])) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y.shape()[0];
}

// ============================================================================
// Metrics Implementation
// ============================================================================

namespace metrics {

double accuracy_score(const Tensor& y_true, const Tensor& y_pred) {
    if (y_true.shape() != y_pred.shape()) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }
    
    int correct = 0;
    for (int i = 0; i < static_cast<int>(y_true.shape()[0]); ++i) {
        if (static_cast<int>(y_true[i]) == static_cast<int>(y_pred[i])) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y_true.shape()[0];
}

double precision_score(const Tensor& y_true, const Tensor& y_pred, const std::string& average) {
    // Simplified implementation for binary classification
    if (average != "binary") {
        throw std::invalid_argument("Only binary average is currently supported");
    }
    
    int true_positives = 0;
    int false_positives = 0;
    
    for (int i = 0; i < static_cast<int>(y_true.shape()[0]); ++i) {
        int true_label = static_cast<int>(y_true[i]);
        int pred_label = static_cast<int>(y_pred[i]);
        
        if (pred_label == 1) {
            if (true_label == 1) {
                true_positives++;
            } else {
                false_positives++;
            }
        }
    }
    
    if (true_positives + false_positives == 0) {
        return 0.0;
    }
    
    return static_cast<double>(true_positives) / (true_positives + false_positives);
}

double recall_score(const Tensor& y_true, const Tensor& y_pred, const std::string& average) {
    // Simplified implementation for binary classification
    if (average != "binary") {
        throw std::invalid_argument("Only binary average is currently supported");
    }
    
    int true_positives = 0;
    int false_negatives = 0;
    
    for (int i = 0; i < static_cast<int>(y_true.shape()[0]); ++i) {
        int true_label = static_cast<int>(y_true[i]);
        int pred_label = static_cast<int>(y_pred[i]);
        
        if (true_label == 1) {
            if (pred_label == 1) {
                true_positives++;
            } else {
                false_negatives++;
            }
        }
    }
    
    if (true_positives + false_negatives == 0) {
        return 0.0;
    }
    
    return static_cast<double>(true_positives) / (true_positives + false_negatives);
}

double f1_score(const Tensor& y_true, const Tensor& y_pred, const std::string& average) {
    double precision = precision_score(y_true, y_pred, average);
    double recall = recall_score(y_true, y_pred, average);
    
    if (precision + recall == 0.0) {
        return 0.0;
    }
    
    return 2.0 * precision * recall / (precision + recall);
}

double mean_squared_error(const Tensor& y_true, const Tensor& y_pred) {
    Tensor residuals = y_true - y_pred;
    return (residuals * residuals).mean().item();
}

double mean_absolute_error(const Tensor& y_true, const Tensor& y_pred) {
    Tensor residuals = y_true - y_pred;
    return residuals.abs().mean().item();
}

double r2_score(const Tensor& y_true, const Tensor& y_pred) {
    Tensor ss_res = ((y_true - y_pred) * (y_true - y_pred)).sum();
    Tensor ss_tot = ((y_true - y_true.mean()) * (y_true - y_true.mean())).sum();
    return 1.0 - (ss_res.item() / ss_tot.item());
}

} // namespace metrics

// ============================================================================
// Model Selection Implementation
// ============================================================================

namespace model_selection {

std::pair<Tensor, Tensor> train_test_split(const Tensor& X, const Tensor& y, 
                                           double test_size, int random_state) {
    if (X.shape()[0] != y.shape()[0]) {
        throw std::invalid_argument("X and y must have the same number of samples");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    int n_test = static_cast<int>(n_samples * test_size);
    int n_train = n_samples - n_test;
    
    // Create indices
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Shuffle indices
    std::mt19937 rng(random_state);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Split indices
    std::vector<int> train_indices(indices.begin(), indices.begin() + n_train);
    std::vector<int> test_indices(indices.begin() + n_train, indices.end());
    
    // Create train/test splits
    Tensor X_train = tensorcore::zeros({static_cast<size_t>(n_train), X.shape()[1]});
    Tensor y_train = tensorcore::zeros({static_cast<size_t>(n_train)});
    Tensor X_test = tensorcore::zeros({static_cast<size_t>(n_test), X.shape()[1]});
    Tensor y_test = tensorcore::zeros({static_cast<size_t>(n_test)});
    
    for (int i = 0; i < n_train; ++i) {
        int idx = train_indices[i];
        for (int j = 0; j < static_cast<int>(X.shape()[1]); ++j) {
            X_train[i * X.shape()[1] + j] = X[idx * X.shape()[1] + j];
        }
        y_train[i] = y[idx];
    }
    
    for (int i = 0; i < n_test; ++i) {
        int idx = test_indices[i];
        for (int j = 0; j < static_cast<int>(X.shape()[1]); ++j) {
            X_test[i * X.shape()[1] + j] = X[idx * X.shape()[1] + j];
        }
        y_test[i] = y[idx];
    }
    
    return std::make_pair(X_train, X_test);
}

} // namespace model_selection

// Tree-based models implementation
// DecisionTreeClassifier implementation
DecisionTreeClassifier::DecisionTreeClassifier(const std::string& criterion, int max_depth,
                                             int min_samples_split, int min_samples_leaf, 
                                             int random_state)
    : criterion_(criterion), max_depth_(max_depth), min_samples_split_(min_samples_split),
      min_samples_leaf_(min_samples_leaf), random_state_(random_state), fitted_(false) {
}

void DecisionTreeClassifier::fit(const Tensor& X, const Tensor& y) {
    [[maybe_unused]] int n_samples = X.shape()[0];
    [[maybe_unused]] int n_features = X.shape()[1];
    
    // Get unique classes
    std::set<int> unique_classes_set;
    for (int i = 0; i < n_samples; ++i) {
        unique_classes_set.insert(static_cast<int>(y[i]));
    }
    classes_.assign(unique_classes_set.begin(), unique_classes_set.end());
    
    // Initialize indices
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    // Build tree
    tree_.clear();
    build_tree(X, y, indices, 0);
    
    fitted_ = true;
}

int DecisionTreeClassifier::build_tree(const Tensor& X, const Tensor& y, 
                                      const std::vector<int>& indices, int depth) {
    int node_id = tree_.size();
    TreeNode node;
    
    // Check stopping criteria
    if (depth >= max_depth_ || static_cast<int>(indices.size()) < min_samples_split_) {
        // Create leaf node
        node.is_leaf = true;
        node.feature = -1;
        node.threshold = 0.0;
        node.left_child = -1;
        node.right_child = -1;
        
        // Find most common class
        std::map<int, int> class_counts;
        for (int idx : indices) {
            int class_val = static_cast<int>(y[idx]);
            class_counts[class_val]++;
        }
        
        int max_count = 0;
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                node.prediction = pair.first;
            }
        }
        
        tree_.push_back(node);
        return node_id;
    }
    
    // Find best split
    int best_feature;
    double best_threshold;
    double best_impurity = find_best_split(X, y, indices, best_feature, best_threshold);
    
    if (best_impurity == -1) {
        // No valid split found, create leaf
        node.is_leaf = true;
        node.feature = -1;
        node.threshold = 0.0;
        node.left_child = -1;
        node.right_child = -1;
        
        // Find most common class
        std::map<int, int> class_counts;
        for (int idx : indices) {
            int class_val = static_cast<int>(y[idx]);
            class_counts[class_val]++;
        }
        
        int max_count = 0;
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                node.prediction = pair.first;
            }
        }
        
        tree_.push_back(node);
        return node_id;
    }
    
    // Split indices
    std::vector<int> left_indices, right_indices;
    for (int idx : indices) {
        if (X[idx * X.shape()[1] + best_feature] <= best_threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }
    
    // Check if split is valid
    if (static_cast<int>(left_indices.size()) < min_samples_leaf_ || static_cast<int>(right_indices.size()) < min_samples_leaf_) {
        // Create leaf node
        node.is_leaf = true;
        node.feature = -1;
        node.threshold = 0.0;
        node.left_child = -1;
        node.right_child = -1;
        
        // Find most common class
        std::map<int, int> class_counts;
        for (int idx : indices) {
            int class_val = static_cast<int>(y[idx]);
            class_counts[class_val]++;
        }
        
        int max_count = 0;
        for (const auto& pair : class_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                node.prediction = pair.first;
            }
        }
        
        tree_.push_back(node);
        return node_id;
    }
    
    // Create internal node
    node.is_leaf = false;
    node.feature = best_feature;
    node.threshold = best_threshold;
    node.prediction = -1;
    
    tree_.push_back(node);
    
    // Recursively build children
    node.left_child = build_tree(X, y, left_indices, depth + 1);
    node.right_child = build_tree(X, y, right_indices, depth + 1);
    
    // Update the node with children
    tree_[node_id].left_child = node.left_child;
    tree_[node_id].right_child = node.right_child;
    
    return node_id;
}

double DecisionTreeClassifier::find_best_split(const Tensor& X, const Tensor& y, 
                                             const std::vector<int>& indices,
                                             int& best_feature, double& best_threshold) {
    [[maybe_unused]] int n_features = X.shape()[1];
    double best_impurity = std::numeric_limits<double>::max();
    best_feature = -1;
    best_threshold = 0.0;
    
    for (int feature = 0; feature < n_features; ++feature) {
        // Get unique values for this feature
        std::set<double> unique_values;
        for (int idx : indices) {
            unique_values.insert(X[idx * n_features + feature]);
        }
        
        // Try each unique value as threshold
        for (double threshold : unique_values) {
            std::vector<int> left_indices, right_indices;
            for (int idx : indices) {
                if (X[idx * n_features + feature] <= threshold) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }
            
            if (static_cast<int>(left_indices.size()) < min_samples_leaf_ || static_cast<int>(right_indices.size()) < min_samples_leaf_) {
                continue;
            }
            
            // Calculate weighted impurity
            double left_impurity = calculate_impurity(y, left_indices);
            double right_impurity = calculate_impurity(y, right_indices);
            double weighted_impurity = (left_indices.size() * left_impurity + 
                                      right_indices.size() * right_impurity) / indices.size();
            
            if (weighted_impurity < best_impurity) {
                best_impurity = weighted_impurity;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
    
    return (best_feature == -1) ? -1 : best_impurity;
}

double DecisionTreeClassifier::calculate_impurity(const Tensor& y, const std::vector<int>& indices) {
    if (criterion_ == "gini") {
        return calculate_gini(y, indices);
    } else if (criterion_ == "entropy" || criterion_ == "log_loss") {
        return calculate_entropy(y, indices);
    } else {
        throw std::invalid_argument("Unknown criterion: " + criterion_);
    }
}

double DecisionTreeClassifier::calculate_gini(const Tensor& y, const std::vector<int>& indices) {
    std::map<int, int> class_counts;
    for (int idx : indices) {
        int class_val = static_cast<int>(y[idx]);
        class_counts[class_val]++;
    }
    
    double gini = 1.0;
    for (const auto& pair : class_counts) {
        double p = static_cast<double>(pair.second) / indices.size();
        gini -= p * p;
    }
    
    return gini;
}

double DecisionTreeClassifier::calculate_entropy(const Tensor& y, const std::vector<int>& indices) {
    std::map<int, int> class_counts;
    for (int idx : indices) {
        int class_val = static_cast<int>(y[idx]);
        class_counts[class_val]++;
    }
    
    double entropy = 0.0;
    for (const auto& pair : class_counts) {
        double p = static_cast<double>(pair.second) / indices.size();
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

Tensor DecisionTreeClassifier::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    Tensor predictions(std::vector<size_t>{static_cast<size_t>(n_samples)});
    
    for (int i = 0; i < n_samples; ++i) {
        Tensor sample = X.slice({{i, i+1}}).flatten();
        predictions[i] = predict_sample(sample);
    }
    
    return predictions;
}

int DecisionTreeClassifier::predict_sample(const Tensor& sample) const {
    int node_id = 0;
    
    while (!tree_[node_id].is_leaf) {
        int feature = tree_[node_id].feature;
        double threshold = tree_[node_id].threshold;
        
        if (sample[feature] <= threshold) {
            node_id = tree_[node_id].left_child;
        } else {
            node_id = tree_[node_id].right_child;
        }
    }
    
    return tree_[node_id].prediction;
}

Tensor DecisionTreeClassifier::predict_proba(const Tensor& X) {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    int n_classes = classes_.size();
    Tensor probabilities(std::vector<size_t>{static_cast<size_t>(n_samples), static_cast<size_t>(n_classes)});
    
    for (int i = 0; i < n_samples; ++i) {
        Tensor sample = X.slice({{i, i+1}}).flatten();
        std::vector<double> probs = predict_proba_sample(sample);
        
        for (int j = 0; j < n_classes; ++j) {
            probabilities[i * n_classes + j] = probs[j];
        }
    }
    
    return probabilities;
}

std::vector<double> DecisionTreeClassifier::predict_proba_sample(const Tensor& sample) const {
    int node_id = 0;
    
    while (!tree_[node_id].is_leaf) {
        int feature = tree_[node_id].feature;
        double threshold = tree_[node_id].threshold;
        
        if (sample[feature] <= threshold) {
            node_id = tree_[node_id].left_child;
        } else {
            node_id = tree_[node_id].right_child;
        }
    }
    
    // For simplicity, return uniform probabilities
    // In a real implementation, you'd track class probabilities at each leaf
    int n_classes = classes_.size();
    std::vector<double> probs(n_classes, 1.0 / n_classes);
    return probs;
}

double DecisionTreeClassifier::score(const Tensor& X, const Tensor& y) const {
    Tensor predictions = predict(X);
    return metrics::accuracy_score(y, predictions);
}

// DecisionTreeRegressor implementation
DecisionTreeRegressor::DecisionTreeRegressor(const std::string& criterion, int max_depth,
                                           int min_samples_split, int min_samples_leaf, 
                                           int random_state)
    : criterion_(criterion), max_depth_(max_depth), min_samples_split_(min_samples_split),
      min_samples_leaf_(min_samples_leaf), random_state_(random_state), fitted_(false) {
}

void DecisionTreeRegressor::fit(const Tensor& X, const Tensor& y) {
    [[maybe_unused]] int n_samples = X.shape()[0];
    
    // Initialize indices
    std::vector<int> indices(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    // Build tree
    tree_.clear();
    build_tree(X, y, indices, 0);
    
    fitted_ = true;
}

int DecisionTreeRegressor::build_tree(const Tensor& X, const Tensor& y, 
                                     const std::vector<int>& indices, int depth) {
    int node_id = tree_.size();
    TreeNode node;
    
    // Check stopping criteria
    if (depth >= max_depth_ || static_cast<int>(indices.size()) < min_samples_split_) {
        // Create leaf node
        node.is_leaf = true;
        node.feature = -1;
        node.threshold = 0.0;
        node.left_child = -1;
        node.right_child = -1;
        
        // Calculate mean value
        double sum = 0.0;
        for (int idx : indices) {
            sum += y[idx];
        }
        node.prediction = sum / indices.size();
        
        tree_.push_back(node);
        return node_id;
    }
    
    // Find best split
    int best_feature;
    double best_threshold;
    double best_impurity = find_best_split(X, y, indices, best_feature, best_threshold);
    
    if (best_impurity == -1) {
        // No valid split found, create leaf
        node.is_leaf = true;
        node.feature = -1;
        node.threshold = 0.0;
        node.left_child = -1;
        node.right_child = -1;
        
        // Calculate mean value
        double sum = 0.0;
        for (int idx : indices) {
            sum += y[idx];
        }
        node.prediction = sum / indices.size();
        
        tree_.push_back(node);
        return node_id;
    }
    
    // Split indices
    std::vector<int> left_indices, right_indices;
    for (int idx : indices) {
        if (X[idx * X.shape()[1] + best_feature] <= best_threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }
    
    // Check if split is valid
    if (static_cast<int>(left_indices.size()) < min_samples_leaf_ || static_cast<int>(right_indices.size()) < min_samples_leaf_) {
        // Create leaf node
        node.is_leaf = true;
        node.feature = -1;
        node.threshold = 0.0;
        node.left_child = -1;
        node.right_child = -1;
        
        // Calculate mean value
        double sum = 0.0;
        for (int idx : indices) {
            sum += y[idx];
        }
        node.prediction = sum / indices.size();
        
        tree_.push_back(node);
        return node_id;
    }
    
    // Create internal node
    node.is_leaf = false;
    node.feature = best_feature;
    node.threshold = best_threshold;
    node.prediction = 0.0;
    
    tree_.push_back(node);
    
    // Recursively build children
    node.left_child = build_tree(X, y, left_indices, depth + 1);
    node.right_child = build_tree(X, y, right_indices, depth + 1);
    
    // Update the node with children
    tree_[node_id].left_child = node.left_child;
    tree_[node_id].right_child = node.right_child;
    
    return node_id;
}

double DecisionTreeRegressor::find_best_split(const Tensor& X, const Tensor& y, 
                                            const std::vector<int>& indices,
                                            int& best_feature, double& best_threshold) {
    [[maybe_unused]] int n_features = X.shape()[1];
    double best_impurity = std::numeric_limits<double>::max();
    best_feature = -1;
    best_threshold = 0.0;
    
    for (int feature = 0; feature < n_features; ++feature) {
        // Get unique values for this feature
        std::set<double> unique_values;
        for (int idx : indices) {
            unique_values.insert(X[idx * n_features + feature]);
        }
        
        // Try each unique value as threshold
        for (double threshold : unique_values) {
            std::vector<int> left_indices, right_indices;
            for (int idx : indices) {
                if (X[idx * n_features + feature] <= threshold) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }
            
            if (static_cast<int>(left_indices.size()) < min_samples_leaf_ || static_cast<int>(right_indices.size()) < min_samples_leaf_) {
                continue;
            }
            
            // Calculate weighted impurity
            double left_impurity = calculate_mse(y, left_indices);
            double right_impurity = calculate_mse(y, right_indices);
            double weighted_impurity = (left_indices.size() * left_impurity + 
                                      right_indices.size() * right_impurity) / indices.size();
            
            if (weighted_impurity < best_impurity) {
                best_impurity = weighted_impurity;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }
    
    return (best_feature == -1) ? -1 : best_impurity;
}

double DecisionTreeRegressor::calculate_mse(const Tensor& y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    
    double mean = 0.0;
    for (int idx : indices) {
        mean += y[idx];
    }
    mean /= indices.size();
    
    double mse = 0.0;
    for (int idx : indices) {
        double diff = y[idx] - mean;
        mse += diff * diff;
    }
    
    return mse / indices.size();
}

double DecisionTreeRegressor::calculate_mae(const Tensor& y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    
    double mean = 0.0;
    for (int idx : indices) {
        mean += y[idx];
    }
    mean /= indices.size();
    
    double mae = 0.0;
    for (int idx : indices) {
        mae += std::abs(y[idx] - mean);
    }
    
    return mae / indices.size();
}

Tensor DecisionTreeRegressor::predict(const Tensor& X) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    [[maybe_unused]] int n_samples = X.shape()[0];
    Tensor predictions(std::vector<size_t>{static_cast<size_t>(n_samples)});
    
    for (int i = 0; i < n_samples; ++i) {
        Tensor sample = X.slice({{i, i+1}}).flatten();
        predictions[i] = predict_sample(sample);
    }
    
    return predictions;
}

double DecisionTreeRegressor::predict_sample(const Tensor& sample) const {
    int node_id = 0;
    
    while (!tree_[node_id].is_leaf) {
        int feature = tree_[node_id].feature;
        double threshold = tree_[node_id].threshold;
        
        if (sample[feature] <= threshold) {
            node_id = tree_[node_id].left_child;
        } else {
            node_id = tree_[node_id].right_child;
        }
    }
    
    return tree_[node_id].prediction;
}

double DecisionTreeRegressor::score(const Tensor& X, const Tensor& y) const {
    Tensor predictions = predict(X);
    return metrics::r2_score(y, predictions);
}

} // namespace sklearn
} // namespace tensorcore
