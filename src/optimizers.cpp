#include "tensorcore/optimizers.hpp"
#include "tensorcore/operations.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tensorcore {

// SGD Optimizer Implementation
SGD::SGD(double learning_rate, double momentum, double dampening, double weight_decay, bool nesterov)
    : learning_rate_(learning_rate), momentum_(momentum), dampening_(dampening), 
      weight_decay_(weight_decay), nesterov_(nesterov) {}

void SGD::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& velocity = velocities_[i];
        
        // Apply weight decay
        if (weight_decay_ > 0.0) {
            param = param - learning_rate_ * weight_decay_ * param;
        }
        
        // Update velocity
        if (momentum_ > 0.0) {
            velocity = momentum_ * velocity + (1.0 - dampening_) * param;
            
            if (nesterov_) {
                param = param - learning_rate_ * (velocity + momentum_ * param);
            } else {
                param = param - learning_rate_ * velocity;
            }
        } else {
            param = param - learning_rate_ * param;
        }
    }
}

void SGD::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients by setting them to zero
        // Note: This assumes gradients are stored in the parameter tensors
        // In a real implementation, you'd have separate gradient storage
    }
}

void SGD::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    if (momentum_ > 0.0) {
        velocities_.push_back(Tensor(parameter.shape(), 0.0));
    }
}

void SGD::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void SGD::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double SGD::get_learning_rate() const {
    return learning_rate_;
}

std::string SGD::get_name() const {
    return "SGD";
}

// Adam Optimizer Implementation
Adam::Adam(double learning_rate, double beta1, double beta2, double eps, double weight_decay)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0) {}

void Adam::step() {
    step_count_++;
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& first_moment = first_moments_[i];
        Tensor& second_moment = second_moments_[i];
        
        // Apply weight decay
        if (weight_decay_ > 0.0) {
            param = param - learning_rate_ * weight_decay_ * param;
        }
        
        // Update biased first moment estimate
        first_moment = beta1_ * first_moment + (1.0 - beta1_) * param;
        
        // Update biased second raw moment estimate
        second_moment = beta2_ * second_moment + (1.0 - beta2_) * param * param;
        
        // Compute bias-corrected first moment estimate
        Tensor first_moment_hat = first_moment / (1.0 - std::pow(beta1_, step_count_));
        
        // Compute bias-corrected second raw moment estimate
        Tensor second_moment_hat = second_moment / (1.0 - std::pow(beta2_, step_count_));
        
        // Update parameters
        param = param - learning_rate_ * first_moment_hat / (second_moment_hat.sqrt() + eps_);
    }
}

void Adam::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients
        // Note: This assumes gradients are stored in the parameter tensors
    }
}

void Adam::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    first_moments_.push_back(Tensor(parameter.shape(), 0.0));
    second_moments_.push_back(Tensor(parameter.shape(), 0.0));
}

void Adam::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void Adam::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double Adam::get_learning_rate() const {
    return learning_rate_;
}

std::string Adam::get_name() const {
    return "Adam";
}

// AdamW Optimizer Implementation
AdamW::AdamW(double learning_rate, double beta1, double beta2, double eps, double weight_decay)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0) {}

void AdamW::step() {
    step_count_++;
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& first_moment = first_moments_[i];
        Tensor& second_moment = second_moments_[i];
        
        // Update biased first moment estimate
        first_moment = beta1_ * first_moment + (1.0 - beta1_) * param;
        
        // Update biased second raw moment estimate
        second_moment = beta2_ * second_moment + (1.0 - beta2_) * param * param;
        
        // Compute bias-corrected first moment estimate
        Tensor first_moment_hat = first_moment / (1.0 - std::pow(beta1_, step_count_));
        
        // Compute bias-corrected second raw moment estimate
        Tensor second_moment_hat = second_moment / (1.0 - std::pow(beta2_, step_count_));
        
        // Update parameters with decoupled weight decay
        param = param - learning_rate_ * (first_moment_hat / (second_moment_hat.sqrt() + eps_) + weight_decay_ * param);
    }
}

void AdamW::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients
    }
}

void AdamW::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    first_moments_.push_back(Tensor(parameter.shape(), 0.0));
    second_moments_.push_back(Tensor(parameter.shape(), 0.0));
}

void AdamW::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void AdamW::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double AdamW::get_learning_rate() const {
    return learning_rate_;
}

std::string AdamW::get_name() const {
    return "AdamW";
}

// RMSprop Optimizer Implementation
RMSprop::RMSprop(double learning_rate, double alpha, double eps, double weight_decay, double momentum)
    : learning_rate_(learning_rate), alpha_(alpha), eps_(eps), 
      weight_decay_(weight_decay), momentum_(momentum) {}

void RMSprop::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& squared_grad = squared_gradients_[i];
        
        // Apply weight decay
        if (weight_decay_ > 0.0) {
            param = param - learning_rate_ * weight_decay_ * param;
        }
        
        // Update squared gradients
        squared_grad = alpha_ * squared_grad + (1.0 - alpha_) * param * param;
        
        // Update parameters
        if (momentum_ > 0.0) {
            Tensor& velocity = velocities_[i];
            velocity = momentum_ * velocity + learning_rate_ * param / (squared_grad.sqrt() + eps_);
            param = param - velocity;
        } else {
            param = param - learning_rate_ * param / (squared_grad.sqrt() + eps_);
        }
    }
}

void RMSprop::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients
    }
}

void RMSprop::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    squared_gradients_.push_back(Tensor(parameter.shape(), 0.0));
    if (momentum_ > 0.0) {
        velocities_.push_back(Tensor(parameter.shape(), 0.0));
    }
}

void RMSprop::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void RMSprop::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double RMSprop::get_learning_rate() const {
    return learning_rate_;
}

std::string RMSprop::get_name() const {
    return "RMSprop";
}

// Adagrad Optimizer Implementation
Adagrad::Adagrad(double learning_rate, double eps, double weight_decay)
    : learning_rate_(learning_rate), eps_(eps), weight_decay_(weight_decay) {}

void Adagrad::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& squared_grad = squared_gradients_[i];
        
        // Apply weight decay
        if (weight_decay_ > 0.0) {
            param = param - learning_rate_ * weight_decay_ * param;
        }
        
        // Update squared gradients
        squared_grad = squared_grad + param * param;
        
        // Update parameters
        param = param - learning_rate_ * param / (squared_grad.sqrt() + eps_);
    }
}

void Adagrad::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients
    }
}

void Adagrad::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    squared_gradients_.push_back(Tensor(parameter.shape(), 0.0));
}

void Adagrad::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void Adagrad::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double Adagrad::get_learning_rate() const {
    return learning_rate_;
}

std::string Adagrad::get_name() const {
    return "Adagrad";
}

// Adadelta Optimizer Implementation
Adadelta::Adadelta(double learning_rate, double rho, double eps, double weight_decay)
    : learning_rate_(learning_rate), rho_(rho), eps_(eps), weight_decay_(weight_decay) {}

void Adadelta::step() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& squared_grad = squared_gradients_[i];
        Tensor& squared_update = squared_updates_[i];
        
        // Apply weight decay
        if (weight_decay_ > 0.0) {
            param = param - learning_rate_ * weight_decay_ * param;
        }
        
        // Update squared gradients
        squared_grad = rho_ * squared_grad + (1.0 - rho_) * param * param;
        
        // Compute update
        Tensor update = param * (squared_update.sqrt() + eps_) / (squared_grad.sqrt() + eps_);
        
        // Update squared updates
        squared_update = rho_ * squared_update + (1.0 - rho_) * update * update;
        
        // Update parameters
        param = param - learning_rate_ * update;
    }
}

void Adadelta::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients
    }
}

void Adadelta::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    squared_gradients_.push_back(Tensor(parameter.shape(), 0.0));
    squared_updates_.push_back(Tensor(parameter.shape(), 0.0));
}

void Adadelta::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void Adadelta::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double Adadelta::get_learning_rate() const {
    return learning_rate_;
}

std::string Adadelta::get_name() const {
    return "Adadelta";
}

// Adamax Optimizer Implementation
Adamax::Adamax(double learning_rate, double beta1, double beta2, double eps, double weight_decay)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0) {}

void Adamax::step() {
    step_count_++;
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& first_moment = first_moments_[i];
        Tensor& second_moment = second_moments_[i];
        
        // Apply weight decay
        if (weight_decay_ > 0.0) {
            param = param - learning_rate_ * weight_decay_ * param;
        }
        
        // Update biased first moment estimate
        first_moment = beta1_ * first_moment + (1.0 - beta1_) * param;
        
        // Update biased second raw moment estimate (max)
        second_moment = maximum(beta2_ * second_moment, param.abs());
        
        // Compute bias-corrected first moment estimate
        Tensor first_moment_hat = first_moment / (1.0 - std::pow(beta1_, step_count_));
        
        // Update parameters
        param = param - learning_rate_ * first_moment_hat / (second_moment + eps_);
    }
}

void Adamax::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients
    }
}

void Adamax::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    first_moments_.push_back(Tensor(parameter.shape(), 0.0));
    second_moments_.push_back(Tensor(parameter.shape(), 0.0));
}

void Adamax::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void Adamax::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double Adamax::get_learning_rate() const {
    return learning_rate_;
}

std::string Adamax::get_name() const {
    return "Adamax";
}

// Nadam Optimizer Implementation
Nadam::Nadam(double learning_rate, double beta1, double beta2, double eps, double weight_decay)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps), 
      weight_decay_(weight_decay), step_count_(0) {}

void Nadam::step() {
    step_count_++;
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& param = parameters_[i];
        Tensor& first_moment = first_moments_[i];
        Tensor& second_moment = second_moments_[i];
        
        // Apply weight decay
        if (weight_decay_ > 0.0) {
            param = param - learning_rate_ * weight_decay_ * param;
        }
        
        // Update biased first moment estimate
        first_moment = beta1_ * first_moment + (1.0 - beta1_) * param;
        
        // Update biased second raw moment estimate
        second_moment = beta2_ * second_moment + (1.0 - beta2_) * param * param;
        
        // Compute bias-corrected first moment estimate
        Tensor first_moment_hat = first_moment / (1.0 - std::pow(beta1_, step_count_));
        
        // Compute bias-corrected second raw moment estimate
        Tensor second_moment_hat = second_moment / (1.0 - std::pow(beta2_, step_count_));
        
        // Nesterov momentum
        Tensor nesterov_momentum = beta1_ * first_moment_hat + (1.0 - beta1_) * param;
        
        // Update parameters
        param = param - learning_rate_ * nesterov_momentum / (second_moment_hat.sqrt() + eps_);
    }
}

void Nadam::zero_grad() {
    for (auto& param : parameters_) {
        // Zero gradients
    }
}

void Nadam::add_parameter(const Tensor& parameter) {
    parameters_.push_back(parameter);
    first_moments_.push_back(Tensor(parameter.shape(), 0.0));
    second_moments_.push_back(Tensor(parameter.shape(), 0.0));
}

void Nadam::add_parameters(const std::vector<Tensor>& parameters) {
    for (const auto& param : parameters) {
        add_parameter(param);
    }
}

void Nadam::set_learning_rate(double lr) {
    learning_rate_ = lr;
}

double Nadam::get_learning_rate() const {
    return learning_rate_;
}

std::string Nadam::get_name() const {
    return "Nadam";
}

// Learning Rate Schedulers
StepLR::StepLR(double step_size, double gamma) : step_size_(step_size), gamma_(gamma) {}

double StepLR::get_learning_rate(int epoch, double base_lr) {
    return base_lr * std::pow(gamma_, epoch / step_size_);
}

std::string StepLR::get_name() const {
    return "StepLR";
}

ExponentialLR::ExponentialLR(double gamma) : gamma_(gamma) {}

double ExponentialLR::get_learning_rate(int epoch, double base_lr) {
    return base_lr * std::pow(gamma_, epoch);
}

std::string ExponentialLR::get_name() const {
    return "ExponentialLR";
}

CosineAnnealingLR::CosineAnnealingLR(int T_max, double eta_min) : T_max_(T_max), eta_min_(eta_min) {}

double CosineAnnealingLR::get_learning_rate(int epoch, double base_lr) {
    return eta_min_ + (base_lr - eta_min_) * (1 + std::cos(M_PI * epoch / T_max_)) / 2;
}

std::string CosineAnnealingLR::get_name() const {
    return "CosineAnnealingLR";
}

// Utility functions
std::unique_ptr<Optimizer> create_optimizer(const std::string& name, double learning_rate) {
    if (name == "SGD") {
        return std::make_unique<SGD>(learning_rate);
    } else if (name == "Adam") {
        return std::make_unique<Adam>(learning_rate);
    } else if (name == "AdamW") {
        return std::make_unique<AdamW>(learning_rate);
    } else if (name == "RMSprop") {
        return std::make_unique<RMSprop>(learning_rate);
    } else if (name == "Adagrad") {
        return std::make_unique<Adagrad>(learning_rate);
    } else if (name == "Adadelta") {
        return std::make_unique<Adadelta>(learning_rate);
    } else if (name == "Adamax") {
        return std::make_unique<Adamax>(learning_rate);
    } else if (name == "Nadam") {
        return std::make_unique<Nadam>(learning_rate);
    } else {
        throw std::invalid_argument("Unknown optimizer: " + name);
    }
}

std::vector<std::string> get_available_optimizers() {
    return {"SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Nadam"};
}

std::unique_ptr<LearningRateScheduler> create_scheduler(const std::string& name, const std::vector<double>& params) {
    if (name == "StepLR") {
        if (params.size() < 2) {
            throw std::invalid_argument("StepLR requires 2 parameters: step_size, gamma");
        }
        return std::make_unique<StepLR>(params[0], params[1]);
    } else if (name == "ExponentialLR") {
        if (params.size() < 1) {
            throw std::invalid_argument("ExponentialLR requires 1 parameter: gamma");
        }
        return std::make_unique<ExponentialLR>(params[0]);
    } else if (name == "CosineAnnealingLR") {
        if (params.size() < 2) {
            throw std::invalid_argument("CosineAnnealingLR requires 2 parameters: T_max, eta_min");
        }
        return std::make_unique<CosineAnnealingLR>(static_cast<int>(params[0]), params[1]);
    } else {
        throw std::invalid_argument("Unknown scheduler: " + name);
    }
}

std::vector<std::string> get_available_schedulers() {
    return {"StepLR", "ExponentialLR", "CosineAnnealingLR"};
}

} // namespace tensorcore
