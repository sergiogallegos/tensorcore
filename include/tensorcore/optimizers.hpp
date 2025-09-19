#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace tensorcore {

/**
 * @brief Optimization algorithms for machine learning
 * 
 * This module provides various optimization algorithms commonly used in machine learning,
 * including SGD variants, adaptive methods, and specialized optimizers.
 */

// Base optimizer class
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual void add_parameter(const Tensor& parameter) = 0;
    virtual void add_parameters(const std::vector<Tensor>& parameters) = 0;
    virtual void set_learning_rate(double lr) = 0;
    virtual double get_learning_rate() const = 0;
    virtual std::string get_name() const = 0;
};

// Stochastic Gradient Descent (SGD)
class SGD : public Optimizer {
public:
    SGD(double learning_rate = 0.01, double momentum = 0.0, double dampening = 0.0, 
        double weight_decay = 0.0, bool nesterov = false);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> velocities_;
};

// Adam optimizer
class Adam : public Optimizer {
public:
    Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, 
         double eps = 1e-8, double weight_decay = 0.0);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    int step_count_;
};

// AdamW optimizer
class AdamW : public Optimizer {
public:
    AdamW(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, 
          double eps = 1e-8, double weight_decay = 0.01);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    int step_count_;
};

// RMSprop optimizer
class RMSprop : public Optimizer {
public:
    RMSprop(double learning_rate = 0.01, double alpha = 0.99, double eps = 1e-8, 
            double weight_decay = 0.0, double momentum = 0.0);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double alpha_;
    double eps_;
    double weight_decay_;
    double momentum_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> squared_gradients_;
    std::vector<Tensor> velocities_;
};

// Adagrad optimizer
class Adagrad : public Optimizer {
public:
    Adagrad(double learning_rate = 0.01, double eps = 1e-10, double weight_decay = 0.0);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double eps_;
    double weight_decay_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> squared_gradients_;
};

// Adadelta optimizer
class Adadelta : public Optimizer {
public:
    Adadelta(double learning_rate = 1.0, double rho = 0.9, double eps = 1e-6, 
             double weight_decay = 0.0);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double rho_;
    double eps_;
    double weight_decay_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> squared_gradients_;
    std::vector<Tensor> squared_updates_;
};

// Adamax optimizer
class Adamax : public Optimizer {
public:
    Adamax(double learning_rate = 0.002, double beta1 = 0.9, double beta2 = 0.999, 
           double eps = 1e-8, double weight_decay = 0.0);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    int step_count_;
};

// Nadam optimizer
class Nadam : public Optimizer {
public:
    Nadam(double learning_rate = 0.002, double beta1 = 0.9, double beta2 = 0.999, 
          double eps = 1e-8, double weight_decay = 0.0);
    
    void step() override;
    void zero_grad() override;
    void add_parameter(const Tensor& parameter) override;
    void add_parameters(const std::vector<Tensor>& parameters) override;
    void set_learning_rate(double lr) override;
    double get_learning_rate() const override;
    std::string get_name() const override;
    
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    int step_count_;
};

// Learning rate schedulers
class LearningRateScheduler {
public:
    virtual ~LearningRateScheduler() = default;
    virtual double get_learning_rate(int epoch, double base_lr) = 0;
    virtual std::string get_name() const = 0;
};

class StepLR : public LearningRateScheduler {
public:
    StepLR(double step_size, double gamma = 0.1);
    double get_learning_rate(int epoch, double base_lr) override;
    std::string get_name() const override;
    
private:
    double step_size_;
    double gamma_;
};

class ExponentialLR : public LearningRateScheduler {
public:
    ExponentialLR(double gamma);
    double get_learning_rate(int epoch, double base_lr) override;
    std::string get_name() const override;
    
private:
    double gamma_;
};

class CosineAnnealingLR : public LearningRateScheduler {
public:
    CosineAnnealingLR(int T_max, double eta_min = 0.0);
    double get_learning_rate(int epoch, double base_lr) override;
    std::string get_name() const override;
    
private:
    int T_max_;
    double eta_min_;
};

// Utility functions
std::unique_ptr<Optimizer> create_optimizer(const std::string& name, double learning_rate = 0.01);
std::vector<std::string> get_available_optimizers();
std::unique_ptr<LearningRateScheduler> create_scheduler(const std::string& name, 
                                                        const std::vector<double>& params);
std::vector<std::string> get_available_schedulers();

} // namespace tensorcore
