#include "tensorcore/layers.hpp"
#include "tensorcore/operations.hpp"
#include "tensorcore/activations.hpp"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace tensorcore {

// Dense (Linear) Layer Implementation
Dense::Dense(int input_size, int output_size, bool use_bias, const std::string& activation)
    : input_size_(input_size), output_size_(output_size), use_bias_(use_bias) {
    
    // Initialize weights with Xavier/Glorot initialization
    double std_dev = std::sqrt(2.0 / (static_cast<double>(input_size) + static_cast<double>(output_size)));
    std::vector<size_t> weight_shape = {static_cast<size_t>(input_size), static_cast<size_t>(output_size)};
    weights_ = Tensor(weight_shape);
    weights_.random_normal(0.0, std_dev);
    
    // Initialize bias if used
    if (use_bias_) {
        std::vector<size_t> bias_shape = {static_cast<size_t>(output_size)};
        bias_ = Tensor(bias_shape, 0.0);
    }
    
    // Initialize gradients
    std::vector<size_t> weight_grad_shape = {static_cast<size_t>(input_size), static_cast<size_t>(output_size)};
    weight_grad_ = Tensor(weight_grad_shape, 0.0);
    if (use_bias_) {
        std::vector<size_t> bias_grad_shape = {static_cast<size_t>(output_size)};
        bias_grad_ = Tensor(bias_grad_shape, 0.0);
    }
    
    // Set activation function
    if (activation == "relu") {
        activation_ = std::make_shared<ActivationFunction>(ReLU);
    } else if (activation == "sigmoid") {
        activation_ = std::make_shared<ActivationFunction>(Sigmoid);
    } else if (activation == "tanh") {
        activation_ = std::make_shared<ActivationFunction>(Tanh);
    } else if (activation == "linear" || activation == "none") {
        activation_ = std::make_shared<ActivationFunction>(Identity);
    } else {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
}

Tensor Dense::forward(const Tensor& input) {
    if (input.shape().size() != 2 || input.shape()[1] != input_size_) {
        throw std::invalid_argument("Input shape mismatch for Dense layer");
    }
    
    // Store input for backward pass
    last_input_ = input;
    
    // Debug output
    std::cout << "Input shape: " << input.shape()[0] << "x" << input.shape()[1] << std::endl;
    std::cout << "Weights shape: " << weights_.shape()[0] << "x" << weights_.shape()[1] << std::endl;
    std::cout << "Input dimensions: " << input.shape().size() << std::endl;
    std::cout << "Weights dimensions: " << weights_.shape().size() << std::endl;
    
    // Linear transformation: output = input * weights + bias
    Tensor linear_output = input.matmul(weights_);
    
    if (use_bias_) {
        // Add bias to each sample
        for (size_t i = 0; i < linear_output.shape()[0]; ++i) {
            for (size_t j = 0; j < output_size_; ++j) {
                linear_output({i, j}) += bias_[j];
            }
        }
    }
    
    // Apply activation function
    Tensor output = activation_->forward(linear_output);
    last_output_ = output;
    
    return output;
}

Tensor Dense::backward(const Tensor& grad_output) {
    if (grad_output.shape() != last_output_.shape()) {
        throw std::invalid_argument("Gradient shape mismatch in Dense backward");
    }
    
    // Apply activation gradient
    Tensor activation_grad = activation_->backward(last_output_, grad_output);
    
    // Compute weight gradients: dW = X^T * grad
    weight_grad_ = last_input_.transpose().matmul(activation_grad);
    
    // Compute bias gradients: db = sum(grad, axis=0)
    if (use_bias_) {
        bias_grad_ = activation_grad.sum(0);
    }
    
    // Compute input gradients: dX = grad * W^T
    Tensor input_grad = activation_grad.matmul(weights_.transpose());
    
    return input_grad;
}

std::vector<Tensor> Dense::get_parameters() const {
    std::vector<Tensor> params = {weights_};
    if (use_bias_) {
        params.push_back(bias_);
    }
    return params;
}

std::vector<Tensor> Dense::get_gradients() const {
    std::vector<Tensor> grads = {weight_grad_};
    if (use_bias_) {
        grads.push_back(bias_grad_);
    }
    return grads;
}

void Dense::zero_grad() {
    weight_grad_.fill(0.0);
    if (use_bias_) {
        bias_grad_.fill(0.0);
    }
}

std::string Dense::get_name() const {
    return "Dense";
}

bool Dense::is_trainable() const {
    return true;
}

// Conv2D Layer Implementation
Conv2D::Conv2D(int input_channels, int output_channels, int kernel_size, 
               int stride, int padding, bool use_bias, const std::string& activation)
    : input_channels_(input_channels), output_channels_(output_channels), 
      kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(use_bias) {
    
    // Initialize weights (output_channels, input_channels, kernel_size, kernel_size)
    double std_dev = std::sqrt(2.0 / (static_cast<double>(input_channels) * static_cast<double>(kernel_size) * static_cast<double>(kernel_size)));
    weights_ = Tensor({static_cast<size_t>(output_channels), static_cast<size_t>(input_channels), static_cast<size_t>(kernel_size), static_cast<size_t>(kernel_size)});
    weights_.random_normal(0.0, std_dev);
    
    // Initialize bias if used
    if (use_bias_) {
        bias_ = Tensor({static_cast<size_t>(output_channels)}, 0.0);
    }
    
    // Initialize gradients
    weight_grad_ = Tensor({static_cast<size_t>(output_channels), static_cast<size_t>(input_channels), static_cast<size_t>(kernel_size), static_cast<size_t>(kernel_size)}, 0.0);
    if (use_bias_) {
        bias_grad_ = Tensor({static_cast<size_t>(output_channels)}, 0.0);
    }
    
    // Set activation function
    if (activation == "relu") {
        activation_ = std::make_shared<ActivationFunction>(ReLU);
    } else if (activation == "sigmoid") {
        activation_ = std::make_shared<ActivationFunction>(Sigmoid);
    } else if (activation == "tanh") {
        activation_ = std::make_shared<ActivationFunction>(Tanh);
    } else if (activation == "linear" || activation == "none") {
        activation_ = std::make_shared<ActivationFunction>(Identity);
    } else {
        throw std::invalid_argument("Unknown activation function: " + activation);
    }
}

Tensor Conv2D::forward(const Tensor& input) {
    // TODO: Implement 2D convolution
    // This is a placeholder implementation
    throw std::runtime_error("Conv2D forward pass not yet implemented");
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    // TODO: Implement 2D convolution backward pass
    throw std::runtime_error("Conv2D backward pass not yet implemented");
}

std::vector<Tensor> Conv2D::get_parameters() const {
    std::vector<Tensor> params = {weights_};
    if (use_bias_) {
        params.push_back(bias_);
    }
    return params;
}

std::vector<Tensor> Conv2D::get_gradients() const {
    std::vector<Tensor> grads = {weight_grad_};
    if (use_bias_) {
        grads.push_back(bias_grad_);
    }
    return grads;
}

void Conv2D::zero_grad() {
    weight_grad_.fill(0.0);
    if (use_bias_) {
        bias_grad_.fill(0.0);
    }
}

std::string Conv2D::get_name() const {
    return "Conv2D";
}

bool Conv2D::is_trainable() const {
    return true;
}

// MaxPool2D Layer Implementation
MaxPool2D::MaxPool2D(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    // TODO: Implement 2D max pooling
    throw std::runtime_error("MaxPool2D forward pass not yet implemented");
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    // TODO: Implement 2D max pooling backward pass
    throw std::runtime_error("MaxPool2D backward pass not yet implemented");
}

std::vector<Tensor> MaxPool2D::get_parameters() const {
    return {}; // No parameters
}

std::vector<Tensor> MaxPool2D::get_gradients() const {
    return {}; // No gradients
}

void MaxPool2D::zero_grad() {
    // No gradients to zero
}

std::string MaxPool2D::get_name() const {
    return "MaxPool2D";
}

bool MaxPool2D::is_trainable() const {
    return false;
}

// AvgPool2D Layer Implementation
AvgPool2D::AvgPool2D(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

Tensor AvgPool2D::forward(const Tensor& input) {
    // TODO: Implement 2D average pooling
    throw std::runtime_error("AvgPool2D forward pass not yet implemented");
}

Tensor AvgPool2D::backward(const Tensor& grad_output) {
    // TODO: Implement 2D average pooling backward pass
    throw std::runtime_error("AvgPool2D backward pass not yet implemented");
}

std::vector<Tensor> AvgPool2D::get_parameters() const {
    return {}; // No parameters
}

std::vector<Tensor> AvgPool2D::get_gradients() const {
    return {}; // No gradients
}

void AvgPool2D::zero_grad() {
    // No gradients to zero
}

std::string AvgPool2D::get_name() const {
    return "AvgPool2D";
}

bool AvgPool2D::is_trainable() const {
    return false;
}

// Dropout Layer Implementation
Dropout::Dropout(double rate, bool training)
    : rate_(rate), training_(training) {
    if (rate < 0.0 || rate >= 1.0) {
        throw std::invalid_argument("Dropout rate must be in range [0, 1)");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    last_input_ = input;
    
    if (!training_ || rate_ == 0.0) {
        return input; // No dropout during inference or if rate is 0
    }
    
    // Create dropout mask
    mask_ = Tensor(input.shape());
    mask_.random_uniform(0.0, 1.0);
    
    // Apply dropout: set to 0 with probability rate_, scale by 1/(1-rate) otherwise
    Tensor output = input;
    double scale = 1.0 / (1.0 - rate_);
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (mask_[i] < rate_) {
            output[i] = 0.0;
        } else {
            output[i] *= scale;
        }
    }
    
    return output;
}

Tensor Dropout::backward(const Tensor& grad_output) {
    if (!training_ || rate_ == 0.0) {
        return grad_output; // No dropout during inference
    }
    
    // Apply same mask as forward pass
    Tensor input_grad = grad_output;
    double scale = 1.0 / (1.0 - rate_);
    
    for (size_t i = 0; i < grad_output.size(); ++i) {
        if (mask_[i] < rate_) {
            input_grad[i] = 0.0;
        } else {
            input_grad[i] *= scale;
        }
    }
    
    return input_grad;
}

std::vector<Tensor> Dropout::get_parameters() const {
    return {}; // No parameters
}

std::vector<Tensor> Dropout::get_gradients() const {
    return {}; // No gradients
}

void Dropout::zero_grad() {
    // No gradients to zero
}

std::string Dropout::get_name() const {
    return "Dropout";
}

bool Dropout::is_trainable() const {
    return false;
}

void Dropout::set_training(bool training) {
    training_ = training;
}

// BatchNorm Layer Implementation
BatchNorm::BatchNorm(int num_features, double eps, double momentum, bool affine, bool training)
    : num_features_(num_features), eps_(eps), momentum_(momentum), 
      affine_(affine), training_(training) {
    
    if (affine_) {
        // Initialize scale and shift parameters
        gamma_ = Tensor({static_cast<size_t>(num_features)}, 1.0);
        beta_ = Tensor({static_cast<size_t>(num_features)}, 0.0);
        
        gamma_grad_ = Tensor({static_cast<size_t>(num_features)}, 0.0);
        beta_grad_ = Tensor({static_cast<size_t>(num_features)}, 0.0);
    }
    
    // Initialize running statistics
    running_mean_ = Tensor({static_cast<size_t>(num_features)}, 0.0);
    running_var_ = Tensor({static_cast<size_t>(num_features)}, 1.0);
}

Tensor BatchNorm::forward(const Tensor& input) {
    last_input_ = input;
    
    if (training_) {
        // Training mode: use batch statistics
        Tensor batch_mean = input.mean(0); // Mean over batch dimension
        Tensor batch_var = input.var(0);   // Variance over batch dimension
        
        // Update running statistics
        running_mean_ = momentum_ * running_mean_ + (1.0 - momentum_) * batch_mean;
        running_var_ = momentum_ * running_var_ + (1.0 - momentum_) * batch_var;
        
        // Normalize
        normalized_input_ = (input - batch_mean) / (batch_var + eps_).sqrt();
    } else {
        // Inference mode: use running statistics
        normalized_input_ = (input - running_mean_) / (running_var_ + eps_).sqrt();
    }
    
    if (affine_) {
        // Apply scale and shift
        last_output_ = gamma_ * normalized_input_ + beta_;
    } else {
        last_output_ = normalized_input_;
    }
    
    return last_output_;
}

Tensor BatchNorm::backward(const Tensor& grad_output) {
    if (!affine_) {
        // TODO: Implement backward pass for non-affine BatchNorm
        return grad_output;
    }
    
    // Compute gradients for gamma and beta
    gamma_grad_ = (grad_output * normalized_input_).sum(0);
    beta_grad_ = grad_output.sum(0);
    
    // TODO: Implement full backward pass for BatchNorm
    // This is a simplified version
    Tensor input_grad = grad_output * gamma_;
    
    return input_grad;
}

std::vector<Tensor> BatchNorm::get_parameters() const {
    if (affine_) {
        return {gamma_, beta_};
    }
    return {};
}

std::vector<Tensor> BatchNorm::get_gradients() const {
    if (affine_) {
        return {gamma_grad_, beta_grad_};
    }
    return {};
}

void BatchNorm::zero_grad() {
    if (affine_) {
        gamma_grad_.fill(0.0);
        beta_grad_.fill(0.0);
    }
}

std::string BatchNorm::get_name() const {
    return "BatchNorm";
}

bool BatchNorm::is_trainable() const {
    return affine_;
}

void BatchNorm::set_training(bool training) {
    training_ = training;
}

// LayerNorm Layer Implementation
LayerNorm::LayerNorm(int normalized_shape, double eps, bool elementwise_affine)
    : normalized_shape_(normalized_shape), eps_(eps), elementwise_affine_(elementwise_affine) {
    
    if (elementwise_affine_) {
        gamma_ = Tensor({static_cast<size_t>(normalized_shape)}, 1.0);
        beta_ = Tensor({static_cast<size_t>(normalized_shape)}, 0.0);
        
        gamma_grad_ = Tensor({static_cast<size_t>(normalized_shape)}, 0.0);
        beta_grad_ = Tensor({static_cast<size_t>(normalized_shape)}, 0.0);
    }
}

Tensor LayerNorm::forward(const Tensor& input) {
    last_input_ = input;
    
    // Compute mean and variance over the last dimension
    Tensor mean = input.mean(-1);
    Tensor var = input.var(-1);
    
    // Normalize
    normalized_input_ = (input - mean) / (var + eps_).sqrt();
    
    if (elementwise_affine_) {
        last_output_ = gamma_ * normalized_input_ + beta_;
    } else {
        last_output_ = normalized_input_;
    }
    
    return last_output_;
}

Tensor LayerNorm::backward(const Tensor& grad_output) {
    if (!elementwise_affine_) {
        return grad_output;
    }
    
    // Compute gradients for gamma and beta
    gamma_grad_ = (grad_output * normalized_input_).sum(-1);
    beta_grad_ = grad_output.sum(-1);
    
    // TODO: Implement full backward pass for LayerNorm
    Tensor input_grad = grad_output * gamma_;
    
    return input_grad;
}

std::vector<Tensor> LayerNorm::get_parameters() const {
    if (elementwise_affine_) {
        return {gamma_, beta_};
    }
    return {};
}

std::vector<Tensor> LayerNorm::get_gradients() const {
    if (elementwise_affine_) {
        return {gamma_grad_, beta_grad_};
    }
    return {};
}

void LayerNorm::zero_grad() {
    if (elementwise_affine_) {
        gamma_grad_.fill(0.0);
        beta_grad_.fill(0.0);
    }
}

std::string LayerNorm::get_name() const {
    return "LayerNorm";
}

bool LayerNorm::is_trainable() const {
    return elementwise_affine_;
}

// LSTM Layer Implementation
LSTM::LSTM(int input_size, int hidden_size, int num_layers, bool bidirectional, double dropout)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
      bidirectional_(bidirectional), dropout_(dropout) {
    
    // TODO: Implement LSTM parameter initialization
    // This is a placeholder
}

Tensor LSTM::forward(const Tensor& input) {
    // TODO: Implement LSTM forward pass
    throw std::runtime_error("LSTM forward pass not yet implemented");
}

Tensor LSTM::backward(const Tensor& grad_output) {
    // TODO: Implement LSTM backward pass
    throw std::runtime_error("LSTM backward pass not yet implemented");
}

std::vector<Tensor> LSTM::get_parameters() const {
    return parameters_;
}

std::vector<Tensor> LSTM::get_gradients() const {
    return gradients_;
}

void LSTM::zero_grad() {
    for (auto& grad : gradients_) {
        grad.fill(0.0);
    }
}

std::string LSTM::get_name() const {
    return "LSTM";
}

bool LSTM::is_trainable() const {
    return true;
}

// Embedding Layer Implementation
Embedding::Embedding(int num_embeddings, int embedding_dim, double padding_idx)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim), padding_idx_(padding_idx) {
    
    // Initialize embedding weights
    double std_dev = std::sqrt(1.0 / static_cast<double>(embedding_dim));
    weights_ = Tensor({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)});
    weights_.random_normal(0.0, std_dev);
    
    weight_grad_ = Tensor({static_cast<size_t>(num_embeddings), static_cast<size_t>(embedding_dim)}, 0.0);
}

Tensor Embedding::forward(const Tensor& input) {
    last_input_ = input;
    
    // TODO: Implement embedding lookup
    // This is a placeholder
    Tensor output = Tensor({input.shape()[0], static_cast<size_t>(embedding_dim_)}, 0.0);
    return output;
}

Tensor Embedding::backward(const Tensor& grad_output) {
    // TODO: Implement embedding backward pass
    Tensor input_grad = Tensor(last_input_.shape(), 0.0);
    return input_grad;
}

std::vector<Tensor> Embedding::get_parameters() const {
    return {weights_};
}

std::vector<Tensor> Embedding::get_gradients() const {
    return {weight_grad_};
}

void Embedding::zero_grad() {
    weight_grad_.fill(0.0);
}

std::string Embedding::get_name() const {
    return "Embedding";
}

bool Embedding::is_trainable() const {
    return true;
}

// Sequential Container Implementation
Sequential::Sequential(const std::vector<std::shared_ptr<Layer>>& layers)
    : layers_(layers) {}

void Sequential::add_layer(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
}

void Sequential::add_layer(const std::string& layer_type, const std::vector<double>& params) {
    // TODO: Implement layer creation from string
    throw std::runtime_error("Layer creation from string not yet implemented");
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;
    layer_outputs_.clear();
    layer_outputs_.reserve(layers_.size());
    
    for (auto& layer : layers_) {
        output = layer->forward(output);
        layer_outputs_.push_back(output);
    }
    
    return output;
}

Tensor Sequential::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    
    // Backward pass through layers in reverse order
    for (int i = layers_.size() - 1; i >= 0; --i) {
        grad = layers_[i]->backward(grad);
    }
    
    return grad;
}

std::vector<Tensor> Sequential::get_parameters() const {
    std::vector<Tensor> params;
    for (auto& layer : layers_) {
        auto layer_params = layer->get_parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

std::vector<Tensor> Sequential::get_gradients() const {
    std::vector<Tensor> grads;
    for (auto& layer : layers_) {
        auto layer_grads = layer->get_gradients();
        grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return grads;
}

void Sequential::zero_grad() {
    for (auto& layer : layers_) {
        layer->zero_grad();
    }
}

std::string Sequential::get_name() const {
    return "Sequential";
}

bool Sequential::is_trainable() const {
    return true;
}

// Utility functions
std::shared_ptr<Layer> create_layer(const std::string& layer_type, const std::vector<double>& params) {
    // TODO: Implement layer creation from string
    throw std::runtime_error("Layer creation from string not yet implemented");
}

std::vector<std::string> get_available_layers() {
    return {"Dense", "Conv2D", "MaxPool2D", "AvgPool2D", "Dropout", "BatchNorm", "LayerNorm", "LSTM", "Embedding"};
}

std::shared_ptr<Sequential> create_mlp(const std::vector<int>& layer_sizes, const std::vector<std::string>& activations) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("MLP requires at least 2 layer sizes");
    }
    
    auto mlp = std::make_shared<Sequential>();
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        std::string activation = (i < activations.size()) ? activations[i] : "relu";
        auto layer = std::make_shared<Dense>(layer_sizes[i], layer_sizes[i + 1], true, activation);
        mlp->add_layer(layer);
    }
    
    return mlp;
}

std::shared_ptr<Sequential> create_cnn(const std::vector<int>& conv_channels, const std::vector<int>& kernel_sizes,
                                       const std::vector<int>& pool_sizes, int input_channels) {
    // TODO: Implement CNN creation
    throw std::runtime_error("CNN creation not yet implemented");
}

} // namespace tensorcore
