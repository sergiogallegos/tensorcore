#pragma once

#include "tensor.hpp"
#include "activations.hpp"
#include <vector>
#include <memory>
#include <string>

namespace tensorcore {

/**
 * @brief Neural network layers
 * 
 * This module provides various neural network layers commonly used in deep learning,
 * including fully connected, convolutional, and specialized layers.
 */

// Base layer class
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor> get_parameters() const = 0;
    virtual std::vector<Tensor> get_gradients() const = 0;
    virtual void zero_grad() = 0;
    virtual std::string get_name() const = 0;
    virtual bool is_trainable() const = 0;
};

// Fully connected (Dense) layer
class Dense : public Layer {
public:
    Dense(int input_size, int output_size, bool use_bias = true, 
          const std::string& activation = "linear");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
private:
    int input_size_;
    int output_size_;
    bool use_bias_;
    Tensor weights_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    ActivationFunction activation_;
    Tensor last_input_;
    Tensor last_output_;
};

// Convolutional layer
class Conv2D : public Layer {
public:
    Conv2D(int input_channels, int output_channels, int kernel_size, 
           int stride = 1, int padding = 0, bool use_bias = true,
           const std::string& activation = "linear");
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
private:
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool use_bias_;
    Tensor weights_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
    ActivationFunction activation_;
    Tensor last_input_;
    Tensor last_output_;
};

// Max pooling layer
class MaxPool2D : public Layer {
public:
    MaxPool2D(int kernel_size, int stride = 1, int padding = 0);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
private:
    int kernel_size_;
    int stride_;
    int padding_;
    Tensor last_input_;
    Tensor last_output_;
    Tensor max_indices_;
};

// Average pooling layer
class AvgPool2D : public Layer {
public:
    AvgPool2D(int kernel_size, int stride = 1, int padding = 0);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
private:
    int kernel_size_;
    int stride_;
    int padding_;
    Tensor last_input_;
    Tensor last_output_;
};

// Dropout layer
class Dropout : public Layer {
public:
    Dropout(double rate = 0.5, bool training = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    void set_training(bool training);
    
private:
    double rate_;
    bool training_;
    Tensor mask_;
    Tensor last_input_;
};

// Batch normalization layer
class BatchNorm : public Layer {
public:
    BatchNorm(int num_features, double eps = 1e-5, double momentum = 0.1, 
              bool affine = true, bool training = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    void set_training(bool training);
    
private:
    int num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool training_;
    Tensor gamma_;
    Tensor beta_;
    Tensor gamma_grad_;
    Tensor beta_grad_;
    Tensor running_mean_;
    Tensor running_var_;
    Tensor last_input_;
    Tensor last_output_;
    Tensor normalized_input_;
};

// Layer normalization layer
class LayerNorm : public Layer {
public:
    LayerNorm(int normalized_shape, double eps = 1e-5, bool elementwise_affine = true);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
private:
    int normalized_shape_;
    double eps_;
    bool elementwise_affine_;
    Tensor gamma_;
    Tensor beta_;
    Tensor gamma_grad_;
    Tensor beta_grad_;
    Tensor last_input_;
    Tensor last_output_;
    Tensor normalized_input_;
};

// Recurrent layer (LSTM)
class LSTM : public Layer {
public:
    LSTM(int input_size, int hidden_size, int num_layers = 1, 
         bool bidirectional = false, double dropout = 0.0);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool bidirectional_;
    double dropout_;
    std::vector<Tensor> parameters_;
    std::vector<Tensor> gradients_;
    Tensor last_input_;
    Tensor last_output_;
    Tensor last_hidden_;
    Tensor last_cell_;
};

// Embedding layer
class Embedding : public Layer {
public:
    Embedding(int num_embeddings, int embedding_dim, double padding_idx = -1);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
private:
    int num_embeddings_;
    int embedding_dim_;
    double padding_idx_;
    Tensor weights_;
    Tensor weight_grad_;
    Tensor last_input_;
};

// Sequential container for layers
class Sequential : public Layer {
public:
    Sequential() = default;
    Sequential(const std::vector<std::shared_ptr<Layer>>& layers);
    
    void add_layer(std::shared_ptr<Layer> layer);
    void add_layer(const std::string& layer_type, const std::vector<double>& params);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor> get_parameters() const override;
    std::vector<Tensor> get_gradients() const override;
    void zero_grad() override;
    std::string get_name() const override;
    bool is_trainable() const override;
    
    size_t size() const { return layers_.size(); }
    std::shared_ptr<Layer> operator[](size_t index) { return layers_[index]; }
    const std::shared_ptr<Layer> operator[](size_t index) const { return layers_[index]; }
    
private:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::vector<Tensor> layer_outputs_;
};

// Utility functions
std::shared_ptr<Layer> create_layer(const std::string& layer_type, 
                                   const std::vector<double>& params);
std::vector<std::string> get_available_layers();
std::shared_ptr<Sequential> create_mlp(const std::vector<int>& layer_sizes, 
                                       const std::vector<std::string>& activations);
std::shared_ptr<Sequential> create_cnn(const std::vector<int>& conv_channels,
                                       const std::vector<int>& kernel_sizes,
                                       const std::vector<int>& pool_sizes,
                                       int input_channels = 1);

} // namespace tensorcore
