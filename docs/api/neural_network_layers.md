# Neural Network Layers

This document provides comprehensive documentation for all neural network layers in TensorCore.

## Table of Contents

1. [Linear Layers](#linear-layers)
2. [Convolutional Layers](#convolutional-layers)
3. [Pooling Layers](#pooling-layers)
4. [Recurrent Layers](#recurrent-layers)
5. [Normalization Layers](#normalization-layers)
6. [Dropout Layers](#dropout-layers)
7. [Layer Properties](#layer-properties)
8. [Performance Considerations](#performance-considerations)

## Linear Layers

### `Linear`

```cpp
/**
 * @brief Linear (fully connected) layer
 * 
 * @details This class implements a linear transformation layer, also known as
 * a fully connected layer. The operation is mathematically defined as:
 * 
 *     y = xW^T + b
 * 
 * where x is the input, W is the weight matrix, b is the bias vector, and y
 * is the output. This layer is commonly used in neural networks for
 * learning linear relationships between inputs and outputs.
 * 
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @param bias Whether to include bias term (default: true)
 * @param weight_init Weight initialization method (default: "xavier")
 * @param bias_init Bias initialization method (default: "zeros")
 * 
 * @example
 * ```cpp
 * // Basic linear layer
 * Linear layer(784, 128);  // 784 inputs, 128 outputs
 * 
 * // Linear layer without bias
 * Linear layer(784, 128, false);
 * 
 * // Custom initialization
 * Linear layer(784, 128, true, "he", "zeros");
 * 
 * // Forward pass
 * Tensor input = {1, 2, 3, 4};  // (4,)
 * Tensor output = layer.forward(input);  // (128,)
 * 
 * // In neural network
 * class MLP {
 * public:
 *     Linear fc1{784, 128};
 *     Linear fc2{128, 64};
 *     Linear fc3{64, 10};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(fc1.forward(x));
 *         x = relu(fc2.forward(x));
 *         return fc3.forward(x);
 *     }
 * };
 * ```
 * 
 * @see Conv2d, LSTM, BatchNorm
 * @since 1.0.0
 */
class Linear {
public:
    Linear(int in_features, int out_features, bool bias = true, 
           const std::string& weight_init = "xavier", 
           const std::string& bias_init = "zeros");
    
    Tensor forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

### `Bilinear`

```cpp
/**
 * @brief Bilinear layer
 * 
 * @details This class implements a bilinear transformation layer, which
 * computes a bilinear function of two inputs. The operation is mathematically
 * defined as:
 * 
 *     y = x1^T W x2 + b
 * 
 * where x1 and x2 are the two inputs, W is the weight matrix, b is the bias
 * vector, and y is the output. This layer is commonly used in attention
 * mechanisms and graph neural networks.
 * 
 * @param in1_features Number of features in first input
 * @param in2_features Number of features in second input
 * @param out_features Number of output features
 * @param bias Whether to include bias term (default: true)
 * 
 * @example
 * ```cpp
 * // Basic bilinear layer
 * Bilinear layer(128, 64, 32);  // 128 and 64 inputs, 32 outputs
 * 
 * // Forward pass
 * Tensor x1 = {1, 2, 3, 4};  // (4,)
 * Tensor x2 = {5, 6, 7, 8};  // (4,)
 * Tensor output = layer.forward(x1, x2);  // (32,)
 * 
 * // In attention mechanism
 * Tensor query = {1, 2, 3, 4};  // (4,)
 * Tensor key = {5, 6, 7, 8};    // (4,)
 * Tensor attention = layer.forward(query, key);  // (32,)
 * ```
 * 
 * @see Linear, Conv2d, LSTM
 * @since 1.0.0
 */
class Bilinear {
public:
    Bilinear(int in1_features, int in2_features, int out_features, bool bias = true);
    
    Tensor forward(const Tensor& input1, const Tensor& input2);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

## Convolutional Layers

### `Conv2d`

```cpp
/**
 * @brief 2D Convolutional layer
 * 
 * @details This class implements a 2D convolutional layer, which applies
 * convolution operations to 2D input data. The operation is mathematically
 * defined as:
 * 
 *     y[i,j] = Σ(m=0 to M-1) Σ(n=0 to N-1) x[i+m,j+n] * w[m,n] + b
 * 
 * where x is the input, w is the kernel, b is the bias, and y is the output.
 * This layer is commonly used in computer vision tasks for feature extraction.
 * 
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_size Size of the convolution kernel
 * @param stride Stride of the convolution (default: 1)
 * @param padding Padding added to input (default: 0)
 * @param dilation Dilation of the convolution (default: 1)
 * @param groups Number of blocked connections (default: 1)
 * @param bias Whether to include bias term (default: true)
 * 
 * @example
 * ```cpp
 * // Basic 2D convolution
 * Conv2d conv1(3, 64, 3);  // 3 input channels, 64 output channels, 3x3 kernel
 * 
 * // Convolution with stride and padding
 * Conv2d conv2(64, 128, 3, 2, 1);  // stride=2, padding=1
 * 
 * // Forward pass
 * Tensor input = zeros({1, 3, 32, 32});  // (batch, channels, height, width)
 * Tensor output = conv1.forward(input);  // (1, 64, 30, 30)
 * 
 * // In CNN
 * class CNN {
 * public:
 *     Conv2d conv1{3, 64, 3, 1, 1};
 *     Conv2d conv2{64, 128, 3, 2, 1};
 *     Conv2d conv3{128, 256, 3, 2, 1};
 *     Linear fc{256 * 4 * 4, 10};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(conv1.forward(x));
 *         x = relu(conv2.forward(x));
 *         x = relu(conv3.forward(x));
 *         x = x.reshape({x.shape[0], -1});  // Flatten
 *         return fc.forward(x);
 *     }
 * };
 * ```
 * 
 * @see Conv1d, Conv3d, MaxPool2d
 * @since 1.0.0
 */
class Conv2d {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0, int dilation = 1, 
           int groups = 1, bool bias = true);
    
    Tensor forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

### `Conv1d`

```cpp
/**
 * @brief 1D Convolutional layer
 * 
 * @details This class implements a 1D convolutional layer, which applies
 * convolution operations to 1D input data. This layer is commonly used
 * for time series analysis and natural language processing.
 * 
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_size Size of the convolution kernel
 * @param stride Stride of the convolution (default: 1)
 * @param padding Padding added to input (default: 0)
 * @param dilation Dilation of the convolution (default: 1)
 * @param groups Number of blocked connections (default: 1)
 * @param bias Whether to include bias term (default: true)
 * 
 * @example
 * ```cpp
 * // Basic 1D convolution
 * Conv1d conv1(1, 64, 3);  // 1 input channel, 64 output channels, 3x1 kernel
 * 
 * // Forward pass
 * Tensor input = zeros({1, 1, 100});  // (batch, channels, length)
 * Tensor output = conv1.forward(input);  // (1, 64, 98)
 * 
 * // In time series model
 * class TimeSeriesCNN {
 * public:
 *     Conv1d conv1{1, 64, 3, 1, 1};
 *     Conv1d conv2{64, 128, 3, 2, 1};
 *     Linear fc{128 * 25, 1};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(conv1.forward(x));
 *         x = relu(conv2.forward(x));
 *         x = x.reshape({x.shape[0], -1});  // Flatten
 *         return fc.forward(x);
 *     }
 * };
 * ```
 * 
 * @see Conv2d, Conv3d, MaxPool1d
 * @since 1.0.0
 */
class Conv1d {
public:
    Conv1d(int in_channels, int out_channels, int kernel_size, 
           int stride = 1, int padding = 0, int dilation = 1, 
           int groups = 1, bool bias = true);
    
    Tensor forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

### `ConvTranspose2d`

```cpp
/**
 * @brief 2D Transpose Convolutional layer
 * 
 * @details This class implements a 2D transpose convolutional layer, also
 * known as a deconvolutional layer. It performs the inverse operation of
 * a regular convolution, upsampling the input. This layer is commonly used
 * in generative models and image segmentation.
 * 
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_size Size of the convolution kernel
 * @param stride Stride of the convolution (default: 1)
 * @param padding Padding added to input (default: 0)
 * @param output_padding Additional padding added to output (default: 0)
 * @param dilation Dilation of the convolution (default: 1)
 * @param groups Number of blocked connections (default: 1)
 * @param bias Whether to include bias term (default: true)
 * 
 * @example
 * ```cpp
 * // Basic transpose convolution
 * ConvTranspose2d conv_transpose(64, 32, 3);  // 64 input, 32 output, 3x3 kernel
 * 
 * // Forward pass
 * Tensor input = zeros({1, 64, 8, 8});  // (batch, channels, height, width)
 * Tensor output = conv_transpose.forward(input);  // (1, 32, 10, 10)
 * 
 * // In generator network
 * class Generator {
 * public:
 *     Linear fc{100, 256 * 4 * 4};
 *     ConvTranspose2d conv1{256, 128, 4, 2, 1};
 *     ConvTranspose2d conv2{128, 64, 4, 2, 1};
 *     ConvTranspose2d conv3{64, 3, 4, 2, 1};
 *     
 *     Tensor forward(const Tensor& z) {
 *         x = fc.forward(z);
 *         x = x.reshape({z.shape[0], 256, 4, 4});
 *         x = relu(conv1.forward(x));
 *         x = relu(conv2.forward(x));
 *         return tanh(conv3.forward(x));
 *     }
 * };
 * ```
 * 
 * @see Conv2d, ConvTranspose1d, MaxPool2d
 * @since 1.0.0
 */
class ConvTranspose2d {
public:
    ConvTranspose2d(int in_channels, int out_channels, int kernel_size, 
                    int stride = 1, int padding = 0, int output_padding = 0, 
                    int dilation = 1, int groups = 1, bool bias = true);
    
    Tensor forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

## Pooling Layers

### `MaxPool2d`

```cpp
/**
 * @brief 2D Max Pooling layer
 * 
 * @details This class implements a 2D max pooling layer, which reduces
 * the spatial dimensions of the input by taking the maximum value in each
 * pooling window. This layer is commonly used in CNNs for downsampling
 * and reducing computational complexity.
 * 
 * @param kernel_size Size of the pooling window
 * @param stride Stride of the pooling (default: kernel_size)
 * @param padding Padding added to input (default: 0)
 * @param dilation Dilation of the pooling (default: 1)
 * @param return_indices Whether to return max indices (default: false)
 * 
 * @example
 * ```cpp
 * // Basic max pooling
 * MaxPool2d pool1(2);  // 2x2 pooling window
 * 
 * // Max pooling with stride
 * MaxPool2d pool2(2, 2);  // 2x2 window, stride=2
 * 
 * // Forward pass
 * Tensor input = zeros({1, 64, 32, 32});  // (batch, channels, height, width)
 * Tensor output = pool1.forward(input);   // (1, 64, 16, 16)
 * 
 * // In CNN
 * class CNN {
 * public:
 *     Conv2d conv1{3, 64, 3, 1, 1};
 *     MaxPool2d pool1{2, 2};
 *     Conv2d conv2{64, 128, 3, 1, 1};
 *     MaxPool2d pool2{2, 2};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(conv1.forward(x));
 *         x = pool1.forward(x);
 *         x = relu(conv2.forward(x));
 *         x = pool2.forward(x);
 *         return x;
 *     }
 * };
 * ```
 * 
 * @see AvgPool2d, MaxPool1d, Conv2d
 * @since 1.0.0
 */
class MaxPool2d {
public:
    MaxPool2d(int kernel_size, int stride = -1, int padding = 0, 
              int dilation = 1, bool return_indices = false);
    
    Tensor forward(const Tensor& input);
    std::vector<Tensor> parameters() const;
};
```

### `AvgPool2d`

```cpp
/**
 * @brief 2D Average Pooling layer
 * 
 * @details This class implements a 2D average pooling layer, which reduces
 * the spatial dimensions of the input by taking the average value in each
 * pooling window. This layer is commonly used in CNNs for downsampling
 * and is less sensitive to outliers than max pooling.
 * 
 * @param kernel_size Size of the pooling window
 * @param stride Stride of the pooling (default: kernel_size)
 * @param padding Padding added to input (default: 0)
 * @param count_include_pad Whether to include padding in average (default: true)
 * 
 * @example
 * ```cpp
 * // Basic average pooling
 * AvgPool2d pool1(2);  // 2x2 pooling window
 * 
 * // Average pooling with stride
 * AvgPool2d pool2(2, 2);  // 2x2 window, stride=2
 * 
 * // Forward pass
 * Tensor input = zeros({1, 64, 32, 32});  // (batch, channels, height, width)
 * Tensor output = pool1.forward(input);   // (1, 64, 16, 16)
 * 
 * // In CNN
 * class CNN {
 * public:
 *     Conv2d conv1{3, 64, 3, 1, 1};
 *     AvgPool2d pool1{2, 2};
 *     Conv2d conv2{64, 128, 3, 1, 1};
 *     AvgPool2d pool2{2, 2};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(conv1.forward(x));
 *         x = pool1.forward(x);
 *         x = relu(conv2.forward(x));
 *         x = pool2.forward(x);
 *         return x;
 *     }
 * };
 * ```
 * 
 * @see MaxPool2d, AvgPool1d, Conv2d
 * @since 1.0.0
 */
class AvgPool2d {
public:
    AvgPool2d(int kernel_size, int stride = -1, int padding = 0, 
              bool count_include_pad = true);
    
    Tensor forward(const Tensor& input);
    std::vector<Tensor> parameters() const;
};
```

## Recurrent Layers

### `LSTM`

```cpp
/**
 * @brief Long Short-Term Memory layer
 * 
 * @details This class implements an LSTM layer, which is a type of recurrent
 * neural network that can learn long-term dependencies. The LSTM uses gates
 * to control the flow of information and prevent the vanishing gradient problem.
 * 
 * @param input_size Number of input features
 * @param hidden_size Number of hidden units
 * @param num_layers Number of LSTM layers (default: 1)
 * @param bias Whether to include bias terms (default: true)
 * @param batch_first Whether input is (batch, seq, feature) (default: false)
 * @param dropout Dropout probability (default: 0.0)
 * @param bidirectional Whether to use bidirectional LSTM (default: false)
 * 
 * @example
 * ```cpp
 * // Basic LSTM
 * LSTM lstm(128, 64);  // 128 input features, 64 hidden units
 * 
 * // Multi-layer LSTM
 * LSTM lstm(128, 64, 2);  // 2 layers
 * 
 * // Bidirectional LSTM
 * LSTM lstm(128, 64, 1, true, false, 0.0, true);
 * 
 * // Forward pass
 * Tensor input = zeros({10, 32, 128});  // (seq_len, batch, input_size)
 * auto [output, (h_n, c_n)] = lstm.forward(input);
 * // output: (seq_len, batch, hidden_size)
 * // h_n: (num_layers, batch, hidden_size)
 * // c_n: (num_layers, batch, hidden_size)
 * 
 * // In sequence model
 * class SequenceModel {
 * public:
 *     LSTM lstm{128, 64, 2};
 *     Linear fc{64, 10};
 *     
 *     Tensor forward(const Tensor& x) {
 *         auto [output, _] = lstm.forward(x);
 *         return fc.forward(output[-1]);  // Use last output
 *     }
 * };
 * ```
 * 
 * @see GRU, RNN, Linear
 * @since 1.0.0
 */
class LSTM {
public:
    LSTM(int input_size, int hidden_size, int num_layers = 1, bool bias = true, 
         bool batch_first = false, double dropout = 0.0, bool bidirectional = false);
    
    std::pair<Tensor, std::pair<Tensor, Tensor>> forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

### `GRU`

```cpp
/**
 * @brief Gated Recurrent Unit layer
 * 
 * @details This class implements a GRU layer, which is a type of recurrent
 * neural network that uses gating mechanisms similar to LSTM but with fewer
 * parameters. GRU is often more efficient than LSTM while maintaining
 * similar performance.
 * 
 * @param input_size Number of input features
 * @param hidden_size Number of hidden units
 * @param num_layers Number of GRU layers (default: 1)
 * @param bias Whether to include bias terms (default: true)
 * @param batch_first Whether input is (batch, seq, feature) (default: false)
 * @param dropout Dropout probability (default: 0.0)
 * @param bidirectional Whether to use bidirectional GRU (default: false)
 * 
 * @example
 * ```cpp
 * // Basic GRU
 * GRU gru(128, 64);  // 128 input features, 64 hidden units
 * 
 * // Multi-layer GRU
 * GRU gru(128, 64, 2);  // 2 layers
 * 
 * // Bidirectional GRU
 * GRU gru(128, 64, 1, true, false, 0.0, true);
 * 
 * // Forward pass
 * Tensor input = zeros({10, 32, 128});  // (seq_len, batch, input_size)
 * auto [output, h_n] = gru.forward(input);
 * // output: (seq_len, batch, hidden_size)
 * // h_n: (num_layers, batch, hidden_size)
 * 
 * // In sequence model
 * class SequenceModel {
 * public:
 *     GRU gru{128, 64, 2};
 *     Linear fc{64, 10};
 *     
 *     Tensor forward(const Tensor& x) {
 *         auto [output, _] = gru.forward(x);
 *         return fc.forward(output[-1]);  // Use last output
 *     }
 * };
 * ```
 * 
 * @see LSTM, RNN, Linear
 * @since 1.0.0
 */
class GRU {
public:
    GRU(int input_size, int hidden_size, int num_layers = 1, bool bias = true, 
        bool batch_first = false, double dropout = 0.0, bool bidirectional = false);
    
    std::pair<Tensor, Tensor> forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

## Normalization Layers

### `BatchNorm2d`

```cpp
/**
 * @brief 2D Batch Normalization layer
 * 
 * @details This class implements 2D batch normalization, which normalizes
 * the input across the batch dimension. This helps with training stability
 * and can accelerate convergence. The operation is mathematically defined as:
 * 
 *     y = (x - μ) / √(σ² + ε) * γ + β
 * 
 * where μ and σ² are the mean and variance of the batch, γ and β are
 * learnable parameters, and ε is a small constant.
 * 
 * @param num_features Number of input features
 * @param eps Small constant for numerical stability (default: 1e-5)
 * @param momentum Momentum for running statistics (default: 0.1)
 * @param affine Whether to use learnable affine parameters (default: true)
 * @param track_running_stats Whether to track running statistics (default: true)
 * 
 * @example
 * ```cpp
 * // Basic batch normalization
 * BatchNorm2d bn(64);  // 64 input features
 * 
 * // Forward pass
 * Tensor input = zeros({32, 64, 32, 32});  // (batch, channels, height, width)
 * Tensor output = bn.forward(input);       // Same shape
 * 
 * // In CNN
 * class CNN {
 * public:
 *     Conv2d conv1{3, 64, 3, 1, 1};
 *     BatchNorm2d bn1{64};
 *     Conv2d conv2{64, 128, 3, 1, 1};
 *     BatchNorm2d bn2{128};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(bn1.forward(conv1.forward(x)));
 *         x = relu(bn2.forward(conv2.forward(x)));
 *         return x;
 *     }
 * };
 * ```
 * 
 * @see LayerNorm, GroupNorm, Conv2d
 * @since 1.0.0
 */
class BatchNorm2d {
public:
    BatchNorm2d(int num_features, double eps = 1e-5, double momentum = 0.1, 
                bool affine = true, bool track_running_stats = true);
    
    Tensor forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

### `LayerNorm`

```cpp
/**
 * @brief Layer Normalization layer
 * 
 * @details This class implements layer normalization, which normalizes
 * the input across the feature dimension. This is commonly used in
 * transformer models and can be more stable than batch normalization
 * for certain tasks.
 * 
 * @param normalized_shape Shape of the input to normalize
 * @param eps Small constant for numerical stability (default: 1e-5)
 * @param elementwise_affine Whether to use learnable affine parameters (default: true)
 * 
 * @example
 * ```cpp
 * // Basic layer normalization
 * LayerNorm ln({128});  // Normalize over 128 features
 * 
 * // Forward pass
 * Tensor input = zeros({32, 128});  // (batch, features)
 * Tensor output = ln.forward(input);  // Same shape
 * 
 * // In transformer
 * class TransformerBlock {
 * public:
 *     MultiHeadAttention attention;
 *     LayerNorm ln1{128};
 *     LayerNorm ln2{128};
 *     Linear fc1{128, 512};
 *     Linear fc2{512, 128};
 *     
 *     Tensor forward(const Tensor& x) {
 *         Tensor attn_out = attention.forward(x);
 *         x = x + attn_out;
 *         x = ln1.forward(x);
 *         
 *         Tensor fc_out = relu(fc1.forward(x));
 *         fc_out = fc2.forward(fc_out);
 *         x = x + fc_out;
 *         return ln2.forward(x);
 *     }
 * };
 * ```
 * 
 * @see BatchNorm2d, GroupNorm, Linear
 * @since 1.0.0
 */
class LayerNorm {
public:
    LayerNorm(const std::vector<int>& normalized_shape, double eps = 1e-5, 
              bool elementwise_affine = true);
    
    Tensor forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};
```

## Dropout Layers

### `Dropout`

```cpp
/**
 * @brief Dropout layer
 * 
 * @details This class implements dropout, which randomly sets some input
 * elements to zero during training to prevent overfitting. The operation
 * is mathematically defined as:
 * 
 *     y = x * mask / (1 - p)
 * 
 * where mask is a binary mask with probability p of being 1, and p is
 * the dropout probability.
 * 
 * @param p Dropout probability (default: 0.5)
 * @param inplace Whether to perform in-place operation (default: false)
 * 
 * @example
 * ```cpp
 * // Basic dropout
 * Dropout dropout(0.5);  // 50% dropout probability
 * 
 * // Forward pass
 * Tensor input = {1, 2, 3, 4, 5};
 * Tensor output = dropout.forward(input);  // Some elements may be zero
 * 
 * // In neural network
 * class MLP {
 * public:
 *     Linear fc1{784, 128};
 *     Dropout dropout1{0.5};
 *     Linear fc2{128, 64};
 *     Dropout dropout2{0.3};
 *     Linear fc3{64, 10};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(fc1.forward(x));
 *         x = dropout1.forward(x);
 *         x = relu(fc2.forward(x));
 *         x = dropout2.forward(x);
 *         return fc3.forward(x);
 *     }
 * };
 * ```
 * 
 * @see Dropout2d, Linear, Conv2d
 * @since 1.0.0
 */
class Dropout {
public:
    Dropout(double p = 0.5, bool inplace = false);
    
    Tensor forward(const Tensor& input);
    std::vector<Tensor> parameters() const;
};
```

### `Dropout2d`

```cpp
/**
 * @brief 2D Dropout layer
 * 
 * @details This class implements 2D dropout, which randomly sets entire
 * feature maps to zero during training. This is commonly used in CNNs
 * and can be more effective than regular dropout for spatial data.
 * 
 * @param p Dropout probability (default: 0.5)
 * @param inplace Whether to perform in-place operation (default: false)
 * 
 * @example
 * ```cpp
 * // Basic 2D dropout
 * Dropout2d dropout2d(0.5);  // 50% dropout probability
 * 
 * // Forward pass
 * Tensor input = zeros({32, 64, 32, 32});  // (batch, channels, height, width)
 * Tensor output = dropout2d.forward(input);  // Some channels may be zero
 * 
 * // In CNN
 * class CNN {
 * public:
 *     Conv2d conv1{3, 64, 3, 1, 1};
 *     Dropout2d dropout1{0.5};
 *     Conv2d conv2{64, 128, 3, 1, 1};
 *     Dropout2d dropout2{0.3};
 *     
 *     Tensor forward(const Tensor& x) {
 *         x = relu(conv1.forward(x));
 *         x = dropout1.forward(x);
 *         x = relu(conv2.forward(x));
 *         x = dropout2.forward(x);
 *         return x;
 *     }
 * };
 * ```
 * 
 * @see Dropout, Conv2d, BatchNorm2d
 * @since 1.0.0
 */
class Dropout2d {
public:
    Dropout2d(double p = 0.5, bool inplace = false);
    
    Tensor forward(const Tensor& input);
    std::vector<Tensor> parameters() const;
};
```

## Layer Properties

### Mathematical Properties

| Layer | Parameters | Memory | Computation | Gradient Flow |
|-------|------------|--------|-------------|---------------|
| Linear | O(in×out) | O(in×out) | O(in×out) | Good |
| Conv2d | O(in×out×k²) | O(in×out×k²) | O(in×out×k²×H×W) | Good |
| MaxPool2d | 0 | O(1) | O(k²×H×W) | Limited |
| LSTM | O(4×in×hidden) | O(4×in×hidden) | O(4×in×hidden×seq) | Good |
| BatchNorm2d | O(features) | O(features) | O(features×H×W) | Good |
| Dropout | 0 | O(1) | O(input) | Good |

### Initialization Methods

```cpp
// Weight initialization
class WeightInit {
public:
    static void xavier_uniform(Tensor& weight, double gain = 1.0);
    static void xavier_normal(Tensor& weight, double gain = 1.0);
    static void kaiming_uniform(Tensor& weight, double a = 0.0, const std::string& mode = "fan_in");
    static void kaiming_normal(Tensor& weight, double a = 0.0, const std::string& mode = "fan_in");
    static void orthogonal(Tensor& weight, double gain = 1.0);
    static void sparse(Tensor& weight, double sparsity, double std = 0.01);
};

// Usage
Tensor weight = zeros({128, 64});
WeightInit::xavier_uniform(weight);
WeightInit::kaiming_normal(weight, 0.0, "fan_in");
```

### Layer Composition

```cpp
// Sequential container
class Sequential {
public:
    Sequential(std::initializer_list<Layer*> layers);
    
    Tensor forward(const Tensor& input);
    void reset_parameters();
    std::vector<Tensor> parameters() const;
};

// Usage
Sequential model({
    new Conv2d(3, 64, 3, 1, 1),
    new BatchNorm2d(64),
    new ReLU(),
    new MaxPool2d(2, 2),
    new Conv2d(64, 128, 3, 1, 1),
    new BatchNorm2d(128),
    new ReLU(),
    new MaxPool2d(2, 2),
    new Flatten(),
    new Linear(128 * 8 * 8, 512),
    new Dropout(0.5),
    new Linear(512, 10)
});
```

## Performance Considerations

### Computational Complexity

| Layer | Forward | Backward | Memory |
|-------|---------|----------|--------|
| Linear | O(in×out) | O(in×out) | O(in×out) |
| Conv2d | O(in×out×k²×H×W) | O(in×out×k²×H×W) | O(in×out×k²) |
| MaxPool2d | O(k²×H×W) | O(k²×H×W) | O(1) |
| LSTM | O(4×in×hidden×seq) | O(4×in×hidden×seq) | O(4×in×hidden) |
| BatchNorm2d | O(features×H×W) | O(features×H×W) | O(features) |
| Dropout | O(input) | O(input) | O(1) |

### Memory Usage

- **Linear Layers**: Memory usage scales with input and output dimensions
- **Convolutional Layers**: Memory usage scales with kernel size and number of channels
- **Recurrent Layers**: Memory usage scales with hidden size and sequence length
- **Normalization Layers**: Memory usage scales with feature dimensions
- **Dropout Layers**: Minimal memory usage

### Best Practices

```cpp
// Good: Use appropriate layer for the task
// For computer vision
Conv2d conv(3, 64, 3, 1, 1);
BatchNorm2d bn(64);
ReLU activation;
MaxPool2d pool(2, 2);

// For natural language processing
Embedding embedding(vocab_size, 128);
LSTM lstm(128, 64, 2);
LayerNorm ln({64});
Dropout dropout(0.5);

// Good: Use proper initialization
Linear fc(784, 128);
WeightInit::xavier_uniform(fc.weight());
WeightInit::zeros(fc.bias());

// Good: Use appropriate activation functions
Tensor x = conv.forward(input);
x = bn.forward(x);
x = relu(x);  // ReLU for hidden layers
x = pool.forward(x);

// Good: Use dropout for regularization
Tensor x = fc1.forward(x);
x = relu(x);
x = dropout.forward(x);  // Dropout after activation
x = fc2.forward(x);
```

## Common Patterns

### CNN Architecture

```cpp
class CNN {
public:
    Conv2d conv1{3, 64, 3, 1, 1};
    BatchNorm2d bn1{64};
    ReLU relu1;
    MaxPool2d pool1{2, 2};
    
    Conv2d conv2{64, 128, 3, 1, 1};
    BatchNorm2d bn2{128};
    ReLU relu2;
    MaxPool2d pool2{2, 2};
    
    Conv2d conv3{128, 256, 3, 1, 1};
    BatchNorm2d bn3{256};
    ReLU relu3;
    MaxPool2d pool3{2, 2};
    
    Flatten flatten;
    Linear fc1{256 * 4 * 4, 512};
    Dropout dropout{0.5};
    Linear fc2{512, 10};
    
    Tensor forward(const Tensor& x) {
        x = relu1.forward(bn1.forward(conv1.forward(x)));
        x = pool1.forward(x);
        
        x = relu2.forward(bn2.forward(conv2.forward(x)));
        x = pool2.forward(x);
        
        x = relu3.forward(bn3.forward(conv3.forward(x)));
        x = pool3.forward(x);
        
        x = flatten.forward(x);
        x = relu1.forward(fc1.forward(x));
        x = dropout.forward(x);
        return fc2.forward(x);
    }
};
```

### Transformer Block

```cpp
class TransformerBlock {
public:
    MultiHeadAttention attention;
    LayerNorm ln1{128};
    LayerNorm ln2{128};
    Linear fc1{128, 512};
    Linear fc2{512, 128};
    Dropout dropout1{0.1};
    Dropout dropout2{0.1};
    
    Tensor forward(const Tensor& x) {
        // Self-attention
        Tensor attn_out = attention.forward(x);
        attn_out = dropout1.forward(attn_out);
        x = x + attn_out;
        x = ln1.forward(x);
        
        // Feed-forward
        Tensor ff_out = fc1.forward(x);
        ff_out = relu(ff_out);
        ff_out = dropout2.forward(ff_out);
        ff_out = fc2.forward(ff_out);
        x = x + ff_out;
        return ln2.forward(x);
    }
};
```

### ResNet Block

```cpp
class ResNetBlock {
public:
    Conv2d conv1{64, 64, 3, 1, 1};
    BatchNorm2d bn1{64};
    Conv2d conv2{64, 64, 3, 1, 1};
    BatchNorm2d bn2{64};
    
    Tensor forward(const Tensor& x) {
        Tensor residual = x;
        
        x = relu(bn1.forward(conv1.forward(x)));
        x = bn2.forward(conv2.forward(x));
        
        x = x + residual;  // Skip connection
        return relu(x);
    }
};
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's neural network layers for their machine learning projects.
