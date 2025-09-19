#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

namespace tensorcore {

/**
 * @brief Automatic differentiation system
 * 
 * This module provides automatic differentiation capabilities for computing
 * gradients of complex functions, enabling backpropagation in neural networks.
 */

// Forward declaration
class ComputationalGraph;

// Node in the computational graph
class GraphNode {
public:
    GraphNode(const Tensor& tensor, const std::vector<std::shared_ptr<GraphNode>>& inputs = {},
              const std::function<Tensor(const std::vector<Tensor>&)>& forward_func = nullptr,
              const std::function<std::vector<Tensor>(const Tensor&, const std::vector<Tensor>&)>& backward_func = nullptr);
    
    const Tensor& get_tensor() const { return tensor_; }
    Tensor& get_tensor() { return tensor_; }
    const std::vector<std::shared_ptr<GraphNode>>& get_inputs() const { return inputs_; }
    const std::function<Tensor(const std::vector<Tensor>&)>& get_forward_func() const { return forward_func_; }
    const std::function<std::vector<Tensor>(const Tensor&, const std::vector<Tensor>&)>& get_backward_func() const { return backward_func_; }
    
    void set_gradient(const Tensor& grad) { gradient_ = grad; }
    const Tensor& get_gradient() const { return gradient_; }
    Tensor& get_gradient() { return gradient_; }
    
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    void add_gradient(const Tensor& grad);
    void zero_grad();
    
private:
    Tensor tensor_;
    std::vector<std::shared_ptr<GraphNode>> inputs_;
    std::function<Tensor(const std::vector<Tensor>&)> forward_func_;
    std::function<std::vector<Tensor>(const Tensor&, const std::vector<Tensor>&)> backward_func_;
    Tensor gradient_;
    bool requires_grad_;
};

// Computational graph for automatic differentiation
class ComputationalGraph {
public:
    ComputationalGraph();
    ~ComputationalGraph() = default;
    
    // Create nodes
    std::shared_ptr<GraphNode> create_node(const Tensor& tensor, 
                                          const std::vector<std::shared_ptr<GraphNode>>& inputs = {},
                                          const std::function<Tensor(const std::vector<Tensor>&)>& forward_func = nullptr,
                                          const std::function<std::vector<Tensor>(const Tensor&, const std::vector<Tensor>&)>& backward_func = nullptr);
    
    // Operations that create new nodes
    std::shared_ptr<GraphNode> add(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b);
    std::shared_ptr<GraphNode> subtract(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b);
    std::shared_ptr<GraphNode> multiply(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b);
    std::shared_ptr<GraphNode> divide(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b);
    std::shared_ptr<GraphNode> power(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b);
    std::shared_ptr<GraphNode> matmul(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b);
    std::shared_ptr<GraphNode> transpose(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> sum(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> sum(const std::shared_ptr<GraphNode>& a, int axis);
    std::shared_ptr<GraphNode> mean(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> mean(const std::shared_ptr<GraphNode>& a, int axis);
    std::shared_ptr<GraphNode> max(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> max(const std::shared_ptr<GraphNode>& a, int axis);
    std::shared_ptr<GraphNode> min(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> min(const std::shared_ptr<GraphNode>& a, int axis);
    std::shared_ptr<GraphNode> abs(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> sqrt(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> exp(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> log(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> sin(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> cos(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> tanh(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> sigmoid(const std::shared_ptr<GraphNode>& a);
    std::shared_ptr<GraphNode> relu(const std::shared_ptr<GraphNode>& a);
    
    // Scalar operations
    std::shared_ptr<GraphNode> add_scalar(const std::shared_ptr<GraphNode>& a, double scalar);
    std::shared_ptr<GraphNode> subtract_scalar(const std::shared_ptr<GraphNode>& a, double scalar);
    std::shared_ptr<GraphNode> multiply_scalar(const std::shared_ptr<GraphNode>& a, double scalar);
    std::shared_ptr<GraphNode> divide_scalar(const std::shared_ptr<GraphNode>& a, double scalar);
    std::shared_ptr<GraphNode> power_scalar(const std::shared_ptr<GraphNode>& a, double scalar);
    
    // Backpropagation
    void backward(const std::shared_ptr<GraphNode>& node);
    void zero_grad();
    
    // Graph management
    void clear();
    size_t size() const { return nodes_.size(); }
    
private:
    std::vector<std::shared_ptr<GraphNode>> nodes_;
    std::unordered_map<std::shared_ptr<GraphNode>, std::vector<std::shared_ptr<GraphNode>>> adj_list_;
    
    void build_adjacency_list();
    void topological_sort(std::vector<std::shared_ptr<GraphNode>>& sorted_nodes);
    void compute_gradients(const std::vector<std::shared_ptr<GraphNode>>& sorted_nodes);
};

// Global computational graph
extern ComputationalGraph global_graph;

// Convenience functions for creating nodes
std::shared_ptr<GraphNode> variable(const Tensor& tensor, bool requires_grad = true);
std::shared_ptr<GraphNode> constant(const Tensor& tensor);

// Gradient computation functions
Tensor compute_gradient(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                       const std::vector<Tensor>& inputs,
                       const std::vector<int>& input_indices = {});

Tensor compute_hessian(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                      const std::vector<Tensor>& inputs,
                      int input_index);

// Higher-order derivatives
std::vector<Tensor> compute_jacobian(const std::function<std::vector<Tensor>(const std::vector<Tensor>&)>& func,
                                    const std::vector<Tensor>& inputs);

Tensor compute_laplacian(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                        const std::vector<Tensor>& inputs);

// Gradient checking
bool check_gradient(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                   const std::vector<Tensor>& inputs,
                   double eps = 1e-5,
                   double tolerance = 1e-4);

// Gradient clipping
void clip_gradients(std::vector<Tensor>& gradients, double max_norm);
void clip_gradients(std::vector<Tensor>& gradients, double min_val, double max_val);

// Gradient accumulation
void accumulate_gradients(std::vector<Tensor>& target, const std::vector<Tensor>& source);
void scale_gradients(std::vector<Tensor>& gradients, double scale);

// Memory management for gradients
void clear_gradients(std::vector<Tensor>& gradients);
void detach_gradients(std::vector<Tensor>& gradients);

// Utility functions
std::vector<Tensor> get_parameters(const std::vector<std::shared_ptr<GraphNode>>& nodes);
std::vector<Tensor> get_gradients(const std::vector<std::shared_ptr<GraphNode>>& nodes);
void zero_gradients(const std::vector<std::shared_ptr<GraphNode>>& nodes);

// Debugging and visualization
void print_graph(const std::shared_ptr<GraphNode>& node);
void save_graph(const std::shared_ptr<GraphNode>& node, const std::string& filename);
std::shared_ptr<GraphNode> load_graph(const std::string& filename);

} // namespace tensorcore
