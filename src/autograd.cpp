#include "tensorcore/autograd.hpp"
#include "tensorcore/operations.hpp"
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <iostream>

namespace tensorcore {

// Global computational graph instance
ComputationalGraph global_graph;

// GraphNode implementation
GraphNode::GraphNode(const Tensor& tensor, 
                     const std::vector<std::shared_ptr<GraphNode>>& inputs,
                     const std::function<Tensor(const std::vector<Tensor>&)>& forward_func,
                     const std::function<std::vector<Tensor>(const Tensor&, const std::vector<Tensor>&)>& backward_func)
    : tensor_(tensor), inputs_(inputs), forward_func_(forward_func), backward_func_(backward_func),
      requires_grad_(tensor.requires_grad()) {
    
    // Initialize gradient tensor with same shape as tensor
    if (requires_grad_) {
        gradient_ = Tensor(tensor_.shape(), 0.0);
    }
}

void GraphNode::add_gradient(const Tensor& grad) {
    if (!requires_grad_) return;
    
    if (gradient_.size() == 0) {
        gradient_ = grad;
    } else {
        // Add gradients (for nodes with multiple outputs)
        for (size_t i = 0; i < gradient_.size(); ++i) {
            gradient_[i] += grad[i];
        }
    }
}

void GraphNode::zero_grad() {
    if (requires_grad_ && gradient_.size() > 0) {
        gradient_.fill(0.0);
    }
}

// ComputationalGraph implementation
ComputationalGraph::ComputationalGraph() {}

std::shared_ptr<GraphNode> ComputationalGraph::create_node(
    const Tensor& tensor,
    const std::vector<std::shared_ptr<GraphNode>>& inputs,
    const std::function<Tensor(const std::vector<Tensor>&)>& forward_func,
    const std::function<std::vector<Tensor>(const Tensor&, const std::vector<Tensor>&)>& backward_func) {
    
    auto node = std::make_shared<GraphNode>(tensor, inputs, forward_func, backward_func);
    nodes_.push_back(node);
    return node;
}

// Forward operations that create new nodes
std::shared_ptr<GraphNode> ComputationalGraph::add(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b) {
    // Forward pass
    Tensor result = a->get_tensor() + b->get_tensor();
    
    // Create backward function
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        return {grad_output, grad_output}; // Both inputs get the same gradient
    };
    
    return create_node(result, {a, b}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::subtract(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b) {
    Tensor result = a->get_tensor() - b->get_tensor();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        return {grad_output, -grad_output}; // Second input gets negative gradient
    };
    
    return create_node(result, {a, b}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::multiply(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b) {
    Tensor result = a->get_tensor() * b->get_tensor();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(x*y) = y, d/dy(x*y) = x
        return {grad_output * inputs[1], grad_output * inputs[0]};
    };
    
    return create_node(result, {a, b}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::divide(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b) {
    Tensor result = a->get_tensor() / b->get_tensor();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(x/y) = 1/y, d/dy(x/y) = -x/y²
        Tensor grad_a = grad_output / inputs[1];
        Tensor grad_b = -grad_output * inputs[0] / (inputs[1] * inputs[1]);
        return {grad_a, grad_b};
    };
    
    return create_node(result, {a, b}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::power(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b) {
    Tensor result = tensorcore::power(a->get_tensor(), b->get_tensor());
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(x^y) = y*x^(y-1), d/dy(x^y) = x^y*ln(x)
        Tensor grad_a = grad_output * inputs[1] * tensorcore::power(inputs[0], inputs[1] - Tensor(inputs[0].shape(), 1.0));
        Tensor grad_b = grad_output * tensorcore::power(inputs[0], inputs[1]) * tensorcore::log(inputs[0]);
        return {grad_a, grad_b};
    };
    
    return create_node(result, {a, b}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::matmul(const std::shared_ptr<GraphNode>& a, const std::shared_ptr<GraphNode>& b) {
    Tensor result = a->get_tensor().matmul(b->get_tensor());
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dA(A*B) = grad_output * B^T, d/dB(A*B) = A^T * grad_output
        Tensor grad_a = grad_output.matmul(inputs[1].transpose());
        Tensor grad_b = inputs[0].transpose().matmul(grad_output);
        return {grad_a, grad_b};
    };
    
    return create_node(result, {a, b}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::transpose(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().transpose();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        return {grad_output.transpose()};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::sum(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().sum();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // Gradient is broadcasted to input shape
        Tensor grad = Tensor(inputs[0].shape(), grad_output[0]);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::sum(const std::shared_ptr<GraphNode>& a, int axis) {
    Tensor result = a->get_tensor().sum(axis);
    
    auto backward_func = [axis](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // Broadcast gradient back to original shape
        Tensor grad = grad_output;
        for (int i = 0; i < axis; ++i) {
            grad = grad.unsqueeze(0);
        }
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::mean(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().mean();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        double scale = 1.0 / static_cast<double>(inputs[0].size());
        Tensor grad = Tensor(inputs[0].shape(), grad_output[0] * scale);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::mean(const std::shared_ptr<GraphNode>& a, int axis) {
    Tensor result = a->get_tensor().mean(axis);
    
    auto backward_func = [axis](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        double scale = 1.0 / static_cast<double>(inputs[0].shape()[axis]);
        Tensor grad = grad_output;
        for (int i = 0; i < axis; ++i) {
            grad = grad.unsqueeze(0);
        }
        grad = grad * scale;
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::max(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().max();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // Find where max occurred and set gradient there
        Tensor grad = Tensor(inputs[0].shape(), 0.0);
        double max_val = inputs[0].max()[0];
        
        for (size_t i = 0; i < inputs[0].size(); ++i) {
            if (std::abs(inputs[0][i] - max_val) < 1e-10) {
                grad[i] = grad_output[0];
            }
        }
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::max(const std::shared_ptr<GraphNode>& a, int axis) {
    Tensor result = a->get_tensor().max(axis);
    
    auto backward_func = [axis](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // TODO: Implement axis-wise max gradient
        Tensor grad = Tensor(inputs[0].shape(), 0.0);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::min(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().min();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        Tensor grad = Tensor(inputs[0].shape(), 0.0);
        double min_val = inputs[0].min()[0];
        
        for (size_t i = 0; i < inputs[0].size(); ++i) {
            if (std::abs(inputs[0][i] - min_val) < 1e-10) {
                grad[i] = grad_output[0];
            }
        }
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::min(const std::shared_ptr<GraphNode>& a, int axis) {
    Tensor result = a->get_tensor().min(axis);
    
    auto backward_func = [axis](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // TODO: Implement axis-wise min gradient
        Tensor grad = Tensor(inputs[0].shape(), 0.0);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::abs(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().abs();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        Tensor grad = grad_output;
        for (size_t i = 0; i < grad.size(); ++i) {
            if (inputs[0][i] < 0) {
                grad[i] = -grad[i];
            }
        }
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::sqrt(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().sqrt();
    
    auto backward_func = [result](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(sqrt(x)) = 1/(2*sqrt(x))
        Tensor grad = grad_output / (2.0 * result);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::exp(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().exp();
    
    auto backward_func = [result](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(exp(x)) = exp(x)
        Tensor grad = grad_output * result;
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::log(const std::shared_ptr<GraphNode>& a) {
    Tensor result = a->get_tensor().log();
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(log(x)) = 1/x
        Tensor grad = grad_output / inputs[0];
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::sin(const std::shared_ptr<GraphNode>& a) {
    Tensor result = tensorcore::sin(a->get_tensor());
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(sin(x)) = cos(x)
        Tensor grad = grad_output * tensorcore::cos(inputs[0]);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::cos(const std::shared_ptr<GraphNode>& a) {
    Tensor result = tensorcore::cos(a->get_tensor());
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(cos(x)) = -sin(x)
        Tensor grad = -grad_output * tensorcore::sin(inputs[0]);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::tanh(const std::shared_ptr<GraphNode>& a) {
    Tensor result = tensorcore::tanh(a->get_tensor());
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(tanh(x)) = 1 - tanh²(x)
        Tensor tanh_val = tensorcore::tanh(inputs[0]);
        Tensor grad = grad_output * (Tensor(inputs[0].shape(), 1.0) - tanh_val * tanh_val);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::sigmoid(const std::shared_ptr<GraphNode>& a) {
    Tensor result = tensorcore::sigmoid(a->get_tensor());
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
        Tensor sigmoid_val = tensorcore::sigmoid(inputs[0]);
        Tensor grad = grad_output * sigmoid_val * (Tensor(inputs[0].shape(), 1.0) - sigmoid_val);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::relu(const std::shared_ptr<GraphNode>& a) {
    Tensor result = tensorcore::relu(a->get_tensor());
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(ReLU(x)) = 1 if x > 0, else 0
        Tensor grad = Tensor(inputs[0].shape(), 0.0);
        for (size_t i = 0; i < inputs[0].size(); ++i) {
            if (inputs[0][i] > 0) {
                grad[i] = grad_output[i];
            }
        }
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

// Scalar operations
std::shared_ptr<GraphNode> ComputationalGraph::add_scalar(const std::shared_ptr<GraphNode>& a, double scalar) {
    Tensor result = a->get_tensor() + scalar;
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        return {grad_output}; // Scalar doesn't contribute to gradient
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::subtract_scalar(const std::shared_ptr<GraphNode>& a, double scalar) {
    Tensor result = a->get_tensor() - scalar;
    
    auto backward_func = [](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        return {grad_output};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::multiply_scalar(const std::shared_ptr<GraphNode>& a, double scalar) {
    Tensor result = a->get_tensor() * scalar;
    
    auto backward_func = [scalar](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        return {grad_output * scalar};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::divide_scalar(const std::shared_ptr<GraphNode>& a, double scalar) {
    Tensor result = a->get_tensor() / scalar;
    
    auto backward_func = [scalar](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        return {grad_output / scalar};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

std::shared_ptr<GraphNode> ComputationalGraph::power_scalar(const std::shared_ptr<GraphNode>& a, double scalar) {
    Tensor result = a->get_tensor().pow(scalar);
    
    auto backward_func = [scalar](const Tensor& grad_output, const std::vector<Tensor>& inputs) -> std::vector<Tensor> {
        // d/dx(x^n) = n*x^(n-1)
        Tensor grad = grad_output * scalar * inputs[0].pow(scalar - 1.0);
        return {grad};
    };
    
    return create_node(result, {a}, nullptr, backward_func);
}

// Backpropagation
void ComputationalGraph::backward(const std::shared_ptr<GraphNode>& node) {
    // Topological sort to get correct order
    std::vector<std::shared_ptr<GraphNode>> sorted_nodes;
    topological_sort(sorted_nodes);
    
    // Initialize gradients
    for (auto& n : nodes_) {
        n->zero_grad();
    }
    
    // Set output gradient to 1
    if (node->requires_grad()) {
        node->set_gradient(Tensor(node->get_tensor().shape(), 1.0));
    }
    
    // Backward pass
    compute_gradients(sorted_nodes);
}

void ComputationalGraph::zero_grad() {
    for (auto& node : nodes_) {
        node->zero_grad();
    }
}

void ComputationalGraph::clear() {
    nodes_.clear();
    adj_list_.clear();
}

void ComputationalGraph::build_adjacency_list() {
    adj_list_.clear();
    
    for (auto& node : nodes_) {
        adj_list_[node] = node->get_inputs();
    }
}

void ComputationalGraph::topological_sort(std::vector<std::shared_ptr<GraphNode>>& sorted_nodes) {
    sorted_nodes.clear();
    std::unordered_set<std::shared_ptr<GraphNode>> visited;
    std::unordered_set<std::shared_ptr<GraphNode>> temp_visited;
    
    std::function<void(std::shared_ptr<GraphNode>)> visit = [&](std::shared_ptr<GraphNode> node) {
        if (temp_visited.find(node) != temp_visited.end()) {
            throw std::runtime_error("Circular dependency detected in computational graph");
        }
        if (visited.find(node) != visited.end()) {
            return;
        }
        
        temp_visited.insert(node);
        for (auto& input : node->get_inputs()) {
            visit(input);
        }
        temp_visited.erase(node);
        visited.insert(node);
        sorted_nodes.push_back(node);
    };
    
    for (auto& node : nodes_) {
        if (visited.find(node) == visited.end()) {
            visit(node);
        }
    }
    
    std::reverse(sorted_nodes.begin(), sorted_nodes.end());
}

void ComputationalGraph::compute_gradients(const std::vector<std::shared_ptr<GraphNode>>& sorted_nodes) {
    for (auto& node : sorted_nodes) {
        if (!node->requires_grad() || node->get_gradient().size() == 0) {
            continue;
        }
        
        // Get input tensors for backward pass
        std::vector<Tensor> input_tensors;
        for (auto& input : node->get_inputs()) {
            input_tensors.push_back(input->get_tensor());
        }
        
        // Compute gradients for inputs
        if (node->get_backward_func()) {
            std::vector<Tensor> input_grads = node->get_backward_func()(node->get_gradient(), input_tensors);
            
            // Add gradients to input nodes
            for (size_t i = 0; i < node->get_inputs().size() && i < input_grads.size(); ++i) {
                if (node->get_inputs()[i]->requires_grad()) {
                    node->get_inputs()[i]->add_gradient(input_grads[i]);
                }
            }
        }
    }
}

// Convenience functions
std::shared_ptr<GraphNode> variable(const Tensor& tensor, bool requires_grad) {
    Tensor tensor_copy = tensor;
    tensor_copy.set_requires_grad(requires_grad);
    return global_graph.create_node(tensor_copy);
}

std::shared_ptr<GraphNode> constant(const Tensor& tensor) {
    return global_graph.create_node(tensor, {}, nullptr, nullptr);
}

// Gradient computation functions
Tensor compute_gradient(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                       const std::vector<Tensor>& inputs,
                       const std::vector<int>& input_indices) {
    // Create variable nodes for inputs
    std::vector<std::shared_ptr<GraphNode>> input_nodes;
    for (size_t i = 0; i < inputs.size(); ++i) {
        bool requires_grad = std::find(input_indices.begin(), input_indices.end(), static_cast<int>(i)) != input_indices.end();
        input_nodes.push_back(variable(inputs[i], requires_grad));
    }
    
    // Forward pass
    std::vector<Tensor> input_tensors;
    for (auto& node : input_nodes) {
        input_tensors.push_back(node->get_tensor());
    }
    Tensor output = func(input_tensors);
    
    // Create output node
    auto output_node = variable(output, true);
    
    // Backward pass
    global_graph.backward(output_node);
    
    // Collect gradients
    std::vector<Tensor> gradients;
    for (int idx : input_indices) {
        if (idx >= 0 && idx < static_cast<int>(input_nodes.size())) {
            gradients.push_back(input_nodes[idx]->get_gradient());
        }
    }
    
    return gradients.empty() ? Tensor({1}, {0.0}) : gradients[0];
}

Tensor compute_hessian(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                      const std::vector<Tensor>& inputs,
                      int input_index) {
    // TODO: Implement Hessian computation
    return Tensor({1}, {0.0});
}

std::vector<Tensor> compute_jacobian(const std::function<std::vector<Tensor>(const std::vector<Tensor>&)>& func,
                                    const std::vector<Tensor>& inputs) {
    // TODO: Implement Jacobian computation
    return {};
}

Tensor compute_laplacian(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                        const std::vector<Tensor>& inputs) {
    // TODO: Implement Laplacian computation
    return Tensor({1}, {0.0});
}

bool check_gradient(const std::function<Tensor(const std::vector<Tensor>&)>& func,
                   const std::vector<Tensor>& inputs,
                   double eps,
                   double tolerance) {
    // TODO: Implement gradient checking
    return true;
}

void clip_gradients(std::vector<Tensor>& gradients, double max_norm) {
    // TODO: Implement gradient clipping
}

void clip_gradients(std::vector<Tensor>& gradients, double min_val, double max_val) {
    // TODO: Implement gradient clipping with min/max values
}

void accumulate_gradients(std::vector<Tensor>& target, const std::vector<Tensor>& source) {
    // TODO: Implement gradient accumulation
}

void scale_gradients(std::vector<Tensor>& gradients, double scale) {
    for (auto& grad : gradients) {
        grad = grad * scale;
    }
}

void clear_gradients(std::vector<Tensor>& gradients) {
    for (auto& grad : gradients) {
        grad.fill(0.0);
    }
}

void detach_gradients(std::vector<Tensor>& gradients) {
    // TODO: Implement gradient detachment
}

std::vector<Tensor> get_parameters(const std::vector<std::shared_ptr<GraphNode>>& nodes) {
    std::vector<Tensor> params;
    for (auto& node : nodes) {
        if (node->requires_grad()) {
            params.push_back(node->get_tensor());
        }
    }
    return params;
}

std::vector<Tensor> get_gradients(const std::vector<std::shared_ptr<GraphNode>>& nodes) {
    std::vector<Tensor> grads;
    for (auto& node : nodes) {
        if (node->requires_grad()) {
            grads.push_back(node->get_gradient());
        }
    }
    return grads;
}

void zero_gradients(const std::vector<std::shared_ptr<GraphNode>>& nodes) {
    for (auto& node : nodes) {
        node->zero_grad();
    }
}

void print_graph(const std::shared_ptr<GraphNode>& node) {
    // TODO: Implement graph visualization
    std::cout << "Graph node: " << node->get_tensor().to_string() << std::endl;
}

void save_graph(const std::shared_ptr<GraphNode>& node, const std::string& filename) {
    // TODO: Implement graph saving
}

std::shared_ptr<GraphNode> load_graph(const std::string& filename) {
    // TODO: Implement graph loading
    return nullptr;
}

} // namespace tensorcore
