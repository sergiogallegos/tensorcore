#include "tensorcore/tensor.hpp"
#include "tensorcore/operations.hpp"
#include "tensorcore/activations.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/utils.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace tensorcore;

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;
    
    // Test default constructor
    Tensor t1;
    assert(t1.size() == 0);
    assert(t1.shape().empty());
    
    // Test shape constructor
    Tensor t2({3, 4});
    assert(t2.size() == 12);
    assert((t2.shape() == std::vector<size_t>{3, 4}));
    
    // Test shape and value constructor
    Tensor t3({2, 3}, 5.0);
    assert(t3.size() == 6);
    assert((t3.shape() == std::vector<size_t>{2, 3}));
    for (size_t i = 0; i < t3.size(); ++i) {
        assert(t3[i] == 5.0);
    }
    
    // Test initializer list constructor
    Tensor t4({1.0, 2.0, 3.0, 4.0});
    assert(t4.size() == 4);
    assert(t4.shape() == std::vector<size_t>{4});
    assert(t4[0] == 1.0);
    assert(t4[1] == 2.0);
    assert(t4[2] == 3.0);
    assert(t4[3] == 4.0);
    
    // Test 2D initializer list constructor
    Tensor t5({{1.0, 2.0}, {3.0, 4.0}});
    assert(t5.size() == 4);
    assert((t5.shape() == std::vector<size_t>{2, 2}));
    assert(t5[0] == 1.0);
    assert(t5[1] == 2.0);
    assert(t5[2] == 3.0);
    assert(t5[3] == 4.0);
    
    std::cout << "  ✓ Tensor creation tests passed" << std::endl;
}

void test_tensor_arithmetic() {
    std::cout << "Testing tensor arithmetic..." << std::endl;
    
    // Test addition
    Tensor a({2, 3}, 1.0);
    Tensor b({2, 3}, 2.0);
    Tensor c = a + b;
    assert(c.size() == 6);
    for (size_t i = 0; i < c.size(); ++i) {
        assert(c[i] == 3.0);
    }
    
    // Test subtraction
    Tensor d = b - a;
    for (size_t i = 0; i < d.size(); ++i) {
        assert(d[i] == 1.0);
    }
    
    // Test multiplication
    Tensor e = a * b;
    for (size_t i = 0; i < e.size(); ++i) {
        assert(e[i] == 2.0);
    }
    
    // Test division
    Tensor f = b / a;
    for (size_t i = 0; i < f.size(); ++i) {
        assert(f[i] == 2.0);
    }
    
    // Test scalar operations
    Tensor g = a + 5.0;
    for (size_t i = 0; i < g.size(); ++i) {
        assert(g[i] == 6.0);
    }
    
    Tensor h = a * 3.0;
    for (size_t i = 0; i < h.size(); ++i) {
        assert(h[i] == 3.0);
    }
    
    std::cout << "  ✓ Tensor arithmetic tests passed" << std::endl;
}

void test_tensor_shape_operations() {
    std::cout << "Testing tensor shape operations..." << std::endl;
    
    // Test reshape
    Tensor a({2, 3}, 1.0);
    Tensor b = a.reshape({3, 2});
    assert((b.shape() == std::vector<size_t>{3, 2}));
    assert(b.size() == 6);
    
    // Test transpose
    Tensor c = a.transpose();
    assert((c.shape() == std::vector<size_t>{3, 2}));
    
    // Test squeeze
    Tensor d({1, 3}, 1.0);
    Tensor e = d.squeeze();
    assert(e.shape() == std::vector<size_t>{3});
    
    // Test unsqueeze
    Tensor f({3}, 1.0);
    Tensor g = f.unsqueeze(0);
    assert((g.shape() == std::vector<size_t>{1, 3}));
    
    std::cout << "  ✓ Tensor shape operations tests passed" << std::endl;
}

void test_tensor_mathematical_operations() {
    std::cout << "Testing tensor mathematical operations..." << std::endl;
    
    Tensor a({2, 3}, 2.0);
    
    // Test sum
    Tensor sum_all = a.sum();
    assert(sum_all.size() == 1);
    assert(sum_all[0] == 12.0); // 2 * 3 * 2.0
    
    Tensor sum_axis = a.sum(0);
    assert(sum_axis.shape() == std::vector<size_t>{3});
    for (size_t i = 0; i < sum_axis.size(); ++i) {
        assert(sum_axis[i] == 4.0); // 2 * 2.0
    }
    
    // Test mean
    Tensor mean_all = a.mean();
    assert(mean_all.size() == 1);
    assert(mean_all[0] == 2.0);
    
    // Test max
    Tensor max_all = a.max();
    assert(max_all.size() == 1);
    assert(max_all[0] == 2.0);
    
    // Test min
    Tensor min_all = a.min();
    assert(min_all.size() == 1);
    assert(min_all[0] == 2.0);
    
    // Test abs
    Tensor b({2, 2}, -3.0);
    Tensor c = b.abs();
    for (size_t i = 0; i < c.size(); ++i) {
        assert(c[i] == 3.0);
    }
    
    // Test sqrt
    Tensor d({2, 2}, 4.0);
    Tensor e = d.sqrt();
    for (size_t i = 0; i < e.size(); ++i) {
        assert(std::abs(e[i] - 2.0) < 1e-10);
    }
    
    // Test exp
    Tensor f({2, 2}, 0.0);
    Tensor g = f.exp();
    for (size_t i = 0; i < g.size(); ++i) {
        assert(std::abs(g[i] - 1.0) < 1e-10);
    }
    
    // Test log
    Tensor h({2, 2}, 1.0);
    Tensor i = h.log();
    for (size_t j = 0; j < i.size(); ++j) {
        assert(std::abs(i[j] - 0.0) < 1e-10);
    }
    
    std::cout << "  ✓ Tensor mathematical operations tests passed" << std::endl;
}

void test_tensor_linear_algebra() {
    std::cout << "Testing tensor linear algebra..." << std::endl;
    
    // Test matrix multiplication
    Tensor a({2, 3}, 1.0);
    Tensor b({3, 2}, 2.0);
    Tensor c = a.matmul(b);
    assert((c.shape() == std::vector<size_t>{2, 2}));
    for (size_t i = 0; i < c.size(); ++i) {
        assert(c[i] == 6.0); // 3 * 1.0 * 2.0
    }
    
    // Test dot product
    Tensor d({3}, 1.0);
    Tensor e({3}, 2.0);
    Tensor f = d.dot(e);
    assert(f.size() == 1);
    assert(f[0] == 6.0); // 3 * 1.0 * 2.0
    
    // Test norm
    Tensor g({3}, 3.0);
    Tensor h = g.norm();
    assert(h.size() == 1);
    assert(std::abs(h[0] - std::sqrt(27.0)) < 1e-10); // sqrt(3 * 3^2)
    
    std::cout << "  ✓ Tensor linear algebra tests passed" << std::endl;
}

void test_activation_functions() {
    std::cout << "Testing activation functions..." << std::endl;
    
    Tensor x({3}, 1.0);
    
    // Test ReLU
    Tensor relu_x = relu(x);
    for (size_t i = 0; i < relu_x.size(); ++i) {
        assert(relu_x[i] == 1.0);
    }
    
    // Test negative ReLU
    Tensor neg_x({3}, -1.0);
    Tensor relu_neg = relu(neg_x);
    for (size_t i = 0; i < relu_neg.size(); ++i) {
        assert(relu_neg[i] == 0.0);
    }
    
    // Test sigmoid
    Tensor sig_x = sigmoid(x);
    for (size_t i = 0; i < sig_x.size(); ++i) {
        assert(sig_x[i] > 0.0 && sig_x[i] < 1.0);
    }
    
    // Test tanh
    Tensor tanh_x = tanh(x);
    for (size_t i = 0; i < tanh_x.size(); ++i) {
        assert(tanh_x[i] > -1.0 && tanh_x[i] < 1.0);
    }
    
    // Test softmax
    Tensor softmax_x = softmax(x);
    [[maybe_unused]] double sum = 0.0;
    for (size_t i = 0; i < softmax_x.size(); ++i) {
        assert(softmax_x[i] > 0.0);
        sum += softmax_x[i];
    }
    assert(std::abs(sum - 1.0) < 1e-10);
    
    std::cout << "  ✓ Activation functions tests passed" << std::endl;
}

void test_loss_functions() {
    std::cout << "Testing loss functions..." << std::endl;
    
    Tensor predictions({3}, 1.0);
    Tensor targets({3}, 2.0);
    
    // Test MSE loss
    Tensor mse = mse_loss(predictions, targets);
    assert(mse.size() == 1);
    assert(mse[0] == 1.0); // (1-2)^2 = 1
    
    // Test MAE loss
    Tensor mae = mae_loss(predictions, targets);
    assert(mae.size() == 1);
    assert(mae[0] == 1.0); // |1-2| = 1
    
    // Test Huber loss
    Tensor huber = huber_loss(predictions, targets);
    assert(huber.size() == 1);
    assert(huber[0] == 0.5); // 0.5 * (1-2)^2 = 0.5
    
    std::cout << "  ✓ Loss functions tests passed" << std::endl;
}

void test_utility_functions() {
    std::cout << "Testing utility functions..." << std::endl;
    
    // Test zeros
    Tensor zeros_tensor = zeros({2, 3});
    assert((zeros_tensor.shape() == std::vector<size_t>{2, 3}));
    for (size_t i = 0; i < zeros_tensor.size(); ++i) {
        assert(zeros_tensor[i] == 0.0);
    }
    
    // Test ones
    Tensor ones_tensor = ones({2, 3});
    assert((ones_tensor.shape() == std::vector<size_t>{2, 3}));
    for (size_t i = 0; i < ones_tensor.size(); ++i) {
        assert(ones_tensor[i] == 1.0);
    }
    
    // Test eye
    Tensor eye_tensor = eye(3);
    assert((eye_tensor.shape() == std::vector<size_t>{3, 3}));
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                assert(eye_tensor({i, j}) == 1.0);
            } else {
                assert(eye_tensor({i, j}) == 0.0);
            }
        }
    }
    
    // Test arange
    Tensor arange_tensor = arange(0.0, 5.0, 1.0);
    assert(arange_tensor.size() == 5);
    for (size_t i = 0; i < arange_tensor.size(); ++i) {
        assert(arange_tensor[i] == static_cast<double>(i));
    }
    
    // Test linspace
    Tensor linspace_tensor = linspace(0.0, 1.0, 5);
    assert(linspace_tensor.size() == 5);
    assert(linspace_tensor[0] == 0.0);
    assert(linspace_tensor[4] == 1.0);
    
    std::cout << "  ✓ Utility functions tests passed" << std::endl;
}

void test_operations_functions() {
    std::cout << "Testing operations functions..." << std::endl;
    
    Tensor a({2, 3}, 1.0);
    Tensor b({2, 3}, 2.0);
    
    // Test add
    Tensor c = add(a, b);
    for (size_t i = 0; i < c.size(); ++i) {
        assert(c[i] == 3.0);
    }
    
    // Test subtract
    Tensor d = subtract(b, a);
    for (size_t i = 0; i < d.size(); ++i) {
        assert(d[i] == 1.0);
    }
    
    // Test multiply
    Tensor e = multiply(a, b);
    for (size_t i = 0; i < e.size(); ++i) {
        assert(e[i] == 2.0);
    }
    
    // Test divide
    Tensor f = divide(b, a);
    for (size_t i = 0; i < f.size(); ++i) {
        assert(f[i] == 2.0);
    }
    
    // Test power
    Tensor power_tensor({2, 3}, 2.0);
    Tensor g = power(power_tensor, power_tensor);
    for (size_t i = 0; i < g.size(); ++i) {
        assert(g[i] == 4.0); // 2^2 = 4
    }
    
    // Test sin
    Tensor h({3}, 0.0);
    Tensor i = sin(h);
    for (size_t j = 0; j < i.size(); ++j) {
        assert(std::abs(i[j] - 0.0) < 1e-10);
    }
    
    // Test cos
    Tensor j = cos(h);
    for (size_t k = 0; k < j.size(); ++k) {
        assert(std::abs(j[k] - 1.0) < 1e-10);
    }
    
    std::cout << "  ✓ Operations functions tests passed" << std::endl;
}

int main() {
    std::cout << "Running comprehensive TensorCore tests..." << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        test_tensor_creation();
        test_tensor_arithmetic();
        test_tensor_shape_operations();
        test_tensor_mathematical_operations();
        test_tensor_linear_algebra();
        test_activation_functions();
        test_loss_functions();
        test_utility_functions();
        test_operations_functions();
        
        std::cout << "=========================================" << std::endl;
        std::cout << "🎉 All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
