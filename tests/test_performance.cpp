#include "tensorcore/tensor.hpp"
#include "tensorcore/operations.hpp"
#include "tensorcore/activations.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/utils.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

using namespace tensorcore;

class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string operation_name;
    
public:
    PerformanceTimer(const std::string& name) : operation_name(name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~PerformanceTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "  " << operation_name << ": " << duration.count() << " μs" << std::endl;
    }
};

void benchmark_tensor_creation() {
    std::cout << "Benchmarking tensor creation..." << std::endl;
    
    const size_t size = 1000000;
    
    {
        PerformanceTimer timer("Create tensor with shape constructor");
        Tensor t({size});
    }
    
    {
        PerformanceTimer timer("Create tensor with value constructor");
        Tensor t({size}, 1.0);
    }
    
    {
        PerformanceTimer timer("Create tensor with initializer list");
        std::vector<double> data(size, 1.0);
        Tensor t({size}, data);
    }
}

void benchmark_tensor_operations() {
    std::cout << "Benchmarking tensor operations..." << std::endl;
    
    const size_t size = 1000000;
    Tensor a({size}, 1.0);
    Tensor b({size}, 2.0);
    
    {
        PerformanceTimer timer("Element-wise addition");
        Tensor c = a + b;
    }
    
    {
        PerformanceTimer timer("Element-wise multiplication");
        Tensor c = a * b;
    }
    
    {
        PerformanceTimer timer("Scalar addition");
        Tensor c = a + 5.0;
    }
    
    {
        PerformanceTimer timer("Scalar multiplication");
        Tensor c = a * 3.0;
    }
}

void benchmark_mathematical_operations() {
    std::cout << "Benchmarking mathematical operations..." << std::endl;
    
    const size_t size = 1000000;
    Tensor a({size}, 2.0);
    
    {
        PerformanceTimer timer("Sum operation");
        Tensor sum_result = a.sum();
    }
    
    {
        PerformanceTimer timer("Mean operation");
        Tensor mean_result = a.mean();
    }
    
    {
        PerformanceTimer timer("Max operation");
        Tensor max_result = a.max();
    }
    
    {
        PerformanceTimer timer("Min operation");
        Tensor min_result = a.min();
    }
    
    {
        PerformanceTimer timer("Abs operation");
        Tensor abs_result = a.abs();
    }
    
    {
        PerformanceTimer timer("Sqrt operation");
        Tensor sqrt_result = a.sqrt();
    }
    
    {
        PerformanceTimer timer("Exp operation");
        Tensor exp_result = a.exp();
    }
    
    {
        PerformanceTimer timer("Log operation");
        Tensor log_result = a.log();
    }
}

void benchmark_activation_functions() {
    std::cout << "Benchmarking activation functions..." << std::endl;
    
    const size_t size = 1000000;
    Tensor a({size}, 1.0);
    
    {
        PerformanceTimer timer("ReLU activation");
        Tensor relu_result = relu(a);
    }
    
    {
        PerformanceTimer timer("Sigmoid activation");
        Tensor sigmoid_result = sigmoid(a);
    }
    
    {
        PerformanceTimer timer("Tanh activation");
        Tensor tanh_result = tanh(a);
    }
    
    {
        PerformanceTimer timer("Softmax activation");
        Tensor softmax_result = softmax(a);
    }
}

void benchmark_loss_functions() {
    std::cout << "Benchmarking loss functions..." << std::endl;
    
    const size_t size = 1000000;
    Tensor predictions({size}, 1.0);
    Tensor targets({size}, 2.0);
    
    {
        PerformanceTimer timer("MSE loss");
        Tensor mse_result = mse_loss(predictions, targets);
    }
    
    {
        PerformanceTimer timer("MAE loss");
        Tensor mae_result = mae_loss(predictions, targets);
    }
    
    {
        PerformanceTimer timer("Huber loss");
        Tensor huber_result = huber_loss(predictions, targets);
    }
    
    {
        PerformanceTimer timer("Cross entropy loss");
        Tensor ce_result = cross_entropy_loss(predictions, targets);
    }
}

void benchmark_linear_algebra() {
    std::cout << "Benchmarking linear algebra operations..." << std::endl;
    
    const size_t matrix_size = 1000;
    Tensor a({matrix_size, matrix_size}, 1.0);
    Tensor b({matrix_size, matrix_size}, 2.0);
    
    {
        PerformanceTimer timer("Matrix multiplication");
        Tensor c = a.matmul(b);
    }
    
    {
        PerformanceTimer timer("Dot product");
        Tensor d({matrix_size}, 1.0);
        Tensor e({matrix_size}, 2.0);
        Tensor f = d.dot(e);
    }
    
    {
        PerformanceTimer timer("Norm calculation");
        Tensor g = a.norm();
    }
}

void benchmark_memory_usage() {
    std::cout << "Benchmarking memory usage..." << std::endl;
    
    const size_t sizes[] = {1000, 10000, 100000, 1000000};
    
    for (size_t size : sizes) {
        std::cout << "  Testing with " << size << " elements:" << std::endl;
        
        {
            PerformanceTimer timer("Create tensor");
            Tensor t({size}, 1.0);
        }
        
        {
            PerformanceTimer timer("Copy tensor");
            Tensor t1({size}, 1.0);
            Tensor t2 = t1.copy();
        }
        
        {
            PerformanceTimer timer("Reshape tensor");
            Tensor t({size}, 1.0);
            Tensor reshaped = t.reshape({size/2, 2});
        }
    }
}

void benchmark_random_operations() {
    std::cout << "Benchmarking random operations..." << std::endl;
    
    const size_t size = 1000000;
    
    {
        PerformanceTimer timer("Random normal generation");
        Tensor random_normal = random_normal({size}, 0.0, 1.0);
    }
    
    {
        PerformanceTimer timer("Random uniform generation");
        Tensor random_uniform = random_uniform({size}, 0.0, 1.0);
    }
}

void benchmark_utility_functions() {
    std::cout << "Benchmarking utility functions..." << std::endl;
    
    const size_t size = 1000000;
    
    {
        PerformanceTimer timer("Zeros creation");
        Tensor zeros_tensor = zeros({size});
    }
    
    {
        PerformanceTimer timer("Ones creation");
        Tensor ones_tensor = ones({size});
    }
    
    {
        PerformanceTimer timer("Eye matrix creation");
        Tensor eye_tensor = eye(1000);
    }
    
    {
        PerformanceTimer timer("Arange creation");
        Tensor arange_tensor = arange(0.0, 1000.0, 1.0);
    }
    
    {
        PerformanceTimer timer("Linspace creation");
        Tensor linspace_tensor = linspace(0.0, 1.0, size);
    }
}

int main() {
    std::cout << "TensorCore Performance Benchmark" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        benchmark_tensor_creation();
        std::cout << std::endl;
        
        benchmark_tensor_operations();
        std::cout << std::endl;
        
        benchmark_mathematical_operations();
        std::cout << std::endl;
        
        benchmark_activation_functions();
        std::cout << std::endl;
        
        benchmark_loss_functions();
        std::cout << std::endl;
        
        benchmark_linear_algebra();
        std::cout << std::endl;
        
        benchmark_memory_usage();
        std::cout << std::endl;
        
        benchmark_random_operations();
        std::cout << std::endl;
        
        benchmark_utility_functions();
        std::cout << std::endl;
        
        std::cout << "=================================" << std::endl;
        std::cout << "🎉 Performance benchmark completed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
