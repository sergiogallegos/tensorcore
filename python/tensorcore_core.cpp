#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "tensorcore/tensor.hpp"
#include "tensorcore/operations.hpp"
#include "tensorcore/activations.hpp"
#include "tensorcore/losses.hpp"
#include "tensorcore/optimizers.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/autograd.hpp"
#include "tensorcore/utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tensorcore_core, m) {
    m.doc() = "TensorCore - Educational Machine Learning Library";
    
    // Version information
    m.attr("__version__") = "1.0.0";
    m.attr("__version_info__") = py::make_tuple(1, 0, 0);
    
    // Core Tensor class
    py::class_<tensorcore::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const tensorcore::Tensor::shape_type&>())
        .def(py::init<const tensorcore::Tensor::shape_type&, const tensorcore::Tensor::data_type&>())
        .def(py::init<const tensorcore::Tensor::shape_type&, tensorcore::Tensor::value_type>())
        .def(py::init<std::initializer_list<tensorcore::Tensor::value_type>>())
        .def(py::init<std::initializer_list<std::initializer_list<tensorcore::Tensor::value_type>>>())
        
        // Copy and move constructors
        .def(py::init<const tensorcore::Tensor&>())
        .def(py::init<tensorcore::Tensor&&>())
        
        // Assignment operators
        .def("__copy__", [](const tensorcore::Tensor& self) { return tensorcore::Tensor(self); })
        .def("__deepcopy__", [](const tensorcore::Tensor& self, py::dict) { return tensorcore::Tensor(self); })
        
        // Access operators
        .def("__getitem__", [](const tensorcore::Tensor& self, size_t index) {
            return self[index];
        })
        .def("__setitem__", [](tensorcore::Tensor& self, size_t index, tensorcore::Tensor::value_type value) {
            self[index] = value;
        })
        .def("__call__", [](const tensorcore::Tensor& self, const std::vector<size_t>& indices) {
            return self(indices);
        })
        
        // Arithmetic operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(py::self + tensorcore::Tensor::value_type())
        .def(py::self - tensorcore::Tensor::value_type())
        .def(py::self * tensorcore::Tensor::value_type())
        .def(py::self / tensorcore::Tensor::value_type())
        .def(tensorcore::Tensor::value_type() + py::self)
        .def(tensorcore::Tensor::value_type() - py::self)
        .def(tensorcore::Tensor::value_type() * py::self)
        .def(tensorcore::Tensor::value_type() / py::self)
        
        // In-place operations
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        .def(py::self += tensorcore::Tensor::value_type())
        .def(py::self -= tensorcore::Tensor::value_type())
        .def(py::self *= tensorcore::Tensor::value_type())
        .def(py::self /= tensorcore::Tensor::value_type())
        
        // Comparison operators
        .def(py::self == py::self)
        .def(py::self != py::self)
        
        // Properties
        .def_property_readonly("shape", &tensorcore::Tensor::shape)
        .def_property_readonly("ndim", &tensorcore::Tensor::ndim)
        .def_property_readonly("size", &tensorcore::Tensor::size)
        .def_property_readonly("data", [](const tensorcore::Tensor& self) { return self.data(); })
        .def_property("requires_grad", &tensorcore::Tensor::requires_grad, &tensorcore::Tensor::set_requires_grad)
        
        // Shape operations
        .def("reshape", &tensorcore::Tensor::reshape)
        .def("transpose", py::overload_cast<>(&tensorcore::Tensor::transpose, py::const_))
        .def("transpose", py::overload_cast<const std::vector<int>&>(&tensorcore::Tensor::transpose, py::const_))
        .def("squeeze", py::overload_cast<>(&tensorcore::Tensor::squeeze, py::const_))
        .def("squeeze", py::overload_cast<int>(&tensorcore::Tensor::squeeze, py::const_))
        .def("unsqueeze", &tensorcore::Tensor::unsqueeze)
        
        // Mathematical operations
        .def("sum", py::overload_cast<>(&tensorcore::Tensor::sum, py::const_))
        .def("sum", py::overload_cast<int>(&tensorcore::Tensor::sum, py::const_))
        .def("mean", py::overload_cast<>(&tensorcore::Tensor::mean, py::const_))
        .def("mean", py::overload_cast<int>(&tensorcore::Tensor::mean, py::const_))
        .def("max", py::overload_cast<>(&tensorcore::Tensor::max, py::const_))
        .def("max", py::overload_cast<int>(&tensorcore::Tensor::max, py::const_))
        .def("min", py::overload_cast<>(&tensorcore::Tensor::min, py::const_))
        .def("min", py::overload_cast<int>(&tensorcore::Tensor::min, py::const_))
        .def("abs", &tensorcore::Tensor::abs)
        .def("sqrt", &tensorcore::Tensor::sqrt)
        .def("exp", &tensorcore::Tensor::exp)
        .def("log", &tensorcore::Tensor::log)
        .def("pow", &tensorcore::Tensor::pow)
        
        // Linear algebra operations
        .def("matmul", &tensorcore::Tensor::matmul)
        .def("dot", &tensorcore::Tensor::dot)
        .def("norm", py::overload_cast<>(&tensorcore::Tensor::norm, py::const_))
        .def("norm", py::overload_cast<int>(&tensorcore::Tensor::norm, py::const_))
        
        // Utility functions
        .def("copy", &tensorcore::Tensor::copy)
        .def("fill", &tensorcore::Tensor::fill)
        .def("random_normal", &tensorcore::Tensor::random_normal)
        .def("random_uniform", &tensorcore::Tensor::random_uniform)
        .def("to_string", &tensorcore::Tensor::to_string)
        .def("print", &tensorcore::Tensor::print)
        .def("__str__", &tensorcore::Tensor::to_string)
        .def("__repr__", &tensorcore::Tensor::to_string)
        
        // Memory management
        .def("data_ptr", &tensorcore::Tensor::data_ptr)
        .def("is_broadcastable", &tensorcore::Tensor::is_broadcastable)
        .def("broadcast_to", &tensorcore::Tensor::broadcast_to)
        .def("slice", &tensorcore::Tensor::slice)
        .def("index", &tensorcore::Tensor::index);
    
    // Mathematical operations
    m.def("add", &tensorcore::add);
    m.def("subtract", &tensorcore::subtract);
    m.def("multiply", &tensorcore::multiply);
    m.def("divide", &tensorcore::divide);
    m.def("power", &tensorcore::power);
    m.def("mod", &tensorcore::mod);
    
    // Scalar operations
    m.def("add_scalar", &tensorcore::add_scalar);
    m.def("subtract_scalar", &tensorcore::subtract_scalar);
    m.def("multiply_scalar", &tensorcore::multiply_scalar);
    m.def("divide_scalar", &tensorcore::divide_scalar);
    m.def("power_scalar", &tensorcore::power_scalar);
    
    // Trigonometric functions
    m.def("sin", &tensorcore::sin);
    m.def("cos", &tensorcore::cos);
    m.def("tan", &tensorcore::tan);
    m.def("asin", &tensorcore::asin);
    m.def("acos", &tensorcore::acos);
    m.def("atan", &tensorcore::atan);
    m.def("atan2", &tensorcore::atan2);
    
    // Hyperbolic functions
    m.def("sinh", &tensorcore::sinh);
    m.def("cosh", &tensorcore::cosh);
    m.def("tanh", &tensorcore::tanh);
    m.def("asinh", &tensorcore::asinh);
    m.def("acosh", &tensorcore::acosh);
    m.def("atanh", &tensorcore::atanh);
    
    // Logarithmic and exponential functions
    m.def("log", &tensorcore::log);
    m.def("log2", &tensorcore::log2);
    m.def("log10", &tensorcore::log10);
    m.def("exp", &tensorcore::exp);
    m.def("exp2", &tensorcore::exp2);
    m.def("expm1", &tensorcore::expm1);
    m.def("log1p", &tensorcore::log1p);
    
    // Power and root functions
    m.def("sqrt", &tensorcore::sqrt);
    m.def("cbrt", &tensorcore::cbrt);
    m.def("square", &tensorcore::square);
    m.def("reciprocal", &tensorcore::reciprocal);
    m.def("rsqrt", &tensorcore::rsqrt);
    
    // Rounding functions
    m.def("floor", &tensorcore::floor);
    m.def("ceil", &tensorcore::ceil);
    m.def("round", &tensorcore::round);
    m.def("trunc", &tensorcore::trunc);
    m.def("rint", &tensorcore::rint);
    
    // Absolute value and sign functions
    m.def("abs", &tensorcore::abs);
    m.def("fabs", &tensorcore::fabs);
    m.def("sign", &tensorcore::sign);
    m.def("copysign", &tensorcore::copysign);
    
    // Comparison operations
    m.def("equal", &tensorcore::equal);
    m.def("not_equal", &tensorcore::not_equal);
    m.def("less", &tensorcore::less);
    m.def("less_equal", &tensorcore::less_equal);
    m.def("greater", &tensorcore::greater);
    m.def("greater_equal", &tensorcore::greater_equal);
    
    // Logical operations
    m.def("logical_and", &tensorcore::logical_and);
    m.def("logical_or", &tensorcore::logical_or);
    m.def("logical_xor", &tensorcore::logical_xor);
    m.def("logical_not", &tensorcore::logical_not);
    
    // Reduction operations
    m.def("sum", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::sum));
    m.def("sum", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::sum));
    m.def("sum", py::overload_cast<const tensorcore::Tensor&, const std::vector<int>&>(&tensorcore::sum));
    m.def("mean", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::mean));
    m.def("mean", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::mean));
    m.def("mean", py::overload_cast<const tensorcore::Tensor&, const std::vector<int>&>(&tensorcore::mean));
    m.def("max", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::max));
    m.def("max", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::max));
    m.def("max", py::overload_cast<const tensorcore::Tensor&, const std::vector<int>&>(&tensorcore::max));
    m.def("min", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::min));
    m.def("min", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::min));
    m.def("min", py::overload_cast<const tensorcore::Tensor&, const std::vector<int>&>(&tensorcore::min));
    m.def("argmax", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::argmax));
    m.def("argmax", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::argmax));
    m.def("argmin", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::argmin));
    m.def("argmin", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::argmin));
    m.def("prod", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::prod));
    m.def("prod", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::prod));
    m.def("prod", py::overload_cast<const tensorcore::Tensor&, const std::vector<int>&>(&tensorcore::prod));
    m.def("std", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::std));
    m.def("std", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::std));
    m.def("var", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::var));
    m.def("var", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::var));
    
    // Linear algebra operations
    m.def("matmul", &tensorcore::matmul);
    m.def("dot", &tensorcore::dot);
    m.def("outer", &tensorcore::outer);
    m.def("cross", &tensorcore::cross);
    m.def("norm", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::norm));
    m.def("norm", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::norm));
    m.def("norm", py::overload_cast<const tensorcore::Tensor&, double>(&tensorcore::norm));
    m.def("norm", py::overload_cast<const tensorcore::Tensor&, int, double>(&tensorcore::norm));
    
    // Matrix operations
    m.def("transpose", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::transpose));
    m.def("transpose", py::overload_cast<const tensorcore::Tensor&, const std::vector<int>&>(&tensorcore::transpose));
    m.def("conjugate", &tensorcore::conjugate);
    m.def("hermitian", &tensorcore::hermitian);
    m.def("trace", &tensorcore::trace);
    m.def("det", &tensorcore::det);
    m.def("inv", &tensorcore::inv);
    m.def("pinv", &tensorcore::pinv);
    m.def("solve", &tensorcore::solve);
    m.def("lstsq", &tensorcore::lstsq);
    
    // Eigenvalue and SVD
    m.def("eig", &tensorcore::eig);
    m.def("svd", &tensorcore::svd);
    m.def("eigh", &tensorcore::eigh);
    
    // Broadcasting operations
    m.def("broadcast_to", &tensorcore::broadcast_to);
    m.def("broadcast_add", &tensorcore::broadcast_add);
    m.def("broadcast_multiply", &tensorcore::broadcast_multiply);
    
    // Utility operations
    m.def("concatenate", &tensorcore::concatenate);
    m.def("stack", &tensorcore::stack);
    m.def("split", &tensorcore::split);
    m.def("tile", &tensorcore::tile);
    m.def("repeat", &tensorcore::repeat);
    m.def("pad", &tensorcore::pad);
    
    // Statistical operations
    m.def("histogram", &tensorcore::histogram);
    m.def("percentile", &tensorcore::percentile);
    m.def("quantile", &tensorcore::quantile);
    m.def("median", py::overload_cast<const tensorcore::Tensor&>(&tensorcore::median));
    m.def("median", py::overload_cast<const tensorcore::Tensor&, int>(&tensorcore::median));
    
    // Window functions
    m.def("hamming", &tensorcore::hamming);
    m.def("hanning", &tensorcore::hanning);
    m.def("blackman", &tensorcore::blackman);
    m.def("bartlett", &tensorcore::bartlett);
    
    // Convolution operations
    m.def("conv1d", &tensorcore::conv1d);
    m.def("conv2d", &tensorcore::conv2d);
    m.def("max_pool1d", &tensorcore::max_pool1d);
    m.def("max_pool2d", &tensorcore::max_pool2d);
    
    // Gradient operations
    m.def("gradient", &tensorcore::gradient);
    m.def("hessian", &tensorcore::hessian);
    
    // Tensor creation functions
    m.def("tensor", [](const std::vector<double>& data) {
        return tensorcore::Tensor({static_cast<size_t>(data.size())}, data);
    });
    m.def("zeros", &tensorcore::zeros);
    m.def("ones", &tensorcore::ones);
    m.def("eye", py::overload_cast<size_t>(&tensorcore::eye));
    m.def("eye", py::overload_cast<size_t, size_t>(&tensorcore::eye));
    m.def("arange", &tensorcore::arange);
    m.def("linspace", &tensorcore::linspace);
    m.def("random_normal", &tensorcore::random_normal);
    m.def("random_uniform", &tensorcore::random_uniform);
    
    // Utility functions
    m.def("set_random_seed", &tensorcore::set_random_seed);
    m.def("create_identity_matrix", &tensorcore::create_identity_matrix);
    m.def("create_zeros", &tensorcore::create_zeros);
    m.def("create_ones", &tensorcore::create_ones);
    m.def("create_range", &tensorcore::create_range);
    
    // Note: Exception classes would be registered here if they were defined
}
