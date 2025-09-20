#pragma once

#include "tensor.hpp"
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <memory>
#include <fstream>

namespace tensorcore {

/**
 * @brief Utility functions and classes
 * 
 * This module provides various utility functions and classes that are useful
 * for machine learning operations, including random number generation,
 * data loading, preprocessing, and other helper functions.
 */

// Random number generation
class RandomGenerator {
public:
    static RandomGenerator& get_instance();
    
    void set_seed(unsigned int seed);
    unsigned int get_seed() const { return seed_; }
    
    // Random number generation
    double uniform(double min = 0.0, double max = 1.0);
    double normal(double mean = 0.0, double std = 1.0);
    int uniform_int(int min, int max);
    bool bernoulli(double p = 0.5);
    
    // Random tensor generation
    Tensor uniform_tensor(const Tensor::shape_type& shape, double min = 0.0, double max = 1.0);
    Tensor normal_tensor(const Tensor::shape_type& shape, double mean = 0.0, double std = 1.0);
    Tensor bernoulli_tensor(const Tensor::shape_type& shape, double p = 0.5);
    
    // Random sampling
    std::vector<int> choice(int n, int k, bool replace = true);
    std::vector<int> permutation(int n);
    void shuffle(std::vector<int>& indices);
    
private:
    RandomGenerator() = default;
    unsigned int seed_;
    std::mt19937 generator_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
};

// Data preprocessing
class DataPreprocessor {
public:
    // Normalization
    static Tensor normalize(const Tensor& data, double mean = 0.0, double std = 1.0);
    static Tensor min_max_scale(const Tensor& data, double min_val = 0.0, double max_val = 1.0);
    static Tensor robust_scale(const Tensor& data);
    static Tensor unit_scale(const Tensor& data);
    
    // Standardization
    static std::pair<Tensor, std::pair<double, double>> standardize(const Tensor& data);
    static Tensor inverse_standardize(const Tensor& data, double mean, double std);
    
    // One-hot encoding
    static Tensor one_hot_encode(const Tensor& labels, int num_classes);
    static Tensor one_hot_decode(const Tensor& one_hot);
    
    // Data splitting
    static std::tuple<Tensor, Tensor, Tensor, Tensor> train_test_split(
        const Tensor& X, const Tensor& y, double test_size = 0.2, unsigned int random_state = 42);
    
    // Cross-validation
    static std::vector<std::tuple<Tensor, Tensor, Tensor, Tensor>> k_fold_split(
        const Tensor& X, const Tensor& y, int k = 5, unsigned int random_state = 42);
    
    // Data augmentation
    static Tensor add_noise(const Tensor& data, double noise_level = 0.1);
    static Tensor random_rotation(const Tensor& data, double max_angle = 15.0);
    static Tensor random_shift(const Tensor& data, double max_shift = 0.1);
    static Tensor random_flip(const Tensor& data, double flip_prob = 0.5);
};

// Performance monitoring
class Timer {
public:
    Timer();
    void start();
    void stop();
    double elapsed_seconds() const;
    double elapsed_milliseconds() const;
    double elapsed_microseconds() const;
    void reset();
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_;
};

// Memory usage monitoring
class MemoryMonitor {
public:
    static size_t get_memory_usage();
    static size_t get_peak_memory_usage();
    static void reset_peak_memory();
    static std::string format_bytes(size_t bytes);
};

// Progress bar
class ProgressBar {
public:
    ProgressBar(int total, int width = 50, char fill_char = '=', char empty_char = ' ');
    void update(int current);
    void finish();
    
private:
    int total_;
    int width_;
    char fill_char_;
    char empty_char_;
    int current_;
    bool finished_;
};

// Configuration management
class Config {
public:
    static Config& get_instance();
    
    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, int value);
    void set(const std::string& key, double value);
    void set(const std::string& key, bool value);
    
    std::string get_string(const std::string& key, const std::string& default_value = "") const;
    int get_int(const std::string& key, int default_value = 0) const;
    double get_double(const std::string& key, double default_value = 0.0) const;
    bool get_bool(const std::string& key, bool default_value = false) const;
    
    void load_from_file(const std::string& filename);
    void save_to_file(const std::string& filename) const;
    
    void clear();
    
private:
    Config() = default;
    std::unordered_map<std::string, std::string> config_;
};

// Logging system
class Logger {
public:
    enum Level { DEBUG, INFO, WARNING, ERROR, CRITICAL };
    
    static Logger& get_instance();
    
    void set_level(Level level);
    void set_output_file(const std::string& filename);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
private:
    Logger() = default;
    Level level_;
    std::string output_file_;
    std::ofstream file_stream_;
    
    void log(Level level, const std::string& message);
    std::string level_to_string(Level level) const;
    std::string get_timestamp() const;
};

// File I/O utilities
class FileUtils {
public:
    // File existence and properties
    static bool exists(const std::string& filename);
    static bool is_file(const std::string& path);
    static bool is_directory(const std::string& path);
    static size_t file_size(const std::string& filename);
    
    // Directory operations
    static std::vector<std::string> list_files(const std::string& directory);
    static std::vector<std::string> list_directories(const std::string& directory);
    static bool create_directory(const std::string& path);
    static bool remove_directory(const std::string& path);
    
    // File operations
    static bool copy_file(const std::string& source, const std::string& destination);
    static bool move_file(const std::string& source, const std::string& destination);
    static bool remove_file(const std::string& filename);
    
    // Path utilities
    static std::string get_filename(const std::string& path);
    static std::string get_directory(const std::string& path);
    static std::string get_extension(const std::string& filename);
    static std::string join_path(const std::string& path1, const std::string& path2);
};

// Data loading utilities
class DataLoader {
public:
    // CSV loading
    static std::pair<Tensor, Tensor> load_csv(const std::string& filename, 
                                             bool has_header = true, 
                                             char delimiter = ',');
    
    // Image loading (basic implementation)
    static Tensor load_image(const std::string& filename);
    static void save_image(const Tensor& image, const std::string& filename);
    
    // Dataset loading
    static std::pair<Tensor, Tensor> load_mnist(const std::string& data_dir);
    static std::pair<Tensor, Tensor> load_cifar10(const std::string& data_dir);
    
    // Text loading
    static std::vector<std::string> load_text_lines(const std::string& filename);
    static std::string load_text(const std::string& filename);
};

// Mathematical utilities
class MathUtils {
public:
    // Constants
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double E = 2.71828182845904523536;
    static constexpr double EPSILON = 1e-8;
    
    // Basic functions
    static double sigmoid(double x);
    static double tanh(double x);
    static double relu(double x);
    static double softmax(const std::vector<double>& x, int index);
    
    // Statistical functions
    static double mean(const std::vector<double>& data);
    static double variance(const std::vector<double>& data);
    static double standard_deviation(const std::vector<double>& data);
    static double correlation(const std::vector<double>& x, const std::vector<double>& y);
    
    // Distance functions
    static double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double manhattan_distance(const std::vector<double>& a, const std::vector<double>& b);
    static double cosine_distance(const std::vector<double>& a, const std::vector<double>& b);
    
    // Numerical stability
    static double log_sum_exp(const std::vector<double>& x);
    static double softmax_stable(const std::vector<double>& x, int index);
};

// String utilities
class StringUtils {
public:
    static std::vector<std::string> split(const std::string& str, char delimiter);
    static std::string join(const std::vector<std::string>& strings, const std::string& delimiter);
    static std::string trim(const std::string& str);
    static std::string to_lower(const std::string& str);
    static std::string to_upper(const std::string& str);
    static bool starts_with(const std::string& str, const std::string& prefix);
    static bool ends_with(const std::string& str, const std::string& suffix);
    static std::string format(const std::string& format, ...);
};

// Global utility functions
void set_random_seed(unsigned int seed);
Tensor create_identity_matrix(int size);
Tensor create_zeros(const Tensor::shape_type& shape);
Tensor create_ones(const Tensor::shape_type& shape);
Tensor create_range(double start, double stop, double step = 1.0);

// Debugging utilities
void print_tensor_info(const Tensor& tensor, const std::string& name = "");
void print_memory_usage();
void print_configuration();

} // namespace tensorcore
