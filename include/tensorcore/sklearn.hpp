#pragma once

#include "tensor.hpp"
#include <vector>
#include <string>
#include <memory>

namespace tensorcore {
namespace sklearn {

/**
 * @brief Scikit-learn style machine learning algorithms for TensorCore
 * 
 * This module provides implementations of popular machine learning algorithms
 * following the scikit-learn API design patterns with fit/predict/transform methods.
 */

// Forward declarations
class BaseEstimator;
class LinearRegression;
class Ridge;
class Lasso;
class ElasticNet;
class LogisticRegression;
class DecisionTreeClassifier;
class DecisionTreeRegressor;
class RandomForestClassifier;
class RandomForestRegressor;
class KMeans;
class DBSCAN;
class StandardScaler;
class MinMaxScaler;
class LabelEncoder;
class OneHotEncoder;

/**
 * @brief Base class for all scikit-learn style estimators
 */
class BaseEstimator {
public:
    virtual ~BaseEstimator() = default;
    virtual void fit(const Tensor& X, const Tensor& y = Tensor()) = 0;
    virtual Tensor predict(const Tensor& X) = 0;
    virtual Tensor transform(const Tensor& X) { return predict(X); }
    virtual Tensor fit_transform(const Tensor& X, const Tensor& y = Tensor()) {
        fit(X, y);
        return transform(X);
    }
    virtual std::string get_params() const { return ""; }
    virtual void set_params(const std::string& params) {}
};

/**
 * @brief Linear Regression
 * 
 * Ordinary least squares linear regression.
 * 
 * Parameters:
 * - fit_intercept: Whether to calculate the intercept (default: true)
 * - normalize: Whether to normalize features (default: false)
 * - copy_X: Whether to copy X (default: true)
 * - n_jobs: Number of jobs for parallel computation (default: 1)
 */
class LinearRegression : public BaseEstimator {
private:
    bool fit_intercept_;
    bool normalize_;
    bool copy_X_;
    int n_jobs_;
    
    Tensor coef_;
    Tensor intercept_;
    bool fitted_;
    
public:
    LinearRegression(bool fit_intercept = true, bool normalize = false, 
                    bool copy_X = true, int n_jobs = 1);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) override;
    
    Tensor get_coef() const { return coef_; }
    Tensor get_intercept() const { return intercept_; }
    double score(const Tensor& X, const Tensor& y) const;
};

/**
 * @brief Ridge Regression
 * 
 * Linear least squares with L2 regularization.
 * 
 * Parameters:
 * - alpha: Regularization strength (default: 1.0)
 * - fit_intercept: Whether to calculate the intercept (default: true)
 * - normalize: Whether to normalize features (default: false)
 * - max_iter: Maximum number of iterations (default: 1000)
 * - tol: Tolerance for convergence (default: 1e-4)
 */
class Ridge : public BaseEstimator {
private:
    double alpha_;
    bool fit_intercept_;
    bool normalize_;
    int max_iter_;
    double tol_;
    
    Tensor coef_;
    Tensor intercept_;
    bool fitted_;
    
public:
    Ridge(double alpha = 1.0, bool fit_intercept = true, bool normalize = false,
          int max_iter = 1000, double tol = 1e-4);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) override;
    
    Tensor get_coef() const { return coef_; }
    Tensor get_intercept() const { return intercept_; }
    double score(const Tensor& X, const Tensor& y) const;
};

/**
 * @brief Lasso Regression
 * 
 * Linear least squares with L1 regularization.
 * 
 * Parameters:
 * - alpha: Regularization strength (default: 1.0)
 * - fit_intercept: Whether to calculate the intercept (default: true)
 * - normalize: Whether to normalize features (default: false)
 * - max_iter: Maximum number of iterations (default: 1000)
 * - tol: Tolerance for convergence (default: 1e-4)
 */
class Lasso : public BaseEstimator {
private:
    double alpha_;
    bool fit_intercept_;
    bool normalize_;
    int max_iter_;
    double tol_;
    
    Tensor coef_;
    Tensor intercept_;
    bool fitted_;
    
public:
    Lasso(double alpha = 1.0, bool fit_intercept = true, bool normalize = false,
          int max_iter = 1000, double tol = 1e-4);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) override;
    
    Tensor get_coef() const { return coef_; }
    Tensor get_intercept() const { return intercept_; }
    double score(const Tensor& X, const Tensor& y) const;
};

/**
 * @brief Elastic Net Regression
 * 
 * Linear least squares with L1 and L2 regularization.
 * 
 * Parameters:
 * - alpha: Regularization strength (default: 1.0)
 * - l1_ratio: Balance between L1 and L2 regularization (default: 0.5)
 * - fit_intercept: Whether to calculate the intercept (default: true)
 * - normalize: Whether to normalize features (default: false)
 * - max_iter: Maximum number of iterations (default: 1000)
 * - tol: Tolerance for convergence (default: 1e-4)
 */
class ElasticNet : public BaseEstimator {
private:
    double alpha_;
    double l1_ratio_;
    bool fit_intercept_;
    bool normalize_;
    int max_iter_;
    double tol_;
    
    Tensor coef_;
    Tensor intercept_;
    bool fitted_;
    
public:
    ElasticNet(double alpha = 1.0, double l1_ratio = 0.5, bool fit_intercept = true, 
               bool normalize = false, int max_iter = 1000, double tol = 1e-4);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) override;
    
    Tensor get_coef() const { return coef_; }
    Tensor get_intercept() const { return intercept_; }
    double score(const Tensor& X, const Tensor& y) const;
};

/**
 * @brief Logistic Regression
 * 
 * Logistic regression for binary and multiclass classification.
 * 
 * Parameters:
 * - penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
 * - C: Inverse regularization strength (default: 1.0)
 * - fit_intercept: Whether to calculate the intercept (default: true)
 * - max_iter: Maximum number of iterations (default: 1000)
 * - tol: Tolerance for convergence (default: 1e-4)
 * - multi_class: Strategy for multiclass ('ovr', 'multinomial', 'auto')
 */
class LogisticRegression : public BaseEstimator {
private:
    std::string penalty_;
    double C_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    std::string multi_class_;
    
    Tensor coef_;
    Tensor intercept_;
    std::vector<int> classes_;
    bool fitted_;
    
public:
    LogisticRegression(const std::string& penalty = "l2", double C = 1.0, 
                       bool fit_intercept = true, int max_iter = 1000, 
                       double tol = 1e-4, const std::string& multi_class = "auto");
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) override;
    Tensor predict_proba(const Tensor& X);
    
    Tensor get_coef() const { return coef_; }
    Tensor get_intercept() const { return intercept_; }
    std::vector<int> get_classes() const { return classes_; }
    double score(const Tensor& X, const Tensor& y) const;
};

/**
 * @brief Decision Tree Classifier
 * 
 * Decision tree for classification.
 * 
 * Parameters:
 * - criterion: Splitting criterion ('gini', 'entropy', 'log_loss')
 * - max_depth: Maximum depth of tree (default: -1 for no limit)
 * - min_samples_split: Minimum samples to split (default: 2)
 * - min_samples_leaf: Minimum samples per leaf (default: 1)
 * - random_state: Random seed (default: 0)
 */
class DecisionTreeClassifier : public BaseEstimator {
private:
    std::string criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int random_state_;
    
    // Tree structure (simplified for educational purposes)
    struct TreeNode {
        int feature;
        double threshold;
        int left_child;
        int right_child;
        int prediction;
        bool is_leaf;
    };
    
    std::vector<TreeNode> tree_;
    std::vector<int> classes_;
    bool fitted_;
    
public:
    DecisionTreeClassifier(const std::string& criterion = "gini", int max_depth = -1,
                          int min_samples_split = 2, int min_samples_leaf = 1, 
                          int random_state = 0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) override;
    Tensor predict_proba(const Tensor& X);
    
    std::vector<int> get_classes() const { return classes_; }
    double score(const Tensor& X, const Tensor& y) const;
};

/**
 * @brief Decision Tree Regressor
 * 
 * Decision tree for regression.
 * 
 * Parameters:
 * - criterion: Splitting criterion ('squared_error', 'friedman_mse', 'absolute_error', 'poisson')
 * - max_depth: Maximum depth of tree (default: -1 for no limit)
 * - min_samples_split: Minimum samples to split (default: 2)
 * - min_samples_leaf: Minimum samples per leaf (default: 1)
 * - random_state: Random seed (default: 0)
 */
class DecisionTreeRegressor : public BaseEstimator {
private:
    std::string criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int random_state_;
    
    // Tree structure (simplified for educational purposes)
    struct TreeNode {
        int feature;
        double threshold;
        int left_child;
        int right_child;
        double prediction;
        bool is_leaf;
    };
    
    std::vector<TreeNode> tree_;
    bool fitted_;
    
public:
    DecisionTreeRegressor(const std::string& criterion = "squared_error", int max_depth = -1,
                         int min_samples_split = 2, int min_samples_leaf = 1, 
                         int random_state = 0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) override;
    
    double score(const Tensor& X, const Tensor& y) const;
};

/**
 * @brief K-Means Clustering
 * 
 * K-means clustering algorithm.
 * 
 * Parameters:
 * - n_clusters: Number of clusters (default: 8)
 * - init: Initialization method ('k-means++', 'random')
 * - n_init: Number of initializations (default: 10)
 * - max_iter: Maximum iterations (default: 300)
 * - tol: Tolerance for convergence (default: 1e-4)
 * - random_state: Random seed (default: 0)
 */
class KMeans : public BaseEstimator {
private:
    int n_clusters_;
    std::string init_;
    int n_init_;
    int max_iter_;
    double tol_;
    int random_state_;
    
    Tensor cluster_centers_;
    std::vector<int> labels_;
    bool fitted_;
    
public:
    KMeans(int n_clusters = 8, const std::string& init = "k-means++", 
           int n_init = 10, int max_iter = 300, double tol = 1e-4, 
           int random_state = 0);
    
    void fit(const Tensor& X, const Tensor& y = Tensor()) override;
    Tensor predict(const Tensor& X) override;
    
    Tensor get_cluster_centers() const { return cluster_centers_; }
    std::vector<int> get_labels() const { return labels_; }
    double inertia() const;
};

/**
 * @brief Standard Scaler
 * 
 * Standardize features by removing the mean and scaling to unit variance.
 * 
 * Parameters:
 * - with_mean: Whether to center the data (default: true)
 * - with_std: Whether to scale to unit variance (default: true)
 * - copy: Whether to copy the data (default: true)
 */
class StandardScaler : public BaseEstimator {
private:
    bool with_mean_;
    bool with_std_;
    bool copy_;
    
    Tensor mean_;
    Tensor scale_;
    bool fitted_;
    
public:
    StandardScaler(bool with_mean = true, bool with_std = true, bool copy = true);
    
    void fit(const Tensor& X, const Tensor& y = Tensor()) override;
    Tensor transform(const Tensor& X) override;
    Tensor fit_transform(const Tensor& X, const Tensor& y = Tensor()) override;
    Tensor inverse_transform(const Tensor& X);
    
    Tensor get_mean() const { return mean_; }
    Tensor get_scale() const { return scale_; }
};

/**
 * @brief Min-Max Scaler
 * 
 * Transform features by scaling each feature to a given range.
 * 
 * Parameters:
 * - feature_range: Desired range (default: (0, 1))
 * - copy: Whether to copy the data (default: true)
 */
class MinMaxScaler : public BaseEstimator {
private:
    std::pair<double, double> feature_range_;
    bool copy_;
    
    Tensor min_;
    Tensor scale_;
    bool fitted_;
    
public:
    MinMaxScaler(const std::pair<double, double>& feature_range = {0, 1}, bool copy = true);
    
    void fit(const Tensor& X, const Tensor& y = Tensor()) override;
    Tensor transform(const Tensor& X) override;
    Tensor fit_transform(const Tensor& X, const Tensor& y = Tensor()) override;
    Tensor inverse_transform(const Tensor& X);
    
    Tensor get_min() const { return min_; }
    Tensor get_scale() const { return scale_; }
};

/**
 * @brief Label Encoder
 * 
 * Encode target labels with value between 0 and n_classes-1.
 */
class LabelEncoder : public BaseEstimator {
private:
    std::vector<int> classes_;
    std::vector<int> inverse_classes_;
    bool fitted_;
    
public:
    LabelEncoder();
    
    void fit(const Tensor& y, const Tensor& X = Tensor()) override;
    Tensor transform(const Tensor& y) override;
    Tensor fit_transform(const Tensor& y, const Tensor& X = Tensor()) override;
    Tensor inverse_transform(const Tensor& y);
    
    std::vector<int> get_classes() const { return classes_; }
};

/**
 * @brief One-Hot Encoder
 * 
 * Encode categorical integer features as a one-hot numeric array.
 * 
 * Parameters:
 * - categories: Categories for each feature (default: "auto")
 * - drop: Whether to drop one category (default: "if_binary")
 * - sparse: Whether to return sparse matrix (default: false)
 * - dtype: Data type of output (default: float)
 */
class OneHotEncoder : public BaseEstimator {
private:
    std::string categories_;
    std::string drop_;
    bool sparse_;
    std::string dtype_;
    
    std::vector<std::vector<int>> categories_list_;
    bool fitted_;
    
public:
    OneHotEncoder(const std::string& categories = "auto", const std::string& drop = "if_binary",
                  bool sparse = false, const std::string& dtype = "float");
    
    void fit(const Tensor& X, const Tensor& y = Tensor()) override;
    Tensor transform(const Tensor& X) override;
    Tensor fit_transform(const Tensor& X, const Tensor& y = Tensor()) override;
    Tensor inverse_transform(const Tensor& X);
    
    std::vector<std::vector<int>> get_categories() const { return categories_list_; }
};

// Utility functions for model evaluation
namespace metrics {
    double accuracy_score(const Tensor& y_true, const Tensor& y_pred);
    double precision_score(const Tensor& y_true, const Tensor& y_pred, const std::string& average = "binary");
    double recall_score(const Tensor& y_true, const Tensor& y_pred, const std::string& average = "binary");
    double f1_score(const Tensor& y_true, const Tensor& y_pred, const std::string& average = "binary");
    double roc_auc_score(const Tensor& y_true, const Tensor& y_pred);
    double mean_squared_error(const Tensor& y_true, const Tensor& y_pred);
    double mean_absolute_error(const Tensor& y_true, const Tensor& y_pred);
    double r2_score(const Tensor& y_true, const Tensor& y_pred);
}

// Utility functions for model selection
namespace model_selection {
    std::pair<Tensor, Tensor> train_test_split(const Tensor& X, const Tensor& y, 
                                               double test_size = 0.25, int random_state = 0);
    Tensor cross_val_score(BaseEstimator& estimator, const Tensor& X, const Tensor& y, 
                          int cv = 5, const std::string& scoring = "accuracy");
}

} // namespace sklearn
} // namespace tensorcore
