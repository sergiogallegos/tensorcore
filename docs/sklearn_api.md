# TensorCore Scikit-learn Style API

TensorCore now includes a comprehensive scikit-learn style machine learning API that provides familiar interfaces for common ML algorithms and utilities.

## 🎯 Overview

The sklearn module in TensorCore provides:

- **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
- **Preprocessing**: StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
- **Metrics**: Accuracy, Precision, Recall, F1-Score, MSE, MAE, R²
- **Model Selection**: train_test_split, cross_val_score, GridSearchCV
- **Consistent API**: All models follow the fit/predict/transform pattern

## 📚 Quick Start

```python
import tensorcore as tc
from tensorcore.sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from tensorcore.sklearn.preprocessing import StandardScaler
from tensorcore.sklearn.metrics import accuracy_score, mean_squared_error
from tensorcore.sklearn.model_selection import train_test_split

# Generate data
X = tc.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
y = tc.tensor([3, 7, 11, 15])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")
```

## 🔧 Linear Models

### LinearRegression

Ordinary least squares linear regression.

```python
from tensorcore.sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True, normalize=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
```

**Parameters:**
- `fit_intercept` (bool): Whether to calculate the intercept
- `normalize` (bool): Whether to normalize features
- `copy_X` (bool): Whether to copy X
- `n_jobs` (int): Number of jobs for parallel computation

### Ridge Regression

Linear regression with L2 regularization.

```python
from tensorcore.sklearn.linear_model import Ridge

model = Ridge(alpha=1.0, fit_intercept=True, normalize=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Parameters:**
- `alpha` (float): Regularization strength
- `fit_intercept` (bool): Whether to calculate the intercept
- `normalize` (bool): Whether to normalize features
- `max_iter` (int): Maximum number of iterations
- `tol` (float): Tolerance for convergence

### Lasso Regression

Linear regression with L1 regularization.

```python
from tensorcore.sklearn.linear_model import Lasso

model = Lasso(alpha=0.1, fit_intercept=True, normalize=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Parameters:**
- `alpha` (float): Regularization strength
- `fit_intercept` (bool): Whether to calculate the intercept
- `normalize` (bool): Whether to normalize features
- `max_iter` (int): Maximum number of iterations
- `tol` (float): Tolerance for convergence

### Elastic Net

Linear regression with L1 and L2 regularization.

```python
from tensorcore.sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Parameters:**
- `alpha` (float): Regularization strength
- `l1_ratio` (float): Balance between L1 and L2 regularization
- `fit_intercept` (bool): Whether to calculate the intercept
- `normalize` (bool): Whether to normalize features
- `max_iter` (int): Maximum number of iterations
- `tol` (float): Tolerance for convergence

### Logistic Regression

Logistic regression for binary and multiclass classification.

```python
from tensorcore.sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

**Parameters:**
- `penalty` (str): Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
- `C` (float): Inverse regularization strength
- `fit_intercept` (bool): Whether to calculate the intercept
- `max_iter` (int): Maximum number of iterations
- `tol` (float): Tolerance for convergence
- `multi_class` (str): Strategy for multiclass classification

## 🔄 Preprocessing

### StandardScaler

Standardize features by removing the mean and scaling to unit variance.

```python
from tensorcore.sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)
X_original = scaler.inverse_transform(X_scaled)
```

**Parameters:**
- `with_mean` (bool): Whether to center the data
- `with_std` (bool): Whether to scale to unit variance
- `copy` (bool): Whether to copy the data

### MinMaxScaler

Transform features by scaling each feature to a given range.

```python
from tensorcore.sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_original = scaler.inverse_transform(X_scaled)
```

**Parameters:**
- `feature_range` (tuple): Desired range of transformed data
- `copy` (bool): Whether to copy the data

### LabelEncoder

Encode target labels with value between 0 and n_classes-1.

```python
from tensorcore.sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_categorical)
y_decoded = encoder.inverse_transform(y_encoded)
```

### OneHotEncoder

Encode categorical integer features as a one-hot numeric array.

```python
from tensorcore.sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(categories='auto', drop='if_binary')
X_encoded = encoder.fit_transform(X_categorical)
X_decoded = encoder.inverse_transform(X_encoded)
```

**Parameters:**
- `categories` (str or list): Categories for each feature
- `drop` (str): Whether to drop one category
- `sparse` (bool): Whether to return sparse matrix
- `dtype` (str): Data type of output

## 📊 Metrics

### Classification Metrics

```python
from tensorcore.sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
```

### Regression Metrics

```python
from tensorcore.sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Classification Report

```python
from tensorcore.sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=['Class A', 'Class B'])
print(report)
```

## 🔍 Model Selection

### Train-Test Split

```python
from tensorcore.sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True
)
```

**Parameters:**
- `test_size` (float): Proportion of dataset to include in the test split
- `random_state` (int): Random seed for reproducibility
- `shuffle` (bool): Whether to shuffle the data before splitting

### Cross-Validation

```python
from tensorcore.sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean CV score: {scores.mean():.4f}")
print(f"Std CV score: {scores.std():.4f}")
```

**Parameters:**
- `cv` (int): Number of cross-validation folds
- `scoring` (str): Scoring metric ('accuracy', 'precision', 'recall', 'f1', 'r2', 'mse', 'mae')

### Grid Search

```python
from tensorcore.sklearn.model_selection import GridSearchCV

param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}

grid_search = GridSearchCV(ElasticNet(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## 🚀 Complete Example

```python
import tensorcore as tc
from tensorcore.sklearn.linear_model import LogisticRegression
from tensorcore.sklearn.preprocessing import StandardScaler
from tensorcore.sklearn.metrics import accuracy_score, classification_report
from tensorcore.sklearn.model_selection import train_test_split, cross_val_score

# Generate classification data
X = tc.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [1, 1], [2, 2]])
y = tc.tensor([0, 0, 0, 0, 1, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
print(f"CV scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

# Classification report
report = classification_report(y_test, y_pred)
print(report)
```

## 🔄 Migration from Scikit-learn

The TensorCore sklearn API is designed to be compatible with scikit-learn. Most code can be migrated with minimal changes:

```python
# Scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# TensorCore (minimal changes)
from tensorcore.sklearn.linear_model import LinearRegression
from tensorcore.sklearn.preprocessing import StandardScaler
from tensorcore.sklearn.metrics import accuracy_score
```

## 🎯 Key Features

- **Familiar API**: Follows scikit-learn conventions
- **High Performance**: Built on TensorCore's optimized C++ backend
- **Educational**: Clear, well-documented implementations
- **Compatible**: Easy migration from scikit-learn
- **Comprehensive**: Covers most common ML algorithms and utilities

## 📈 Performance

TensorCore's sklearn implementation is optimized for performance:

- **SIMD Optimizations**: Vectorized operations
- **Memory Efficient**: Optimized memory usage
- **Parallel Processing**: Multi-threaded operations where applicable
- **C++ Backend**: Fast numerical computations

## 🚧 Future Enhancements

Planned additions to the sklearn module:

- **Tree Models**: DecisionTreeClassifier, DecisionTreeRegressor, RandomForest
- **Clustering**: KMeans, DBSCAN, AgglomerativeClustering
- **SVM**: Support Vector Machines for classification and regression
- **Naive Bayes**: GaussianNB, MultinomialNB, BernoulliNB
- **Ensemble Methods**: VotingClassifier, BaggingClassifier, AdaBoostClassifier

## 📚 Educational Value

The TensorCore sklearn implementation is designed for educational purposes:

- **Transparent**: Clear, readable implementations
- **Well-Documented**: Comprehensive documentation and examples
- **Mathematical**: Shows the underlying mathematics
- **Modular**: Easy to understand and extend

This makes TensorCore an excellent tool for learning machine learning algorithms and understanding how they work under the hood.
