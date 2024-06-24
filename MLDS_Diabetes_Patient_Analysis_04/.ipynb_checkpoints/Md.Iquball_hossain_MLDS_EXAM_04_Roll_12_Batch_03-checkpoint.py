import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step I: Load the diabetes dataset and split into training and testing sets
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Step II: Linear regression without PCA
start_time = time.time()
lr_no_pca = LinearRegression()
lr_no_pca.fit(X_train, y_train)
train_eval_time_no_pca = time.time() - start_time
y_pred_no_pca = lr_no_pca.predict(X_test)
mse_no_pca = mean_squared_error(y_test, y_pred_no_pca)

# Step III: Apply PCA to reduce dimensionality to 5 features
# Subtract mean from each feature
X_train_mean = np.mean(X_train, axis=0)
X_train_centered = X_train - X_train_mean

# Compute covariance matrix
cov_matrix = np.cov(X_train_centered, rowvar=False)

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
top_indices = sorted_indices[:5]
top_eigenvectors = eigenvectors[:, top_indices]

# Transform training and testing sets
X_train_pca = np.dot(X_train_centered, top_eigenvectors)
X_test_pca = np.dot(X_test - X_train_mean, top_eigenvectors)

# Step IV: Linear regression with PCA
start_time = time.time()
lr_with_pca = LinearRegression()
lr_with_pca.fit(X_train_pca, y_train)
train_eval_time_with_pca = time.time() - start_time
y_pred_with_pca = lr_with_pca.predict(X_test_pca)
mse_with_pca = mean_squared_error(y_test, y_pred_with_pca)

# Step V: Comparison
print("Comparison of Linear Regression with and without PCA:")
print("a) Time taken for training and evaluation:")
print("   Without PCA: {:.4f} seconds".format(train_eval_time_no_pca))
print("   With PCA: {:.4f} seconds".format(train_eval_time_with_pca))
print("b) Mean Squared Error (MSE) on testing set:")
print("   Without PCA: {:.4f}".format(mse_no_pca))
print("   With PCA: {:.4f}".format(mse_with_pca))
