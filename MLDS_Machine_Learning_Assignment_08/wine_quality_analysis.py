import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score

# Load the dataset
file_path = 'WineQT.csv'
data = pd.read_csv(file_path)

# Drop the 'Id' column as it's not needed for analysis
data = data.drop('Id', axis=1)

# Calculate the correlation matrix
corr_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split data into features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert quality scores into binary classification: good (1) and bad (0)
y_binary = (y >= 6).astype(int)

# Split the data into training and testing sets for classification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

# Split the data into training and testing sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)

# Random Forest Regressor
rfr = RandomForestRegressor(random_state=42)

# Perform cross-validation for classification
cv_scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring='accuracy')

# Perform cross-validation for regression
cv_scores_reg = cross_val_score(rfr, X_train_reg, y_train_reg, cv=kf, scoring='neg_mean_squared_error')

# Fit and evaluate the models for classification
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Fit and evaluate the models for regression
rfr.fit(X_train_reg, y_train_reg)
y_pred_reg = rfr.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

# Print results
print("Classification Metrics:")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")
print(f"Test Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:")
print(classification_report_str)

print("\nRegression Metrics:")
print(f"Cross-Validation MSE: {-cv_scores_reg.mean():.2f}")
print(f"Test MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
