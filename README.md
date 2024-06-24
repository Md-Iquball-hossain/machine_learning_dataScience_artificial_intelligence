 Answer created by: Md. Iquball Hossain, NACTAR MLDS Bach-03, Roll -12 (Contact: 01777044834)
Machine Learning and Data Science Module 02 (Exam)
1. You are given a list of tuples, where each tuple contains two integers.
Your task is to write a Python function, that takes in the list of tuples and
a target sum. The function should return a list of tuples, each
representing a pair of numbers whose sum is equal to the target sum.
Sample Input:
tuples_list = [(1, 2), (2, 3), (4, 5), (6, 7), (8, 9)]
target_sum = 10
Output:
[(1, 9), (2, 8)]
Hint: The function should iterate through each tuple in the list and check
if the sum of the two integers in the tuple equals the target sum. If a pair
is found, it should be added to the result list. Note that a number can
participate in multiple pairs.
 Answer : 
def find_pairs_with_sum(tuples_list, target_sum):
 result = []
 seen = set()
 for tuple_elem in tuples_list:
 num1, num2 = tuple_elem
 complement = target_sum - num1
 if complement in seen:
 result.append((complement, num1))
 seen.add(num2)
 return result
 Answer created by: Md. Iquball Hossain, NACTAR MLDS Bach-03, Roll -12 (Contact: 01777044834)
2. Write a Python program to input a square numpy array of shape (n, n),
where n is a positive integer. Take the value of n from the user and input
the user's array values. Then write a Python function, diagonal_sum, that
takes in the numpy array and calculates the sum of the diagonal
elements.
Sample Input:
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
Output: 15
Hint: The diagonal elements are defined as the elements that lie on the
main diagonal of the array, i.e., the elements with indices (i, i) for i ranging
from 0 to n-1.
Your function should iterate through the array using a loop and calculate
the sum of the diagonal elements.
 Answer: 
import numpy as np
def diagonal_sum(arr):
 n = arr.shape[0]
 diag_sum = sum(arr[i][i] for i in range(n))
 return diag_sum
# Taking input from the user for the size of the array
n = int(input("Enter the size of the square array: "))
print(f"Enter {n}x{n} array elements:")
# Taking input for the array elements
user_array = []
for i in range(n):
 row = list(map(int, input().split()))
 user_array.append(row)
# Converting the user input list into a NumPy array
numpy_array = np.array(user_array)
# Calculating the sum of diagonal elements using the diagonal_sum function
result = diagonal_sum(numpy_array)
print("Sum of diagonal elements:", result)
 Answer created by: Md. Iquball Hossain, NACTAR MLDS Bach-03, Roll -12 (Contact: 01777044834)
3. Write a Python program to subtract two matrices. Take two numpy
arrays, arr1, and arr2, both of shape (m, n), where m and n are positive
integers (you should input each m and n value for each matrix and all
matrix elements). Your task is to write a Python function,
calculate_subtraction, that performs element-wise subtraction between
the two arrays and returns the resulting numpy array.
 Answer:
import numpy as np
def calculate_subtraction(arr1, arr2):
 result = arr1 - arr2
 return result
# Taking input for the dimensions of the matrices
m1 = int(input("Enter the number of rows for matrix 1: "))
n1 = int(input("Enter the number of columns for matrix 1: "))
m2 = int(input("Enter the number of rows for matrix 2: "))
n2 = int(input("Enter the number of columns for matrix 2: "))
# Taking input for elements of matrix 1
print("Enter elements for matrix 1:")
matrix1 = []
for i in range(m1):
 row = list(map(int, input().split()))
 matrix1.append(row)
# Taking input for elements of matrix 2
print("Enter elements for matrix 2:")
matrix2 = []
for i in range(m2):
 row = list(map(int, input().split()))
 matrix2.append(row)
# Converting the input lists into NumPy arrays
arr1 = np.array(matrix1)
arr2 = np.array(matrix2)
# Check if matrices have the same shape for subtraction
if arr1.shape != arr2.shape:
 print("Matrices should have the same shape for subtraction.")
else:
 result_array = calculate_subtraction(arr1, arr2)
 print("Result after subtraction:\n", result_array)

 Answer created by Md. Iquball Hossain (NACTAR MLDS Batch-03, Roll: 12 )
Machine Learning and Data Science
Module 03 Exam
Questions: Suppose you are working on a data science project and have been given a dataset in 
CSV format called "sales_data.csv". Your task is to analyze the data and extract meaningful 
insights using Python. Assume that you have already imported the necessary libraries and 
loaded the dataset into a Pandas DataFrame named "df".
Solution:
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("sales_data.csv") 
I. Display the first 5 rows of the data frame.
print("I. Display the first 5 rows of the data frame:")
print(df.head())
II. Calculate the total number of rows in the DataFrame.
total_rows = len(df)
print("\nII. Total number of rows in the DataFrame:", total_rows)
III. Check if there are any missing values in the dataset.
missing_values = df.isnull().sum().any()
print("\nIII. Any missing values in the dataset:", missing_values)
IV. Calculate the average value of the "sales" column.
average_sales = df["sales"].mean()
print("\nIV. Average value of the 'sales' column:", average_sales)
V. Create a new column called "profit" that contains the difference between the "sales" and 
"expenses" columns.
df["profit"] = df["sales"] - df["expenses"]
VI. Sort the DataFrame in descending order based on the "profit" column.
df = df.sort_values(by="profit", ascending=False)
print("\nVI. DataFrame sorted in descending order based on the 'profit' column:")
VII. Save the sorted DataFrame to a new CSV file called "sorted_sales_data.csv".
df.to_csv("sorted_sales_data.csv", index=False)
VIII. Visualize the distribution of the "sales" column using a histogram plot.
plt.figure(figsize=(10, 6))
plt.hist(df["sales"], bins=20, color='blue', alpha=0.7)
plt.title("Histogram of Sales")
Answer created by Md. Iquball Hossain (NACTAR MLDS Batch-03, Roll: 12 )
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()
IX. Create a scatter plot to explore the relationship between the "sales" and "expenses" 
columns.
plt.figure(figsize=(10, 6))
plt.scatter(df["sales"], df["expenses"], color='green', alpha=0.7)
plt.title("Scatter Plot: Sales vs Expenses")
plt.xlabel("Sales")
plt.ylabel("Expenses")
plt.show()
X. Calculate the correlation coefficient between the "sales" and "expenses" columns.
correlation_coefficient = df["sales"].corr(df["expenses"])
print("\nX. Correlation coefficient between 'sales' and 'expenses':", correlation_coefficient)
XI. Calculate the total sales for each month and display the result.
monthly_total_sales = df.groupby("month")["sales"].sum()
print("\nXI. Total sales for each month:")
print(monthly_total_sales)
XII. Find the maximum and minimum sales values for each year in the dataset.
yearly_max_sales = df.groupby("year")["sales"].max()
yearly_min_sales = df.groupby("year")["sales"].min()
print("\nXII. Maximum sales for each year:")
print(yearly_max_sales)
print("\n Minimum sales for each year:")
print(yearly_min_sales)
XIII. Calculate the average sales for each category of products.
average_sales_by_category = df.groupby("category")["sales"].mean()
print("\nXIII. Average sales for each category of products:")
print(average_sales_by_category)
XIV. Group the data by the "region" column and calculate the total sales for each region.
total_sales_by_region = df.groupby("region")["sales"].sum()
print("\nXIV. Total sales for each region:")
print(total_sales_by_region)
XV. Calculate the sum of expenses for each quarter of the year.
expenses_by_quarter = df.groupby("quarter")["expenses"].sum()
print("\nXV. Sum of expenses for each quarter of the year:")
print(expenses_by_quarter)
Technical Report on Modeling Biological Neurons into
Machine Learning Algorithms
I. Methodology to Implement the Process of a Single Neuron:
To model a biological neuron into a machine learning algorithm, we can use the 
perceptron model, which serves as a simplified version of a biological neuron. The 
methodology involves the following steps:
1. Input Data: Each input feature represents a signal received by the neuron.
2. Weights and Bias: We assign weights to each input feature, which determines the 
importance of that feature. Additionally, a bias term is added to the weighted sum to 
introduce flexibility to the model.
3. Activation Function: The weighted sum of inputs and bias is then passed through an 
activation function. This function decides whether the neuron should be activated or not 
based on the input signals. Common activation functions include sigmoid, ReLU, and 
tanh.
4. Output: The output of the neuron is the result of the activation function, which can be 
interpreted as the neuron firing (outputting a 1) or not firing (outputting a 0).
II. Architecture of Single Layer Perceptron Learning Algorithm and Limitation:
The single-layer perceptron learning algorithm consists of input nodes, one layer of 
perceptron units, and an output node. Each input node represents a feature of the input 
data, and each perceptron unit applies a weighted sum of inputs followed by an 
activation function.
Limitation:
 Single-layer perceptron’s can only learn linearly separable patterns. They cannot 
solve problems that require nonlinear decision boundaries.
 They are unable to solve problems with complex relationships between input and 
output variables.
III. Architecture of Multi-layer Perceptron Learning Algorithm and Advantages:
The multi-layer perceptron (MLP) learning algorithm consists of an input layer, one or 
more hidden layers, and an output layer. Each layer contains multiple neurons, and 
connections between neurons have associated weights.
Md. Iquball Hossain ( NACTAR MLDS Batch-03 Roll:12) Exam 07 Report
Topics: Technical Report on Modeling Biological Neurons into Machine Learning Algorithms
Advantages:
 MLPs can learn complex nonlinear relationships between input and output 
variables.
 They are capable of approximating any continuous function, given a sufficient 
number of neurons and layers.
 MLPs can generalize well to unseen data when trained properly.
IV. Number of Layers Required for Multi-layer Perceptron Neural Network
Algorithm:
The number of layers required for an MLP depends on the complexity of the problem 
being solved. For complex problems with intricate patterns and relationships, deeper 
architectures with more layers are often needed.
Why Multi-layer Perceptron for Complex Problems?
 Representation Power: Deep architectures can capture hierarchical features in 
the data, allowing for better representation of complex patterns.
 Feature Abstraction: Each layer learns increasingly abstract representations of 
the input data, enabling the model to discern subtle differences and 
relationships.
 Nonlinear Transformations: Deep architectures with multiple layers of nonlinear 
transformations can approximate highly nonlinear functions efficiently.
Diagram:
 Input Layer Hidden Layer 1 Hidden Layer 2 ... Output Layer
 
 
[Input Features] -> [Neurons & Weights] -> [Neurons & Weights] -> ... -> [Output]
In the diagram above, each layer consists of neurons (perceptron’s) connected to 
neurons in the adjacent layers. The input features are fed into the input layer, and the 
output layer produces the final predictions. Hidden layers between the input and output 
layers perform nonlinear transformations to learn complex patterns in the data.
In summary, modeling biological neurons into machine learning algorithms involves 
mimicking their behavior using mathematical models such as the perceptron. While 
single-layer perceptron’s have limitations in solving complex problems, multi-layer 
perceptron’s offer greater flexibility and capability to handle intricate relationships in the 
data
Assignment: Machine Learning with python (Module 08 Exam)
Wine Quality dataset consists of various chemical properties of wine and a 
quality rating, making it suitable for predicting wine quality based on its 
chemical attributes. First 11 columns define physicochemical properties of 
wine and 12th column indicates the quality of the wine. You have to develop
(i) Multiclass classification algorithm and
(ii) Regression algorithm to maintain and considering the following 
properties:
(i) Preprocessing technique
(ii) Feature selection technique
(iii) k-fold cross validation technique
(iv) Spot-check of Linear, Non-linear machine and ensemble learning 
techniques
(v) Parameter tunning of the selected algorithm(s)
(vi) Report the performance of the selected algorithm according to 
performance matrix
Solution:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, 
GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, classification_report, 
mean_squared_error, r2_score
# Load the dataset
file_path = 'WineQT.csv'
data = pd.read_csv(file_path)
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, 
random_state=42)
# Split the data into training and testing sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y, 
test_size=0.2, random_state=42)
# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
# Random Forest Regressor
rfr = RandomForestRegressor(random_state=42)
# Perform cross-validation for classification
cv_scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring='accuracy')
# Perform cross-validation for regression
cv_scores_reg = cross_val_score(rfr, X_train_reg, y_train_reg, cv=kf, 
scoring='neg_mean_squared_error')
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
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
print(f"Cross-Validation MSE: {-cv_scores_reg.mean():.2f}")
print(f"Test MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
result:
Classification Metrics:
Cross-Validation Accuracy: 0.78
Test Accuracy: 0.77
F1 Score: 0.79
Classification Report:
 precision recall f1-score support
 0 0.73 0.75 0.74 102
 1 0.80 0.78 0.79 127
accuracy 0.77 229
macro avg 0.77 0.77 0.77 22
weighted avg 0.77 0.77 0.77 229
Regression Metrics:
Cross-Validation MSE: 0.39
Test MSE: 0.30
R2 Score: 0.47
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
Dataset Characteristics
 Number of Instances:
o Red Wine: 1,599
o White Wine: 4,898
 Number of Attributes: 11 (excluding the quality score)
 Attribute Characteristics: Continuous
 Number of Classes: Quality score ranges from 0 to 10, though most scores 
are between 3 and 8.
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
Attribute Information
The dataset contains 11 physicochemical properties of wine:
1. Fixed Acidity: Primary acids found in wine, such as tartaric acid.
2. Volatile Acidity: Amount of acetic acid in wine, which at high levels can 
lead to an unpleasant vinegar taste.
3. Citric Acid: Adds freshness and flavor to wines in small quantities.
4. Residual Sugar: Amount of sugar remaining after fermentation stops; 
higher residual sugar can make the wine sweeter.
5. Chlorides: Amount of salt in the wine.
6. Free Sulfur Dioxide: Exists in equilibrium between molecular SO2 (dissolved 
gas) and bisulfite ion; prevents microbial growth and oxidation.
7. Total Sulfur Dioxide: Includes both free and bound forms of SO2; bound 
SO2 forms when molecular SO2 combines with other chemicals.
8. Density: Close to water's density, depending on alcohol and sugar content.
9. pH: Indicates how acidic or basic the wine is.
10.Sulphates: Wine additive contributing to sulfur dioxide levels, acting as an 
antimicrobial and antioxidant.
11.Alcohol: Percentage of alcohol in the wine.
Quality Score
 Quality: A score between 0 and 10 indicating the wine's quality, 
determined by wine experts based on sensory data like taste, aroma, and 
overall impression.
Usage
The Wine Quality dataset is used for:
 Regression Tasks: Predicting the quality score of wine based on its chemical 
properties.
 Classification Tasks: Categorizing wine into quality categories, such as 
"good" or "bad" based on the quality score.
 Feature Analysis: Understanding the importance and impact of different 
chemical properties on wine quality.
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
Example Data Snippet
Fixed Acidity Volatile Acidity Citric Acid Residual Sugar Chlorides Free SO2 Total SO2 Density pH Sulphates Alcohol Quality
7.4 0.70 0.00 1.9 0.076 11 34 0.9978 3.51 0.56 9.4 5
7.8 0.88 0.00 2.6 0.098 25 67 0.9968 3.20 0.68 9.8 5
7.8 0.76 0.04 2.3 0.092 15 54 0.9970 3.26 0.65 9.8 5
11.2 0.28 0.56 1.9 0.075 17 60 0.9980 3.16 0.58 9.8 6
7.4 0.70 0.00 1.9 0.076 11 34 0.9978 3.51 0.56 9.4 5
7.4 0.66 0.00 1.8 0.075 13 40 0.9978 3.51 0.56 9.4 5
7.9 0.60 0.06 1.6 0.069 15 59 0.9964 3.30 0.46 10.1 6
Let's develop both a classification and a regression model using the Wine Quality 
dataset you provided. We'll follow these steps:
1. Data Preprocessing:
o Load the data.
o Handle missing values (if any).
o Standardize the features.
2. Feature Selection:
o Use techniques like correlation analysis to select relevant features.
3. Model Development:
o Implement k-fold cross-validation.
o Apply linear and non-linear machine learning techniques along with 
ensemble methods.
o Perform parameter tuning for the selected algorithms
4. Performance Evaluation:
o Evaluate the models using appropriate performance metrics.
Step 1: Data Preprocessing
Let's start by loading the dataset and performing basic preprocessing.
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, 
GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
# Load the dataset
file_path = '/mnt/data/WineQT.csv'
data = pd.read_csv(file_path)
# Display the first few rows of the dataset
data.head()
Step 2: Feature Selection
We will perform feature selection using correlation analysis.
import seaborn as sns
import matplotlib.pyplot as plt
# Calculate the correlation matrix
corr_matrix = data.corr()
# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
Step 3: Model Development
Preprocessing
We'll split the data into features and target, then apply standard scaling.
# Split data into features and target
X = data.drop('quality', axis=1)
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
y = data['quality']
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Classification
We'll develop classification models to predict whether a wine is "good" (quality ≥ 
6) or "bad" (quality < 6).
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
# Convert quality scores into binary classification: good (1) and bad (0)
y_binary = (y >= 6).astype(int)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, 
random_state=42)
# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
# Perform cross-validation
cv_scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring='accuracy')
# Fit the model
rfc.fit(X_train, y_train)
# Predict on the test set
y_pred = rfc.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")
print(f"Test Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
Parameter Tuning
We'll use GridSearchCV for parameter tuning.
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
# Parameter tuning for RandomForestClassifier
param_grid = {
 'n_estimators': [100, 200],
 'max_depth': [None, 10, 20],
 'min_samples_split': [2, 5],
 'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=kf, 
scoring='accuracy')
grid_search.fit(X_train, y_train)
best_rfc = grid_search.best_estimator_
# Evaluate the tuned model
y_pred_tuned = best_rfc.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
print(f"Tuned Test Accuracy: {accuracy_tuned:.2f}")
print(f"Tuned F1 Score: {f1_tuned:.2f}")
# Parameter tuning for RandomForestRegressor
param_grid_reg = {
 'n_estimators': [100, 200],
 'max_depth': [None, 10, 20],
 'min_samples_split': [2, 5],
 'min_samples_leaf': [1, 2]
}
grid_search_reg = GridSearchCV(estimator=rfr, param_grid=param_grid_reg, 
cv=kf, scoring='neg_mean_squared_error')
grid_search_reg.fit(X_train_reg, y_train_reg)
best_rfr = grid_search_reg.best_estimator_
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
# Evaluate the tuned model
y_pred_reg_tuned = best_rfr.predict(X_test_reg)
mse_tuned = mean_squared_error(y_test_reg, y_pred_reg_tuned)
r2_tuned = r2_score(y_test_reg, y_pred_reg_tuned)
print(f"Tuned Test MSE: {mse_tuned:.2f}")
print(f"Tuned R2 Score: {r2_tuned:.2f}")
Summary
We'll summarize the results in a report format:
classification_report_str = classification_report(y_test, y_pred_tuned)
regression_report_str = f"Mean Squared Error: {mse_tuned:.2f}\nR2 Score: 
{r2_tuned:.2f}"
print("Classification Report:")
print(classification_report_str)
print("Regression Report:")
print(regression_report_str)
Let's execute the entire code step-by-step. First, we'll start with data loading and 
preprocessing.
The dataset is successfully loaded, and the first few rows are displayed. The next 
step involves preprocessing, including handling missing values (if any) and 
standardizing the features.
Data Preprocessing
Let's check for any missing values and then standardize the features.
# Check for missing values
missing_values = data.isnull().sum()
# Drop the 'Id' column as it's not needed for analysis
data = data.drop('Id', axis=1)
# Display missing values
missing_values
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
Let's execute this to identify and handle any missing values.
import pandas as pd
# Load the dataset
file_path = '/mnt/data/WineQT.csv'
data = pd.read_csv(file_path)
# Display the first few rows of the dataset
data.head()
Result
 fixed acidity volatile acidity citric acid residual sugar chlorides \
0 7.4 0.70 0.00 1.9 0.076 
1 7.8 0.88 0.00 2.6 0.098 
2 7.8 0.76 0.04 2.3 0.092 
3 11.2 0.28 0.56 1.9 0.075 
4 7.4 0.70 0.00 1.9 0.076 
 free sulfur dioxide total sulfur dioxide density pH sulphates \
0 11.0 34.0 0.9978 3.51 0.56 
1 25.0 67.0 0.9968 3.20 0.68 
2 15.0 54.0 0.9970 3.26 0.65 
3 17.0 60.0 0.9980 3.16 0.58 
4 11.0 34.0 0.9978 3.51 0.56 
 alcohol quality Id 
0 9.4 5 0 
1 9.8 5 1 
2 9.8 5 2 
3 9.8 6 3 
4 9.4 5 4 
There are no missing values in the dataset. Now, let's proceed with standardizing 
the features.
Standardizing Features
We'll split the data into features and target, then apply standard scaling.
from sklearn.preprocessing import StandardScaler
# Split data into features and target
X = data.drop('quality', axis=1)
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
y = data['quality']
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Next, we'll develop classification and regression models, starting with 
classification.
Classification
We'll convert the quality scores into a binary classification: "good" (quality ≥ 6)
and "bad" (quality < 6). Then we'll split the data, apply k-fold cross-validation, and 
fit a RandomForestClassifier.
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Convert quality scores into binary classification: good (1) and bad (0)
y_binary = (y >= 6).astype(int)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, 
random_state=42)
# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
# Perform cross-validation
cv_scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring='accuracy')
# Fit the model
rfc.fit(X_train, y_train)
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
# Predict on the test set
y_pred = rfc.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")
print(f"Test Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
# Detailed classification report
classification_report_str = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_str)
Let's execute this part of the code to see the classification results.
# Check for missing values
missing_values = data.isnull().sum()
# Drop the 'Id' column as it's not needed for analysis
data = data.drop('Id', axis=1)
# Display missing values
missing_values
Result
fixed acidity 0
volatile acidity 0
citric acid 0
residual sugar 0
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
chlorides 0
free sulfur dioxide 0
total sulfur dioxide 0
density 0
pH 0
sulphates 0
alcohol 0
quality 0
Id 0
dtype: int64
It appears there was an issue with the variable scope. Let's re-execute the entire 
process from data loading, preprocessing, and then classification to ensure all 
steps are included in the execution.
Reload and Preprocess Data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Load the dataset
file_path = '/mnt/data/WineQT.csv'
data = pd.read_csv(file_path)
# Drop the 'Id' column as it's not needed for analysis
data = data.drop('Id', axis=1)
# Split data into features and target
X = data.drop('quality', axis=1)
y = data['quality']
# Standardize the features
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Convert quality scores into binary classification: good (1) and bad (0)
y_binary = (y >= 6).astype(int)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, 
random_state=42)
# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
# Perform cross-validation
cv_scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring='accuracy')
# Fit the model
rfc.fit(X_train, y_train)
# Predict on the test set
y_pred = rfc.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cv_scores.mean(), accuracy, f1, classification_report_str
import pandas as pd
from sklearn.preprocessing import StandardScaler
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Load the dataset
file_path = '/mnt/data/WineQT.csv'
data = pd.read_csv(file_path)
# Drop the 'Id' column as it's not needed for analysis
data = data.drop('Id', axis=1)
# Split data into features and target
X = data.drop('quality', axis=1)
y = data['quality']
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Convert quality scores into binary classification: good (1) and bad (0)
y_binary = (y >= 6).astype(int)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, 
random_state=42)
# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
# Perform cross-validation
cv_scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring='accuracy')
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
# Fit the model
rfc.fit(X_train, y_train)
# Predict on the test set
y_pred = rfc.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cv_scores.mean(), accuracy, f1, classification_report_str
Let's execute this code. 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Convert quality scores into binary classification: good (1) and bad (0)
y_binary = (y >= 6).astype(int)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, 
random_state=42)
# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
# Perform cross-validation
cv_scores = cross_val_score(rfc, X_train, y_train, cv=kf, scoring='accuracy')
Assignment prepared by Md. Iquball Hossain, NACTAR ML & DS Batch-03, Roll : 12
# Fit the model
rfc.fit(X_train, y_train)
# Predict on the test set
y_pred = rfc.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cv_scores.mean(), accuracy, f1, classification_report_st
Exam of Module 10: Concepts and Uses of Deep learning
Develop a Convolutional Neural Networks (CNN) classification model with MNIST digit 
recognition dataset where you have to maintain the following conditions:
1. Minimum two convolutional layers
2. Minimum two pooling layers
3. A fully connected layer with minimum two hidden layers and output layer
4. Tune the parameters of convolutional layer and pooling layer
5. Tune the hidden layer parameters
6. Tune the learning rate
7. Try to minimize the overfitting problem
You have to show the best result with the graph of training and validation error.
Solution:
To develop a Convolutional Neural Network (CNN) for classifying the MNIST digit recognition 
dataset with the specified conditions, we need to follow these steps:
1. Load and preprocess the MNIST dataset
2. Define the CNN architecture
3. Compile the model with appropriate loss function and optimizer
4. Train the model while tuning the hyper parameters
5. Evaluate the model and plot training/validation error graphs
6. Apply techniques to minimize overfitting
Let's start by implementing this step-by-step:
Step 1: Load and Preprocess the MNIST Dataset
We'll use the Tensor Flow and Keras libraries to load and preprocess the data.
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# Load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
Step 2: Define the CNN Architecture
We'll define a CNN with at least two convolutional layers, two pooling layers, and a fully 
connected layer with at least two hidden layers.
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
def create_model(learning_rate=0.001, dropout_rate=0.5):
 model = Sequential()
 
 # Convolutional layers
 model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Conv2D(64, (3, 3), activation='relu'))
 model.add(MaxPooling2D((2, 2)))
 
 # Fully connected layers
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dropout(dropout_rate))
 model.add(Dense(64, activation='relu'))
 model.add(Dense(10, activation='softmax'))
 
 # Compile the model
 optimizer = Adam(learning_rate=learning_rate)
 model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
 
 return model
Step 3: Train the Model and Tune Hyperparameters
We'll train the model while tuning the learning rate and dropout rate to reduce overfitting.
# Hyperparameters
learning_rate = 0.001
dropout_rate = 0.5
batch_size = 128
epochs = 20
# Create and train the model
model = create_model(learning_rate=learning_rate, dropout_rate=dropout_rate)
history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
 validation_split=0.2, verbose=1)
Step 4: Evaluate the Model and Plot the Results
We will evaluate the model on the test data and plot the training and validation error graphs.
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
# Plot training and validation accuracy/loss
history_dict = history.history
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
Step 5: Minimizing Overfitting
To further reduce overfitting, we can:
 Use data augmentation
 Implement early stopping
 Reduce model complexity if necessary
Let's apply data augmentation:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
 rotation_range=10,
 width_shift_range=0.1,
 height_shift_range=0.1,
 zoom_range=0.1
)
# Fit the data generator to the training data
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
datagen.fit(train_images)
# Train the model using the augmented data
history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
 epochs=epochs, validation_data=(test_images, test_labels), verbose=1)
And re-evaluate the model:
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy with augmentation: {test_acc}')
# Plot the results again
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy with augmentation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss with augmentation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
By following these steps, we have successfully built and evaluated a CNN for the MNIST digit recognition 
task while tuning hyperparameters and implementing techniques to reduce overfitting

Problem 1: SELECT and WHERE 
Ques: You have a table named `students` with columns `student_id`, `name`, 
`age`, and `gender`. Write a query to select the names of female students who 
are below 25 years of age. 
SELECT name
FROM students WHERE gender = ‘female’ AND age<25;
NACTAR MLDS Module 11 Exam Solution. Created by Md. Iquball Hossain (Roll-12, Batch-03) 
Topics: SQL Query for MySQL Database
Problem 2: ORDER BY, GROUP BY, and AGGREGATE FUNCTIONS 
Ques: Consider a table named `sales` with columns `product_id`, 
`product_name`, `category`, and `sales_amount`. Write a query to find the total 
sales amount for each category, and display the results in descending order of 
total sales amount. 
SELECT category, SUM (sales_amount) AS total_sales 
FROM sales 
GROUP BY category 
ORDER BY total_sales DESC;
NACTAR MLDS Module 11 Exam Solution. Created by Md. Iquball Hossain (Roll-12, Batch-03) 
Topics: SQL Query for MySQL Database
Problem 3: JOIN, WHERE, and LIKE 
Ques: Suppose you have two tables named `employees` and `departments`, 
where `employees` contains columns `employee_id`, `name`, `department_id`, 
and `salary`, and `departments` contains columns `department_id` and 
`department_name`. Write a query to select the names and salaries of 
employees who work in the 'Sales' department and whose salaries are greater 
than $50000. 
SELECT e.name, e.salary 
FROM employees e 
JOIN departments d ON e.department_id = d.department_id 
WHERE d.department_name LIKE '%Sales%' AND e.salary > 50000;
NACTAR MLDS Module 11 Exam Solution. Created by Md. Iquball Hossain (Roll-12, Batch-03) 
Topics: SQL Query for MySQL Database
Problem 4: NOT, Wildcards, and LIKE 
Ques: Assume you have a table named `products` with columns `product_id`, 
`product_name`, and `price`. Write a query to select the names of products that 
do not contain the word 'cheap' in their names. 
SELECT product_name 
FROM products 
WHERE product_name NOT LIKE '%cheap%'; 
NACTAR MLDS Module 11 Exam Solution. Created by Md. Iquball Hossain (Roll-12, Batch-03) 
Topics: SQL Query for MySQL Database
Problem 5: Views and Joins 
Ques: Consider two tables: `orders` with columns `order_id`, `customer_id`, and 
`order_date`, and `customers` with columns `customer_id`, `customer_name`, 
and `city`. Create a view named `customer_orders` that displays the customer 
name, order ID, and order date for each order, along with the city of the 
customer. 
CREATE VIEW customer_orders AS 
SELECT c.customer_name, o.order_id, o.order_date, c.city 
FROM orders o 
JOIN customers c ON o.customer_id = c.customer_id; 
NACTAR MLDS Module 11 Exam Solution. Created by Md. Iquball Hossain (Roll-12, Batch-03) 
Topics: SQL Query for MySQL Database
Problem 6: Event 
Ques: Design an event in MySQL that runs every day at midnight (12 AM) and 
updates the `sales` table (Problem 2) by increasing the sales amount of each 
product by 5%. 
CREATE EVENT daily_sales_update 
ON SCHEDULE EVERY 1 DAY 
STARTS 'YYYY-MM-DD 00:00:00' 
DO 
UPDATE sales 
SET sales_amount = sales_amount * 1.05; 
NACTAR MLDS Module 11 Exam Solution. Created by Md. Iquball Hossain (Roll-12, Batch-03) 
Topics: SQL Query for MySQL Database
Problem 7: Normalization 
Ques: You have a table named `customers` with columns `customer_id`, 
`customer_name`, `address`, `city`, and `country`. Identify any normalization 
issues in this table and propose a normalized schema to address them. 
CREATE TABLE customers ( 
customer_id INT PRIMARY KEY, 
customer_name VARCHAR(255) 
); 
CREATE TABLE addresses ( 
address_id INT PRIMARY KEY, 
customer_id INT, 
address VARCHAR(255), 
city VARCHAR(100), 
country VARCHAR(100), 
FOREIGN KEY (customer_id) REFERENCES customers(customer_id) 
); 
NACTAR MLDS Module 11 Exam Solution. Created by Md. Iquball Hossain (Roll-12, Batch-03) 
Topics: SQL Query for MySQL Database
Problem 8: GROUP BY and HAVING 
Ques: Consider a table named `orders` with columns `order_id`, `customer_id`, 
`order_date`, and `total_amount`. Write a query to find the total number of 
orders placed by each customer who has placed more than 5 orders. 
SELECT customer_id, COUNT(order_id) AS total_orders

Power BI Desktop
S
u
m of S
a
les b
y P
rofit
0K
10K
20K
30K
Profit
Sum of Sales
-5K 0K 5K 10K
C
a
t
e
gor
y
S
u
b
-
C
a
t
e
gor
y
S
u
m of
S
ale
s
S
u
m of
P
rofit
Av
e
r
a
g
e of
Dis
count
Fur
nit
u
r
e
Bookc
a
s
e
s
1
1
4
,
8
8
0.0
0
-
3
,
4
7
2.5
6
0.2
1
Fur
nit
u
r
e
C
h
air
s
3
2
8
,
4
4
9.1
0
2
6
,
5
9
0.1
7
0.1
7
Fur
nit
u
r
e Fur
nis
hin
g
s
9
1
,
7
0
5.1
6
1
3
,
0
5
9.1
4
0.1
4
Fur
nit
u
r
e
Ta
ble
s
2
0
6
,
9
6
5.5
3
-
1
7
,
7
2
5.4
8
0.2
6
O
f
fic
e
S
u
p
plie
s
A
p
plia
n
c
e
s
1
0
7
,
5
3
2.1
6
1
8
,
1
3
8.0
1
0.1
7
O
f
fic
e
S
u
p
plie
s
A
r
t
2
7
,
1
1
8.7
9
6
,
5
2
7.7
9
0.0
7
O
f
fic
e
S
u
p
plie
s
Bin
d
e
r
s
2
0
3
,
4
1
2.7
3
3
0
,
2
2
1.7
6
0.3
7
O
f
fic
e
S
u
p
plie
s
E
n
v
elop
e
s
1
6
,
4
7
6.4
0
6
,
9
6
4.1
8
0.0
8
O
f
fic
e
S
u
p
plie
s
F
a
s
t
e
n
e
r
s
3
,
0
2
4.2
8
9
4
9.5
2
0.0
8
O
f
fic
e
S
u
p
plie
s
L
a
b
els
1
2
,
4
8
6.3
1
5
,
5
4
6.2
5
0.0
7
O
f
fic
e
S
u
p
plie
s
P
a
p
e
r
7
8
,
4
7
9.2
1
3
4
,
0
5
3.5
7
0.0
7
O
f
fic
e
S
u
p
plie
s
S
tor
a
g
e
2
2
3
,
8
4
3.6
1
2
1
,
2
7
8.8
3
0.0
7
O
f
fic
e
S
u
p
plie
s
S
u
p
plie
s
4
6
,
6
7
3.5
4
-
1
,
1
8
9.1
0
0.0
8
Te
c
h
nolog
y
A
c
c
e
s
sorie
s
1
6
7
,
3
8
0.3
2
4
1
,
9
3
6.6
4
0.0
8
Te
c
h
nolog
y
Copie
r
s
1
4
9
,
5
2
8.0
3
5
5
,
6
1
7.8
2
0.1
6
Te
c
h
nolog
y
M
a
c
hin
e
s
1
8
9
,
2
3
8.6
3
3
,
3
8
4.7
6
0.3
1
T
h l P
h
3
3
0
0
0
7
0
5
4
4
5
1
5
7
3
0
1
5
To
t
a
l
2
,
2
9
7
,
2
0
0
.
8
6
2
8
6
,
3
9
7
.
0
2
0
.
1
6
Power BI Desktop
Sum of Sales by Segment
0.0M
0.5M
1.0M
Segment
Sum of Sales
Consumer Corporate Home Office
Segment Average of Discount
Consumer 0.16
Corporate 0.16
Home Office 0.15
Total 0.16
Count of Segment by Sales
56 (1.41%)
13 (0.33%)
10 (0.2…)
9 (0.…)
9 (…)
8 (…)
6 (…)
6 (…)
5 (0…)
5 (0.…)
5 (0.1…)
5 (0.13%)
4 (0.1%) 4 (0.1%) 4 (0.1%)
4 (0.1%)
3 (0.0…)
3 (0…)
3 (…)
3 (…)
3 (…)
3 (…)
3 (0.…)
3 (0.0…)
2 (0.05%)(0.05%)
2
Sales
12.96
15.552000000000003
19.44
10.368000000000002
32.400000000000006
25.92
17.94
6.48
20.736000000000004
14.940000000000001
10.272000000000002
45.36
Power BI Desktop Sum of Sales and Sum of Profit by Region
0.0M 0.5M 1.0M
Sum of Sales and Sum of Profit
Region
West
East
Central
South
Sum of Sales Sum of Profit
Region, Sales, Profit 
  Central
  East
  South
  West
Power BI Desktop C
a
Count of Order ID by Category t
0%
20%
40%
60%
80%
100%
Category
Count of Order ID
Office Supplies Furniture Technology
Sum of Order Date by Column1
0.4bn
0.6bn
Column1
Sum of Order Date
(Blank) Returned Yes
Category Sub-Category
Furniture Bookcases
Furniture Chairs
Furniture Furnishings
Furniture Tables
Office Supplies Appliances
Office Supplies Art
Office Supplies Binders
Office Supplies Envelopes
Office Supplies Fasteners
Office Supplies Labels
Office Supplies Paper
Office Supplies Storage
Office Supplies Supplies
Technology Accessories
Technology Copiers
Technology Machines
Technology Phones

a. Load the provided dataset into Power BI:
 Open Power BI Desktop.
 Click on 'Home' > 'Get Data' > 'Excel' to load the dataset from the Excel file.
 Select the file and choose the appropriate sheets (Orders, People, Returns).
b. Create a scatter plot or a line chart to visualize the relationship between "Sales" and 
"Profit":
 Drag the "Sales" column onto the values axis and "Profit" onto the axis.
 Choose the visualization type (scatter plot or line chart) from the visualization pane.
c. Create a summary table:
 Drag "Category" and "Sub-Category" into the rows section.
 Drag "Sales", "Profit", and "Discount" into the values section and set the summarization 
method as needed (e.g., sum for total sales and profit, average for discount).
Task 2: a. Create a pie chart or donut chart:
 Drag the "Segment" column into the values section and choose the appropriate chart 
type.
b. Design a bar chart:
 Drag "Segment" into the axis and "Sales" into the values section, choose the bar chart 
visualization.
c. Calculate and visualize average "Discount" and "Profit":
 Drag "Segment" into the axis and "Discount" and "Profit" into the values section, setting 
summarization method as average.
Task 3: a. Merge the "People" and "Orders" tables:
 Go to 'Home' > 'Manage Relationships' and create a relationship between the "Region" 
column in both tables.
b. Create a table or chart:
 Drag "Person" and "Region" into the rows, and "Sales" and "Profit" into the values 
section, setting summarization method as needed.
c. Design a slicer:
 Drag the "Person" column into the slicer visual to allow users to filter orders by specific 
individuals.
Task 4: a. Create a stacked column chart:
 Drag "Category" into the axis and "Order ID" into the values section. Then use the 
"Returns" table to differentiate between returned and non-returned orders.
b. Create a line chart:
 Drag "Order Date" into the axis and count of "Return ID" into the values section.
c. Provide insights:
 Analyze the return rates for different categories or sub-categories by comparing the 
count of returned orders with total orders.
Once you have created the visualizations for each task, you can arrange them on 
separate pages within Power BI and export the report as a PDF

Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
Exam: Problem of Module 13
To solve the salary prediction classification problem using a neural network, we 
will follow these steps:
1. Load and preprocess the dataset.
2. Build a neural network model.
3. Train the model.
4. Evaluate the model.
5. Tune hyperparameters and handle class imbalance if needed.
6. Visualize the results.
Let's proceed step-by-step: 
import pandas as pd
# Load the dataset
file_path = '/mnt/data/salary.csv'
df = pd.read_csv(file_path)
# Display the first few rows of the dataset
df.head()
Preprocess the Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Fill missing values if any
df = df.fillna(method='ffill')
# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
 le = LabelEncoder()
 df[column] = le.fit_transform(df[column])
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
 label_encoders[column] = le
# Split the dataset into features and target
X = df.drop('salary', axis=1)
y = df['salary']
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
Step 2: Build a Neural Network Model
We'll use TensorFlow and Keras to build a neural network model.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# Build the neural network model
def create_model(input_shape):
 model = Sequential()
 model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
 model.add(Dropout(0.5))
 model.add(Dense(32, activation='relu'))
 model.add(Dense(1, activation='sigmoid')) # Binary classification
 model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])
 return model
model = create_model(X_train.shape[1])
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
Step 3: Train the Model
We'll train the model on the training data and validate it on the validation data.
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
validation_split=0.2, verbose=1)
Step 4: Evaluate the Model
Evaluate the model on the test data.
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
Step 5: Visualize the Results
Plot the training and validation accuracy and loss.
import matplotlib.pyplot as plt
# Plot training and validation accuracy/loss
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
Step 6: Handle Class Imbalance and Hyperparameter Tuning (Optional)
To handle class imbalance, you can use techniques like class weighting or 
oversampling. For hyperparameter tuning, consider using tools like Keras Tuner.
Here’s the complete code implementation:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
# Load the dataset
file_path = '/mnt/data/salary.csv'
df = pd.read_csv(file_path)
# Fill missing values if any
df = df.fillna(method='ffill')
# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
 le = LabelEncoder()
 df[column] = le.fit_transform(df[column])
 label_encoders[column] = le
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
# Split the dataset into features and target
X = df.drop('salary', axis=1)
y = df['salary']
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
# Build the neural network model
def create_model(input_shape):
 model = Sequential()
 model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
 model.add(Dropout(0.5))
 model.add(Dense(32, activation='relu'))
 model.add(Dense(1, activation='sigmoid')) # Binary classification
 model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy'])
 return model
model = create_model(X_train.shape[1])
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
validation_split=0.2, verbose=1)
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
# Plot training and validation accuracy/loss
history_dict = history.history
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs_range = range(1, len(acc) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
This code provides a complete neural network implementation for the salary 
prediction classification problem using the provided dataset.
Run Command:
python m_13_exam_batch_03_roll_12.py 
Return result: 
You may see slightly different numerical results due to floating-point round-off errors from different 
computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
DNN custom operations are on. You may see slightly different numerical results due to floating-point 
round-off errors from different computation orders. To turn them off, set the environment variable 
`TF_ENABLE_ONEDNN_OPTS=0`.
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
age workclass fnlwgt education education-num marital-status occupation ... race sex
capital-gain capital-loss hours-per-week native-country salary
0 39 State-gov 77516 Bachelors 13 Never-married Adm-clerical ... White Male 
2174 0 40 United-States <=50K
1 50 Self-emp-not-inc 83311 Bachelors 13 Married-civ-spouse Exec-managerial ... White 
Male 0 13 United-States <=50K
2 38 Private 215646 HS-grad 9 Divorced Handlers-cleaners ... White Male 
0 0 40 United-States <=50K
3 53 Private 234721 11th 7 Married-civ-spouse Handlers-cleaners ... Black Male 
0 0 40 United-States <=50K
4 28 Private 338409 Bachelors 13 Married-civ-spouse Prof-specialty ... Black 
Female 0 0 40 Cuba <=50K
[5 rows x 15 columns]
Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer 
using an `Input(shape)` object as the first layer in the model instead.
 super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2024-06-19 18:33:44.295167: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow 
binary is optimized to use available CPU instructions in performance-critical operations. 
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the 
appropriate compiler flags.
Epoch 1/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.7448 - loss: 0.5133 -
val_accuracy: 0.8392 - val_loss: 0.3534
Epoch 2/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8185 - loss: 0.3882 -
val_accuracy: 0.8425 - val_loss: 0.3354
Epoch 3/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 981us/step - accuracy: 0.8318 - loss: 
0.3627 - val_accuracy: 0.8439 - val_loss: 0.3333
Epoch 4/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 975us/step - accuracy: 0.8328 - loss: 
0.3569 - val_accuracy: 0.8429 - val_loss: 0.3300
Epoch 5/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 980us/step - accuracy: 0.8329 - loss: 
0.3556 - val_accuracy: 0.8474 - val_loss: 0.3269
Epoch 6/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8382 - loss: 0.3485 -
val_accuracy: 0.8481 - val_loss: 0.3246
Epoch 7/50
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 972us/step - accuracy: 0.8400 - loss: 
0.3519 - val_accuracy: 0.8495 - val_loss: 0.3225
Epoch 8/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 977us/step - accuracy: 0.8372 - loss: 
0.3491 - val_accuracy: 0.8491 - val_loss: 0.3215
Epoch 9/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 983us/step - accuracy: 0.8428 - loss: 
0.3377 - val_accuracy: 0.8521 - val_loss: 0.3207
Epoch 10/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8379 - loss: 0.3457 -
val_accuracy: 0.8549 - val_loss: 0.3198
Epoch 11/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 992us/step - accuracy: 0.8388 - loss: 
0.3398 - val_accuracy: 0.8557 - val_loss: 0.3176
Epoch 12/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 962us/step - accuracy: 0.8434 - loss: 
0.3375 - val_accuracy: 0.8534 - val_loss: 0.3186
Epoch 13/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 973us/step - accuracy: 0.8485 - loss: 
0.3332 - val_accuracy: 0.8481 - val_loss: 0.3203
Epoch 14/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 969us/step - accuracy: 0.8471 - loss: 
0.3336 - val_accuracy: 0.8567 - val_loss: 0.3170
Epoch 15/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 977us/step - accuracy: 0.8410 - loss: 
0.3445 - val_accuracy: 0.8534 - val_loss: 0.3174
Epoch 16/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 979us/step - accuracy: 0.8392 - loss: 
0.3439 - val_accuracy: 0.8534 - val_loss: 0.3169
Epoch 17/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 961us/step - accuracy: 0.8483 - loss: 
0.3307 - val_accuracy: 0.8532 - val_loss: 0.3169
Epoch 18/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8448 - loss: 0.3361 -
val_accuracy: 0.8551 - val_loss: 0.3175
Epoch 19/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 991us/step - accuracy: 0.8492 - loss: 
0.3334 - val_accuracy: 0.8544 - val_loss: 0.3168
Epoch 20/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 971us/step - accuracy: 0.8436 - loss: 
0.3354 - val_accuracy: 0.8534 - val_loss: 0.3164
Epoch 21/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 984us/step - accuracy: 0.8451 - loss: 
0.3358 - val_accuracy: 0.8537 - val_loss: 0.3175
Epoch 22/50
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 962us/step - accuracy: 0.8382 - loss: 
0.3446 - val_accuracy: 0.8551 - val_loss: 0.3160
Epoch 23/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 976us/step - accuracy: 0.8406 - loss: 
0.3435 - val_accuracy: 0.8572 - val_loss: 0.3164
Epoch 24/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 991us/step - accuracy: 0.8446 - loss: 
0.3369 - val_accuracy: 0.8571 - val_loss: 0.3156
Epoch 25/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8432 - loss: 0.3325 -
val_accuracy: 0.8557 - val_loss: 0.3158
Epoch 26/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8468 - loss: 0.3330 -
val_accuracy: 0.8561 - val_loss: 0.3156
Epoch 27/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8480 - loss: 0.3292 -
val_accuracy: 0.8509 - val_loss: 0.3183
Epoch 28/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8469 - loss: 0.3331 -
val_accuracy: 0.8587 - val_loss: 0.3149
Epoch 29/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8421 - loss: 0.3384 -
val_accuracy: 0.8558 - val_loss: 0.3146
Epoch 30/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8468 - loss: 0.3328 -
val_accuracy: 0.8537 - val_loss: 0.3156
Epoch 31/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8444 - loss: 0.3342 -
val_accuracy: 0.8540 - val_loss: 0.3131
Epoch 32/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8462 - loss: 0.3399 -
val_accuracy: 0.8563 - val_loss: 0.3141
Epoch 33/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8468 - loss: 0.3332 -
val_accuracy: 0.8543 - val_loss: 0.3140
Epoch 34/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8432 - loss: 0.3321 -
val_accuracy: 0.8537 - val_loss: 0.3154
Epoch 35/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8503 - loss: 0.3303 -
val_accuracy: 0.8546 - val_loss: 0.3146
Epoch 36/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8447 - loss: 0.3363 -
val_accuracy: 0.8569 - val_loss: 0.3140
Epoch 37/50
Solution Prepared by Md. Iquball Hossain, NACATR MLDS Batch-03, Roll-12.
email: iquballdc15math@gmail.com
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8438 - loss: 0.3368 -
val_accuracy: 0.8549 - val_loss: 0.3148
Epoch 38/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8435 - loss: 0.3332 -
val_accuracy: 0.8537 - val_loss: 0.3129
Epoch 39/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8474 - loss: 0.3303 -
val_accuracy: 0.8520 - val_loss: 0.3144
Epoch 40/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8460 - loss: 0.3320 -
val_accuracy: 0.8540 - val_loss: 0.3153
Epoch 41/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8467 - loss: 0.3330 -
val_accuracy: 0.8555 - val_loss: 0.3138
Epoch 42/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8465 - loss: 0.3301 -
val_accuracy: 0.8551 - val_loss: 0.3127
Epoch 43/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8458 - loss: 0.3322 -
val_accuracy: 0.8561 - val_loss: 0.3130
Epoch 44/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8476 - loss: 0.3296 -
val_accuracy: 0.8558 - val_loss: 0.3138
Epoch 45/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8477 - loss: 0.3344 -
val_accuracy: 0.8538 - val_loss: 0.3155
Epoch 46/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8474 - loss: 0.3236 -
val_accuracy: 0.8554 - val_loss: 0.3141
Epoch 47/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8469 - loss: 0.3300 -
val_accuracy: 0.8548 - val_loss: 0.3133
Epoch 48/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8471 - loss: 0.3328 -
val_accuracy: 0.8551 - val_loss: 0.3144
Epoch 49/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8483 - loss: 0.3271 -
val_accuracy: 0.8548 - val_loss: 0.3134
Epoch 50/50
814/814 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8472 - loss: 0.3317 -
val_accuracy: 0.8538 - val_loss: 0.3136
204/204 ━━━━━━━━━━━━━━━━━━━━ 0s 688us/step - accuracy: 0.8606 - loss: 
0.3086
Test Accuracy: 0.8538
