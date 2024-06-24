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
