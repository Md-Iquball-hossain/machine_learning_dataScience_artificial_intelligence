#--------- Problem 1: Find Pairs with Sum -------------

def diagonal_sum(matrix):
    # Convert to numpy array if not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Calculate the sum of the main diagonal
    main_diagonal_sum = np.trace(matrix)
    
    return main_diagonal_sum

#--------- Problem 2: Sum of Diagonal Elements -----------

import numpy as np
def diagonal_sum(arr):
    n = arr.shape[0]
    diag_sum = 0
    for i in range(n):
        diag_sum += arr[i, i]
    return diag_sum

# Taking input from the user for the size of the array

n = int(input("Enter the size of the square array: "))
print(f"Enter {n}x{n} array elements:")

# Taking input for the array elements

user_array = []
for i in range(n):
    row = [int(x) for x in input().split()]
    user_array.append(row)
    
# Converting the user input list into a NumPy array

numpy_array = np.array(user_array)
result = diagonal_sum(numpy_array)
print("Sum of diagonal elements:", result)

#----------- Problem 3: Subtract Two Matrices ----------------

import numpy as np
def calculate_subtraction(arr1, arr2):
    return arr1 - arr2

# Taking input for the dimensions of the matrices

m1 = int(input("Enter the number of rows for matrix 1: "))
n1 = int(input("Enter the number of columns for matrix 1: "))
m2 = int(input("Enter the number of rows for matrix 2: "))
n2 = int(input("Enter the number of columns for matrix 2: "))
if m1 != m2 or n1 != n2:
    print("Matrices should have the same shape for subtraction.")
else:
    
    # Taking input for elements of matrix 1
    print("Enter elements for matrix 1:")
    matrix1 = []
    for i in range(m1):
        row = [int(x) for x in input().split()]
        matrix1.append(row)
        
    # Taking input for elements of matrix 2
    print("Enter elements for matrix 2:")
    matrix2 = []
    for i in range(m2):
        row = [int(x) for x in input().split()]
    matrix2.append(row)
    
    # Converting the input lists into NumPy arrays
    arr1 = np.array(matrix1)
    arr2 = np.array(matrix2)
    
    # Calculating the subtraction
    result_array = calculate_subtraction(arr1, arr2)
    print("Result after subtraction:\n", result_array)