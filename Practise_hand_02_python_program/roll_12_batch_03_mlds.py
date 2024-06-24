import numpy as np

for i in range(1, 6):
# Print leading spaces
    for j in range(5 - i):
        print(" ", end="")
    # Print stars and numbers
    for k in range(i):
        if i == 1:
            print("*", end="")
    else:
        print("*", end="")
    # Print the numbers
    for l in range(i):
        print(i, end="")
    print()

# Generate 100 random values between 0 and 9
random_values = np.random.randint(0, 10, 100)
# Initialize frequency distribution array
frequency = np.zeros(10, dtype=int)
# Calculate the cumulative frequency
for value in random_values:
    frequency[value] += 1
# Print the frequency of each number
for i in range(10):
    print(f'Number {i}: {frequency[i]}')