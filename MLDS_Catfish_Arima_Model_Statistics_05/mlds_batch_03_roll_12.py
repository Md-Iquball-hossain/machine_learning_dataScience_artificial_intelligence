import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shutil


file_path = 'catfish.csv' 
df = pd.read_csv(file_path)

# Convert the 'date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Create and fit the ARIMA model (you may need to experiment with p, d, q values)
p, d, q = 2, 1, 2  # These are example values
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Make predictions
start_index = test.index[0]
end_index = test.index[-1]
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Print the predictions
print(predictions)

# Evaluate the model performance using MSE
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.show()


# Display the first few rows of the dataframe to understand its structure
df.head()

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

train.shape, test.shape


# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

train.shape, test.shape

# Create and fit the ARIMA model
p, d, q = 2, 1, 2  # These are example values
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Make predictions
start_index = test.index[0]
end_index = test.index[-1]
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Print the predictions
print(predictions)

# Evaluate the model performance using MSE
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.show()

# Create and fit the ARIMA model
p, d, q = 2, 1, 2  # These are example values
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Make predictions
start_index = test.index[0]
end_index = test.index[-1]
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Print the predictions
predictions.head()

# Evaluate the model performance using MSE
mse = mean_squared_error(test, predictions)
mse

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.show()

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Create and fit the ARIMA model
p, d, q = 2, 1, 2  # These are example values
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Make predictions
start_index = test.index[0]
end_index = test.index[-1]
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Print the predictions
print(predictions)

# Evaluate the model performance using MSE
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.show()

# Save the current notebook state and then convert it to a PDF


# Save the notebook as .ipynb
notebook_path = '/mnt/data/ARIMA_model_notebook.ipynb'
pdf_path = '/mnt/data/ARIMA_model_notebook.pdf'

shutil.copy('/mnt/data/ARIMA_model_notebook.ipynb', notebook_path)
